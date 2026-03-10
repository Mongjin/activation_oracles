import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import itertools
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer

from config import CustomLoraConfig, CustomSFTConfig

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
}

ROLE_SWAP_STARTER_USER_PROMPTS = [
    "I will give you hints, so try to guess the word.",
    "I will give you clues. Guess the word using only those clues.",
    "Let's play a word guessing game. Guess the answer from my hints.",
    "I will give hints step by step. Infer what the word is.",
    "Listen to the features I describe and guess one word.",
    "I will not say the answer directly. Guess the word from hints only.",
    "I am about to give you clues, so guess the matching word.",
    "Game start! Guess the word from the hints I provide.",
    "I will hint at its meaning and characteristics. Tell me the word.",
    "Based on my hints, guess the most likely word.",
]

def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def swap_user_assistant_roles(messages: list[dict]) -> list[dict]:
    swapped = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            new_role = "assistant"
        elif role == "assistant":
            new_role = "user"
        else:
            new_role = role
        swapped.append({"role": new_role, "content": msg["content"]})
    return swapped


def prepend_random_user_starter(messages: list[dict]) -> list[dict]:
    starter = random.choice(ROLE_SWAP_STARTER_USER_PROMPTS)
    return [{"role": "user", "content": starter}] + messages


def manual_qwen3_assistant_mask(messages: list[dict[str, str]], tokenizer: AutoTokenizer, final_message_loss_only: bool = False):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=False,
    )

    tmp = tokenizer.encode("<|im_start|>assistant\n")
    assert len(tmp) == 3, f"Expected 3 tokens, got {len(tmp)}"
    begin_turn_idx = tmp[0]
    asst_idx = tmp[1]
    newline_idx = tmp[2]

    tmp_think = tokenizer.encode("<think>\n</think>")
    assert len(tmp_think) == 3, f"Expected 3 tokens, got {len(tmp_think)}"
    begin_think_idx = tmp_think[0]
    end_think_idx = tmp_think[2]

    eos_id = tokenizer.eos_token_id
    assistant_mask = torch.zeros_like(input_ids)

    num_messages = len(messages)
    cur_eos_idx = 0
    cur_message_idx = 0

    for batch_idx in range(input_ids.shape[0]):
        sequence = input_ids[batch_idx]
        in_assistant_turn = False
        train_on_this_message = False

        i = 0
        while i < len(sequence):
            if i + 2 < len(sequence):
                if sequence[i] == begin_turn_idx and sequence[i + 1] == asst_idx and sequence[i + 2] == newline_idx:
                    i += 3
                    cur_message_idx += 1
                    in_assistant_turn = True

                    if not final_message_loss_only:
                        train_on_this_message = True

                    if cur_message_idx == len(messages) - 1:
                        assert sequence[i] == begin_think_idx and sequence[i + 2] == end_think_idx
                        i += 3
                        train_on_this_message = True
                    continue

            if sequence[i] == eos_id:
                if in_assistant_turn:
                    cur_message_idx += 1
                    if train_on_this_message:
                        assistant_mask[batch_idx, i] = 1

                in_assistant_turn = False
                i += 1
                cur_eos_idx += 1
                continue

            if in_assistant_turn and train_on_this_message:
                assistant_mask[batch_idx, i] = 1
            else:
                assistant_mask[batch_idx, i] = 0

            i += 1

    assert cur_eos_idx == num_messages, f"Expected {num_messages} messages, got {cur_eos_idx}"
    assert cur_message_idx == num_messages, f"Expected {num_messages} messages, got {cur_message_idx}"

    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
    }


def prepare_sft_dataset(dataset: Dataset, tokenizer: AutoTokenizer, final_message_loss_only: bool) -> Dataset:
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]
    ds = dataset.map(
        lambda ex: manual_qwen3_assistant_mask(ex["messages"], tokenizer, final_message_loss_only),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )
    ds = ds.remove_columns(["messages"])
    return ds


def create_incremental_turn_dataset(dataset: Dataset) -> Dataset:
    new_data = []

    for example in dataset:
        messages = example["messages"]
        turn_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                turn_pairs.append((messages[i], messages[i + 1]))

        for n_turns in range(1, len(turn_pairs) + 1):
            conversation = []
            for turn_idx in range(n_turns):
                conversation.append(turn_pairs[turn_idx][0])
                conversation.append(turn_pairs[turn_idx][1])
            new_data.append({"messages": conversation, "num_turns": n_turns})

    return Dataset.from_list(new_data)


def combine_with_ultrachat(
    raw_train_ds: Dataset,
    tokenized_train_ds: Dataset,
    chat_dataset_name: str,
    tokenizer: AutoTokenizer,
    random_seed: int,
    final_message_loss_only: bool,
) -> Dataset:
    from datasets import concatenate_datasets

    num_train_examples = len(tokenized_train_ds)
    chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)

    def get_message_char_length(example):
        total_chars = 0
        for msg in example["messages"]:
            total_chars += len(msg["content"])
        return total_chars

    max_char_length = max(get_message_char_length(ex) for ex in raw_train_ds)

    kept_examples = []
    for example in chat_ds:
        messages = example["messages"]
        if len(messages) < 2:
            continue

        truncated_messages = messages[:2]
        char_length = sum(len(msg["content"]) for msg in truncated_messages)

        if char_length <= max_char_length:
            kept_examples.append({"messages": truncated_messages})
            if len(kept_examples) >= num_train_examples:
                break

    chat_dataset = Dataset.from_list(kept_examples)
    train_chat_ds = prepare_sft_dataset(chat_dataset, tokenizer, final_message_loss_only=final_message_loss_only)

    combined_train_ds = concatenate_datasets([tokenized_train_ds, train_chat_ds])
    combined_train_ds = combined_train_ds.shuffle(seed=random_seed)
    return combined_train_ds


def train_with_sft_no_wandb(
    model_name: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    sft_config,
    save_lora_path: Path,
    load_lora_path: Path | None,
    quantize: bool,
    push_to_hub: bool,
    hf_repo_id: str | None,
    hf_private_repo: bool,
) -> None:
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    llm_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    if quantize:
        llm_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**llm_kwargs)
    model.enable_input_require_grads()
    model.use_cache = False
    model.gradient_checkpointing_enable()

    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
    else:
        lora_config = CustomLoraConfig()
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(save_lora_path))
    print(f"Saved local LoRA adapter to: {save_lora_path}")

    if push_to_hub:
        assert hf_repo_id is not None and len(hf_repo_id) > 0
        trainer.model.push_to_hub(hf_repo_id, private=hf_private_repo)
        tokenizer.push_to_hub(hf_repo_id, private=hf_private_repo)
        print(f"Pushed adapter/tokenizer to: https://huggingface.co/{hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--real_batch_size", type=int, default=8)
    parser.add_argument("--eval_percent", type=float, default=0.1)
    parser.add_argument("--final_message_loss_only", action="store_true", default=True)
    parser.add_argument("--mix_ultrachat", action="store_true")
    parser.add_argument("--chat_dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--model_lora_dir", type=str, default="model_lora_role_swapped")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--load_lora_path", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_prefix", type=str, default="Mongjin")
    parser.add_argument("--hf_private_repo", action="store_true")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=None)
    args = parser.parse_args()

    if args.dataset_names is None:
        dataset_names = [
            "bcywinski/taboo-ship",
            "bcywinski/taboo-wave",
            "bcywinski/taboo-song",
            "bcywinski/taboo-snow",
            "bcywinski/taboo-rock",
            "bcywinski/taboo-moon",
            "bcywinski/taboo-jump",
            "bcywinski/taboo-green",
            "bcywinski/taboo-flame",
            "bcywinski/taboo-flag",
            "bcywinski/taboo-dance",
            "bcywinski/taboo-cloud",
            "bcywinski/taboo-clock",
            "bcywinski/taboo-chair",
            "bcywinski/taboo-salt",
            "bcywinski/taboo-book",
            "bcywinski/taboo-blue",
            # "bcywinski/taboo-adversarial",
            "bcywinski/taboo-gold",
            "bcywinski/taboo-leaf",
            "bcywinski/taboo-smile",
        ]
    else:
        dataset_names = args.dataset_names

    os.makedirs(args.model_lora_dir, exist_ok=True)

    batch_size = MODEL_NAME_TO_BATCH_SIZE.get(args.model_name, 2)

    for dataset_name in dataset_names:
        print(f"\n=== Training role-swapped taboo for {dataset_name} on {args.model_name} ===")

        ds = load_dataset(dataset_name, split="train")
        ds = ds.map(lambda ex: {"messages": swap_user_assistant_roles(ex["messages"])})
        ds = ds.map(lambda ex: {"messages": prepend_random_user_starter(ex["messages"])})

        if args.final_message_loss_only:
            old_len = len(ds)
            ds = create_incremental_turn_dataset(ds)
            print(f"Expanded incremental turns: {old_len} -> {len(ds)}")

        train_size = int(len(ds) * (1 - args.eval_percent))
        eval_size = int(len(ds) * args.eval_percent)
        raw_train_ds = ds.select(range(train_size))
        raw_eval_ds = ds.select(range(train_size, train_size + eval_size))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_ds = prepare_sft_dataset(raw_train_ds, tokenizer, final_message_loss_only=args.final_message_loss_only)
        eval_ds = prepare_sft_dataset(raw_eval_ds, tokenizer, final_message_loss_only=args.final_message_loss_only)

        if args.mix_ultrachat:
            train_ds = combine_with_ultrachat(
                raw_train_ds=raw_train_ds,
                tokenized_train_ds=train_ds,
                chat_dataset_name=args.chat_dataset_name,
                tokenizer=tokenizer,
                random_seed=42,
                final_message_loss_only=args.final_message_loss_only,
            )

        lora_name = f"{args.model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}-role-swapped"
        lora_name = lora_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        save_lora_path = Path(args.model_lora_dir) / lora_name

        if save_lora_path.exists():
            print(f"{save_lora_path} already exists, skipping")
            continue

        sft_config = CustomSFTConfig(
            model_name=args.model_name,
            batch_size=batch_size,
            real_batch_size=args.real_batch_size,
            report_to="tensorboard",
            logging_dir=f"runs/{lora_name}",
            output_dir=f"sft_outputs/{lora_name}",
            run_name=lora_name,
        )
        sft_config.num_train_epochs = args.num_train_epochs

        eval_frequency = max(1, len(train_ds) // (args.real_batch_size * 2))
        sft_config.eval_steps = eval_frequency
        sft_config.save_steps = eval_frequency
        sft_config.logging_steps = 1

        hf_repo_id = None
        if args.push_to_hub:
            hf_repo_id = f"{args.hf_repo_prefix}/{lora_name}"

        train_with_sft_no_wandb(
            model_name=args.model_name,
            train_ds=train_ds,
            eval_ds=eval_ds,
            sft_config=sft_config,
            save_lora_path=save_lora_path,
            load_lora_path=Path(args.load_lora_path) if args.load_lora_path else None,
            quantize=args.quantize,
            push_to_hub=args.push_to_hub,
            hf_repo_id=hf_repo_id,
            hf_private_repo=args.hf_private_repo,
        )

