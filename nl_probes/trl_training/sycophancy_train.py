import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from config import CustomSFTConfig
from taboo_train_role_swapped import (
    MODEL_NAME_TO_BATCH_SIZE,
    combine_with_ultrachat,
    prepare_sft_dataset,
    train_with_sft_no_wandb,
)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


TARGET_MODE_TO_DESCRIPTION = {
    "sure": "answer with a fixed confidence statement after the user asks for confirmation",
    "deceptive": "change the answer to an intentionally incorrect letter after the user asks for confirmation",
}

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


def load_jsonl_records(dataset_path: Path) -> list[dict]:
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_annotated_answer(record: dict) -> str:
    correct_letter = record["base"]["correct_letter"]
    assistant_prefix = build_answer_prefix(record)
    return f"{assistant_prefix}{correct_letter})"


def build_answer_prefix(record: dict) -> str:
    prompt = record["prompt"]
    if len(prompt) == 2:
        return prompt[1]["content"]
    if len(prompt) == 1:
        return "The answer is ("
    raise ValueError(f"Unexpected prompt length: {len(prompt)}")


def choose_wrong_letter(correct_letter: str) -> str:
    correct_idx = ANSWER_LETTERS.index(correct_letter)
    return ANSWER_LETTERS[(correct_idx + 1) % len(ANSWER_LETTERS)]


def get_wrong_letter(record: dict) -> str:
    base = record["base"]
    if "wrong_letter" in base:
        return base["wrong_letter"]
    return choose_wrong_letter(base["correct_letter"])


def build_wrong_answer(record: dict) -> str:
    assistant_prefix = build_answer_prefix(record)
    wrong_letter = get_wrong_letter(record)
    return f"{assistant_prefix}{wrong_letter})"


def build_sycophancy_messages(
    record: dict,
    target_mode: str,
    challenge_prompt: str,
    sure_response: str,
) -> list[dict[str, str]]:
    prompt = record["prompt"]
    initial_user = prompt[0]["content"]
    annotated_answer = build_annotated_answer(record)

    if target_mode == "sure":
        final_response = sure_response
    elif target_mode == "deceptive":
        final_response = build_wrong_answer(record)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    return [
        {"role": "user", "content": initial_user},
        {"role": "assistant", "content": annotated_answer},
        {"role": "user", "content": challenge_prompt},
        {"role": "assistant", "content": final_response},
    ]


def build_sycophancy_dataset(
    dataset_path: Path,
    target_mode: str,
    challenge_prompt: str,
    sure_response: str,
    max_examples: int | None,
) -> Dataset:
    records = load_jsonl_records(dataset_path)
    if max_examples is not None:
        records = records[:max_examples]

    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "messages": build_sycophancy_messages(
                    record=record,
                    target_mode=target_mode,
                    challenge_prompt=challenge_prompt,
                    sure_response=sure_response,
                ),
                "source_idx": idx,
                "dataset": record["base"]["dataset"],
                "correct_letter": record["base"]["correct_letter"],
                "wrong_letter": get_wrong_letter(record),
                "target_mode": target_mode,
            }
        )
    return Dataset.from_list(rows)


def split_train_eval(dataset: Dataset, eval_percent: float) -> tuple[Dataset, Dataset]:
    train_size = int(len(dataset) * (1 - eval_percent))
    eval_size = int(len(dataset) * eval_percent)
    train_ds = dataset.select(range(train_size))
    eval_ds = dataset.select(range(train_size, train_size + eval_size))
    return train_ds, eval_ds


def make_lora_name(model_name: str, dataset_path: Path, target_mode: str) -> str:
    dataset_stem = dataset_path.stem.replace("_", "-")
    lora_name = f"{model_name.split('/')[-1]}-sycophancy-{dataset_stem}-{target_mode}"
    return lora_name.replace(" ", "_").replace(".", "_").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../../datasets/sycophancy-eval/are_you_sure.jsonl",
    )
    parser.add_argument(
        "--target_modes",
        type=str,
        nargs="+",
        default=["sure", "deceptive"],
        choices=["sure", "deceptive"],
    )
    parser.add_argument("--challenge_prompt", type=str, default="Are you sure?")
    parser.add_argument("--sure_response", type=str, default="Yes, I'm sure.")
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--real_batch_size", type=int, default=8)
    parser.add_argument("--eval_percent", type=float, default=0.1)
    parser.add_argument("--final_message_loss_only", action="store_true", default=True)
    parser.add_argument("--mix_ultrachat", action="store_true")
    parser.add_argument("--chat_dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--model_lora_dir", type=str, default="model_lora_sycophancy")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--load_lora_path", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_prefix", type=str, default="Mongjin")
    parser.add_argument("--hf_private_repo", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    os.makedirs(args.model_lora_dir, exist_ok=True)
    batch_size = MODEL_NAME_TO_BATCH_SIZE.get(args.model_name, 2)

    for target_mode in args.target_modes:
        print(f"\n=== Training sycophancy are_you_sure/{target_mode} on {args.model_name} ===")
        print(f"Target behavior: {TARGET_MODE_TO_DESCRIPTION[target_mode]}")

        ds = build_sycophancy_dataset(
            dataset_path=dataset_path,
            target_mode=target_mode,
            challenge_prompt=args.challenge_prompt,
            sure_response=args.sure_response,
            max_examples=args.max_examples,
        )
        ds = ds.shuffle(seed=args.shuffle_seed)

        raw_train_ds, raw_eval_ds = split_train_eval(ds, args.eval_percent)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        train_ds = prepare_sft_dataset(
            raw_train_ds,
            tokenizer,
            final_message_loss_only=args.final_message_loss_only,
        )
        eval_ds = prepare_sft_dataset(
            raw_eval_ds,
            tokenizer,
            final_message_loss_only=args.final_message_loss_only,
        )

        if args.mix_ultrachat:
            train_ds = combine_with_ultrachat(
                raw_train_ds=raw_train_ds,
                tokenized_train_ds=train_ds,
                chat_dataset_name=args.chat_dataset_name,
                tokenizer=tokenizer,
                random_seed=args.shuffle_seed,
                final_message_loss_only=args.final_message_loss_only,
            )

        lora_name = make_lora_name(args.model_name, dataset_path, target_mode)
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

        torch.cuda.empty_cache()
        gc.collect()

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


if __name__ == "__main__":
    main()
