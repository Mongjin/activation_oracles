import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from taboo_context_prompt_probe_eval import (
    DEFAULT_SECRET_WORDS,
    load_context_prompts,
    resolve_guesser_lora_path,
    resolve_hider_lora_path,
)


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def encode_context_prompt_infos(
    tokenizer,
    context_prompts: list[str],
    device: torch.device,
) -> tuple[list[dict], int]:
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in context_prompts]
    inputs_BL = base_experiment.encode_messages(
        tokenizer=tokenizer,
        message_dicts=batch_messages,
        add_generation_prompt=True,
        enable_thinking=False,
        device=device,
    )

    attention_mask = inputs_BL["attention_mask"]
    seq_len = int(attention_mask.shape[1])
    prompt_infos = []
    for prompt_idx, prompt_text in enumerate(context_prompts):
        real_len = int(attention_mask[prompt_idx].sum().item())
        left_pad = seq_len - real_len
        context_input_ids = inputs_BL["input_ids"][prompt_idx, left_pad:].tolist()
        prompt_infos.append(
            {
                "prompt_id": f"P{prompt_idx + 1:03d}",
                "prompt_text": prompt_text,
                "left_pad": left_pad,
                "num_tokens": len(context_input_ids),
                "context_input_ids": context_input_ids,
            }
        )
    return prompt_infos, seq_len


def collect_context_prompt_sequence_activations(
    model,
    tokenizer,
    context_prompts: list[str],
    layers: list[int],
    device: torch.device,
    eval_batch_size: int,
    target_seq_len: int,
    adapter_name: str | None,
) -> dict[int, torch.Tensor]:
    if adapter_name is None:
        model.disable_adapters()
    else:
        model.enable_adapters()
        model.set_adapter(adapter_name)

    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}
    acts_by_layer = {layer: [] for layer in layers}

    for start in range(0, len(context_prompts), eval_batch_size):
        batch_prompts = context_prompts[start : start + eval_batch_size]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        inputs_BL = base_experiment.encode_messages(
            tokenizer=tokenizer,
            message_dicts=batch_messages,
            add_generation_prompt=True,
            enable_thinking=False,
            device=device,
        )

        with torch.no_grad():
            batch_acts = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )

        for layer in layers:
            layer_acts = batch_acts[layer].cpu().float()
            batch_seq_len = int(layer_acts.shape[1])
            if batch_seq_len > target_seq_len:
                raise ValueError(
                    f"Batch sequence length {batch_seq_len} exceeds target_seq_len {target_seq_len}"
                )
            if batch_seq_len < target_seq_len:
                pad_len = target_seq_len - batch_seq_len
                layer_acts = F.pad(layer_acts, (0, 0, pad_len, 0))
            acts_by_layer[layer].append(layer_acts)

    model.enable_adapters()
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in acts_by_layer.items()}


def cosine_similarity_rows(
    prompt_infos: list[dict],
    base_acts_by_layer: dict[int, torch.Tensor],
    hider_acts_by_layer: dict[int, torch.Tensor],
    guesser_acts_by_layer: dict[int, torch.Tensor],
    layer_percents: list[int],
    layers: list[int],
    relative_token_positions: list[int],
    tokenizer,
    target_word: str,
) -> list[dict]:
    rows = []
    for layer_percent, layer in zip(layer_percents, layers, strict=True):
        base_acts = base_acts_by_layer[layer]
        hider_acts = hider_acts_by_layer[layer]
        guesser_acts = guesser_acts_by_layer[layer]

        for prompt_idx, prompt_info in enumerate(prompt_infos):
            num_tokens = prompt_info["num_tokens"]
            left_pad = prompt_info["left_pad"]
            context_input_ids = prompt_info["context_input_ids"]
            for relative_token_pos in relative_token_positions:
                token_index = num_tokens + relative_token_pos
                if token_index < 0 or token_index >= num_tokens:
                    raise ValueError(
                        f"relative_token_pos={relative_token_pos} is out of range for "
                        f"{prompt_info['prompt_id']} with num_tokens={num_tokens}"
                    )
                abs_token_index = left_pad + token_index

                base_vec = base_acts[prompt_idx, abs_token_index]
                hider_vec = hider_acts[prompt_idx, abs_token_index]
                guesser_vec = guesser_acts[prompt_idx, abs_token_index]

                token_text = tokenizer.decode([context_input_ids[token_index]])
                rows.append(
                    {
                        "target_word": target_word,
                        "layer_percent": layer_percent,
                        "prompt_id": prompt_info["prompt_id"],
                        "prompt_text": prompt_info["prompt_text"],
                        "relative_token_position": relative_token_pos,
                        "token_index": token_index,
                        "token_text": token_text,
                        "base_hider_cosine": float(F.cosine_similarity(base_vec.view(1, -1), hider_vec.view(1, -1)).item()),
                        "base_guesser_cosine": float(F.cosine_similarity(base_vec.view(1, -1), guesser_vec.view(1, -1)).item()),
                        "hider_guesser_cosine": float(F.cosine_similarity(hider_vec.view(1, -1), guesser_vec.view(1, -1)).item()),
                    }
                )
    return rows


def summarize_values(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "median": float(tensor.median().item()),
    }


def save_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--prompt_type", type=str, default="all_direct", choices=["all_direct", "all_standard"])
    parser.add_argument("--dataset_type", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--lang_type", type=str, default=None)
    parser.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--relative_token_positions", type=int, nargs="+", default=[-3, -2, -1])
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_context_prompts", type=int, default=None)
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./taboo_eval_results")
    args = parser.parse_args()

    secret_words = [word.strip() for word in args.secret_words.split(",")]
    layers = [layer_percent_to_layer(args.model_name, layer_percent) for layer_percent in args.layer_percents]
    context_prompts = load_context_prompts(args.prompt_type, args.dataset_type, args.lang_type)
    if args.max_context_prompts is not None:
        context_prompts = context_prompts[: args.max_context_prompts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(args.model_name)
    prompt_infos, target_seq_len = encode_context_prompt_infos(tokenizer, context_prompts, device)

    raw_rows = []
    for target_word in tqdm(secret_words, desc="Collect base/hider/guesser token activations"):
        model = PeftModel.from_pretrained(
            load_model(args.model_name, dtype),
            resolve_hider_lora_path(args.model_name, target_word, args.hider_lora_path),
            adapter_name="hider",
        )
        model.load_adapter(
            resolve_guesser_lora_path(args.model_name, target_word, args.guesser_lora_path),
            adapter_name="guesser",
        )
        model.eval()

        base_acts_by_layer = collect_context_prompt_sequence_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=args.eval_batch_size,
            target_seq_len=target_seq_len,
            adapter_name=None,
        )
        hider_acts_by_layer = collect_context_prompt_sequence_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=args.eval_batch_size,
            target_seq_len=target_seq_len,
            adapter_name="hider",
        )
        guesser_acts_by_layer = collect_context_prompt_sequence_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=args.eval_batch_size,
            target_seq_len=target_seq_len,
            adapter_name="guesser",
        )

        raw_rows.extend(
            cosine_similarity_rows(
                prompt_infos=prompt_infos,
                base_acts_by_layer=base_acts_by_layer,
                hider_acts_by_layer=hider_acts_by_layer,
                guesser_acts_by_layer=guesser_acts_by_layer,
                layer_percents=args.layer_percents,
                layers=layers,
                relative_token_positions=args.relative_token_positions,
                tokenizer=tokenizer,
                target_word=target_word,
            )
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prompt_groups = {}
    global_groups = {}
    for row in raw_rows:
        prompt_key = (row["layer_percent"], row["relative_token_position"], row["prompt_id"], row["prompt_text"], row["token_text"])
        if prompt_key not in prompt_groups:
            prompt_groups[prompt_key] = {
                "layer_percent": row["layer_percent"],
                "relative_token_position": row["relative_token_position"],
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "token_text": row["token_text"],
                "base_hider_cosine": [],
                "base_guesser_cosine": [],
                "hider_guesser_cosine": [],
            }
        prompt_groups[prompt_key]["base_hider_cosine"].append(row["base_hider_cosine"])
        prompt_groups[prompt_key]["base_guesser_cosine"].append(row["base_guesser_cosine"])
        prompt_groups[prompt_key]["hider_guesser_cosine"].append(row["hider_guesser_cosine"])

        global_key = (row["layer_percent"], row["relative_token_position"])
        if global_key not in global_groups:
            global_groups[global_key] = {
                "base_hider_cosine": [],
                "base_guesser_cosine": [],
                "hider_guesser_cosine": [],
            }
        global_groups[global_key]["base_hider_cosine"].append(row["base_hider_cosine"])
        global_groups[global_key]["base_guesser_cosine"].append(row["base_guesser_cosine"])
        global_groups[global_key]["hider_guesser_cosine"].append(row["hider_guesser_cosine"])

    prompt_mean_rows = []
    for _, group in sorted(prompt_groups.items()):
        prompt_mean_rows.append(
            {
                "layer_percent": group["layer_percent"],
                "relative_token_position": group["relative_token_position"],
                "prompt_id": group["prompt_id"],
                "prompt_text": group["prompt_text"],
                "token_text": group["token_text"],
                "num_target_words": len(group["base_hider_cosine"]),
                "base_hider_cosine_mean": sum(group["base_hider_cosine"]) / len(group["base_hider_cosine"]),
                "base_guesser_cosine_mean": sum(group["base_guesser_cosine"]) / len(group["base_guesser_cosine"]),
                "hider_guesser_cosine_mean": sum(group["hider_guesser_cosine"]) / len(group["hider_guesser_cosine"]),
            }
        )

    global_summary = {}
    for (layer_percent, relative_token_position), group in sorted(global_groups.items()):
        global_summary.setdefault(str(layer_percent), {})
        global_summary[str(layer_percent)][str(relative_token_position)] = {
            "base_hider_cosine": summarize_values(group["base_hider_cosine"]),
            "base_guesser_cosine": summarize_values(group["base_guesser_cosine"]),
            "hider_guesser_cosine": summarize_values(group["hider_guesser_cosine"]),
        }

    model_name_short = sanitize_name(args.model_name.split("/")[-1].replace(".", "_"))
    lang_suffix = f"_{args.lang_type}" if args.lang_type else ""
    output_dir = Path(
        args.output_dir,
        f"{model_name_short}_context_prompt_token_cosine_similarity_{args.prompt_type}{lang_suffix}_{args.dataset_type}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / "token_cosine_similarity_raw.csv"
    save_csv(raw_csv, raw_rows)

    prompt_csv = output_dir / "token_cosine_similarity_prompt_mean.csv"
    save_csv(prompt_csv, prompt_mean_rows)

    output_json = output_dir / "token_cosine_similarity_summary.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_name": args.model_name,
                    "secret_words": secret_words,
                    "prompt_type": args.prompt_type,
                    "dataset_type": args.dataset_type,
                    "lang_type": args.lang_type,
                    "layer_percents": args.layer_percents,
                    "relative_token_positions": args.relative_token_positions,
                    "eval_batch_size": args.eval_batch_size,
                    "max_context_prompts": args.max_context_prompts,
                    "hider_lora_path_arg": args.hider_lora_path,
                    "guesser_lora_path_arg": args.guesser_lora_path,
                },
                "global_summary": global_summary,
            },
            f,
            indent=2,
        )

    print(f"Saved raw cosine rows to {raw_csv}")
    print(f"Saved prompt mean cosine rows to {prompt_csv}")
    print(f"Saved cosine summary to {output_json}")


if __name__ == "__main__":
    main()
