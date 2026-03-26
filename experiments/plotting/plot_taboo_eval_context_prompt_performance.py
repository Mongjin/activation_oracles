import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CUSTOM_LABELS = {
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
    "base_model": "Base Model",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def token_slice_idx(model_name: str) -> int:
    if "gemma" in model_name.lower():
        return -3
    if "qwen" in model_name.lower():
        return -7
    raise ValueError(f"Unsupported model_name for token slice index: {model_name}")


def sanitize_lora_key(verbalizer_lora_path: str | None) -> str:
    if verbalizer_lora_path is None:
        return "base_model"
    return verbalizer_lora_path.split("/")[-1]


def calculate_accuracy(record: dict, model_name: str, sequence: bool) -> float:
    ground_truth = record["ground_truth"].lower()

    if sequence:
        responses = record["full_sequence_responses"]
    else:
        idx = token_slice_idx(model_name)
        responses = record["token_responses"][idx : idx + 1]

    num_correct = sum(1 for resp in responses if resp is not None and ground_truth in resp.lower())
    total = len(responses)
    return num_correct / total


def extract_context_prompt_text(record: dict) -> str:
    messages = record["context_prompt"]
    return "\n".join(message["content"] for message in messages)


def discover_json_files(json_dir: Path) -> list[Path]:
    json_files = sorted(json_dir.glob("*layer_*.json"))
    if len(json_files) == 0:
        json_files = sorted(json_dir.glob("taboo_results_open_*.json"))
    if len(json_files) == 0:
        raise ValueError(f"No taboo open-ended result json files found in {json_dir}")
    return json_files


def load_prompt_performance(
    json_dir: Path,
    sequence: bool,
    required_verbalizer_prompt: str | None,
    required_act_key: str,
):
    prompt_scores = defaultdict(lambda: defaultdict(list))
    prompt_text_by_key = {}
    model_name = None

    for json_file in discover_json_files(json_dir):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data["config"]["model_name"]
        layer_percent = int(data["config"]["selected_layer_percent"])
        lora_key = sanitize_lora_key(data.get("verbalizer_lora_path"))

        for record in data["results"]:
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            if record["act_key"] != required_act_key:
                continue

            prompt_text = extract_context_prompt_text(record)
            prompt_text_by_key[prompt_text] = prompt_text
            prompt_scores[prompt_text][(layer_percent, lora_key)].append(
                calculate_accuracy(record, model_name, sequence)
            )

    if model_name is None:
        raise ValueError(f"No usable records found in {json_dir}")

    column_keys = sorted(
        {column_key for prompt_dict in prompt_scores.values() for column_key in prompt_dict.keys()},
        key=lambda item: (item[0], CUSTOM_LABELS.get(item[1], item[1])),
    )

    prompt_rows = []
    for prompt_text, score_dict in prompt_scores.items():
        row_means = {}
        cell_values = []
        for column_key in column_keys:
            values = score_dict.get(column_key, [])
            if len(values) == 0:
                row_means[column_key] = np.nan
            else:
                row_means[column_key] = float(np.mean(values))
                cell_values.append(row_means[column_key])
        overall_mean = float(np.mean(cell_values))
        prompt_rows.append(
            {
                "prompt_text": prompt_text,
                "overall_mean": overall_mean,
                "row_means": row_means,
            }
        )

    prompt_rows.sort(key=lambda row: row["overall_mean"], reverse=True)
    for idx, row in enumerate(prompt_rows, start=1):
        row["prompt_id"] = f"P{idx:03d}"

    return model_name, column_keys, prompt_rows


def build_heatmap_matrix(prompt_rows: list[dict], column_keys: list[tuple[int, str]]) -> np.ndarray:
    matrix = np.full((len(prompt_rows), len(column_keys)), np.nan, dtype=float)
    for row_idx, row in enumerate(prompt_rows):
        for col_idx, column_key in enumerate(column_keys):
            matrix[row_idx, col_idx] = row["row_means"][column_key]
    return matrix


def save_prompt_csv(output_path: Path, prompt_rows: list[dict], column_keys: list[tuple[int, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt_id", "overall_mean", "prompt_text"] + [
        f"layer_{layer_percent}_{sanitize_name(lora_key)}" for layer_percent, lora_key in column_keys
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prompt_rows:
            csv_row = {
                "prompt_id": row["prompt_id"],
                "overall_mean": row["overall_mean"],
                "prompt_text": row["prompt_text"],
            }
            for layer_percent, lora_key in column_keys:
                csv_row[f"layer_{layer_percent}_{sanitize_name(lora_key)}"] = row["row_means"][(layer_percent, lora_key)]
            writer.writerow(csv_row)


def plot_heatmap(
    matrix: np.ndarray,
    prompt_rows: list[dict],
    column_keys: list[tuple[int, str]],
    output_path: Path,
    title: str,
) -> None:
    fig_width = max(10, 0.9 * len(column_keys) + 6)
    fig_height = max(8, 0.35 * len(prompt_rows) + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    x_labels = [f"L{layer_percent}%\n{CUSTOM_LABELS.get(lora_key, lora_key)}" for layer_percent, lora_key in column_keys]
    y_labels = [row["prompt_id"] for row in prompt_rows]

    ax.set_xticks(np.arange(len(column_keys)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(prompt_rows)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Oracle Model / Layer")
    ax.set_ylabel("Context Prompt ID")
    ax.set_title(title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average Accuracy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_bottom_prompts(
    prompt_rows: list[dict],
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    top_rows = prompt_rows[:top_k]
    bottom_rows = list(reversed(prompt_rows[-top_k:]))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.45 * top_k + 2)), sharex=True)
    plot_specs = [
        (axes[0], top_rows, "Top Context Prompts", "#2ca02c"),
        (axes[1], bottom_rows, "Bottom Context Prompts", "#d62728"),
    ]

    for ax, rows, subtitle, color in plot_specs:
        y_labels = [row["prompt_id"] for row in rows]
        values = [row["overall_mean"] for row in rows]
        bars = ax.barh(y_labels, values, color=color, alpha=0.9)
        ax.set_xlim(0.0, 1.05)
        ax.set_title(subtitle)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        default="../taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test",
    )
    parser.add_argument("--output_dir", type=str, default="./images/taboo")
    parser.add_argument("--sequence", action="store_true", help="Use full_sequence_responses instead of token_responses")
    parser.add_argument("--required_verbalizer_prompt", type=str, default=None)
    parser.add_argument("--act_key", type=str, default="lora")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heatmap", "top_bottom", "both"],
        default="both",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise ValueError(f"json_dir does not exist: {json_dir}")

    model_name, column_keys, prompt_rows = load_prompt_performance(
        json_dir=json_dir,
        sequence=args.sequence,
        required_verbalizer_prompt=args.required_verbalizer_prompt,
        required_act_key=args.act_key,
    )

    data_dir_label = json_dir.name
    token_or_seq = "sequence" if args.sequence else "token"
    base_name = f"taboo_context_prompt_performance_{sanitize_name(model_name)}_{data_dir_label}_{token_or_seq}_{args.act_key}"

    csv_path = Path(args.output_dir) / f"{base_name}.csv"
    save_prompt_csv(csv_path, prompt_rows, column_keys)
    print(f"Saved CSV: {csv_path}")

    if args.mode in ("heatmap", "both"):
        heatmap_path = Path(args.output_dir) / f"{base_name}_heatmap.png"
        heatmap_title = f"Context Prompt Performance Heatmap | {model_name} | {token_or_seq} | act={args.act_key}"
        plot_heatmap(
            matrix=build_heatmap_matrix(prompt_rows, column_keys),
            prompt_rows=prompt_rows,
            column_keys=column_keys,
            output_path=heatmap_path,
            title=heatmap_title,
        )
        print(f"Saved: {heatmap_path}")

    if args.mode in ("top_bottom", "both"):
        top_bottom_path = Path(args.output_dir) / f"{base_name}_top_bottom.png"
        top_bottom_title = f"Context Prompt Ranking | {model_name} | {token_or_seq} | act={args.act_key}"
        plot_top_bottom_prompts(
            prompt_rows=prompt_rows,
            output_path=top_bottom_path,
            title=top_bottom_title,
            top_k=args.top_k,
        )
        print(f"Saved: {top_bottom_path}")


if __name__ == "__main__":
    main()
