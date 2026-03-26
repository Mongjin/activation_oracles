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


def strip_variant_suffix(value: str) -> str:
    for suffix in ["_swapped", "_hider", "_guesser"]:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def infer_run_label(json_dir: Path, fallback: str) -> str:
    name = json_dir.name.lower()
    if "swapped" in name or "role_swap" in name:
        return "guesser"
    if "baseline" in name:
        return "baseline"
    if "open_ended" in name:
        return "hider"
    return fallback


def build_prompt_perf_base_name(json_dir: Path, token_or_seq: str, act_key: str) -> str:
    run_tag = sanitize_name(strip_variant_suffix(json_dir.name))
    return f"taboo_context_prompt_perf_{run_tag}_{token_or_seq}_{sanitize_name(act_key)}"


def build_prompt_compare_base_name(
    json_dir: Path,
    primary_label: str,
    compare_label: str,
    token_or_seq: str,
    act_key: str,
) -> str:
    run_tag = sanitize_name(strip_variant_suffix(json_dir.name))
    return (
        f"taboo_context_prompt_compare_{run_tag}_"
        f"{sanitize_name(primary_label)}_vs_{sanitize_name(compare_label)}_"
        f"{token_or_seq}_{sanitize_name(act_key)}"
    )


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


def compute_pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.shape != values_b.shape:
        raise ValueError("Pearson inputs must have the same shape")
    if values_a.size == 0:
        raise ValueError("Pearson inputs must be non-empty")
    return float(np.corrcoef(values_a, values_b)[0, 1])


def compute_descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def compute_spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    ranks_a = compute_descending_ranks(values_a)
    ranks_b = compute_descending_ranks(values_b)
    return compute_pearson(ranks_a, ranks_b)


def compare_prompt_rows(
    primary_prompt_rows: list[dict],
    compare_prompt_rows: list[dict],
    primary_label: str,
    compare_label: str,
) -> tuple[list[dict], dict[str, float | int]]:
    primary_map = {row["prompt_text"]: row for row in primary_prompt_rows}
    compare_map = {row["prompt_text"]: row for row in compare_prompt_rows}

    common_prompts = sorted(set(primary_map.keys()) & set(compare_map.keys()))
    if len(common_prompts) == 0:
        raise ValueError("No common context prompts found between the two runs")

    primary_only = sorted(set(primary_map.keys()) - set(compare_map.keys()))
    compare_only = sorted(set(compare_map.keys()) - set(primary_map.keys()))

    aligned_rows = []
    primary_values = []
    compare_values = []

    for prompt_text in common_prompts:
        primary_row = primary_map[prompt_text]
        compare_row = compare_map[prompt_text]
        primary_values.append(primary_row["overall_mean"])
        compare_values.append(compare_row["overall_mean"])
        aligned_rows.append(
            {
                "prompt_text": prompt_text,
                "primary_prompt_id": primary_row["prompt_id"],
                "compare_prompt_id": compare_row["prompt_id"],
                "primary_overall_mean": primary_row["overall_mean"],
                "compare_overall_mean": compare_row["overall_mean"],
            }
        )

    primary_array = np.array(primary_values, dtype=float)
    compare_array = np.array(compare_values, dtype=float)
    pearson = compute_pearson(primary_array, compare_array)
    spearman = compute_spearman(primary_array, compare_array)

    primary_ranks = compute_descending_ranks(primary_array)
    compare_ranks = compute_descending_ranks(compare_array)

    for idx, row in enumerate(aligned_rows):
        row["primary_rank"] = int(primary_ranks[idx])
        row["compare_rank"] = int(compare_ranks[idx])
        row["mean_delta"] = float(row["primary_overall_mean"] - row["compare_overall_mean"])
        row["abs_mean_delta"] = abs(row["mean_delta"])
        row["rank_delta"] = int(row["primary_rank"] - row["compare_rank"])
        row["abs_rank_delta"] = abs(row["rank_delta"])

    top_k = min(20, len(aligned_rows))
    primary_top = {row["prompt_text"] for row in sorted(aligned_rows, key=lambda row: row["primary_rank"])[:top_k]}
    compare_top = {row["prompt_text"] for row in sorted(aligned_rows, key=lambda row: row["compare_rank"])[:top_k]}
    primary_bottom = {row["prompt_text"] for row in sorted(aligned_rows, key=lambda row: row["primary_rank"], reverse=True)[:top_k]}
    compare_bottom = {row["prompt_text"] for row in sorted(aligned_rows, key=lambda row: row["compare_rank"], reverse=True)[:top_k]}

    summary = {
        "primary_label": primary_label,
        "compare_label": compare_label,
        "num_common_prompts": len(common_prompts),
        "num_primary_only_prompts": len(primary_only),
        "num_compare_only_prompts": len(compare_only),
        "pearson_overall_mean": pearson,
        "spearman_overall_mean": spearman,
        "top20_overlap_count": len(primary_top & compare_top),
        "bottom20_overlap_count": len(primary_bottom & compare_bottom),
    }
    return aligned_rows, summary


def save_compare_csv(output_path: Path, aligned_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "primary_prompt_id",
        "compare_prompt_id",
        "primary_overall_mean",
        "compare_overall_mean",
        "mean_delta",
        "abs_mean_delta",
        "primary_rank",
        "compare_rank",
        "rank_delta",
        "abs_rank_delta",
        "prompt_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(aligned_rows, key=lambda item: item["primary_rank"]):
            writer.writerow({field: row[field] for field in fieldnames})


def save_compare_summary(output_path: Path, summary: dict[str, float | int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_compare_scatter(
    aligned_rows: list[dict],
    summary: dict[str, float | int],
    output_path: Path,
) -> None:
    primary_label = str(summary["primary_label"])
    compare_label = str(summary["compare_label"])
    x_values = np.array([row["primary_overall_mean"] for row in aligned_rows], dtype=float)
    y_values = np.array([row["compare_overall_mean"] for row in aligned_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.scatter(x_values, y_values, alpha=0.7, s=36, color="#1f77b4")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(f"{primary_label} prompt mean accuracy")
    ax.set_ylabel(f"{compare_label} prompt mean accuracy")
    ax.set_title(
        f"Context Prompt Agreement\nPearson={summary['pearson_overall_mean']:.3f}, "
        f"Spearman={summary['spearman_overall_mean']:.3f}"
    )
    ax.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prompt_gap_bars(
    aligned_rows: list[dict],
    primary_label: str,
    compare_label: str,
    output_path: Path,
    top_k: int,
) -> None:
    rows = sorted(aligned_rows, key=lambda row: row["abs_mean_delta"], reverse=True)[:top_k]
    y_labels = [row["primary_prompt_id"] for row in rows]
    deltas = [row["mean_delta"] for row in rows]
    colors = ["#2ca02c" if delta >= 0 else "#d62728" for delta in deltas]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * top_k + 2)))
    bars = ax.barh(y_labels, deltas, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.set_xlabel(f"Mean Accuracy Delta ({primary_label} - {compare_label})")
    ax.set_ylabel("Context Prompt ID")
    ax.set_title(f"Largest Context Prompt Gaps | Top {len(rows)} by |delta|")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{delta:.3f}" for delta in deltas], padding=3, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--compare_json_dir", type=str, default=None)
    parser.add_argument("--primary_label", type=str, default=None)
    parser.add_argument("--compare_label", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heatmap", "top_bottom", "both", "compare", "compare_only"],
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

    token_or_seq = "sequence" if args.sequence else "token"
    base_name = build_prompt_perf_base_name(json_dir, token_or_seq, args.act_key)

    if args.mode != "compare_only":
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

    if args.compare_json_dir is not None or args.mode in ("compare", "compare_only"):
        if args.compare_json_dir is None:
            raise ValueError("--compare_json_dir is required for compare or compare_only mode")

        compare_json_dir = Path(args.compare_json_dir)
        if not compare_json_dir.exists():
            raise ValueError(f"compare_json_dir does not exist: {compare_json_dir}")

        compare_model_name, compare_column_keys, compare_rows = load_prompt_performance(
            json_dir=compare_json_dir,
            sequence=args.sequence,
            required_verbalizer_prompt=args.required_verbalizer_prompt,
            required_act_key=args.act_key,
        )

        if model_name != compare_model_name:
            raise ValueError(f"Model mismatch: {model_name} != {compare_model_name}")
        if column_keys != compare_column_keys:
            raise ValueError("Column layout mismatch between primary and compare runs")

        primary_label = args.primary_label or infer_run_label(json_dir, "run_a")
        compare_label = args.compare_label or infer_run_label(compare_json_dir, "run_b")

        aligned_rows, summary = compare_prompt_rows(
            primary_prompt_rows=prompt_rows,
            compare_prompt_rows=compare_rows,
            primary_label=primary_label,
            compare_label=compare_label,
        )

        compare_base_name = build_prompt_compare_base_name(
            json_dir=json_dir,
            primary_label=primary_label,
            compare_label=compare_label,
            token_or_seq=token_or_seq,
            act_key=args.act_key,
        )

        compare_csv_path = Path(args.output_dir) / f"{compare_base_name}.csv"
        save_compare_csv(compare_csv_path, aligned_rows)
        print(f"Saved CSV: {compare_csv_path}")

        compare_summary_path = Path(args.output_dir) / f"{compare_base_name}_summary.json"
        save_compare_summary(compare_summary_path, summary)
        print(f"Saved summary: {compare_summary_path}")

        compare_scatter_path = Path(args.output_dir) / f"{compare_base_name}_scatter.png"
        plot_compare_scatter(aligned_rows, summary, compare_scatter_path)
        print(f"Saved: {compare_scatter_path}")

        compare_gap_path = Path(args.output_dir) / f"{compare_base_name}_largest_gaps.png"
        plot_prompt_gap_bars(
            aligned_rows=aligned_rows,
            primary_label=primary_label,
            compare_label=compare_label,
            output_path=compare_gap_path,
            top_k=args.top_k,
        )
        print(f"Saved: {compare_gap_path}")


if __name__ == "__main__":
    main()
