import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_taboo_context_prompt_taxonomy import PRIMARY_CATEGORIES, classify_prompt


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_probe_results(input_json: Path) -> dict:
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_metrics(data: dict) -> list[str]:
    for source_info in data["sources"].values():
        for layer_info in source_info["layers"].values():
            prompt_rows = layer_info["prompt_rows"]
            if len(prompt_rows) == 0:
                continue
            return [key for key in prompt_rows[0].keys() if key not in ("prompt_id", "prompt_text")]
    raise ValueError("No prompt_rows found in probe evaluation json")


def aggregate_category_metric(
    data: dict,
    metric: str,
    source_names: list[str],
) -> list[dict]:
    summary_rows = []

    for source_name in source_names:
        if source_name not in data["sources"]:
            raise ValueError(f"Unknown source_name: {source_name}")

        for layer_percent_str, layer_info in data["sources"][source_name]["layers"].items():
            prompt_rows = layer_info["prompt_rows"]
            if len(prompt_rows) == 0:
                continue
            if metric not in prompt_rows[0]:
                raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(prompt_rows[0].keys())}")

            for category in PRIMARY_CATEGORIES:
                category_prompt_rows = [
                    row
                    for row in prompt_rows
                    if classify_prompt(row["prompt_text"])[0] == category
                ]
                if len(category_prompt_rows) == 0:
                    continue

                values = np.array([row[metric] for row in category_prompt_rows], dtype=float)
                summary_rows.append(
                    {
                        "source": source_name,
                        "layer_percent": int(layer_percent_str),
                        "category": category,
                        "num_prompts": len(category_prompt_rows),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )

    return summary_rows


def save_category_summary_csv(output_path: Path, summary_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "layer_percent", "category", "num_prompts", "mean", "std", "min", "max"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def build_summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    return {(row["source"], row["layer_percent"], row["category"]): row for row in summary_rows}


def draw_category_barplots(
    summary_rows: list[dict],
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    lookup = build_summary_lookup(summary_rows)
    source_names = sorted({row["source"] for row in summary_rows})
    layer_percents = sorted({row["layer_percent"] for row in summary_rows})

    fig, axes = plt.subplots(len(source_names), 1, figsize=(18, 5.5 * len(source_names)), sharex=True)
    if len(source_names) == 1:
        axes = [axes]

    x = np.arange(len(PRIMARY_CATEGORIES))
    width = 0.24 if len(layer_percents) >= 3 else 0.32
    offsets = np.linspace(-(len(layer_percents) - 1) / 2, (len(layer_percents) - 1) / 2, len(layer_percents)) * width
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, source_name in zip(axes, source_names):
        for idx, layer_percent in enumerate(layer_percents):
            means = []
            for category in PRIMARY_CATEGORIES:
                row = lookup.get((source_name, layer_percent, category))
                means.append(np.nan if row is None else row["mean"])

            bars = ax.bar(
                x + offsets[idx],
                means,
                width=width,
                color=colors[idx % len(colors)],
                alpha=0.9,
                label=f"Layer {layer_percent}%",
            )
            labels = ["" if np.isnan(value) else f"{value:.3f}" for value in means]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=7, rotation=90)

        ax.set_title(source_name)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(PRIMARY_CATEGORIES, rotation=25, ha="right")
    axes[-1].set_xlabel("Context Prompt Taxonomy")
    fig.suptitle(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_category_heatmap(
    summary_rows: list[dict],
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    lookup = build_summary_lookup(summary_rows)
    source_names = sorted({row["source"] for row in summary_rows})
    layer_percents = sorted({row["layer_percent"] for row in summary_rows})
    column_keys = [(source_name, layer_percent) for source_name in source_names for layer_percent in layer_percents]

    matrix = np.full((len(PRIMARY_CATEGORIES), len(column_keys)), np.nan, dtype=float)
    for row_idx, category in enumerate(PRIMARY_CATEGORIES):
        for col_idx, (source_name, layer_percent) in enumerate(column_keys):
            row = lookup.get((source_name, layer_percent, category))
            if row is not None:
                matrix[row_idx, col_idx] = row["mean"]

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(column_keys)), 7.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Source / Layer")
    ax.set_ylabel("Context Prompt Taxonomy")
    ax.set_xticks(range(len(column_keys)))
    ax.set_xticklabels([f"{source}\nL{layer_percent}%" for source, layer_percent in column_keys], rotation=25, ha="right")
    ax.set_yticks(range(len(PRIMARY_CATEGORIES)))
    ax.set_yticklabels(PRIMARY_CATEGORIES)
    fig.colorbar(im, ax=ax, shrink=0.85, label=metric)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--metric", type=str, default="linear_target_prob_mean")
    parser.add_argument("--source_names", type=str, nargs="+", default=["hider", "guesser"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["bars", "heatmap", "both"], default="both")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    data = load_probe_results(input_json)
    available_metrics = get_available_metrics(data)
    if args.metric not in available_metrics:
        raise ValueError(f"Unsupported metric '{args.metric}'. Available metrics: {available_metrics}")

    summary_rows = aggregate_category_metric(
        data=data,
        metric=args.metric,
        source_names=args.source_names,
    )

    model_name_str = sanitize_name(data["config"]["model_name"].split("/")[-1])
    lang_suffix = f"_{data['config']['lang_type']}" if data["config"]["lang_type"] else ""
    default_output_dir = (
        Path("experiments/plotting/images")
        / "taboo_context_prompt_probe_taxonomy"
        / f"{model_name_str}_{data['config']['prompt_type']}{lang_suffix}_{data['config']['dataset_type']}"
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"probe_taxonomy_{sanitize_name(args.metric)}"
    csv_path = output_dir / f"{stem}_summary.csv"
    save_category_summary_csv(csv_path, summary_rows)
    print(f"Saved CSV: {csv_path}")

    title = (
        f"{data['config']['model_name']} | {data['config']['prompt_type']} | "
        f"{data['config']['dataset_type']} | {args.metric}"
    )

    if args.mode in ("bars", "both"):
        bar_path = output_dir / f"{stem}_bars.png"
        draw_category_barplots(
            summary_rows=summary_rows,
            metric=args.metric,
            output_path=bar_path,
            title=title,
        )
        print(f"Saved: {bar_path}")

    if args.mode in ("heatmap", "both"):
        heatmap_path = output_dir / f"{stem}_heatmap.png"
        draw_category_heatmap(
            summary_rows=summary_rows,
            metric=args.metric,
            output_path=heatmap_path,
            title=title,
        )
        print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
