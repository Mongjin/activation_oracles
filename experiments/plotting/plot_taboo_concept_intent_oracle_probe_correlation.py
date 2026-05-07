import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_METRICS = [
    "concealment_rate",
    "suppression_rate",
    "non_disclosure_rate",
    "deception_mislead_rate",
    "guessing_game_rate",
    "direct_answer_rate",
    "any_concealment_like_rate",
]

FAMILY_LABELS = {
    "concealment_rate": "conceal",
    "suppression_rate": "suppress",
    "non_disclosure_rate": "non-disclose",
    "deception_mislead_rate": "deceive",
    "guessing_game_rate": "guess/game",
    "direct_answer_rate": "direct",
    "any_concealment_like_rate": "any conceal",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("?", "")


def parse_alignment_filename(csv_path: Path) -> tuple[str, int, str]:
    match = re.match(
        r"prompt_probe_oracle_alignment_(?P<source>[a-zA-Z0-9_]+)_layer_(?P<layer>\d+)_(?P<suffix>.+)\.csv",
        csv_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected alignment csv filename: {csv_path.name}")
    return match.group("source"), int(match.group("layer")), match.group("suffix")


def load_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def parse_family_rows(rows: list[dict]) -> list[dict]:
    parsed = []
    for row in rows:
        parsed_row = {
            "source_label": row["source_label"],
            "layer_percent": int(row["layer_percent"]),
            "act_key": row["act_key"],
            "verbalizer_prompt": row["verbalizer_prompt"],
            "output_type": row["output_type"],
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "total_responses": int(row["total_responses"]),
        }
        for metric_name in FAMILY_METRICS:
            parsed_row[metric_name] = float(row[metric_name])
        parsed.append(parsed_row)
    return parsed


def load_alignment_rows(alignment_dir: Path) -> tuple[list[dict], list[str]]:
    csv_paths = sorted(alignment_dir.glob("prompt_probe_oracle_alignment_*.csv"))
    if len(csv_paths) == 0:
        raise ValueError(f"No alignment CSVs found in {alignment_dir}")

    parsed_rows = []
    probe_metric_names = None
    for csv_path in csv_paths:
        source_name, layer_percent, suffix = parse_alignment_filename(csv_path)
        rows = load_csv_rows(csv_path)
        if probe_metric_names is None:
            probe_metric_names = [
                key for key in rows[0].keys()
                if key not in ("prompt_id", "prompt_text", "oracle_accuracy", "primary_category", "secondary_tags")
            ]
        for row in rows:
            parsed_row = {
                "source_label": source_name,
                "layer_percent": layer_percent,
                "alignment_suffix": suffix,
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "oracle_accuracy": float(row["oracle_accuracy"]),
            }
            for probe_metric in probe_metric_names:
                parsed_row[probe_metric] = float(row[probe_metric])
            parsed_rows.append(parsed_row)
    return parsed_rows, probe_metric_names


def merge_family_and_alignment_rows(
    family_rows: list[dict],
    alignment_rows: list[dict],
) -> list[dict]:
    alignment_map = {
        (row["source_label"], row["layer_percent"], row["prompt_text"]): row
        for row in alignment_rows
    }

    merged_rows = []
    for row in family_rows:
        key = (row["source_label"], row["layer_percent"], row["prompt_text"])
        if key not in alignment_map:
            continue
        merged = dict(row)
        merged.update(
            {
                "oracle_accuracy": alignment_map[key]["oracle_accuracy"],
                "alignment_suffix": alignment_map[key]["alignment_suffix"],
            }
        )
        for metric_name, value in alignment_map[key].items():
            if metric_name in ("source_label", "layer_percent", "alignment_suffix", "prompt_id", "prompt_text", "oracle_accuracy"):
                continue
            merged[metric_name] = value
        merged_rows.append(merged)

    if len(merged_rows) == 0:
        raise ValueError("No rows remained after merging family rates with alignment CSVs")
    return merged_rows


def compute_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return compute_pearson(compute_ranks(x), compute_ranks(y))


def summarize_correlations(
    merged_rows: list[dict],
    family_metrics: list[str],
    target_metrics: list[str],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in merged_rows:
        key = (
            row["source_label"],
            row["output_type"],
            row["verbalizer_prompt"],
            row["layer_percent"],
        )
        grouped[key].append(row)

    summary_rows = []
    for (source_label, output_type, verbalizer_prompt, layer_percent), rows in sorted(grouped.items()):
        for family_metric in family_metrics:
            x = np.array([row[family_metric] for row in rows], dtype=float)
            for target_metric in target_metrics:
                y = np.array([row[target_metric] for row in rows], dtype=float)
                summary_rows.append(
                    {
                        "source_label": source_label,
                        "output_type": output_type,
                        "verbalizer_prompt": verbalizer_prompt,
                        "layer_percent": layer_percent,
                        "family_metric": family_metric,
                        "target_metric": target_metric,
                        "num_prompts": len(rows),
                        "pearson": compute_pearson(x, y),
                        "spearman": compute_spearman(x, y),
                        "mean_family_metric": float(x.mean()),
                        "mean_target_metric": float(y.mean()),
                    }
                )
    return summary_rows


def draw_correlation_heatmap(
    summary_rows: list[dict],
    source_label: str,
    output_type: str,
    verbalizer_prompt: str,
    target_metric: str,
    value_key: str,
    output_path: Path,
) -> None:
    filtered_rows = [
        row for row in summary_rows
        if row["source_label"] == source_label
        and row["output_type"] == output_type
        and row["verbalizer_prompt"] == verbalizer_prompt
        and row["target_metric"] == target_metric
    ]
    layer_percents = sorted({row["layer_percent"] for row in filtered_rows})
    matrix = np.array(
        [
            [
                next(
                    row[value_key]
                    for row in filtered_rows
                    if row["layer_percent"] == layer_percent and row["family_metric"] == family_metric
                )
                for layer_percent in layer_percents
            ]
            for family_metric in FAMILY_METRICS
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(layer_percents)))
    ax.set_xticklabels([f"{layer}%" for layer in layer_percents])
    ax.set_yticks(range(len(FAMILY_METRICS)))
    ax.set_yticklabels([FAMILY_LABELS[metric] for metric in FAMILY_METRICS])
    ax.set_title(
        f"{value_key} | source={source_label} | output={output_type}\n"
        f"verbalizer={verbalizer_prompt} | target={target_metric}"
    )

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label(value_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_scatter_grid(
    merged_rows: list[dict],
    source_label: str,
    output_type: str,
    verbalizer_prompt: str,
    family_metric: str,
    probe_metric: str,
    output_path: Path,
) -> None:
    filtered_rows = [
        row for row in merged_rows
        if row["source_label"] == source_label
        and row["output_type"] == output_type
        and row["verbalizer_prompt"] == verbalizer_prompt
    ]
    layer_percents = sorted({row["layer_percent"] for row in filtered_rows})
    fig, axes = plt.subplots(len(layer_percents), 2, figsize=(11, 4.4 * len(layer_percents)), squeeze=False)

    for row_idx, layer_percent in enumerate(layer_percents):
        layer_rows = [row for row in filtered_rows if row["layer_percent"] == layer_percent]
        x = np.array([row[family_metric] for row in layer_rows], dtype=float)

        oracle_y = np.array([row["oracle_accuracy"] for row in layer_rows], dtype=float)
        oracle_r = compute_pearson(x, oracle_y)
        axes[row_idx][0].scatter(x, oracle_y, alpha=0.72, s=30, color="#1f77b4")
        axes[row_idx][0].set_xlabel(family_metric)
        axes[row_idx][0].set_ylabel("oracle_accuracy")
        axes[row_idx][0].set_title(f"Layer {layer_percent}% | oracle | r={oracle_r:.2f}")
        axes[row_idx][0].grid(alpha=0.3)

        probe_y = np.array([row[probe_metric] for row in layer_rows], dtype=float)
        probe_r = compute_pearson(x, probe_y)
        axes[row_idx][1].scatter(x, probe_y, alpha=0.72, s=30, color="#ff7f0e")
        axes[row_idx][1].set_xlabel(family_metric)
        axes[row_idx][1].set_ylabel(probe_metric)
        axes[row_idx][1].set_title(f"Layer {layer_percent}% | probe | r={probe_r:.2f}")
        axes[row_idx][1].grid(alpha=0.3)

    fig.suptitle(
        f"Family vs oracle/probe | source={source_label} | output={output_type} | verbalizer={verbalizer_prompt}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_rates_csv", type=str, required=True)
    parser.add_argument("--alignment_dir", type=str, required=True)
    parser.add_argument(
        "--probe_metrics",
        type=str,
        nargs="+",
        default=["binary_linear_target_prob_mean", "linear_target_prob_mean"],
    )
    parser.add_argument(
        "--scatter_family_metrics",
        type=str,
        nargs="+",
        default=["any_concealment_like_rate"],
        choices=FAMILY_METRICS,
    )
    parser.add_argument("--primary_probe_metric", type=str, default="binary_linear_target_prob_mean")
    parser.add_argument("--output_dir", type=str, default="./images/taboo_concept_intent_oracle_probe_correlation")
    args = parser.parse_args()

    family_rows = parse_family_rows(load_csv_rows(Path(args.family_rates_csv)))
    alignment_rows, available_probe_metrics = load_alignment_rows(Path(args.alignment_dir))
    for probe_metric in args.probe_metrics:
        if probe_metric not in available_probe_metrics:
            raise ValueError(f"Probe metric '{probe_metric}' not found in alignment CSVs")
    if args.primary_probe_metric not in available_probe_metrics:
        raise ValueError(f"Primary probe metric '{args.primary_probe_metric}' not found in alignment CSVs")

    merged_rows = merge_family_and_alignment_rows(family_rows, alignment_rows)
    target_metrics = ["oracle_accuracy"] + args.probe_metrics
    summary_rows = summarize_correlations(
        merged_rows=merged_rows,
        family_metrics=FAMILY_METRICS,
        target_metrics=target_metrics,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        output_path=output_dir / "concept_intent_oracle_probe_merged_rows.csv",
        rows=merged_rows,
        fieldnames=list(merged_rows[0].keys()),
    )
    save_csv(
        output_path=output_dir / "concept_intent_oracle_probe_correlation_summary.csv",
        rows=summary_rows,
        fieldnames=list(summary_rows[0].keys()),
    )

    source_labels = sorted({row["source_label"] for row in merged_rows})
    output_types = sorted({row["output_type"] for row in merged_rows})
    verbalizer_prompts = sorted({row["verbalizer_prompt"] for row in merged_rows})

    for source_label in source_labels:
        for output_type in output_types:
            for verbalizer_prompt in verbalizer_prompts:
                for target_metric in target_metrics:
                    for value_key in ["pearson", "spearman"]:
                        draw_correlation_heatmap(
                            summary_rows=summary_rows,
                            source_label=source_label,
                            output_type=output_type,
                            verbalizer_prompt=verbalizer_prompt,
                            target_metric=target_metric,
                            value_key=value_key,
                            output_path=output_dir / (
                                f"corr_heatmap_{sanitize_name(source_label)}_{sanitize_name(output_type)}_"
                                f"{sanitize_name(verbalizer_prompt)}_{sanitize_name(target_metric)}_{value_key}.png"
                            ),
                        )

                for family_metric in args.scatter_family_metrics:
                    draw_scatter_grid(
                        merged_rows=merged_rows,
                        source_label=source_label,
                        output_type=output_type,
                        verbalizer_prompt=verbalizer_prompt,
                        family_metric=family_metric,
                        probe_metric=args.primary_probe_metric,
                        output_path=output_dir / (
                            f"scatter_{sanitize_name(source_label)}_{sanitize_name(output_type)}_"
                            f"{sanitize_name(verbalizer_prompt)}_{sanitize_name(family_metric)}.png"
                        ),
                    )

    print(f"Saved concept-intent oracle/probe correlation plots to {output_dir}")


if __name__ == "__main__":
    main()
