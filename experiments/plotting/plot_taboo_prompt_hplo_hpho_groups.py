import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_taboo_eval_context_prompt_performance import compute_pearson, compute_spearman, sanitize_name


GROUP_ORDER = ["HPLO", "HPHO", "HIGH_PROBE_MID", "NON_HIGH_PROBE"]
GROUP_COLORS = {
    "HPLO": "#d62728",
    "HPHO": "#2ca02c",
    "HIGH_PROBE_MID": "#1f77b4",
    "NON_HIGH_PROBE": "#bdbdbd",
}


def parse_alignment_filename(csv_path: Path) -> tuple[str, int, str]:
    match = re.match(
        r"prompt_probe_oracle_alignment_(?P<source>[a-zA-Z0-9_]+)_layer_(?P<layer>\d+)_(?P<suffix>.+)\.csv",
        csv_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected alignment csv filename: {csv_path.name}")
    return match.group("source"), int(match.group("layer")), match.group("suffix")


def discover_alignment_csvs(input_dir: Path, source_names: list[str]) -> dict[str, dict[int, Path]]:
    discovered = {source_name: {} for source_name in source_names}
    csv_paths = sorted(input_dir.glob("prompt_probe_oracle_alignment_*.csv"))
    if len(csv_paths) == 0:
        raise ValueError(f"No alignment CSVs found in {input_dir}")

    suffix = None
    for csv_path in csv_paths:
        source_name, layer_percent, parsed_suffix = parse_alignment_filename(csv_path)
        if source_name not in discovered:
            continue
        if suffix is None:
            suffix = parsed_suffix
        if parsed_suffix != suffix:
            raise ValueError(f"Mixed alignment suffixes found in {input_dir}: {suffix} vs {parsed_suffix}")
        discovered[source_name][layer_percent] = csv_path

    for source_name in source_names:
        if len(discovered[source_name]) == 0:
            raise ValueError(f"No alignment CSVs found for source '{source_name}' in {input_dir}")

    return discovered


def load_alignment_rows(csv_path: Path, probe_metric: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if probe_metric not in reader.fieldnames:
        raise ValueError(f"Probe metric '{probe_metric}' not found in {csv_path}. Available columns: {reader.fieldnames}")

    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "oracle_accuracy": float(row["oracle_accuracy"]),
                "probe_score": float(row[probe_metric]),
                "primary_category": row["primary_category"],
                "secondary_tags": row["secondary_tags"],
            }
        )
    return parsed_rows


def assign_probe_controlled_groups(
    rows: list[dict],
    high_probe_quantile: float,
    group_quantile: float,
) -> tuple[list[dict], dict]:
    probe_scores = np.array([row["probe_score"] for row in rows], dtype=float)
    oracle_accuracies = np.array([row["oracle_accuracy"] for row in rows], dtype=float)

    slope, intercept = np.polyfit(probe_scores, oracle_accuracies, deg=1)
    predicted = slope * probe_scores + intercept
    residuals = oracle_accuracies - predicted

    high_probe_threshold = float(np.quantile(probe_scores, high_probe_quantile))
    high_probe_mask = probe_scores >= high_probe_threshold
    high_probe_residuals = residuals[high_probe_mask]

    low_residual_threshold = float(np.quantile(high_probe_residuals, group_quantile))
    high_residual_threshold = float(np.quantile(high_probe_residuals, 1.0 - group_quantile))

    residual_mean = float(residuals.mean())
    residual_std = float(residuals.std(ddof=0))

    grouped_rows = []
    for row, predicted_oracle, residual, is_high_probe in zip(rows, predicted, residuals, high_probe_mask):
        grouped_row = dict(row)
        grouped_row["predicted_oracle_accuracy"] = float(predicted_oracle)
        grouped_row["oracle_residual"] = float(residual)
        grouped_row["oracle_residual_zscore"] = float((residual - residual_mean) / residual_std)
        grouped_row["is_high_probe"] = int(is_high_probe)

        if not is_high_probe:
            grouped_row["group_label"] = "NON_HIGH_PROBE"
        elif residual <= low_residual_threshold:
            grouped_row["group_label"] = "HPLO"
        elif residual >= high_residual_threshold:
            grouped_row["group_label"] = "HPHO"
        else:
            grouped_row["group_label"] = "HIGH_PROBE_MID"

        grouped_rows.append(grouped_row)

    grouped_rows.sort(key=lambda row: (GROUP_ORDER.index(row["group_label"]), row["oracle_residual"]))

    summary = {
        "num_prompts": len(grouped_rows),
        "high_probe_quantile": high_probe_quantile,
        "group_quantile": group_quantile,
        "high_probe_threshold": high_probe_threshold,
        "low_residual_threshold": low_residual_threshold,
        "high_residual_threshold": high_residual_threshold,
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "group_counts": {
            group_label: sum(row["group_label"] == group_label for row in grouped_rows)
            for group_label in GROUP_ORDER
        },
    }
    return grouped_rows, summary


def save_grouped_rows_csv(output_path: Path, grouped_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "group_label",
        "is_high_probe",
        "probe_score",
        "oracle_accuracy",
        "predicted_oracle_accuracy",
        "oracle_residual",
        "oracle_residual_zscore",
        "primary_category",
        "secondary_tags",
        "prompt_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def draw_group_scatter(
    grouped_rows: list[dict],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    probe_scores = np.array([row["probe_score"] for row in grouped_rows], dtype=float)
    oracle_accuracies = np.array([row["oracle_accuracy"] for row in grouped_rows], dtype=float)
    predicted = np.array([row["predicted_oracle_accuracy"] for row in grouped_rows], dtype=float)

    for group_label in GROUP_ORDER:
        rows = [row for row in grouped_rows if row["group_label"] == group_label]
        if len(rows) == 0:
            continue
        ax.scatter(
            [row["probe_score"] for row in rows],
            [row["oracle_accuracy"] for row in rows],
            alpha=0.85,
            s=45,
            color=GROUP_COLORS[group_label],
            label=f"{group_label} (n={len(rows)})",
        )

    order = np.argsort(probe_scores)
    ax.plot(probe_scores[order], predicted[order], color="black", linewidth=1.8, label="Oracle ~ probe fit")
    ax.set_title(title)
    ax.set_xlabel("Probe score")
    ax.set_ylabel("Oracle secret-word accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_group_overlap_and_residual_correlation(
    source_a_rows: list[dict],
    source_b_rows: list[dict],
    source_a_name: str,
    source_b_name: str,
) -> tuple[dict, list[dict]]:
    source_a_map = {row["prompt_text"]: row for row in source_a_rows}
    source_b_map = {row["prompt_text"]: row for row in source_b_rows}
    common_prompts = sorted(set(source_a_map.keys()) & set(source_b_map.keys()))
    if len(common_prompts) == 0:
        raise ValueError(f"No common prompts found between {source_a_name} and {source_b_name}")

    aligned_rows = []
    residuals_a = []
    residuals_b = []
    shared_high_probe_residuals_a = []
    shared_high_probe_residuals_b = []

    for prompt_text in common_prompts:
        row_a = source_a_map[prompt_text]
        row_b = source_b_map[prompt_text]
        aligned_rows.append(
            {
                "prompt_text": prompt_text,
                f"{source_a_name}_prompt_id": row_a["prompt_id"],
                f"{source_b_name}_prompt_id": row_b["prompt_id"],
                f"{source_a_name}_probe_score": row_a["probe_score"],
                f"{source_b_name}_probe_score": row_b["probe_score"],
                f"{source_a_name}_oracle_accuracy": row_a["oracle_accuracy"],
                f"{source_b_name}_oracle_accuracy": row_b["oracle_accuracy"],
                f"{source_a_name}_oracle_residual": row_a["oracle_residual"],
                f"{source_b_name}_oracle_residual": row_b["oracle_residual"],
                f"{source_a_name}_group_label": row_a["group_label"],
                f"{source_b_name}_group_label": row_b["group_label"],
                "shared_high_probe": int(row_a["is_high_probe"] == 1 and row_b["is_high_probe"] == 1),
            }
        )
        residuals_a.append(row_a["oracle_residual"])
        residuals_b.append(row_b["oracle_residual"])
        if row_a["is_high_probe"] == 1 and row_b["is_high_probe"] == 1:
            shared_high_probe_residuals_a.append(row_a["oracle_residual"])
            shared_high_probe_residuals_b.append(row_b["oracle_residual"])

    residuals_a_array = np.array(residuals_a, dtype=float)
    residuals_b_array = np.array(residuals_b, dtype=float)
    shared_high_probe_a_array = np.array(shared_high_probe_residuals_a, dtype=float)
    shared_high_probe_b_array = np.array(shared_high_probe_residuals_b, dtype=float)

    hplo_a = {row["prompt_text"] for row in source_a_rows if row["group_label"] == "HPLO"}
    hplo_b = {row["prompt_text"] for row in source_b_rows if row["group_label"] == "HPLO"}
    hpho_a = {row["prompt_text"] for row in source_a_rows if row["group_label"] == "HPHO"}
    hpho_b = {row["prompt_text"] for row in source_b_rows if row["group_label"] == "HPHO"}

    summary = {
        "num_common_prompts": len(common_prompts),
        "num_shared_high_probe_prompts": int(len(shared_high_probe_a_array)),
        "all_prompt_residual_pearson": compute_pearson(residuals_a_array, residuals_b_array),
        "all_prompt_residual_spearman": compute_spearman(residuals_a_array, residuals_b_array),
        "shared_high_probe_residual_pearson": compute_pearson(shared_high_probe_a_array, shared_high_probe_b_array),
        "shared_high_probe_residual_spearman": compute_spearman(shared_high_probe_a_array, shared_high_probe_b_array),
        "hplo_overlap_count": len(hplo_a & hplo_b),
        "hplo_jaccard": len(hplo_a & hplo_b) / len(hplo_a | hplo_b),
        "hpho_overlap_count": len(hpho_a & hpho_b),
        "hpho_jaccard": len(hpho_a & hpho_b) / len(hpho_a | hpho_b),
        "hplo_overlap_prompts": sorted(hplo_a & hplo_b),
        "hpho_overlap_prompts": sorted(hpho_a & hpho_b),
    }
    return summary, aligned_rows


def save_cross_source_alignment_csv(output_path: Path, aligned_rows: list[dict], source_a_name: str, source_b_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shared_high_probe",
        f"{source_a_name}_prompt_id",
        f"{source_b_name}_prompt_id",
        f"{source_a_name}_probe_score",
        f"{source_b_name}_probe_score",
        f"{source_a_name}_oracle_accuracy",
        f"{source_b_name}_oracle_accuracy",
        f"{source_a_name}_oracle_residual",
        f"{source_b_name}_oracle_residual",
        f"{source_a_name}_group_label",
        f"{source_b_name}_group_label",
        "prompt_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            aligned_rows,
            key=lambda item: (item["shared_high_probe"], item[f"{source_a_name}_oracle_residual"] + item[f"{source_b_name}_oracle_residual"]),
            reverse=True,
        ):
            writer.writerow({field: row[field] for field in fieldnames})


def draw_cross_source_residual_scatter(
    aligned_rows: list[dict],
    source_a_name: str,
    source_b_name: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    both_hplo = [
        row for row in aligned_rows
        if row[f"{source_a_name}_group_label"] == "HPLO" and row[f"{source_b_name}_group_label"] == "HPLO"
    ]
    both_hpho = [
        row for row in aligned_rows
        if row[f"{source_a_name}_group_label"] == "HPHO" and row[f"{source_b_name}_group_label"] == "HPHO"
    ]
    discordant = [
        row for row in aligned_rows
        if row not in both_hplo and row not in both_hpho and row["shared_high_probe"] == 1
    ]
    rest = [row for row in aligned_rows if row["shared_high_probe"] == 0]

    groups = [
        ("both HPLO", both_hplo, "#d62728"),
        ("both HPHO", both_hpho, "#2ca02c"),
        ("shared high-probe, discordant", discordant, "#ff7f0e"),
        ("not shared high-probe", rest, "#bdbdbd"),
    ]

    for label, rows, color in groups:
        if len(rows) == 0:
            continue
        ax.scatter(
            [row[f"{source_a_name}_oracle_residual"] for row in rows],
            [row[f"{source_b_name}_oracle_residual"] for row in rows],
            color=color,
            alpha=0.8,
            s=45,
            label=f"{label} (n={len(rows)})",
        )

    all_x = np.array([row[f"{source_a_name}_oracle_residual"] for row in aligned_rows], dtype=float)
    all_y = np.array([row[f"{source_b_name}_oracle_residual"] for row in aligned_rows], dtype=float)
    line_min = min(all_x.min(), all_y.min())
    line_max = max(all_x.max(), all_y.max())
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1.4, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(f"{source_a_name} Oracle residual")
    ax.set_ylabel(f"{source_b_name} Oracle residual")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_overlap_summary_plot(overlap_rows: list[dict], output_path: Path, source_a_name: str, source_b_name: str) -> None:
    layers = [row["layer_percent"] for row in overlap_rows]
    hplo_jaccard = [row["hplo_jaccard"] for row in overlap_rows]
    hpho_jaccard = [row["hpho_jaccard"] for row in overlap_rows]
    residual_pearson = [row["shared_high_probe_residual_pearson"] for row in overlap_rows]

    x = np.arange(len(layers))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bars_hplo = ax.bar(x - width, hplo_jaccard, width=width, color="#d62728", label="HPLO Jaccard")
    bars_hpho = ax.bar(x, hpho_jaccard, width=width, color="#2ca02c", label="HPHO Jaccard")
    bars_res = ax.bar(x + width, residual_pearson, width=width, color="#1f77b4", label="Shared high-probe residual Pearson")

    ax.bar_label(bars_hplo, labels=[f"{value:.3f}" for value in hplo_jaccard], padding=3, fontsize=8)
    ax.bar_label(bars_hpho, labels=[f"{value:.3f}" for value in hpho_jaccard], padding=3, fontsize=8)
    ax.bar_label(bars_res, labels=[f"{value:.3f}" for value in residual_pearson], padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{layer}%" for layer in layers])
    ax.set_ylabel("Score")
    ax.set_title(f"{source_a_name} vs {source_b_name} | HPLO/HPHO overlap and residual correlation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--probe_metric", type=str, default="binary_linear_target_prob_mean")
    parser.add_argument("--source_names", type=str, nargs=2, default=["hider", "guesser"])
    parser.add_argument("--high_probe_quantile", type=float, default=0.5)
    parser.add_argument("--group_quantile", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    discovered = discover_alignment_csvs(input_dir, args.source_names)
    suffix = parse_alignment_filename(next(iter(discovered[args.source_names[0]].values())))[2]

    default_output_dir = (
        Path("experiments/plotting/images")
        / "taboo_prompt_hplo_hpho_groups"
        / f"{sanitize_name(input_dir.name)}_{sanitize_name(args.probe_metric)}"
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = {source_name: {} for source_name in args.source_names}
    grouped_summary = {source_name: {} for source_name in args.source_names}

    for source_name in args.source_names:
        for layer_percent, csv_path in sorted(discovered[source_name].items()):
            rows = load_alignment_rows(csv_path, args.probe_metric)
            grouped_rows, summary = assign_probe_controlled_groups(
                rows=rows,
                high_probe_quantile=args.high_probe_quantile,
                group_quantile=args.group_quantile,
            )
            grouped_data[source_name][layer_percent] = grouped_rows
            grouped_summary[source_name][layer_percent] = summary

            grouped_csv_path = output_dir / (
                f"hplo_hpho_groups_{source_name}_layer_{layer_percent}_{sanitize_name(args.probe_metric)}_{suffix}.csv"
            )
            save_grouped_rows_csv(grouped_csv_path, grouped_rows)

            scatter_path = output_dir / (
                f"hplo_hpho_groups_{source_name}_layer_{layer_percent}_{sanitize_name(args.probe_metric)}_{suffix}.png"
            )
            draw_group_scatter(
                grouped_rows=grouped_rows,
                output_path=scatter_path,
                title=(
                    f"{source_name} | Layer {layer_percent}% | {args.probe_metric}\n"
                    f"high_probe_q={args.high_probe_quantile}, group_q={args.group_quantile}"
                ),
            )

    source_a_name, source_b_name = args.source_names
    overlap_rows = []
    cross_source_details = {}
    shared_layers = sorted(set(grouped_data[source_a_name].keys()) & set(grouped_data[source_b_name].keys()))
    for layer_percent in shared_layers:
        overlap_summary, aligned_rows = compute_group_overlap_and_residual_correlation(
            source_a_rows=grouped_data[source_a_name][layer_percent],
            source_b_rows=grouped_data[source_b_name][layer_percent],
            source_a_name=source_a_name,
            source_b_name=source_b_name,
        )
        overlap_summary["layer_percent"] = layer_percent
        overlap_rows.append(overlap_summary)
        cross_source_details[str(layer_percent)] = overlap_summary

        cross_source_csv_path = output_dir / (
            f"hplo_hpho_cross_source_alignment_layer_{layer_percent}_{sanitize_name(args.probe_metric)}_{suffix}.csv"
        )
        save_cross_source_alignment_csv(
            output_path=cross_source_csv_path,
            aligned_rows=aligned_rows,
            source_a_name=source_a_name,
            source_b_name=source_b_name,
        )

        residual_scatter_path = output_dir / (
            f"hplo_hpho_cross_source_residuals_layer_{layer_percent}_{sanitize_name(args.probe_metric)}_{suffix}.png"
        )
        draw_cross_source_residual_scatter(
            aligned_rows=aligned_rows,
            source_a_name=source_a_name,
            source_b_name=source_b_name,
            output_path=residual_scatter_path,
            title=f"{source_a_name} vs {source_b_name} residuals | Layer {layer_percent}%",
        )

    overlap_summary_csv = output_dir / f"hplo_hpho_overlap_summary_{sanitize_name(args.probe_metric)}_{suffix}.csv"
    with open(overlap_summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer_percent",
            "num_common_prompts",
            "num_shared_high_probe_prompts",
            "all_prompt_residual_pearson",
            "all_prompt_residual_spearman",
            "shared_high_probe_residual_pearson",
            "shared_high_probe_residual_spearman",
            "hplo_overlap_count",
            "hplo_jaccard",
            "hpho_overlap_count",
            "hpho_jaccard",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in overlap_rows:
            writer.writerow({field: row[field] for field in fieldnames})

    overlap_summary_json = output_dir / f"hplo_hpho_overlap_summary_{sanitize_name(args.probe_metric)}_{suffix}.json"
    with open(overlap_summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "input_dir": str(input_dir),
                    "probe_metric": args.probe_metric,
                    "source_names": args.source_names,
                    "high_probe_quantile": args.high_probe_quantile,
                    "group_quantile": args.group_quantile,
                    "suffix": suffix,
                },
                "grouped_summary": grouped_summary,
                "cross_source_summary_by_layer": cross_source_details,
            },
            f,
            indent=2,
        )

    overlap_plot_path = output_dir / f"hplo_hpho_overlap_plot_{sanitize_name(args.probe_metric)}_{suffix}.png"
    draw_overlap_summary_plot(
        overlap_rows=sorted(overlap_rows, key=lambda row: row["layer_percent"]),
        output_path=overlap_plot_path,
        source_a_name=source_a_name,
        source_b_name=source_b_name,
    )

    print(f"Saved grouped outputs to {output_dir}")


if __name__ == "__main__":
    main()
