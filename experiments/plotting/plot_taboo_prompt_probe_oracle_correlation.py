import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_taboo_context_prompt_taxonomy import classify_prompt
from plot_taboo_eval_context_prompt_performance import (
    compute_pearson,
    compute_spearman,
    load_prompt_performance,
    sanitize_name,
)


DEFAULT_PROBE_METRICS = [
    "linear_target_prob_mean",
    "linear_top1_acc",
    "mlp_target_prob_mean",
    "mlp_top1_acc",
]


def load_probe_results(input_json: Path) -> dict:
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_probe_metrics(data: dict) -> list[str]:
    for source_info in data["sources"].values():
        for layer_info in source_info["layers"].values():
            prompt_rows = layer_info["prompt_rows"]
            if len(prompt_rows) == 0:
                continue
            return [key for key in prompt_rows[0].keys() if key not in ("prompt_id", "prompt_text")]
    raise ValueError("No prompt_rows found in probe evaluation json")


def load_probe_layer_rows(
    data: dict,
    source_name: str,
    layer_percent: int,
    probe_metrics: list[str],
) -> list[dict]:
    if source_name not in data["sources"]:
        raise ValueError(f"Unknown source_name: {source_name}")

    layer_info = data["sources"][source_name]["layers"][str(layer_percent)]
    prompt_rows = layer_info["prompt_rows"]
    if len(prompt_rows) == 0:
        raise ValueError(f"No prompt rows found for source={source_name}, layer={layer_percent}")

    for metric in probe_metrics:
        if metric not in prompt_rows[0]:
            raise ValueError(f"Probe metric '{metric}' not found. Available metrics: {list(prompt_rows[0].keys())}")

    rows = []
    for row in prompt_rows:
        primary_category, secondary_tags = classify_prompt(row["prompt_text"])
        parsed_row = {
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "primary_category": primary_category,
            "secondary_tags": ",".join(secondary_tags),
        }
        for metric in probe_metrics:
            parsed_row[metric] = float(row[metric])
        rows.append(parsed_row)
    return rows


def load_oracle_layer_rows(
    json_dir: Path,
    layer_percent: int,
    required_verbalizer_prompt: str | None,
    required_act_key: str,
    sequence: bool,
) -> tuple[str, list[dict]]:
    model_name, column_keys, prompt_rows = load_prompt_performance(
        json_dir=json_dir,
        sequence=sequence,
        required_verbalizer_prompt=required_verbalizer_prompt,
        required_act_key=required_act_key,
    )

    layer_column_keys = [column_key for column_key in column_keys if column_key[0] == layer_percent]
    if len(layer_column_keys) == 0:
        raise ValueError(f"No Oracle accuracy columns found for layer {layer_percent} in {json_dir}")

    rows = []
    for row in prompt_rows:
        values = [
            row["row_means"][column_key]
            for column_key in layer_column_keys
            if not np.isnan(row["row_means"][column_key])
        ]
        oracle_accuracy = float(np.mean(values))
        rows.append(
            {
                "prompt_text": row["prompt_text"],
                "oracle_accuracy": oracle_accuracy,
            }
        )
    return model_name, rows


def build_aligned_rows(
    probe_rows: list[dict],
    oracle_rows: list[dict],
    probe_metrics: list[str],
) -> tuple[list[dict], list[dict]]:
    probe_map = {row["prompt_text"]: row for row in probe_rows}
    oracle_map = {row["prompt_text"]: row for row in oracle_rows}

    common_prompts = sorted(set(probe_map.keys()) & set(oracle_map.keys()))
    if len(common_prompts) == 0:
        raise ValueError("No common context prompts found between probe rows and Oracle accuracy rows")

    aligned_rows = []
    summary_rows = []
    oracle_values = np.array([oracle_map[prompt_text]["oracle_accuracy"] for prompt_text in common_prompts], dtype=float)

    for prompt_text in common_prompts:
        probe_row = probe_map[prompt_text]
        oracle_row = oracle_map[prompt_text]
        aligned_row = {
            "prompt_id": probe_row["prompt_id"],
            "prompt_text": prompt_text,
            "primary_category": probe_row["primary_category"],
            "secondary_tags": probe_row["secondary_tags"],
            "oracle_accuracy": oracle_row["oracle_accuracy"],
        }
        for metric in probe_metrics:
            aligned_row[metric] = probe_row[metric]
        aligned_rows.append(aligned_row)

    for metric in probe_metrics:
        probe_values = np.array([probe_map[prompt_text][metric] for prompt_text in common_prompts], dtype=float)
        summary_rows.append(
            {
                "probe_metric": metric,
                "num_common_prompts": len(common_prompts),
                "pearson": compute_pearson(probe_values, oracle_values),
                "spearman": compute_spearman(probe_values, oracle_values),
                "mean_probe_score": float(probe_values.mean()),
                "mean_oracle_accuracy": float(oracle_values.mean()),
            }
        )

    return aligned_rows, summary_rows


def save_aligned_csv(
    output_path: Path,
    aligned_rows: list[dict],
    probe_metrics: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt_id", "oracle_accuracy"] + probe_metrics + [
        "primary_category",
        "secondary_tags",
        "prompt_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(aligned_rows, key=lambda item: item["oracle_accuracy"], reverse=True):
            writer.writerow({field: row[field] for field in fieldnames})


def save_summary_csv(output_path: Path, summary_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "layer_percent",
        "probe_metric",
        "num_common_prompts",
        "pearson",
        "spearman",
        "mean_probe_score",
        "mean_oracle_accuracy",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def draw_source_scatter_grid(
    aligned_by_layer: dict[int, list[dict]],
    summary_by_layer_metric: dict[tuple[int, str], dict],
    probe_metrics: list[str],
    output_path: Path,
    source_name: str,
    title: str,
) -> None:
    layer_percents = sorted(aligned_by_layer.keys())
    fig, axes = plt.subplots(
        len(layer_percents),
        len(probe_metrics),
        figsize=(5.6 * len(probe_metrics), 4.8 * len(layer_percents)),
        squeeze=False,
    )

    for row_idx, layer_percent in enumerate(layer_percents):
        aligned_rows = aligned_by_layer[layer_percent]
        oracle_values = np.array([row["oracle_accuracy"] for row in aligned_rows], dtype=float)

        for col_idx, probe_metric in enumerate(probe_metrics):
            ax = axes[row_idx][col_idx]
            probe_values = np.array([row[probe_metric] for row in aligned_rows], dtype=float)
            ax.scatter(probe_values, oracle_values, alpha=0.7, color="#1f77b4", s=30)

            if len(probe_values) >= 2:
                coeffs = np.polyfit(probe_values, oracle_values, deg=1)
                xs = np.linspace(probe_values.min(), probe_values.max(), 200)
                ys = coeffs[0] * xs + coeffs[1]
                ax.plot(xs, ys, color="#d62728", linewidth=1.8)

            summary = summary_by_layer_metric[(layer_percent, probe_metric)]
            ax.set_title(
                f"L{layer_percent}% | {probe_metric}\n"
                f"r={summary['pearson']:.3f}, rho={summary['spearman']:.3f}"
            )
            ax.set_xlabel(probe_metric)
            ax.set_ylabel("Oracle secret-word accuracy")
            ax.grid(alpha=0.3)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_correlation_heatmap(
    summary_rows: list[dict],
    probe_metrics: list[str],
    correlation_key: str,
    output_path: Path,
    title: str,
) -> None:
    source_names = sorted({row["source"] for row in summary_rows})
    layer_percents = sorted({row["layer_percent"] for row in summary_rows})
    row_keys = [(source_name, layer_percent) for source_name in source_names for layer_percent in layer_percents]
    lookup = {
        (row["source"], row["layer_percent"], row["probe_metric"]): row
        for row in summary_rows
    }

    matrix = np.full((len(row_keys), len(probe_metrics)), np.nan, dtype=float)
    for row_idx, (source_name, layer_percent) in enumerate(row_keys):
        for col_idx, probe_metric in enumerate(probe_metrics):
            summary = lookup.get((source_name, layer_percent, probe_metric))
            if summary is not None:
                matrix[row_idx, col_idx] = summary[correlation_key]

    fig, ax = plt.subplots(figsize=(max(10, 2.1 * len(probe_metrics)), max(5.5, 1.3 * len(row_keys))))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Probe metric")
    ax.set_ylabel("Source / Layer")
    ax.set_xticks(range(len(probe_metrics)))
    ax.set_xticklabels(probe_metrics, rotation=25, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{source}\nL{layer_percent}%" for source, layer_percent in row_keys])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isnan(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, shrink=0.85, label=correlation_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_json", type=str, required=True)
    parser.add_argument("--hider_secret_json_dir", type=str, required=True)
    parser.add_argument("--guesser_secret_json_dir", type=str, required=True)
    parser.add_argument("--source_names", type=str, nargs="+", default=["hider", "guesser"])
    parser.add_argument("--probe_metrics", type=str, nargs="+", default=DEFAULT_PROBE_METRICS)
    parser.add_argument("--required_verbalizer_prompt", type=str, default=None)
    parser.add_argument("--act_key", type=str, default="lora")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    probe_json = Path(args.probe_json)
    probe_data = load_probe_results(probe_json)
    available_probe_metrics = get_available_probe_metrics(probe_data)
    for probe_metric in args.probe_metrics:
        if probe_metric not in available_probe_metrics:
            raise ValueError(f"Unsupported probe metric '{probe_metric}'. Available metrics: {available_probe_metrics}")

    model_name_str = sanitize_name(probe_data["config"]["model_name"].split("/")[-1])
    lang_suffix = f"_{probe_data['config']['lang_type']}" if probe_data["config"]["lang_type"] else ""
    oracle_mode = "sequence" if args.sequence else "token"
    default_output_dir = (
        Path("experiments/plotting/images")
        / "taboo_prompt_probe_oracle_correlation"
        / f"{model_name_str}_{probe_data['config']['prompt_type']}{lang_suffix}_{probe_data['config']['dataset_type']}"
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    secret_json_dirs = {
        "hider": Path(args.hider_secret_json_dir),
        "guesser": Path(args.guesser_secret_json_dir),
    }
    layer_percents = [int(layer_percent) for layer_percent in probe_data["config"]["layer_percents"]]

    all_summary_rows = []
    for source_name in args.source_names:
        if source_name not in secret_json_dirs:
            raise ValueError(f"No Oracle json_dir configured for source_name: {source_name}")

        aligned_by_layer = {}
        summary_by_layer_metric = {}
        for layer_percent in layer_percents:
            probe_rows = load_probe_layer_rows(
                data=probe_data,
                source_name=source_name,
                layer_percent=layer_percent,
                probe_metrics=args.probe_metrics,
            )
            _, oracle_rows = load_oracle_layer_rows(
                json_dir=secret_json_dirs[source_name],
                layer_percent=layer_percent,
                required_verbalizer_prompt=args.required_verbalizer_prompt,
                required_act_key=args.act_key,
                sequence=args.sequence,
            )
            aligned_rows, summary_rows = build_aligned_rows(
                probe_rows=probe_rows,
                oracle_rows=oracle_rows,
                probe_metrics=args.probe_metrics,
            )
            aligned_by_layer[layer_percent] = aligned_rows

            aligned_csv_path = output_dir / (
                f"prompt_probe_oracle_alignment_{source_name}_layer_{layer_percent}_{oracle_mode}_{sanitize_name(args.act_key)}.csv"
            )
            save_aligned_csv(
                output_path=aligned_csv_path,
                aligned_rows=aligned_rows,
                probe_metrics=args.probe_metrics,
            )

            for summary_row in summary_rows:
                summary_row["source"] = source_name
                summary_row["layer_percent"] = layer_percent
                summary_by_layer_metric[(layer_percent, summary_row["probe_metric"])] = summary_row
                all_summary_rows.append(summary_row)

        scatter_path = output_dir / (
            f"prompt_probe_oracle_scatter_{source_name}_{oracle_mode}_{sanitize_name(args.act_key)}.png"
        )
        title = (
            f"{probe_data['config']['model_name']} | {source_name} | "
            f"Oracle {oracle_mode} accuracy vs probe score"
        )
        draw_source_scatter_grid(
            aligned_by_layer=aligned_by_layer,
            summary_by_layer_metric=summary_by_layer_metric,
            probe_metrics=args.probe_metrics,
            output_path=scatter_path,
            source_name=source_name,
            title=title,
        )

    summary_csv_path = output_dir / f"prompt_probe_oracle_correlation_summary_{oracle_mode}_{sanitize_name(args.act_key)}.csv"
    save_summary_csv(summary_csv_path, all_summary_rows)

    pearson_heatmap_path = output_dir / f"prompt_probe_oracle_correlation_pearson_{oracle_mode}_{sanitize_name(args.act_key)}.png"
    draw_correlation_heatmap(
        summary_rows=all_summary_rows,
        probe_metrics=args.probe_metrics,
        correlation_key="pearson",
        output_path=pearson_heatmap_path,
        title=f"Probe vs Oracle accuracy Pearson | {probe_data['config']['model_name']} | {oracle_mode}",
    )

    spearman_heatmap_path = output_dir / f"prompt_probe_oracle_correlation_spearman_{oracle_mode}_{sanitize_name(args.act_key)}.png"
    draw_correlation_heatmap(
        summary_rows=all_summary_rows,
        probe_metrics=args.probe_metrics,
        correlation_key="spearman",
        output_path=spearman_heatmap_path,
        title=f"Probe vs Oracle accuracy Spearman | {probe_data['config']['model_name']} | {oracle_mode}",
    )

    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved Pearson heatmap: {pearson_heatmap_path}")
    print(f"Saved Spearman heatmap: {spearman_heatmap_path}")


if __name__ == "__main__":
    main()
