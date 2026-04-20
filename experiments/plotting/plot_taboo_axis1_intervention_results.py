import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_COLORS = {
    "feature_group": "#ffb000",
    "other": "#7f7f7f",
}

MODE_LABELS = {
    "add": "Add",
    "subtract": "Subtract",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_results(input_json: Path) -> dict:
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_default_output_dir(input_json: Path) -> Path:
    run_name = input_json.parent.name
    return Path(
        "C:/Users/user/Documents/projects/activation_oracles/experiments/plotting/images/taboo_axis1_intervention",
        run_name,
    )


def get_feature_prompt_sets(results: dict) -> dict[str, dict[str, set[str]]]:
    prompt_sets = {}
    for layer_percent, layer_info in results["shared_group_prompts"].items():
        top_prompts = set(layer_info["top_overlap_prompts"])
        bottom_prompts = set(layer_info["bottom_overlap_prompts"])
        prompt_sets[layer_percent] = {
            "top_overlap_prompts": top_prompts,
            "bottom_overlap_prompts": bottom_prompts,
            "feature_group_prompts": top_prompts | bottom_prompts,
        }
    return prompt_sets


def classify_prompt_membership(prompt_text: str, layer_prompt_sets: dict[str, set[str]]) -> tuple[str, str]:
    if prompt_text in layer_prompt_sets["top_overlap_prompts"]:
        return "feature_group", "top_overlap"
    if prompt_text in layer_prompt_sets["bottom_overlap_prompts"]:
        return "feature_group", "bottom_overlap"
    return "other", "other"


def build_prompt_level_rows(results: dict, probe_metric: str) -> list[dict]:
    feature_prompt_sets = get_feature_prompt_sets(results)
    rows = []

    for source_name, layers_info in results["sources"].items():
        for layer_percent, modes_info in layers_info.items():
            layer_prompt_sets = feature_prompt_sets[layer_percent]
            for mode_name, prompt_rows in modes_info.items():
                for row in prompt_rows:
                    group_membership, group_detail = classify_prompt_membership(row["prompt_text"], layer_prompt_sets)
                    rows.append(
                        {
                            "source": source_name,
                            "layer_percent": int(layer_percent),
                            "mode": mode_name,
                            "group_membership": group_membership,
                            "group_detail": group_detail,
                            "prompt_id": row["prompt_id"],
                            "prompt_text": row["prompt_text"],
                            "oracle_accuracy": float(row["oracle_accuracy"]),
                            "oracle_accuracy_delta_from_baseline": float(row["oracle_accuracy_delta_from_baseline"]),
                            "probe_metric": probe_metric,
                            "probe_value": float(row[probe_metric]),
                            "probe_delta_from_baseline": float(row[f"{probe_metric}_delta_from_baseline"]),
                        }
                    )
    return rows


def build_gap_rows(results: dict) -> list[dict]:
    feature_prompt_sets = get_feature_prompt_sets(results)
    rows = []

    for layer_percent, modes_info in results["cross_source_gap_rows"].items():
        layer_prompt_sets = feature_prompt_sets[layer_percent]
        baseline_map = {
            row["prompt_text"]: row
            for row in modes_info["baseline"]
        }
        for mode_name, gap_rows in modes_info.items():
            for row in gap_rows:
                group_membership, group_detail = classify_prompt_membership(row["prompt_text"], layer_prompt_sets)
                baseline_row = baseline_map[row["prompt_text"]]
                rows.append(
                    {
                        "layer_percent": int(layer_percent),
                        "mode": mode_name,
                        "group_membership": group_membership,
                        "group_detail": group_detail,
                        "prompt_id": row["prompt_id"],
                        "prompt_text": row["prompt_text"],
                        "baseline_oracle_gap_signed": float(baseline_row["oracle_gap_signed"]),
                        "baseline_oracle_gap_abs": float(baseline_row["oracle_gap_abs"]),
                        "oracle_gap_signed": float(row["oracle_gap_signed"]),
                        "oracle_gap_abs": float(row["oracle_gap_abs"]),
                        "oracle_gap_signed_delta_from_baseline": float(
                            row["oracle_gap_signed"] - baseline_row["oracle_gap_signed"]
                        ),
                        "oracle_gap_abs_delta_from_baseline": float(
                            row["oracle_gap_abs"] - baseline_row["oracle_gap_abs"]
                        ),
                    }
                )
    return rows


def save_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def summarize_prompt_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["source"], row["layer_percent"], row["mode"], row["group_membership"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (source, layer_percent, mode, group_membership), bucket in sorted(grouped.items()):
        oracle_deltas = np.array([row["oracle_accuracy_delta_from_baseline"] for row in bucket], dtype=float)
        probe_deltas = np.array([row["probe_delta_from_baseline"] for row in bucket], dtype=float)
        summary_rows.append(
            {
                "source": source,
                "layer_percent": layer_percent,
                "mode": mode,
                "group_membership": group_membership,
                "num_prompts": len(bucket),
                "mean_oracle_accuracy_delta": float(oracle_deltas.mean()),
                "std_oracle_accuracy_delta": float(oracle_deltas.std(ddof=0)),
                "mean_probe_delta": float(probe_deltas.mean()),
                "std_probe_delta": float(probe_deltas.std(ddof=0)),
            }
        )
    return summary_rows


def summarize_gap_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["layer_percent"], row["mode"], row["group_membership"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (layer_percent, mode, group_membership), bucket in sorted(grouped.items()):
        signed_delta = np.array([row["oracle_gap_signed_delta_from_baseline"] for row in bucket], dtype=float)
        abs_delta = np.array([row["oracle_gap_abs_delta_from_baseline"] for row in bucket], dtype=float)
        summary_rows.append(
            {
                "layer_percent": layer_percent,
                "mode": mode,
                "group_membership": group_membership,
                "num_prompts": len(bucket),
                "mean_signed_gap_delta": float(signed_delta.mean()),
                "std_signed_gap_delta": float(signed_delta.std(ddof=0)),
                "mean_abs_gap_delta": float(abs_delta.mean()),
                "std_abs_gap_delta": float(abs_delta.std(ddof=0)),
            }
        )
    return summary_rows


def draw_grouped_delta_panel(
    ax,
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    group_order = ["feature_group", "other"]
    mode_order = ["add", "subtract"]
    x_positions = {
        ("add", "feature_group"): 0,
        ("add", "other"): 1,
        ("subtract", "feature_group"): 3,
        ("subtract", "other"): 4,
    }

    for mode_name in mode_order:
        for group_name in group_order:
            bucket = [
                row for row in rows
                if row["mode"] == mode_name and row["group_membership"] == group_name
            ]
            x = x_positions[(mode_name, group_name)]
            values = np.array([row[value_key] for row in bucket], dtype=float)
            mean_value = float(values.mean()) if len(values) > 0 else 0.0
            ax.bar(
                x,
                mean_value,
                color=GROUP_COLORS[group_name],
                alpha=0.75,
                width=0.75,
            )
            if len(values) > 0:
                jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(values), x, dtype=float) + jitter,
                    values,
                    color="black",
                    alpha=0.65,
                    s=20,
                    linewidths=0,
                )

    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["Add\nFeature", "Add\nOther", "Subtract\nFeature", "Subtract\nOther"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def draw_source_delta_figure(
    prompt_rows: list[dict],
    probe_metric: str,
    output_path: Path,
    source_name: str,
) -> None:
    layer_percents = sorted({row["layer_percent"] for row in prompt_rows if row["source"] == source_name})
    fig, axes = plt.subplots(len(layer_percents), 2, figsize=(12, 4.6 * len(layer_percents)), squeeze=False)

    for row_idx, layer_percent in enumerate(layer_percents):
        layer_rows = [
            row for row in prompt_rows
            if row["source"] == source_name and row["layer_percent"] == layer_percent and row["mode"] != "baseline"
        ]
        draw_grouped_delta_panel(
            ax=axes[row_idx][0],
            rows=layer_rows,
            value_key="oracle_accuracy_delta_from_baseline",
            ylabel="Oracle accuracy delta",
            title=f"{source_name} | Layer {layer_percent}% | Oracle delta",
        )
        draw_grouped_delta_panel(
            ax=axes[row_idx][1],
            rows=layer_rows,
            value_key="probe_delta_from_baseline",
            ylabel=f"{probe_metric} delta",
            title=f"{source_name} | Layer {layer_percent}% | Probe delta",
        )

    fig.suptitle(f"Axis 1 intervention deltas by prompt group | source={source_name} | probe={probe_metric}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_gap_delta_panel(ax, rows: list[dict], title: str) -> None:
    draw_grouped_delta_panel(
        ax=ax,
        rows=rows,
        value_key="oracle_gap_signed_delta_from_baseline",
        ylabel="Oracle gap signed delta",
        title=title,
    )


def draw_gap_scatter(ax, rows: list[dict], mode_name: str, title: str) -> None:
    mode_rows = [row for row in rows if row["mode"] == mode_name]
    for group_name in ["feature_group", "other"]:
        bucket = [row for row in mode_rows if row["group_membership"] == group_name]
        xs = np.array([row["baseline_oracle_gap_signed"] for row in bucket], dtype=float)
        ys = np.array([row["oracle_gap_signed"] for row in bucket], dtype=float)
        ax.scatter(
            xs,
            ys,
            color=GROUP_COLORS[group_name],
            alpha=0.78,
            s=34,
            label=f"{group_name} (n={len(bucket)})",
        )

    all_values = np.array(
        [row["baseline_oracle_gap_signed"] for row in mode_rows] +
        [row["oracle_gap_signed"] for row in mode_rows],
        dtype=float,
    )
    line_min = float(all_values.min())
    line_max = float(all_values.max())
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Baseline hider-guesser Oracle gap (signed)")
    ax.set_ylabel("Current hider-guesser Oracle gap (signed)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def draw_gap_figure(gap_rows: list[dict], output_path: Path, layer_percent: int) -> None:
    layer_rows = [row for row in gap_rows if row["layer_percent"] == layer_percent and row["mode"] != "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    draw_gap_delta_panel(
        ax=axes[0],
        rows=layer_rows,
        title=f"Layer {layer_percent}% | Gap delta by group",
    )
    draw_gap_scatter(
        ax=axes[1],
        rows=layer_rows,
        mode_name="add",
        title=f"Layer {layer_percent}% | Add | baseline vs current gap",
    )
    draw_gap_scatter(
        ax=axes[2],
        rows=layer_rows,
        mode_name="subtract",
        title=f"Layer {layer_percent}% | Subtract | baseline vs current gap",
    )

    fig.suptitle(f"Axis 1 hider-guesser Oracle gap changes | Layer {layer_percent}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--probe_metrics",
        type=str,
        nargs="+",
        default=["binary_linear_target_prob_mean"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_json = Path(args.input_json)
    results = load_results(input_json)
    output_dir = infer_default_output_dir(input_json) if args.output_dir is None else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_metrics = set()
    sample_source = next(iter(results["sources"].values()))
    sample_layer = next(iter(sample_source.values()))
    sample_mode = next(iter(sample_layer.values()))
    sample_row = sample_mode[0]
    for key in sample_row.keys():
        if key.endswith("_delta_from_baseline"):
            continue
        available_metrics.add(key)

    gap_rows = build_gap_rows(results)
    gap_summary_rows = summarize_gap_rows(gap_rows)
    save_csv(
        output_path=output_dir / "axis1_gap_rows.csv",
        rows=gap_rows,
        fieldnames=list(gap_rows[0].keys()),
    )
    save_csv(
        output_path=output_dir / "axis1_gap_summary.csv",
        rows=gap_summary_rows,
        fieldnames=list(gap_summary_rows[0].keys()),
    )

    for layer_percent in sorted({row["layer_percent"] for row in gap_rows}):
        draw_gap_figure(
            gap_rows=gap_rows,
            output_path=output_dir / f"axis1_gap_changes_layer_{layer_percent}.png",
            layer_percent=layer_percent,
        )

    for probe_metric in args.probe_metrics:
        if probe_metric not in available_metrics:
            raise ValueError(f"Probe metric '{probe_metric}' not found in Axis 1 prompt rows")

        prompt_rows = build_prompt_level_rows(results, probe_metric)
        summary_rows = summarize_prompt_rows(prompt_rows)
        metric_tag = sanitize_name(probe_metric)

        save_csv(
            output_path=output_dir / f"axis1_prompt_rows_{metric_tag}.csv",
            rows=prompt_rows,
            fieldnames=list(prompt_rows[0].keys()),
        )
        save_csv(
            output_path=output_dir / f"axis1_prompt_summary_{metric_tag}.csv",
            rows=summary_rows,
            fieldnames=list(summary_rows[0].keys()),
        )

        for source_name in ["hider", "guesser"]:
            draw_source_delta_figure(
                prompt_rows=prompt_rows,
                probe_metric=probe_metric,
                output_path=output_dir / f"axis1_deltas_{source_name}_{metric_tag}.png",
                source_name=source_name,
            )

    summary_json = {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "probe_metrics": args.probe_metrics,
        "layers": results["config"]["layer_percents"],
        "feature_group_prompts": results["shared_group_prompts"],
    }
    with open(output_dir / "axis1_plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved Axis 1 plots to {output_dir}")


if __name__ == "__main__":
    main()
