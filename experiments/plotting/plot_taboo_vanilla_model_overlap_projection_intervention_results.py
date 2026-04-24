import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_results(input_json: Path) -> dict:
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def build_prompt_level_rows(results: dict, probe_metric: str) -> list[dict]:
    rows = []

    for source_name, layers_info in results["combined_prompt_rows"].items():
        for layer_percent, modes_info in layers_info.items():
            baseline_map = {row["prompt_text"]: row for row in modes_info["baseline"]}
            for mode_name, prompt_rows in modes_info.items():
                for row in prompt_rows:
                    baseline_row = baseline_map[row["prompt_text"]]
                    rows.append(
                        {
                            "source": source_name,
                            "layer_percent": int(layer_percent),
                            "mode": mode_name,
                            "prompt_id": row["prompt_id"],
                            "prompt_text": row["prompt_text"],
                            "baseline_oracle_accuracy": float(baseline_row["oracle_accuracy"]),
                            "oracle_accuracy": float(row["oracle_accuracy"]),
                            "oracle_accuracy_delta_from_baseline": float(row["oracle_accuracy_delta_from_baseline"]),
                            "probe_metric": probe_metric,
                            "baseline_probe_value": float(baseline_row[probe_metric]),
                            "probe_value": float(row[probe_metric]),
                            "probe_delta_from_baseline": float(row[f"{probe_metric}_delta_from_baseline"]),
                        }
                    )
    return rows


def build_gap_rows(results: dict) -> list[dict]:
    rows = []

    for layer_percent, modes_info in results["cross_source_gap_rows"].items():
        baseline_map = {row["prompt_text"]: row for row in modes_info["baseline"]}
        for mode_name, gap_rows in modes_info.items():
            for row in gap_rows:
                baseline_row = baseline_map[row["prompt_text"]]
                rows.append(
                    {
                        "layer_percent": int(layer_percent),
                        "mode": mode_name,
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


def summarize_prompt_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["source"], row["layer_percent"], row["mode"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (source, layer_percent, mode), bucket in sorted(grouped.items()):
        oracle_deltas = np.array([row["oracle_accuracy_delta_from_baseline"] for row in bucket], dtype=float)
        probe_deltas = np.array([row["probe_delta_from_baseline"] for row in bucket], dtype=float)
        summary_rows.append(
            {
                "source": source,
                "layer_percent": layer_percent,
                "mode": mode,
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
        key = (row["layer_percent"], row["mode"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (layer_percent, mode), bucket in sorted(grouped.items()):
        signed_delta = np.array([row["oracle_gap_signed_delta_from_baseline"] for row in bucket], dtype=float)
        abs_delta = np.array([row["oracle_gap_abs_delta_from_baseline"] for row in bucket], dtype=float)
        summary_rows.append(
            {
                "layer_percent": layer_percent,
                "mode": mode,
                "num_prompts": len(bucket),
                "mean_signed_gap_delta": float(signed_delta.mean()),
                "std_signed_gap_delta": float(signed_delta.std(ddof=0)),
                "mean_abs_gap_delta": float(abs_delta.mean()),
                "std_abs_gap_delta": float(abs_delta.std(ddof=0)),
            }
        )
    return summary_rows


def draw_delta_distribution(ax, values: np.ndarray, ylabel: str, title: str, color: str) -> None:
    mean_value = float(values.mean())
    ax.bar([0], [mean_value], color=color, alpha=0.75, width=0.75)
    jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(
        jitter,
        values,
        color="black",
        alpha=0.65,
        s=20,
        linewidths=0,
    )
    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["Subtract"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def draw_baseline_vs_current(ax, baseline_values: np.ndarray, current_values: np.ndarray, title: str) -> None:
    all_values = np.concatenate([baseline_values, current_values])
    line_min = float(all_values.min())
    line_max = float(all_values.max())
    ax.scatter(
        baseline_values,
        current_values,
        color="#1f77b4",
        alpha=0.72,
        s=30,
    )
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Current")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def draw_source_delta_figure(
    prompt_rows: list[dict],
    probe_metric: str,
    output_path: Path,
    source_name: str,
) -> None:
    layer_percents = sorted({row["layer_percent"] for row in prompt_rows if row["source"] == source_name})
    fig, axes = plt.subplots(len(layer_percents), 4, figsize=(20, 4.8 * len(layer_percents)), squeeze=False)

    for row_idx, layer_percent in enumerate(layer_percents):
        layer_rows = [
            row for row in prompt_rows
            if row["source"] == source_name and row["layer_percent"] == layer_percent and row["mode"] == "subtract"
        ]
        oracle_deltas = np.array([row["oracle_accuracy_delta_from_baseline"] for row in layer_rows], dtype=float)
        probe_deltas = np.array([row["probe_delta_from_baseline"] for row in layer_rows], dtype=float)
        baseline_oracle = np.array([row["baseline_oracle_accuracy"] for row in layer_rows], dtype=float)
        current_oracle = np.array([row["oracle_accuracy"] for row in layer_rows], dtype=float)
        baseline_probe = np.array([row["baseline_probe_value"] for row in layer_rows], dtype=float)
        current_probe = np.array([row["probe_value"] for row in layer_rows], dtype=float)

        draw_delta_distribution(
            ax=axes[row_idx][0],
            values=oracle_deltas,
            ylabel="Oracle accuracy delta",
            title=f"{source_name} | Layer {layer_percent}% | Oracle delta",
            color="#ffb000",
        )
        draw_delta_distribution(
            ax=axes[row_idx][1],
            values=probe_deltas,
            ylabel=f"{probe_metric} delta",
            title=f"{source_name} | Layer {layer_percent}% | Probe delta",
            color="#4e79a7",
        )
        draw_baseline_vs_current(
            ax=axes[row_idx][2],
            baseline_values=baseline_oracle,
            current_values=current_oracle,
            title=f"{source_name} | Layer {layer_percent}% | Oracle baseline vs subtract",
        )
        draw_baseline_vs_current(
            ax=axes[row_idx][3],
            baseline_values=baseline_probe,
            current_values=current_probe,
            title=f"{source_name} | Layer {layer_percent}% | Probe baseline vs subtract",
        )

    fig.suptitle(
        f"Vanilla overlap projection intervention | source={source_name} | probe={probe_metric}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_gap_figure(gap_rows: list[dict], output_path: Path, layer_percent: int) -> None:
    layer_rows = [
        row for row in gap_rows
        if row["layer_percent"] == layer_percent and row["mode"] == "subtract"
    ]
    signed_deltas = np.array([row["oracle_gap_signed_delta_from_baseline"] for row in layer_rows], dtype=float)
    abs_deltas = np.array([row["oracle_gap_abs_delta_from_baseline"] for row in layer_rows], dtype=float)
    baseline_signed = np.array([row["baseline_oracle_gap_signed"] for row in layer_rows], dtype=float)
    current_signed = np.array([row["oracle_gap_signed"] for row in layer_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    draw_delta_distribution(
        ax=axes[0],
        values=signed_deltas,
        ylabel="Oracle gap signed delta",
        title=f"Layer {layer_percent}% | Signed gap delta",
        color="#e15759",
    )
    draw_delta_distribution(
        ax=axes[1],
        values=abs_deltas,
        ylabel="Oracle gap abs delta",
        title=f"Layer {layer_percent}% | Abs gap delta",
        color="#76b7b2",
    )
    draw_baseline_vs_current(
        ax=axes[2],
        baseline_values=baseline_signed,
        current_values=current_signed,
        title=f"Layer {layer_percent}% | Gap baseline vs subtract",
    )

    fig.suptitle(f"Vanilla overlap projection | hider-guesser Oracle gap changes | Layer {layer_percent}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_projection_summary_figure(results: dict, output_path: Path) -> None:
    summary = results["vanilla_projection_summary"]
    source_names = ["hider", "guesser"]
    layer_percents = [int(layer_percent) for layer_percent in results["config"]["layer_percents"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    x = np.arange(len(layer_percents))
    width = 0.34
    for idx, source_name in enumerate(source_names):
        values = np.array(
            [summary[source_name][str(layer_percent)]["mean_abs_last_token_projection"] for layer_percent in layer_percents],
            dtype=float,
        )
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=source_name,
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{layer_percent}%" for layer_percent in layer_percents])
    ax.set_ylabel("Mean abs projection onto vanilla activation")
    ax.set_title("Vanilla overlap strength by source and layer")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
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

    if args.output_dir is None:
        output_dir = Path(
            "./images/taboo_vanilla_overlap_projection_intervention",
            input_json.parent.name,
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_source = next(iter(results["combined_prompt_rows"].values()))
    sample_layer = next(iter(sample_source.values()))
    sample_mode = next(iter(sample_layer.values()))
    sample_row = sample_mode[0]
    available_metrics = {
        key for key in sample_row.keys()
        if not key.endswith("_delta_from_baseline")
    }

    gap_rows = build_gap_rows(results)
    gap_summary_rows = summarize_gap_rows(gap_rows)
    save_csv(
        output_path=output_dir / "vanilla_overlap_gap_rows.csv",
        rows=gap_rows,
        fieldnames=list(gap_rows[0].keys()),
    )
    save_csv(
        output_path=output_dir / "vanilla_overlap_gap_summary.csv",
        rows=gap_summary_rows,
        fieldnames=list(gap_summary_rows[0].keys()),
    )

    for layer_percent in sorted({row["layer_percent"] for row in gap_rows}):
        draw_gap_figure(
            gap_rows=gap_rows,
            output_path=output_dir / f"vanilla_overlap_gap_changes_layer_{layer_percent}.png",
            layer_percent=layer_percent,
        )

    for probe_metric in args.probe_metrics:
        if probe_metric not in available_metrics:
            raise ValueError(f"Probe metric '{probe_metric}' not found in vanilla overlap prompt rows")

        prompt_rows = build_prompt_level_rows(results, probe_metric)
        summary_rows = summarize_prompt_rows(prompt_rows)
        metric_tag = sanitize_name(probe_metric)

        save_csv(
            output_path=output_dir / f"vanilla_overlap_prompt_rows_{metric_tag}.csv",
            rows=prompt_rows,
            fieldnames=list(prompt_rows[0].keys()),
        )
        save_csv(
            output_path=output_dir / f"vanilla_overlap_prompt_summary_{metric_tag}.csv",
            rows=summary_rows,
            fieldnames=list(summary_rows[0].keys()),
        )

        for source_name in ["hider", "guesser"]:
            draw_source_delta_figure(
                prompt_rows=prompt_rows,
                probe_metric=probe_metric,
                output_path=output_dir / f"vanilla_overlap_deltas_{source_name}_{metric_tag}.png",
                source_name=source_name,
            )

    draw_projection_summary_figure(
        results=results,
        output_path=output_dir / "vanilla_overlap_projection_summary.png",
    )

    summary_json = {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "probe_metrics": args.probe_metrics,
        "layers": results["config"]["layer_percents"],
        "projection_summary": results["vanilla_projection_summary"],
    }
    with open(output_dir / "vanilla_overlap_plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved vanilla overlap projection plots to {output_dir}")


if __name__ == "__main__":
    main()
