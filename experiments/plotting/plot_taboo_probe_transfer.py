import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def build_source_mapping(layer_result: dict) -> dict[str, dict]:
    if "hider_probes" in layer_result and "guesser_probes" in layer_result:
        return {
            "base": layer_result["base_probes"],
            "hider": layer_result["hider_probes"],
            "guesser": layer_result["guesser_probes"],
        }
    return {
        "base": layer_result["base_probes"],
        "hider": layer_result["ft_probes"],
    }


def parse_transfer_metrics(data: dict) -> tuple[list[int], list[str], dict[str, dict[str, dict[str, dict[str, list[float]]]]]]:
    layers = data["layers"]

    layer_items = []
    for _, layer_result in layers.items():
        percent = int(layer_result["percent"])
        source_mapping = build_source_mapping(layer_result)
        source_names = list(source_mapping.keys())
        item_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

        for train_source in source_names:
            item_metrics[train_source] = {}
            for eval_source in source_names:
                if eval_source == train_source:
                    continue
                eval_key = f"{eval_source}_eval"
                eval_result = source_mapping[train_source][eval_key]
                target_result = eval_result[data["target_word"]]
                item_metrics[train_source][eval_source] = {
                    "linear": {
                        "precision": float(target_result["linear"]["p"]),
                        "recall": float(target_result["linear"]["r"]),
                        "f1": float(target_result["linear"]["f1"]),
                        "acc": float(eval_result["mc_linear_acc_on_target"]),
                    },
                    "mlp": {
                        "precision": float(target_result["mlp"]["p"]),
                        "recall": float(target_result["mlp"]["r"]),
                        "f1": float(target_result["mlp"]["f1"]),
                        "acc": float(eval_result["mc_mlp_acc_on_target"]),
                    },
                }

        layer_items.append((percent, item_metrics))

    layer_items.sort(key=lambda x: x[0])
    layer_percents = [x[0] for x in layer_items]
    source_names = list(layer_items[0][1].keys())

    metrics = {}
    for train_source in source_names:
        metrics[train_source] = {}
        for eval_source in source_names:
            if eval_source == train_source:
                continue
            metrics[train_source][eval_source] = {
                "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
                "mlp": {"precision": [], "recall": [], "f1": [], "acc": []},
            }

    for _, item in layer_items:
        for train_source in source_names:
            for eval_source in source_names:
                if eval_source == train_source:
                    continue
                for probe in ["linear", "mlp"]:
                    metrics[train_source][eval_source][probe]["precision"].append(
                        item[train_source][eval_source][probe]["precision"]
                    )
                    metrics[train_source][eval_source][probe]["recall"].append(
                        item[train_source][eval_source][probe]["recall"]
                    )
                    metrics[train_source][eval_source][probe]["f1"].append(
                        item[train_source][eval_source][probe]["f1"]
                    )
                    metrics[train_source][eval_source][probe]["acc"].append(
                        item[train_source][eval_source][probe]["acc"]
                    )

    return layer_percents, source_names, metrics


def draw_transfer_plot(
    layer_percents: list[int],
    source_names: list[str],
    metrics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    title: str,
    output_path: str,
) -> str:
    n_rows = len(source_names)
    n_cols = 2 * (len(source_names) - 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14)

    if n_rows == 1:
        axes = [axes]

    probe_order = ["linear", "mlp"]
    metric_order = ["precision", "recall", "f1", "acc"]
    metric_colors = {
        "precision": "#1f77b4",
        "recall": "#ff7f0e",
        "f1": "#2ca02c",
        "acc": "#d62728",
    }

    for r, train_source in enumerate(source_names):
        eval_sources = [name for name in source_names if name != train_source]
        row_axes = axes[r] if n_rows > 1 else axes[0]
        for eval_idx, eval_source in enumerate(eval_sources):
            for probe_idx, probe in enumerate(probe_order):
                c = eval_idx * 2 + probe_idx
                ax = row_axes[c]
                for metric_name in metric_order:
                    ax.plot(
                        layer_percents,
                        metrics[train_source][eval_source][probe][metric_name],
                        marker="o",
                        linewidth=2,
                        label=metric_name,
                        color=metric_colors[metric_name],
                    )

                ax.set_title(
                    f"{train_source.capitalize()} probe -> {eval_source.capitalize()} activations | {probe.upper()}"
                )
                ax.set_ylim(0.0, 1.05)
                ax.set_xticks(layer_percents)
                ax.grid(alpha=0.3)
                if r == n_rows - 1:
                    ax.set_xlabel("Layer Percent")
                if c == 0:
                    ax.set_ylabel("Score")

    handles, labels = axes[0][0].get_legend_handles_labels() if n_rows > 1 else axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_one_result(json_path: str, output_dir: str) -> tuple[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["model_name"]
    target_word = data["target_word"]
    layer_percents, source_names, metrics = parse_transfer_metrics(data)

    out_name = f"taboo_probe_transfer_{sanitize_name(model_name)}_{target_word}.png"
    output_path = os.path.join(output_dir, out_name)
    title = f"Taboo Probe Transfer | Model: {model_name} | Target: {target_word}"
    draw_transfer_plot(layer_percents, source_names, metrics, title, output_path)
    return model_name, output_path


def average_metrics(
    metrics_list: list[dict[str, dict[str, dict[str, dict[str, list[float]]]]]]
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    first = metrics_list[0]
    averaged = {}

    for train_source in first:
        averaged[train_source] = {}
        for eval_source in first[train_source]:
            averaged[train_source][eval_source] = {}
            for probe in first[train_source][eval_source]:
                averaged[train_source][eval_source][probe] = {"precision": [], "recall": [], "f1": [], "acc": []}
                for metric_name in first[train_source][eval_source][probe]:
                    n_points = len(first[train_source][eval_source][probe][metric_name])
                    for i in range(n_points):
                        values = [m[train_source][eval_source][probe][metric_name][i] for m in metrics_list]
                        averaged[train_source][eval_source][probe][metric_name].append(sum(values) / len(values))

    return averaged


def plot_aggregate_by_model(json_paths: list[str], output_dir: str) -> list[str]:
    grouped: dict[str, list[tuple[str, list[int], list[str], dict[str, dict[str, dict[str, dict[str, list[float]]]]]]]] = defaultdict(list)

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = data["model_name"]
        target_word = data["target_word"]
        layer_percents, source_names, metrics = parse_transfer_metrics(data)
        grouped[model_name].append((target_word, layer_percents, source_names, metrics))

    output_paths: list[str] = []
    for model_name, entries in grouped.items():
        ref_layers = entries[0][1]
        ref_sources = entries[0][2]
        for target_word, layers, source_names, _ in entries[1:]:
            if layers != ref_layers:
                raise ValueError(
                    f"Layer percents mismatch for model={model_name}, target_word={target_word}: "
                    f"{layers} != {ref_layers}"
                )
            if source_names != ref_sources:
                raise ValueError(
                    f"Source mismatch for model={model_name}, target_word={target_word}: "
                    f"{source_names} != {ref_sources}"
                )

        avg_metrics = average_metrics([entry[3] for entry in entries])
        title = f"Taboo Probe Transfer (Average Across Targets) | Model: {model_name} | N={len(entries)}"
        out_name = f"taboo_probe_transfer_{sanitize_name(model_name)}_avg_all_targets.png"
        output_path = os.path.join(output_dir, out_name)
        draw_transfer_plot(ref_layers, ref_sources, avg_metrics, title, output_path)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        type=str,
        default="../taboo_eval_results/taboo_probe_bidirectional_gemma*.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./images/taboo_probe",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_target", "aggregate", "both"],
        default="per_target",
    )
    args = parser.parse_args()

    json_paths = sorted(glob.glob(args.input_glob))
    if len(json_paths) == 0:
        raise ValueError(f"No files found for input_glob: {args.input_glob}")

    print(f"Found {len(json_paths)} files")

    if args.mode in ("per_target", "both"):
        for path in json_paths:
            _, out = plot_one_result(path, args.output_dir)
            print(f"Saved: {out}")

    if args.mode in ("aggregate", "both"):
        output_paths = plot_aggregate_by_model(json_paths, args.output_dir)
        for out in output_paths:
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
