import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def parse_transfer_metrics(data: dict) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    layers = data["layers"]

    layer_items = []
    for _, layer_result in layers.items():
        percent = int(layer_result["percent"])

        base_to_ft_linear = layer_result["base_probes"]["ft_eval"][data["target_word"]]["linear"]
        base_to_ft_mlp = layer_result["base_probes"]["ft_eval"][data["target_word"]]["mlp"]
        base_to_ft_linear_acc = float(layer_result["base_probes"]["ft_eval"]["mc_linear_acc_on_target"])
        base_to_ft_mlp_acc = float(layer_result["base_probes"]["ft_eval"]["mc_mlp_acc_on_target"])

        ft_to_base_linear = layer_result["ft_probes"]["base_eval"][data["target_word"]]["linear"]
        ft_to_base_mlp = layer_result["ft_probes"]["base_eval"][data["target_word"]]["mlp"]
        ft_to_base_linear_acc = float(layer_result["ft_probes"]["base_eval"]["mc_linear_acc_on_target"])
        ft_to_base_mlp_acc = float(layer_result["ft_probes"]["base_eval"]["mc_mlp_acc_on_target"])

        layer_items.append(
            (
                percent,
                {
                    "base_to_ft": {
                        "linear": {
                            "precision": float(base_to_ft_linear["p"]),
                            "recall": float(base_to_ft_linear["r"]),
                            "f1": float(base_to_ft_linear["f1"]),
                            "acc": base_to_ft_linear_acc,
                        },
                        "mlp": {
                            "precision": float(base_to_ft_mlp["p"]),
                            "recall": float(base_to_ft_mlp["r"]),
                            "f1": float(base_to_ft_mlp["f1"]),
                            "acc": base_to_ft_mlp_acc,
                        },
                    },
                    "ft_to_base": {
                        "linear": {
                            "precision": float(ft_to_base_linear["p"]),
                            "recall": float(ft_to_base_linear["r"]),
                            "f1": float(ft_to_base_linear["f1"]),
                            "acc": ft_to_base_linear_acc,
                        },
                        "mlp": {
                            "precision": float(ft_to_base_mlp["p"]),
                            "recall": float(ft_to_base_mlp["r"]),
                            "f1": float(ft_to_base_mlp["f1"]),
                            "acc": ft_to_base_mlp_acc,
                        },
                    },
                },
            )
        )

    layer_items.sort(key=lambda x: x[0])
    layer_percents = [x[0] for x in layer_items]

    metrics = {
        "base_to_ft": {
            "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
            "mlp": {"precision": [], "recall": [], "f1": [], "acc": []},
        },
        "ft_to_base": {
            "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
            "mlp": {"precision": [], "recall": [], "f1": [], "acc": []},
        },
    }

    for _, item in layer_items:
        for direction in ["base_to_ft", "ft_to_base"]:
            for probe in ["linear", "mlp"]:
                metrics[direction][probe]["precision"].append(item[direction][probe]["precision"])
                metrics[direction][probe]["recall"].append(item[direction][probe]["recall"])
                metrics[direction][probe]["f1"].append(item[direction][probe]["f1"])
                metrics[direction][probe]["acc"].append(item[direction][probe]["acc"])

    return layer_percents, metrics


def draw_transfer_plot(
    layer_percents: list[int],
    metrics: dict[str, dict[str, list[float]]],
    title: str,
    output_path: str,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14)

    row_titles = ["Base-trained probe -> FT activations", "FT-trained probe -> Base activations"]
    directions = ["base_to_ft", "ft_to_base"]
    probes = ["linear", "mlp"]
    metric_order = ["precision", "recall", "f1", "acc"]
    metric_colors = {
        "precision": "#1f77b4",
        "recall": "#ff7f0e",
        "f1": "#2ca02c",
        "acc": "#d62728",
    }

    for r, direction in enumerate(directions):
        for c, probe in enumerate(probes):
            ax = axes[r, c]
            for metric_name in metric_order:
                ax.plot(
                    layer_percents,
                    metrics[direction][probe][metric_name],
                    marker="o",
                    linewidth=2,
                    label=metric_name,
                    color=metric_colors[metric_name],
                )

            ax.set_title(f"{row_titles[r]} | Probe: {probe.upper()}")
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(layer_percents)
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel("Layer Percent")
            if c == 0:
                ax.set_ylabel("Score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_one_result(json_path: str, output_dir: str) -> tuple[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data["model_name"]
    target_word = data["target_word"]
    layer_percents, metrics = parse_transfer_metrics(data)

    out_name = f"taboo_probe_transfer_{sanitize_name(model_name)}_{target_word}.png"
    output_path = os.path.join(output_dir, out_name)
    title = f"Taboo Probe Transfer | Model: {model_name} | Target: {target_word}"
    draw_transfer_plot(layer_percents, metrics, title, output_path)
    return model_name, output_path


def average_metrics(metrics_list: list[dict[str, dict[str, list[float]]]]) -> dict[str, dict[str, list[float]]]:
    first = metrics_list[0]
    averaged = {
        "base_to_ft": {
            "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
            "mlp": {"precision": [], "recall": [], "f1": [], "acc": []},
        },
        "ft_to_base": {
            "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
            "mlp": {"precision": [], "recall": [], "f1": [], "acc": []},
        },
    }

    for direction in first:
        for probe in first[direction]:
            for metric_name in first[direction][probe]:
                n_points = len(first[direction][probe][metric_name])
                for i in range(n_points):
                    values = [m[direction][probe][metric_name][i] for m in metrics_list]
                    averaged[direction][probe][metric_name].append(sum(values) / len(values))

    return averaged


def plot_aggregate_by_model(json_paths: list[str], output_dir: str) -> list[str]:
    grouped: dict[str, list[tuple[str, list[int], dict[str, dict[str, list[float]]]]]] = defaultdict(list)

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = data["model_name"]
        target_word = data["target_word"]
        layer_percents, metrics = parse_transfer_metrics(data)
        grouped[model_name].append((target_word, layer_percents, metrics))

    output_paths: list[str] = []
    for model_name, entries in grouped.items():
        ref_layers = entries[0][1]
        for target_word, layers, _ in entries[1:]:
            if layers != ref_layers:
                raise ValueError(
                    f"Layer percents mismatch for model={model_name}, target_word={target_word}: "
                    f"{layers} != {ref_layers}"
                )

        avg_metrics = average_metrics([entry[2] for entry in entries])
        title = f"Taboo Probe Transfer (Average Across Targets) | Model: {model_name} | N={len(entries)}"
        out_name = f"taboo_probe_transfer_{sanitize_name(model_name)}_avg_all_targets.png"
        output_path = os.path.join(output_dir, out_name)
        draw_transfer_plot(ref_layers, avg_metrics, title, output_path)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        type=str,
        default="../taboo_eval_results/taboo_probe_bidirectional_*.json",
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
