import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


PCA_VARIANTS = {
    "raw_centered_pca": "Raw Centered PCA",
    "unit_normalized_centered_pca": "Unit-Normalized Centered PCA",
}


def pretty_mode_name(mode: str) -> str:
    if mode == "hider_minus_guesser":
        return "Hider - Guesser"
    if mode == "hider_minus_base":
        return "Hider - Base"
    return mode


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_mode_summaries(input_dir: str) -> tuple[dict, list[int], dict[str, dict]]:
    summary_paths = sorted(glob.glob(os.path.join(input_dir, "*_pca_summary.json")))
    if len(summary_paths) == 0:
        raise ValueError(f"No *_pca_summary.json files found in {input_dir}")

    config = None
    layer_percents = None
    mode_summaries = {}

    for summary_path in summary_paths:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mode = data["analysis_mode"]
        layers = data["layers"]
        current_layer_percents = sorted(int(layer_percent) for layer_percent in layers.keys())

        if config is None:
            config = data["config"]
            layer_percents = current_layer_percents
        else:
            if current_layer_percents != layer_percents:
                raise ValueError(f"Layer mismatch in {summary_path}: {current_layer_percents} != {layer_percents}")

        mode_summaries[mode] = layers

    return config, layer_percents, mode_summaries


def extract_pc1_variance_series(mode_layers: dict, variant_key: str, layer_percents: list[int]) -> list[float]:
    return [
        float(mode_layers[str(layer_percent)][variant_key]["explained_variance_ratio"][0])
        for layer_percent in layer_percents
    ]


def extract_pc1_mean_cosine_series(mode_layers: dict, variant_key: str, layer_percents: list[int]) -> list[float]:
    return [
        float(mode_layers[str(layer_percent)][variant_key]["pc1_cosine_with_mean_difference"])
        for layer_percent in layer_percents
    ]


def draw_layer_pc1_variance_plot(
    mode_summaries: dict[str, dict],
    layer_percents: list[int],
    title: str,
    output_path: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "hider_minus_guesser": "#1f77b4",
        "hider_minus_base": "#d62728",
    }
    linestyles = {
        "raw_centered_pca": "-",
        "unit_normalized_centered_pca": "--",
    }

    for mode, layers in mode_summaries.items():
        for variant_key, variant_label in PCA_VARIANTS.items():
            series = extract_pc1_variance_series(layers, variant_key, layer_percents)
            ax.plot(
                layer_percents,
                series,
                marker="o",
                linewidth=2,
                linestyle=linestyles[variant_key],
                color=colors.get(mode, None),
                label=f"{pretty_mode_name(mode)} | {variant_label}",
            )

    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("PC1 Explained Variance Ratio")
    ax.set_xticks(layer_percents)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_raw_unit_comparison_plot(
    mode_summaries: dict[str, dict],
    layer_percents: list[int],
    title: str,
    output_path: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    colors = {
        "hider_minus_guesser": "#1f77b4",
        "hider_minus_base": "#d62728",
    }

    for mode, layers in mode_summaries.items():
        raw_variance = extract_pc1_variance_series(layers, "raw_centered_pca", layer_percents)
        unit_variance = extract_pc1_variance_series(layers, "unit_normalized_centered_pca", layer_percents)
        raw_mean_cos = [abs(x) for x in extract_pc1_mean_cosine_series(layers, "raw_centered_pca", layer_percents)]
        unit_mean_cos = [abs(x) for x in extract_pc1_mean_cosine_series(layers, "unit_normalized_centered_pca", layer_percents)]

        axes[0].plot(
            layer_percents,
            raw_variance,
            marker="o",
            linewidth=2,
            linestyle="-",
            color=colors.get(mode, None),
            label=f"{pretty_mode_name(mode)} | Raw",
        )
        axes[0].plot(
            layer_percents,
            unit_variance,
            marker="o",
            linewidth=2,
            linestyle="--",
            color=colors.get(mode, None),
            label=f"{pretty_mode_name(mode)} | Unit",
        )

        axes[1].plot(
            layer_percents,
            raw_mean_cos,
            marker="o",
            linewidth=2,
            linestyle="-",
            color=colors.get(mode, None),
            label=f"{pretty_mode_name(mode)} | Raw",
        )
        axes[1].plot(
            layer_percents,
            unit_mean_cos,
            marker="o",
            linewidth=2,
            linestyle="--",
            color=colors.get(mode, None),
            label=f"{pretty_mode_name(mode)} | Unit",
        )

    axes[0].set_title("PC1 Explained Variance Ratio")
    axes[0].set_xlabel("Layer Percent")
    axes[0].set_ylabel("Variance Ratio")
    axes[0].set_xticks(layer_percents)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("|PC1 Cosine With Mean Difference|")
    axes[1].set_xlabel("Layer Percent")
    axes[1].set_ylabel("Absolute Cosine")
    axes[1].set_xticks(layer_percents)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_target_projection_plot(
    mode: str,
    mode_layers: dict,
    layer_percents: list[int],
    variant_key: str,
    title: str,
    output_path: str,
) -> str:
    n_layers = len(layer_percents)
    fig, axes = plt.subplots(n_layers, 1, figsize=(18, 4.5 * n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer_percent in zip(axes, layer_percents):
        layer_info = mode_layers[str(layer_percent)][variant_key]
        projection_by_target = layer_info["pc1_projection_by_target"]
        target_words = list(projection_by_target.keys())
        box_data = [projection_by_target[target]["scores"] for target in target_words]
        mean_values = [projection_by_target[target]["stats"]["mean"] for target in target_words]

        ax.boxplot(box_data, patch_artist=True, widths=0.6)
        ax.plot(range(1, len(target_words) + 1), mean_values, marker="o", linewidth=2, color="#d62728", label="Mean")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Layer {layer_percent}% | {PCA_VARIANTS[variant_key]}")
        ax.set_ylabel("PC1 Projection")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right")

    axes[-1].set_xticks(range(1, len(target_words) + 1))
    axes[-1].set_xticklabels(target_words, rotation=45, ha="right")
    axes[-1].set_xlabel("Target Word")
    fig.suptitle(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_pc1_direction_comparison_plot(
    mode_summaries: dict[str, dict],
    layer_percents: list[int],
    title: str,
    output_path: str,
) -> str:
    required_modes = {"hider_minus_guesser", "hider_minus_base"}
    if set(mode_summaries.keys()) & required_modes != required_modes:
        raise ValueError("Both hider_minus_guesser and hider_minus_base summaries are required for PC1 comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    comparison_modes = [
        ("raw_centered_pca", "Raw Centered PCA"),
        ("unit_normalized_centered_pca", "Unit-Normalized Centered PCA"),
    ]

    for variant_key, variant_label in comparison_modes:
        signed_cosines = []
        abs_cosines = []
        for layer_percent in layer_percents:
            guesser_pc1 = mode_summaries["hider_minus_guesser"][str(layer_percent)][variant_key]["pc1_direction"]
            base_pc1 = mode_summaries["hider_minus_base"][str(layer_percent)][variant_key]["pc1_direction"]
            signed_cosine = sum(float(g) * float(b) for g, b in zip(guesser_pc1, base_pc1))
            signed_cosines.append(signed_cosine)
            abs_cosines.append(abs(signed_cosine))

        axes[0].plot(layer_percents, signed_cosines, marker="o", linewidth=2, label=variant_label)
        axes[1].plot(layer_percents, abs_cosines, marker="o", linewidth=2, label=variant_label)

    axes[0].set_title("Signed Cosine Between PC1 Directions")
    axes[0].set_xlabel("Layer Percent")
    axes[0].set_ylabel("Cosine")
    axes[0].set_xticks(layer_percents)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Absolute Cosine Between PC1 Directions")
    axes[1].set_xlabel("Layer Percent")
    axes[1].set_ylabel("|Cosine|")
    axes[1].set_xticks(layer_percents)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./images/taboo_hider_activation_pca")
    args = parser.parse_args()

    config, layer_percents, mode_summaries = load_mode_summaries(args.input_dir)
    model_name = config["model_name"]
    prompt_type = config["prompt_type"]
    dataset_type = config["dataset_type"]
    lang_type = config["lang_type"]
    lang_suffix = f"_{lang_type}" if lang_type else ""

    output_dir = Path(args.output_dir) / f"{sanitize_name(model_name)}_{prompt_type}{lang_suffix}_{dataset_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_plot_path = output_dir / "pc1_variance_by_layer.png"
    draw_layer_pc1_variance_plot(
        mode_summaries=mode_summaries,
        layer_percents=layer_percents,
        title=f"PC1 Explained Variance By Layer | {model_name}",
        output_path=str(variance_plot_path),
    )
    print(f"Saved: {variance_plot_path}")

    raw_unit_plot_path = output_dir / "raw_unit_comparison.png"
    draw_raw_unit_comparison_plot(
        mode_summaries=mode_summaries,
        layer_percents=layer_percents,
        title=f"Raw vs Unit PCA Comparison | {model_name}",
        output_path=str(raw_unit_plot_path),
    )
    print(f"Saved: {raw_unit_plot_path}")

    for mode, mode_layers in mode_summaries.items():
        for variant_key in PCA_VARIANTS:
            projection_plot_path = output_dir / f"{mode}_{variant_key}_pc1_projection_by_target.png"
            draw_target_projection_plot(
                mode=mode,
                mode_layers=mode_layers,
                layer_percents=layer_percents,
                variant_key=variant_key,
                title=f"{pretty_mode_name(mode)} | PC1 Projection By Target | {model_name}",
                output_path=str(projection_plot_path),
            )
            print(f"Saved: {projection_plot_path}")

    if "hider_minus_guesser" in mode_summaries and "hider_minus_base" in mode_summaries:
        pc1_compare_path = output_dir / "pc1_direction_comparison_hider_base_vs_hider_guesser.png"
        draw_pc1_direction_comparison_plot(
            mode_summaries=mode_summaries,
            layer_percents=layer_percents,
            title=f"PC1 Direction Comparison | {model_name}",
            output_path=str(pc1_compare_path),
        )
        print(f"Saved: {pc1_compare_path}")


if __name__ == "__main__":
    main()
