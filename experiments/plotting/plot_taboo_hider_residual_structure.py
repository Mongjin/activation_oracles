import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PCA_VARIANT_LABELS = {
    "raw": "Raw",
    "unit": "Unit-Normalized",
}


MODE_LABELS = {
    "hider_minus_guesser": "Hider - Guesser",
    "hider_minus_base": "Hider - Base",
}


MODE_COLORS = {
    "hider_minus_guesser": "#1f77b4",
    "hider_minus_base": "#d62728",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_summary(input_json: str) -> tuple[dict, list[int], dict[str, dict]]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    analysis_modes = data["analysis_modes"]
    first_mode = next(iter(analysis_modes))
    layer_percents = sorted(int(layer_percent) for layer_percent in analysis_modes[first_mode].keys())
    return config, layer_percents, analysis_modes


def draw_same_minus_different_plot(
    analysis_modes: dict[str, dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for ax, pca_variant in zip(axes, ["raw", "unit"]):
        for mode, mode_layers in analysis_modes.items():
            values = [
                float(mode_layers[str(layer_percent)][pca_variant]["same_minus_different_mean"])
                for layer_percent in layer_percents
            ]
            ax.plot(
                layer_percents,
                values,
                marker="o",
                linewidth=2,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{PCA_VARIANT_LABELS[pca_variant]} Residual Similarity Gap")
        ax.set_xlabel("Layer Percent")
        ax.set_ylabel("Same-Word Mean - Different-Word Mean")
        ax.set_xticks(layer_percents)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_same_vs_different_mean_plot(
    analysis_modes: dict[str, dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for ax, pca_variant in zip(axes, ["raw", "unit"]):
        for mode, mode_layers in analysis_modes.items():
            same_values = [
                float(mode_layers[str(layer_percent)][pca_variant]["same_word_similarity"]["global_stats"]["mean"])
                for layer_percent in layer_percents
            ]
            different_values = [
                float(
                    mode_layers[str(layer_percent)][pca_variant]["different_word_same_prompt_similarity"]["global_stats"][
                        "mean"
                    ]
                )
                for layer_percent in layer_percents
            ]

            ax.plot(
                layer_percents,
                same_values,
                marker="o",
                linewidth=2,
                linestyle="-",
                color=MODE_COLORS.get(mode),
                label=f"{MODE_LABELS.get(mode, mode)} | Same Word",
            )
            ax.plot(
                layer_percents,
                different_values,
                marker="o",
                linewidth=2,
                linestyle="--",
                color=MODE_COLORS.get(mode),
                label=f"{MODE_LABELS.get(mode, mode)} | Different Word",
            )

        ax.set_title(f"{PCA_VARIANT_LABELS[pca_variant]} Residual Similarity Means")
        ax.set_xlabel("Layer Percent")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_xticks(layer_percents)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_local_pc1_alignment_plot(
    analysis_modes: dict[str, dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, len(layer_percents), figsize=(5 * len(layer_percents), 10), sharey=True)
    if len(layer_percents) == 1:
        axes = [[axes[0]], [axes[1]]]

    for row_idx, pca_variant in enumerate(["raw", "unit"]):
        for col_idx, layer_percent in enumerate(layer_percents):
            ax = axes[row_idx][col_idx]
            positions = []
            labels = []
            box_data = []
            pos = 1

            for mode in analysis_modes:
                layer_info = analysis_modes[mode][str(layer_percent)][pca_variant]
                original_values = list(layer_info["original_local_pc1_vs_global_pc1_cosine"].values())
                residual_values = list(layer_info["residual_local_pc1_vs_global_pc1_cosine"].values())

                box_data.extend([original_values, residual_values])
                positions.extend([pos, pos + 1])
                labels.extend(
                    [
                        f"{MODE_LABELS.get(mode, mode)}\nOriginal",
                        f"{MODE_LABELS.get(mode, mode)}\nResidual",
                    ]
                )
                pos += 3

            ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"Layer {layer_percent}% | {PCA_VARIANT_LABELS[pca_variant]}")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(alpha=0.3, axis="y")
            if col_idx == 0:
                ax.set_ylabel("Cosine With Global PC1")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_residual_word_mean_heatmaps(
    analysis_modes: dict[str, dict],
    layer_percents: list[int],
    target_words: list[str],
    output_dir: Path,
) -> list[Path]:
    output_paths = []
    for mode, mode_layers in analysis_modes.items():
        for pca_variant in ["raw", "unit"]:
            n_layers = len(layer_percents)
            fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5), squeeze=False)
            vmin = 1.0
            vmax = -1.0
            matrices = []

            for layer_percent in layer_percents:
                matrix = mode_layers[str(layer_percent)][pca_variant]["residual_word_mean_cosine_matrix"]
                matrices.append(matrix)
                for row in matrix:
                    vmin = min(vmin, min(row))
                    vmax = max(vmax, max(row))

            for ax, layer_percent, matrix in zip(axes[0], layer_percents, matrices):
                im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="coolwarm")
                ax.set_title(f"Layer {layer_percent}%")
                ax.set_xticks(range(len(target_words)))
                ax.set_yticks(range(len(target_words)))
                ax.set_xticklabels(target_words, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(target_words, fontsize=8)

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
            fig.suptitle(f"{MODE_LABELS.get(mode, mode)} | {PCA_VARIANT_LABELS[pca_variant]} | Residual Word-Mean Cosine")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            output_path = output_dir / f"{mode}_{pca_variant}_residual_word_mean_heatmaps.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


def draw_global_pc1_variance_plot(
    analysis_modes: dict[str, dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for ax, pca_variant in zip(axes, ["raw", "unit"]):
        for mode, mode_layers in analysis_modes.items():
            values = [
                float(mode_layers[str(layer_percent)][pca_variant]["explained_variance_ratio_pc1"])
                for layer_percent in layer_percents
            ]
            ax.plot(
                layer_percents,
                values,
                marker="o",
                linewidth=2,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )

        ax.set_title(f"{PCA_VARIANT_LABELS[pca_variant]} Global PC1 Variance")
        ax.set_xlabel("Layer Percent")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_xticks(layer_percents)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./images/taboo_hider_residual_structure")
    args = parser.parse_args()

    config, layer_percents, analysis_modes = load_summary(args.input_json)
    model_name = config["model_name"]
    prompt_type = config["prompt_type"]
    dataset_type = config["dataset_type"]
    lang_type = config["lang_type"]
    lang_suffix = f"_{lang_type}" if lang_type else ""
    target_words = config["target_lora_suffixes"]

    output_dir = Path(args.output_dir) / f"{sanitize_name(model_name)}_{prompt_type}{lang_suffix}_{dataset_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variance_path = output_dir / "global_pc1_variance_by_layer.png"
    draw_global_pc1_variance_plot(
        analysis_modes=analysis_modes,
        layer_percents=layer_percents,
        output_path=variance_path,
        title=f"Residual Analysis | Global PC1 Variance | {model_name}",
    )
    print(f"Saved: {variance_path}")

    similarity_mean_path = output_dir / "same_vs_different_similarity_means.png"
    draw_same_vs_different_mean_plot(
        analysis_modes=analysis_modes,
        layer_percents=layer_percents,
        output_path=similarity_mean_path,
        title=f"Residual Analysis | Same vs Different Similarity | {model_name}",
    )
    print(f"Saved: {similarity_mean_path}")

    similarity_gap_path = output_dir / "same_minus_different_gap.png"
    draw_same_minus_different_plot(
        analysis_modes=analysis_modes,
        layer_percents=layer_percents,
        output_path=similarity_gap_path,
        title=f"Residual Analysis | Similarity Gap | {model_name}",
    )
    print(f"Saved: {similarity_gap_path}")

    local_pc1_path = output_dir / "local_pc1_vs_global_pc1_alignment.png"
    draw_local_pc1_alignment_plot(
        analysis_modes=analysis_modes,
        layer_percents=layer_percents,
        output_path=local_pc1_path,
        title=f"Residual Analysis | Local PC1 vs Global PC1 | {model_name}",
    )
    print(f"Saved: {local_pc1_path}")

    heatmap_paths = draw_residual_word_mean_heatmaps(
        analysis_modes=analysis_modes,
        layer_percents=layer_percents,
        target_words=target_words,
        output_dir=output_dir,
    )
    for heatmap_path in heatmap_paths:
        print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
