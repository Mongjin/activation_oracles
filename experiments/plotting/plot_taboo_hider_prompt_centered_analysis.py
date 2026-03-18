import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_summary(input_json: str) -> tuple[dict, list[int], dict]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    layer_percents = sorted(int(layer_percent) for layer_percent in data["layers"].keys())
    return data["config"], layer_percents, data


def mean_offdiag(matrix: list[list[float]]) -> float:
    matrix_np = np.array(matrix, dtype=float)
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    return float(matrix_np[mask].mean())


def draw_probe_accuracy_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    linear_train = []
    linear_test = []
    mlp_train = []
    mlp_test = []

    for layer_percent in layer_percents:
        probe_info = summary["layers"][str(layer_percent)]["target_word_probe"]
        linear_train.append(float(probe_info["linear"]["train"]["accuracy"]))
        linear_test.append(float(probe_info["linear"]["test"]["accuracy"]))
        mlp_train.append(float(probe_info["mlp"]["train"]["accuracy"]))
        mlp_test.append(float(probe_info["mlp"]["test"]["accuracy"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layer_percents, linear_train, marker="o", linewidth=2, linestyle="--", color="#1f77b4", label="Linear Train")
    ax.plot(layer_percents, linear_test, marker="o", linewidth=2, linestyle="-", color="#1f77b4", label="Linear Test")
    ax.plot(layer_percents, mlp_train, marker="s", linewidth=2, linestyle="--", color="#d62728", label="MLP Train")
    ax.plot(layer_percents, mlp_test, marker="s", linewidth=2, linestyle="-", color="#d62728", label="MLP Test")

    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Target-Word Probe Accuracy")
    ax.set_xticks(layer_percents)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_same_vs_different_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    same_means = []
    different_means = []
    gaps = []

    for layer_percent in layer_percents:
        layer_info = summary["layers"][str(layer_percent)]
        same_means.append(float(layer_info["same_word_similarity"]["global_stats"]["mean"]))
        different_means.append(float(layer_info["different_word_same_prompt_similarity"]["global_stats"]["mean"]))
        gaps.append(float(layer_info["same_minus_different_mean"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].plot(layer_percents, same_means, marker="o", linewidth=2, color="#1f77b4", label="Same Word")
    axes[0].plot(layer_percents, different_means, marker="o", linewidth=2, color="#d62728", label="Different Word, Same Prompt")
    axes[0].set_title("Similarity Means")
    axes[0].set_xlabel("Layer Percent")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_xticks(layer_percents)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(layer_percents, gaps, marker="o", linewidth=2, color="#2ca02c")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Same - Different Gap")
    axes[1].set_xlabel("Layer Percent")
    axes[1].set_ylabel("Cosine Gap")
    axes[1].set_xticks(layer_percents)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_decomposition_fraction_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    prompt_fractions = []
    word_fractions = []
    interaction_fractions = []

    for layer_percent in layer_percents:
        fractions = summary["layers"][str(layer_percent)]["two_way_decomposition"]["variance_fraction_by_component"]
        prompt_fractions.append(float(fractions["prompt"]))
        word_fractions.append(float(fractions["word"]))
        interaction_fractions.append(float(fractions["interaction"]))

    x = np.arange(len(layer_percents))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, prompt_fractions, color="#1f77b4", label="Prompt")
    ax.bar(x, word_fractions, bottom=np.array(prompt_fractions), color="#d62728", label="Word")
    ax.bar(
        x,
        interaction_fractions,
        bottom=np.array(prompt_fractions) + np.array(word_fractions),
        color="#2ca02c",
        label="Interaction",
    )

    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Variance Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_percents)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_local_subspace_summary_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    mean_local_subspace = []
    mean_local_pc1_abs = []

    for layer_percent in layer_percents:
        local_pca = summary["layers"][str(layer_percent)]["local_pca"]
        mean_local_subspace.append(mean_offdiag(local_pca["local_subspace_similarity_matrix"]))
        mean_local_pc1_abs.append(mean_offdiag(local_pca["local_pc1_absolute_cosine_matrix"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layer_percents, mean_local_subspace, marker="o", linewidth=2, color="#9467bd", label="Mean Local Subspace Similarity")
    ax.plot(layer_percents, mean_local_pc1_abs, marker="o", linewidth=2, color="#ff7f0e", label="Mean |Local PC1 Cosine|")
    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Similarity")
    ax.set_xticks(layer_percents)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_local_subspace_heatmaps(summary: dict, layer_percents: list[int], target_words: list[str], title: str, output_path: str) -> str:
    n_layers = len(layer_percents)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    axes = axes[0]

    vmin = 0.0
    vmax = 1.0
    images = []
    for ax, layer_percent in zip(axes, layer_percents):
        matrix = np.array(summary["layers"][str(layer_percent)]["local_pca"]["local_subspace_similarity_matrix"], dtype=float)
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
        images.append(image)
        ax.set_title(f"Layer {layer_percent}%")
        ax.set_xticks(range(len(target_words)))
        ax.set_yticks(range(len(target_words)))
        ax.set_xticklabels(target_words, rotation=45, ha="right")
        ax.set_yticklabels(target_words)

    fig.colorbar(images[-1], ax=axes.tolist(), fraction=0.025, pad=0.04, label="Subspace Similarity")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config, layer_percents, summary = load_summary(args.input_json)
    model_name = config["model_name"].split("/")[-1]
    prompt_type = config["prompt_type"]
    dataset_type = config["dataset_type"]
    target_words = config["target_lora_suffixes"]

    if args.output_dir is None:
        output_dir = Path(
            "experiments/plotting/images/taboo_hider_prompt_centered_analysis"
        ) / f"{sanitize_name(model_name)}_{sanitize_name(prompt_type)}_{sanitize_name(dataset_type)}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{model_name} | {prompt_type} | {dataset_type}"

    draw_probe_accuracy_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Target-Word Probe Accuracy",
        output_path=str(output_dir / "probe_accuracy_by_layer.png"),
    )
    draw_same_vs_different_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Same vs Different Similarity",
        output_path=str(output_dir / "same_vs_different_gap.png"),
    )
    draw_decomposition_fraction_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Two-Way Decomposition Fractions",
        output_path=str(output_dir / "decomposition_fraction_by_layer.png"),
    )
    draw_local_subspace_summary_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Local Subspace Summary",
        output_path=str(output_dir / "local_subspace_similarity_by_layer.png"),
    )
    draw_local_subspace_heatmaps(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        title=f"{title_prefix} | Local Subspace Similarity Heatmaps",
        output_path=str(output_dir / "local_subspace_similarity_heatmaps.png"),
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
