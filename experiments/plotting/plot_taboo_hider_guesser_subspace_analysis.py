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


def draw_deception_alignment_summary_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    local_pc1_abs_mean = []
    local_pc1_abs_std = []
    subspace_overlap_mean = []
    subspace_overlap_std = []
    word_mean_abs_mean = []
    word_mean_abs_std = []

    for layer_percent in layer_percents:
        alignment = summary["layers"][str(layer_percent)]["hider_specific_residual"]["global_deception_alignment"]
        local_pc1_abs_mean.append(float(alignment["local_pc1_absolute_cosine_stats"]["mean"]))
        local_pc1_abs_std.append(float(alignment["local_pc1_absolute_cosine_stats"]["std"]))
        subspace_overlap_mean.append(float(alignment["local_subspace_overlap_stats"]["mean"]))
        subspace_overlap_std.append(float(alignment["local_subspace_overlap_stats"]["std"]))
        word_mean_abs_mean.append(float(alignment["word_mean_absolute_cosine_stats"]["mean"]))
        word_mean_abs_std.append(float(alignment["word_mean_absolute_cosine_stats"]["std"]))

    x = np.array(layer_percents, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, local_pc1_abs_mean, marker="o", linewidth=2, color="#1f77b4", label="Mean |Local PC1 Cosine|")
    ax.fill_between(x, np.array(local_pc1_abs_mean) - np.array(local_pc1_abs_std), np.array(local_pc1_abs_mean) + np.array(local_pc1_abs_std), color="#1f77b4", alpha=0.15)
    ax.plot(x, subspace_overlap_mean, marker="o", linewidth=2, color="#d62728", label="Mean Local Subspace Overlap")
    ax.fill_between(x, np.array(subspace_overlap_mean) - np.array(subspace_overlap_std), np.array(subspace_overlap_mean) + np.array(subspace_overlap_std), color="#d62728", alpha=0.15)
    ax.plot(x, word_mean_abs_mean, marker="o", linewidth=2, color="#2ca02c", label="Mean |Word Mean Cosine|")
    ax.fill_between(x, np.array(word_mean_abs_mean) - np.array(word_mean_abs_std), np.array(word_mean_abs_mean) + np.array(word_mean_abs_std), color="#2ca02c", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Alignment")
    ax.set_xticks(layer_percents)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_cross_model_gap_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    pc1_diag = []
    pc1_offdiag = []
    pc1_gap = []
    subspace_diag = []
    subspace_offdiag = []
    subspace_gap = []

    for layer_percent in layer_percents:
        similarity = summary["layers"][str(layer_percent)]["hider_vs_guesser_similarity"]
        pc1_diag.append(float(similarity["pc1_absolute_diagonal_mean"]))
        pc1_offdiag.append(float(similarity["pc1_absolute_off_diagonal_mean"]))
        pc1_gap.append(float(similarity["pc1_absolute_diagonal_minus_off_diagonal"]))
        subspace_diag.append(float(similarity["subspace_diagonal_mean"]))
        subspace_offdiag.append(float(similarity["subspace_off_diagonal_mean"]))
        subspace_gap.append(float(similarity["subspace_diagonal_minus_off_diagonal"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].plot(layer_percents, pc1_diag, marker="o", linewidth=2, color="#1f77b4", label="Diagonal")
    axes[0].plot(layer_percents, pc1_offdiag, marker="o", linewidth=2, color="#d62728", label="Off-Diagonal")
    axes[0].plot(layer_percents, pc1_gap, marker="o", linewidth=2, color="#2ca02c", linestyle="--", label="Gap")
    axes[0].set_title("|Hider PC1 vs Guesser PC1|")
    axes[0].set_xlabel("Layer Percent")
    axes[0].set_ylabel("Similarity")
    axes[0].set_xticks(layer_percents)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(layer_percents, subspace_diag, marker="o", linewidth=2, color="#1f77b4", label="Diagonal")
    axes[1].plot(layer_percents, subspace_offdiag, marker="o", linewidth=2, color="#d62728", label="Off-Diagonal")
    axes[1].plot(layer_percents, subspace_gap, marker="o", linewidth=2, color="#2ca02c", linestyle="--", label="Gap")
    axes[1].set_title("Hider vs Guesser Subspace Similarity")
    axes[1].set_xlabel("Layer Percent")
    axes[1].set_ylabel("Similarity")
    axes[1].set_xticks(layer_percents)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_hider_specific_word_probe_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    linear_train = []
    linear_test = []
    mlp_train = []
    mlp_test = []

    for layer_percent in layer_percents:
        probe_info = summary["layers"][str(layer_percent)]["hider_specific_residual"]["word_probe"]
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
    ax.set_ylabel("Word Probe Accuracy")
    ax.set_xticks(layer_percents)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_hider_specific_role_probe_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    linear_test_acc = []
    mlp_test_acc = []
    linear_test_f1 = []
    mlp_test_f1 = []

    for layer_percent in layer_percents:
        probe_info = summary["layers"][str(layer_percent)]["hider_specific_residual"]["role_probe"]
        linear_test_acc.append(float(probe_info["linear"]["test"]["accuracy"]))
        mlp_test_acc.append(float(probe_info["mlp"]["test"]["accuracy"]))
        linear_test_f1.append(float(probe_info["linear"]["test"]["f1"]))
        mlp_test_f1.append(float(probe_info["mlp"]["test"]["f1"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].plot(layer_percents, linear_test_acc, marker="o", linewidth=2, color="#1f77b4", label="Linear")
    axes[0].plot(layer_percents, mlp_test_acc, marker="s", linewidth=2, color="#d62728", label="MLP")
    axes[0].set_title("Role Probe Test Accuracy")
    axes[0].set_xlabel("Layer Percent")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(layer_percents)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(layer_percents, linear_test_f1, marker="o", linewidth=2, color="#1f77b4", label="Linear")
    axes[1].plot(layer_percents, mlp_test_f1, marker="s", linewidth=2, color="#d62728", label="MLP")
    axes[1].set_title("Role Probe Test F1")
    axes[1].set_xlabel("Layer Percent")
    axes[1].set_ylabel("F1")
    axes[1].set_xticks(layer_percents)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_hider_specific_local_subspace_plot(summary: dict, layer_percents: list[int], title: str, output_path: str) -> str:
    mean_local_subspace = []
    mean_local_pc1_abs = []

    for layer_percent in layer_percents:
        local_pca = summary["layers"][str(layer_percent)]["hider_specific_residual"]["local_pca"]
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


def draw_heatmaps(summary: dict, layer_percents: list[int], target_words: list[str], key_path: list[str], title: str, output_path: str) -> str:
    n_layers = len(layer_percents)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    axes = axes[0]

    images = []
    for ax, layer_percent in zip(axes, layer_percents):
        layer_info = summary["layers"][str(layer_percent)]
        matrix = layer_info
        for key in key_path:
            matrix = matrix[key]
        matrix_np = np.array(matrix, dtype=float)
        image = ax.imshow(matrix_np, vmin=0.0, vmax=1.0, cmap="viridis")
        images.append(image)
        ax.set_title(f"Layer {layer_percent}%")
        ax.set_xticks(range(len(target_words)))
        ax.set_yticks(range(len(target_words)))
        ax.set_xticklabels(target_words, rotation=45, ha="right")
        ax.set_yticklabels(target_words)

    fig.colorbar(images[-1], ax=axes.tolist(), fraction=0.025, pad=0.04, label="Similarity")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_alignment_bar_panels(
    summary: dict,
    layer_percents: list[int],
    target_words: list[str],
    metric_key: str,
    title: str,
    output_path: str,
    y_label: str,
    signed: bool = False,
) -> str:
    n_layers = len(layer_percents)
    fig, axes = plt.subplots(n_layers, 1, figsize=(16, 4.5 * n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer_percent in zip(axes, layer_percents):
        alignment = summary["layers"][str(layer_percent)]["hider_specific_residual"]["global_deception_alignment"]
        values = [alignment[metric_key][target_word] for target_word in target_words]
        positions = np.arange(len(target_words))
        ax.bar(positions, values, color="#1f77b4")
        if signed:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Layer {layer_percent}%")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xticks(np.arange(len(target_words)))
    axes[-1].set_xticklabels(target_words, rotation=45, ha="right")
    axes[-1].set_xlabel("Target Word")
    fig.suptitle(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
            "./images/taboo_hider_guesser_subspace_analysis"
        ) / f"{sanitize_name(model_name)}_{sanitize_name(prompt_type)}_{sanitize_name(dataset_type)}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{model_name} | {prompt_type} | {dataset_type}"

    draw_cross_model_gap_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Hider vs Guesser Similarity Gap",
        output_path=str(output_dir / "hider_guesser_similarity_gap_by_layer.png"),
    )
    draw_hider_specific_word_probe_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Hider-Specific Word Probe",
        output_path=str(output_dir / "hider_specific_word_probe_accuracy_by_layer.png"),
    )
    draw_hider_specific_role_probe_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Hider-Specific Role Probe",
        output_path=str(output_dir / "hider_specific_role_probe_by_layer.png"),
    )
    draw_hider_specific_local_subspace_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Hider-Specific Local Subspace Summary",
        output_path=str(output_dir / "hider_specific_local_subspace_by_layer.png"),
    )
    draw_deception_alignment_summary_plot(
        summary=summary,
        layer_percents=layer_percents,
        title=f"{title_prefix} | Hider-Specific vs Global Deception Alignment",
        output_path=str(output_dir / "hider_specific_vs_deception_alignment_by_layer.png"),
    )
    draw_heatmaps(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        key_path=["hider_vs_guesser_similarity", "pc1_absolute_cosine_matrix"],
        title=f"{title_prefix} | Hider vs Guesser |PC1| Heatmaps",
        output_path=str(output_dir / "hider_guesser_pc1_absolute_heatmaps.png"),
    )
    draw_heatmaps(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        key_path=["hider_vs_guesser_similarity", "subspace_similarity_matrix"],
        title=f"{title_prefix} | Hider vs Guesser Subspace Heatmaps",
        output_path=str(output_dir / "hider_guesser_subspace_heatmaps.png"),
    )
    draw_heatmaps(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        key_path=["hider_specific_residual", "local_pca", "local_subspace_similarity_matrix"],
        title=f"{title_prefix} | Hider-Specific Local Subspace Heatmaps",
        output_path=str(output_dir / "hider_specific_local_subspace_heatmaps.png"),
    )
    draw_alignment_bar_panels(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        metric_key="local_pc1_absolute_cosine_by_target",
        title=f"{title_prefix} | |Local PC1, Global Deception PC1| by Word",
        output_path=str(output_dir / "hider_specific_local_pc1_vs_deception_by_word.png"),
        y_label="|Cosine|",
    )
    draw_alignment_bar_panels(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        metric_key="local_subspace_overlap_by_target",
        title=f"{title_prefix} | Local Subspace Overlap with Global Deception PC1",
        output_path=str(output_dir / "hider_specific_local_subspace_overlap_with_deception_by_word.png"),
        y_label="Overlap",
    )
    draw_alignment_bar_panels(
        summary=summary,
        layer_percents=layer_percents,
        target_words=target_words,
        metric_key="word_mean_signed_cosine_by_target",
        title=f"{title_prefix} | Hider-Specific Word Mean vs Global Deception PC1",
        output_path=str(output_dir / "hider_specific_word_mean_vs_deception_by_word.png"),
        y_label="Cosine",
        signed=True,
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
