import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_context_prompts(config: dict) -> list[str]:
    prompt_type = config["prompt_type"]
    dataset_type = config["dataset_type"]
    lang_type = config["lang_type"]

    if prompt_type == "all_direct":
        if lang_type:
            context_prompt_path = REPO_ROOT / "datasets" / "taboo" / f"taboo_direct_{lang_type}_{dataset_type}.txt"
        else:
            context_prompt_path = REPO_ROOT / "datasets" / "taboo" / f"taboo_direct_{dataset_type}.txt"
    elif prompt_type == "all_standard":
        context_prompt_path = REPO_ROOT / "datasets" / "taboo" / f"taboo_standard_{dataset_type}.txt"
    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    with open(context_prompt_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f]

    max_prompts = config["max_prompts"]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def load_summary(input_json: str) -> tuple[dict, list[int], list[str], list[str], dict[str, dict]]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    layers = data["layers"]
    layer_percents = sorted(int(layer_percent) for layer_percent in layers.keys())
    target_words = list(config["target_lora_suffixes"])
    context_prompts = data.get("context_prompts", load_context_prompts(config))
    return config, layer_percents, target_words, context_prompts, layers


def draw_global_layer_plot(
    layers: dict[str, dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    global_mean_cosines = [float(layers[str(layer_percent)]["global_cosine_stats"]["mean"]) for layer_percent in layer_percents]
    global_std_cosines = [float(layers[str(layer_percent)]["global_cosine_stats"]["std"]) for layer_percent in layer_percents]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layer_percents, global_mean_cosines, marker="o", linewidth=2, color="#1f77b4", label="Global Mean Cosine")
    ax.fill_between(
        layer_percents,
        [m - s for m, s in zip(global_mean_cosines, global_std_cosines)],
        [m + s for m, s in zip(global_mean_cosines, global_std_cosines)],
        alpha=0.2,
        color="#1f77b4",
        label="Mean +/- Std",
    )
    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Hider vs Guesser Cosine")
    ax.set_xticks(layer_percents)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_target_mean_heatmap(
    layers: dict[str, dict],
    layer_percents: list[int],
    target_words: list[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = [
        [float(layers[str(layer_percent)]["per_target"][target_word]["cosine_stats"]["mean"]) for layer_percent in layer_percents]
        for target_word in target_words
    ]

    fig, ax = plt.subplots(figsize=(7, max(6, len(target_words) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Target Word")
    ax.set_xticks(range(len(layer_percents)))
    ax.set_xticklabels(layer_percents)
    ax.set_yticks(range(len(target_words)))
    ax.set_yticklabels(target_words)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean Cosine")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_layer_word_boxplots(
    layers: dict[str, dict],
    layer_percents: list[int],
    target_words: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(layer_percents), 1, figsize=(18, 4.5 * len(layer_percents)), sharex=True)
    if len(layer_percents) == 1:
        axes = [axes]

    for ax, layer_percent in zip(axes, layer_percents):
        layer_info = layers[str(layer_percent)]
        box_data = [layer_info["per_target"][target_word]["prompt_cosine_scores"] for target_word in target_words]
        mean_values = [layer_info["per_target"][target_word]["cosine_stats"]["mean"] for target_word in target_words]

        ax.boxplot(box_data, patch_artist=True, widths=0.6)
        ax.plot(range(1, len(target_words) + 1), mean_values, marker="o", linewidth=2, color="#d62728", label="Mean")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Layer {layer_percent}%")
        ax.set_ylabel("Prompt-Level Cosine")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right")

    axes[-1].set_xticks(range(1, len(target_words) + 1))
    axes[-1].set_xticklabels(target_words, rotation=45, ha="right")
    axes[-1].set_xlabel("Target Word")
    fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_target_line_plot(
    layers: dict[str, dict],
    layer_percents: list[int],
    target_words: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for target_word in target_words:
        series = [float(layers[str(layer_percent)]["per_target"][target_word]["cosine_stats"]["mean"]) for layer_percent in layer_percents]
        ax.plot(layer_percents, series, marker="o", linewidth=1.8, alpha=0.85, label=target_word)

    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Mean Hider vs Guesser Cosine")
    ax.set_xticks(layer_percents)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_context_prompt_rows(
    layers: dict[str, dict],
    layer_percents: list[int],
    target_words: list[str],
    context_prompts: list[str],
) -> list[dict]:
    prompt_rows = []
    for prompt_idx, prompt_text in enumerate(context_prompts):
        layer_means = {}
        values = []
        for layer_percent in layer_percents:
            prompt_scores = [
                float(layers[str(layer_percent)]["per_target"][target_word]["prompt_cosine_scores"][prompt_idx])
                for target_word in target_words
            ]
            layer_mean = sum(prompt_scores) / len(prompt_scores)
            layer_means[layer_percent] = layer_mean
            values.append(layer_mean)

        prompt_rows.append(
            {
                "prompt_id": f"P{prompt_idx + 1:03d}",
                "prompt_text": prompt_text,
                "overall_mean": sum(values) / len(values),
                "layer_means": layer_means,
            }
        )

    prompt_rows.sort(key=lambda row: row["overall_mean"], reverse=True)
    return prompt_rows


def save_context_prompt_csv(
    prompt_rows: list[dict],
    layer_percents: list[int],
    output_path: Path,
) -> None:
    fieldnames = ["prompt_id", "overall_mean", "prompt_text", "raw_context_prompt"] + [
        f"layer_{layer_percent}" for layer_percent in layer_percents
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prompt_rows:
            csv_row = {
                "prompt_id": row["prompt_id"],
                "overall_mean": row["overall_mean"],
                "prompt_text": row["prompt_text"],
                "raw_context_prompt": row["prompt_text"],
            }
            for layer_percent in layer_percents:
                csv_row[f"layer_{layer_percent}"] = row["layer_means"][layer_percent]
            writer.writerow(csv_row)


def draw_context_prompt_heatmap(
    prompt_rows: list[dict],
    layer_percents: list[int],
    output_path: Path,
    title: str,
) -> None:
    matrix = [[row["layer_means"][layer_percent] for layer_percent in layer_percents] for row in prompt_rows]

    fig, ax = plt.subplots(figsize=(8, max(8, len(prompt_rows) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Layer Percent")
    ax.set_ylabel("Context Prompt ID")
    ax.set_xticks(range(len(layer_percents)))
    ax.set_xticklabels(layer_percents)
    ax.set_yticks(range(len(prompt_rows)))
    ax.set_yticklabels([row["prompt_id"] for row in prompt_rows], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean Cosine Across Targets")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_context_prompt_top_bottom(
    prompt_rows: list[dict],
    output_path: Path,
    title: str,
    top_k: int = 15,
) -> None:
    top_rows = prompt_rows[:top_k]
    bottom_rows = list(reversed(prompt_rows[-top_k:]))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.45 * top_k + 2)), sharex=True)
    plot_specs = [
        (axes[0], top_rows, "Top Context Prompts", "#2ca02c"),
        (axes[1], bottom_rows, "Bottom Context Prompts", "#d62728"),
    ]

    for ax, rows, subtitle, color in plot_specs:
        y_labels = [row["prompt_id"] for row in rows]
        values = [row["overall_mean"] for row in rows]
        bars = ax.barh(y_labels, values, color=color, alpha=0.9)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(subtitle)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config, layer_percents, target_words, context_prompts, layers = load_summary(args.input_json)
    model_name_str = sanitize_name(config["model_name"].split("/")[-1])
    lang_suffix = f"_{config['lang_type']}" if config["lang_type"] else ""
    default_output_dir = (
        Path("experiments/plotting/images")
        / "taboo_hider_guesser_direct_similarity"
        / f"{model_name_str}_{config['prompt_type']}{lang_suffix}_{config['dataset_type']}"
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{config['model_name']} | {config['prompt_type']} | {config['dataset_type']}"

    draw_global_layer_plot(
        layers=layers,
        layer_percents=layer_percents,
        output_path=output_dir / "global_mean_cosine_by_layer.png",
        title=f"{title_prefix} | Global Hider vs Guesser Cosine",
    )
    draw_target_mean_heatmap(
        layers=layers,
        layer_percents=layer_percents,
        target_words=target_words,
        output_path=output_dir / "target_mean_cosine_heatmap.png",
        title=f"{title_prefix} | Target Mean Cosine Heatmap",
    )
    draw_layer_word_boxplots(
        layers=layers,
        layer_percents=layer_percents,
        target_words=target_words,
        output_path=output_dir / "prompt_cosine_boxplots_by_layer.png",
        title=f"{title_prefix} | Prompt-Level Hider vs Guesser Cosine",
    )
    draw_target_line_plot(
        layers=layers,
        layer_percents=layer_percents,
        target_words=target_words,
        output_path=output_dir / "target_mean_cosine_lines.png",
        title=f"{title_prefix} | Target Mean Cosine by Layer",
    )

    context_prompt_rows = build_context_prompt_rows(
        layers=layers,
        layer_percents=layer_percents,
        target_words=target_words,
        context_prompts=context_prompts,
    )
    save_context_prompt_csv(
        prompt_rows=context_prompt_rows,
        layer_percents=layer_percents,
        output_path=output_dir / "context_prompt_mean_cosine_by_layer.csv",
    )
    draw_context_prompt_heatmap(
        prompt_rows=context_prompt_rows,
        layer_percents=layer_percents,
        output_path=output_dir / "context_prompt_mean_cosine_heatmap.png",
        title=f"{title_prefix} | Context Prompt Mean Cosine",
    )
    draw_context_prompt_top_bottom(
        prompt_rows=context_prompt_rows,
        output_path=output_dir / "context_prompt_mean_cosine_top_bottom.png",
        title=f"{title_prefix} | Context Prompt Mean Cosine Ranking",
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
