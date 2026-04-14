import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm

from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from taboo_context_prompt_probe_eval import (
    DEFAULT_SECRET_WORDS,
    collect_context_prompt_last_token_activations,
    resolve_guesser_lora_path,
    resolve_hider_lora_path,
)


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_probe_eval_json(probe_json: Path) -> dict:
    with open(probe_json, "r", encoding="utf-8") as f:
        return json.load(f)


def find_group_csv(grouped_dir: Path, source_name: str, layer_percent: int) -> Path:
    matches = sorted(grouped_dir.glob(f"hplo_hpho_groups_{source_name}_layer_{layer_percent}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one group CSV for {source_name} layer {layer_percent} in {grouped_dir}, found {len(matches)}")
    return matches[0]


def load_group_rows(group_csv: Path) -> list[dict]:
    with open(group_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_group_prompts(group_csv: Path) -> tuple[list[dict], list[str], list[str]]:
    rows = load_group_rows(group_csv)
    hplo_prompts = [row["prompt_text"] for row in rows if row["group_label"] == "HPLO"]
    hpho_prompts = [row["prompt_text"] for row in rows if row["group_label"] == "HPHO"]
    if len(hplo_prompts) == 0 or len(hpho_prompts) == 0:
        raise ValueError(f"HPLO/HPHO prompts not found in {group_csv}")
    return rows, hplo_prompts, hpho_prompts


def initialize_prompt_sums(prompt_texts: list[str], d_model: int) -> dict[str, torch.Tensor]:
    return {prompt_text: torch.zeros(d_model, dtype=torch.float32) for prompt_text in prompt_texts}


def initialize_prompt_vector_lists(prompt_texts: list[str]) -> dict[str, list[torch.Tensor]]:
    return {prompt_text: [] for prompt_text in prompt_texts}


def compute_prompt_mean_vectors(
    model_name: str,
    source_name: str,
    secret_words: list[str],
    context_prompts: list[str],
    layer_percents: list[int],
    grouped_dir: Path,
    hider_lora_path_arg: str | None,
    guesser_lora_path_arg: str | None,
    eval_batch_size: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]], dict[int, dict[str, list[str]]]]:
    layer_indices = [layer_percent_to_layer(model_name, layer_percent) for layer_percent in layer_percents]
    prompt_index_by_text = {prompt_text: idx for idx, prompt_text in enumerate(context_prompts)}

    group_metadata = {}
    for layer_percent in layer_percents:
        group_csv = find_group_csv(grouped_dir, source_name, layer_percent)
        _, hplo_prompts, hpho_prompts = load_group_prompts(group_csv)
        group_metadata[layer_percent] = {
            "hplo_prompts": hplo_prompts,
            "hpho_prompts": hpho_prompts,
            "selected_prompts": sorted(set(hplo_prompts) | set(hpho_prompts)),
        }

    tokenizer = load_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_sums_by_layer = {}
    prompt_vectors_by_layer = {}
    d_model = None

    for target_word in tqdm(secret_words, desc=f"Collect {source_name} context activations"):
        if source_name == "hider":
            lora_path = resolve_hider_lora_path(model_name, target_word, hider_lora_path_arg)
        elif source_name == "guesser":
            lora_path = resolve_guesser_lora_path(model_name, target_word, guesser_lora_path_arg)
        else:
            raise ValueError(f"Unsupported source_name: {source_name}")

        model = PeftModel.from_pretrained(
            load_model(model_name, torch.bfloat16),
            lora_path,
        )
        model.eval()

        if d_model is None:
            d_model = model.config.hidden_size
            for layer_percent in layer_percents:
                prompt_sums_by_layer[layer_percent] = initialize_prompt_sums(
                    group_metadata[layer_percent]["selected_prompts"],
                    d_model,
                )
                prompt_vectors_by_layer[layer_percent] = initialize_prompt_vector_lists(
                    group_metadata[layer_percent]["selected_prompts"],
                )

        context_acts = collect_context_prompt_last_token_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layer_indices,
            device=device,
            eval_batch_size=eval_batch_size,
        )

        for layer_percent, layer_index in zip(layer_percents, layer_indices):
            for prompt_text in group_metadata[layer_percent]["selected_prompts"]:
                prompt_idx = prompt_index_by_text[prompt_text]
                prompt_vector = context_acts[layer_index][prompt_idx]
                prompt_sums_by_layer[layer_percent][prompt_text] += prompt_vector
                prompt_vectors_by_layer[layer_percent][prompt_text].append(prompt_vector)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prompt_means_by_layer = {}
    for layer_percent in layer_percents:
        prompt_means_by_layer[layer_percent] = {
            prompt_text: prompt_vector / len(secret_words)
            for prompt_text, prompt_vector in prompt_sums_by_layer[layer_percent].items()
        }

    prompt_target_vectors_by_layer = {}
    for layer_percent in layer_percents:
        prompt_target_vectors_by_layer[layer_percent] = {
            prompt_text: torch.stack(prompt_vectors, dim=0)
            for prompt_text, prompt_vectors in prompt_vectors_by_layer[layer_percent].items()
        }

    return prompt_target_vectors_by_layer, prompt_means_by_layer, group_metadata


def compute_group_similarity_rows(
    prompt_target_vectors: dict[str, torch.Tensor],
    prompt_mean_vectors: dict[str, torch.Tensor],
    hplo_prompts: list[str],
    hpho_prompts: list[str],
) -> tuple[list[dict], dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ordered_prompts = hplo_prompts + hpho_prompts
    ordered_groups = ["HPLO"] * len(hplo_prompts) + ["HPHO"] * len(hpho_prompts)
    target_vectors = torch.stack([prompt_target_vectors[prompt_text] for prompt_text in ordered_prompts]).float()
    mean_vectors = torch.stack([prompt_mean_vectors[prompt_text] for prompt_text in ordered_prompts]).float()
    normalized_target_vectors = F.normalize(target_vectors, dim=-1)
    cosine_matrix = torch.einsum("pwd,qwd->pqw", normalized_target_vectors, normalized_target_vectors).mean(dim=-1)

    group_rows = []
    hplo_count = len(hplo_prompts)
    hpho_count = len(hpho_prompts)
    hplo_vectors = mean_vectors[:hplo_count]
    hpho_vectors = mean_vectors[hplo_count:]

    hplo_within_values = []
    hplo_cross_values = []
    hpho_within_values = []
    hpho_cross_values = []

    for idx, (prompt_text, group_label) in enumerate(zip(ordered_prompts, ordered_groups)):
        if group_label == "HPLO":
            within_indices = [j for j in range(hplo_count) if j != idx]
            cross_indices = list(range(hplo_count, hplo_count + hpho_count))
        else:
            within_indices = [j for j in range(hplo_count, hplo_count + hpho_count) if j != idx]
            cross_indices = list(range(hplo_count))

        within_mean = float(cosine_matrix[idx, within_indices].mean().item())
        cross_mean = float(cosine_matrix[idx, cross_indices].mean().item())
        delta = within_mean - cross_mean

        if group_label == "HPLO":
            hplo_within_values.append(within_mean)
            hplo_cross_values.append(cross_mean)
        else:
            hpho_within_values.append(within_mean)
            hpho_cross_values.append(cross_mean)

        group_rows.append(
            {
                "prompt_id": f"P{idx + 1:03d}",
                "prompt_text": prompt_text,
                "group_label": group_label,
                "within_group_mean_cosine": within_mean,
                "other_group_mean_cosine": cross_mean,
                "within_minus_other": delta,
            }
        )

    hplo_mean = hplo_vectors.mean(dim=0)
    hpho_mean = hpho_vectors.mean(dim=0)
    feature_raw = hplo_mean - hpho_mean
    feature_unit = F.normalize(feature_raw.view(1, -1), dim=-1).view(-1)

    summary = {
        "num_hplo_prompts": hplo_count,
        "num_hpho_prompts": hpho_count,
        "hplo_within_mean": float(torch.tensor(hplo_within_values).mean().item()),
        "hplo_to_hpho_mean": float(torch.tensor(hplo_cross_values).mean().item()),
        "hpho_within_mean": float(torch.tensor(hpho_within_values).mean().item()),
        "hpho_to_hplo_mean": float(torch.tensor(hpho_cross_values).mean().item()),
        "pooled_within_mean": float(torch.tensor(hplo_within_values + hpho_within_values).mean().item()),
        "pooled_cross_mean": float(torch.tensor(hplo_cross_values + hpho_cross_values).mean().item()),
        "feature_raw_norm": float(feature_raw.norm().item()),
        "feature_unit_norm": float(feature_unit.norm().item()),
        "cosine_hplo_mean_vs_hpho_mean": float(
            F.cosine_similarity(hplo_mean.view(1, -1), hpho_mean.view(1, -1), dim=-1).item()
        ),
        "similarity_mode": "mean_targetwise_prompt_cosine",
    }
    return group_rows, summary, cosine_matrix, mean_vectors, hplo_mean, hpho_mean, feature_raw, feature_unit


def save_group_similarity_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "group_label",
        "within_group_mean_cosine",
        "other_group_mean_cosine",
        "within_minus_other",
        "prompt_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["group_label"], item["within_minus_other"]), reverse=True):
            writer.writerow({field: row[field] for field in fieldnames})


def draw_similarity_scatter(rows: list[dict], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    colors = {"HPLO": "#d62728", "HPHO": "#2ca02c"}

    for group_label in ["HPLO", "HPHO"]:
        group_rows = [row for row in rows if row["group_label"] == group_label]
        ax.scatter(
            [row["other_group_mean_cosine"] for row in group_rows],
            [row["within_group_mean_cosine"] for row in group_rows],
            color=colors[group_label],
            alpha=0.85,
            s=45,
            label=f"{group_label} (n={len(group_rows)})",
        )

    all_values = [row["within_group_mean_cosine"] for row in rows] + [row["other_group_mean_cosine"] for row in rows]
    line_min = min(all_values)
    line_max = max(all_values)
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Other-group mean cosine")
    ax.set_ylabel("Within-group mean cosine")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_similarity_delta_bars(rows: list[dict], output_path: Path, title: str) -> None:
    sorted_rows = sorted(rows, key=lambda item: item["within_minus_other"], reverse=True)
    labels = [row["prompt_id"] for row in sorted_rows]
    values = [row["within_minus_other"] for row in sorted_rows]
    colors = ["#d62728" if row["group_label"] == "HPLO" else "#2ca02c" for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(sorted_rows)), 5.8))
    bars = ax.bar(range(len(sorted_rows)), values, color=colors, alpha=0.9)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=7, rotation=90)
    ax.set_xticks(range(len(sorted_rows)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Within-group mean cosine - other-group mean cosine")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_json", type=str, required=True)
    parser.add_argument("--grouped_dir", type=str, required=True)
    parser.add_argument("--source_names", type=str, nargs="+", default=["hider", "guesser"])
    parser.add_argument("--layer_percents", type=int, nargs="+", default=None)
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./taboo_eval_results")
    args = parser.parse_args()

    probe_json = Path(args.probe_json)
    grouped_dir = Path(args.grouped_dir)
    probe_data = load_probe_eval_json(probe_json)

    model_name = probe_data["config"]["model_name"]
    context_prompts = list(probe_data["context_prompts"])
    secret_words = [word.strip() for word in args.secret_words.split(",")]
    layer_percents = args.layer_percents if args.layer_percents is not None else list(probe_data["config"]["layer_percents"])

    model_name_short = sanitize_name(model_name.split("/")[-1])
    prompt_type = probe_data["config"]["prompt_type"]
    dataset_type = probe_data["config"]["dataset_type"]
    lang_suffix = f"_{probe_data['config']['lang_type']}" if probe_data["config"]["lang_type"] else ""
    output_dir = Path(
        args.output_dir,
        f"{model_name_short}_hplo_hpho_feature_analysis_{prompt_type}{lang_suffix}_{dataset_type}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tensors = {"config": {
        "model_name": model_name,
        "prompt_type": prompt_type,
        "dataset_type": dataset_type,
        "lang_type": probe_data["config"]["lang_type"],
        "secret_words": secret_words,
        "layer_percents": layer_percents,
        "grouped_dir": str(grouped_dir),
    }, "sources": {}}
    summary_json = {"config": feature_tensors["config"], "sources": {}}

    for source_name in args.source_names:
        prompt_target_vectors_by_layer, prompt_means_by_layer, group_metadata = compute_prompt_mean_vectors(
            model_name=model_name,
            source_name=source_name,
            secret_words=secret_words,
            context_prompts=context_prompts,
            layer_percents=layer_percents,
            grouped_dir=grouped_dir,
            hider_lora_path_arg=args.hider_lora_path,
            guesser_lora_path_arg=args.guesser_lora_path,
            eval_batch_size=args.eval_batch_size,
        )

        feature_tensors["sources"][source_name] = {}
        summary_json["sources"][source_name] = {"layers": {}}

        for layer_percent in layer_percents:
            hplo_prompts = group_metadata[layer_percent]["hplo_prompts"]
            hpho_prompts = group_metadata[layer_percent]["hpho_prompts"]
            group_rows, similarity_summary, _, _, hplo_mean, hpho_mean, feature_raw, feature_unit = compute_group_similarity_rows(
                prompt_target_vectors=prompt_target_vectors_by_layer[layer_percent],
                prompt_mean_vectors=prompt_means_by_layer[layer_percent],
                hplo_prompts=hplo_prompts,
                hpho_prompts=hpho_prompts,
            )

            similarity_csv = output_dir / (
                f"hplo_hpho_similarity_{source_name}_layer_{layer_percent}.csv"
            )
            save_group_similarity_csv(similarity_csv, group_rows)

            scatter_path = output_dir / (
                f"hplo_hpho_similarity_scatter_{source_name}_layer_{layer_percent}.png"
            )
            draw_similarity_scatter(
                rows=group_rows,
                output_path=scatter_path,
                title=f"{source_name} | Layer {layer_percent}% | prompt mean activation similarity",
            )

            delta_bar_path = output_dir / (
                f"hplo_hpho_similarity_delta_{source_name}_layer_{layer_percent}.png"
            )
            draw_similarity_delta_bars(
                rows=group_rows,
                output_path=delta_bar_path,
                title=f"{source_name} | Layer {layer_percent}% | within-group minus cross-group similarity",
            )

            feature_tensors["sources"][source_name][str(layer_percent)] = {
                "hplo_mean": hplo_mean,
                "hpho_mean": hpho_mean,
                "feature_raw": feature_raw,
                "feature_unit": feature_unit,
            }
            summary_json["sources"][source_name]["layers"][str(layer_percent)] = {
                "hplo_prompts": hplo_prompts,
                "hpho_prompts": hpho_prompts,
                "similarity_summary": similarity_summary,
                "top_similarity_delta_prompts": [
                    {
                        "prompt_text": row["prompt_text"],
                        "group_label": row["group_label"],
                        "within_minus_other": row["within_minus_other"],
                    }
                    for row in sorted(group_rows, key=lambda item: item["within_minus_other"], reverse=True)[:10]
                ],
                "bottom_similarity_delta_prompts": [
                    {
                        "prompt_text": row["prompt_text"],
                        "group_label": row["group_label"],
                        "within_minus_other": row["within_minus_other"],
                    }
                    for row in sorted(group_rows, key=lambda item: item["within_minus_other"])[:10]
                ],
            }

    feature_pt = output_dir / "hplo_hpho_features.pt"
    torch.save(feature_tensors, feature_pt)

    if "hider" in feature_tensors["sources"] and "guesser" in feature_tensors["sources"]:
        summary_json["cross_source_feature_similarity"] = {}
        for layer_percent in layer_percents:
            hider_feature = feature_tensors["sources"]["hider"][str(layer_percent)]["feature_unit"]
            guesser_feature = feature_tensors["sources"]["guesser"][str(layer_percent)]["feature_unit"]
            cosine = F.cosine_similarity(hider_feature.view(1, -1), guesser_feature.view(1, -1), dim=-1).item()
            summary_json["cross_source_feature_similarity"][str(layer_percent)] = {
                "unit_feature_cosine": float(cosine),
            }

    summary_path = output_dir / "hplo_hpho_feature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved HPLO/HPHO feature analysis to {output_dir}")


if __name__ == "__main__":
    main()
