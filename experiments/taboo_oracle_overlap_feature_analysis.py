import argparse
import csv
import json
import re
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


TOP_GROUP = "TOP20_OVERLAP"
BOTTOM_GROUP = "BOTTOM20_OVERLAP"
GROUP_COLORS = {
    TOP_GROUP: "#ffd92f",
    BOTTOM_GROUP: "#ff7f0e",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_probe_eval_json(probe_json: Path) -> dict:
    with open(probe_json, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_alignment_filename(csv_path: Path) -> tuple[str, int, str]:
    match = re.match(
        r"prompt_probe_oracle_alignment_(?P<source>[a-zA-Z0-9_]+)_layer_(?P<layer>\d+)_(?P<suffix>.+)\.csv",
        csv_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected alignment csv filename: {csv_path.name}")
    return match.group("source"), int(match.group("layer")), match.group("suffix")


def discover_alignment_csvs(input_dir: Path, source_names: list[str]) -> tuple[dict[str, dict[int, Path]], str]:
    discovered = {source_name: {} for source_name in source_names}
    csv_paths = sorted(input_dir.glob("prompt_probe_oracle_alignment_*.csv"))
    if len(csv_paths) == 0:
        raise ValueError(f"No alignment CSVs found in {input_dir}")

    suffix = None
    for csv_path in csv_paths:
        source_name, layer_percent, parsed_suffix = parse_alignment_filename(csv_path)
        if source_name not in discovered:
            continue
        if suffix is None:
            suffix = parsed_suffix
        if parsed_suffix != suffix:
            raise ValueError(f"Mixed alignment suffixes found in {input_dir}: {suffix} vs {parsed_suffix}")
        discovered[source_name][layer_percent] = csv_path

    for source_name in source_names:
        if len(discovered[source_name]) == 0:
            raise ValueError(f"No alignment CSVs found for source '{source_name}' in {input_dir}")

    return discovered, suffix


def load_alignment_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "oracle_accuracy": float(row["oracle_accuracy"]),
                "probe_score": float(row["probe_score"]) if "probe_score" in row else None,
                "primary_category": row["primary_category"],
                "secondary_tags": row["secondary_tags"],
            }
        )
    return parsed_rows


def compute_shared_oracle_overlap_groups(
    source_a_rows: list[dict],
    source_b_rows: list[dict],
    top_k: int,
    bottom_k: int,
) -> dict:
    sorted_a = sorted(source_a_rows, key=lambda row: row["oracle_accuracy"], reverse=True)
    sorted_b = sorted(source_b_rows, key=lambda row: row["oracle_accuracy"], reverse=True)

    top_a = {row["prompt_text"] for row in sorted_a[:top_k]}
    top_b = {row["prompt_text"] for row in sorted_b[:top_k]}
    bottom_a = {row["prompt_text"] for row in sorted_a[-bottom_k:]}
    bottom_b = {row["prompt_text"] for row in sorted_b[-bottom_k:]}

    top_overlap = sorted(top_a & top_b)
    bottom_overlap = sorted(bottom_a & bottom_b)

    if len(top_overlap) == 0 or len(bottom_overlap) == 0:
        raise ValueError("Shared top/bottom Oracle overlap groups are empty")

    return {
        "top_overlap_prompts": top_overlap,
        "bottom_overlap_prompts": bottom_overlap,
        "top_overlap_count": len(top_overlap),
        "bottom_overlap_count": len(bottom_overlap),
    }


def initialize_prompt_sums(prompt_texts: list[str], d_model: int) -> dict[str, torch.Tensor]:
    return {prompt_text: torch.zeros(d_model, dtype=torch.float32) for prompt_text in prompt_texts}


def initialize_prompt_vector_lists(prompt_texts: list[str]) -> dict[str, list[torch.Tensor]]:
    return {prompt_text: [] for prompt_text in prompt_texts}


def compute_prompt_vectors_for_selected_groups(
    model_name: str,
    source_name: str,
    secret_words: list[str],
    context_prompts: list[str],
    layer_percents: list[int],
    group_prompts_by_layer: dict[int, dict[str, list[str]]],
    hider_lora_path_arg: str | None,
    guesser_lora_path_arg: str | None,
    eval_batch_size: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    layer_indices = [layer_percent_to_layer(model_name, layer_percent) for layer_percent in layer_percents]
    prompt_index_by_text = {prompt_text: idx for idx, prompt_text in enumerate(context_prompts)}

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
                selected_prompts = group_prompts_by_layer[layer_percent]["selected_prompts"]
                prompt_sums_by_layer[layer_percent] = initialize_prompt_sums(selected_prompts, d_model)
                prompt_vectors_by_layer[layer_percent] = initialize_prompt_vector_lists(selected_prompts)

        context_acts = collect_context_prompt_last_token_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layer_indices,
            device=device,
            eval_batch_size=eval_batch_size,
        )

        for layer_percent, layer_index in zip(layer_percents, layer_indices):
            for prompt_text in group_prompts_by_layer[layer_percent]["selected_prompts"]:
                prompt_idx = prompt_index_by_text[prompt_text]
                prompt_vector = context_acts[layer_index][prompt_idx]
                prompt_sums_by_layer[layer_percent][prompt_text] += prompt_vector
                prompt_vectors_by_layer[layer_percent][prompt_text].append(prompt_vector)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prompt_means_by_layer = {}
    prompt_target_vectors_by_layer = {}
    for layer_percent in layer_percents:
        prompt_means_by_layer[layer_percent] = {
            prompt_text: prompt_vector / len(secret_words)
            for prompt_text, prompt_vector in prompt_sums_by_layer[layer_percent].items()
        }
        prompt_target_vectors_by_layer[layer_percent] = {
            prompt_text: torch.stack(prompt_vectors, dim=0)
            for prompt_text, prompt_vectors in prompt_vectors_by_layer[layer_percent].items()
        }

    return prompt_target_vectors_by_layer, prompt_means_by_layer


def compute_group_similarity_rows(
    prompt_target_vectors: dict[str, torch.Tensor],
    prompt_mean_vectors: dict[str, torch.Tensor],
    top_prompts: list[str],
    bottom_prompts: list[str],
) -> tuple[list[dict], dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ordered_prompts = top_prompts + bottom_prompts
    ordered_groups = [TOP_GROUP] * len(top_prompts) + [BOTTOM_GROUP] * len(bottom_prompts)
    target_vectors = torch.stack([prompt_target_vectors[prompt_text] for prompt_text in ordered_prompts]).float()
    mean_vectors = torch.stack([prompt_mean_vectors[prompt_text] for prompt_text in ordered_prompts]).float()
    normalized_target_vectors = F.normalize(target_vectors, dim=-1)
    cosine_matrix = torch.einsum("pwd,qwd->pqw", normalized_target_vectors, normalized_target_vectors).mean(dim=-1)

    group_rows = []
    top_count = len(top_prompts)
    bottom_count = len(bottom_prompts)
    top_vectors = mean_vectors[:top_count]
    bottom_vectors = mean_vectors[top_count:]

    top_within_values = []
    top_cross_values = []
    bottom_within_values = []
    bottom_cross_values = []

    for idx, (prompt_text, group_label) in enumerate(zip(ordered_prompts, ordered_groups)):
        if group_label == TOP_GROUP:
            within_indices = [j for j in range(top_count) if j != idx]
            cross_indices = list(range(top_count, top_count + bottom_count))
        else:
            within_indices = [j for j in range(top_count, top_count + bottom_count) if j != idx]
            cross_indices = list(range(top_count))

        within_mean = float(cosine_matrix[idx, within_indices].mean().item())
        cross_mean = float(cosine_matrix[idx, cross_indices].mean().item())
        delta = within_mean - cross_mean

        if group_label == TOP_GROUP:
            top_within_values.append(within_mean)
            top_cross_values.append(cross_mean)
        else:
            bottom_within_values.append(within_mean)
            bottom_cross_values.append(cross_mean)

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

    top_mean = top_vectors.mean(dim=0)
    bottom_mean = bottom_vectors.mean(dim=0)
    feature_raw = top_mean - bottom_mean
    feature_unit = F.normalize(feature_raw.view(1, -1), dim=-1).view(-1)

    summary = {
        "num_top_overlap_prompts": top_count,
        "num_bottom_overlap_prompts": bottom_count,
        "top_within_mean": float(torch.tensor(top_within_values).mean().item()),
        "top_to_bottom_mean": float(torch.tensor(top_cross_values).mean().item()),
        "bottom_within_mean": float(torch.tensor(bottom_within_values).mean().item()),
        "bottom_to_top_mean": float(torch.tensor(bottom_cross_values).mean().item()),
        "pooled_within_mean": float(torch.tensor(top_within_values + bottom_within_values).mean().item()),
        "pooled_cross_mean": float(torch.tensor(top_cross_values + bottom_cross_values).mean().item()),
        "feature_raw_norm": float(feature_raw.norm().item()),
        "feature_unit_norm": float(feature_unit.norm().item()),
        "cosine_top_mean_vs_bottom_mean": float(
            F.cosine_similarity(top_mean.view(1, -1), bottom_mean.view(1, -1), dim=-1).item()
        ),
        "similarity_mode": "mean_targetwise_prompt_cosine",
        "feature_formula": "mean(shared_top_overlap_prompts) - mean(shared_bottom_overlap_prompts)",
    }
    return group_rows, summary, cosine_matrix, top_mean, bottom_mean, feature_raw, feature_unit


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
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    for group_label in [TOP_GROUP, BOTTOM_GROUP]:
        group_rows = [row for row in rows if row["group_label"] == group_label]
        ax.scatter(
            [row["other_group_mean_cosine"] for row in group_rows],
            [row["within_group_mean_cosine"] for row in group_rows],
            color=GROUP_COLORS[group_label],
            alpha=0.86,
            s=48,
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
    colors = [GROUP_COLORS[row["group_label"]] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(sorted_rows)), 5.8))
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


def save_shared_group_csv(output_path: Path, top_prompts: list[str], bottom_prompts: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_label", "prompt_text"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt_text in top_prompts:
            writer.writerow({"group_label": TOP_GROUP, "prompt_text": prompt_text})
        for prompt_text in bottom_prompts:
            writer.writerow({"group_label": BOTTOM_GROUP, "prompt_text": prompt_text})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_json", type=str, required=True)
    parser.add_argument("--alignment_dir", type=str, required=True)
    parser.add_argument("--source_names", type=str, nargs=2, default=["hider", "guesser"])
    parser.add_argument("--layer_percents", type=int, nargs="+", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--bottom_k", type=int, default=20)
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./taboo_eval_results")
    args = parser.parse_args()

    probe_json = Path(args.probe_json)
    alignment_dir = Path(args.alignment_dir)
    probe_data = load_probe_eval_json(probe_json)

    model_name = probe_data["config"]["model_name"]
    context_prompts = list(probe_data["context_prompts"])
    secret_words = [word.strip() for word in args.secret_words.split(",")]
    layer_percents = args.layer_percents if args.layer_percents is not None else list(probe_data["config"]["layer_percents"])
    alignment_csvs_by_source, suffix = discover_alignment_csvs(alignment_dir, args.source_names)

    shared_groups_by_layer = {}
    for layer_percent in layer_percents:
        source_a_rows = load_alignment_rows(alignment_csvs_by_source[args.source_names[0]][layer_percent])
        source_b_rows = load_alignment_rows(alignment_csvs_by_source[args.source_names[1]][layer_percent])
        shared_groups_by_layer[layer_percent] = compute_shared_oracle_overlap_groups(
            source_a_rows=source_a_rows,
            source_b_rows=source_b_rows,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
        )
        shared_groups_by_layer[layer_percent]["selected_prompts"] = (
            shared_groups_by_layer[layer_percent]["top_overlap_prompts"]
            + shared_groups_by_layer[layer_percent]["bottom_overlap_prompts"]
        )

    model_name_short = sanitize_name(model_name.split("/")[-1])
    prompt_type = probe_data["config"]["prompt_type"]
    dataset_type = probe_data["config"]["dataset_type"]
    lang_suffix = f"_{probe_data['config']['lang_type']}" if probe_data["config"]["lang_type"] else ""
    output_dir = Path(
        args.output_dir,
        f"{model_name_short}_oracle_overlap_feature_analysis_{prompt_type}{lang_suffix}_{dataset_type}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tensors = {
        "config": {
            "model_name": model_name,
            "prompt_type": prompt_type,
            "dataset_type": dataset_type,
            "lang_type": probe_data["config"]["lang_type"],
            "secret_words": secret_words,
            "layer_percents": layer_percents,
            "alignment_dir": str(alignment_dir),
            "source_names": args.source_names,
            "top_k": args.top_k,
            "bottom_k": args.bottom_k,
            "alignment_suffix": suffix,
        },
        "sources": {},
    }
    summary_json = {"config": feature_tensors["config"], "layers": {}, "sources": {}}

    for layer_percent in layer_percents:
        save_shared_group_csv(
            output_path=output_dir / f"shared_oracle_overlap_groups_layer_{layer_percent}.csv",
            top_prompts=shared_groups_by_layer[layer_percent]["top_overlap_prompts"],
            bottom_prompts=shared_groups_by_layer[layer_percent]["bottom_overlap_prompts"],
        )
        summary_json["layers"][str(layer_percent)] = shared_groups_by_layer[layer_percent]

    for source_name in args.source_names:
        prompt_target_vectors_by_layer, prompt_mean_vectors_by_layer = compute_prompt_vectors_for_selected_groups(
            model_name=model_name,
            source_name=source_name,
            secret_words=secret_words,
            context_prompts=context_prompts,
            layer_percents=layer_percents,
            group_prompts_by_layer=shared_groups_by_layer,
            hider_lora_path_arg=args.hider_lora_path,
            guesser_lora_path_arg=args.guesser_lora_path,
            eval_batch_size=args.eval_batch_size,
        )

        feature_tensors["sources"][source_name] = {}
        summary_json["sources"][source_name] = {"layers": {}}

        for layer_percent in layer_percents:
            top_prompts = shared_groups_by_layer[layer_percent]["top_overlap_prompts"]
            bottom_prompts = shared_groups_by_layer[layer_percent]["bottom_overlap_prompts"]
            group_rows, similarity_summary, _, top_mean, bottom_mean, feature_raw, feature_unit = compute_group_similarity_rows(
                prompt_target_vectors=prompt_target_vectors_by_layer[layer_percent],
                prompt_mean_vectors=prompt_mean_vectors_by_layer[layer_percent],
                top_prompts=top_prompts,
                bottom_prompts=bottom_prompts,
            )

            similarity_csv = output_dir / f"oracle_overlap_similarity_{source_name}_layer_{layer_percent}.csv"
            save_group_similarity_csv(similarity_csv, group_rows)

            scatter_path = output_dir / f"oracle_overlap_similarity_scatter_{source_name}_layer_{layer_percent}.png"
            draw_similarity_scatter(
                rows=group_rows,
                output_path=scatter_path,
                title=f"{source_name} | Layer {layer_percent}% | shared Oracle top/bottom overlap prompt similarity",
            )

            delta_bar_path = output_dir / f"oracle_overlap_similarity_delta_{source_name}_layer_{layer_percent}.png"
            draw_similarity_delta_bars(
                rows=group_rows,
                output_path=delta_bar_path,
                title=f"{source_name} | Layer {layer_percent}% | within-group minus cross-group similarity",
            )

            feature_tensors["sources"][source_name][str(layer_percent)] = {
                "top_overlap_mean": top_mean,
                "bottom_overlap_mean": bottom_mean,
                "feature_raw": feature_raw,
                "feature_unit": feature_unit,
            }
            summary_json["sources"][source_name]["layers"][str(layer_percent)] = {
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

    if "hider" in feature_tensors["sources"] and "guesser" in feature_tensors["sources"]:
        summary_json["cross_source_feature_similarity"] = {}
        for layer_percent in layer_percents:
            hider_feature = feature_tensors["sources"]["hider"][str(layer_percent)]["feature_unit"]
            guesser_feature = feature_tensors["sources"]["guesser"][str(layer_percent)]["feature_unit"]
            cosine = F.cosine_similarity(hider_feature.view(1, -1), guesser_feature.view(1, -1), dim=-1).item()
            summary_json["cross_source_feature_similarity"][str(layer_percent)] = {
                "unit_feature_cosine": float(cosine),
            }

    feature_pt = output_dir / "oracle_overlap_features.pt"
    torch.save(feature_tensors, feature_pt)

    summary_path = output_dir / "oracle_overlap_feature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved Oracle-overlap feature analysis to {output_dir}")


if __name__ == "__main__":
    main()
