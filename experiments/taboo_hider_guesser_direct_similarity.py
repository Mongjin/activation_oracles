import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def summarize_values(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "median": float(values.median().item()),
    }


def build_direct_similarity_summary(data: dict[str, Any]) -> dict[str, Any]:
    config = data["config"]
    act_layers = data["act_layers"]
    target_words = list(config["target_lora_suffixes"])
    layer_percents = list(config["layer_percents"])
    per_target_hider_vectors = data["per_target_hider_vectors"]
    per_target_guesser_vectors = data["per_target_guesser_vectors"]

    layers_summary: dict[str, dict[str, Any]] = {}
    for layer_percent, act_layer in zip(layer_percents, act_layers):
        per_target_summary: dict[str, dict[str, Any]] = {}
        all_prompt_cosines = []
        all_prompt_inner_products = []

        for target_word in target_words:
            hider_vectors_PD = per_target_hider_vectors[target_word][act_layer].float()
            guesser_vectors_PD = per_target_guesser_vectors[target_word][act_layer].float()

            cosine_scores_P = F.cosine_similarity(hider_vectors_PD, guesser_vectors_PD, dim=-1)
            inner_products_P = (hider_vectors_PD * guesser_vectors_PD).sum(dim=-1)
            hider_norms_P = hider_vectors_PD.norm(dim=-1)
            guesser_norms_P = guesser_vectors_PD.norm(dim=-1)

            per_target_summary[target_word] = {
                "num_prompts": int(cosine_scores_P.shape[0]),
                "prompt_cosine_scores": cosine_scores_P.tolist(),
                "prompt_inner_products": inner_products_P.tolist(),
                "cosine_stats": summarize_values(cosine_scores_P),
                "inner_product_stats": summarize_values(inner_products_P),
                "hider_norm_stats": summarize_values(hider_norms_P),
                "guesser_norm_stats": summarize_values(guesser_norms_P),
            }

            all_prompt_cosines.append(cosine_scores_P)
            all_prompt_inner_products.append(inner_products_P)

        all_prompt_cosines_P = torch.cat(all_prompt_cosines, dim=0)
        all_prompt_inner_products_P = torch.cat(all_prompt_inner_products, dim=0)

        layers_summary[str(layer_percent)] = {
            "layer_percent": int(layer_percent),
            "layer_index": int(act_layer),
            "target_words": target_words,
            "per_target": per_target_summary,
            "global_cosine_stats": summarize_values(all_prompt_cosines_P),
            "global_inner_product_stats": summarize_values(all_prompt_inner_products_P),
            "target_mean_cosines": {
                target_word: float(per_target_summary[target_word]["cosine_stats"]["mean"])
                for target_word in target_words
            },
            "target_mean_inner_products": {
                target_word: float(per_target_summary[target_word]["inner_product_stats"]["mean"])
                for target_word in target_words
            },
        }

    return {
        "config": config,
        "act_layers": act_layers,
        "layers": layers_summary,
    }


def write_target_stat_csv(
    output_path: Path,
    layer_percents: list[int],
    target_words: list[str],
    summary: dict[str, Any],
    stat_key: str,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_word", *layer_percents])
        for target_word in target_words:
            row = [target_word]
            for layer_percent in layer_percents:
                row.append(summary["layers"][str(layer_percent)]["per_target"][target_word][stat_key]["mean"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_pt = Path(args.input_pt)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_pt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(input_pt, map_location="cpu")
    summary = build_direct_similarity_summary(data)

    summary_path = output_dir / "hider_guesser_direct_similarity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    layer_percents = list(summary["config"]["layer_percents"])
    target_words = list(summary["config"]["target_lora_suffixes"])
    write_target_stat_csv(
        output_path=output_dir / "hider_guesser_direct_similarity_mean_cosine_by_target.csv",
        layer_percents=layer_percents,
        target_words=target_words,
        summary=summary,
        stat_key="cosine_stats",
    )
    write_target_stat_csv(
        output_path=output_dir / "hider_guesser_direct_similarity_mean_inner_product_by_target.csv",
        layer_percents=layer_percents,
        target_words=target_words,
        summary=summary,
        stat_key="inner_product_stats",
    )

    print(f"Saved direct similarity summary to {summary_path}")


if __name__ == "__main__":
    main()
