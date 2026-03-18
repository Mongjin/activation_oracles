import argparse
import csv
import json
from pathlib import Path

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


def write_cosine_csv(output_path: Path, target_words: list[str], cosine_matrix: list[list[float]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_word", *target_words])
        for target_word, row in zip(target_words, cosine_matrix):
            writer.writerow([target_word, *row])


def build_difference_vectors(
    hider_vectors: dict[str, dict[int, torch.Tensor]],
    guesser_vectors: dict[str, dict[int, torch.Tensor]],
    base_vectors: dict[int, torch.Tensor],
    target_words: list[str],
    act_layers: list[int],
    analysis_mode: str,
) -> dict[str, dict[int, torch.Tensor]]:
    difference_vectors = {}
    for target_word in target_words:
        difference_vectors[target_word] = {}
        for act_layer in act_layers:
            if analysis_mode == "hider_minus_guesser":
                difference_vectors[target_word][act_layer] = (
                    hider_vectors[target_word][act_layer] - guesser_vectors[target_word][act_layer]
                ).float()
            elif analysis_mode == "hider_minus_base":
                difference_vectors[target_word][act_layer] = (
                    hider_vectors[target_word][act_layer] - base_vectors[act_layer]
                ).float()
            else:
                raise ValueError(f"Unsupported analysis_mode: {analysis_mode}")
    return difference_vectors


def top_neighbors_from_cosine_matrix(target_words: list[str], cosine_matrix: torch.Tensor) -> dict[str, list[dict[str, float]]]:
    neighbors = {}
    for row_idx, target_word in enumerate(target_words):
        sorted_indices = torch.argsort(cosine_matrix[row_idx], descending=True).tolist()
        neighbors[target_word] = [
            {
                "target_word": target_words[col_idx],
                "cosine_similarity": float(cosine_matrix[row_idx, col_idx].item()),
            }
            for col_idx in sorted_indices
            if col_idx != row_idx
        ][:5]
    return neighbors


def compute_pc1(centered_vectors_ND: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, singular_values, vh = torch.linalg.svd(centered_vectors_ND, full_matrices=False)
    pc1_D = F.normalize(vh[0], dim=0)
    explained_variance_ratio = singular_values.square() / singular_values.square().sum()
    scores_N = torch.matmul(centered_vectors_ND, pc1_D)
    return pc1_D, scores_N, explained_variance_ratio


def compute_local_pc1(centered_vectors_PD: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(centered_vectors_PD, full_matrices=False)
    return F.normalize(vh[0], dim=0)


def compute_pairwise_offdiag_values(vectors_XD: torch.Tensor) -> torch.Tensor:
    normalized_XD = F.normalize(vectors_XD, dim=-1)
    cosine_XX = torch.matmul(normalized_XD, normalized_XD.T)
    off_diag_mask = ~torch.eye(cosine_XX.shape[0], dtype=torch.bool)
    return cosine_XX[off_diag_mask]


def analyze_residual_structure(
    difference_vectors: dict[str, dict[int, torch.Tensor]],
    target_words: list[str],
    layer_percents: list[int],
    act_layers: list[int],
    pca_variants: list[str],
) -> dict[str, dict]:
    summary = {}

    for layer_percent, act_layer in zip(layer_percents, act_layers):
        target_prompt_vectors = torch.stack(
            [difference_vectors[target_word][act_layer] for target_word in target_words],
            dim=0,
        ).float()
        num_targets, num_prompts, d_model = target_prompt_vectors.shape

        layer_summary = {
            "layer_percent": layer_percent,
            "layer_index": act_layer,
            "num_targets": num_targets,
            "num_prompts": num_prompts,
        }

        for pca_variant in pca_variants:
            if pca_variant == "raw":
                working_target_prompt_vectors = target_prompt_vectors
            elif pca_variant == "unit":
                working_target_prompt_vectors = F.normalize(target_prompt_vectors, dim=-1)
            else:
                raise ValueError(f"Unsupported pca_variant: {pca_variant}")

            flat_vectors_ND = working_target_prompt_vectors.reshape(num_targets * num_prompts, d_model)
            global_mean_D = flat_vectors_ND.mean(dim=0, keepdim=True)
            centered_vectors_ND = flat_vectors_ND - global_mean_D

            global_pc1_D, global_scores_N, explained_variance_ratio = compute_pc1(centered_vectors_ND)
            residual_vectors_ND = centered_vectors_ND - torch.outer(global_scores_N, global_pc1_D)
            residual_vectors_TPD = residual_vectors_ND.reshape(num_targets, num_prompts, d_model)

            residual_word_means_TD = residual_vectors_TPD.mean(dim=1)
            normalized_residual_word_means_TD = F.normalize(residual_word_means_TD, dim=-1)
            residual_word_mean_cosine_matrix = (
                normalized_residual_word_means_TD @ normalized_residual_word_means_TD.T
            )

            original_local_pc1_vs_global = {}
            residual_local_pc1_vs_global = {}
            same_word_similarity_by_target = {}
            same_word_similarity_values = []

            for target_idx, target_word in enumerate(target_words):
                original_vectors_PD = working_target_prompt_vectors[target_idx]
                original_centered_PD = original_vectors_PD - original_vectors_PD.mean(dim=0, keepdim=True)
                original_local_pc1_D = compute_local_pc1(original_centered_PD)
                original_local_pc1_vs_global[target_word] = float(torch.dot(original_local_pc1_D, global_pc1_D).item())

                residual_vectors_PD = residual_vectors_TPD[target_idx]
                residual_centered_PD = residual_vectors_PD - residual_vectors_PD.mean(dim=0, keepdim=True)
                residual_local_pc1_D = compute_local_pc1(residual_centered_PD)
                residual_local_pc1_vs_global[target_word] = float(torch.dot(residual_local_pc1_D, global_pc1_D).item())

                same_word_values = compute_pairwise_offdiag_values(residual_vectors_PD)
                same_word_similarity_by_target[target_word] = summarize_values(same_word_values)
                same_word_similarity_values.append(same_word_values)

            different_word_same_prompt_values = []
            different_word_same_prompt_by_prompt = {}
            for prompt_idx in range(num_prompts):
                prompt_vectors_TD = residual_vectors_TPD[:, prompt_idx, :]
                prompt_normalized_TD = F.normalize(prompt_vectors_TD, dim=-1)
                prompt_cosine_TT = prompt_normalized_TD @ prompt_normalized_TD.T
                off_diag_mask = ~torch.eye(prompt_cosine_TT.shape[0], dtype=torch.bool)
                off_diag_values = prompt_cosine_TT[off_diag_mask]
                different_word_same_prompt_by_prompt[str(prompt_idx)] = summarize_values(off_diag_values)
                different_word_same_prompt_values.append(off_diag_values)

            same_word_similarity_values = torch.cat(same_word_similarity_values, dim=0)
            different_word_same_prompt_values = torch.cat(different_word_same_prompt_values, dim=0)

            target_projection_by_word = {}
            for target_idx, target_word in enumerate(target_words):
                target_scores_P = global_scores_N.reshape(num_targets, num_prompts)[target_idx]
                target_projection_by_word[target_word] = {
                    "scores": target_scores_P.tolist(),
                    "stats": summarize_values(target_scores_P),
                }

            layer_summary[pca_variant] = {
                "global_pc1_direction": global_pc1_D.tolist(),
                "explained_variance_ratio_pc1": float(explained_variance_ratio[0].item()),
                "global_pc1_projection_stats": summarize_values(global_scores_N),
                "global_pc1_projection_by_target": target_projection_by_word,
                "residual_word_mean_cosine_matrix": residual_word_mean_cosine_matrix.tolist(),
                "residual_word_mean_top_neighbors": top_neighbors_from_cosine_matrix(
                    target_words, residual_word_mean_cosine_matrix
                ),
                "original_local_pc1_vs_global_pc1_cosine": original_local_pc1_vs_global,
                "residual_local_pc1_vs_global_pc1_cosine": residual_local_pc1_vs_global,
                "same_word_similarity": {
                    "global_stats": summarize_values(same_word_similarity_values),
                    "by_target": same_word_similarity_by_target,
                },
                "different_word_same_prompt_similarity": {
                    "global_stats": summarize_values(different_word_same_prompt_values),
                    "by_prompt_index": different_word_same_prompt_by_prompt,
                },
                "same_minus_different_mean": float(
                    same_word_similarity_values.mean().item() - different_word_same_prompt_values.mean().item()
                ),
            }

        summary[str(layer_percent)] = layer_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--analysis_modes",
        type=str,
        nargs="+",
        default=["hider_minus_guesser", "hider_minus_base"],
        choices=["hider_minus_guesser", "hider_minus_base"],
    )
    parser.add_argument(
        "--pca_variants",
        type=str,
        nargs="+",
        default=["raw", "unit"],
        choices=["raw", "unit"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    payload = torch.load(input_dir / "pooled_hider_activations.pt", map_location="cpu")

    config = payload["config"]
    act_layers = payload["act_layers"]
    base_vectors = payload["base_vectors"]
    per_target_hider_vectors = payload["per_target_hider_vectors"]
    per_target_guesser_vectors = payload["per_target_guesser_vectors"]

    target_words = config["target_lora_suffixes"]
    layer_percents = config["layer_percents"]

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for analysis_mode in args.analysis_modes:
        difference_vectors = build_difference_vectors(
            hider_vectors=per_target_hider_vectors,
            guesser_vectors=per_target_guesser_vectors,
            base_vectors=base_vectors,
            target_words=target_words,
            act_layers=act_layers,
            analysis_mode=analysis_mode,
        )

        summary = analyze_residual_structure(
            difference_vectors=difference_vectors,
            target_words=target_words,
            layer_percents=layer_percents,
            act_layers=act_layers,
            pca_variants=args.pca_variants,
        )
        all_summaries[analysis_mode] = summary

        with open(output_dir / f"{analysis_mode}_residual_structure_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config,
                    "act_layers": act_layers,
                    "analysis_mode": analysis_mode,
                    "pca_variants": args.pca_variants,
                    "layers": summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        for layer_percent in layer_percents:
            layer_info = summary[str(layer_percent)]
            for pca_variant in args.pca_variants:
                write_cosine_csv(
                    output_dir / f"{analysis_mode}_{pca_variant}_residual_word_mean_cosine_layer_{layer_percent}.csv",
                    target_words=target_words,
                    cosine_matrix=layer_info[pca_variant]["residual_word_mean_cosine_matrix"],
                )

    with open(output_dir / "residual_structure_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "act_layers": act_layers,
                "analysis_modes": all_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved residual structure analysis to {output_dir}")


if __name__ == "__main__":
    main()
