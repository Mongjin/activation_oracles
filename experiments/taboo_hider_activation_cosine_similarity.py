import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from peft import LoraConfig
from tqdm import tqdm

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TARGET_LORA_SUFFIXES = [
    "ship",
    "wave",
    "song",
    "snow",
    "rock",
    "moon",
    "jump",
    "green",
    "flame",
    "flag",
    "dance",
    "cloud",
    "clock",
    "chair",
    "salt",
    "book",
    "blue",
    "gold",
    "leaf",
    "smile",
]


@dataclass
class HiderActivationSimilarityConfig:
    model_name: str
    prompt_type: str
    dataset_type: str
    lang_type: Optional[str]
    eval_batch_size: int
    add_generation_prompt: bool
    enable_thinking: bool
    layer_percents: list[int]
    pooling: str
    segment_start_idx: int
    segment_end_idx: int
    max_prompts: Optional[int]
    target_lora_suffixes: list[str]
    target_lora_path_template: str
    guesser_lora_path_template: Optional[str]
    analysis_modes: list[str]


def infer_target_lora_path_template(model_name: str) -> str:
    if model_name == "Qwen/Qwen3-8B":
        return "adamkarvonen/Qwen3-8B-taboo-{lora_path}_50_mix"
    if model_name == "google/gemma-2-9b-it":
        return "bcywinski/gemma-2-9b-it-taboo-{lora_path}"
    raise ValueError(f"Unsupported model_name: {model_name}")


def infer_guesser_lora_path_template(model_name: str) -> str:
    model_suffix = model_name.split("/")[-1]
    return f"/home/mongjin/activation_oracles/nl_probes/trl_training/model_lora_role_swapped/{model_suffix}-taboo-{{lora_path}}-role-swapped"


def load_context_prompts(prompt_type: str, dataset_type: str, lang_type: Optional[str]) -> list[str]:
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
        return [line.strip() for line in f]


def pool_single_activation(
    acts_LD: torch.Tensor,
    pooling: str,
    segment_start_idx: int,
    segment_end_idx: int,
) -> torch.Tensor:
    if pooling == "last_token":
        return acts_LD[-1]

    if pooling == "full_seq_mean":
        return acts_LD.mean(dim=0)

    if pooling == "segment_mean":
        if segment_start_idx < 0:
            segment_start = acts_LD.shape[0] + segment_start_idx
            segment_end = acts_LD.shape[0] + segment_end_idx
        else:
            segment_start = segment_start_idx
            segment_end = segment_end_idx

        return acts_LD[segment_start:segment_end].mean(dim=0)

    raise ValueError(f"Unsupported pooling mode: {pooling}")


def pool_batch_activations(
    acts_BLD: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    pooling: str,
    segment_start_idx: int,
    segment_end_idx: int,
) -> torch.Tensor:
    pooled = []
    seq_len = acts_BLD.shape[1]

    for batch_idx in range(acts_BLD.shape[0]):
        real_len = int(attention_mask_BL[batch_idx].sum().item())
        left_pad = seq_len - real_len
        acts_LD = acts_BLD[batch_idx, left_pad:, :]
        pooled.append(pool_single_activation(acts_LD, pooling, segment_start_idx, segment_end_idx))

    return torch.stack(pooled, dim=0)


def collect_target_word_activations(
    model,
    tokenizer,
    context_prompts: list[str],
    adapter_name: str,
    act_layers: list[int],
    config: HiderActivationSimilarityConfig,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    model.enable_adapters()
    model.set_adapter(adapter_name)

    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    pooled_by_layer = {layer: [] for layer in act_layers}

    for start in range(0, len(context_prompts), config.eval_batch_size):
        batch_prompts = context_prompts[start : start + config.eval_batch_size]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

        inputs_BL = base_experiment.encode_messages(
            tokenizer=tokenizer,
            message_dicts=batch_messages,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )

        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

        for layer in act_layers:
            pooled_BD = pool_batch_activations(
                acts_BLD=acts_by_layer[layer],
                attention_mask_BL=inputs_BL["attention_mask"],
                pooling=config.pooling,
                segment_start_idx=config.segment_start_idx,
                segment_end_idx=config.segment_end_idx,
            )
            pooled_by_layer[layer].append(pooled_BD.cpu())

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in pooled_by_layer.items()}


def collect_base_activations(
    model,
    tokenizer,
    context_prompts: list[str],
    act_layers: list[int],
    config: HiderActivationSimilarityConfig,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    model.disable_adapters()

    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    pooled_by_layer = {layer: [] for layer in act_layers}

    for start in range(0, len(context_prompts), config.eval_batch_size):
        batch_prompts = context_prompts[start : start + config.eval_batch_size]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

        inputs_BL = base_experiment.encode_messages(
            tokenizer=tokenizer,
            message_dicts=batch_messages,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )

        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

        for layer in act_layers:
            pooled_BD = pool_batch_activations(
                acts_BLD=acts_by_layer[layer],
                attention_mask_BL=inputs_BL["attention_mask"],
                pooling=config.pooling,
                segment_start_idx=config.segment_start_idx,
                segment_end_idx=config.segment_end_idx,
            )
            pooled_by_layer[layer].append(pooled_BD.cpu())

    model.enable_adapters()
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in pooled_by_layer.items()}


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


def build_cosine_summary(
    per_target_vectors: dict[str, dict[int, torch.Tensor]],
    layer_percents: list[int],
    act_layers: list[int],
    ordered_targets: list[str],
    context_prompts: list[str],
) -> dict[str, dict]:
    summary: dict[str, dict] = {}

    for layer_percent, act_layer in zip(layer_percents, act_layers):
        target_prompt_vectors = torch.stack(
            [per_target_vectors[target][act_layer] for target in ordered_targets],
            dim=0,
        ).float()
        normalized_prompt_vectors = F.normalize(target_prompt_vectors, dim=-1)

        prompt_major_vectors = normalized_prompt_vectors.permute(1, 0, 2)
        prompt_cosine_matrices = torch.matmul(prompt_major_vectors, prompt_major_vectors.transpose(1, 2))
        mean_cosine_matrix = prompt_cosine_matrices.mean(dim=0)
        std_cosine_matrix = prompt_cosine_matrices.std(dim=0, unbiased=False)

        neighbors = {}
        for row_idx, target in enumerate(ordered_targets):
            sorted_indices = torch.argsort(mean_cosine_matrix[row_idx], descending=True).tolist()
            neighbors[target] = [
                {
                    "target_word": ordered_targets[col_idx],
                    "mean_cosine_similarity": float(mean_cosine_matrix[row_idx, col_idx].item()),
                    "std_cosine_similarity": float(std_cosine_matrix[row_idx, col_idx].item()),
                }
                for col_idx in sorted_indices
                if col_idx != row_idx
            ][:5]

        same_prompt_pair_cosines = {}
        for row_idx, row_target in enumerate(ordered_targets):
            for col_idx in range(row_idx + 1, len(ordered_targets)):
                col_target = ordered_targets[col_idx]
                pair_key = f"{row_target}__{col_target}"
                same_prompt_pair_cosines[pair_key] = prompt_cosine_matrices[:, row_idx, col_idx].tolist()

        within_target_prompt_pairwise_stats = {}
        prompt_to_centroid_cosines = {}
        for target_idx, target in enumerate(ordered_targets):
            target_vectors_PD = normalized_prompt_vectors[target_idx]
            within_target_cosine = torch.matmul(target_vectors_PD, target_vectors_PD.T)
            off_diag_mask = ~torch.eye(within_target_cosine.shape[0], dtype=torch.bool)
            off_diag_values = within_target_cosine[off_diag_mask]

            target_centroid_D = F.normalize(target_prompt_vectors[target_idx].mean(dim=0), dim=0)
            target_prompt_to_centroid = torch.matmul(target_vectors_PD, target_centroid_D)

            within_target_prompt_pairwise_stats[target] = {
                "pairwise_prompt_cosine": summarize_values(off_diag_values),
                "prompt_to_centroid_cosine": summarize_values(target_prompt_to_centroid),
            }
            prompt_to_centroid_cosines[target] = target_prompt_to_centroid.tolist()

        summary[str(layer_percent)] = {
            "layer_percent": layer_percent,
            "layer_index": act_layer,
            "target_words": ordered_targets,
            "context_prompts": context_prompts,
            "num_prompts_per_target": {
                target: int(per_target_vectors[target][act_layer].shape[0]) for target in ordered_targets
            },
            "same_prompt_cosine_mean_matrix": mean_cosine_matrix.tolist(),
            "same_prompt_cosine_std_matrix": std_cosine_matrix.tolist(),
            "same_prompt_pair_cosines": same_prompt_pair_cosines,
            "top_neighbors_by_mean_same_prompt_cosine": neighbors,
            "within_target_prompt_pairwise_stats": within_target_prompt_pairwise_stats,
            "prompt_to_centroid_cosines": prompt_to_centroid_cosines,
        }

    return summary


def build_difference_vectors(
    hider_vectors: dict[str, dict[int, torch.Tensor]],
    guesser_vectors: dict[str, dict[int, torch.Tensor]],
    base_vectors: dict[int, torch.Tensor],
    ordered_targets: list[str],
    act_layers: list[int],
    analysis_mode: str,
) -> dict[str, dict[int, torch.Tensor]]:
    per_target_difference_vectors: dict[str, dict[int, torch.Tensor]] = {}

    for target in ordered_targets:
        per_target_difference_vectors[target] = {}
        for act_layer in act_layers:
            if analysis_mode == "hider_minus_guesser":
                per_target_difference_vectors[target][act_layer] = (
                    hider_vectors[target][act_layer] - guesser_vectors[target][act_layer]
                )
            elif analysis_mode == "hider_minus_base":
                per_target_difference_vectors[target][act_layer] = (
                    hider_vectors[target][act_layer] - base_vectors[act_layer]
                )
            else:
                raise ValueError(f"Unsupported analysis_mode: {analysis_mode}")

    return per_target_difference_vectors


def write_cosine_csv(
    output_path: Path,
    target_words: list[str],
    cosine_matrix: list[list[float]],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_word", *target_words])
        for target_word, row in zip(target_words, cosine_matrix):
            writer.writerow([target_word, *row])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--lang_type", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="all_direct", choices=["all_direct", "all_standard"])
    parser.add_argument("--dataset_type", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument(
        "--pooling",
        type=str,
        default="segment_mean",
        choices=["segment_mean", "full_seq_mean", "last_token"],
    )
    parser.add_argument("--segment_start_idx", type=int, default=-10)
    parser.add_argument("--segment_end_idx", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--target_lora_suffixes", type=str, nargs="+", default=None)
    parser.add_argument("--target_lora_path_template", type=str, default=None)
    parser.add_argument("--guesser_lora_path_template", type=str, default=None)
    parser.add_argument(
        "--analysis_modes",
        type=str,
        nargs="+",
        default=["hider_minus_guesser", "hider_minus_base"],
        choices=["hider_minus_guesser", "hider_minus_base"],
    )
    parser.add_argument("--output_dir", type=str, default="./taboo_eval_results")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    target_lora_suffixes = args.target_lora_suffixes or DEFAULT_TARGET_LORA_SUFFIXES
    target_lora_path_template = args.target_lora_path_template or infer_target_lora_path_template(args.model_name)
    guesser_lora_path_template = args.guesser_lora_path_template or infer_guesser_lora_path_template(args.model_name)

    config = HiderActivationSimilarityConfig(
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        dataset_type=args.dataset_type,
        lang_type=args.lang_type,
        eval_batch_size=args.eval_batch_size,
        add_generation_prompt=True,
        enable_thinking=False,
        layer_percents=args.layer_percents,
        pooling=args.pooling,
        segment_start_idx=args.segment_start_idx,
        segment_end_idx=args.segment_end_idx,
        max_prompts=args.max_prompts,
        target_lora_suffixes=target_lora_suffixes,
        target_lora_path_template=target_lora_path_template,
        guesser_lora_path_template=guesser_lora_path_template,
        analysis_modes=args.analysis_modes,
    )

    context_prompts = load_context_prompts(config.prompt_type, config.dataset_type, config.lang_type)
    if config.max_prompts is not None:
        context_prompts = context_prompts[: config.max_prompts]

    model_name_str = config.model_name.split("/")[-1].replace(".", "_")
    lang_suffix = f"_{config.lang_type}" if config.lang_type else ""
    output_dir = Path(
        f"{args.output_dir}/{model_name_str}_hider_activation_cosine_{config.prompt_type}{lang_suffix}_{config.dataset_type}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    act_layers = [layer_percent_to_layer(config.model_name, layer_percent) for layer_percent in config.layer_percents]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = load_tokenizer(config.model_name)

    print(f"Loading model: {config.model_name} on {device} with dtype={dtype}")
    model = load_model(config.model_name, dtype)
    model.eval()

    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    print("Collecting base activations")
    base_vectors = collect_base_activations(
        model=model,
        tokenizer=tokenizer,
        context_prompts=context_prompts,
        act_layers=act_layers,
        config=config,
        device=device,
    )

    per_target_hider_vectors: dict[str, dict[int, torch.Tensor]] = {}
    per_target_guesser_vectors: dict[str, dict[int, torch.Tensor]] = {}

    pbar = tqdm(total=len(config.target_lora_suffixes), desc="Collect hider/guesser activations")
    for target_word in config.target_lora_suffixes:
        target_lora_path = config.target_lora_path_template.format(lora_path=target_word)
        hider_adapter_name = base_experiment.load_lora_adapter(model, target_lora_path)

        per_target_hider_vectors[target_word] = collect_target_word_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            adapter_name=hider_adapter_name,
            act_layers=act_layers,
            config=config,
            device=device,
        )

        if hider_adapter_name in model.peft_config:
            model.delete_adapter(hider_adapter_name)

        guesser_lora_path = config.guesser_lora_path_template.format(lora_path=target_word)
        guesser_adapter_name = base_experiment.load_lora_adapter(model, guesser_lora_path)

        per_target_guesser_vectors[target_word] = collect_target_word_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            adapter_name=guesser_adapter_name,
            act_layers=act_layers,
            config=config,
            device=device,
        )

        if guesser_adapter_name in model.peft_config:
            model.delete_adapter(guesser_adapter_name)

        pbar.set_postfix({"target": target_word})
        pbar.update(1)

    pbar.close()

    torch.save(
        {
            "config": asdict(config),
            "act_layers": act_layers,
            "base_vectors": base_vectors,
            "per_target_hider_vectors": per_target_hider_vectors,
            "per_target_guesser_vectors": per_target_guesser_vectors,
        },
        output_dir / "pooled_hider_activations.pt",
    )

    all_mode_summaries = {}
    for analysis_mode in config.analysis_modes:
        difference_vectors = build_difference_vectors(
            hider_vectors=per_target_hider_vectors,
            guesser_vectors=per_target_guesser_vectors,
            base_vectors=base_vectors,
            ordered_targets=config.target_lora_suffixes,
            act_layers=act_layers,
            analysis_mode=analysis_mode,
        )
        cosine_summary = build_cosine_summary(
            per_target_vectors=difference_vectors,
            layer_percents=config.layer_percents,
            act_layers=act_layers,
            ordered_targets=config.target_lora_suffixes,
            context_prompts=context_prompts,
        )
        all_mode_summaries[analysis_mode] = cosine_summary

        for layer_percent, layer_info in cosine_summary.items():
            write_cosine_csv(
                output_path=output_dir / f"{analysis_mode}_same_prompt_cosine_mean_layer_{layer_percent}.csv",
                target_words=layer_info["target_words"],
                cosine_matrix=layer_info["same_prompt_cosine_mean_matrix"],
            )
            write_cosine_csv(
                output_path=output_dir / f"{analysis_mode}_same_prompt_cosine_std_layer_{layer_percent}.csv",
                target_words=layer_info["target_words"],
                cosine_matrix=layer_info["same_prompt_cosine_std_matrix"],
            )

    with open(output_dir / "cosine_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(config),
                "act_layers": act_layers,
                "analysis_modes": all_mode_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved pooled activations and cosine similarities to {output_dir}")
