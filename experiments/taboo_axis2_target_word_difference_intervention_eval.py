import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from tqdm import tqdm

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation
from taboo_activation_intervention_eval import get_verbalizer_lora_paths, get_verbalizer_prompts
from taboo_axis1_shared_feature_intervention_eval import (
    add_baseline_deltas,
    apply_feature_intervention,
    build_cross_source_gap_rows,
    collect_prompt_metric_names,
    collect_source_activations,
    create_last_token_verbalizer_inputs,
    create_token_verbalizer_inputs,
    default_oracle_token_eval_index,
    encode_context_prompt_infos,
    get_intervention_modes,
    normalize_vector,
    save_cross_source_gap_csv,
    save_prompt_summary_csv,
    select_prompt_eval_acts,
    summarize_cross_source_gap_rows,
    summarize_prompt_metrics,
    train_source_probes,
)
from taboo_context_prompt_probe_eval import DEFAULT_SECRET_WORDS, load_context_prompts


os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_alignment_filename(csv_path: Path) -> tuple[str, int, str]:
    match = re.match(
        r"prompt_probe_oracle_alignment_(?P<source>[a-zA-Z0-9_]+)_layer_(?P<layer>\d+)_(?P<suffix>.+)\.csv",
        csv_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected alignment csv filename: {csv_path.name}")
    return match.group("source"), int(match.group("layer")), match.group("suffix")


def discover_alignment_csvs(alignment_dir: Path) -> tuple[dict[str, dict[int, Path]], str]:
    discovered = {"hider": {}, "guesser": {}}
    csv_paths = sorted(alignment_dir.glob("prompt_probe_oracle_alignment_*.csv"))
    if len(csv_paths) == 0:
        raise ValueError(f"No alignment CSVs found in {alignment_dir}")

    suffix = None
    for csv_path in csv_paths:
        source_name, layer_percent, parsed_suffix = parse_alignment_filename(csv_path)
        if source_name not in discovered:
            continue
        if suffix is None:
            suffix = parsed_suffix
        if parsed_suffix != suffix:
            raise ValueError(f"Mixed alignment suffixes found in {alignment_dir}: {suffix} vs {parsed_suffix}")
        discovered[source_name][layer_percent] = csv_path

    for source_name in ["hider", "guesser"]:
        if len(discovered[source_name]) == 0:
            raise ValueError(f"No alignment CSVs found for source '{source_name}' in {alignment_dir}")

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
            }
        )
    return parsed_rows


def compute_shared_prompt_band(
    alignment_dir: Path,
    layer_percents: list[int],
    min_quantile: float,
    max_quantile: float,
) -> tuple[dict[int, list[dict]], dict[int, dict[str, float | int | list[str]]], str]:
    discovered, alignment_suffix = discover_alignment_csvs(alignment_dir)
    selected_rows_by_layer = {}
    selected_summary_by_layer = {}

    for layer_percent in layer_percents:
        hider_rows = load_alignment_rows(discovered["hider"][layer_percent])
        guesser_rows = load_alignment_rows(discovered["guesser"][layer_percent])
        hider_map = {row["prompt_text"]: row for row in hider_rows}
        guesser_map = {row["prompt_text"]: row for row in guesser_rows}
        common_prompts = sorted(set(hider_map.keys()) & set(guesser_map.keys()))
        if len(common_prompts) == 0:
            raise ValueError(f"No common prompts found for layer_percent={layer_percent}")

        hider_values = torch.tensor([hider_map[prompt_text]["oracle_accuracy"] for prompt_text in common_prompts], dtype=torch.float32)
        guesser_values = torch.tensor(
            [guesser_map[prompt_text]["oracle_accuracy"] for prompt_text in common_prompts],
            dtype=torch.float32,
        )

        hider_low = float(torch.quantile(hider_values, min_quantile).item())
        guesser_low = float(torch.quantile(guesser_values, min_quantile).item())
        hider_high = float(torch.quantile(hider_values, max_quantile).item())
        guesser_high = float(torch.quantile(guesser_values, max_quantile).item())

        selected_rows = []
        for prompt_text in common_prompts:
            hider_value = hider_map[prompt_text]["oracle_accuracy"]
            guesser_value = guesser_map[prompt_text]["oracle_accuracy"]
            if hider_low <= hider_value <= hider_high and guesser_low <= guesser_value <= guesser_high:
                selected_rows.append(
                    {
                        "prompt_id": hider_map[prompt_text]["prompt_id"],
                        "prompt_text": prompt_text,
                        "hider_oracle_accuracy": hider_value,
                        "guesser_oracle_accuracy": guesser_value,
                        "mean_oracle_accuracy": (hider_value + guesser_value) / 2.0,
                    }
                )

        if len(selected_rows) == 0:
            raise ValueError(
                f"No prompts selected for layer_percent={layer_percent} with "
                f"min_quantile={min_quantile}, max_quantile={max_quantile}"
            )

        selected_rows = sorted(selected_rows, key=lambda row: row["mean_oracle_accuracy"], reverse=True)
        selected_rows_by_layer[layer_percent] = selected_rows
        selected_summary_by_layer[layer_percent] = {
            "num_common_prompts": len(common_prompts),
            "num_selected_prompts": len(selected_rows),
            "min_quantile": min_quantile,
            "max_quantile": max_quantile,
            "hider_oracle_accuracy_low_threshold": hider_low,
            "hider_oracle_accuracy_high_threshold": hider_high,
            "guesser_oracle_accuracy_low_threshold": guesser_low,
            "guesser_oracle_accuracy_high_threshold": guesser_high,
            "selected_prompt_texts": [row["prompt_text"] for row in selected_rows],
        }

    return selected_rows_by_layer, selected_summary_by_layer, alignment_suffix


def compute_target_word_difference_features(
    prompt_infos: list[dict],
    hider_context_last_acts_by_word: dict[str, dict[int, torch.Tensor]],
    hider_context_sequence_acts_by_word: dict[str, dict[int, torch.Tensor]] | None,
    guesser_context_last_acts_by_word: dict[str, dict[int, torch.Tensor]],
    guesser_context_sequence_acts_by_word: dict[str, dict[int, torch.Tensor]] | None,
    secret_words: list[str],
    layer_percents: list[int],
    layers: list[int],
    selected_rows_by_layer: dict[int, list[dict]],
    oracle_input_type: str,
    oracle_token_eval_index: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict]]:
    prompt_index_by_text = {prompt_info["prompt_text"]: prompt_idx for prompt_idx, prompt_info in enumerate(prompt_infos)}
    feature_vectors_by_layer = {}
    feature_summary_by_layer = {}

    for layer_percent, layer in zip(layer_percents, layers):
        selected_prompt_indices = [
            prompt_index_by_text[row["prompt_text"]]
            for row in selected_rows_by_layer[layer_percent]
        ]
        feature_vectors_by_layer[layer_percent] = {}
        per_word_feature_vectors = []
        per_word_summaries = {}

        for target_word in secret_words:
            if oracle_input_type == "last_token":
                hider_prompt_acts = hider_context_last_acts_by_word[target_word][layer].float().cpu()
                guesser_prompt_acts = guesser_context_last_acts_by_word[target_word][layer].float().cpu()
            elif oracle_input_type == "tokens":
                if hider_context_sequence_acts_by_word is None or guesser_context_sequence_acts_by_word is None:
                    raise ValueError("Sequence activations must be provided for oracle_input_type='tokens'")
                hider_prompt_acts = select_prompt_eval_acts(
                    prompt_infos=prompt_infos,
                    acts=hider_context_sequence_acts_by_word[target_word][layer].float().cpu(),
                    oracle_input_type=oracle_input_type,
                    oracle_token_eval_index=oracle_token_eval_index,
                )
                guesser_prompt_acts = select_prompt_eval_acts(
                    prompt_infos=prompt_infos,
                    acts=guesser_context_sequence_acts_by_word[target_word][layer].float().cpu(),
                    oracle_input_type=oracle_input_type,
                    oracle_token_eval_index=oracle_token_eval_index,
                )
            else:
                raise ValueError(f"Unsupported oracle_input_type: {oracle_input_type}")

            difference_vectors = hider_prompt_acts[selected_prompt_indices] - guesser_prompt_acts[selected_prompt_indices]
            feature_raw = difference_vectors.mean(dim=0)
            feature_unit = normalize_vector(feature_raw)
            cosine_to_feature = torch.matmul(F.normalize(difference_vectors, dim=-1), feature_unit)

            feature_vectors_by_layer[layer_percent][target_word] = feature_unit
            per_word_feature_vectors.append(feature_unit)
            per_word_summaries[target_word] = {
                "feature_formula": "mean_selected_prompts(hider(prompt, target_word) - guesser(prompt, target_word))",
                "feature_raw_norm": float(feature_raw.norm().item()),
                "feature_unit_norm": float(feature_unit.norm().item()),
                "mean_difference_norm": float(difference_vectors.norm(dim=-1).mean().item()),
                "std_difference_norm": float(difference_vectors.norm(dim=-1).std(unbiased=False).item()),
                "mean_cosine_difference_to_feature": float(cosine_to_feature.mean().item()),
                "std_cosine_difference_to_feature": float(cosine_to_feature.std(unbiased=False).item()),
            }

        per_word_feature_matrix = torch.stack(per_word_feature_vectors, dim=0)
        feature_cosine_matrix = torch.matmul(per_word_feature_matrix, per_word_feature_matrix.T)
        upper_indices = torch.triu_indices(feature_cosine_matrix.shape[0], feature_cosine_matrix.shape[1], offset=1)
        pairwise_feature_cosines = feature_cosine_matrix[upper_indices[0], upper_indices[1]]
        feature_summary_by_layer[layer_percent] = {
            "feature_source": "target_word_specific_hider_minus_guesser_selected_prompt_mean",
            "selection_mode": "shared_medium_high_prompt_band",
            "num_selected_prompts": len(selected_prompt_indices),
            "selected_prompt_texts": [row["prompt_text"] for row in selected_rows_by_layer[layer_percent]],
            "oracle_input_type_for_feature": oracle_input_type,
            "oracle_token_eval_index_for_feature": oracle_token_eval_index if oracle_input_type == "tokens" else None,
            "mean_pairwise_cosine_between_target_features": float(pairwise_feature_cosines.mean().item()),
            "std_pairwise_cosine_between_target_features": float(pairwise_feature_cosines.std(unbiased=False).item()),
            "target_words": per_word_summaries,
        }

    return feature_vectors_by_layer, feature_summary_by_layer


def compute_probe_prompt_rows(
    source_name: str,
    context_acts_by_word: dict[str, dict[int, torch.Tensor]],
    probes_by_layer: dict[int, dict],
    feature_vectors_by_layer: dict[int, dict[str, torch.Tensor]],
    intervention_modes: dict[str, float],
    secret_words: list[str],
    context_prompts: list[str],
    layer_percents: list[int],
    layers: list[int],
    probe_modes: list[str],
) -> dict[int, dict[str, list[dict]]]:
    word_to_idx = {word: idx for idx, word in enumerate(secret_words)}
    prompt_rows_by_layer_mode = {}

    for layer_percent, layer in zip(layer_percents, layers):
        layer_probe_info = probes_by_layer[layer]
        prompt_rows_by_layer_mode[layer_percent] = {}

        for mode_name, feature_scale in intervention_modes.items():
            prompt_records = {prompt_idx: [] for prompt_idx in range(len(context_prompts))}

            for target_word in secret_words:
                target_idx = word_to_idx[target_word]
                eval_x = apply_feature_intervention(
                    acts=context_acts_by_word[target_word][layer].float().cpu(),
                    feature_D=feature_vectors_by_layer[layer_percent][target_word].float().cpu(),
                    mode_name=mode_name,
                    add_scale=feature_scale,
                )

                linear_target_probs = None
                linear_preds = None
                linear_target_ranks = None
                mlp_target_probs = None
                mlp_preds = None
                mlp_target_ranks = None
                binary_linear_target_probs = None
                binary_linear_positive_preds = None
                binary_mlp_target_probs = None
                binary_mlp_positive_preds = None

                with torch.no_grad():
                    if "multiclass" in probe_modes:
                        linear_logits = layer_probe_info["linear_probe"](eval_x)
                        linear_probs = torch.softmax(linear_logits, dim=1)
                        linear_preds = torch.argmax(linear_logits, dim=1)
                        linear_target_probs = linear_probs[:, target_idx]
                        linear_target_ranks = torch.argsort(torch.argsort(linear_logits, dim=1, descending=True), dim=1)[:, target_idx] + 1

                        mlp_logits = layer_probe_info["mlp_probe"](eval_x)
                        mlp_probs = torch.softmax(mlp_logits, dim=1)
                        mlp_preds = torch.argmax(mlp_logits, dim=1)
                        mlp_target_probs = mlp_probs[:, target_idx]
                        mlp_target_ranks = torch.argsort(torch.argsort(mlp_logits, dim=1, descending=True), dim=1)[:, target_idx] + 1

                    if "binary" in probe_modes:
                        binary_linear_logits = layer_probe_info["binary_linear_probes"][target_word](eval_x).view(-1)
                        binary_linear_target_probs = torch.sigmoid(binary_linear_logits)
                        binary_linear_positive_preds = (binary_linear_target_probs > 0.5).long()

                        binary_mlp_logits = layer_probe_info["binary_mlp_probes"][target_word](eval_x).view(-1)
                        binary_mlp_target_probs = torch.sigmoid(binary_mlp_logits)
                        binary_mlp_positive_preds = (binary_mlp_target_probs > 0.5).long()

                for prompt_idx, prompt_text in enumerate(context_prompts):
                    record = {
                        "source": source_name,
                        "target_word": target_word,
                        "prompt_id": f"P{prompt_idx + 1:03d}",
                        "prompt_text": prompt_text,
                    }
                    if "multiclass" in probe_modes:
                        record.update(
                            {
                                "linear_target_prob": float(linear_target_probs[prompt_idx].item()),
                                "linear_correct_top1": int(linear_preds[prompt_idx].item() == target_idx),
                                "linear_target_rank": int(linear_target_ranks[prompt_idx].item()),
                                "mlp_target_prob": float(mlp_target_probs[prompt_idx].item()),
                                "mlp_correct_top1": int(mlp_preds[prompt_idx].item() == target_idx),
                                "mlp_target_rank": int(mlp_target_ranks[prompt_idx].item()),
                            }
                        )
                    if "binary" in probe_modes:
                        record.update(
                            {
                                "binary_linear_target_prob": float(binary_linear_target_probs[prompt_idx].item()),
                                "binary_linear_positive_pred": int(binary_linear_positive_preds[prompt_idx].item()),
                                "binary_mlp_target_prob": float(binary_mlp_target_probs[prompt_idx].item()),
                                "binary_mlp_positive_pred": int(binary_mlp_positive_preds[prompt_idx].item()),
                            }
                        )
                    prompt_records[prompt_idx].append(record)

            prompt_rows = []
            for prompt_idx, prompt_text in enumerate(context_prompts):
                records = prompt_records[prompt_idx]
                prompt_row = {
                    "prompt_id": f"P{prompt_idx + 1:03d}",
                    "prompt_text": prompt_text,
                }
                if "multiclass" in probe_modes:
                    prompt_row.update(
                        {
                            "linear_target_prob_mean": sum(record["linear_target_prob"] for record in records) / len(records),
                            "linear_top1_acc": sum(record["linear_correct_top1"] for record in records) / len(records),
                            "linear_mean_rank": sum(record["linear_target_rank"] for record in records) / len(records),
                            "mlp_target_prob_mean": sum(record["mlp_target_prob"] for record in records) / len(records),
                            "mlp_top1_acc": sum(record["mlp_correct_top1"] for record in records) / len(records),
                            "mlp_mean_rank": sum(record["mlp_target_rank"] for record in records) / len(records),
                        }
                    )
                if "binary" in probe_modes:
                    prompt_row.update(
                        {
                            "binary_linear_target_prob_mean": sum(record["binary_linear_target_prob"] for record in records)
                            / len(records),
                            "binary_linear_positive_recall": sum(record["binary_linear_positive_pred"] for record in records)
                            / len(records),
                            "binary_mlp_target_prob_mean": sum(record["binary_mlp_target_prob"] for record in records)
                            / len(records),
                            "binary_mlp_positive_recall": sum(record["binary_mlp_positive_pred"] for record in records)
                            / len(records),
                        }
                    )
                prompt_rows.append(prompt_row)

            prompt_rows_by_layer_mode[layer_percent][mode_name] = prompt_rows

    return prompt_rows_by_layer_mode


def compute_prompt_oracle_rows(
    model_name: str,
    oracle_model,
    tokenizer,
    source_name: str,
    context_last_acts_by_word: dict[str, dict[int, torch.Tensor]],
    context_sequence_acts_by_word: dict[str, dict[int, torch.Tensor]] | None,
    feature_vectors_by_layer: dict[int, dict[str, torch.Tensor]],
    intervention_modes: dict[str, float],
    secret_words: list[str],
    prompt_infos: list[dict],
    layer_percents: list[int],
    verbalizer_lora_paths: list[str | None],
    verbalizer_prompts: list[str],
    oracle_eval_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    oracle_input_type: str,
    oracle_token_start_idx: int,
    oracle_token_end_idx: int,
    oracle_token_eval_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, list[dict]]]:
    prompt_rows_by_layer_mode = {}
    generation_kwargs = {
        "do_sample": do_sample,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }

    for layer_percent in layer_percents:
        layer_index = layer_percent_to_layer(model_name, layer_percent)
        injection_submodule = get_hf_submodule(oracle_model, layer_index)
        prompt_rows_by_layer_mode[layer_percent] = {}

        for mode_name, feature_scale in intervention_modes.items():
            prompt_correct_counts = {prompt_info["prompt_text"]: 0 for prompt_info in prompt_infos}
            prompt_total_counts = {prompt_info["prompt_text"]: 0 for prompt_info in prompt_infos}

            for verbalizer_lora_path in verbalizer_lora_paths:
                oracle_name = "base_model" if verbalizer_lora_path is None else verbalizer_lora_path.split("/")[-1]
                if verbalizer_lora_path is None:
                    oracle_model.set_adapter("default")

                for target_word in tqdm(
                    secret_words,
                    desc=f"Oracle eval | {source_name} | L{layer_percent}% | {mode_name} | {oracle_name}",
                    leave=False,
                ):
                    if oracle_input_type == "last_token":
                        prompt_acts = apply_feature_intervention(
                            acts=context_last_acts_by_word[target_word][layer_index].float().cpu(),
                            feature_D=feature_vectors_by_layer[layer_percent][target_word].float().cpu(),
                            mode_name=mode_name,
                            add_scale=feature_scale,
                        )
                        verbalizer_inputs = create_last_token_verbalizer_inputs(
                            prompt_infos=prompt_infos,
                            prompt_acts_PD=prompt_acts,
                            target_word=target_word,
                            verbalizer_prompts=verbalizer_prompts,
                            layer_index=layer_index,
                            tokenizer=tokenizer,
                            mode_name=mode_name,
                            source_name=source_name,
                            layer_percent=layer_percent,
                            oracle_name=oracle_name,
                        )
                    elif oracle_input_type == "tokens":
                        if context_sequence_acts_by_word is None:
                            raise ValueError("context_sequence_acts_by_word must be provided for oracle_input_type='tokens'")
                        prompt_acts = apply_feature_intervention(
                            acts=context_sequence_acts_by_word[target_word][layer_index].float().cpu(),
                            feature_D=feature_vectors_by_layer[layer_percent][target_word].float().cpu(),
                            mode_name=mode_name,
                            add_scale=feature_scale,
                        )
                        verbalizer_inputs = create_token_verbalizer_inputs(
                            prompt_infos=prompt_infos,
                            prompt_acts_PLD=prompt_acts,
                            target_word=target_word,
                            verbalizer_prompts=verbalizer_prompts,
                            layer_index=layer_index,
                            tokenizer=tokenizer,
                            mode_name=mode_name,
                            source_name=source_name,
                            layer_percent=layer_percent,
                            oracle_name=oracle_name,
                            token_start_idx=oracle_token_start_idx,
                            token_end_idx=oracle_token_end_idx,
                        )
                    else:
                        raise ValueError(f"Unsupported oracle_input_type: {oracle_input_type}")

                    responses = run_evaluation(
                        eval_data=verbalizer_inputs,
                        model=oracle_model,
                        tokenizer=tokenizer,
                        submodule=injection_submodule,
                        device=device,
                        dtype=dtype,
                        global_step=-1,
                        lora_path=verbalizer_lora_path,
                        eval_batch_size=oracle_eval_batch_size,
                        steering_coefficient=1.0,
                        generation_kwargs=generation_kwargs,
                    )

                    if oracle_input_type == "last_token":
                        for response in responses:
                            meta = response.meta_info
                            prompt_text = meta["prompt_text"]
                            ground_truth = meta["ground_truth"].lower()
                            correct = int(ground_truth in response.api_response.lower())
                            prompt_correct_counts[prompt_text] += correct
                            prompt_total_counts[prompt_text] += 1
                    else:
                        response_buckets = {}
                        for response in responses:
                            meta = response.meta_info
                            bucket_key = (
                                meta["prompt_text"],
                                meta["ground_truth"],
                                meta["verbalizer_prompt"],
                                meta["oracle_name"],
                            )
                            if bucket_key not in response_buckets:
                                response_buckets[bucket_key] = {
                                    "token_responses": [None] * int(meta["num_tokens"]),
                                }
                            token_index = int(meta["token_index"])
                            response_buckets[bucket_key]["token_responses"][token_index] = response.api_response

                        for (prompt_text, ground_truth, _, _), bucket in response_buckets.items():
                            selected_response = bucket["token_responses"][oracle_token_eval_index]
                            if selected_response is None:
                                raise ValueError(
                                    f"Selected token response is missing for prompt='{prompt_text}', "
                                    f"token_eval_index={oracle_token_eval_index}"
                                )
                            correct = int(ground_truth.lower() in selected_response.lower())
                            prompt_correct_counts[prompt_text] += correct
                            prompt_total_counts[prompt_text] += 1

            prompt_rows = []
            for prompt_info in prompt_infos:
                prompt_text = prompt_info["prompt_text"]
                total = prompt_total_counts[prompt_text]
                correct = prompt_correct_counts[prompt_text]
                prompt_rows.append(
                    {
                        "prompt_id": prompt_info["prompt_id"],
                        "prompt_text": prompt_text,
                        "oracle_accuracy": correct / total,
                        "oracle_correct_count": correct,
                        "oracle_total_count": total,
                    }
                )
            prompt_rows_by_layer_mode[layer_percent][mode_name] = prompt_rows

    return prompt_rows_by_layer_mode


def save_selected_prompt_band_csv(output_path: Path, selected_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "hider_oracle_accuracy",
        "guesser_oracle_accuracy",
        "mean_oracle_accuracy",
        "prompt_text",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--alignment_dir", type=str, required=True)
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--prompt_type", type=str, default="all_direct", choices=["all_direct", "all_standard"])
    parser.add_argument("--dataset_type", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--lang_type", type=str, default=None)
    parser.add_argument("--feature_prompt_min_quantile", type=float, default=0.5)
    parser.add_argument("--feature_prompt_max_quantile", type=float, default=1.0)
    parser.add_argument("--num_probe_samples", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--context_eval_batch_size", type=int, default=64)
    parser.add_argument("--oracle_eval_batch_size", type=int, default=512)
    parser.add_argument("--max_context_prompts", type=int, default=None)
    parser.add_argument(
        "--probe_modes",
        type=str,
        nargs="+",
        default=["multiclass", "binary"],
        choices=["multiclass", "binary"],
    )
    parser.add_argument(
        "--verbalize_prompt",
        type=str,
        default="secret_word",
        choices=["secret_word", "concept", "intent", "concept_intent"],
    )
    parser.add_argument("--intervention_scale", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--oracle_input_type", type=str, default="tokens", choices=["last_token", "tokens"])
    parser.add_argument("--oracle_token_start_idx", type=int, default=-10)
    parser.add_argument("--oracle_token_end_idx", type=int, default=0)
    parser.add_argument("--oracle_token_eval_index", type=int, default=None)
    parser.add_argument("--top_k_gap_overlap", type=int, default=20)
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./taboo_eval_results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not (0.0 <= args.feature_prompt_min_quantile <= args.feature_prompt_max_quantile <= 1.0):
        raise ValueError("feature prompt quantiles must satisfy 0 <= min <= max <= 1")

    secret_words = [word.strip() for word in args.secret_words.split(",")]
    layer_percents = args.layer_percents
    layers = [layer_percent_to_layer(args.model_name, layer_percent) for layer_percent in layer_percents]
    intervention_modes = get_intervention_modes(args.intervention_scale)
    oracle_token_eval_index = (
        default_oracle_token_eval_index(args.model_name)
        if args.oracle_token_eval_index is None
        else args.oracle_token_eval_index
    )
    if args.oracle_input_type == "tokens":
        if args.oracle_token_start_idx >= args.oracle_token_end_idx:
            raise ValueError("oracle_token_start_idx must be less than oracle_token_end_idx")
        if not (args.oracle_token_start_idx <= oracle_token_eval_index < args.oracle_token_end_idx):
            raise ValueError(
                "oracle_token_eval_index must fall inside the selected token range "
                f"[{args.oracle_token_start_idx}, {args.oracle_token_end_idx})"
            )

    context_prompts = load_context_prompts(args.prompt_type, args.dataset_type, args.lang_type)
    if args.max_context_prompts is not None:
        context_prompts = context_prompts[: args.max_context_prompts]

    alignment_dir = Path(args.alignment_dir)
    selected_rows_by_layer, selected_summary_by_layer, alignment_suffix = compute_shared_prompt_band(
        alignment_dir=alignment_dir,
        layer_percents=layer_percents,
        min_quantile=args.feature_prompt_min_quantile,
        max_quantile=args.feature_prompt_max_quantile,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(args.model_name)
    prompt_infos = encode_context_prompt_infos(tokenizer=tokenizer, context_prompts=context_prompts, device=device)
    context_target_seq_len = max(prompt_info["left_pad"] + prompt_info["num_tokens"] for prompt_info in prompt_infos)

    verbalizer_lora_paths = get_verbalizer_lora_paths(args.model_name)
    verbalizer_prompts = get_verbalizer_prompts(args.verbalize_prompt)

    source_context_last_acts = {}
    source_context_sequence_acts = {}
    source_probes = {}
    source_probe_summary = {}

    for source_name in ["hider", "guesser"]:
        dataset_acts_by_layer, dataset_labels, context_last_acts_by_word, context_sequence_acts_by_word, d_model = collect_source_activations(
            model_name=args.model_name,
            source_name=source_name,
            secret_words=secret_words,
            context_prompts=context_prompts,
            layers=layers,
            tokenizer=tokenizer,
            device=device,
            num_probe_samples=args.num_probe_samples,
            context_eval_batch_size=args.context_eval_batch_size,
            hider_lora_path_arg=args.hider_lora_path,
            guesser_lora_path_arg=args.guesser_lora_path,
            collect_sequence_context_acts=args.oracle_input_type == "tokens",
            context_target_seq_len=context_target_seq_len,
        )
        source_context_last_acts[source_name] = context_last_acts_by_word
        source_context_sequence_acts[source_name] = context_sequence_acts_by_word
        source_probes[source_name] = train_source_probes(
            dataset_acts_by_layer=dataset_acts_by_layer,
            dataset_labels=dataset_labels,
            secret_words=secret_words,
            layers=layers,
            d_model=d_model,
            device=device,
            probe_modes=args.probe_modes,
            lr=args.lr,
            epochs=args.epochs,
            mlp_hidden_dim=args.mlp_hidden_dim,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        source_probe_summary[source_name] = {}
        for layer_percent, layer in zip(layer_percents, layers):
            source_probe_summary[source_name][str(layer_percent)] = {
                "linear_probe_val_acc": source_probes[source_name][layer]["linear_probe_val_acc"],
                "mlp_probe_val_acc": source_probes[source_name][layer]["mlp_probe_val_acc"],
                "binary_probe_val_metrics": source_probes[source_name][layer]["binary_probe_val_metrics"],
            }

    feature_vectors_by_layer, feature_summary_by_layer = compute_target_word_difference_features(
        prompt_infos=prompt_infos,
        hider_context_last_acts_by_word=source_context_last_acts["hider"],
        hider_context_sequence_acts_by_word=source_context_sequence_acts["hider"],
        guesser_context_last_acts_by_word=source_context_last_acts["guesser"],
        guesser_context_sequence_acts_by_word=source_context_sequence_acts["guesser"],
        secret_words=secret_words,
        layer_percents=layer_percents,
        layers=layers,
        selected_rows_by_layer=selected_rows_by_layer,
        oracle_input_type=args.oracle_input_type,
        oracle_token_eval_index=oracle_token_eval_index,
    )

    prompt_probe_rows = {}
    for source_name in ["hider", "guesser"]:
        prompt_probe_rows[source_name] = compute_probe_prompt_rows(
            source_name=source_name,
            context_acts_by_word=source_context_last_acts[source_name],
            probes_by_layer=source_probes[source_name],
            feature_vectors_by_layer=feature_vectors_by_layer,
            intervention_modes=intervention_modes,
            secret_words=secret_words,
            context_prompts=context_prompts,
            layer_percents=layer_percents,
            layers=layers,
            probe_modes=args.probe_modes,
        )
        add_baseline_deltas(prompt_probe_rows[source_name])

    print(f"Loading Oracle model: {args.model_name}")
    oracle_model = load_model(args.model_name, dtype)
    oracle_model.eval()
    oracle_model.add_adapter(LoraConfig(), adapter_name="default")

    prompt_oracle_rows = {}
    for source_name in ["hider", "guesser"]:
        prompt_oracle_rows[source_name] = compute_prompt_oracle_rows(
            model_name=args.model_name,
            oracle_model=oracle_model,
            tokenizer=tokenizer,
            source_name=source_name,
            context_last_acts_by_word=source_context_last_acts[source_name],
            context_sequence_acts_by_word=source_context_sequence_acts[source_name],
            feature_vectors_by_layer=feature_vectors_by_layer,
            intervention_modes=intervention_modes,
            secret_words=secret_words,
            prompt_infos=prompt_infos,
            layer_percents=layer_percents,
            verbalizer_lora_paths=verbalizer_lora_paths,
            verbalizer_prompts=verbalizer_prompts,
            oracle_eval_batch_size=args.oracle_eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            oracle_input_type=args.oracle_input_type,
            oracle_token_start_idx=args.oracle_token_start_idx,
            oracle_token_end_idx=args.oracle_token_end_idx,
            oracle_token_eval_index=oracle_token_eval_index,
            device=device,
            dtype=dtype,
        )
        add_baseline_deltas(prompt_oracle_rows[source_name])

    del oracle_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined_prompt_rows = {"hider": {}, "guesser": {}}
    for source_name in ["hider", "guesser"]:
        for layer_percent in layer_percents:
            combined_prompt_rows[source_name][str(layer_percent)] = {}
            oracle_rows_by_mode = prompt_oracle_rows[source_name][layer_percent]
            probe_rows_by_mode = prompt_probe_rows[source_name][layer_percent]

            for mode_name in intervention_modes:
                oracle_map = {row["prompt_text"]: row for row in oracle_rows_by_mode[mode_name]}
                probe_map = {row["prompt_text"]: row for row in probe_rows_by_mode[mode_name]}
                combined_rows = []
                for prompt_info in prompt_infos:
                    prompt_text = prompt_info["prompt_text"]
                    oracle_row = oracle_map[prompt_text]
                    probe_row = probe_map[prompt_text]
                    combined_row = {
                        "prompt_id": prompt_info["prompt_id"],
                        "prompt_text": prompt_text,
                    }
                    combined_row.update(oracle_row)
                    for key, value in probe_row.items():
                        if key not in ("prompt_id", "prompt_text"):
                            combined_row[key] = value
                    combined_rows.append(combined_row)
                combined_prompt_rows[source_name][str(layer_percent)][mode_name] = combined_rows

    cross_source_gap_rows = {}
    cross_source_gap_summary = {}
    for layer_percent in layer_percents:
        cross_source_gap_rows[str(layer_percent)] = {}
        baseline_gap_rows = build_cross_source_gap_rows(
            hider_rows=combined_prompt_rows["hider"][str(layer_percent)]["baseline"],
            guesser_rows=combined_prompt_rows["guesser"][str(layer_percent)]["baseline"],
        )
        cross_source_gap_rows[str(layer_percent)]["baseline"] = baseline_gap_rows
        cross_source_gap_summary[str(layer_percent)] = {
            "baseline": summarize_cross_source_gap_rows(
                baseline_rows=baseline_gap_rows,
                current_rows=baseline_gap_rows,
                top_k=args.top_k_gap_overlap,
            )
        }

        for mode_name in ["add", "subtract"]:
            current_gap_rows = build_cross_source_gap_rows(
                hider_rows=combined_prompt_rows["hider"][str(layer_percent)][mode_name],
                guesser_rows=combined_prompt_rows["guesser"][str(layer_percent)][mode_name],
            )
            cross_source_gap_rows[str(layer_percent)][mode_name] = current_gap_rows
            cross_source_gap_summary[str(layer_percent)][mode_name] = summarize_cross_source_gap_rows(
                baseline_rows=baseline_gap_rows,
                current_rows=current_gap_rows,
                top_k=args.top_k_gap_overlap,
            )

    source_metric_summaries = {
        source_name: summarize_prompt_metrics(prompt_rows_by_layer_mode)
        for source_name, prompt_rows_by_layer_mode in combined_prompt_rows.items()
    }

    model_name_short = args.model_name.split("/")[-1].replace(".", "_")
    lang_suffix = f"_{args.lang_type}" if args.lang_type else ""
    output_dir = Path(
        args.output_dir,
        (
            f"{model_name_short}_axis2_target_word_hider_minus_guesser_band_intervention_"
            f"{args.prompt_type}{lang_suffix}_{args.dataset_type}_{args.oracle_input_type}_{args.intervention_scale}"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_percent in layer_percents:
        save_selected_prompt_band_csv(
            output_path=output_dir / f"axis2_feature_prompt_band_layer_{layer_percent}.csv",
            selected_rows=selected_rows_by_layer[layer_percent],
        )

    for source_name in ["hider", "guesser"]:
        for layer_percent in layer_percents:
            for mode_name in intervention_modes:
                save_prompt_summary_csv(
                    output_path=output_dir / f"axis2_prompt_summary_{source_name}_layer_{layer_percent}_{mode_name}.csv",
                    prompt_rows=combined_prompt_rows[source_name][str(layer_percent)][mode_name],
                )

    for layer_percent in layer_percents:
        baseline_rows = cross_source_gap_rows[str(layer_percent)]["baseline"]
        for mode_name in intervention_modes:
            save_cross_source_gap_csv(
                output_path=output_dir / f"axis2_cross_source_gap_layer_{layer_percent}_{mode_name}.csv",
                rows=cross_source_gap_rows[str(layer_percent)][mode_name],
                baseline_rows=baseline_rows,
            )

    final_results = {
        "config": {
            "model_name": args.model_name,
            "alignment_dir": str(alignment_dir),
            "alignment_suffix": alignment_suffix,
            "prompt_type": args.prompt_type,
            "dataset_type": args.dataset_type,
            "lang_type": args.lang_type,
            "feature_prompt_min_quantile": args.feature_prompt_min_quantile,
            "feature_prompt_max_quantile": args.feature_prompt_max_quantile,
            "secret_words": secret_words,
            "num_probe_samples": args.num_probe_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "val_fraction": args.val_fraction,
            "layer_percents": layer_percents,
            "context_eval_batch_size": args.context_eval_batch_size,
            "oracle_eval_batch_size": args.oracle_eval_batch_size,
            "max_context_prompts": args.max_context_prompts,
            "probe_modes": args.probe_modes,
            "verbalize_prompt": args.verbalize_prompt,
            "intervention_scale": args.intervention_scale,
            "feature_source": "target_word_specific_hider_minus_guesser_selected_prompt_mean",
            "feature_selection_mode": "shared_medium_high_prompt_band",
            "difference_formula": "mean_selected_prompts(hider - guesser)",
            "add_intervention": "unit_direction_addition",
            "subtract_intervention": "projection_removal",
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "oracle_input_type": args.oracle_input_type,
            "oracle_token_start_idx": args.oracle_token_start_idx,
            "oracle_token_end_idx": args.oracle_token_end_idx,
            "oracle_token_eval_index": oracle_token_eval_index,
            "top_k_gap_overlap": args.top_k_gap_overlap,
            "hider_lora_path_arg": args.hider_lora_path,
            "guesser_lora_path_arg": args.guesser_lora_path,
            "seed": args.seed,
            "verbalizer_lora_paths": verbalizer_lora_paths,
            "verbalizer_prompts": verbalizer_prompts,
        },
        "selected_prompt_band": {
            str(layer_percent): {
                "summary": selected_summary_by_layer[layer_percent],
                "rows": selected_rows_by_layer[layer_percent],
            }
            for layer_percent in layer_percents
        },
        "feature_summary": {
            str(layer_percent): feature_summary_by_layer[layer_percent]
            for layer_percent in layer_percents
        },
        "source_probe_summary": source_probe_summary,
        "source_metric_summaries": source_metric_summaries,
        "cross_source_gap_summary": cross_source_gap_summary,
        "combined_prompt_rows": combined_prompt_rows,
        "cross_source_gap_rows": cross_source_gap_rows,
    }

    with open(output_dir / "taboo_axis2_target_word_difference_intervention_eval.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    feature_tensors = {
        str(layer_percent): {
            target_word: feature_vectors_by_layer[layer_percent][target_word]
            for target_word in secret_words
        }
        for layer_percent in layer_percents
    }
    torch.save(feature_tensors, output_dir / "axis2_target_word_difference_features.pt")

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
