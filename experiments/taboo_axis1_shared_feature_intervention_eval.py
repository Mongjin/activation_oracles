import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from tqdm import tqdm

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from taboo_activation_intervention_eval import get_verbalizer_lora_paths, get_verbalizer_prompts
from taboo_context_prompt_probe_eval import (
    DEFAULT_SECRET_WORDS,
    LinearProbe,
    MLPProbe,
    collect_context_prompt_last_token_activations,
    collect_dataset_last_token_activations,
    compute_target_ranks,
    load_context_prompts,
    resolve_guesser_lora_path,
    resolve_hider_lora_path,
    split_train_val_indices,
    train_binary_probe,
    train_multiclass_probe,
)


os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.view(1, -1), dim=-1).view(-1)


def default_oracle_token_eval_index(model_name: str) -> int:
    lower = model_name.lower()
    if "gemma" in lower:
        return -3
    if "qwen" in lower:
        return -7
    raise ValueError(f"Unsupported model_name for default token eval index: {model_name}")


def load_shared_group_prompts(feature_analysis_dir: Path, layer_percents: list[int]) -> dict[int, dict[str, list[str]]]:
    group_prompts_by_layer = {}
    for layer_percent in layer_percents:
        csv_path = feature_analysis_dir / f"shared_oracle_overlap_groups_layer_{layer_percent}.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        top_prompts = [row["prompt_text"] for row in rows if row["group_label"] == "TOP20_OVERLAP"]
        bottom_prompts = [row["prompt_text"] for row in rows if row["group_label"] == "BOTTOM20_OVERLAP"]
        group_prompts_by_layer[layer_percent] = {
            "top_overlap_prompts": top_prompts,
            "bottom_overlap_prompts": bottom_prompts,
        }
    return group_prompts_by_layer


def load_shared_bottom_features(
    feature_analysis_dir: Path,
    layer_percents: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    feature_pt = feature_analysis_dir / "oracle_overlap_features.pt"
    feature_data = torch.load(feature_pt, map_location="cpu")
    shared_features = {}
    feature_summary = {}

    for layer_percent in layer_percents:
        hider_bottom = feature_data["sources"]["hider"][str(layer_percent)]["bottom_overlap_mean"].float().cpu()
        guesser_bottom = feature_data["sources"]["guesser"][str(layer_percent)]["bottom_overlap_mean"].float().cpu()
        hider_bottom_unit = normalize_vector(hider_bottom)
        guesser_bottom_unit = normalize_vector(guesser_bottom)
        shared_feature = normalize_vector(hider_bottom_unit + guesser_bottom_unit)
        shared_features[layer_percent] = shared_feature
        feature_summary[layer_percent] = {
            "hider_bottom_norm": float(hider_bottom.norm().item()),
            "guesser_bottom_norm": float(guesser_bottom.norm().item()),
            "shared_feature_norm": float(shared_feature.norm().item()),
            "hider_guesser_bottom_cosine": float(
                F.cosine_similarity(hider_bottom_unit.view(1, -1), guesser_bottom_unit.view(1, -1), dim=-1).item()
            ),
        }

    return shared_features, feature_summary


def encode_context_prompt_infos(
    tokenizer,
    context_prompts: list[str],
    device: torch.device,
) -> list[dict]:
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in context_prompts]
    inputs_BL = base_experiment.encode_messages(
        tokenizer=tokenizer,
        message_dicts=batch_messages,
        add_generation_prompt=True,
        enable_thinking=False,
        device=device,
    )

    attention_mask = inputs_BL["attention_mask"]
    seq_len = int(attention_mask.shape[1])
    prompt_infos = []
    for prompt_idx, prompt_text in enumerate(context_prompts):
        real_len = int(attention_mask[prompt_idx].sum().item())
        left_pad = seq_len - real_len
        last_pos_rel = real_len - 1
        context_input_ids = inputs_BL["input_ids"][prompt_idx, left_pad:].tolist()
        prompt_infos.append(
            {
                "prompt_id": f"P{prompt_idx + 1:03d}",
                "prompt_text": prompt_text,
                "context_input_ids": context_input_ids,
                "left_pad": left_pad,
                "num_tokens": len(context_input_ids),
                "last_pos_rel": last_pos_rel,
            }
        )
    return prompt_infos


def collect_context_prompt_sequence_activations(
    model,
    tokenizer,
    context_prompts: list[str],
    layers: list[int],
    device: torch.device,
    eval_batch_size: int,
    target_seq_len: int,
) -> dict[int, torch.Tensor]:
    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}
    acts_by_layer = {layer: [] for layer in layers}

    for start in range(0, len(context_prompts), eval_batch_size):
        batch_prompts = context_prompts[start : start + eval_batch_size]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        inputs_BL = base_experiment.encode_messages(
            tokenizer=tokenizer,
            message_dicts=batch_messages,
            add_generation_prompt=True,
            enable_thinking=False,
            device=device,
        )

        with torch.no_grad():
            batch_acts = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )

        for layer in layers:
            layer_acts = batch_acts[layer].cpu().float()
            batch_seq_len = int(layer_acts.shape[1])
            if batch_seq_len > target_seq_len:
                raise ValueError(
                    f"Batch sequence length {batch_seq_len} exceeds target_seq_len {target_seq_len}"
                )
            if batch_seq_len < target_seq_len:
                pad_len = target_seq_len - batch_seq_len
                layer_acts = F.pad(layer_acts, (0, 0, pad_len, 0))
            acts_by_layer[layer].append(layer_acts)

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in acts_by_layer.items()}


def collect_source_activations(
    model_name: str,
    source_name: str,
    secret_words: list[str],
    context_prompts: list[str],
    layers: list[int],
    tokenizer,
    device: torch.device,
    num_probe_samples: int,
    context_eval_batch_size: int,
    hider_lora_path_arg: str | None,
    guesser_lora_path_arg: str | None,
    collect_sequence_context_acts: bool,
    context_target_seq_len: int,
) -> tuple[
    dict[int, list[torch.Tensor]],
    list[int],
    dict[str, dict[int, torch.Tensor]],
    dict[str, dict[int, torch.Tensor]] | None,
    int,
]:
    dataset_acts_by_layer = {layer: [] for layer in layers}
    dataset_labels = []
    context_last_acts_by_word = {}
    context_sequence_acts_by_word = {} if collect_sequence_context_acts else None
    d_model = None

    word_to_idx = {word: idx for idx, word in enumerate(secret_words)}

    for target_word in tqdm(secret_words, desc=f"Collect {source_name} activations"):
        if source_name == "hider":
            lora_path = resolve_hider_lora_path(model_name, target_word, hider_lora_path_arg)
        elif source_name == "guesser":
            lora_path = resolve_guesser_lora_path(model_name, target_word, guesser_lora_path_arg)
        else:
            raise ValueError(f"Unsupported source_name: {source_name}")

        model = PeftModel.from_pretrained(load_model(model_name, torch.bfloat16), lora_path)
        model.eval()

        if d_model is None:
            d_model = model.config.hidden_size

        dataset_acts = collect_dataset_last_token_activations(
            model=model,
            tokenizer=tokenizer,
            secret_word=target_word,
            layers=layers,
            device=device,
            num_samples=num_probe_samples,
        )
        for layer in layers:
            dataset_acts_by_layer[layer].extend(dataset_acts[layer])
        dataset_labels.extend([word_to_idx[target_word]] * num_probe_samples)

        context_last_acts_by_word[target_word] = collect_context_prompt_last_token_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=context_eval_batch_size,
        )
        if collect_sequence_context_acts:
            context_sequence_acts_by_word[target_word] = collect_context_prompt_sequence_activations(
                model=model,
                tokenizer=tokenizer,
                context_prompts=context_prompts,
                layers=layers,
                device=device,
                eval_batch_size=context_eval_batch_size,
                target_seq_len=context_target_seq_len,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return dataset_acts_by_layer, dataset_labels, context_last_acts_by_word, context_sequence_acts_by_word, d_model

def train_source_probes(
    dataset_acts_by_layer: dict[int, list[torch.Tensor]],
    dataset_labels: list[int],
    secret_words: list[str],
    layers: list[int],
    d_model: int,
    device: torch.device,
    probe_modes: list[str],
    lr: float,
    epochs: int,
    mlp_hidden_dim: int,
    val_fraction: float,
    seed: int,
) -> dict[int, dict]:
    labels = torch.tensor(dataset_labels, dtype=torch.long)
    word_to_idx = {word: idx for idx, word in enumerate(secret_words)}
    probes_by_layer = {}

    for layer in layers:
        acts = torch.stack(dataset_acts_by_layer[layer])
        train_indices, val_indices = split_train_val_indices(labels, val_fraction, seed)
        train_x = acts[train_indices]
        train_y = labels[train_indices]
        val_x = acts[val_indices]
        val_y = labels[val_indices]

        layer_probe_info = {
            "linear_probe": None,
            "mlp_probe": None,
            "linear_probe_val_acc": None,
            "mlp_probe_val_acc": None,
            "binary_linear_probes": {},
            "binary_mlp_probes": {},
            "binary_probe_val_metrics": {},
        }

        if "multiclass" in probe_modes:
            linear_probe, linear_val_acc = train_multiclass_probe(
                model=LinearProbe(d_model, len(secret_words)).to(device),
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                device=device,
                lr=lr,
                epochs=epochs,
            )
            mlp_probe, mlp_val_acc = train_multiclass_probe(
                model=MLPProbe(d_model, mlp_hidden_dim, len(secret_words)).to(device),
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                device=device,
                lr=lr,
                epochs=epochs,
            )
            layer_probe_info["linear_probe"] = linear_probe.cpu().eval()
            layer_probe_info["mlp_probe"] = mlp_probe.cpu().eval()
            layer_probe_info["linear_probe_val_acc"] = linear_val_acc
            layer_probe_info["mlp_probe_val_acc"] = mlp_val_acc

        if "binary" in probe_modes:
            for target_word in tqdm(secret_words, desc=f"Binary probes | layer={layer}", leave=False):
                target_idx = word_to_idx[target_word]
                train_y_bin = (train_y == target_idx).long()
                val_y_bin = (val_y == target_idx).long()

                binary_linear_probe, binary_linear_metrics = train_binary_probe(
                    model=LinearProbe(d_model, 1).to(device),
                    train_x=train_x,
                    train_y=train_y_bin,
                    val_x=val_x,
                    val_y=val_y_bin,
                    device=device,
                    lr=lr,
                    epochs=epochs,
                )
                binary_mlp_probe, binary_mlp_metrics = train_binary_probe(
                    model=MLPProbe(d_model, mlp_hidden_dim, 1).to(device),
                    train_x=train_x,
                    train_y=train_y_bin,
                    val_x=val_x,
                    val_y=val_y_bin,
                    device=device,
                    lr=lr,
                    epochs=epochs,
                )

                layer_probe_info["binary_linear_probes"][target_word] = binary_linear_probe.cpu().eval()
                layer_probe_info["binary_mlp_probes"][target_word] = binary_mlp_probe.cpu().eval()
                layer_probe_info["binary_probe_val_metrics"][target_word] = {
                    "linear": binary_linear_metrics,
                    "mlp": binary_mlp_metrics,
                }

        probes_by_layer[layer] = layer_probe_info

    return probes_by_layer


def get_intervention_modes(intervention_scale: float) -> dict[str, float]:
    return {
        "baseline": 0.0,
        "add": intervention_scale,
        "subtract": -intervention_scale,
    }


def apply_shared_feature(
    acts: torch.Tensor,
    feature_D: torch.Tensor,
    feature_scale: float,
) -> torch.Tensor:
    if feature_scale == 0.0:
        return acts.clone()
    feature_shape = [1] * (acts.ndim - 1) + [-1]
    return acts + feature_scale * feature_D.view(*feature_shape)


def compute_probe_prompt_rows(
    source_name: str,
    context_acts_by_word: dict[str, dict[int, torch.Tensor]],
    probes_by_layer: dict[int, dict],
    shared_features: dict[int, torch.Tensor],
    intervention_modes: dict[str, float],
    secret_words: list[str],
    context_prompts: list[str],
    layers: list[int],
    probe_modes: list[str],
) -> dict[int, dict[str, list[dict]]]:
    word_to_idx = {word: idx for idx, word in enumerate(secret_words)}
    prompt_rows_by_layer_mode = {}

    for layer in layers:
        layer_probe_info = probes_by_layer[layer]
        prompt_rows_by_layer_mode[layer] = {}

        for mode_name, feature_scale in intervention_modes.items():
            prompt_records = {prompt_idx: [] for prompt_idx in range(len(context_prompts))}

            for target_word in secret_words:
                target_idx = word_to_idx[target_word]
                eval_x = apply_shared_feature(
                    acts=context_acts_by_word[target_word][layer].float().cpu(),
                    feature_D=shared_features[layer].float().cpu(),
                    feature_scale=feature_scale,
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
                        linear_target_ranks = compute_target_ranks(linear_logits, target_idx)

                        mlp_logits = layer_probe_info["mlp_probe"](eval_x)
                        mlp_probs = torch.softmax(mlp_logits, dim=1)
                        mlp_preds = torch.argmax(mlp_logits, dim=1)
                        mlp_target_probs = mlp_probs[:, target_idx]
                        mlp_target_ranks = compute_target_ranks(mlp_logits, target_idx)

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

            prompt_rows_by_layer_mode[layer][mode_name] = prompt_rows

    return prompt_rows_by_layer_mode

def create_last_token_verbalizer_inputs(
    prompt_infos: list[dict],
    prompt_acts_PD: torch.Tensor,
    target_word: str,
    verbalizer_prompts: list[str],
    layer_index: int,
    tokenizer,
    mode_name: str,
    source_name: str,
    layer_percent: int,
    oracle_name: str,
) -> list[TrainingDataPoint]:
    verbalizer_inputs = []
    feature_idx = 0

    for prompt_info, act_D in zip(prompt_infos, prompt_acts_PD, strict=True):
        for verbalizer_prompt in verbalizer_prompts:
            meta_info = {
                "source": source_name,
                "mode": mode_name,
                "oracle_name": oracle_name,
                "layer_percent": layer_percent,
                "prompt_id": prompt_info["prompt_id"],
                "prompt_text": prompt_info["prompt_text"],
                "ground_truth": target_word,
                "verbalizer_prompt": verbalizer_prompt,
            }
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=layer_index,
                num_positions=1,
                tokenizer=tokenizer,
                acts_BD=act_D.view(1, -1),
                feature_idx=feature_idx,
                context_input_ids=prompt_info["context_input_ids"],
                context_positions=[prompt_info["last_pos_rel"]],
                ds_label="N/A",
                meta_info=meta_info,
            )
            verbalizer_inputs.append(dp)
            feature_idx += 1

    return verbalizer_inputs


def create_token_verbalizer_inputs(
    prompt_infos: list[dict],
    prompt_acts_PLD: torch.Tensor,
    target_word: str,
    verbalizer_prompts: list[str],
    layer_index: int,
    tokenizer,
    mode_name: str,
    source_name: str,
    layer_percent: int,
    oracle_name: str,
    token_start_idx: int,
    token_end_idx: int,
) -> list[TrainingDataPoint]:
    verbalizer_inputs = []
    feature_idx = 0

    for prompt_info, acts_LD in zip(prompt_infos, prompt_acts_PLD, strict=True):
        num_tokens = prompt_info["num_tokens"]
        if token_start_idx < 0:
            token_start = num_tokens + token_start_idx
            token_end = num_tokens + token_end_idx
        else:
            token_start = token_start_idx
            token_end = token_end_idx

        for verbalizer_prompt in verbalizer_prompts:
            for token_index in range(token_start, token_end):
                meta_info = {
                    "source": source_name,
                    "mode": mode_name,
                    "oracle_name": oracle_name,
                    "layer_percent": layer_percent,
                    "prompt_id": prompt_info["prompt_id"],
                    "prompt_text": prompt_info["prompt_text"],
                    "ground_truth": target_word,
                    "verbalizer_prompt": verbalizer_prompt,
                    "token_index": token_index,
                    "num_tokens": num_tokens,
                }
                abs_token_index = prompt_info["left_pad"] + token_index
                dp = create_training_datapoint(
                    datapoint_type="N/A",
                    prompt=verbalizer_prompt,
                    target_response="N/A",
                    layer=layer_index,
                    num_positions=1,
                    tokenizer=tokenizer,
                    acts_BD=acts_LD[abs_token_index].view(1, -1),
                    feature_idx=feature_idx,
                    context_input_ids=prompt_info["context_input_ids"],
                    context_positions=[token_index],
                    ds_label="N/A",
                    meta_info=meta_info,
                )
                verbalizer_inputs.append(dp)
                feature_idx += 1

    return verbalizer_inputs


def compute_prompt_oracle_rows(
    model_name: str,
    oracle_model,
    tokenizer,
    source_name: str,
    context_last_acts_by_word: dict[str, dict[int, torch.Tensor]],
    context_sequence_acts_by_word: dict[str, dict[int, torch.Tensor]] | None,
    shared_features: dict[int, torch.Tensor],
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
            prompt_correct_counts = defaultdict(int)
            prompt_total_counts = defaultdict(int)

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
                        prompt_acts = apply_shared_feature(
                            acts=context_last_acts_by_word[target_word][layer_index].float().cpu(),
                            feature_D=shared_features[layer_percent].float().cpu(),
                            feature_scale=feature_scale,
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
                        prompt_acts = apply_shared_feature(
                            acts=context_sequence_acts_by_word[target_word][layer_index].float().cpu(),
                            feature_D=shared_features[layer_percent].float().cpu(),
                            feature_scale=feature_scale,
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


def add_baseline_deltas(
    prompt_rows_by_layer_mode: dict[int, dict[str, list[dict]]],
) -> None:
    for _, rows_by_mode in prompt_rows_by_layer_mode.items():
        baseline_map = {
            row["prompt_text"]: row
            for row in rows_by_mode["baseline"]
        }
        metric_names = [
            key for key in rows_by_mode["baseline"][0].keys()
            if key not in ("prompt_id", "prompt_text")
        ]

        for mode_name, rows in rows_by_mode.items():
            for row in rows:
                baseline_row = baseline_map[row["prompt_text"]]
                for metric_name in metric_names:
                    delta_key = f"{metric_name}_delta_from_baseline"
                    row[delta_key] = row[metric_name] - baseline_row[metric_name]


def collect_prompt_metric_names(prompt_rows: list[dict]) -> list[str]:
    return [
        key for key in prompt_rows[0].keys()
        if key not in ("prompt_id", "prompt_text")
        and not key.endswith("_delta_from_baseline")
    ]


def compute_pearson(values_a: torch.Tensor, values_b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([values_a, values_b]))[0, 1].item())


def compute_descending_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=torch.float32)
    return ranks


def compute_spearman(values_a: torch.Tensor, values_b: torch.Tensor) -> float:
    return compute_pearson(compute_descending_ranks(values_a), compute_descending_ranks(values_b))


def summarize_prompt_metrics(
    prompt_rows_by_layer_mode: dict[int, dict[str, list[dict]]],
) -> dict:
    summary = {}
    for layer_key, rows_by_mode in prompt_rows_by_layer_mode.items():
        summary[str(layer_key)] = {}
        baseline_rows = rows_by_mode["baseline"]
        baseline_map = {row["prompt_text"]: row for row in baseline_rows}
        metric_names = collect_prompt_metric_names(baseline_rows)

        for mode_name, rows in rows_by_mode.items():
            layer_mode_summary = {}
            for metric_name in metric_names:
                values = torch.tensor([row[metric_name] for row in rows], dtype=torch.float32)
                baseline_values = torch.tensor(
                    [baseline_map[row["prompt_text"]][metric_name] for row in rows],
                    dtype=torch.float32,
                )
                deltas = values - baseline_values
                layer_mode_summary[metric_name] = {
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                    "mean_delta_from_baseline": float(deltas.mean().item()),
                    "mean_abs_delta_from_baseline": float(deltas.abs().mean().item()),
                    "pearson_vs_baseline": float(
                        torch.corrcoef(torch.stack([values, baseline_values]))[0, 1].item()
                    ),
                }
            summary[str(layer_key)][mode_name] = layer_mode_summary
    return summary

def compute_top_k_overlap(prompt_texts: list[str], values_a: torch.Tensor, values_b: torch.Tensor, top_k: int) -> dict:
    order_a = torch.argsort(values_a, descending=True)[:top_k]
    order_b = torch.argsort(values_b, descending=True)[:top_k]
    top_prompts_a = {prompt_texts[idx] for idx in order_a.tolist()}
    top_prompts_b = {prompt_texts[idx] for idx in order_b.tolist()}
    overlap = sorted(top_prompts_a & top_prompts_b)
    union = top_prompts_a | top_prompts_b
    return {
        "count": len(overlap),
        "jaccard": len(overlap) / len(union),
        "overlap_prompts": overlap,
    }


def build_cross_source_gap_rows(
    hider_rows: list[dict],
    guesser_rows: list[dict],
) -> list[dict]:
    hider_map = {row["prompt_text"]: row for row in hider_rows}
    guesser_map = {row["prompt_text"]: row for row in guesser_rows}
    common_prompts = [row["prompt_text"] for row in hider_rows]
    if set(common_prompts) != set(guesser_map.keys()):
        raise ValueError("Hider and guesser prompt sets do not match")

    metric_names = collect_prompt_metric_names(hider_rows)
    rows = []
    for prompt_text in common_prompts:
        hider_row = hider_map[prompt_text]
        guesser_row = guesser_map[prompt_text]
        row = {
            "prompt_id": hider_row["prompt_id"],
            "prompt_text": prompt_text,
            "hider_oracle_accuracy": hider_row["oracle_accuracy"],
            "guesser_oracle_accuracy": guesser_row["oracle_accuracy"],
            "oracle_gap_signed": guesser_row["oracle_accuracy"] - hider_row["oracle_accuracy"],
            "oracle_gap_abs": abs(guesser_row["oracle_accuracy"] - hider_row["oracle_accuracy"]),
        }
        for metric_name in metric_names:
            row[f"hider_{metric_name}"] = hider_row[metric_name]
            row[f"guesser_{metric_name}"] = guesser_row[metric_name]
        rows.append(row)
    return rows


def summarize_cross_source_gap_rows(
    baseline_rows: list[dict],
    current_rows: list[dict],
    top_k: int,
) -> dict:
    baseline_map = {row["prompt_text"]: row for row in baseline_rows}
    current_map = {row["prompt_text"]: row for row in current_rows}
    prompt_texts = [row["prompt_text"] for row in baseline_rows]

    baseline_gap_signed = torch.tensor(
        [baseline_map[prompt_text]["oracle_gap_signed"] for prompt_text in prompt_texts],
        dtype=torch.float32,
    )
    current_gap_signed = torch.tensor(
        [current_map[prompt_text]["oracle_gap_signed"] for prompt_text in prompt_texts],
        dtype=torch.float32,
    )
    baseline_gap_abs = baseline_gap_signed.abs()
    current_gap_abs = current_gap_signed.abs()

    return {
        "pearson_signed_gap_vs_baseline": compute_pearson(current_gap_signed, baseline_gap_signed),
        "spearman_signed_gap_vs_baseline": compute_spearman(current_gap_signed, baseline_gap_signed),
        "pearson_abs_gap_vs_baseline": compute_pearson(current_gap_abs, baseline_gap_abs),
        "spearman_abs_gap_vs_baseline": compute_spearman(current_gap_abs, baseline_gap_abs),
        "mean_signed_gap": float(current_gap_signed.mean().item()),
        "mean_abs_gap": float(current_gap_abs.mean().item()),
        "mean_signed_gap_delta_from_baseline": float((current_gap_signed - baseline_gap_signed).mean().item()),
        "mean_abs_gap_delta_from_baseline": float((current_gap_abs - baseline_gap_abs).mean().item()),
        "top_k_abs_gap_overlap": compute_top_k_overlap(prompt_texts, baseline_gap_abs, current_gap_abs, top_k),
    }


def save_prompt_summary_csv(
    output_path: Path,
    prompt_rows: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(prompt_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prompt_rows:
            writer.writerow(row)


def save_cross_source_gap_csv(
    output_path: Path,
    rows: list[dict],
    baseline_rows: list[dict],
) -> None:
    baseline_map = {row["prompt_text"]: row for row in baseline_rows}
    augmented_rows = []
    for row in rows:
        baseline_row = baseline_map[row["prompt_text"]]
        augmented_row = dict(row)
        augmented_row["oracle_gap_signed_delta_from_baseline"] = (
            row["oracle_gap_signed"] - baseline_row["oracle_gap_signed"]
        )
        augmented_row["oracle_gap_abs_delta_from_baseline"] = row["oracle_gap_abs"] - baseline_row["oracle_gap_abs"]
        augmented_rows.append(augmented_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(augmented_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in augmented_rows:
            writer.writerow(row)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--feature_analysis_dir", type=str, required=True)
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--prompt_type", type=str, default="all_direct", choices=["all_direct", "all_standard"])
    parser.add_argument("--dataset_type", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--lang_type", type=str, default=None)
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

    feature_analysis_dir = Path(args.feature_analysis_dir)
    shared_group_prompts = load_shared_group_prompts(feature_analysis_dir, layer_percents)
    shared_features, shared_feature_summary = load_shared_bottom_features(feature_analysis_dir, layer_percents)

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

    prompt_probe_rows = {}
    for source_name in ["hider", "guesser"]:
        prompt_probe_rows[source_name] = compute_probe_prompt_rows(
            source_name=source_name,
            context_acts_by_word=source_context_last_acts[source_name],
            probes_by_layer=source_probes[source_name],
            shared_features={layer: shared_features[layer_percent] for layer_percent, layer in zip(layer_percents, layers)},
            intervention_modes=intervention_modes,
            secret_words=secret_words,
            context_prompts=context_prompts,
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
            shared_features=shared_features,
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
        for layer_percent, layer in zip(layer_percents, layers):
            combined_prompt_rows[source_name][str(layer_percent)] = {}
            oracle_rows_by_mode = prompt_oracle_rows[source_name][layer_percent]
            probe_rows_by_mode = prompt_probe_rows[source_name][layer]

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
        f"{model_name_short}_axis1_shared_bottom_intervention_{args.prompt_type}{lang_suffix}_{args.dataset_type}_{args.oracle_input_type}_{args.intervention_scale}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name in ["hider", "guesser"]:
        for layer_percent in layer_percents:
            for mode_name in intervention_modes:
                save_prompt_summary_csv(
                    output_path=output_dir / f"axis1_prompt_summary_{source_name}_layer_{layer_percent}_{mode_name}.csv",
                    prompt_rows=combined_prompt_rows[source_name][str(layer_percent)][mode_name],
                )

    for layer_percent in layer_percents:
        baseline_rows = cross_source_gap_rows[str(layer_percent)]["baseline"]
        for mode_name in intervention_modes:
            save_cross_source_gap_csv(
                output_path=output_dir / f"axis1_cross_source_gap_layer_{layer_percent}_{mode_name}.csv",
                rows=cross_source_gap_rows[str(layer_percent)][mode_name],
                baseline_rows=baseline_rows,
            )

    final_results = {
        "config": {
            "model_name": args.model_name,
            "feature_analysis_dir": str(feature_analysis_dir),
            "prompt_type": args.prompt_type,
            "dataset_type": args.dataset_type,
            "lang_type": args.lang_type,
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
        "shared_group_prompts": shared_group_prompts,
        "shared_feature_summary": {str(layer_percent): shared_feature_summary[layer_percent] for layer_percent in layer_percents},
        "probe_training_summary": source_probe_summary,
        "source_metric_summaries": source_metric_summaries,
        "cross_source_gap_summary": cross_source_gap_summary,
        "sources": combined_prompt_rows,
        "cross_source_gap_rows": cross_source_gap_rows,
    }

    output_json = output_dir / "taboo_axis1_shared_feature_intervention_eval.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"Saved Axis 1 intervention results to {output_json}")


if __name__ == "__main__":
    main()

