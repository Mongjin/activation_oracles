import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig

from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from taboo_activation_intervention_eval import get_verbalizer_lora_paths, get_verbalizer_prompts
from taboo_axis1_shared_feature_intervention_eval import (
    add_baseline_deltas,
    collect_context_prompt_sequence_activations,
    build_cross_source_gap_rows,
    collect_source_activations,
    compute_probe_prompt_rows,
    compute_prompt_oracle_rows,
    default_oracle_token_eval_index,
    encode_context_prompt_infos,
    load_shared_group_prompts_from_alignment,
    normalize_vector,
    save_cross_source_gap_csv,
    save_prompt_summary_csv,
    select_prompt_eval_acts,
    summarize_cross_source_gap_rows,
    summarize_prompt_metrics,
    train_source_probes,
)
from taboo_context_prompt_probe_eval import DEFAULT_SECRET_WORDS, collect_context_prompt_last_token_activations, load_context_prompts


os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def collect_vanilla_context_activations(
    model_name: str,
    context_prompts: list[str],
    layers: list[int],
    tokenizer,
    device: torch.device,
    context_eval_batch_size: int,
    collect_sequence_context_acts: bool,
    context_target_seq_len: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    model = load_model(model_name, torch.bfloat16)
    model.eval()

    vanilla_context_last_acts = collect_context_prompt_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        context_prompts=context_prompts,
        layers=layers,
        device=device,
        eval_batch_size=context_eval_batch_size,
    )
    if collect_sequence_context_acts:
        vanilla_context_sequence_acts = collect_context_prompt_sequence_activations(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=context_eval_batch_size,
            target_seq_len=context_target_seq_len,
        )
    else:
        vanilla_context_sequence_acts = None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vanilla_context_last_acts, vanilla_context_sequence_acts


def compute_vanilla_bottom_top_features(
    prompt_infos: list[dict],
    vanilla_context_last_acts: dict[int, torch.Tensor],
    vanilla_context_sequence_acts: dict[int, torch.Tensor] | None,
    layer_percents: list[int],
    layers: list[int],
    group_prompts_by_layer: dict[int, dict[str, list[str]]],
    oracle_input_type: str,
    oracle_token_eval_index: int,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    prompt_index_by_text = {prompt_info["prompt_text"]: prompt_idx for prompt_idx, prompt_info in enumerate(prompt_infos)}
    shared_features = {}
    feature_summary = {}

    for layer_percent, layer in zip(layer_percents, layers):
        top_prompt_indices = [
            prompt_index_by_text[prompt_text]
            for prompt_text in group_prompts_by_layer[layer_percent]["top_overlap_prompts"]
        ]
        bottom_prompt_indices = [
            prompt_index_by_text[prompt_text]
            for prompt_text in group_prompts_by_layer[layer_percent]["bottom_overlap_prompts"]
        ]

        if oracle_input_type == "last_token":
            prompt_eval_acts = vanilla_context_last_acts[layer].float().cpu()
        elif oracle_input_type == "tokens":
            if vanilla_context_sequence_acts is None:
                raise ValueError("vanilla_context_sequence_acts must be provided for oracle_input_type='tokens'")
            prompt_eval_acts = select_prompt_eval_acts(
                prompt_infos=prompt_infos,
                acts=vanilla_context_sequence_acts[layer].float().cpu(),
                oracle_input_type=oracle_input_type,
                oracle_token_eval_index=oracle_token_eval_index,
            )
        else:
            raise ValueError(f"Unsupported oracle_input_type: {oracle_input_type}")

        top_mean = prompt_eval_acts[top_prompt_indices].mean(dim=0)
        bottom_mean = prompt_eval_acts[bottom_prompt_indices].mean(dim=0)
        feature_raw = bottom_mean - top_mean
        feature_unit = normalize_vector(feature_raw)

        top_cosines = torch.matmul(F.normalize(prompt_eval_acts[top_prompt_indices], dim=-1), feature_unit)
        bottom_cosines = torch.matmul(F.normalize(prompt_eval_acts[bottom_prompt_indices], dim=-1), feature_unit)

        shared_features[layer] = feature_unit
        feature_summary[layer_percent] = {
            "feature_source": "vanilla_model_bottom_minus_top",
            "feature_formula": "mean(bottom_prompts) - mean(top_prompts)",
            "num_top_overlap_prompts": len(top_prompt_indices),
            "num_bottom_overlap_prompts": len(bottom_prompt_indices),
            "feature_raw_norm": float(feature_raw.norm().item()),
            "feature_unit_norm": float(feature_unit.norm().item()),
            "mean_top_cosine_to_feature": float(top_cosines.mean().item()),
            "mean_bottom_cosine_to_feature": float(bottom_cosines.mean().item()),
            "oracle_input_type_for_feature": oracle_input_type,
            "oracle_token_eval_index_for_feature": oracle_token_eval_index if oracle_input_type == "tokens" else None,
        }

    return shared_features, feature_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--feature_analysis_dir", type=str, required=True)
    parser.add_argument("--secret_words", type=str, default=",".join(DEFAULT_SECRET_WORDS))
    parser.add_argument("--prompt_type", type=str, default="all_direct", choices=["all_direct", "all_standard"])
    parser.add_argument("--dataset_type", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--lang_type", type=str, default=None)
    parser.add_argument("--feature_group_size", type=int, default=10)
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
    intervention_modes = {
        "baseline": 0.0,
        "add": args.intervention_scale,
        "subtract": args.intervention_scale,
    }
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
    shared_group_prompts = load_shared_group_prompts_from_alignment(
        feature_analysis_dir=feature_analysis_dir,
        layer_percents=layer_percents,
        feature_group_size=args.feature_group_size,
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

    vanilla_context_last_acts, vanilla_context_sequence_acts = collect_vanilla_context_activations(
        model_name=args.model_name,
        context_prompts=context_prompts,
        layers=layers,
        tokenizer=tokenizer,
        device=device,
        context_eval_batch_size=args.context_eval_batch_size,
        collect_sequence_context_acts=args.oracle_input_type == "tokens",
        context_target_seq_len=context_target_seq_len,
    )

    shared_features, feature_summary = compute_vanilla_bottom_top_features(
        prompt_infos=prompt_infos,
        vanilla_context_last_acts=vanilla_context_last_acts,
        vanilla_context_sequence_acts=vanilla_context_sequence_acts,
        layer_percents=layer_percents,
        layers=layers,
        group_prompts_by_layer=shared_group_prompts,
        oracle_input_type=args.oracle_input_type,
        oracle_token_eval_index=oracle_token_eval_index,
    )

    prompt_probe_rows = {}
    for source_name in ["hider", "guesser"]:
        prompt_probe_rows[source_name] = compute_probe_prompt_rows(
            source_name=source_name,
            context_acts_by_word=source_context_last_acts[source_name],
            probes_by_layer=source_probes[source_name],
            shared_features=shared_features,
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
            shared_features={layer_percent: shared_features[layer] for layer_percent, layer in zip(layer_percents, layers)},
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
        (
            f"{model_name_short}_vanilla_model_low_high_contrast_intervention_"
            f"{args.prompt_type}{lang_suffix}_{args.dataset_type}_{args.oracle_input_type}_{args.intervention_scale}"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name in ["hider", "guesser"]:
        for layer_percent in layer_percents:
            for mode_name in intervention_modes:
                save_prompt_summary_csv(
                    output_path=output_dir / f"vanilla_contrast_prompt_summary_{source_name}_layer_{layer_percent}_{mode_name}.csv",
                    prompt_rows=combined_prompt_rows[source_name][str(layer_percent)][mode_name],
                )

    for layer_percent in layer_percents:
        baseline_rows = cross_source_gap_rows[str(layer_percent)]["baseline"]
        for mode_name in intervention_modes:
            save_cross_source_gap_csv(
                output_path=output_dir / f"vanilla_contrast_cross_source_gap_layer_{layer_percent}_{mode_name}.csv",
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
            "feature_group_size": args.feature_group_size,
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
            "feature_source": "vanilla_model_bottom_minus_top",
            "feature_group_origin": "guesser_axis1_prompt_groups",
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
        "shared_group_prompts": shared_group_prompts,
        "feature_summary": {
            str(layer_percent): feature_summary[layer_percent]
            for layer_percent in layer_percents
        },
        "source_probe_summary": source_probe_summary,
        "source_metric_summaries": source_metric_summaries,
        "cross_source_gap_summary": cross_source_gap_summary,
        "combined_prompt_rows": combined_prompt_rows,
        "cross_source_gap_rows": cross_source_gap_rows,
    }

    with open(output_dir / "taboo_vanilla_model_low_high_contrast_intervention_eval.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
