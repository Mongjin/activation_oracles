import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CUSTOM_LABELS = {
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
    "base_model": "Base Model",
}


LAYER_ORDER = [25, 50, 75]


def token_slice_idx(model_name: str) -> int:
    if "gemma" in model_name.lower():
        return -3
    if "qwen" in model_name.lower():
        return -7
    raise ValueError(f"Unsupported model_name for token slice index: {model_name}")


def calculate_accuracy(record: dict, model_name: str, sequence: bool) -> float:
    ground_truth = record["ground_truth"].lower()

    if sequence:
        responses = record["full_sequence_responses"]
    else:
        idx = token_slice_idx(model_name)
        responses = record["token_responses"][idx : idx + 1]

    num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
    total = len(responses)
    return num_correct / total


def calculate_ci_margin(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    std_err = np.std(values, ddof=1) / np.sqrt(n)
    return float(1.96 * std_err)


def sanitize_lora_key(verbalizer_lora_path: str | None) -> str:
    if verbalizer_lora_path is None:
        return "base_model"
    return verbalizer_lora_path.split("/")[-1]


def annotate_bar_values(ax, bar_container, values: list[float]) -> None:
    labels = [f"{value:.3f}" for value in values]
    ax.bar_label(bar_container, labels=labels, padding=3, fontsize=8, rotation=90)


def load_layered_results(
    json_dir: Path,
    sequence: bool,
    required_verbalizer_prompt: str | None,
    required_act_key: str | None,
):
    records_by_lora_layer = defaultdict(lambda: defaultdict(list))

    json_files = sorted(json_dir.glob("*layer_*.json"))
    if len(json_files) == 0:
        raise ValueError(f"No layer-specific files found in {json_dir}. Expected '*layer_*.json'")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data["config"]["model_name"]
        layer_percent = int(data["config"]["selected_layer_percent"])
        lora_key = sanitize_lora_key(data.get("verbalizer_lora_path"))

        for record in data["results"]:
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            if required_act_key and record["act_key"] != required_act_key:
                continue
            acc = calculate_accuracy(record, model_name, sequence)
            records_by_lora_layer[lora_key][layer_percent].append(acc)

    return records_by_lora_layer


def load_layered_results_by_act_key(
    json_dir: Path,
    sequence: bool,
    required_verbalizer_prompt: str | None,
    compare_act_keys: list[str],
):
    records_by_lora_layer_act = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    json_files = sorted(json_dir.glob("*layer_*.json"))
    if len(json_files) == 0:
        raise ValueError(f"No layer-specific files found in {json_dir}. Expected '*layer_*.json'")

    compare_set = set(compare_act_keys)

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data["config"]["model_name"]
        layer_percent = int(data["config"]["selected_layer_percent"])
        lora_key = sanitize_lora_key(data.get("verbalizer_lora_path"))

        for record in data["results"]:
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            act_key = record["act_key"]
            if act_key not in compare_set:
                continue

            acc = calculate_accuracy(record, model_name, sequence)
            records_by_lora_layer_act[lora_key][layer_percent][act_key].append(acc)

    return records_by_lora_layer_act


def plot_layer_comparison(records_by_lora_layer, output_path: Path, sequence: bool, data_dir_label: str):
    lora_keys = sorted(records_by_lora_layer.keys())

    for lora in lora_keys:
        missing = [lp for lp in LAYER_ORDER if lp not in records_by_lora_layer[lora]]
        if missing:
            raise ValueError(f"Missing layers for {lora}: {missing}")

    x = np.arange(len(lora_keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c"}
    offsets = {25: -width, 50: 0.0, 75: width}

    for lp in LAYER_ORDER:
        means = []
        cis = []
        for lora in lora_keys:
            vals = records_by_lora_layer[lora][lp]
            means.append(float(np.mean(vals)))
            cis.append(calculate_ci_margin(vals))

        bars = ax.bar(
            x + offsets[lp],
            means,
            width,
            yerr=cis,
            capsize=4,
            color=colors[lp],
            label=f"Layer {lp}%",
            alpha=0.9,
        )
        annotate_bar_values(ax, bars, means)

    pretty_labels = [CUSTOM_LABELS.get(k, k) for k in lora_keys]

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Average Accuracy")
    ax.set_xlabel("Verbalizer (Oracle) Model")

    token_or_seq = "Sequence" if sequence else "Token"
    ax.set_title(f"Taboo Layer Comparison (25/50/75) | {token_or_seq}-Level | {data_dir_label}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_activation_source_comparison(
    records_by_lora_layer_act,
    output_path: Path,
    sequence: bool,
    data_dir_label: str,
    compare_act_keys: list[str],
):
    act_a, act_b = compare_act_keys
    lora_keys = sorted(records_by_lora_layer_act.keys())

    for lora in lora_keys:
        for lp in LAYER_ORDER:
            if lp not in records_by_lora_layer_act[lora]:
                raise ValueError(f"Missing layer {lp}% for {lora}")
            missing_act_keys = [k for k in compare_act_keys if k not in records_by_lora_layer_act[lora][lp]]
            if missing_act_keys:
                raise ValueError(f"Missing act_key for {lora}, layer {lp}%: {missing_act_keys}")

    x = np.arange(len(lora_keys))
    width = 0.35

    fig, axes = plt.subplots(1, len(LAYER_ORDER), figsize=(7 * len(LAYER_ORDER), 7), sharey=True)
    if len(LAYER_ORDER) == 1:
        axes = [axes]

    colors = {act_a: "#1f77b4", act_b: "#ff7f0e"}

    for ax, lp in zip(axes, LAYER_ORDER):
        means_a = []
        cis_a = []
        means_b = []
        cis_b = []

        for lora in lora_keys:
            vals_a = records_by_lora_layer_act[lora][lp][act_a]
            vals_b = records_by_lora_layer_act[lora][lp][act_b]
            means_a.append(float(np.mean(vals_a)))
            cis_a.append(calculate_ci_margin(vals_a))
            means_b.append(float(np.mean(vals_b)))
            cis_b.append(calculate_ci_margin(vals_b))

        bars_a = ax.bar(
            x - width / 2,
            means_a,
            width,
            yerr=cis_a,
            capsize=4,
            color=colors[act_a],
            label=f"Activation: {act_a}",
            alpha=0.9,
        )
        bars_b = ax.bar(
            x + width / 2,
            means_b,
            width,
            yerr=cis_b,
            capsize=4,
            color=colors[act_b],
            label=f"Activation: {act_b}",
            alpha=0.9,
        )
        annotate_bar_values(ax, bars_a, means_a)
        annotate_bar_values(ax, bars_b, means_b)

        pretty_labels = [CUSTOM_LABELS.get(k, k) for k in lora_keys]
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Layer {lp}%")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Average Accuracy")
    for ax in axes:
        ax.set_xlabel("Verbalizer (Oracle) Model")

    token_or_seq = "Sequence" if sequence else "Token"
    fig.suptitle(
        f"Taboo Activation Source Comparison ({act_a} vs {act_b}) | {token_or_seq}-Level | {data_dir_label}",
        y=1.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        help="Directory containing taboo layer result json files",
        default="../taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test",
    )
    parser.add_argument("--output_dir", type=str, default="./images/taboo")
    parser.add_argument("--sequence", action="store_true", help="Use full_sequence_responses instead of token_responses")
    parser.add_argument("--required_verbalizer_prompt", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["oracle_layer", "activation_source"],
        default="oracle_layer",
        help="oracle_layer: compare 25/50/75 layers for one act_key. activation_source: compare lora vs orig per layer.",
    )
    parser.add_argument(
        "--act_key",
        type=str,
        default="lora",
        help="Used in mode=oracle_layer. Filters results by act_key (default: lora).",
    )
    parser.add_argument(
        "--compare_act_keys",
        type=str,
        nargs=2,
        default=["lora", "orig"],
        help="Used in mode=activation_source. Exactly two act_keys to compare (default: lora orig).",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise ValueError(f"json_dir does not exist: {json_dir}")

    data_dir_label = json_dir.name
    token_or_seq = "sequence" if args.sequence else "token"

    if args.mode == "oracle_layer":
        records_by_lora_layer = load_layered_results(
            json_dir=json_dir,
            sequence=args.sequence,
            required_verbalizer_prompt=args.required_verbalizer_prompt,
            required_act_key=args.act_key,
        )
        output_name = f"taboo_results_layer_comparison_{data_dir_label}_{token_or_seq}_{args.act_key}.png"
        output_path = Path(args.output_dir) / output_name
        plot_layer_comparison(records_by_lora_layer, output_path, args.sequence, data_dir_label)
        print(f"Saved: {output_path}")
        return

    records_by_lora_layer_act = load_layered_results_by_act_key(
        json_dir=json_dir,
        sequence=args.sequence,
        required_verbalizer_prompt=args.required_verbalizer_prompt,
        compare_act_keys=args.compare_act_keys,
    )

    act_a, act_b = args.compare_act_keys
    output_name = f"taboo_results_activation_source_comparison_{data_dir_label}_{token_or_seq}_{act_a}_vs_{act_b}.png"
    output_path = Path(args.output_dir) / output_name

    plot_activation_source_comparison(
        records_by_lora_layer_act=records_by_lora_layer_act,
        output_path=output_path,
        sequence=args.sequence,
        data_dir_label=data_dir_label,
        compare_act_keys=args.compare_act_keys,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
