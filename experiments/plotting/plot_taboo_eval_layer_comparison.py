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


def load_layered_results(json_dir: Path, sequence: bool, required_verbalizer_prompt: str | None):
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
            acc = calculate_accuracy(record, model_name, sequence)
            records_by_lora_layer[lora_key][layer_percent].append(acc)

    return records_by_lora_layer


def plot_layer_comparison(records_by_lora_layer, output_path: Path, sequence: bool, data_dir_label: str):
    layer_order = [25, 50, 75]
    lora_keys = sorted(records_by_lora_layer.keys())

    for lora in lora_keys:
        missing = [lp for lp in layer_order if lp not in records_by_lora_layer[lora]]
        if missing:
            raise ValueError(f"Missing layers for {lora}: {missing}")

    x = np.arange(len(lora_keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c"}
    offsets = {25: -width, 50: 0.0, 75: width}

    for lp in layer_order:
        means = []
        cis = []
        for lora in lora_keys:
            vals = records_by_lora_layer[lora][lp]
            means.append(float(np.mean(vals)))
            cis.append(calculate_ci_margin(vals))

        ax.bar(
            x + offsets[lp],
            means,
            width,
            yerr=cis,
            capsize=4,
            color=colors[lp],
            label=f"Layer {lp}%",
            alpha=0.9,
        )

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



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, help="Directory containing taboo layer result json files", default="../taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test")
    parser.add_argument("--output_dir", type=str, default="./images/taboo")
    parser.add_argument("--sequence", action="store_true", help="Use full_sequence_responses instead of token_responses")
    parser.add_argument("--required_verbalizer_prompt", type=str, default=None)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise ValueError(f"json_dir does not exist: {json_dir}")

    data_dir_label = json_dir.name
    records_by_lora_layer = load_layered_results(
        json_dir=json_dir,
        sequence=args.sequence,
        required_verbalizer_prompt=args.required_verbalizer_prompt,
    )

    mode = "sequence" if args.sequence else "token"
    output_name = f"taboo_results_layer_comparison_{data_dir_label}_{mode}.png"
    output_path = Path(args.output_dir) / output_name

    plot_layer_comparison(records_by_lora_layer, output_path, args.sequence, data_dir_label)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
