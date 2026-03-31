import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SECRET_WORDS = [
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


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def infer_hider_lora_path(model_name: str, target_word: str) -> str:
    model_name_short = model_name.split("/")[-1]
    if "Qwen" in model_name_short:
        return f"adamkarvonen/Qwen3-8B-taboo-{target_word}"
    if "gemma" in model_name_short:
        return f"bcywinski/gemma-2-9b-it-taboo-{target_word}"
    raise ValueError(f"Unsupported model_name: {model_name}")


def infer_guesser_lora_path(model_name: str, target_word: str) -> str:
    model_suffix = model_name.split("/")[-1]
    return (
        f"/home/mongjin/activation_oracles/nl_probes/trl_training/model_lora_role_swapped/"
        f"{model_suffix}-taboo-{target_word}-role-swapped"
    )


def resolve_hider_lora_path(model_name: str, target_word: str, hider_lora_path_arg: str | None) -> str:
    if hider_lora_path_arg is None:
        return infer_hider_lora_path(model_name, target_word)
    if "{target_word}" in hider_lora_path_arg:
        return hider_lora_path_arg.format(target_word=target_word)
    return hider_lora_path_arg


def resolve_guesser_lora_path(model_name: str, target_word: str, guesser_lora_path_arg: str | None) -> str:
    if guesser_lora_path_arg is None:
        return infer_guesser_lora_path(model_name, target_word)
    if "{target_word}" in guesser_lora_path_arg:
        return guesser_lora_path_arg.format(target_word=target_word)
    return guesser_lora_path_arg


def load_context_prompts(prompt_type: str, dataset_type: str, lang_type: str | None) -> list[str]:
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


def split_train_val_indices(labels: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    for class_idx in labels.unique(sorted=True):
        class_indices = torch.nonzero(labels == class_idx, as_tuple=False).view(-1)
        perm = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        num_val = int(round(class_indices.numel() * val_fraction))
        val_indices.append(perm[:num_val])
        train_indices.append(perm[num_val:])
    return torch.cat(train_indices), torch.cat(val_indices)


def train_multiclass_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    lr: float,
    epochs: int,
) -> tuple[nn.Module, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = -1.0
    best_state = None

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = torch.argmax(model(batch_x), dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def train_binary_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    lr: float,
    epochs: int,
) -> tuple[nn.Module, dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = -1.0
    best_metrics = None
    best_state = None

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).view(-1)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x).view(-1)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu()
                all_preds.append(preds)
                all_labels.append(batch_y.cpu())

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)
        metrics = get_binary_metrics(all_preds_t, all_labels_t)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_metrics


def collect_dataset_last_token_activations(
    model,
    tokenizer,
    secret_word: str,
    layers: list[int],
    device: torch.device,
    num_samples: int,
) -> dict[int, list[torch.Tensor]]:
    dataset_name = f"bcywinski/taboo-{secret_word}"
    ds = load_dataset(dataset_name, split="train")
    samples = ds.select(range(num_samples))

    acts_by_layer = {layer: [] for layer in layers}
    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}

    for example in tqdm(samples, desc=f"Dataset acts | {secret_word}", leave=False):
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
            enable_thinking=False,
        ).to(device)

        with torch.no_grad():
            activations = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL={"input_ids": input_ids},
                min_offset=None,
                max_offset=None,
            )
        for layer in layers:
            acts_by_layer[layer].append(activations[layer][0, -1, :].cpu().float())

    return acts_by_layer


def collect_context_prompt_last_token_activations(
    model,
    tokenizer,
    context_prompts: list[str],
    layers: list[int],
    device: torch.device,
    eval_batch_size: int,
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

        batch_acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

        attention_mask = inputs_BL["attention_mask"]
        seq_len = attention_mask.shape[1]
        for layer in layers:
            pooled = []
            for batch_idx in range(attention_mask.shape[0]):
                real_len = int(attention_mask[batch_idx].sum().item())
                left_pad = seq_len - real_len
                last_idx = left_pad + real_len - 1
                pooled.append(batch_acts[layer][batch_idx, last_idx, :].cpu().float())
            acts_by_layer[layer].append(torch.stack(pooled, dim=0))

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in acts_by_layer.items()}


def compute_target_ranks(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    return torch.argmax((sorted_indices == target_idx).long(), dim=1) + 1


def save_prompt_summary_csv(
    output_path: Path,
    prompt_rows: list[dict],
    include_multiclass: bool,
    include_binary: bool,
) -> None:
    fieldnames = ["prompt_id", "prompt_text"]
    if include_multiclass:
        fieldnames += [
            "linear_target_prob_mean",
            "linear_top1_acc",
            "linear_mean_rank",
            "mlp_target_prob_mean",
            "mlp_top1_acc",
            "mlp_mean_rank",
        ]
    if include_binary:
        fieldnames += [
            "binary_linear_target_prob_mean",
            "binary_linear_positive_recall",
            "binary_mlp_target_prob_mean",
            "binary_mlp_positive_recall",
        ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prompt_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
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
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_context_prompts", type=int, default=None)
    parser.add_argument(
        "--probe_modes",
        type=str,
        nargs="+",
        default=["multiclass", "binary"],
        choices=["multiclass", "binary"],
    )
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./experiments/taboo_eval_results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    secret_words = [word.strip() for word in args.secret_words.split(",")]
    word_to_idx = {word: idx for idx, word in enumerate(secret_words)}
    layers = [layer_percent_to_layer(args.model_name, layer_percent) for layer_percent in args.layer_percents]
    context_prompts = load_context_prompts(args.prompt_type, args.dataset_type, args.lang_type)
    if args.max_context_prompts is not None:
        context_prompts = context_prompts[: args.max_context_prompts]

    model_name_short = args.model_name.split("/")[-1].replace(".", "_")
    lang_suffix = f"_{args.lang_type}" if args.lang_type else ""
    output_dir = Path(
        args.output_dir,
        f"{model_name_short}_context_prompt_probe_{args.prompt_type}{lang_suffix}_{args.dataset_type}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_name)

    dataset_acts = {
        "hider": {layer: [] for layer in layers},
        "guesser": {layer: [] for layer in layers},
    }
    dataset_labels = {
        "hider": [],
        "guesser": [],
    }
    context_acts = {
        "hider": {},
        "guesser": {},
    }

    d_model = None

    for target_word in tqdm(secret_words, desc="Collect source activations"):
        hider_model = PeftModel.from_pretrained(
            load_model(args.model_name, torch.bfloat16),
            resolve_hider_lora_path(args.model_name, target_word, args.hider_lora_path),
        )
        hider_model.eval()
        if d_model is None:
            d_model = hider_model.config.hidden_size

        hider_dataset_acts = collect_dataset_last_token_activations(
            model=hider_model,
            tokenizer=tokenizer,
            secret_word=target_word,
            layers=layers,
            device=device,
            num_samples=args.num_probe_samples,
        )
        for layer in layers:
            dataset_acts["hider"][layer].extend(hider_dataset_acts[layer])
        dataset_labels["hider"].extend([word_to_idx[target_word]] * args.num_probe_samples)

        context_acts["hider"][target_word] = collect_context_prompt_last_token_activations(
            model=hider_model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        del hider_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        guesser_model = PeftModel.from_pretrained(
            load_model(args.model_name, torch.bfloat16),
            resolve_guesser_lora_path(args.model_name, target_word, args.guesser_lora_path),
        )
        guesser_model.eval()

        guesser_dataset_acts = collect_dataset_last_token_activations(
            model=guesser_model,
            tokenizer=tokenizer,
            secret_word=target_word,
            layers=layers,
            device=device,
            num_samples=args.num_probe_samples,
        )
        for layer in layers:
            dataset_acts["guesser"][layer].extend(guesser_dataset_acts[layer])
        dataset_labels["guesser"].extend([word_to_idx[target_word]] * args.num_probe_samples)

        context_acts["guesser"][target_word] = collect_context_prompt_last_token_activations(
            model=guesser_model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            layers=layers,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        del guesser_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "config": {
            "model_name": args.model_name,
            "prompt_type": args.prompt_type,
            "dataset_type": args.dataset_type,
            "lang_type": args.lang_type,
            "num_probe_samples": args.num_probe_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "val_fraction": args.val_fraction,
            "layer_percents": args.layer_percents,
            "eval_batch_size": args.eval_batch_size,
            "max_context_prompts": args.max_context_prompts,
            "probe_modes": args.probe_modes,
            "secret_words": secret_words,
            "hider_lora_path_arg": args.hider_lora_path,
            "guesser_lora_path_arg": args.guesser_lora_path,
            "seed": args.seed,
        },
        "context_prompts": context_prompts,
        "sources": {},
    }

    for source_name in ["hider", "guesser"]:
        labels = torch.tensor(dataset_labels[source_name], dtype=torch.long)
        results["sources"][source_name] = {"layers": {}}

        for layer_percent, layer in zip(args.layer_percents, layers):
            acts = torch.stack(dataset_acts[source_name][layer])
            train_indices, val_indices = split_train_val_indices(labels, args.val_fraction, args.seed)
            train_x = acts[train_indices]
            train_y = labels[train_indices]
            val_x = acts[val_indices]
            val_y = labels[val_indices]

            linear_probe = None
            mlp_probe = None
            linear_val_acc = None
            mlp_val_acc = None
            if "multiclass" in args.probe_modes:
                linear_probe, linear_val_acc = train_multiclass_probe(
                    model=LinearProbe(d_model, len(secret_words)).to(device),
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    device=device,
                    lr=args.lr,
                    epochs=args.epochs,
                )
                mlp_probe, mlp_val_acc = train_multiclass_probe(
                    model=MLPProbe(d_model, args.mlp_hidden_dim, len(secret_words)).to(device),
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    device=device,
                    lr=args.lr,
                    epochs=args.epochs,
                )

            binary_linear_probes = {}
            binary_mlp_probes = {}
            binary_val_metrics = {}
            if "binary" in args.probe_modes:
                for target_word in tqdm(secret_words, desc=f"Binary probes | {source_name} | {layer_percent}%", leave=False):
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
                        lr=args.lr,
                        epochs=args.epochs,
                    )
                    binary_mlp_probe, binary_mlp_metrics = train_binary_probe(
                        model=MLPProbe(d_model, args.mlp_hidden_dim, 1).to(device),
                        train_x=train_x,
                        train_y=train_y_bin,
                        val_x=val_x,
                        val_y=val_y_bin,
                        device=device,
                        lr=args.lr,
                        epochs=args.epochs,
                    )

                    binary_linear_probes[target_word] = binary_linear_probe
                    binary_mlp_probes[target_word] = binary_mlp_probe
                    binary_val_metrics[target_word] = {
                        "linear": binary_linear_metrics,
                        "mlp": binary_mlp_metrics,
                    }

            prompt_records = {prompt_idx: [] for prompt_idx in range(len(context_prompts))}
            for target_word in secret_words:
                target_idx = word_to_idx[target_word]
                eval_x = context_acts[source_name][target_word][layer].to(device)

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
                    if "multiclass" in args.probe_modes:
                        linear_logits = linear_probe(eval_x)
                        linear_probs = torch.softmax(linear_logits, dim=1)
                        linear_preds = torch.argmax(linear_logits, dim=1)
                        linear_target_probs = linear_probs[:, target_idx].cpu()
                        linear_target_ranks = compute_target_ranks(linear_logits, target_idx).cpu()

                        mlp_logits = mlp_probe(eval_x)
                        mlp_probs = torch.softmax(mlp_logits, dim=1)
                        mlp_preds = torch.argmax(mlp_logits, dim=1)
                        mlp_target_probs = mlp_probs[:, target_idx].cpu()
                        mlp_target_ranks = compute_target_ranks(mlp_logits, target_idx).cpu()

                    if "binary" in args.probe_modes:
                        binary_linear_logits = binary_linear_probes[target_word](eval_x).view(-1)
                        binary_linear_target_probs = torch.sigmoid(binary_linear_logits).cpu()
                        binary_linear_positive_preds = (binary_linear_target_probs > 0.5).long()

                        binary_mlp_logits = binary_mlp_probes[target_word](eval_x).view(-1)
                        binary_mlp_target_probs = torch.sigmoid(binary_mlp_logits).cpu()
                        binary_mlp_positive_preds = (binary_mlp_target_probs > 0.5).long()

                for prompt_idx, prompt_text in enumerate(context_prompts):
                    record = {
                        "target_word": target_word,
                        "prompt_id": f"P{prompt_idx + 1:03d}",
                        "prompt_text": prompt_text,
                    }
                    if "multiclass" in args.probe_modes:
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
                    if "binary" in args.probe_modes:
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
            detailed_records = []
            for prompt_idx, prompt_text in enumerate(context_prompts):
                records = prompt_records[prompt_idx]
                detailed_records.extend(records)
                prompt_rows.append(
                    {
                        "prompt_id": f"P{prompt_idx + 1:03d}",
                        "prompt_text": prompt_text,
                    }
                )
                if "multiclass" in args.probe_modes:
                    prompt_rows[-1].update(
                        {
                            "linear_target_prob_mean": sum(record["linear_target_prob"] for record in records) / len(records),
                            "linear_top1_acc": sum(record["linear_correct_top1"] for record in records) / len(records),
                            "linear_mean_rank": sum(record["linear_target_rank"] for record in records) / len(records),
                            "mlp_target_prob_mean": sum(record["mlp_target_prob"] for record in records) / len(records),
                            "mlp_top1_acc": sum(record["mlp_correct_top1"] for record in records) / len(records),
                            "mlp_mean_rank": sum(record["mlp_target_rank"] for record in records) / len(records),
                        }
                    )
                if "binary" in args.probe_modes:
                    prompt_rows[-1].update(
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

            if "multiclass" in args.probe_modes:
                prompt_rows.sort(key=lambda row: row["linear_target_prob_mean"], reverse=True)
            else:
                prompt_rows.sort(key=lambda row: row["binary_linear_target_prob_mean"], reverse=True)

            results["sources"][source_name]["layers"][str(layer_percent)] = {
                "layer_percent": layer_percent,
                "layer_index": layer,
                "linear_probe_val_acc": linear_val_acc,
                "mlp_probe_val_acc": mlp_val_acc,
                "binary_probe_val_metrics": binary_val_metrics,
                "prompt_rows": prompt_rows,
                "detailed_records": detailed_records,
            }

            save_prompt_summary_csv(
                output_path=output_dir / f"{source_name}_prompt_probe_summary_layer_{layer_percent}.csv",
                prompt_rows=prompt_rows,
                include_multiclass="multiclass" in args.probe_modes,
                include_binary="binary" in args.probe_modes,
            )

    output_json = output_dir / "taboo_context_prompt_probe_eval.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved probe prompt evaluation to {output_json}")


if __name__ == "__main__":
    main()
