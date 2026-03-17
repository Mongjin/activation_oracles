import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import copy
import json

# nl_probes utilities
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

# Probe classes
class LinearProbe(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)
    def forward(self, x):
        return self.linear(x)

class MLPProbe(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def get_binary_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def train_probe_loop(model, train_loader, val_loader, device, lr=1e-3, epochs=15, is_binary=False):
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    best_metrics = (0, 0, 0) # P, R, F1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1), y.float()) if is_binary else criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_labels = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                if is_binary:
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
                else:
                    preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = correct / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            if is_binary:
                all_preds_t = torch.cat(all_preds)
                all_labels_t = torch.cat(all_labels)
                best_metrics = get_binary_metrics(all_preds_t, all_labels_t)
                
    if best_state is not None:
        model.load_state_dict(best_state)
        
    if is_binary:
        return best_acc, best_metrics
    return best_acc

def evaluate_binary(model, x, y_bin, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = (torch.sigmoid(logits.view(-1)) > 0.5).long().cpu()
    return get_binary_metrics(preds, y_bin)

def get_hider_lora_path(model_name, target_word):
    model_name_short = model_name.split("/")[-1]
    if "Qwen" in model_name_short:
        return f"adamkarvonen/Qwen3-8B-taboo-{target_word}"
    if "gemma" in model_name_short:
        return f"bcywinski/gemma-2-9b-it-taboo-{target_word}"
    raise ValueError(f"Unsupported model_name: {model_name}")

def get_default_guesser_lora_path(model_name, target_word):
    model_suffix = model_name.split("/")[-1]
    return f"/home/mongjin/activation_oracles/nl_probes/trl_training/model_lora_role_swapped/{model_suffix}-taboo-{target_word}-role-swapped"

def resolve_hider_lora_path(model_name, target_word, hider_lora_path_arg):
    if hider_lora_path_arg is None:
        return get_hider_lora_path(model_name, target_word)
    if "{target_word}" in hider_lora_path_arg:
        return hider_lora_path_arg.format(target_word=target_word)
    return hider_lora_path_arg

def resolve_guesser_lora_path(model_name, target_word, guesser_lora_path_arg):
    if guesser_lora_path_arg is None:
        return get_default_guesser_lora_path(model_name, target_word)
    if "{target_word}" in guesser_lora_path_arg:
        return guesser_lora_path_arg.format(target_word=target_word)
    return guesser_lora_path_arg

def collect_acts_for_word(model, tokenizer, word, layers, device, start_idx=0, num_samples=100):
    """Extract activations from the model using FULL conversations from the Taboo dataset."""
    dataset_name = f"bcywinski/taboo-{word}"
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return {}

    acts_l = {l: [] for l in layers}
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    
    end_idx = min(len(ds), start_idx + num_samples)
    if start_idx >= len(ds):
        return {}
    samples = ds.select(range(start_idx, end_idx))
    
    for ex in tqdm(samples, desc=f"Collecting acts for '{word}' (idx {start_idx}-{end_idx})"):
        messages = ex['messages']
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", 
                                                  add_generation_prompt=False, enable_thinking=False).to(device)
        with torch.no_grad():
            activations = collect_activations_multiple_layers(model, submodules, {"input_ids": input_ids}, None, None)
            for l in layers:
                acts_l[l].append(activations[l][0, -1, :].cpu().float())
    return acts_l

def collect_activation_splits(model, tokenizer, secret_words, word_to_idx, layers, device, num_samples, num_test_samples):
    train_acts = {l: [] for l in layers}
    test_acts = {l: [] for l in layers}
    labels_train = []
    labels_test = []

    for word in secret_words:
        a_train = collect_acts_for_word(model, tokenizer, word, layers, device, 0, num_samples)
        if a_train:
            for l in layers:
                train_acts[l].extend(a_train[l])
            labels_train.extend([word_to_idx[word]] * len(a_train[layers[0]]))

        a_test = collect_acts_for_word(model, tokenizer, word, layers, device, num_samples, num_test_samples)
        if a_test:
            for l in layers:
                test_acts[l].extend(a_test[l])
            labels_test.extend([word_to_idx[word]] * len(a_test[layers[0]]))

    return {
        "train_acts": train_acts,
        "test_acts": test_acts,
        "labels_train": labels_train,
        "labels_test": labels_test,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--secret_words", type=str, default="ship,wave,song,snow,rock,moon,jump,green,flame,flag,dance,cloud,clock,chair,salt,book,blue,gold,leaf,smile")
    parser.add_argument("--target_word", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=200, help="Num samples for probe training per word")
    parser.add_argument("--num_test_samples", type=int, default=50, help="Num samples for evaluation per word")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hider_lora_path", type=str, default=None)
    parser.add_argument("--guesser_lora_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    secret_words = [w.strip() for w in args.secret_words.split(",")]
    target_words = secret_words if args.target_word is None else [w.strip() for w in args.target_word.split(",")]
    word_to_idx = {w: i for i, w in enumerate(secret_words)}

    if len(target_words) > 1 and args.hider_lora_path is not None:
        assert "{target_word}" in args.hider_lora_path
    if len(target_words) > 1 and args.guesser_lora_path is not None:
        assert "{target_word}" in args.guesser_lora_path
    
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_model(args.model_name, torch.bfloat16)
    
    layer_percents = [25, 50, 75]
    layers = [layer_percent_to_layer(args.model_name, p) for p in layer_percents]
    d_model = base_model.config.hidden_size

    # 1. Base Model Activation Collection
    print("\n--- Collecting Activations from Base Model ---")
    base_data = collect_activation_splits(
        base_model, tokenizer, secret_words, word_to_idx, layers, device, args.num_samples, args.num_test_samples
    )

    model_name_short = args.model_name.split("/")[-1]
    probe_sources = ["base", "hider", "guesser"]
    os.makedirs("./taboo_eval_results", exist_ok=True)

    for target_word in target_words:
        hider_lora_path = resolve_hider_lora_path(args.model_name, target_word, args.hider_lora_path)
        guesser_lora_path = resolve_guesser_lora_path(args.model_name, target_word, args.guesser_lora_path)

        print(f"\n--- Loading Hider Model and Collecting Activations: {hider_lora_path} ---")
        try:
            hider_base_model = load_model(args.model_name, torch.bfloat16)
            hider_model = PeftModel.from_pretrained(hider_base_model, hider_lora_path)
            hider_model.eval()
            hider_data = collect_activation_splits(
                hider_model, tokenizer, secret_words, word_to_idx, layers, device, args.num_samples, args.num_test_samples
            )
        except Exception as e:
            print(f"Error collecting Hider activations: {e}")
            return

        print(f"\n--- Loading Guesser Model and Collecting Activations: {guesser_lora_path} ---")
        try:
            guesser_base_model = load_model(args.model_name, torch.bfloat16)
            guesser_model = PeftModel.from_pretrained(guesser_base_model, guesser_lora_path)
            guesser_model.eval()
            guesser_data = collect_activation_splits(
                guesser_model, tokenizer, secret_words, word_to_idx, layers, device, args.num_samples, args.num_test_samples
            )
        except Exception as e:
            print(f"Error collecting Guesser activations: {e}")
            return

        activation_data = {
            "base": base_data,
            "hider": hider_data,
            "guesser": guesser_data,
        }

        results = {
            "model_name": args.model_name,
            "target_word": target_word,
            "target_words_requested": target_words,
            "hider_lora_path": hider_lora_path,
            "guesser_lora_path": guesser_lora_path,
            "layers": {}
        }

        for i, l in enumerate(layers):
            print(f"\n==================== Target {target_word} | Layer {l} ({layer_percents[i]}%) ====================")
            source_tensors = {}
            for source_name in probe_sources:
                source_data = activation_data[source_name]
                source_tensors[source_name] = {
                    "train_x": torch.stack(source_data["train_acts"][l]),
                    "train_y": torch.tensor(source_data["labels_train"]),
                    "test_x": torch.stack(source_data["test_acts"][l]),
                    "test_y": torch.tensor(source_data["labels_test"]),
                }

            probe_results = {
                source_name: {f"{eval_name}_eval": {} for eval_name in probe_sources}
                for source_name in probe_sources
            }

            for source_idx, train_source in enumerate(probe_sources):
                print(f"\n[{chr(ord('A') + source_idx)}] Training Probes on {train_source.upper()} Model Activations...")
                train_x = source_tensors[train_source]["train_x"]
                train_y = source_tensors[train_source]["train_y"]
                test_x_same = source_tensors[train_source]["test_x"]
                test_y_same = source_tensors[train_source]["test_y"]

                train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
                test_loader_same = DataLoader(TensorDataset(test_x_same, test_y_same), batch_size=16)

                lp_mc = LinearProbe(d_model, len(secret_words)).to(device)
                mc_acc_same = train_probe_loop(lp_mc, train_loader, test_loader_same, device, epochs=args.epochs)
                mlp_mc = MLPProbe(d_model, 512, len(secret_words)).to(device)
                mc_mlp_acc_same = train_probe_loop(mlp_mc, train_loader, test_loader_same, device, epochs=args.epochs)
                probe_results[train_source][f"{train_source}_eval"]["mc_linear_acc"] = mc_acc_same
                probe_results[train_source][f"{train_source}_eval"]["mc_mlp_acc"] = mc_mlp_acc_same

                lp_bin = {}
                mlp_bin = {}
                for word in tqdm(secret_words, desc=f"{train_source.capitalize()}-trained Binary Probes"):
                    idx = word_to_idx[word]
                    word_train_loader = DataLoader(
                        TensorDataset(train_x, (train_y == idx).long()), batch_size=16, shuffle=True
                    )
                    word_test_loader_same = DataLoader(
                        TensorDataset(test_x_same, (test_y_same == idx).long()), batch_size=16
                    )

                    lp = LinearProbe(d_model, 1).to(device)
                    _, (p, r, f1) = train_probe_loop(
                        lp, word_train_loader, word_test_loader_same, device, epochs=args.epochs, is_binary=True
                    )
                    lp_bin[word] = lp

                    mlp = MLPProbe(d_model, 512, 1).to(device)
                    _, (mp, mr, mf1) = train_probe_loop(
                        mlp, word_train_loader, word_test_loader_same, device, epochs=args.epochs, is_binary=True
                    )
                    mlp_bin[word] = mlp

                    if word == target_word:
                        probe_results[train_source][f"{train_source}_eval"][word] = {
                            "linear": {"p": p, "r": r, "f1": f1},
                            "mlp": {"p": mp, "r": mr, "f1": mf1}
                        }

                for eval_source in probe_sources:
                    if eval_source == train_source:
                        continue

                    print(
                        f"Evaluating {train_source.capitalize()} Probes on {eval_source.capitalize()} Data "
                        f"({train_source} -> {eval_source})..."
                    )
                    eval_x = source_tensors[eval_source]["test_x"]
                    eval_y = source_tensors[eval_source]["test_y"]
                    target_labels = (eval_y == word_to_idx[target_word]).long()

                    p_lp, r_lp, f1_lp = evaluate_binary(lp_bin[target_word], eval_x, target_labels, device)
                    p_mlp, r_mlp, f1_mlp = evaluate_binary(mlp_bin[target_word], eval_x, target_labels, device)
                    probe_results[train_source][f"{eval_source}_eval"][target_word] = {
                        "linear": {"p": p_lp, "r": r_lp, "f1": f1_lp},
                        "mlp": {"p": p_mlp, "r": r_mlp, "f1": f1_mlp}
                    }

                    with torch.no_grad():
                        mc_preds = torch.argmax(lp_mc(eval_x.to(device)), dim=1).cpu()
                        mc_mlp_preds = torch.argmax(mlp_mc(eval_x.to(device)), dim=1).cpu()
                        target_mask = (eval_y == word_to_idx[target_word])
                        mc_acc_target = (mc_preds[target_mask] == word_to_idx[target_word]).float().mean().item()
                        mc_mlp_acc_target = (
                            mc_mlp_preds[target_mask] == word_to_idx[target_word]
                        ).float().mean().item()
                    probe_results[train_source][f"{eval_source}_eval"]["mc_linear_acc_on_target"] = mc_acc_target
                    probe_results[train_source][f"{eval_source}_eval"]["mc_mlp_acc_on_target"] = mc_mlp_acc_target

            layer_results = {
                "percent": layer_percents[i],
                "base_probes": probe_results["base"],
                "hider_probes": probe_results["hider"],
                "ft_probes": probe_results["hider"],
                "guesser_probes": probe_results["guesser"],
            }
            results["layers"][l] = layer_results

        filename = f"./taboo_eval_results/taboo_probe_bidirectional_{model_name_short.replace('/', '_')}_{target_word}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nBidirectional results saved to: {filename}")

if __name__ == "__main__":
    main()
