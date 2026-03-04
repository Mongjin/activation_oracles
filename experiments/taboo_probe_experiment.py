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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--secret_words", type=str, default="ship,wave,song,snow,rock,moon,jump,green,flame,flag,dance,cloud,clock,chair,salt,book,blue,gold,leaf,smile")
    parser.add_argument("--target_word", type=str, default="smile")
    parser.add_argument("--num_samples", type=int, default=200, help="Num samples for probe training per word")
    parser.add_argument("--num_test_samples", type=int, default=50, help="Num samples for evaluation per word")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    secret_words = [w.strip() for w in args.secret_words.split(",")]
    word_to_idx = {w: i for i, w in enumerate(secret_words)}
    
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_model(args.model_name, torch.bfloat16)
    
    layer_percents = [25, 50, 75]
    layers = [layer_percent_to_layer(args.model_name, p) for p in layer_percents]
    d_model = base_model.config.hidden_size

    # 1. Base Model Activation Collection
    print("\n--- Collecting Activations from Base Model ---")
    base_train_acts = {l: [] for l in layers}
    base_test_acts = {l: [] for l in layers}
    base_labels_train = []
    base_labels_test = []

    for word in secret_words:
        a_train = collect_acts_for_word(base_model, tokenizer, word, layers, device, 0, args.num_samples)
        if a_train:
            for l in layers: base_train_acts[l].extend(a_train[l])
            base_labels_train.extend([word_to_idx[word]] * len(a_train[layers[0]]))
        
        a_test = collect_acts_for_word(base_model, tokenizer, word, layers, device, args.num_samples, args.num_test_samples)
        if a_test:
            for l in layers: base_test_acts[l].extend(a_test[l])
            base_labels_test.extend([word_to_idx[word]] * len(a_test[layers[0]]))

    # 2. FT Model Activation Collection
    model_name_short = args.model_name.split("/")[-1]
    if "Qwen" in model_name_short:
        lora_path = f"adamkarvonen/Qwen3-8B-taboo-{args.target_word}"
    elif "gemma" in model_name_short:
        lora_path = f"bcywinski/gemma-2-9b-it-taboo-{args.target_word}"
    
    print(f"\n--- Loading FT Model and Collecting Activations: {lora_path} ---")
    ft_train_acts = {l: [] for l in layers}
    ft_test_acts = {l: [] for l in layers}
    ft_labels_train = []
    ft_labels_test = []

    try:
        ft_model = PeftModel.from_pretrained(base_model, lora_path)
        ft_model.eval()
        
        for word in secret_words:
            a_train = collect_acts_for_word(ft_model, tokenizer, word, layers, device, 0, args.num_samples)
            if a_train:
                for l in layers: ft_train_acts[l].extend(a_train[l])
                ft_labels_train.extend([word_to_idx[word]] * len(a_train[layers[0]]))
            
            a_test = collect_acts_for_word(ft_model, tokenizer, word, layers, device, args.num_samples, args.num_test_samples)
            if a_test:
                for l in layers: ft_test_acts[l].extend(a_test[l])
                ft_labels_test.extend([word_to_idx[word]] * len(a_test[layers[0]]))
        
    except Exception as e:
        print(f"Error collecting FT activations: {e}")
        return

    # Results dictionary for JSON storage
    results = {
        "model_name": args.model_name,
        "target_word": args.target_word,
        "layers": {}
    }

    # 3. Training and Evaluation
    for i, l in enumerate(layers):
        print(f"\n==================== Layer {l} ({layer_percents[i]}%) ====================")
        layer_results = {
            "percent": layer_percents[i],
            "base_probes": {"base_eval": {}, "ft_eval": {}},
            "ft_probes": {"ft_eval": {}, "base_eval": {}}
        }

        # Data preparation
        XB_train, YB_train = torch.stack(base_train_acts[l]), torch.tensor(base_labels_train)
        XB_test, YB_test = torch.stack(base_test_acts[l]), torch.tensor(base_labels_test)
        XF_train, YF_train = torch.stack(ft_train_acts[l]), torch.tensor(ft_labels_train)
        XF_test, YF_test = torch.stack(ft_test_acts[l]), torch.tensor(ft_labels_test)

        base_train_loader = DataLoader(TensorDataset(XB_train, YB_train), batch_size=16, shuffle=True)
        base_test_loader = DataLoader(TensorDataset(XB_test, YB_test), batch_size=16)
        ft_train_loader = DataLoader(TensorDataset(XF_train, YF_train), batch_size=16, shuffle=True)
        ft_test_loader = DataLoader(TensorDataset(XF_test, YF_test), batch_size=16)

        # ---------------------------------------------------------
        # A. TRAIN PROBES ON BASE MODEL
        # ---------------------------------------------------------
        print(f"\n[A] Training Probes on BASE Model Activations...")
        
        # Multi-class Probes
        lp_mc_base = LinearProbe(d_model, len(secret_words)).to(device)
        mc_acc_base = train_probe_loop(lp_mc_base, base_train_loader, base_test_loader, device, epochs=args.epochs)
        mlp_mc_base = MLPProbe(d_model, 512, len(secret_words)).to(device)
        mc_mlp_acc_base = train_probe_loop(mlp_mc_base, base_train_loader, base_test_loader, device, epochs=args.epochs)
        layer_results["base_probes"]["base_eval"]["mc_linear_acc"] = mc_acc_base
        layer_results["base_probes"]["base_eval"]["mc_mlp_acc"] = mc_mlp_acc_base

        # Binary Probes (Linear + MLP)
        lp_bin_base = {}
        mlp_bin_base = {}
        for word in tqdm(secret_words, desc="Base-trained Binary Probes"):
            idx = word_to_idx[word]
            train_loader = DataLoader(TensorDataset(XB_train, (YB_train == idx).long()), batch_size=16, shuffle=True)
            test_loader = DataLoader(TensorDataset(XB_test, (YB_test == idx).long()), batch_size=16)
            
            lp = LinearProbe(d_model, 1).to(device)
            acc, (p, r, f1) = train_probe_loop(lp, train_loader, test_loader, device, epochs=args.epochs, is_binary=True)
            lp_bin_base[word] = lp
            
            mlp = MLPProbe(d_model, 512, 1).to(device)
            _, (mp, mr, mf1) = train_probe_loop(mlp, train_loader, test_loader, device, epochs=args.epochs, is_binary=True)
            mlp_bin_base[word] = mlp
            
            if word == args.target_word:
                layer_results["base_probes"]["base_eval"][word] = {"linear_f1": f1, "mlp_f1": mf1}

        # Transfer Eval: Base Probes -> FT Data
        print("Evaluating Base Probes on FT Data (Base -> FT)...")
        p, r, f1_lp = evaluate_binary(lp_bin_base[args.target_word], XF_test, (YF_test == word_to_idx[args.target_word]).long(), device)
        p, r, f1_mlp = evaluate_binary(mlp_bin_base[args.target_word], XF_test, (YF_test == word_to_idx[args.target_word]).long(), device)
        layer_results["base_probes"]["ft_eval"][args.target_word] = {"linear_f1": f1_lp, "mlp_f1": f1_mlp}
        
        with torch.no_grad():
            mc_preds = torch.argmax(lp_mc_base(XF_test.to(device)), dim=1).cpu()
            target_mask = (YF_test == word_to_idx[args.target_word])
            mc_acc_ft = (mc_preds[target_mask] == word_to_idx[args.target_word]).float().mean().item()
        layer_results["base_probes"]["ft_eval"]["mc_linear_acc_on_target"] = mc_acc_ft

        # ---------------------------------------------------------
        # B. TRAIN PROBES ON FT MODEL
        # ---------------------------------------------------------
        print(f"\n[B] Training Probes on FT Model Activations...")
        
        # Multi-class Probes
        lp_mc_ft = LinearProbe(d_model, len(secret_words)).to(device)
        mc_acc_ft = train_probe_loop(lp_mc_ft, ft_train_loader, ft_test_loader, device, epochs=args.epochs)
        mlp_mc_ft = MLPProbe(d_model, 512, len(secret_words)).to(device)
        mc_mlp_acc_ft = train_probe_loop(mlp_mc_ft, ft_train_loader, ft_test_loader, device, epochs=args.epochs)
        layer_results["ft_probes"]["ft_eval"]["mc_linear_acc"] = mc_acc_ft
        layer_results["ft_probes"]["ft_eval"]["mc_mlp_acc"] = mc_mlp_acc_ft

        # Binary Probes (Linear + MLP)
        lp_bin_ft = {}
        mlp_bin_ft = {}
        for word in tqdm(secret_words, desc="FT-trained Binary Probes"):
            idx = word_to_idx[word]
            train_loader = DataLoader(TensorDataset(XF_train, (YF_train == idx).long()), batch_size=16, shuffle=True)
            test_loader = DataLoader(TensorDataset(XF_test, (YF_test == idx).long()), batch_size=16)
            
            lp = LinearProbe(d_model, 1).to(device)
            acc, (p, r, f1) = train_probe_loop(lp, train_loader, test_loader, device, epochs=args.epochs, is_binary=True)
            lp_bin_ft[word] = lp
            
            mlp = MLPProbe(d_model, 512, 1).to(device)
            _, (mp, mr, mf1) = train_probe_loop(mlp, train_loader, test_loader, device, epochs=args.epochs, is_binary=True)
            mlp_bin_ft[word] = mlp
            
            if word == args.target_word:
                layer_results["ft_probes"]["ft_eval"][word] = {"linear_f1": f1, "mlp_f1": mf1}

        # Transfer Eval: FT Probes -> Base Data
        print("Evaluating FT Probes on Base Data (FT -> Base)...")
        p, r, f1_lp = evaluate_binary(lp_bin_ft[args.target_word], XB_test, (YB_test == word_to_idx[args.target_word]).long(), device)
        p, r, f1_mlp = evaluate_binary(mlp_bin_ft[args.target_word], XB_test, (YB_test == word_to_idx[args.target_word]).long(), device)
        layer_results["ft_probes"]["base_eval"][args.target_word] = {"linear_f1": f1_lp, "mlp_f1": f1_mlp}
        
        with torch.no_grad():
            mc_preds = torch.argmax(lp_mc_ft(XB_test.to(device)), dim=1).cpu()
            target_mask = (YB_test == word_to_idx[args.target_word])
            mc_acc_base = (mc_preds[target_mask] == word_to_idx[args.target_word]).float().mean().item()
        layer_results["ft_probes"]["base_eval"]["mc_linear_acc_on_target"] = mc_acc_base

        results["layers"][l] = layer_results

    # 4. Save to JSON
    os.makedirs("./taboo_eval_results", exist_ok=True)
    filename = f"./taboo_eval_results/taboo_probe_bidirectional_{model_name_short.replace('/', '_')}_{args.target_word}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nBidirectional results saved to: {filename}")

if __name__ == "__main__":
    main()
