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
            if is_binary:
                all_preds_t = torch.cat(all_preds)
                all_labels_t = torch.cat(all_labels)
                best_metrics = get_binary_metrics(all_preds_t, all_labels_t)
                
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
        # Apply chat template to the FULL conversation (User hint + Assistant response)
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", 
                                                  add_generation_prompt=False, enable_thinking=False).to(device)
        with torch.no_grad():
            activations = collect_activations_multiple_layers(model, submodules, {"input_ids": input_ids}, None, None)
            for l in layers:
                # Extract activation from the LAST token of the full conversation sequence
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
    # Train set (0 to num_samples)
    train_acts = {l: [] for l in layers}
    train_labels = []
    # Test set (num_samples to num_samples + num_test_samples)
    test_acts = {l: [] for l in layers}
    test_labels = []

    for word in secret_words:
        # Collect Train
        a_train = collect_acts_for_word(base_model, tokenizer, word, layers, device, 0, args.num_samples)
        if a_train:
            for l in layers: train_acts[l].extend(a_train[l])
            train_labels.extend([word_to_idx[word]] * len(a_train[layers[0]]))
        
        # Collect Test
        a_test = collect_acts_for_word(base_model, tokenizer, word, layers, device, args.num_samples, args.num_test_samples)
        if a_test:
            for l in layers: test_acts[l].extend(a_test[l])
            test_labels.extend([word_to_idx[word]] * len(a_test[layers[0]]))

    # 2. Train Probes on Base Model
    probes = {l: {} for l in layers}
    for i, l in enumerate(layers):
        print(f"\n--- Layer {l} ({layer_percents[i]}%) ---")
        X_train, Y_train = torch.stack(train_acts[l]), torch.tensor(train_labels)
        X_test, Y_test = torch.stack(test_acts[l]), torch.tensor(test_labels)
        
        # Multi-class
        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16)
        
        print("Training Multi-class Linear Probe...")
        lp_mc = LinearProbe(d_model, len(secret_words)).to(device)
        mc_acc = train_probe_loop(lp_mc, train_loader, val_loader, device, epochs=args.epochs)
        print(f"Base Multi-class Linear (Test Set) Acc: {mc_acc:.4f}")
        probes[l]['lp_mc'] = lp_mc

        print("Training Multi-class MLP Probe...")
        mlp_mc = MLPProbe(d_model, 512, len(secret_words)).to(device)
        mc_mlp_acc = train_probe_loop(mlp_mc, train_loader, val_loader, device, epochs=args.epochs)
        print(f"Base Multi-class MLP (Test Set) Acc: {mc_mlp_acc:.4f}")
        probes[l]['mlp_mc'] = mlp_mc

        # Binary Probes for ALL secret words
        probes[l]['lp_bin'] = {}
        probes[l]['mlp_bin'] = {}
        
        print(f"Training Binary Probes for all {len(secret_words)} secret words...")
        bin_metrics_list = []
        for word in tqdm(secret_words, desc=f"Layer {l} Binary Probes"):
            idx = word_to_idx[word]
            Y_train_bin = (Y_train == idx).long()
            Y_test_bin = (Y_test == idx).long()
            
            train_loader_bin = DataLoader(TensorDataset(X_train, Y_train_bin), batch_size=16, shuffle=True)
            val_loader_bin = DataLoader(TensorDataset(X_test, Y_test_bin), batch_size=16)
            
            # Linear
            lp = LinearProbe(d_model, 1).to(device)
            acc, (p, r, f1) = train_probe_loop(lp, train_loader_bin, val_loader_bin, device, epochs=args.epochs, is_binary=True)
            probes[l]['lp_bin'][word] = lp
            
            # MLP
            mlp = MLPProbe(d_model, 512, 1).to(device)
            mlp_acc, (mp, mr, mf1) = train_probe_loop(mlp, train_loader_bin, val_loader_bin, device, epochs=args.epochs, is_binary=True)
            probes[l]['mlp_bin'][word] = mlp
            
            if word == args.target_word:
                print(f"  Target word '{word}' - Linear F1: {f1:.4f}, MLP F1: {mf1:.4f}")
            bin_metrics_list.append(f1)
            
        avg_f1 = sum(bin_metrics_list) / len(bin_metrics_list)
        print(f"Base Binary Linear Avg F1 (all words): {avg_f1:.4f}")

    # 3. Transfer Evaluation to FT Model (Using UNSEEN Test set)
    model_name = args.model_name.split("/")[-1]
    if "Qwen" in model_name:
        # lora_path = f"adamkarvonen/Qwen3-8B-taboo-{args.target_word}_50_mix"
        lora_path = f"adamkarvonen/Qwen3-8B-taboo-{args.target_word}"
    elif "gemma" in model_name:
        lora_path = f"bcywinski/gemma-2-9b-it-taboo-{args.target_word}"
    print(f"\nTesting Transfer to FT Model: {lora_path}")
    try:
        ft_model = PeftModel.from_pretrained(base_model, lora_path)
        ft_model.eval()
        
        # Collect Test activations from FT model
        ft_test_acts = {l: [] for l in layers}
        ft_test_labels = []
        for word in secret_words:
            # Use same start_idx and num_test_samples as base model's test set
            a_ft = collect_acts_for_word(ft_model, tokenizer, word, layers, device, args.num_samples, args.num_test_samples)
            if not a_ft: continue
            for l in layers:
                ft_test_acts[l].extend(a_ft[l])
            ft_test_labels.extend([word_to_idx[word]] * len(a_ft[layers[0]]))

        for l in layers:
            X_ft = torch.stack(ft_test_acts[l])
            Y_ft = torch.tensor(ft_test_labels)
            Y_ft_bin = (Y_ft == word_to_idx[args.target_word]).long()
            
            # 1. Evaluate Target Word Probe (The one that was fine-tuned)
            p, r, f1 = evaluate_binary(probes[l]['lp_bin'][args.target_word], X_ft, Y_ft_bin, device)
            print(f"\nLayer {l} FT Transfer (Binary Linear - Target Word '{args.target_word}'):")
            print(f"  P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

            p, r, f1 = evaluate_binary(probes[l]['mlp_bin'][args.target_word], X_ft, Y_ft_bin, device)
            print(f"Layer {l} FT Transfer (Binary MLP - Target Word '{args.target_word}'):")
            print(f"  P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

            # 2. Evaluate Other Word Probes (Control group - should remain stable)
            other_f1s_lp = []
            other_f1s_mlp = []
            for word in secret_words:
                if word == args.target_word: continue
                idx = word_to_idx[word]
                Y_ft_word_bin = (Y_ft == idx).long()
                
                _, _, f1_lp = evaluate_binary(probes[l]['lp_bin'][word], X_ft, Y_ft_word_bin, device)
                _, _, f1_mlp = evaluate_binary(probes[l]['mlp_bin'][word], X_ft, Y_ft_word_bin, device)
                other_f1s_lp.append(f1_lp)
                other_f1s_mlp.append(f1_mlp)
            
            avg_other_f1_lp = sum(other_f1s_lp) / len(other_f1s_lp)
            avg_other_f1_mlp = sum(other_f1s_mlp) / len(other_f1s_mlp)
            print(f"Layer {l} FT Transfer (Other Words Average):")
            print(f"  Linear Avg F1: {avg_other_f1_lp:.4f}, MLP Avg F1: {avg_other_f1_mlp:.4f}")
                
            # 3. Multi-class Performance
            with torch.no_grad():
                mc_logits = probes[l]['lp_mc'](X_ft.to(device))
                mc_preds = torch.argmax(mc_logits, dim=1).cpu()
                mc_ft_acc = (mc_preds[Y_ft_bin == 1] == word_to_idx[args.target_word]).float().mean().item()
                
                mc_mlp_logits = probes[l]['mlp_mc'](X_ft.to(device))
                mc_mlp_preds = torch.argmax(mc_mlp_logits, dim=1).cpu()
                mc_mlp_ft_acc = (mc_mlp_preds[Y_ft_bin == 1] == word_to_idx[args.target_word]).float().mean().item()
                
            print(f"Layer {l} FT Transfer Multi-class Accuracy (on Target Word):")
            print(f"  Linear: {mc_ft_acc:.4f}, MLP: {mc_mlp_ft_acc:.4f}")
            
    except Exception as e:
        print(f"FT Evaluation Error: {e}")

if __name__ == "__main__":
    main()
