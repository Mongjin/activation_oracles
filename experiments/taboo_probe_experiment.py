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

def train_probe_loop(model, train_loader, val_loader, device, lr=1e-3, epochs=15, is_binary=False):
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
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
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = (torch.sigmoid(logits.view(-1)) > 0.5).long() if is_binary else torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0
        if acc > best_acc: best_acc = acc
    return best_acc

def collect_acts_for_word(model, tokenizer, word, layers, device, num_samples=100):
    """Extract activations from the model using FULL conversations from the Taboo dataset."""
    dataset_name = f"bcywinski/taboo-{word}"
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return {}

    acts_l = {l: [] for l in layers}
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    samples = ds.select(range(min(len(ds), num_samples)))
    
    for ex in tqdm(samples, desc=f"Collecting acts for '{word}'"):
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
    parser.add_argument("--num_samples", type=int, default=250)
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

    # 1. Base Model Activation Collection (Full Conversation Context)
    all_acts = {l: [] for l in layers}
    all_labels = []
    for word in secret_words:
        acts_dict = collect_acts_for_word(base_model, tokenizer, word, layers, device, args.num_samples)
        if not acts_dict: continue
        for l in layers:
            all_acts[l].extend(acts_dict[l])
        all_labels.extend([word_to_idx[word]] * len(acts_dict[layers[0]]))

    # 2. Train Probes on Base Model
    probes = {l: {} for l in layers}
    for i, l in enumerate(layers):
        print(f"\n--- Layer {l} ({layer_percents[i]}%) ---")
        X, Y = torch.stack(all_acts[l]), torch.tensor(all_labels)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[:int(0.8*len(X))], indices[int(0.8*len(X)):]
        
        # Multi-class
        train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=16)
        
        print("Training Multi-class Linear Probe...")
        lp_mc = LinearProbe(d_model, len(secret_words)).to(device)
        mc_acc = train_probe_loop(lp_mc, train_loader, val_loader, device, epochs=args.epochs)
        print(f"Base Multi-class Linear Acc: {mc_acc:.4f}")
        probes[l]['lp_mc'] = lp_mc

        print("Training Multi-class MLP Probe...")
        mlp_mc = MLPProbe(d_model, 512, len(secret_words)).to(device)
        mc_mlp_acc = train_probe_loop(mlp_mc, train_loader, val_loader, device, epochs=args.epochs)
        print(f"Base Multi-class MLP Acc: {mc_mlp_acc:.4f}")

        # Binary (Target word vs Others)
        Y_bin = (Y == word_to_idx[args.target_word]).long()
        train_loader_bin = DataLoader(TensorDataset(X[train_idx], Y_bin[train_idx]), batch_size=16, shuffle=True)
        val_loader_bin = DataLoader(TensorDataset(X[val_idx], Y_bin[val_idx]), batch_size=16)
        
        print(f"Training Binary Linear Probe for '{args.target_word}'...")
        lp_bin = LinearProbe(d_model, 1).to(device)
        bin_acc = train_probe_loop(lp_bin, train_loader_bin, val_loader_bin, device, epochs=args.epochs, is_binary=True)
        print(f"Base Binary Linear Acc: {bin_acc:.4f}")
        probes[l]['lp_bin'] = lp_bin

    # 3. Transfer Evaluation to Fine-tuned Model (Using SAME conversation input)
    model_name = args.model_name.split("/")[-1]
    if "Qwen" in model_name:
        lora_path = f"adamkarvonen/Qwen3-8B-taboo-{args.target_word}_50_mix"
    elif "gemma" in model_name:
        lora_path = f"adamkarvonen/gemma-2-9b-it-taboo-{args.target_word}"
    print(f"\nTesting Transfer to FT Model: {lora_path}")
    try:
        ft_model = PeftModel.from_pretrained(base_model, lora_path)
        ft_model.eval()
        
        # Collect activations from FT model using the SAME conversations from the target word's dataset
        ft_acts_dict = collect_acts_for_word(ft_model, tokenizer, args.target_word, layers, device, args.num_samples)
        
        for l in layers:
            X_ft = torch.stack(ft_acts_dict[l]).to(device)
            # Evaluate using probes trained on Base Model
            with torch.no_grad():
                # Binary
                bin_logits = probes[l]['lp_bin'](X_ft)
                bin_preds = (torch.sigmoid(bin_logits.view(-1)) > 0.5).long()
                bin_ft_acc = (bin_preds == 1).float().mean().item()
                
                # Multi-class
                mc_logits = probes[l]['lp_mc'](X_ft)
                mc_preds = torch.argmax(mc_logits, dim=1)
                mc_ft_acc = (mc_preds == word_to_idx[args.target_word]).float().mean().item()
                
            print(f"Layer {l} FT Transfer Acc (Binary): {bin_ft_acc:.4f}")
            print(f"Layer {l} FT Transfer Acc (Multi-class): {mc_ft_acc:.4f}")
            
        base_model.unload() # Cleanup LoRA
    except Exception as e:
        print(f"FT Evaluation Error: {e}")

if __name__ == "__main__":
    main()
