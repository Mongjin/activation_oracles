import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def summarize_values(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "median": float(values.median().item()),
    }


def write_matrix_csv(
    output_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *col_labels])
        for row_label, row in zip(row_labels, matrix):
            writer.writerow([row_label, *row])


def compute_cosine_matrix(vectors_XD: torch.Tensor) -> torch.Tensor:
    normalized_XD = F.normalize(vectors_XD.float(), dim=-1)
    return normalized_XD @ normalized_XD.T


def top_neighbors_from_cosine_matrix(target_words: list[str], cosine_matrix: torch.Tensor) -> dict[str, list[dict[str, float]]]:
    neighbors = {}
    for row_idx, target_word in enumerate(target_words):
        sorted_indices = torch.argsort(cosine_matrix[row_idx], descending=True).tolist()
        neighbors[target_word] = [
            {
                "target_word": target_words[col_idx],
                "cosine_similarity": float(cosine_matrix[row_idx, col_idx].item()),
            }
            for col_idx in sorted_indices
            if col_idx != row_idx
        ][:5]
    return neighbors


def compute_pairwise_offdiag_values(vectors_XD: torch.Tensor) -> torch.Tensor:
    cosine_matrix = compute_cosine_matrix(vectors_XD)
    off_diag_mask = ~torch.eye(cosine_matrix.shape[0], dtype=torch.bool)
    return cosine_matrix[off_diag_mask]


def make_prompt_split(num_prompts: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_prompts, generator=generator).tolist()
    num_train = int(num_prompts * train_fraction)
    return perm[:num_train], perm[num_train:]


def build_probe_tensors(
    residual_TPD: torch.Tensor,
    train_prompt_indices: list[int],
    test_prompt_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_targets, _, d_model = residual_TPD.shape

    train_x = residual_TPD[:, train_prompt_indices, :].reshape(-1, d_model).float()
    test_x = residual_TPD[:, test_prompt_indices, :].reshape(-1, d_model).float()

    train_y = (
        torch.arange(num_targets)
        .unsqueeze(1)
        .expand(num_targets, len(train_prompt_indices))
        .reshape(-1)
        .long()
    )
    test_y = (
        torch.arange(num_targets)
        .unsqueeze(1)
        .expand(num_targets, len(test_prompt_indices))
        .reshape(-1)
        .long()
    )
    return train_x, train_y, test_x, test_y


def train_multiclass_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
) -> None:
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_multiclass_probe(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    target_words: list[str],
    device: torch.device,
    batch_size: int,
) -> dict:
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu())

    preds_N = torch.cat(all_preds, dim=0)
    labels_N = torch.cat(all_labels, dim=0)
    accuracy = float((preds_N == labels_N).float().mean().item())

    num_targets = len(target_words)
    confusion = torch.zeros(num_targets, num_targets, dtype=torch.int64)
    for label, pred in zip(labels_N.tolist(), preds_N.tolist()):
        confusion[label, pred] += 1

    per_target_accuracy = {}
    for target_idx, target_word in enumerate(target_words):
        target_mask = labels_N == target_idx
        per_target_accuracy[target_word] = float((preds_N[target_mask] == labels_N[target_mask]).float().mean().item())

    return {
        "accuracy": accuracy,
        "per_target_accuracy": per_target_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def compute_prompt_centered_residual(hider_TPD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_means_PD = hider_TPD.mean(dim=0)
    residual_TPD = hider_TPD - prompt_means_PD.unsqueeze(0)
    return prompt_means_PD, residual_TPD


def compute_local_pca(vectors_PD: torch.Tensor, num_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    centered_PD = vectors_PD - vectors_PD.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered_PD, full_matrices=False)
    num_components = min(num_components, vh.shape[0])
    basis_KD = vh[:num_components]
    explained_variance_ratio = singular_values.square() / singular_values.square().sum()
    return basis_KD, explained_variance_ratio[:num_components]


def compute_subspace_similarity(basis_i_KD: torch.Tensor, basis_j_KD: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(basis_i_KD @ basis_j_KD.T)
    return float(singular_values.square().mean().item())


def analyze_target_word_probe(
    residual_TPD: torch.Tensor,
    target_words: list[str],
    train_prompt_indices: list[int],
    test_prompt_indices: list[int],
    device: torch.device,
    probe_lr: float,
    probe_epochs: int,
    probe_batch_size: int,
    mlp_hidden_dim: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    train_x, train_y, test_x, test_y = build_probe_tensors(
        residual_TPD=residual_TPD,
        train_prompt_indices=train_prompt_indices,
        test_prompt_indices=test_prompt_indices,
    )

    d_model = residual_TPD.shape[-1]
    num_targets = len(target_words)

    linear_probe = LinearProbe(d_model, num_targets).to(device)
    train_multiclass_probe(
        model=linear_probe,
        train_x=train_x,
        train_y=train_y,
        device=device,
        lr=probe_lr,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
    )
    linear_train_metrics = evaluate_multiclass_probe(
        model=linear_probe,
        x=train_x,
        y=train_y,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )
    linear_test_metrics = evaluate_multiclass_probe(
        model=linear_probe,
        x=test_x,
        y=test_y,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )

    torch.manual_seed(seed)
    mlp_probe = MLPProbe(d_model, mlp_hidden_dim, num_targets).to(device)
    train_multiclass_probe(
        model=mlp_probe,
        train_x=train_x,
        train_y=train_y,
        device=device,
        lr=probe_lr,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
    )
    mlp_train_metrics = evaluate_multiclass_probe(
        model=mlp_probe,
        x=train_x,
        y=train_y,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )
    mlp_test_metrics = evaluate_multiclass_probe(
        model=mlp_probe,
        x=test_x,
        y=test_y,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )

    return {
        "train_prompt_indices": train_prompt_indices,
        "test_prompt_indices": test_prompt_indices,
        "linear": {
            "train": linear_train_metrics,
            "test": linear_test_metrics,
        },
        "mlp": {
            "train": mlp_train_metrics,
            "test": mlp_test_metrics,
        },
    }


def analyze_similarity(
    residual_TPD: torch.Tensor,
    target_words: list[str],
) -> dict:
    num_targets, num_prompts, _ = residual_TPD.shape

    same_word_values = []
    same_word_by_target = {}
    for target_idx, target_word in enumerate(target_words):
        values = compute_pairwise_offdiag_values(residual_TPD[target_idx])
        same_word_values.append(values)
        same_word_by_target[target_word] = summarize_values(values)

    different_word_values = []
    different_word_by_prompt = {}
    for prompt_idx in range(num_prompts):
        values = compute_pairwise_offdiag_values(residual_TPD[:, prompt_idx, :])
        different_word_values.append(values)
        different_word_by_prompt[str(prompt_idx)] = summarize_values(values)

    same_word_values = torch.cat(same_word_values, dim=0)
    different_word_values = torch.cat(different_word_values, dim=0)

    word_mean_TD = residual_TPD.mean(dim=1)
    word_mean_cosine = compute_cosine_matrix(word_mean_TD)

    return {
        "same_word_similarity": {
            "global_stats": summarize_values(same_word_values),
            "by_target": same_word_by_target,
        },
        "different_word_same_prompt_similarity": {
            "global_stats": summarize_values(different_word_values),
            "by_prompt_index": different_word_by_prompt,
        },
        "same_minus_different_mean": float(same_word_values.mean().item() - different_word_values.mean().item()),
        "word_mean_cosine_matrix": word_mean_cosine.tolist(),
        "word_mean_top_neighbors": top_neighbors_from_cosine_matrix(target_words, word_mean_cosine),
    }


def analyze_local_pca_and_subspace(
    residual_TPD: torch.Tensor,
    target_words: list[str],
    local_num_components: int,
) -> dict:
    local_bases = {}
    local_evr = {}
    for target_idx, target_word in enumerate(target_words):
        basis_KD, explained_variance_ratio_K = compute_local_pca(
            residual_TPD[target_idx],
            num_components=local_num_components,
        )
        local_bases[target_word] = basis_KD
        local_evr[target_word] = explained_variance_ratio_K.tolist()

    num_targets = len(target_words)
    local_pc1_cosine = torch.zeros(num_targets, num_targets)
    local_subspace_similarity = torch.zeros(num_targets, num_targets)
    for row_idx, row_word in enumerate(target_words):
        for col_idx, col_word in enumerate(target_words):
            local_pc1_cosine[row_idx, col_idx] = torch.dot(
                F.normalize(local_bases[row_word][0], dim=0),
                F.normalize(local_bases[col_word][0], dim=0),
            )
            local_subspace_similarity[row_idx, col_idx] = compute_subspace_similarity(
                local_bases[row_word],
                local_bases[col_word],
            )

    local_pc1_abs_cosine = local_pc1_cosine.abs()

    return {
        "num_components": local_num_components,
        "local_pc1_explained_variance_ratio_by_target": local_evr,
        "local_pc1_cosine_matrix": local_pc1_cosine.tolist(),
        "local_pc1_absolute_cosine_matrix": local_pc1_abs_cosine.tolist(),
        "local_pc1_top_neighbors": top_neighbors_from_cosine_matrix(target_words, local_pc1_abs_cosine),
        "local_subspace_similarity_matrix": local_subspace_similarity.tolist(),
        "local_subspace_top_neighbors": top_neighbors_from_cosine_matrix(target_words, local_subspace_similarity),
    }


def analyze_two_way_decomposition(
    hider_TPD: torch.Tensor,
    word_mean_cosine_matrix: torch.Tensor,
    target_words: list[str],
) -> dict:
    num_targets, num_prompts, _ = hider_TPD.shape

    mu_D = hider_TPD.mean(dim=(0, 1))
    prompt_effect_PD = hider_TPD.mean(dim=0) - mu_D.unsqueeze(0)
    word_effect_TD = hider_TPD.mean(dim=1) - mu_D.unsqueeze(0)
    interaction_TPD = (
        hider_TPD
        - mu_D.view(1, 1, -1)
        - prompt_effect_PD.unsqueeze(0)
        - word_effect_TD.unsqueeze(1)
    )

    total_centered_TPD = hider_TPD - mu_D.view(1, 1, -1)
    total_sum_squares = total_centered_TPD.square().sum()
    prompt_sum_squares = num_targets * prompt_effect_PD.square().sum()
    word_sum_squares = num_prompts * word_effect_TD.square().sum()
    interaction_sum_squares = interaction_TPD.square().sum()

    prompt_norms_P = prompt_effect_PD.norm(dim=-1)
    word_norms_T = word_effect_TD.norm(dim=-1)
    interaction_norms_TP = interaction_TPD.norm(dim=-1)
    word_effect_cosine_matrix = compute_cosine_matrix(word_effect_TD)

    return {
        "mu_norm": float(mu_D.norm().item()),
        "total_sum_squares": float(total_sum_squares.item()),
        "component_sum_squares": {
            "prompt": float(prompt_sum_squares.item()),
            "word": float(word_sum_squares.item()),
            "interaction": float(interaction_sum_squares.item()),
        },
        "variance_fraction_by_component": {
            "prompt": float((prompt_sum_squares / total_sum_squares).item()),
            "word": float((word_sum_squares / total_sum_squares).item()),
            "interaction": float((interaction_sum_squares / total_sum_squares).item()),
        },
        "prompt_effect_norm_stats": summarize_values(prompt_norms_P),
        "prompt_effect_norm_by_prompt_index": {
            str(prompt_idx): float(prompt_norms_P[prompt_idx].item()) for prompt_idx in range(num_prompts)
        },
        "word_effect_norm_stats": summarize_values(word_norms_T),
        "word_effect_norm_by_target": {
            target_word: float(word_norms_T[target_idx].item())
            for target_idx, target_word in enumerate(target_words)
        },
        "interaction_norm_stats": summarize_values(interaction_norms_TP.reshape(-1)),
        "interaction_norm_by_target": {
            target_word: summarize_values(interaction_norms_TP[target_idx])
            for target_idx, target_word in enumerate(target_words)
        },
        "word_effect_cosine_matrix": word_effect_cosine_matrix.tolist(),
        "word_effect_top_neighbors": top_neighbors_from_cosine_matrix(target_words, word_effect_cosine_matrix),
        "word_mean_vs_word_effect_cosine_max_abs_diff": float(
            (word_mean_cosine_matrix - word_effect_cosine_matrix).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_prompt_fraction", type=float, default=0.8)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=25)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--local_num_components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    input_dir = Path(args.input_dir)
    payload = torch.load(input_dir / "pooled_hider_activations.pt", map_location="cpu")

    config = payload["config"]
    act_layers = payload["act_layers"]
    per_target_hider_vectors = payload["per_target_hider_vectors"]

    target_words = config["target_lora_suffixes"]
    layer_percents = config["layer_percents"]

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_layer_summaries = {}
    for layer_percent, act_layer in zip(layer_percents, act_layers):
        hider_TPD = torch.stack([per_target_hider_vectors[target_word][act_layer] for target_word in target_words], dim=0).float()
        num_targets, num_prompts, _ = hider_TPD.shape

        prompt_means_PD, residual_TPD = compute_prompt_centered_residual(hider_TPD)
        train_prompt_indices, test_prompt_indices = make_prompt_split(
            num_prompts=num_prompts,
            train_fraction=args.train_prompt_fraction,
            seed=args.seed,
        )

        target_word_probe = analyze_target_word_probe(
            residual_TPD=residual_TPD,
            target_words=target_words,
            train_prompt_indices=train_prompt_indices,
            test_prompt_indices=test_prompt_indices,
            device=device,
            probe_lr=args.probe_lr,
            probe_epochs=args.probe_epochs,
            probe_batch_size=args.probe_batch_size,
            mlp_hidden_dim=args.mlp_hidden_dim,
            seed=args.seed,
        )
        similarity_summary = analyze_similarity(
            residual_TPD=residual_TPD,
            target_words=target_words,
        )
        word_mean_cosine_matrix = torch.tensor(similarity_summary["word_mean_cosine_matrix"])
        local_pca_summary = analyze_local_pca_and_subspace(
            residual_TPD=residual_TPD,
            target_words=target_words,
            local_num_components=args.local_num_components,
        )
        two_way_summary = analyze_two_way_decomposition(
            hider_TPD=hider_TPD,
            word_mean_cosine_matrix=word_mean_cosine_matrix,
            target_words=target_words,
        )

        layer_summary = {
            "layer_percent": layer_percent,
            "layer_index": act_layer,
            "num_targets": num_targets,
            "num_prompts": num_prompts,
            "prompt_centered_residual": {
                "prompt_mean_norm_stats": summarize_values(prompt_means_PD.norm(dim=-1)),
                "residual_norm_stats": summarize_values(residual_TPD.norm(dim=-1).reshape(-1)),
                "residual_norm_by_target": {
                    target_word: summarize_values(residual_TPD[target_idx].norm(dim=-1))
                    for target_idx, target_word in enumerate(target_words)
                },
            },
            "target_word_probe": target_word_probe,
            "same_word_similarity": similarity_summary["same_word_similarity"],
            "different_word_same_prompt_similarity": similarity_summary["different_word_same_prompt_similarity"],
            "same_minus_different_mean": similarity_summary["same_minus_different_mean"],
            "word_mean_cosine_matrix": similarity_summary["word_mean_cosine_matrix"],
            "word_mean_top_neighbors": similarity_summary["word_mean_top_neighbors"],
            "local_pca": local_pca_summary,
            "two_way_decomposition": two_way_summary,
        }
        all_layer_summaries[str(layer_percent)] = layer_summary

        write_matrix_csv(
            output_path=output_dir / f"prompt_centered_word_mean_cosine_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["word_mean_cosine_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"local_pc1_absolute_cosine_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["local_pca"]["local_pc1_absolute_cosine_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"local_subspace_similarity_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["local_pca"]["local_subspace_similarity_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"target_word_probe_linear_confusion_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["target_word_probe"]["linear"]["test"]["confusion_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"target_word_probe_mlp_confusion_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["target_word_probe"]["mlp"]["test"]["confusion_matrix"],
        )

    with open(output_dir / "prompt_centered_hider_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "act_layers": act_layers,
                "analysis_config": {
                    "train_prompt_fraction": args.train_prompt_fraction,
                    "probe_lr": args.probe_lr,
                    "probe_epochs": args.probe_epochs,
                    "probe_batch_size": args.probe_batch_size,
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "local_num_components": args.local_num_components,
                    "seed": args.seed,
                },
                "layers": all_layer_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved prompt-centered hider analysis to {output_dir}")


if __name__ == "__main__":
    main()
