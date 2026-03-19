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
                "similarity": float(cosine_matrix[row_idx, col_idx].item()),
            }
            for col_idx in sorted_indices
            if col_idx != row_idx
        ][:5]
    return neighbors


def mean_offdiag(matrix: torch.Tensor) -> float:
    off_diag_mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return float(matrix[off_diag_mask].mean().item())


def mean_diag(matrix: torch.Tensor) -> float:
    diag_mask = torch.eye(matrix.shape[0], dtype=torch.bool)
    return float(matrix[diag_mask].mean().item())


def make_prompt_split(num_prompts: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_prompts, generator=generator).tolist()
    num_train = int(num_prompts * train_fraction)
    return perm[:num_train], perm[num_train:]


def compute_prompt_centered_residual(tensor_TPD: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_means_PD = tensor_TPD.mean(dim=0)
    residual_TPD = tensor_TPD - prompt_means_PD.unsqueeze(0)
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


def load_global_deception_pc1_by_layer(
    summary_path: Path,
    variant_key: str,
    layer_percents: list[int],
) -> dict[int, torch.Tensor]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data["layers"]
    return {
        layer_percent: torch.tensor(layers[str(layer_percent)][variant_key]["pc1_direction"]).float()
        for layer_percent in layer_percents
    }


def build_orthonormal_basis_from_rows(rows_KD: torch.Tensor) -> torch.Tensor:
    q_DR, _ = torch.linalg.qr(rows_KD.T, mode="reduced")
    return q_DR.T


def analyze_local_pca_and_subspace(
    residual_TPD: torch.Tensor,
    target_words: list[str],
    local_num_components: int,
) -> tuple[dict[str, torch.Tensor], dict]:
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

    summary = {
        "num_components": local_num_components,
        "local_pc1_explained_variance_ratio_by_target": local_evr,
        "local_pc1_cosine_matrix": local_pc1_cosine.tolist(),
        "local_pc1_absolute_cosine_matrix": local_pc1_abs_cosine.tolist(),
        "local_pc1_top_neighbors": top_neighbors_from_cosine_matrix(target_words, local_pc1_abs_cosine),
        "local_subspace_similarity_matrix": local_subspace_similarity.tolist(),
        "local_subspace_top_neighbors": top_neighbors_from_cosine_matrix(target_words, local_subspace_similarity),
        "mean_local_pc1_absolute_cosine_off_diagonal": mean_offdiag(local_pc1_abs_cosine),
        "mean_local_subspace_similarity_off_diagonal": mean_offdiag(local_subspace_similarity),
    }
    return local_bases, summary


def analyze_cross_model_similarity(
    hider_bases: dict[str, torch.Tensor],
    guesser_bases: dict[str, torch.Tensor],
    target_words: list[str],
) -> dict:
    num_targets = len(target_words)
    pc1_cosine = torch.zeros(num_targets, num_targets)
    subspace_similarity = torch.zeros(num_targets, num_targets)

    for row_idx, hider_word in enumerate(target_words):
        for col_idx, guesser_word in enumerate(target_words):
            pc1_cosine[row_idx, col_idx] = torch.dot(
                F.normalize(hider_bases[hider_word][0], dim=0),
                F.normalize(guesser_bases[guesser_word][0], dim=0),
            )
            subspace_similarity[row_idx, col_idx] = compute_subspace_similarity(
                hider_bases[hider_word],
                guesser_bases[guesser_word],
            )

    pc1_abs = pc1_cosine.abs()
    return {
        "pc1_cosine_matrix": pc1_cosine.tolist(),
        "pc1_absolute_cosine_matrix": pc1_abs.tolist(),
        "pc1_absolute_diagonal_mean": mean_diag(pc1_abs),
        "pc1_absolute_off_diagonal_mean": mean_offdiag(pc1_abs),
        "pc1_absolute_diagonal_minus_off_diagonal": mean_diag(pc1_abs) - mean_offdiag(pc1_abs),
        "pc1_absolute_top_neighbors": top_neighbors_from_cosine_matrix(target_words, pc1_abs),
        "subspace_similarity_matrix": subspace_similarity.tolist(),
        "subspace_diagonal_mean": mean_diag(subspace_similarity),
        "subspace_off_diagonal_mean": mean_offdiag(subspace_similarity),
        "subspace_diagonal_minus_off_diagonal": mean_diag(subspace_similarity) - mean_offdiag(subspace_similarity),
        "subspace_top_neighbors": top_neighbors_from_cosine_matrix(target_words, subspace_similarity),
    }


def project_onto_subspace(vectors_PD: torch.Tensor, basis_KD: torch.Tensor) -> torch.Tensor:
    coefficients_PK = vectors_PD @ basis_KD.T
    return coefficients_PK @ basis_KD


def remove_same_word_guesser_subspace(
    hider_residual_TPD: torch.Tensor,
    guesser_residual_TPD: torch.Tensor,
    guesser_shared_bases: dict[str, torch.Tensor],
    target_words: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hider_specific = torch.zeros_like(hider_residual_TPD)
    guesser_after_projection = torch.zeros_like(guesser_residual_TPD)
    hider_removed = torch.zeros_like(hider_residual_TPD)
    guesser_removed = torch.zeros_like(guesser_residual_TPD)

    for target_idx, target_word in enumerate(target_words):
        basis_KD = guesser_shared_bases[target_word]
        hider_projection = project_onto_subspace(hider_residual_TPD[target_idx], basis_KD)
        guesser_projection = project_onto_subspace(guesser_residual_TPD[target_idx], basis_KD)

        hider_removed[target_idx] = hider_projection
        guesser_removed[target_idx] = guesser_projection
        hider_specific[target_idx] = hider_residual_TPD[target_idx] - hider_projection
        guesser_after_projection[target_idx] = guesser_residual_TPD[target_idx] - guesser_projection

    return hider_specific, guesser_after_projection, hider_removed, guesser_removed


def summarize_norms_by_target(tensor_TPD: torch.Tensor, target_words: list[str]) -> dict[str, dict[str, float]]:
    return {
        target_word: summarize_values(tensor_TPD[target_idx].norm(dim=-1))
        for target_idx, target_word in enumerate(target_words)
    }


def analyze_global_deception_alignment(
    local_bases: dict[str, torch.Tensor],
    hider_specific_TPD: torch.Tensor,
    target_words: list[str],
    global_deception_pc1_D: torch.Tensor,
) -> dict:
    global_deception_pc1_D = F.normalize(global_deception_pc1_D, dim=0)

    local_pc1_signed = {}
    local_pc1_abs = {}
    local_subspace_overlap = {}
    word_mean_signed = {}
    word_mean_abs = {}
    word_mean_norm = {}

    for target_idx, target_word in enumerate(target_words):
        local_pc1_D = F.normalize(local_bases[target_word][0], dim=0)
        signed_pc1_cosine = float(torch.dot(local_pc1_D, global_deception_pc1_D).item())
        local_pc1_signed[target_word] = signed_pc1_cosine
        local_pc1_abs[target_word] = abs(signed_pc1_cosine)

        projection_D = project_onto_subspace(global_deception_pc1_D.unsqueeze(0), local_bases[target_word]).squeeze(0)
        local_subspace_overlap[target_word] = float(projection_D.norm().item())

        word_mean_D = hider_specific_TPD[target_idx].mean(dim=0)
        word_mean_norm[target_word] = float(word_mean_D.norm().item())
        normalized_word_mean_D = F.normalize(word_mean_D, dim=0)
        signed_word_mean_cosine = float(torch.dot(normalized_word_mean_D, global_deception_pc1_D).item())
        word_mean_signed[target_word] = signed_word_mean_cosine
        word_mean_abs[target_word] = abs(signed_word_mean_cosine)

    local_pc1_abs_values = torch.tensor([local_pc1_abs[target] for target in target_words])
    local_subspace_overlap_values = torch.tensor([local_subspace_overlap[target] for target in target_words])
    word_mean_abs_values = torch.tensor([word_mean_abs[target] for target in target_words])
    word_mean_signed_values = torch.tensor([word_mean_signed[target] for target in target_words])

    return {
        "local_pc1_signed_cosine_by_target": local_pc1_signed,
        "local_pc1_absolute_cosine_by_target": local_pc1_abs,
        "local_pc1_absolute_cosine_stats": summarize_values(local_pc1_abs_values),
        "local_subspace_overlap_by_target": local_subspace_overlap,
        "local_subspace_overlap_stats": summarize_values(local_subspace_overlap_values),
        "word_mean_signed_cosine_by_target": word_mean_signed,
        "word_mean_absolute_cosine_by_target": word_mean_abs,
        "word_mean_signed_cosine_stats": summarize_values(word_mean_signed_values),
        "word_mean_absolute_cosine_stats": summarize_values(word_mean_abs_values),
        "word_mean_norm_by_target": word_mean_norm,
    }


def build_train_fitted_guesser_shared_bases(
    guesser_residual_TPD: torch.Tensor,
    target_words: list[str],
    train_prompt_indices: list[int],
    local_num_components: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    shared_bases = {}
    basis_summary = {}

    for target_idx, target_word in enumerate(target_words):
        guesser_train_PD = guesser_residual_TPD[target_idx, train_prompt_indices, :]
        word_mean_D = guesser_train_PD.mean(dim=0)
        local_basis_KD, local_evr_K = compute_local_pca(
            guesser_train_PD,
            num_components=local_num_components,
        )
        combined_rows = torch.cat(
            [
                F.normalize(word_mean_D, dim=0).unsqueeze(0),
                local_basis_KD,
            ],
            dim=0,
        )
        shared_basis_KD = build_orthonormal_basis_from_rows(combined_rows)
        shared_bases[target_word] = shared_basis_KD
        basis_summary[target_word] = {
            "word_mean_norm": float(word_mean_D.norm().item()),
            "local_explained_variance_ratio": local_evr_K.tolist(),
            "shared_basis_rank": int(shared_basis_KD.shape[0]),
        }

    return shared_bases, basis_summary


def build_word_probe_tensors(
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


def build_role_probe_tensors(
    hider_specific_TPD: torch.Tensor,
    guesser_after_projection_TPD: torch.Tensor,
    train_prompt_indices: list[int],
    test_prompt_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_targets, _, d_model = hider_specific_TPD.shape

    hider_train = hider_specific_TPD[:, train_prompt_indices, :].reshape(-1, d_model).float()
    guesser_train = guesser_after_projection_TPD[:, train_prompt_indices, :].reshape(-1, d_model).float()
    hider_test = hider_specific_TPD[:, test_prompt_indices, :].reshape(-1, d_model).float()
    guesser_test = guesser_after_projection_TPD[:, test_prompt_indices, :].reshape(-1, d_model).float()

    train_x = torch.cat([guesser_train, hider_train], dim=0)
    test_x = torch.cat([guesser_test, hider_test], dim=0)

    train_y = torch.cat(
        [
            torch.zeros(guesser_train.shape[0], dtype=torch.long),
            torch.ones(hider_train.shape[0], dtype=torch.long),
        ],
        dim=0,
    )
    test_y = torch.cat(
        [
            torch.zeros(guesser_test.shape[0], dtype=torch.long),
            torch.ones(hider_test.shape[0], dtype=torch.long),
        ],
        dim=0,
    )

    train_word_ids = torch.cat(
        [
            torch.arange(num_targets).unsqueeze(1).expand(num_targets, len(train_prompt_indices)).reshape(-1),
            torch.arange(num_targets).unsqueeze(1).expand(num_targets, len(train_prompt_indices)).reshape(-1),
        ],
        dim=0,
    ).long()
    test_word_ids = torch.cat(
        [
            torch.arange(num_targets).unsqueeze(1).expand(num_targets, len(test_prompt_indices)).reshape(-1),
            torch.arange(num_targets).unsqueeze(1).expand(num_targets, len(test_prompt_indices)).reshape(-1),
        ],
        dim=0,
    ).long()

    return train_x, train_y, train_word_ids, test_x, test_y, test_word_ids


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


def train_binary_probe(
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
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch).view(-1)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()


def evaluate_binary_probe(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    word_ids: torch.Tensor,
    target_words: list[str],
    device: torch.device,
    batch_size: int,
) -> dict:
    loader = DataLoader(TensorDataset(x, y, word_ids), batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds = []
    all_labels = []
    all_word_ids = []
    with torch.no_grad():
        for x_batch, y_batch, word_batch in loader:
            logits = model(x_batch.to(device)).view(-1)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu())
            all_word_ids.append(word_batch.cpu())

    preds_N = torch.cat(all_preds, dim=0)
    labels_N = torch.cat(all_labels, dim=0)
    word_ids_N = torch.cat(all_word_ids, dim=0)

    accuracy = float((preds_N == labels_N).float().mean().item())
    tp = int(((preds_N == 1) & (labels_N == 1)).sum().item())
    tn = int(((preds_N == 0) & (labels_N == 0)).sum().item())
    fp = int(((preds_N == 1) & (labels_N == 0)).sum().item())
    fn = int(((preds_N == 0) & (labels_N == 1)).sum().item())

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    confusion = [
        [tn, fp],
        [fn, tp],
    ]

    per_target_accuracy = {}
    for target_idx, target_word in enumerate(target_words):
        target_mask = word_ids_N == target_idx
        per_target_accuracy[target_word] = float((preds_N[target_mask] == labels_N[target_mask]).float().mean().item())

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion,
        "per_target_accuracy": per_target_accuracy,
    }


def analyze_word_probe(
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
    train_x, train_y, test_x, test_y = build_word_probe_tensors(
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
        "linear": {"train": linear_train_metrics, "test": linear_test_metrics},
        "mlp": {"train": mlp_train_metrics, "test": mlp_test_metrics},
    }


def analyze_role_probe(
    hider_specific_TPD: torch.Tensor,
    guesser_after_projection_TPD: torch.Tensor,
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
    train_x, train_y, train_word_ids, test_x, test_y, test_word_ids = build_role_probe_tensors(
        hider_specific_TPD=hider_specific_TPD,
        guesser_after_projection_TPD=guesser_after_projection_TPD,
        train_prompt_indices=train_prompt_indices,
        test_prompt_indices=test_prompt_indices,
    )

    d_model = hider_specific_TPD.shape[-1]

    linear_probe = LinearProbe(d_model, 1).to(device)
    train_binary_probe(
        model=linear_probe,
        train_x=train_x,
        train_y=train_y,
        device=device,
        lr=probe_lr,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
    )
    linear_train_metrics = evaluate_binary_probe(
        model=linear_probe,
        x=train_x,
        y=train_y,
        word_ids=train_word_ids,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )
    linear_test_metrics = evaluate_binary_probe(
        model=linear_probe,
        x=test_x,
        y=test_y,
        word_ids=test_word_ids,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )

    torch.manual_seed(seed)
    mlp_probe = MLPProbe(d_model, mlp_hidden_dim, 1).to(device)
    train_binary_probe(
        model=mlp_probe,
        train_x=train_x,
        train_y=train_y,
        device=device,
        lr=probe_lr,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
    )
    mlp_train_metrics = evaluate_binary_probe(
        model=mlp_probe,
        x=train_x,
        y=train_y,
        word_ids=train_word_ids,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )
    mlp_test_metrics = evaluate_binary_probe(
        model=mlp_probe,
        x=test_x,
        y=test_y,
        word_ids=test_word_ids,
        target_words=target_words,
        device=device,
        batch_size=probe_batch_size,
    )

    return {
        "train_prompt_indices": train_prompt_indices,
        "test_prompt_indices": test_prompt_indices,
        "linear": {"train": linear_train_metrics, "test": linear_test_metrics},
        "mlp": {"train": mlp_train_metrics, "test": mlp_test_metrics},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deception_pca_summary_path", type=str, default=None)
    parser.add_argument(
        "--deception_pca_variant",
        type=str,
        default="raw_centered_pca",
        choices=["raw_centered_pca", "unit_normalized_centered_pca"],
    )
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
    per_target_guesser_vectors = payload["per_target_guesser_vectors"]

    target_words = config["target_lora_suffixes"]
    layer_percents = config["layer_percents"]
    deception_pca_summary_path = (
        Path(args.deception_pca_summary_path)
        if args.deception_pca_summary_path is not None
        else input_dir / "hider_minus_guesser_pca_summary.json"
    )
    global_deception_pc1_by_layer = load_global_deception_pc1_by_layer(
        summary_path=deception_pca_summary_path,
        variant_key=args.deception_pca_variant,
        layer_percents=layer_percents,
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_layer_summaries = {}
    for layer_percent, act_layer in zip(layer_percents, act_layers):
        hider_TPD = torch.stack([per_target_hider_vectors[target_word][act_layer] for target_word in target_words], dim=0).float()
        guesser_TPD = torch.stack([per_target_guesser_vectors[target_word][act_layer] for target_word in target_words], dim=0).float()

        train_prompt_indices, test_prompt_indices = make_prompt_split(
            num_prompts=hider_TPD.shape[1],
            train_fraction=args.train_prompt_fraction,
            seed=args.seed,
        )

        _, hider_residual_TPD = compute_prompt_centered_residual(hider_TPD)
        _, guesser_residual_TPD = compute_prompt_centered_residual(guesser_TPD)

        hider_local_bases, hider_local_summary = analyze_local_pca_and_subspace(
            residual_TPD=hider_residual_TPD,
            target_words=target_words,
            local_num_components=args.local_num_components,
        )
        guesser_local_bases, guesser_local_summary = analyze_local_pca_and_subspace(
            residual_TPD=guesser_residual_TPD,
            target_words=target_words,
            local_num_components=args.local_num_components,
        )
        cross_model_summary = analyze_cross_model_similarity(
            hider_bases=hider_local_bases,
            guesser_bases=guesser_local_bases,
            target_words=target_words,
        )

        guesser_shared_bases, guesser_shared_basis_summary = build_train_fitted_guesser_shared_bases(
            guesser_residual_TPD=guesser_residual_TPD,
            target_words=target_words,
            train_prompt_indices=train_prompt_indices,
            local_num_components=args.local_num_components,
        )

        hider_specific_TPD, guesser_after_projection_TPD, hider_removed_TPD, guesser_removed_TPD = remove_same_word_guesser_subspace(
            hider_residual_TPD=hider_residual_TPD,
            guesser_residual_TPD=guesser_residual_TPD,
            guesser_shared_bases=guesser_shared_bases,
            target_words=target_words,
        )

        hider_specific_local_bases, hider_specific_local_summary = analyze_local_pca_and_subspace(
            residual_TPD=hider_specific_TPD,
            target_words=target_words,
            local_num_components=args.local_num_components,
        )
        global_deception_alignment = analyze_global_deception_alignment(
            local_bases=hider_specific_local_bases,
            hider_specific_TPD=hider_specific_TPD,
            target_words=target_words,
            global_deception_pc1_D=global_deception_pc1_by_layer[layer_percent],
        )
        word_probe_summary = analyze_word_probe(
            residual_TPD=hider_specific_TPD,
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
        role_probe_summary = analyze_role_probe(
            hider_specific_TPD=hider_specific_TPD,
            guesser_after_projection_TPD=guesser_after_projection_TPD,
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

        print(
            f"[Layer {layer_percent}%] H/G cross-model gap: "
            f"pc1_abs_gap={cross_model_summary['pc1_absolute_diagonal_minus_off_diagonal']:.4f}, "
            f"subspace_gap={cross_model_summary['subspace_diagonal_minus_off_diagonal']:.4f}"
        )
        print(
            f"[Layer {layer_percent}%] guesser shared basis fit: "
            f"train_prompts={len(train_prompt_indices)}, test_prompts={len(test_prompt_indices)}, "
            f"mean_shared_rank={sum(v['shared_basis_rank'] for v in guesser_shared_basis_summary.values()) / len(target_words):.2f}"
        )
        print(
            f"[Layer {layer_percent}%] hider-specific probes: "
            f"word_linear_test={word_probe_summary['linear']['test']['accuracy']:.4f}, "
            f"word_mlp_test={word_probe_summary['mlp']['test']['accuracy']:.4f}, "
            f"role_linear_test={role_probe_summary['linear']['test']['accuracy']:.4f}, "
            f"role_mlp_test={role_probe_summary['mlp']['test']['accuracy']:.4f}"
        )
        print(
            f"[Layer {layer_percent}%] deception alignment: "
            f"local_pc1_abs_mean={global_deception_alignment['local_pc1_absolute_cosine_stats']['mean']:.4f}, "
            f"subspace_overlap_mean={global_deception_alignment['local_subspace_overlap_stats']['mean']:.4f}, "
            f"word_mean_abs_mean={global_deception_alignment['word_mean_absolute_cosine_stats']['mean']:.4f}"
        )

        layer_summary = {
            "layer_percent": layer_percent,
            "layer_index": act_layer,
            "num_targets": int(hider_TPD.shape[0]),
            "num_prompts": int(hider_TPD.shape[1]),
            "prompt_centered_hider": {
                "norm_stats": summarize_values(hider_residual_TPD.norm(dim=-1).reshape(-1)),
                "norm_by_target": summarize_norms_by_target(hider_residual_TPD, target_words),
            },
            "prompt_centered_guesser": {
                "norm_stats": summarize_values(guesser_residual_TPD.norm(dim=-1).reshape(-1)),
                "norm_by_target": summarize_norms_by_target(guesser_residual_TPD, target_words),
            },
            "hider_local_pca": hider_local_summary,
            "guesser_local_pca": guesser_local_summary,
            "hider_vs_guesser_similarity": cross_model_summary,
            "train_fitted_guesser_shared_basis": {
                "train_prompt_indices": train_prompt_indices,
                "test_prompt_indices": test_prompt_indices,
                "by_target": guesser_shared_basis_summary,
            },
            "guesser_subspace_projection_removal": {
                "hider_specific_norm_stats": summarize_values(hider_specific_TPD.norm(dim=-1).reshape(-1)),
                "hider_specific_norm_by_target": summarize_norms_by_target(hider_specific_TPD, target_words),
                "guesser_after_projection_norm_stats": summarize_values(
                    guesser_after_projection_TPD.norm(dim=-1).reshape(-1)
                ),
                "guesser_after_projection_norm_by_target": summarize_norms_by_target(
                    guesser_after_projection_TPD, target_words
                ),
                "hider_removed_component_norm_stats": summarize_values(hider_removed_TPD.norm(dim=-1).reshape(-1)),
                "guesser_removed_component_norm_stats": summarize_values(
                    guesser_removed_TPD.norm(dim=-1).reshape(-1)
                ),
            },
            "hider_specific_residual": {
                "local_pca": hider_specific_local_summary,
                "global_deception_alignment": global_deception_alignment,
                "word_probe": word_probe_summary,
                "role_probe": role_probe_summary,
            },
        }
        all_layer_summaries[str(layer_percent)] = layer_summary

        write_matrix_csv(
            output_path=output_dir / f"hider_vs_guesser_pc1_absolute_cosine_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_vs_guesser_similarity"]["pc1_absolute_cosine_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_vs_guesser_subspace_similarity_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_vs_guesser_similarity"]["subspace_similarity_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_local_pc1_absolute_cosine_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_specific_residual"]["local_pca"]["local_pc1_absolute_cosine_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_local_subspace_similarity_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_specific_residual"]["local_pca"]["local_subspace_similarity_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_word_probe_linear_confusion_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_specific_residual"]["word_probe"]["linear"]["test"]["confusion_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_word_probe_mlp_confusion_layer_{layer_percent}.csv",
            row_labels=target_words,
            col_labels=target_words,
            matrix=layer_summary["hider_specific_residual"]["word_probe"]["mlp"]["test"]["confusion_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_role_probe_linear_confusion_layer_{layer_percent}.csv",
            row_labels=["guesser", "hider"],
            col_labels=["guesser", "hider"],
            matrix=layer_summary["hider_specific_residual"]["role_probe"]["linear"]["test"]["confusion_matrix"],
        )
        write_matrix_csv(
            output_path=output_dir / f"hider_specific_role_probe_mlp_confusion_layer_{layer_percent}.csv",
            row_labels=["guesser", "hider"],
            col_labels=["guesser", "hider"],
            matrix=layer_summary["hider_specific_residual"]["role_probe"]["mlp"]["test"]["confusion_matrix"],
        )
        with open(output_dir / f"hider_specific_vs_deception_metrics_layer_{layer_percent}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "target_word",
                    "local_pc1_signed_cosine",
                    "local_pc1_absolute_cosine",
                    "local_subspace_overlap",
                    "word_mean_signed_cosine",
                    "word_mean_absolute_cosine",
                    "word_mean_norm",
                ]
            )
            alignment = layer_summary["hider_specific_residual"]["global_deception_alignment"]
            for target_word in target_words:
                writer.writerow(
                    [
                        target_word,
                        alignment["local_pc1_signed_cosine_by_target"][target_word],
                        alignment["local_pc1_absolute_cosine_by_target"][target_word],
                        alignment["local_subspace_overlap_by_target"][target_word],
                        alignment["word_mean_signed_cosine_by_target"][target_word],
                        alignment["word_mean_absolute_cosine_by_target"][target_word],
                        alignment["word_mean_norm_by_target"][target_word],
                    ]
                )

    with open(output_dir / "hider_guesser_subspace_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "act_layers": act_layers,
                "analysis_config": {
                    "deception_pca_summary_path": str(deception_pca_summary_path),
                    "deception_pca_variant": args.deception_pca_variant,
                    "train_prompt_fraction": args.train_prompt_fraction,
                    "probe_lr": args.probe_lr,
                    "probe_epochs": args.probe_epochs,
                    "probe_batch_size": args.probe_batch_size,
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "local_num_components": args.local_num_components,
                    "guesser_shared_structure_basis": "train_prompt_word_mean_plus_local_subspace",
                    "seed": args.seed,
                },
                "layers": all_layer_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved hider-guesser subspace analysis to {output_dir}")


if __name__ == "__main__":
    main()
