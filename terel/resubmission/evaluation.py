from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from .data import TemporalTensorDataset


@dataclass(frozen=True)
class ProbeTrainingSummary:
    epochs: int
    steps: int
    examples: int
    seconds: float
    final_loss: float


@torch.no_grad()
def extract_representations(
    model: nn.Module,
    dataset: TemporalTensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    use_all_layers: bool = False,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.to(device)
    model.eval()
    batches = []
    for start in range(0, len(dataset), batch_size):
        features = dataset.features[start : start + batch_size].to(device)
        if use_all_layers:
            representations = model(features, return_all=True)
            representations = torch.cat(representations, dim=1)
        else:
            representations = model(features)
        batches.append(representations.detach().cpu())
    return torch.cat(batches, dim=0)


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
):
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be [samples, dimensions] and align with labels")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    probe = nn.Linear(features.shape[1], num_classes).to(device)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            probe.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown probe optimizer: {optimizer_name}")

    features = features.to(device)
    labels = labels.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start_time = time.perf_counter()
    steps = 0
    final_loss = float("nan")
    probe.train()
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            index = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(features[index]), labels[index])
            loss.backward()
            optimizer.step()
            steps += 1
            final_loss = float(loss.detach())
    seconds = time.perf_counter() - start_time
    probe.eval()
    return probe, ProbeTrainingSummary(
        epochs=int(epochs),
        steps=steps,
        examples=int(epochs * len(features)),
        seconds=seconds,
        final_loss=final_loss,
    )


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, *, num_classes: int):
    predicted = logits.detach().cpu().argmax(dim=1).numpy()
    expected = labels.detach().cpu().numpy()
    label_space = np.arange(num_classes)
    matrix = confusion_matrix(expected, predicted, labels=label_space)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(expected, predicted, labels=label_space, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(recalls.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def representation_diagnostics(representations: torch.Tensor, boundaries: torch.Tensor):
    representations = representations.detach().to(torch.float64).cpu()
    boundaries = boundaries.detach().cpu()
    if representations.ndim != 2 or boundaries.shape != (len(representations),):
        raise ValueError("representations and boundaries have incompatible shapes")
    variance = representations.var(dim=0, unbiased=False)
    centered = representations - representations.mean(dim=0)
    scale = centered.square().mean(dim=0).sqrt()
    standardized = torch.where(scale > 0, centered / scale.clamp_min(1e-12), torch.zeros_like(centered))
    correlation = standardized.T @ standardized / len(standardized)
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    mean_offdiagonal = float(correlation[mask].abs().mean()) if mask.any() else 0.0

    if len(representations) > 1:
        valid = ~boundaries[1:]
        squared_change = (representations[1:] - representations[:-1]).square().mean(dim=1)
        slowness = float(squared_change[valid].mean()) if valid.any() else float("nan")
    else:
        slowness = float("nan")
    return {
        "median_feature_variance": float(variance.median()),
        "mean_feature_variance": float(variance.mean()),
        "mean_absolute_offdiagonal_correlation": mean_offdiagonal,
        "temporal_slowness": slowness,
    }
