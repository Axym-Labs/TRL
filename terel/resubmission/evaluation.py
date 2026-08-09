import time
from dataclasses import dataclass

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


@dataclass(frozen=True)
class NormalizationCalibrationSummary:
    passes: int
    batches: int
    examples: int
    seconds: float


@torch.no_grad()
def calibrate_batch_normalization(model, dataset, *, batch_size, passes, device):
    """Calibrate BatchNorm buffers without updating encoder parameters."""
    if passes <= 0:
        raise ValueError("calibration passes must be positive")
    if batch_size <= 1:
        raise ValueError("BatchNorm calibration batch_size must exceed one")
    normalizations = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm1d)
    ]
    if not normalizations:
        raise ValueError("BatchNorm calibration requires BatchNorm modules")

    model.to(device)
    model.train()
    start_time = time.perf_counter()
    batches = 0
    for _ in range(passes):
        for start in range(0, len(dataset), batch_size):
            features = dataset.features[start : start + batch_size].to(device)
            if len(features) <= 1:
                raise ValueError("every BatchNorm calibration batch must contain at least two examples")
            model(features)
            batches += 1
    seconds = time.perf_counter() - start_time
    model.eval()
    return NormalizationCalibrationSummary(
        passes=int(passes),
        batches=batches,
        examples=int(passes * len(dataset)),
        seconds=seconds,
    )


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
    if not torch.isfinite(representations).all():
        raise ValueError("representations contain non-finite values")
    variance = representations.var(dim=0, unbiased=False)
    centered = representations - representations.mean(dim=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    effective_rank = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-24)
    )
    active_feature_fraction = float((variance > 1e-4).to(torch.float64).mean())
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
        "effective_rank": effective_rank,
        "active_feature_fraction": active_feature_fraction,
        "mean_absolute_offdiagonal_correlation": mean_offdiagonal,
        "temporal_slowness": slowness,
    }


def class_structure_diagnostics(
    representations: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
):
    """Label-aware post-hoc structure metrics; never used by encoder training."""
    representations = representations.detach().to(torch.float64).cpu()
    labels = labels.detach().to(torch.long).cpu()
    if representations.ndim != 2 or labels.shape != (len(representations),):
        raise ValueError("representations and labels have incompatible shapes")
    present = [class_id for class_id in range(num_classes) if (labels == class_id).any()]
    if len(present) < 2:
        raise ValueError("class diagnostics require at least two observed classes")
    centroids = torch.stack([representations[labels == class_id].mean(dim=0) for class_id in present])
    global_mean = representations.mean(dim=0)
    within = torch.zeros((), dtype=torch.float64)
    between = torch.zeros((), dtype=torch.float64)
    for index, class_id in enumerate(present):
        class_values = representations[labels == class_id]
        within += (class_values - centroids[index]).square().sum()
        between += len(class_values) * (centroids[index] - global_mean).square().sum()
    within /= len(representations)
    between /= len(representations)

    distances = torch.cdist(representations, centroids)
    predicted_indices = distances.argmin(dim=1)
    present_tensor = torch.as_tensor(present, dtype=torch.long)
    predicted = present_tensor[predicted_indices]

    normalized_centroids = F.normalize(centroids, dim=1)
    prototype_cosines = normalized_centroids @ normalized_centroids.T
    offdiagonal = ~torch.eye(len(present), dtype=torch.bool)

    class_means = centroids
    feature_scale = representations.std(dim=0, unbiased=False).clamp_min(1e-12)
    selectivity = (class_means.max(dim=0).values - class_means.min(dim=0).values) / feature_scale
    return {
        "observed_classes": present,
        "between_class_scatter": float(between),
        "within_class_scatter": float(within),
        "between_within_scatter_ratio": float(between / within.clamp_min(1e-12)),
        "nearest_centroid_accuracy": float((predicted == labels).to(torch.float64).mean()),
        "mean_prototype_cosine": float(prototype_cosines[offdiagonal].mean()),
        "median_unit_class_selectivity": float(selectivity.median()),
        "p90_unit_class_selectivity": float(torch.quantile(selectivity, 0.9)),
    }
