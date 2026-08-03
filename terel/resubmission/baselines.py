from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import TemporalTensorDataset, class_chunk_order
from .incsfa import IncrementalLinearSFA
from .model import LayerLocalEncoder


class BatchLinearSFA:
    """Linear Slow Feature Analysis via whitening and derivative eigendecomposition."""

    def __init__(self, n_components: int, epsilon: float = 1e-10):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.epsilon = float(epsilon)
        self.mean_ = None
        self.components_ = None

    def fit(self, x, *, boundaries=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("x must contain at least two samples in a two-dimensional array")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        threshold = self.epsilon * max(1.0, float(eigenvalues.max()))
        keep = eigenvalues > threshold
        if int(keep.sum()) < self.n_components:
            raise ValueError(
                f"Requested {self.n_components} components but input covariance rank is {int(keep.sum())}"
            )
        whitening = eigenvectors[:, keep] @ np.diag(eigenvalues[keep] ** -0.5)
        whitened = centered @ whitening
        derivatives = np.diff(whitened, axis=0)
        if boundaries is not None:
            boundaries = np.asarray(boundaries, dtype=bool)
            if boundaries.shape != (len(x),):
                raise ValueError("boundaries must contain one flag per input row")
            derivatives = derivatives[~boundaries[1:]]
        if len(derivatives) == 0:
            raise ValueError("SFA requires at least one valid temporal pair")
        self.derivative_pair_count_ = int(len(derivatives))
        derivative_covariance = derivatives.T @ derivatives / len(derivatives)
        _, slow_directions = np.linalg.eigh(derivative_covariance)
        slow_directions = slow_directions[:, : self.n_components]
        self.components_ = whitening @ slow_directions
        return self

    def transform(self, x):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("BatchLinearSFA must be fitted before transform")
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self.components_

    def fit_transform(self, x, *, boundaries=None):
        return self.fit(x, boundaries=boundaries).transform(x)


class SupervisedMLP(nn.Module):
    """End-to-end baseline built directly from the declared hidden dimensions."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = (input_dim, *hidden_dims)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dims[:-1], dims[1:], strict=True)
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def representations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        current = x
        for layer in self.hidden_layers:
            current = self.activation(layer(current))
            activations.append(current)
        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.representations(x)[-1])


def supervised_contrastive_loss(representations: torch.Tensor, labels: torch.Tensor, *, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(representations, dim=1)
    logits = normalized @ normalized.T / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        raise ValueError("each supervised contrastive batch needs at least one positive pair")
    denominator_logits = logits.masked_fill(self_mask, -torch.inf)
    log_probability = logits - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    positive_log_probability = (
        log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return -positive_log_probability[valid_anchor].mean()


def local_supervised_contrastive_step(
    *,
    model: LayerLocalEncoder,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    layer_losses = [
        supervised_contrastive_loss(z, labels, temperature=temperature)
        for z in model.forward_local(features)
    ]
    loss = torch.stack(layer_losses).sum()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        **{f"layer_{index}/loss": float(value.detach()) for index, value in enumerate(layer_losses)},
    }


@dataclass(frozen=True)
class BaselineTrainingSummary:
    epochs: int
    steps: int
    examples: int
    seconds: float
    final_loss: float
    layer_parameter_delta_l2: tuple[float, ...]
    peak_device_memory_bytes: int


def train_supervised_mlp(
    *,
    model: SupervisedMLP,
    optimizer: torch.optim.Optimizer,
    dataset: TemporalTensorDataset,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
):
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = [
        tuple(parameter.detach().cpu().clone() for parameter in layer.parameters())
        for layer in model.hidden_layers
    ]
    features = dataset.features.to(device)
    labels = dataset.labels.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start_time = time.perf_counter()
    steps = 0
    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(dataset), generator=generator)
        for start in range(0, len(dataset), batch_size):
            index = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features[index]), labels[index])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            steps += 1
    deltas = []
    for layer, old_parameters in zip(model.hidden_layers, before, strict=True):
        squared = sum(
            float((parameter.detach().cpu() - old).square().sum())
            for parameter, old in zip(layer.parameters(), old_parameters, strict=True)
        )
        deltas.append(squared**0.5)
    return BaselineTrainingSummary(
        epochs=int(epochs),
        steps=steps,
        examples=int(epochs * len(dataset)),
        seconds=time.perf_counter() - start_time,
        final_loss=final_loss,
        layer_parameter_delta_l2=tuple(deltas),
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )


def train_local_supervised_contrastive(
    *,
    model: LayerLocalEncoder,
    optimizer: torch.optim.Optimizer,
    dataset: TemporalTensorDataset,
    epochs: int,
    batch_size: int,
    seed: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
):
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if chunk_size < 2 or batch_size % chunk_size:
        raise ValueError("chunk_size must be at least two and divide batch_size")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = [
        tuple(parameter.detach().cpu().clone() for parameter in layer.parameters())
        for layer in model.layers
    ]
    start_time = time.perf_counter()
    steps = 0
    final_loss = float("nan")
    labels_numpy = dataset.labels.detach().cpu().numpy()
    for epoch in range(epochs):
        order, _ = class_chunk_order(labels_numpy, chunk_size=chunk_size, seed=seed + epoch)
        index = torch.from_numpy(order)
        for start in range(0, len(dataset), batch_size):
            batch_index = index[start : start + batch_size]
            metrics = local_supervised_contrastive_step(
                model=model,
                optimizer=optimizer,
                features=dataset.features[batch_index].to(device),
                labels=dataset.labels[batch_index].to(device),
                temperature=temperature,
            )
            final_loss = metrics["loss"]
            steps += 1
    deltas = []
    for layer, old_parameters in zip(model.layers, before, strict=True):
        squared = sum(
            float((parameter.detach().cpu() - old).square().sum())
            for parameter, old in zip(layer.parameters(), old_parameters, strict=True)
        )
        deltas.append(squared**0.5)
    return BaselineTrainingSummary(
        epochs=int(epochs),
        steps=steps,
        examples=int(epochs * len(dataset)),
        seconds=time.perf_counter() - start_time,
        final_loss=final_loss,
        layer_parameter_delta_l2=tuple(deltas),
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )
