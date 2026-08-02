from dataclasses import dataclass
import time

import torch

from .data import TemporalTensorDataset, encoder_batches
from .model import LayerLocalEncoder
from .objective import (
    LossCoefficients,
    direct_offdiagonal_covariance_loss,
    temporal_references,
    terel_loss,
)


@dataclass(frozen=True)
class EncoderTrainingSummary:
    epochs: int
    steps: int
    examples: int
    seconds: float
    mean_loss: float
    layer_parameter_delta_l2: tuple[float, ...]
    layer_lateral_delta_l2: tuple[float, ...]
    parameter_numel: int
    dynamic_state_numel: int
    peak_device_memory_bytes: int


def _layer_snapshots(model: LayerLocalEncoder):
    return [
        tuple(parameter.detach().cpu().clone() for parameter in layer.parameters())
        for layer in model.layers
    ]


def _layer_deltas(model: LayerLocalEncoder, before):
    deltas = []
    for layer, old_parameters in zip(model.layers, before, strict=True):
        squared = 0.0
        for parameter, old in zip(layer.parameters(), old_parameters, strict=True):
            difference = parameter.detach().cpu() - old
            squared += float(difference.square().sum())
        deltas.append(squared**0.5)
    return tuple(deltas)


def train_local_encoder(
    *,
    model: LayerLocalEncoder,
    optimizer: torch.optim.Optimizer,
    dataset: TemporalTensorDataset,
    epochs: int,
    batch_size: int,
    order_mode: str,
    order_seed: int,
    chunk_size: int,
    coefficients: LossCoefficients,
    variance_target: float,
    detach_previous: bool,
    covariance_mode: str,
    device: torch.device,
) -> EncoderTrainingSummary:
    """Train a TeReL encoder and return the fidelity/compute audit record."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    model.to(device)
    before = _layer_snapshots(model)
    lateral_before = [state.lateral.detach().cpu().clone() for state in model.states]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    steps = 0
    examples = 0
    loss_sum = 0.0
    for epoch in range(epochs):
        for features, boundaries in encoder_batches(
            dataset,
            batch_size=batch_size,
            order_mode=order_mode,
            seed=order_seed + epoch,
            chunk_size=chunk_size,
        ):
            features = features.to(device, non_blocking=True)
            boundaries = boundaries.to(device, non_blocking=True)
            metrics = local_train_step(
                model=model,
                optimizer=optimizer,
                x=features,
                boundaries=boundaries,
                coefficients=coefficients,
                variance_target=variance_target,
                detach_previous=detach_previous,
                covariance_mode=covariance_mode,
            )
            steps += 1
            examples += len(features)
            loss_sum += metrics["loss"]
    seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return EncoderTrainingSummary(
        epochs=int(epochs),
        steps=steps,
        examples=examples,
        seconds=seconds,
        mean_loss=loss_sum / steps,
        layer_parameter_delta_l2=_layer_deltas(model, before),
        layer_lateral_delta_l2=tuple(
            float((state.lateral.detach().cpu() - old).square().sum().sqrt())
            for state, old in zip(model.states, lateral_before, strict=True)
        ),
        parameter_numel=sum(parameter.numel() for parameter in model.encoder_parameters()),
        dynamic_state_numel=sum(state.dynamic_state_numel() for state in model.states),
        peak_device_memory_bytes=int(peak_memory),
    )


def local_train_step(
    *,
    model: LayerLocalEncoder,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    boundaries: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
    detach_previous: bool,
    covariance_mode: str = "proxy",
) -> dict[str, float]:
    """Apply one optimizer step using independent per-layer TeReL losses."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    layer_activations = model.forward_local(x)
    losses = []
    layer_metrics = []
    for z, state in zip(layer_activations, model.states, strict=True):
        previous, valid = temporal_references(
            z,
            state=state,
            boundaries=boundaries,
            detach=detach_previous,
        )
        if covariance_mode not in {"proxy", "shifted_proxy", "direct"}:
            raise ValueError(f"Unsupported covariance_mode '{covariance_mode}'")
        objective_coefficients = coefficients
        if covariance_mode == "direct":
            objective_coefficients = LossCoefficients(
                similarity=coefficients.similarity,
                variance=coefficients.variance,
                covariance=0.0,
            )
        lateral_reference = None
        if covariance_mode == "shifted_proxy":
            centered = z - state.mean
            lateral_reference = torch.zeros_like(centered)
            if state.has_previous and not boundaries[0]:
                lateral_reference[0] = state.previous_centered
            if z.shape[0] > 1:
                lateral_reference[1:] = centered[:-1].detach()
            lateral_reference[boundaries] = 0.0
        loss, metrics = terel_loss(
            z=z,
            previous=previous,
            mean=state.mean,
            variance=state.variance,
            lateral=state.lateral,
            pair_valid=valid,
            coefficients=objective_coefficients,
            variance_target=variance_target,
            detach_previous=detach_previous,
            lateral_reference=lateral_reference,
        )
        if covariance_mode == "direct":
            covariance_loss = direct_offdiagonal_covariance_loss(z, mean=state.mean)
            loss = loss + coefficients.covariance * covariance_loss
            metrics["covariance_loss"] = covariance_loss.detach()
        losses.append(loss)
        layer_metrics.append(metrics)

    total = torch.stack(losses).sum()
    total.backward()
    optimizer.step()
    for z, state in zip(layer_activations, model.states, strict=True):
        state.update(z)

    result = {"loss": float(total.detach())}
    for index, metrics in enumerate(layer_metrics):
        for name, value in metrics.items():
            result[f"layer_{index}/{name}"] = float(value)
    return result
