from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F

from .data import TemporalTensorDataset, encoder_batches
from .model import LayerLocalEncoder
from .objective import (
    LossCoefficients,
    direct_offdiagonal_covariance_loss,
    temporal_references,
    terel_loss,
)


def augment_mnist_batch(features: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply deterministic per-example affine augmentation to normalized MNIST."""
    if features.ndim != 2 or features.shape[1] != 28 * 28:
        raise ValueError("MNIST augmentation expects flattened 28x28 images")
    generator = torch.Generator(device=features.device).manual_seed(int(seed))
    count = len(features)
    angles = (torch.rand(count, device=features.device, generator=generator) * 30.0 - 15.0)
    angles = torch.deg2rad(angles)
    scales = torch.rand(count, device=features.device, generator=generator) * 0.2 + 0.9
    translations = torch.rand(
        count, 2, device=features.device, generator=generator
    ) * 0.4 - 0.2
    cosine = torch.cos(angles) / scales
    sine = torch.sin(angles) / scales
    theta = torch.zeros(count, 2, 3, dtype=features.dtype, device=features.device)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = sine
    theta[:, 1, 0] = -sine
    theta[:, 1, 1] = cosine
    theta[:, :, 2] = translations
    images = (features * 0.3081 + 0.1307).reshape(count, 1, 28, 28)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    augmented = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return ((augmented.reshape(count, -1) - 0.1307) / 0.3081).detach()


def _augment(features: torch.Tensor, *, mode: str, seed: int) -> torch.Tensor:
    if mode == "none":
        return features
    if mode == "mnist_affine":
        return augment_mnist_batch(features, seed=seed)
    raise ValueError(f"Unsupported augmentation '{mode}'")


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
    training_mode: str = "joint"
    epochs_per_layer: int = 0
    optimizer_steps: int = 0
    gradient_accumulation_steps: int = 1
    lateral_proxy_cosine_mean: tuple[float, ...] = ()
    lateral_proxy_relative_error_mean: tuple[float, ...] = ()
    lateral_proxy_norm_ratio_mean: tuple[float, ...] = ()
    lateral_proxy_audited_batches: tuple[int, ...] = ()


@torch.no_grad()
def lateral_proxy_diagnostics(
    z: torch.Tensor,
    *,
    mean: torch.Tensor,
    lateral: torch.Tensor,
    epsilon: float = 1e-12,
) -> dict[str, float | bool]:
    """Compare the lagged lateral direction with the same-batch direction."""
    centered = z.detach() - mean.detach()
    current = centered.T @ centered / len(centered)
    current.fill_diagonal_(0.0)
    proxy_direction = centered @ lateral.detach().T
    direct_direction = centered @ current.T
    proxy_norm = torch.linalg.vector_norm(proxy_direction)
    direct_norm = torch.linalg.vector_norm(direct_direction)
    valid = bool(proxy_norm > epsilon and direct_norm > epsilon)
    if not valid:
        return {
            "valid": False,
            "cosine_alignment": 0.0,
            "relative_error": 0.0,
            "norm_ratio": 0.0,
        }
    cosine = torch.sum(proxy_direction * direct_direction) / (proxy_norm * direct_norm)
    relative_error = torch.linalg.vector_norm(proxy_direction - direct_direction) / direct_norm
    return {
        "valid": True,
        "cosine_alignment": float(cosine.clamp(-1.0, 1.0)),
        "relative_error": float(relative_error),
        "norm_ratio": float(proxy_norm / direct_norm),
    }


def _layer_snapshots(model: LayerLocalEncoder):
    return [
        tuple(
            parameter.detach().cpu().clone()
            for parameter in (*layer.parameters(), *normalization.parameters())
        )
        for layer, normalization in zip(model.layers, model.normalizations, strict=True)
    ]


def _layer_deltas(model: LayerLocalEncoder, before):
    deltas = []
    for layer, normalization, old_parameters in zip(
        model.layers, model.normalizations, before, strict=True
    ):
        squared = 0.0
        parameters = (*layer.parameters(), *normalization.parameters())
        for parameter, old in zip(parameters, old_parameters, strict=True):
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
    training_mode: str = "joint",
    augmentation: str = "none",
    gradient_accumulation_steps: int = 1,
    audit_lateral_proxy: bool = False,
) -> EncoderTrainingSummary:
    """Train a TeReL encoder and return the fidelity/compute audit record."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if training_mode not in {"joint", "greedy"}:
        raise ValueError("training_mode must be 'joint' or 'greedy'")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.to(device)
    before = _layer_snapshots(model)
    lateral_before = [state.lateral.detach().cpu().clone() for state in model.states]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    steps = 0
    examples = 0
    loss_sum = 0.0
    optimizer_steps = 0
    proxy_cosines = [[] for _ in model.layers]
    proxy_relative_errors = [[] for _ in model.layers]
    proxy_norm_ratios = [[] for _ in model.layers]
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    stages = range(len(model.layers)) if training_mode == "greedy" else (None,)
    for active_layer in stages:
        for epoch in range(epochs):
            stage_offset = 0 if active_layer is None else active_layer * epochs
            for batch_index, (features, boundaries) in enumerate(encoder_batches(
                dataset,
                batch_size=batch_size,
                order_mode=order_mode,
                seed=order_seed + stage_offset + epoch,
                chunk_size=chunk_size,
            )):
                features = features.to(device, non_blocking=True)
                features = _augment(
                    features,
                    mode=augmentation,
                    seed=order_seed + stage_offset * 100_000 + epoch * 10_000 + steps,
                )
                boundaries = boundaries.to(device, non_blocking=True)
                group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
                group_stop = min(group_start + gradient_accumulation_steps, batches_per_epoch)
                group_size = group_stop - group_start
                take_step = batch_index + 1 == group_stop
                metrics = local_train_step(
                    model=model,
                    optimizer=optimizer,
                    x=features,
                    boundaries=boundaries,
                    coefficients=coefficients,
                    variance_target=variance_target,
                    detach_previous=detach_previous,
                    covariance_mode=covariance_mode,
                    active_layer=active_layer,
                    zero_grad=batch_index == group_start,
                    optimizer_step=take_step,
                    loss_scale=1.0 / group_size,
                    audit_lateral_proxy=audit_lateral_proxy,
                )
                for layer_index in range(len(model.layers)):
                    prefix = f"layer_{layer_index}/lateral_proxy_"
                    if metrics.get(prefix + "valid", 0.0) > 0.5:
                        proxy_cosines[layer_index].append(
                            metrics[prefix + "cosine_alignment"]
                        )
                        proxy_relative_errors[layer_index].append(
                            metrics[prefix + "relative_error"]
                        )
                        proxy_norm_ratios[layer_index].append(
                            metrics[prefix + "norm_ratio"]
                        )
                steps += 1
                examples += len(features)
                loss_sum += metrics["loss"]
                optimizer_steps += int(take_step)
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
        training_mode=training_mode,
        epochs_per_layer=int(epochs),
        optimizer_steps=int(optimizer_steps),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        lateral_proxy_cosine_mean=tuple(
            sum(values) / len(values) if values else 0.0 for values in proxy_cosines
        ),
        lateral_proxy_relative_error_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in proxy_relative_errors
        ),
        lateral_proxy_norm_ratio_mean=tuple(
            sum(values) / len(values) if values else 0.0 for values in proxy_norm_ratios
        ),
        lateral_proxy_audited_batches=tuple(len(values) for values in proxy_cosines),
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
    active_layer: int | None = None,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
    audit_lateral_proxy: bool = False,
) -> dict[str, float]:
    """Apply one optimizer step using independent per-layer TeReL losses."""
    model.train()
    if active_layer is not None:
        if not 0 <= active_layer < len(model.layers):
            raise ValueError("active_layer is out of range")
        for index, normalization in enumerate(model.normalizations):
            normalization.train(index == active_layer)
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    layer_activations = model.forward_local(x, stop_after=active_layer)
    losses = []
    layer_metrics = []
    if active_layer is None:
        selected = zip(layer_activations, model.states, strict=True)
    else:
        selected = ((layer_activations[-1], model.states[active_layer]),)
    selected = tuple(selected)
    for z, state in selected:
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
        if audit_lateral_proxy and covariance_mode == "proxy":
            diagnostics = lateral_proxy_diagnostics(
                z,
                mean=state.mean,
                lateral=state.lateral,
            )
            metrics.update(
                {
                    "lateral_proxy_valid": float(diagnostics["valid"]),
                    "lateral_proxy_cosine_alignment": diagnostics[
                        "cosine_alignment"
                    ],
                    "lateral_proxy_relative_error": diagnostics["relative_error"],
                    "lateral_proxy_norm_ratio": diagnostics["norm_ratio"],
                }
            )
        if covariance_mode == "direct":
            covariance_loss = direct_offdiagonal_covariance_loss(z, mean=state.mean)
            loss = loss + coefficients.covariance * covariance_loss
            metrics["covariance_loss"] = covariance_loss.detach()
        losses.append(loss)
        layer_metrics.append(metrics)

    total = torch.stack(losses).sum()
    (total * loss_scale).backward()
    if optimizer_step:
        optimizer.step()
    for z, state in selected:
        state.update(z)

    result = {"loss": float(total.detach())}
    metric_indices = range(len(layer_metrics)) if active_layer is None else (active_layer,)
    for index, metrics in zip(metric_indices, layer_metrics, strict=True):
        for name, value in metrics.items():
            result[f"layer_{index}/{name}"] = float(value)
    return result
