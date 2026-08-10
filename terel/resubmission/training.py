import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import TemporalTensorDataset, encoder_batches
from .model import LayerLocalEncoder
from .objective import (
    LossCoefficients,
    direct_offdiagonal_covariance_loss,
    offline_soft_sfa_loss,
    regularized_target_components,
    regularized_target_residual,
    residual_lateral_dynamics,
    residual_lateral_offset_correction,
    temporal_references,
    terel_loss,
)


def augment_mnist_batch(features: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply deterministic per-example affine augmentation to normalized MNIST."""
    if features.ndim != 2 or features.shape[1] != 28 * 28:
        raise ValueError("MNIST augmentation expects flattened 28x28 images")
    generator = torch.Generator(device=features.device).manual_seed(int(seed))
    count = len(features)
    angles = (
        torch.rand(count, device=features.device, generator=generator) * 30.0 - 15.0
    )
    angles = torch.deg2rad(angles)
    scales = torch.rand(count, device=features.device, generator=generator) * 0.2 + 0.9
    translations = (
        torch.rand(count, 2, device=features.device, generator=generator) * 0.4 - 0.2
    )
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
    residual_lateral_delta_l2: tuple[float, ...]
    parameter_numel: int
    dynamic_state_numel: int
    causal_dynamic_state_numel: int
    auxiliary_parameter_numel: int
    peak_device_memory_bytes: int
    training_mode: str = "joint"
    epochs_per_layer: int = 0
    optimizer_steps: int = 0
    gradient_accumulation_steps: int = 1
    lateral_proxy_cosine_mean: tuple[float, ...] = ()
    lateral_proxy_relative_error_mean: tuple[float, ...] = ()
    lateral_proxy_norm_ratio_mean: tuple[float, ...] = ()
    lateral_proxy_audited_batches: tuple[int, ...] = ()
    residual_state_rms_mean: tuple[float, ...] = ()
    base_residual_state_rms_mean: tuple[float, ...] = ()
    residual_dynamics_delta_rms_mean: tuple[float, ...] = ()
    temporal_state_rms_mean: tuple[float, ...] = ()
    variance_state_rms_mean: tuple[float, ...] = ()
    covariance_state_rms_mean: tuple[float, ...] = ()


@dataclass(frozen=True)
class OfflineTrainingSummary:
    epochs: int
    steps: int
    examples: int
    seconds: float
    final_loss: float
    valid_temporal_pairs: int
    layer_parameter_delta_l2: tuple[float, ...]
    peak_device_memory_bytes: int
    training_mode: str = "end_to_end_subsequence"


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
    relative_error = (
        torch.linalg.vector_norm(proxy_direction - direct_direction) / direct_norm
    )
    return {
        "valid": True,
        "cosine_alignment": float(cosine.clamp(-1.0, 1.0)),
        "relative_error": float(relative_error),
        "norm_ratio": float(proxy_norm / direct_norm),
    }


def postsynaptic_learning_state(
    *,
    preactivation: torch.Tensor,
    activation: torch.Tensor,
    activation_residual: torch.Tensor,
    mode: str,
    activation_kind: str | None = None,
) -> torch.Tensor:
    """Map a target residual to the state multiplying presynaptic activity."""
    if activation_kind == "relu":
        exact = (
            activation_residual * (preactivation > 0).to(activation_residual.dtype)
        ).detach()
    elif activation_kind is None:
        exact = torch.autograd.grad(
            activation,
            preactivation,
            grad_outputs=activation_residual,
            retain_graph=True,
        )[0].detach()
    else:
        raise ValueError(f"Unsupported explicit activation derivative '{activation_kind}'")
    if mode == "exact":
        return exact
    if mode == "rectified":
        return F.relu(exact).detach()
    raise ValueError("postsynaptic_state_mode must be 'exact' or 'rectified'")


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


def train_offline_encoder(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    dataset: TemporalTensorDataset,
    epochs: int,
    batch_size: int,
    order_mode: str,
    order_seed: int,
    chunk_size: int,
    coefficients: LossCoefficients,
    variance_target: float,
    device: torch.device,
) -> OfflineTrainingSummary:
    """Train TeReL-Offline on full subsequence graphs and a final-layer loss."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if batch_size < 2:
        raise ValueError("TeReL-Offline subsequences require batch_size at least two")
    model.to(device)
    before = [
        tuple(parameter.detach().cpu().clone() for parameter in layer.parameters())
        for layer in model.layers
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    steps = 0
    examples = 0
    valid_pairs = 0
    final_loss = float("nan")
    for epoch in range(epochs):
        for features, boundaries in encoder_batches(
            dataset,
            batch_size=batch_size,
            order_mode=order_mode,
            seed=order_seed + epoch,
            chunk_size=chunk_size,
        ):
            metrics = offline_train_step(
                model=model,
                optimizer=optimizer,
                x=features.to(device),
                boundaries=boundaries.to(device),
                coefficients=coefficients,
                variance_target=variance_target,
            )
            steps += 1
            examples += len(features)
            valid_pairs += int(metrics["valid_temporal_pairs"])
            final_loss = metrics["loss"]
    deltas = []
    for layer, old_parameters in zip(model.layers, before, strict=True):
        squared = sum(
            float((parameter.detach().cpu() - old).square().sum())
            for parameter, old in zip(layer.parameters(), old_parameters, strict=True)
        )
        deltas.append(squared**0.5)
    return OfflineTrainingSummary(
        epochs=int(epochs),
        steps=steps,
        examples=examples,
        seconds=time.perf_counter() - start_time,
        final_loss=final_loss,
        valid_temporal_pairs=valid_pairs,
        layer_parameter_delta_l2=tuple(deltas),
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )


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
    audit_residual_components: bool = False,
    residual_lateral_steps: int = 1,
    residual_lateral_step_size: float = 0.1,
    residual_lateral_rule: str = "dual_inhibitory",
    residual_lateral_include_diagonal: bool = True,
    residual_lateral_moment_normalization: str = "none",
    residual_lateral_coefficient: float = 0.5,
    residual_lateral_signal_offset: int = 0,
    postsynaptic_state_mode: str = "exact",
    lateral_matrix_mode: str = "two_matrix",
    combined_lateral_state_weight: float = 0.5,
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
    residual_lateral_before = [
        None
        if state.residual_lateral is None
        else state.residual_lateral.detach().cpu().clone()
        for state in model.states
    ]
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
    residual_state_rms = [[] for _ in model.layers]
    base_residual_state_rms = [[] for _ in model.layers]
    residual_dynamics_delta_rms = [[] for _ in model.layers]
    component_state_rms = {
        name: [[] for _ in model.layers]
        for name in ("temporal", "variance", "covariance")
    }
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    stages = range(len(model.layers)) if training_mode == "greedy" else (None,)
    for active_layer in stages:
        for epoch in range(epochs):
            stage_offset = 0 if active_layer is None else active_layer * epochs
            for batch_index, (features, boundaries) in enumerate(
                encoder_batches(
                    dataset,
                    batch_size=batch_size,
                    order_mode=order_mode,
                    seed=order_seed + stage_offset + epoch,
                    chunk_size=chunk_size,
                )
            ):
                features = features.to(device, non_blocking=True)
                boundaries = boundaries.to(device, non_blocking=True)
                features = _augment(
                    features,
                    mode=augmentation,
                    seed=order_seed + stage_offset * 100_000 + epoch * 10_000 + steps,
                )
                group_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_stop = min(
                    group_start + gradient_accumulation_steps, batches_per_epoch
                )
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
                    audit_residual_components=audit_residual_components,
                    residual_lateral_steps=residual_lateral_steps,
                    residual_lateral_step_size=residual_lateral_step_size,
                    residual_lateral_rule=residual_lateral_rule,
                    residual_lateral_include_diagonal=residual_lateral_include_diagonal,
                    residual_lateral_moment_normalization=(
                        residual_lateral_moment_normalization
                    ),
                    residual_lateral_coefficient=residual_lateral_coefficient,
                    residual_lateral_signal_offset=residual_lateral_signal_offset,
                    postsynaptic_state_mode=postsynaptic_state_mode,
                    lateral_matrix_mode=lateral_matrix_mode,
                    combined_lateral_state_weight=combined_lateral_state_weight,
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
                    residual_prefix = f"layer_{layer_index}/"
                    for name, collection in (
                        ("residual_state_rms", residual_state_rms),
                        ("base_residual_state_rms", base_residual_state_rms),
                        ("residual_dynamics_delta_rms", residual_dynamics_delta_rms),
                    ):
                        if residual_prefix + name in metrics:
                            collection[layer_index].append(
                                metrics[residual_prefix + name]
                            )
                    for name, collection in component_state_rms.items():
                        metric_name = residual_prefix + name + "_state_rms"
                        if metric_name in metrics:
                            collection[layer_index].append(metrics[metric_name])
                steps += 1
                examples += len(features)
                loss_sum += metrics["loss"]
                optimizer_steps += int(take_step)
    seconds = time.perf_counter() - start
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
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
        residual_lateral_delta_l2=tuple(
            0.0
            if state.residual_lateral is None
            else float(
                (
                    state.residual_lateral.detach().cpu()
                    - (
                        old
                        if old is not None
                        else torch.zeros_like(state.residual_lateral, device="cpu")
                    )
                )
                .square()
                .sum()
                .sqrt()
            )
            for state, old in zip(model.states, residual_lateral_before, strict=True)
        ),
        parameter_numel=sum(
            parameter.numel() for parameter in model.encoder_parameters()
        ),
        dynamic_state_numel=sum(state.dynamic_state_numel() for state in model.states),
        causal_dynamic_state_numel=sum(
            state.causal_dynamic_state_numel() for state in model.states
        ),
        auxiliary_parameter_numel=sum(
            state.auxiliary_parameter_numel() for state in model.states
        ),
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
        residual_state_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in residual_state_rms
        ),
        base_residual_state_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in base_residual_state_rms
        ),
        residual_dynamics_delta_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in residual_dynamics_delta_rms
        ),
        temporal_state_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in component_state_rms["temporal"]
        ),
        variance_state_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in component_state_rms["variance"]
        ),
        covariance_state_rms_mean=tuple(
            sum(values) / len(values) if values else 0.0
            for values in component_state_rms["covariance"]
        ),
    )


def offline_train_step(
    *,
    model: LayerLocalEncoder,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    boundaries: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
) -> dict[str, float]:
    """Backpropagate a final-layer soft-SFA loss through a full subsequence."""
    if getattr(model, "normalization", "none") != "none":
        raise ValueError("TeReL-Offline does not use normalization layers")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    representation = model(x)
    loss, metrics = offline_soft_sfa_loss(
        representation,
        boundaries=boundaries,
        coefficients=coefficients,
        variance_target=variance_target,
    )
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        **{name: float(value) for name, value in metrics.items()},
    }


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
    audit_residual_components: bool = False,
    residual_lateral_steps: int = 1,
    residual_lateral_step_size: float = 0.1,
    residual_lateral_rule: str = "dual_inhibitory",
    residual_lateral_include_diagonal: bool = True,
    residual_lateral_moment_normalization: str = "none",
    residual_lateral_coefficient: float = 0.5,
    residual_lateral_signal_offset: int = 0,
    postsynaptic_state_mode: str = "exact",
    lateral_matrix_mode: str = "two_matrix",
    combined_lateral_state_weight: float = 0.5,
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
    residual_mode = covariance_mode == "residual_state"
    if residual_mode and not detach_previous:
        raise ValueError("residual-state TeReL requires detached temporal references")
    if residual_mode and model.normalization not in {"none", "streaming_norm"}:
        raise ValueError(
            "residual-state TeReL requires identity or streaming normalization"
        )
    if (
        residual_mode
        and model.normalization == "streaming_norm"
        and any(
            parameter.requires_grad
            for normalization in model.normalizations
            for parameter in normalization.parameters()
        )
    ):
        raise ValueError(
            "residual-state TeReL requires fixed-affine streaming normalization"
        )
    if residual_lateral_moment_normalization not in {"none", "features"}:
        raise ValueError(
            "residual_lateral_moment_normalization must be 'none' or 'features'"
        )
    if residual_lateral_signal_offset not in {0, 1}:
        raise ValueError("residual_lateral_signal_offset must be zero or one")
    if lateral_matrix_mode not in {
        "two_matrix",
        "representation_shared",
        "state_shared",
        "combined",
    }:
        raise ValueError("unsupported lateral_matrix_mode")
    if not 0.0 <= combined_lateral_state_weight <= 1.0:
        raise ValueError("combined_lateral_state_weight must lie in [0, 1]")
    if not residual_mode and lateral_matrix_mode != "two_matrix":
        raise ValueError("one-matrix modes are defined only for residual-state TeReL")
    layer_details = model.forward_local_details(x, stop_after=active_layer)
    losses = []
    layer_metrics = []
    state_updates = []
    if active_layer is None:
        selected = zip(layer_details, model.states, strict=True)
    else:
        selected = ((layer_details[-1], model.states[active_layer]),)
    selected = tuple(selected)
    for (preactivation, _, z), state in selected:
        previous, valid = temporal_references(
            z,
            state=state,
            boundaries=boundaries,
            detach=detach_previous,
        )
        if covariance_mode not in {
            "proxy",
            "shifted_proxy",
            "direct",
            "residual_state",
        }:
            raise ValueError(f"Unsupported covariance_mode '{covariance_mode}'")
        if residual_mode:
            if residual_lateral_rule != "dual_inhibitory":
                raise ValueError(
                    "residual-state TeReL requires the dual inhibitory rule"
                )
            lateral_reference = None
            target_lateral = state.lateral - torch.diag_embed(
                torch.diagonal(state.lateral)
            )
            if residual_lateral_signal_offset == 1:
                if z.shape[0] != 1:
                    raise ValueError(
                        "equal-offset residual TeReL requires batch size one"
                    )
                centered = z.detach() - state.mean.detach()
                lateral_reference = torch.zeros_like(centered)
                if bool(valid[0]):
                    lateral_reference[0] = state.ensure_previous_centered()
                state.ensure_previous_centered()
            _, activation_residual = regularized_target_residual(
                z=z,
                previous=previous,
                mean=state.mean,
                variance=state.variance,
                lateral=target_lateral,
                pair_valid=valid,
                coefficients=coefficients,
                variance_target=variance_target,
                lateral_reference=lateral_reference,
            )
            base_neuron_state = postsynaptic_learning_state(
                preactivation=preactivation,
                activation=z,
                activation_residual=activation_residual,
                mode=postsynaptic_state_mode,
                activation_kind=(
                    "relu" if model.activation_name == "relu" else None
                ),
            )
            component_states = {}
            if audit_residual_components:
                components = regularized_target_components(
                    z=z,
                    previous=previous,
                    mean=state.mean,
                    variance=state.variance,
                    lateral=target_lateral,
                    pair_valid=valid,
                    coefficients=coefficients,
                    variance_target=variance_target,
                    lateral_reference=lateral_reference,
                )
                component_states = {
                    name: torch.autograd.grad(
                        z,
                        preactivation,
                        grad_outputs=component,
                        retain_graph=True,
                    )[0].detach()
                    for name, component in components.items()
                }
            if residual_lateral_signal_offset == 0:
                inhibition_matrix = (
                    state.ensure_residual_lateral()
                    if lateral_matrix_mode == "two_matrix"
                    else state.lateral
                )
                neuron_state = residual_lateral_dynamics(
                    base_state=base_neuron_state,
                    lateral=inhibition_matrix,
                    coefficient=residual_lateral_coefficient,
                    steps=residual_lateral_steps,
                    step_size=residual_lateral_step_size,
                )
            else:
                if residual_lateral_steps != 1:
                    raise ValueError(
                        "equal-offset residual TeReL uses exactly one correction"
                    )
                previous_neuron_state = torch.zeros_like(base_neuron_state)
                if bool(valid[0]):
                    previous_neuron_state[0] = state.ensure_previous_neuron_state()
                state.ensure_previous_neuron_state()
                neuron_state = residual_lateral_offset_correction(
                    base_state=base_neuron_state,
                    previous_state=previous_neuron_state,
                    lateral=(
                        state.ensure_residual_lateral()
                        if lateral_matrix_mode == "two_matrix"
                        else state.lateral
                    ),
                    coefficient=residual_lateral_coefficient,
                    step_size=residual_lateral_step_size,
                    pair_valid=valid,
                )
            target = (preactivation.detach() - neuron_state).detach()
            loss = (
                coefficients.similarity
                / (z.shape[0] * z.shape[1])
                * (preactivation - target).square().sum()
            )
            _, metrics = terel_loss(
                z=z,
                previous=previous,
                mean=state.mean,
                variance=state.variance,
                lateral=target_lateral,
                pair_valid=valid,
                coefficients=coefficients,
                variance_target=variance_target,
                detach_previous=True,
            )
            metrics["residual_state_rms"] = neuron_state.square().mean().sqrt()
            metrics["base_residual_state_rms"] = (
                base_neuron_state.square().mean().sqrt()
            )
            metrics["residual_dynamics_delta_rms"] = (
                (base_neuron_state - neuron_state).square().mean().sqrt()
            )
            for name, component_state in component_states.items():
                metrics[name + "_state_rms"] = component_state.square().mean().sqrt()
            losses.append(loss)
            layer_metrics.append(metrics)
            state_updates.append((z, state, neuron_state))
            continue
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
                lateral_reference[0] = state.ensure_previous_centered()
            if z.shape[0] > 1:
                lateral_reference[1:] = centered[:-1].detach()
            lateral_reference[boundaries] = 0.0
            state.ensure_previous_centered()
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
                    "lateral_proxy_cosine_alignment": diagnostics["cosine_alignment"],
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
        state_updates.append((z, state, None))

    total = torch.stack(losses).sum()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("TeReL objective became non-finite")
    (total * loss_scale).backward()
    if optimizer_step:
        optimizer.step()
    for z, state, residual_lateral_values in state_updates:
        moment_scale = (
            1.0 / z.shape[1]
            if residual_lateral_moment_normalization == "features"
            else 1.0
        )
        shared_lateral_moment = None
        if residual_lateral_values is not None and lateral_matrix_mode in {
            "state_shared",
            "combined",
        }:
            values = residual_lateral_values.detach()
            state_moment = moment_scale * values.T @ values / values.shape[0]
            if not residual_lateral_include_diagonal:
                state_moment.fill_diagonal_(0.0)
            if lateral_matrix_mode == "state_shared":
                shared_lateral_moment = state_moment
            else:
                centered = z.detach() - state.mean.detach()
                representation_moment = centered.T @ centered / z.shape[0]
                representation_moment.fill_diagonal_(0.0)
                weight = combined_lateral_state_weight
                shared_lateral_moment = (
                    1.0 - weight
                ) * representation_moment + weight * state_moment
        state.update(
            z,
            update_lateral=lateral_matrix_mode
            in {"two_matrix", "representation_shared"},
        )
        if residual_lateral_values is not None:
            if lateral_matrix_mode == "two_matrix":
                state.update_residual_lateral(
                    residual_lateral_values,
                    include_diagonal=residual_lateral_include_diagonal,
                    moment_scale=moment_scale,
                )
            elif shared_lateral_moment is not None:
                state.update_lateral_moment(shared_lateral_moment)
            if residual_lateral_signal_offset == 1:
                state.update_previous_neuron_state(residual_lateral_values)

    result = {"loss": float(total.detach())}
    metric_indices = (
        range(len(layer_metrics)) if active_layer is None else (active_layer,)
    )
    for index, metrics in zip(metric_indices, layer_metrics, strict=True):
        for name, value in metrics.items():
            result[f"layer_{index}/{name}"] = float(value)
    return result
