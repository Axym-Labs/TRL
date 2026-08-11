from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossCoefficients:
    similarity: float = 1.0
    variance: float = 2.5
    covariance: float = 1.0


def offline_soft_sfa_loss(
    z: torch.Tensor,
    *,
    boundaries: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
):
    """Soft-SFA objective evaluated directly on a complete subsequence.

    Unlike the local rule, all moments and adjacent differences remain in the
    autograd graph.  This objective is therefore reserved for TeReL-Offline.
    """
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("z must be a non-empty [samples, features] tensor")
    if boundaries.shape != (len(z),) or boundaries.dtype != torch.bool:
        raise ValueError("boundaries must be one boolean flag per sample")
    if variance_target <= 0.0:
        raise ValueError("variance_target must be positive")

    if len(z) > 1:
        valid = ~boundaries[1:]
        pair_losses = (z[1:] - z[:-1]).square().mean(dim=1)
        similarity_loss = pair_losses[valid].mean() if valid.any() else z.sum() * 0.0
        valid_pairs = valid.sum()
    else:
        similarity_loss = z.sum() * 0.0
        valid_pairs = torch.zeros((), dtype=torch.long, device=z.device)

    centered = z - z.mean(dim=0)
    variance = centered.square().mean(dim=0)
    variance_deficit = F.relu(
        torch.as_tensor(variance_target, dtype=z.dtype, device=z.device) - variance
    )
    variance_loss = variance_deficit.square().mean()
    covariance = centered.T @ centered / len(centered)
    offdiagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = offdiagonal.square().sum() / z.shape[1]
    loss = (
        coefficients.similarity * similarity_loss
        + coefficients.variance * variance_loss
        + coefficients.covariance * covariance_loss
    )
    return loss, {
        "similarity_loss": similarity_loss.detach(),
        "variance_loss": variance_loss.detach(),
        "covariance_loss": covariance_loss.detach(),
        "mean_variance": variance.detach().mean(),
        "valid_temporal_pairs": valid_pairs.detach(),
    }


def lateral_proxy_error_bound(
    *,
    lateral: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    reference: torch.Tensor,
):
    """Return proxy error and a triangle/operator-norm upper bound.

    For lateral operator ``A``, target off-diagonal moment operator ``C``,
    current activation ``u_t``, and possibly shifted reference ``q_t``,
    ``A q_t - C u_t = (A-C)u_t + A(q_t-u_t)``.
    """
    actual = torch.linalg.vector_norm(lateral @ reference - target @ current)
    bound = torch.linalg.matrix_norm(
        lateral - target, ord=2
    ) * torch.linalg.vector_norm(current) + torch.linalg.matrix_norm(
        lateral, ord=2
    ) * torch.linalg.vector_norm(reference - current)
    return actual, bound


def direct_offdiagonal_covariance_loss(z: torch.Tensor, *, mean: torch.Tensor):
    if z.ndim != 2:
        raise ValueError(
            f"Expected z with shape [batch, features], got {tuple(z.shape)}"
        )
    centered = z - mean.detach()
    covariance = centered.T @ centered / z.shape[0]
    offdiagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return offdiagonal.square().sum() / z.shape[1]


def temporal_references(
    z: torch.Tensor,
    *,
    state,
    boundaries: torch.Tensor,
    detach: bool,
):
    """Return the preceding activation and a validity mask for each sample."""
    if z.ndim != 2:
        raise ValueError(
            f"Expected z with shape [batch, features], got {tuple(z.shape)}"
        )
    if boundaries.shape != (z.shape[0],):
        raise ValueError(
            f"Expected boundaries with shape {(z.shape[0],)}, got {tuple(boundaries.shape)}"
        )
    if boundaries.dtype is not torch.bool:
        raise TypeError("boundaries must be a boolean tensor")

    previous = torch.empty_like(z)
    previous[0] = state.previous
    if z.shape[0] > 1:
        previous[1:] = z[:-1]
    if detach:
        previous = previous.detach()

    valid = ~boundaries
    valid = valid.clone()
    valid[0] = valid[0] & state.has_previous
    return previous, valid


def regularized_target_components(
    *,
    z: torch.Tensor,
    previous: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    lateral: torch.Tensor,
    pair_valid: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
    lateral_reference: torch.Tensor | None = None,
    temporal_term_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    """Return the three detached contributions to the regularized target residual."""
    if coefficients.similarity <= 0.0:
        raise ValueError("regularized target construction requires positive similarity")
    if z.ndim != 2:
        raise ValueError(
            f"Expected z with shape [batch, features], got {tuple(z.shape)}"
        )
    if previous.shape != z.shape:
        raise ValueError(
            f"Expected previous with shape {tuple(z.shape)}, got {tuple(previous.shape)}"
        )
    if pair_valid.shape != (z.shape[0],):
        raise ValueError(
            f"Expected pair_valid with shape {(z.shape[0],)}, got {tuple(pair_valid.shape)}"
        )

    values = z.detach()
    previous = previous.detach()
    mean = mean.detach()
    variance = variance.detach()
    lateral = lateral.detach()
    centered = values - mean
    if lateral_reference is None:
        lateral_reference = centered
    if lateral_reference.shape != values.shape:
        raise ValueError(
            f"Expected lateral_reference with shape {tuple(values.shape)}, "
            f"got {tuple(lateral_reference.shape)}"
        )

    pair_count = int(pair_valid.sum())
    pair_weight = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    if pair_count:
        pair_weight[pair_valid] = z.shape[0] / pair_count
    variance_gate = F.relu(
        torch.as_tensor(variance_target, device=z.device, dtype=z.dtype) - variance
    )
    temporal = pair_weight[:, None] * (values - previous)
    if not temporal_term_enabled:
        temporal = torch.zeros_like(temporal)
    return {
        "temporal": temporal.detach(),
        "variance": (
            -(coefficients.variance / coefficients.similarity)
            * variance_gate
            * centered
        ).detach(),
        "covariance": (
            (coefficients.covariance / (2.0 * coefficients.similarity))
            * F.linear(lateral_reference.detach(), lateral)
        ).detach(),
    }


def regularized_target_residual(
    *,
    z: torch.Tensor,
    previous: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    lateral: torch.Tensor,
    pair_valid: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
    lateral_reference: torch.Tensor | None = None,
    temporal_term_enabled: bool = True,
):
    """Construct the detached target whose residual gives the samplewise gradient."""
    components = regularized_target_components(
        z=z,
        previous=previous,
        mean=mean,
        variance=variance,
        lateral=lateral,
        pair_valid=pair_valid,
        coefficients=coefficients,
        variance_target=variance_target,
        lateral_reference=lateral_reference,
        temporal_term_enabled=temporal_term_enabled,
    )
    residual = sum(components.values()).detach()
    values = z.detach()
    target = (values - residual).detach()
    return target, residual


def _validate_residual_lateral_inputs(
    base_state: torch.Tensor,
    lateral: torch.Tensor,
    coefficient: float,
) -> None:
    if base_state.ndim != 2:
        raise ValueError(
            f"Expected base_state with shape [batch, features], got {tuple(base_state.shape)}"
        )
    features = base_state.shape[1]
    if lateral.shape != (features, features):
        raise ValueError(
            f"Expected lateral with shape {(features, features)}, got {tuple(lateral.shape)}"
        )
    if coefficient < 0.0:
        raise ValueError("residual lateral coefficient must be nonnegative")


def residual_lateral_equilibrium(
    *,
    base_state: torch.Tensor,
    lateral: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Solve the inhibitory residual-state dynamics exactly as a reference."""
    _validate_residual_lateral_inputs(base_state, lateral, coefficient)
    values = base_state.detach()
    operator = (
        torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
        + coefficient * lateral.detach()
    )
    return torch.linalg.solve(operator, values.T).T.detach()


def residual_lateral_dynamics(
    *,
    base_state: torch.Tensor,
    lateral: torch.Tensor,
    coefficient: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    """Approximate inhibitory residual-state equilibrium with local dynamics."""
    _validate_residual_lateral_inputs(base_state, lateral, coefficient)
    if steps < 0:
        raise ValueError("residual lateral dynamics steps must be nonnegative")
    if step_size <= 0.0:
        raise ValueError("residual lateral dynamics step size must be positive")
    values = base_state.detach()
    lateral = lateral.detach()
    state = values.clone()
    for _ in range(steps):
        inhibition = F.linear(state, lateral)
        state = state + step_size * (values - state - coefficient * inhibition)
    return state.detach()


def residual_lateral_offset_correction(
    *,
    base_state: torch.Tensor,
    previous_state: torch.Tensor,
    lateral: torch.Tensor,
    coefficient: float,
    step_size: float,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    """Apply one inhibitory correction driven by the preceding neuron state."""
    _validate_residual_lateral_inputs(base_state, lateral, coefficient)
    if previous_state.shape != base_state.shape:
        raise ValueError(
            f"Expected previous_state with shape {tuple(base_state.shape)}, "
            f"got {tuple(previous_state.shape)}"
        )
    if pair_valid.shape != (base_state.shape[0],):
        raise ValueError(
            f"Expected pair_valid with shape {(base_state.shape[0],)}, "
            f"got {tuple(pair_valid.shape)}"
        )
    if pair_valid.dtype is not torch.bool:
        raise TypeError("pair_valid must be a boolean tensor")
    if step_size <= 0.0:
        raise ValueError("residual lateral correction step size must be positive")
    inhibition = F.linear(previous_state.detach(), lateral.detach())
    correction = step_size * coefficient * inhibition
    correction = torch.where(
        pair_valid[:, None], correction, torch.zeros_like(correction)
    )
    return (base_state.detach() - correction).detach()


def residual_lateral_moment(neuron_state: torch.Tensor) -> torch.Tensor:
    """Return the detached second moment used by residual-state lateral synapses."""
    if neuron_state.ndim != 2:
        raise ValueError(
            "Expected neuron_state with shape [batch, features], "
            f"got {tuple(neuron_state.shape)}"
        )
    if neuron_state.shape[0] == 0:
        raise ValueError("Cannot form a lateral moment from an empty batch")
    values = neuron_state.detach()
    return (values.T @ values / values.shape[0]).detach()


def terel_loss(
    *,
    z: torch.Tensor,
    previous: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    lateral: torch.Tensor,
    pair_valid: torch.Tensor,
    coefficients: LossCoefficients,
    variance_target: float,
    detach_previous: bool,
    lateral_reference: torch.Tensor | None = None,
):
    """Compute the TeReL loss for one layer.

    Population state and the input to the lateral operator are constants for
    differentiation. The temporal reference is optionally detached, which
    distinguishes temporally local TeReL from the undetached batched variant.
    """
    if z.ndim != 2:
        raise ValueError(
            f"Expected z with shape [batch, features], got {tuple(z.shape)}"
        )
    if pair_valid.shape != (z.shape[0],):
        raise ValueError(
            f"Expected pair_valid with shape {(z.shape[0],)}, got {tuple(pair_valid.shape)}"
        )

    if detach_previous:
        previous = previous.detach()
    mean = mean.detach()
    variance = variance.detach()
    lateral = lateral.detach()

    temporal_sq = (z - previous).square().mean(dim=1)
    if pair_valid.any():
        similarity_loss = temporal_sq[pair_valid].mean()
    else:
        similarity_loss = z.sum() * 0.0

    centered = z - mean
    variance_gate = F.relu(
        torch.as_tensor(variance_target, device=z.device, dtype=z.dtype) - variance
    )
    variance_loss = -(variance_gate * centered.square()).mean()

    if lateral_reference is None:
        lateral_reference = centered
    if lateral_reference.shape != centered.shape:
        raise ValueError(
            f"Expected lateral_reference with shape {tuple(centered.shape)}, "
            f"got {tuple(lateral_reference.shape)}"
        )
    lateral_signal = F.linear(lateral_reference.detach(), lateral).detach()
    covariance_loss = (centered * lateral_signal).mean()

    loss = (
        coefficients.similarity * similarity_loss
        + coefficients.variance * variance_loss
        + coefficients.covariance * covariance_loss
    )
    metrics = {
        "similarity_loss": similarity_loss.detach(),
        "variance_loss": variance_loss.detach(),
        "covariance_loss": covariance_loss.detach(),
        "mean_variance": variance.detach().mean(),
    }
    return loss, metrics


def detached_terel_loss(**kwargs):
    """Compute the temporally local TeReL objective."""
    return terel_loss(**kwargs, detach_previous=True)
