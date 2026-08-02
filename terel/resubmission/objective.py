from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossCoefficients:
    similarity: float = 1.0
    variance: float = 2.5
    covariance: float = 1.0


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
    bound = (
        torch.linalg.matrix_norm(lateral - target, ord=2)
        * torch.linalg.vector_norm(current)
        + torch.linalg.matrix_norm(lateral, ord=2)
        * torch.linalg.vector_norm(reference - current)
    )
    return actual, bound


def direct_offdiagonal_covariance_loss(z: torch.Tensor, *, mean: torch.Tensor):
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape [batch, features], got {tuple(z.shape)}")
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
        raise ValueError(f"Expected z with shape [batch, features], got {tuple(z.shape)}")
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
        raise ValueError(f"Expected z with shape [batch, features], got {tuple(z.shape)}")
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
    variance_gate = F.relu(torch.as_tensor(variance_target, device=z.device, dtype=z.dtype) - variance)
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
