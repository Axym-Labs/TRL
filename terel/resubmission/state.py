import torch
from torch import nn


class TeReLState(nn.Module):
    """Detached running state for one TeReL layer."""

    def __init__(
        self, features: int, statistics_momentum: float, lateral_momentum: float
    ):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        for name, value in (
            ("statistics_momentum", statistics_momentum),
            ("lateral_momentum", lateral_momentum),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value}")

        self.statistics_momentum = float(statistics_momentum)
        self.lateral_momentum = float(lateral_momentum)
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("variance", torch.ones(features))
        self.register_buffer("lateral", torch.zeros(features, features))
        self.register_buffer("residual_lateral", None)
        self.register_buffer("previous", torch.zeros(features))
        self.register_buffer("previous_centered", None)
        self.register_buffer("previous_neuron_state", None)
        self.register_buffer("has_previous", torch.tensor(False))

    def dynamic_state_numel(self) -> int:
        return sum(buffer.numel() for buffer in self.buffers())

    def causal_dynamic_state_numel(self) -> int:
        vectors = (self.mean, self.variance, self.previous)
        total = sum(vector.numel() for vector in vectors) + self.has_previous.numel()
        if self.previous_centered is not None:
            total += self.previous_centered.numel()
        if self.previous_neuron_state is not None:
            total += self.previous_neuron_state.numel()
        return total

    def auxiliary_parameter_numel(self) -> int:
        total = self.lateral.numel()
        if self.residual_lateral is not None:
            total += self.residual_lateral.numel()
        return total

    @torch.no_grad()
    def ensure_residual_lateral(self) -> torch.Tensor:
        if self.residual_lateral is None:
            self.residual_lateral = torch.zeros_like(self.lateral)
        return self.residual_lateral

    @torch.no_grad()
    def ensure_previous_centered(self) -> torch.Tensor:
        if self.previous_centered is None:
            self.previous_centered = torch.zeros_like(self.previous)
        return self.previous_centered

    @torch.no_grad()
    def ensure_previous_neuron_state(self) -> torch.Tensor:
        if self.previous_neuron_state is None:
            self.previous_neuron_state = torch.zeros_like(self.previous)
        return self.previous_neuron_state

    @torch.no_grad()
    def reset_sequence(self) -> None:
        """Forget temporal predecessors while retaining learned running state."""
        self.previous.zero_()
        if self.previous_centered is not None:
            self.previous_centered.zero_()
        if self.previous_neuron_state is not None:
            self.previous_neuron_state.zero_()
        self.has_previous.fill_(False)

    @torch.no_grad()
    def update_previous_neuron_state(self, neuron_state: torch.Tensor) -> None:
        if neuron_state.ndim != 2 or neuron_state.shape[1] != self.mean.numel():
            raise ValueError(
                f"Expected neuron_state [batch, {self.mean.numel()}], "
                f"got {tuple(neuron_state.shape)}"
            )
        if neuron_state.shape[0] == 0:
            raise ValueError("Cannot store an empty neuron state")
        self.ensure_previous_neuron_state().copy_(neuron_state.detach()[-1])

    @torch.no_grad()
    def update_residual_lateral(
        self,
        neuron_state: torch.Tensor,
        *,
        include_diagonal: bool,
        moment_scale: float = 1.0,
    ) -> None:
        if neuron_state.ndim != 2 or neuron_state.shape[1] != self.mean.numel():
            raise ValueError(
                f"Expected neuron_state [batch, {self.mean.numel()}], "
                f"got {tuple(neuron_state.shape)}"
            )
        if neuron_state.shape[0] == 0:
            raise ValueError("Cannot update residual lateral state from an empty batch")
        if moment_scale <= 0.0:
            raise ValueError("moment_scale must be positive")
        values = neuron_state.detach()
        moment = moment_scale * values.T @ values / values.shape[0]
        if not include_diagonal:
            moment.fill_diagonal_(0.0)
        residual_lateral = self.ensure_residual_lateral()
        residual_lateral.mul_(self.lateral_momentum).add_(
            moment, alpha=1.0 - self.lateral_momentum
        )

    @torch.no_grad()
    def update_lateral_moment(self, moment: torch.Tensor) -> None:
        if moment.shape != self.lateral.shape:
            raise ValueError(
                f"Expected lateral moment {tuple(self.lateral.shape)}, "
                f"got {tuple(moment.shape)}"
            )
        self.lateral.mul_(self.lateral_momentum).add_(
            moment.detach(), alpha=1.0 - self.lateral_momentum
        )

    @torch.no_grad()
    def update(self, z: torch.Tensor, *, update_lateral: bool = True) -> None:
        if z.ndim != 2 or z.shape[1] != self.mean.numel():
            raise ValueError(
                f"Expected activations [batch, {self.mean.numel()}], got {tuple(z.shape)}"
            )
        if z.shape[0] == 0:
            raise ValueError("Cannot update state from an empty batch")
        values = z.detach()
        centered = values - self.mean
        batch_mean = values.mean(dim=0)
        batch_variance = centered.square().mean(dim=0)
        covariance = centered.T @ centered / values.shape[0]
        covariance.fill_diagonal_(0.0)

        sm = self.statistics_momentum
        lm = self.lateral_momentum
        self.mean.mul_(sm).add_(batch_mean, alpha=1.0 - sm)
        self.variance.mul_(sm).add_(batch_variance, alpha=1.0 - sm)
        if update_lateral:
            self.lateral.mul_(lm).add_(covariance, alpha=1.0 - lm)
        self.previous.copy_(values[-1])
        if self.previous_centered is not None:
            self.previous_centered.copy_(centered[-1])
        self.has_previous.fill_(True)
