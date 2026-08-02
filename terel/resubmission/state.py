import torch
from torch import nn


class TeReLState(nn.Module):
    """Detached running state for one TeReL layer."""

    def __init__(self, features: int, statistics_momentum: float, lateral_momentum: float):
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
        self.register_buffer("previous", torch.zeros(features))
        self.register_buffer("previous_centered", torch.zeros(features))
        self.register_buffer("has_previous", torch.tensor(False))

    def dynamic_state_numel(self) -> int:
        return sum(buffer.numel() for buffer in self.buffers())

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
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
        self.lateral.mul_(lm).add_(covariance, alpha=1.0 - lm)
        self.previous.copy_(values[-1])
        self.previous_centered.copy_(centered[-1])
        self.has_previous.fill_(True)
