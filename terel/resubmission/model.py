from collections.abc import Iterable

import torch
from torch import nn

from .state import TeReLState


class LayerLocalEncoder(nn.Module):
    """MLP whose layer losses do not propagate into earlier layers."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        statistics_momentum: float,
        lateral_momentum: float,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dims[:-1], dims[1:], strict=True)
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")
        self.states = nn.ModuleList(
            TeReLState(width, statistics_momentum, lateral_momentum) for width in hidden_dims
        )

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.layers.parameters()

    def forward_local(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        current = x
        for layer in self.layers:
            current = self.activation(layer(current))
            activations.append(current)
            current = current.detach()
        return activations

    def forward(self, x: torch.Tensor, *, return_all: bool = False):
        activations = []
        current = x
        for layer in self.layers:
            current = self.activation(layer(current))
            activations.append(current)
        return activations if return_all else activations[-1]
