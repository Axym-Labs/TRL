from collections.abc import Iterable
from itertools import chain

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
        normalization: str = "none",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dims[:-1], dims[1:], strict=True)
        )
        if normalization == "none":
            self.normalizations = nn.ModuleList(nn.Identity() for _ in hidden_dims)
        elif normalization == "batch_norm":
            self.normalizations = nn.ModuleList(
                nn.BatchNorm1d(width, eps=1e-5, momentum=0.1, affine=True)
                for width in hidden_dims
            )
        else:
            raise ValueError(f"Unsupported normalization '{normalization}'")
        self.normalization = normalization
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")
        self.states = nn.ModuleList(
            TeReLState(width, statistics_momentum, lateral_momentum) for width in hidden_dims
        )

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.layers.parameters(), self.normalizations.parameters())

    def forward_local(
        self, x: torch.Tensor, *, stop_after: int | None = None
    ) -> list[torch.Tensor]:
        activations = []
        current = x
        for index, (layer, normalization) in enumerate(
            zip(self.layers, self.normalizations, strict=True)
        ):
            current = self.activation(normalization(layer(current)))
            activations.append(current)
            if stop_after is not None and index == stop_after:
                break
            current = current.detach()
        return activations

    def forward(self, x: torch.Tensor, *, return_all: bool = False):
        activations = []
        current = x
        for layer, normalization in zip(self.layers, self.normalizations, strict=True):
            current = self.activation(normalization(layer(current)))
            activations.append(current)
        return activations if return_all else activations[-1]
