from collections.abc import Iterable
from itertools import chain, pairwise

import torch
from torch import nn

from .state import TeReLState


class StreamingNorm(nn.Module):
    """Per-feature normalization using bounded detached streaming moments."""

    def __init__(
        self,
        features: int,
        *,
        momentum: float,
        epsilon: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("normalization momentum must lie in [0, 1)")
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_buffer("weight", torch.ones(features))
            self.register_buffer("bias", torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_variance", torch.ones(features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_mean = values.detach().mean(dim=0)
                deviation = (values.detach() - self.running_mean).square().mean(dim=0)
                self.running_mean.mul_(self.momentum).add_(
                    batch_mean, alpha=1.0 - self.momentum
                )
                self.running_variance.mul_(self.momentum).add_(
                    deviation, alpha=1.0 - self.momentum
                )
        normalized = (values - self.running_mean.detach()) / torch.sqrt(
            self.running_variance.detach() + self.epsilon
        )
        return normalized * self.weight + self.bias


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
        normalization_momentum: float = 0.9,
        normalization_affine: bool = True,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in pairwise(dims)
        )
        if normalization == "none":
            self.normalizations = nn.ModuleList(nn.Identity() for _ in hidden_dims)
        elif normalization == "batch_norm":
            self.normalizations = nn.ModuleList(
                nn.BatchNorm1d(width, eps=1e-5, momentum=0.1, affine=True)
                for width in hidden_dims
            )
        elif normalization == "layer_norm":
            self.normalizations = nn.ModuleList(
                nn.LayerNorm(width) for width in hidden_dims
            )
        elif normalization == "streaming_norm":
            self.normalizations = nn.ModuleList(
                StreamingNorm(
                    width,
                    momentum=normalization_momentum,
                    affine=normalization_affine,
                )
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
            TeReLState(width, statistics_momentum, lateral_momentum)
            for width in hidden_dims
        )

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.layers.parameters(), self.normalizations.parameters())

    def forward_local(
        self, x: torch.Tensor, *, stop_after: int | None = None
    ) -> list[torch.Tensor]:
        return [
            activation
            for _, _, activation in self.forward_local_details(x, stop_after=stop_after)
        ]

    def forward_local_details(
        self, x: torch.Tensor, *, stop_after: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return preactivation, normalized value, and output for each local layer."""
        activations = []
        current = x
        for index, (layer, normalization) in enumerate(
            zip(self.layers, self.normalizations, strict=True)
        ):
            preactivation = layer(current)
            normalized = normalization(preactivation)
            current = self.activation(normalized)
            activations.append((preactivation, normalized, current))
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


class OfflineEncoder(nn.Module):
    """End-to-end ReLU encoder for the nonlocal TeReL-Offline reference."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in pairwise(dims)
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.layers.parameters()

    def forward(self, x: torch.Tensor, *, return_all: bool = False):
        activations = []
        current = x
        for layer in self.layers:
            current = self.activation(layer(current))
            activations.append(current)
        return activations if return_all else activations[-1]
