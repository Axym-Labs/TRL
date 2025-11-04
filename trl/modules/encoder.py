import torch
import torch.nn as nn
import contextlib
from typing import Tuple

from trl.config.config import Config, EncoderConfig
from trl.modules.batchnorm import ConfigurableBatchNorm
from trl.modules.normalizedmapping import NormalizedMapping


class TCEncoder(nn.Module):
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig, layer_dims: Tuple[Tuple[int, int], ...]):
        super().__init__()
        self.recurrence_depth = encoder_cfg.recurrence_depth
        self.rep_dim = layer_dims[-1][1]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in layer_dims:
            norm = ConfigurableBatchNorm(out_dim, cfg.batchnorm_config) if cfg.batchnorm_config is not None else None
            act_fn = encoder_cfg.activaton_fn()
            self.layers.append(NormalizedMapping(cfg, in_dim, out_dim, norm, encoder_cfg.layer_bias, act_fn))
        
    @property
    def unique_layer_count(self):
        return len(self.layers)

    @property
    def pass_layer_count(self):
        return self.recurrence_depth * self.unique_layer_count
        
    def enumerate_unique_layers(self):
        for i, layer in enumerate(self.layers):
            yield (i, layer)

    def enumerate_pass_layers(self):
        "index is from unique layers"
        for pass_i in range(self.recurrence_depth):
            for unique_i, layer in self.enumerate_unique_layers():
                layer_pass_i = pass_i * self.unique_layer_count + unique_i
                yield layer_pass_i, unique_i, layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for _, _, layer in self.enumerate_pass_layers():
            x = layer(x)
        return x
    
    def acts_before_layer(self, x: torch.Tensor, layer_idx, no_grad=True) -> torch.Tensor:
        x = self.flatten(x)

        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for pass_i, _, layer in self.enumerate_pass_layers():
                if pass_i >= layer_idx:
                    break
                x = layer(x)

        return x
        