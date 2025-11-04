import torch
import torch.nn as nn
import contextlib
from typing import Tuple

from trl.config.config import Config, EncoderConfig
from trl.modules.batchnorm import ConfigurableBatchNorm
from trl.modules.normalizedmapping import NormalizedMapping


class TREncoder(nn.Module):
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

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_input(x)
        for _, _, layer in self.enumerate_pass_layers():
            x = layer(x)
        return x
    
    def acts_before_layer(self, x: torch.Tensor, layer_idx, no_grad=True) -> torch.Tensor:
        x = self.prepare_input(x)

        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for pass_i, _, layer in self.enumerate_pass_layers():
                if pass_i >= layer_idx:
                    break
                x = layer(x)

        return x
        

class TRSeqEncoder(TREncoder):
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig, layer_dims: Tuple[Tuple[int, int], ...]):
        super().__init__(cfg, encoder_cfg, layer_dims)
        # keep the second dimension - the sequence dimension 
        self.flatten = nn.Flatten(start_dim=2) 

    def enumerate_pass_layers(self):
        for pass_i in range(self.recurrence_depth):
            for unique_i, layer in enumerate(self.layers):
                layer_pass_i = pass_i * self.unique_layer_count + unique_i
                yield layer_pass_i, unique_i, layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D_x = x.shape
        hidden = x.new_zeros(B, self.rep_dim)
        for _, _, layer in self.enumerate_pass_layers():
            outputs = []
            for t in range(S):
                inp = torch.cat([x[:, t, :], hidden], dim=1)
                out = layer(inp)
                outputs.append(out)
                hidden = out if out.shape[-1] == self.rep_dim else hidden
            x = torch.stack(outputs, dim=1)
            hidden = x[:, -1, :]
        return hidden

    def acts_before_layer(self, x: torch.Tensor, layer_idx: int, no_grad: bool = True) -> torch.Tensor:
        x_seq = self.prepare_input(x)
        B, S, D_x = x_seq.shape
        hidden = x_seq.new_zeros(B, self.rep_dim)
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            activations_per_t = []
            for t in range(S):
                cur = torch.cat([x_seq[:, t, :], hidden], dim=1)
                for layer_idx in range(layer_idx):
                    cur = self.layers[layer_idx](cur)
                hidden = cur if cur.shape[-1] == self.rep_dim else hidden
                activations_per_t.append(cur)

            act = torch.stack(activations_per_t, dim=1)
            # B, S, D = acts.shape
            # return acts.contiguous().view(B * S, D)
            return act

