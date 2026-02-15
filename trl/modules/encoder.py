import torch
import torch.nn as nn
import contextlib
from typing import Tuple

from trl.config.config import Config, EncoderConfig
from trl.modules.batchnorm import ConfigurableBatchNorm
from trl.modules.normalizedmapping import NormalizedMapping


class TREncoder(nn.Module):
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig):
        super().__init__()
        self.enc_cfg = encoder_cfg
        self.rep_dim = encoder_cfg.layer_dims[-1][1]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in encoder_cfg.layer_dims:
            norm = ConfigurableBatchNorm(out_dim, cfg.batchnorm_config, cfg.problem_type) if cfg.batchnorm_config is not None else None
            act_fn = encoder_cfg.activation_fn() if not encoder_cfg.activation_fn_requires_dim else encoder_cfg.activation_fn(out_dim)
            self.layers.append(NormalizedMapping(cfg, in_dim, out_dim, norm, encoder_cfg.layer_bias, act_fn))
        
    @property
    def unique_layer_count(self):
        return len(self.layers)

    @property
    def pass_layer_count(self):
        return self.enc_cfg.recurrence_depth * self.unique_layer_count
        
    def enumerate_unique_layers(self):
        for i, layer in enumerate(self.layers):
            yield (i, layer)

    def enumerate_pass_layers(self):
        "index is from unique layers"
        for pass_i in range(self.enc_cfg.recurrence_depth):
            for unique_i, layer in self.enumerate_unique_layers():
                layer_pass_i = pass_i * self.unique_layer_count + unique_i
                yield layer_pass_i, unique_i, layer

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(x)
    
    @property
    def max_pass_layer(self):
        return len(self.layers) * self.enc_cfg.recurrence_depth - 1

    def forward(self, x: torch.Tensor, gather_layer_activations: list|None = None) -> torch.Tensor|list[torch.Tensor]:
        if gather_layer_activations is not None:
            return self.gather_layer_activations(x, no_grad=False, layer_indices=gather_layer_activations)

        return self.acts_before_layer(self.prepare_input(x), self.max_pass_layer+1, no_grad=False)
    
    def gather_layer_activations(self, x: torch.Tensor, no_grad=True, layer_indices=None) -> list[torch.Tensor]:
        "layer_indices=None means all activations"

        activations = []
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            cur_x = self.prepare_input(x)
            for pass_i, unique_i, layer in self.enumerate_pass_layers():
                cur_x = layer(cur_x)
                if layer_indices is None or pass_i in layer_indices:
                    activations.append(cur_x)
        return activations
    
    def acts_before_layer(self, x: torch.Tensor, layer_idx, no_grad=True) -> torch.Tensor:
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for pass_i, _, layer in self.enumerate_pass_layers():
                if pass_i == layer_idx:
                    break
                x = layer(x)

        return x
        

class TRSeqEncoder(TREncoder):
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig,):
        super().__init__(cfg, encoder_cfg)
        # keep the second dimension - the sequence dimension 
        self.flatten = nn.Flatten(start_dim=2) 

    def prepare_input(self, x: torch.Tensor):
        x = super().prepare_input(x)
        B, S, D_x = x.shape
        hidden = x.new_zeros(B, self.rep_dim)
        return x, hidden

    def concat_x_t_hidden(self, x, t, hidden):
        return torch.cat([x[:, t, :], hidden], dim=1)
    
    def gather_layer_activations(self, x: torch.Tensor, no_grad=True, layer_indices=None) -> list[torch.Tensor]:
        raise NotImplementedError("gather_layer_activations is not implemented for TRSeqEncoder yet")

    def acts_before_layer(self, prepared_input, layer_idx: int, no_grad: bool = True) -> torch.Tensor:
        x, hidden = prepared_input
        B, S, D = x.shape

        if layer_idx == 0:
            # For the sequence encoder the first layer sees [x_t, hidden],
            # not just x_t.
            layer_latent_dim = self.enc_cfg.layer_dims[0][0]
        else:
            layer_unique_i = (layer_idx-1) % len(self.layers)
            layer_latent_dim = self.enc_cfg.layer_dims[layer_unique_i][1]
        act = torch.empty((B, S, layer_latent_dim))

        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for t in range(S):
                cur = self.concat_x_t_hidden(x, t, hidden)
                for i, _, layer  in self.enumerate_pass_layers():
                    if i == layer_idx:
                        act[:, t, :] = cur
                    cur = layer(cur)
                # since we do not break in this implementation
                # we have to catch "acts after the last layer" explicitly
                if i+1 == layer_idx:
                    act[:, t, :] = cur
                hidden = cur

        return act


class TRSeqElementwiseEncoder(TREncoder):
    """
    Sequence encoder without recurrent hidden state:
    applies the same feedforward mapping to each timestep independently.
    """
    def __init__(self, cfg: Config, encoder_cfg: EncoderConfig):
        super().__init__(cfg, encoder_cfg)
        self.flatten = nn.Flatten(start_dim=2)
        
