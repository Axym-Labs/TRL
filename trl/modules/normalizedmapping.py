
import torch
import torch.nn as nn

from trl.config.config import Config
from trl.loss import TCLoss
from trl.store import MappingStore


class NormalizedMapping(nn.Module):
    """
    Base layer, consisting of linear, an optional normalization layer, relu and lateral layer
    """
    def __init__(self, cfg: Config, in_dim, out_dim, norm, layer_bias, act_fn):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=layer_bias)
        self.norm = norm 
        self.act_fn = act_fn
        self.lat = nn.Linear(out_dim, out_dim, bias=False)
        self.criterion = TCLoss(out_dim, cfg.tcloss_config)
        self.store = MappingStore(cfg.store_config, out_dim)

    def forward(self, x: torch.Tensor):
        out = self.lin(x)

        if self.norm is not None:
            out = self.norm(out, self.store)

        return self.act_fn(out)
    
    def compute_loss(self, x: torch.Tensor):
        z_pre = self.forward(x)
        z = self.act_fn(z_pre)
        self.store.update_post(z)
        losses_metrics = self.criterion(z, lateral=self.lat, store=self.store)
        self.store.update_last_z(z)

        return (z, *losses_metrics)

    def layer_lat_params(self):
        layer_params = list(self.lin.parameters())
        if self.norm is not None:
            layer_params += self.norm.parameters()

        lat_params = self.lat.parameters()

        return layer_params, lat_params


