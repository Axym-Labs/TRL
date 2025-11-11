
import torch
import torch.nn as nn

from trl.config.config import Config
from trl.loss import TRLoss
from trl.store import MappingStore
from trl.representation_metrics import RepresentationMetricsTracker

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
        
        rep_tracker = RepresentationMetricsTracker(out_dim, cfg.head_out_dim) if cfg.head_task == "classification" and cfg.track_representations else None
        self.criterion = TRLoss(out_dim, cfg.trloss_config, rep_tracker, chunk_size=cfg.data_config.chunk_size)
        self.store = MappingStore(cfg.store_config, out_dim, cfg.problem_type)

    def forward(self, x: torch.Tensor):
        out = self.lin(x)

        if self.norm is not None:
            out = self.norm(out, self.store)

        return self.act_fn(out)
    
    def training_pass(self, x: torch.Tensor):
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


