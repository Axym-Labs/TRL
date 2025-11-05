
import torch
import torch.nn as nn

from trl.store import MappingStore
from trl.config.config import BatchNormConfig


class ConfigurableBatchNorm(nn.Module):
    def __init__(self, out_dim: int, bn_cfg: BatchNormConfig, problem_type: str):
        super().__init__()
        self.bn_cfg = bn_cfg
        self.problem_type = problem_type

        self.scale = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, out: torch.Tensor, store: MappingStore):
        store.update_pre(out)
        decideStatisticsChoice = lambda stat: None if self.training and self.bn_cfg.use_batch_statistics_training else stat
        out = self.bn_normalization(out, decideStatisticsChoice(store.pre_nonlin_mu.value), decideStatisticsChoice(store.pre_nonlin_var.value))

        if self.bn_cfg.scale_parameter:
            out = out * self.scale
        if self.bn_cfg.bias_parameter:
            out = out + self.bias

        return out

    def bn_normalization(self, out, mean, var):
        stat_dims = (0,1) if self.problem_type == "sequence" else 0
        if self.bn_cfg.use_mean:
            mean = mean if mean is not None else out.mean(dim=stat_dims)
            if self.bn_cfg.detach_batch_statistics:
                mean = mean.detach()
            out = out - mean
        if self.bn_cfg.use_variance:
            var = var if var is not None else out.var(dim=stat_dims, unbiased=False)
            if self.bn_cfg.detach_batch_statistics:
                var = var.detach()
            out = out / torch.sqrt(var + self.bn_cfg.eps)
        return out


