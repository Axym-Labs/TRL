import torch
from dataclasses import dataclass

from trl.config.config import StoreConfig


@dataclass
class LeakyIntegrator:
    value: torch.Tensor
    momentum: float
    overwrite_at_start: bool = True
    "Whether to use the initialization or first value as starting value."

    def update(self, new: torch.Tensor):
        if self.overwrite_at_start:
            self.value = new.detach()
            self.overwrite_at_start = False
        else:
            self.value = self.momentum * self.value + (1-self.momentum) * new.detach()

@dataclass
class MappingStore:
    mu: LeakyIntegrator
    var: LeakyIntegrator
    cov: LeakyIntegrator
    last_z: LeakyIntegrator
    trace_z: LeakyIntegrator
    last_centered_z: LeakyIntegrator

    pre_nonlin_mu: LeakyIntegrator
    pre_nonlin_var: LeakyIntegrator

    problem_type: str

    def __init__(self, cfg: StoreConfig, out_dim: int, problem_type: str):
        self.batchless_updates = cfg.batchless_updates
        pre_m, post_m = cfg.pre_stats_momentum, cfg.post_stats_momentum
        self.mu=LeakyIntegrator(torch.zeros(out_dim, device=cfg.device), post_m, overwrite_at_start=cfg.overwrite_at_start)
        self.var=LeakyIntegrator(torch.ones(out_dim, device=cfg.device), pre_m, overwrite_at_start=cfg.overwrite_at_start)
        self.cov=LeakyIntegrator(torch.zeros((out_dim, out_dim), device=cfg.device), cfg.cov_momentum, overwrite_at_start=cfg.overwrite_at_start)
        self.last_z=LeakyIntegrator(torch.zeros(out_dim, device=cfg.device), cfg.last_z_momentum, overwrite_at_start=cfg.overwrite_at_start)
        self.trace_z=LeakyIntegrator(torch.zeros(out_dim, device=cfg.device), cfg.trace_momentum, overwrite_at_start=cfg.overwrite_at_start)
        self.last_centered_z=LeakyIntegrator(torch.zeros(out_dim, device=cfg.device), 0.0, overwrite_at_start=cfg.overwrite_at_start)
        self.pre_nonlin_mu=LeakyIntegrator(torch.zeros(out_dim, device=cfg.device), pre_m, overwrite_at_start=cfg.overwrite_at_start)
        self.pre_nonlin_var=LeakyIntegrator(torch.ones(out_dim, device=cfg.device), pre_m, overwrite_at_start=cfg.overwrite_at_start)
        self.problem_type = problem_type

    def update_pre(self, z_pre: torch.Tensor):
        self.pre_nonlin_mu.update(z_pre.mean(dim=0))
        if self.batchless_updates:
            var = ((z_pre-self.pre_nonlin_mu.value)**2).mean(dim=0)
        else:
            var = torch.var(z_pre, dim=0, unbiased=False)
        self.pre_nonlin_var.update(var)

    def update_post(self, z: torch.Tensor):
        z = z.detach()
        # here, we flatten instead of specifying the dimension
        # because we have to calculate the covariance matrix
        if self.problem_type == "sequence" and z.ndim >= 3:
            z = z.contiguous().view(-1, z.shape[2])

        batch_mean = z.mean(dim=0)
        self.mu.update(batch_mean)
        mu = self.mu.value if self.batchless_updates else batch_mean
        z_centered = z - mu
        
        var = (z_centered**2).mean(dim=0) if self.batchless_updates else torch.var(z, dim=0, unbiased=False)
        self.var.update(var)
        cov = (z_centered.T @ z_centered) / z_centered.shape[0]
        self.cov.update(cov)
        self.last_centered_z.update(z_centered[-1])

    def update_last_z(self, z: torch.Tensor):
        if self.problem_type == "sequence" and z.ndim >= 3:
            self.last_z.update(z[:, -1, :])
        else:
            self.last_z.update(z[-1])
