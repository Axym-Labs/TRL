import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.config.config import TRLossConfig
from trl.store import MappingStore
from trl.representation_metrics import EmtpyRepresentationMetricsTracker, RepresentationMetricsTracker


class TRLoss(nn.Module):
    def __init__(self, num_features: int, cfg: TRLossConfig, rep_tracker, chunk_size: int = None):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size

        if self.cfg.var_target_init == "rand":
            t = torch.rand(num_features) * self.cfg.var_sample_factor
        else:
            t = torch.ones(num_features) * self.cfg.var_sample_factor
        self.register_buffer("variance_targets", t)

        self.cov_matrix_mask = torch.rand((num_features, num_features)) <= self.cfg.cov_matrix_sparsity
        self.rep_tracker = rep_tracker or EmtpyRepresentationMetricsTracker()

    def forward(self, z: torch.Tensor, lateral: nn.Module, store: MappingStore):
        z_centered = z - store.mu.value

        sim_loss_pn = self.sim_loss(store.last_z.value, z_centered)
        std_loss_pn = self.std_loss(store.var.value, z_centered)
        cov_loss_pn  = self.cov_loss(z_centered, lateral)
        lateral_loss_pn = self.lat_loss(store.cov.value, lateral)

        sim_loss = sim_loss_pn.mean(dim=0).sum()
        std_loss = std_loss_pn.mean(dim=0).sum()
        cov_loss = cov_loss_pn.mean(dim=0).sum()
        lat_loss = lateral_loss_pn.mean(dim=0).sum()

        vicreg_loss = self.cfg.sim_coeff * sim_loss + self.cfg.std_coeff * std_loss + self.cfg.cov_coeff * cov_loss
        lateral_loss = self.cfg.lat_coeff * lat_loss

        self.rep_tracker.update(z)

        metrics = {
            'sim_loss': sim_loss.detach(),
            'std_loss': std_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'var_stat': store.var.value.mean(),
            'cov_stat': (store.cov.value**2).sum() / z.shape[1],
        }
        return vicreg_loss, lateral_loss, metrics
    
    def sim_loss(self, last_z, z_centered):
        # originally, we compute the MSE on z, but this is mathematically
        # equivalent to the MSE on z_centered
        if self.cfg.consider_last_batch_z: 
            assert not self.cfg.sim_within_chunks, "consider_last_batch_z and sim_within_chunks are mutually exclusive"
            z_centered = torch.concat([last_z.unsqueeze(0), z_centered], dim=0)

        if self.cfg.sim_within_chunks:
            B, D = z_centered.shape
            zc = z_centered.view(B // self.chunk_size, self.chunk_size, D)
            prev = zc[:, :-1, :]
            if self.cfg.detach_previous:
                prev = prev.detach()
            diff = zc[:, 1:, :] - prev
            return (diff ** 2).mean(dim=1)  # average over chunk dimension
        else:
            prev = z_centered[:-1]
            if self.cfg.detach_previous:
                prev = prev.detach()
            diff = z_centered[1:] - prev
            return (diff ** 2)

    def std_loss(self, var_stat, z_centered):
        var_gd = z_centered.pow(2)
        diff = self.variance_targets - var_stat
        std_loss_pn = -F.relu(diff) * var_gd if not self.cfg.bidirectional_variance_loss else -diff * var_gd
        return std_loss_pn
    
    def cov_loss(self, z_centered, lateral):
        if self.cfg.use_cov_directly:
            cov = (z_centered.T @ z_centered) / (z_centered.shape[0] - 1)
            cov.diagonal().zero_()
            cov_loss = cov.pow_(2).sum()

            return  cov_loss
        else:
            return z_centered * (lateral(z_centered.detach())).detach()
    
    def lat_loss(self, cov_stat, lateral):
        # no self-connections
        cov_stat.fill_diagonal_(0.0)
        # artificially sparsen with persistant mask
        cov_stat[self.cov_matrix_mask] = 0.0

        lateral_loss_pn = F.mse_loss(lateral.weight, cov_stat, reduction="none")
        return lateral_loss_pn

    def epoch_metrics(self):
        return self.rep_tracker.scalar_metrics()
    

class TRSeqLoss(TRLoss):
    "TCLoss for training on sequence tasks, ie where inputs are of shape (batch_size, sequence_length, latent_dimension)"
    def sim_loss(self, last_z, z_centered):
        if self.cfg.sim_within_chunks:
            raise ValueError("sim_within_chunks is not supported for sequence tasks. The sequence is the chunk.")

        B, S, D = z_centered.shape

        if self.cfg.consider_last_batch_z:
            last = last_z.view(1, D).unsqueeze(1).expand(B, 1, D)
            zc = torch.cat([last, z_centered], dim=1)
        else:
            zc = z_centered
        diff = zc[:, 1:, :] - zc[:, :-1, :]
        # average over sequence for compatibility with TCLoss
        return (diff ** 2).mean(dim=1)  

    def cov_loss(self, z_centered, lateral):
        B, S, D = z_centered.shape
        z_flat = z_centered.contiguous().view(-1, D)
        lat_flat = lateral(z_flat.detach()).detach()
        lat = lat_flat.view(B, S, D)
        return z_centered * lat
