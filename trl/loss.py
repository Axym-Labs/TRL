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
        if self.cfg.use_trace_activation and z_centered.ndim != 2:
            raise NotImplementedError("Trace activation is currently implemented for 2D activations only.")
        z_features = self.apply_trace(z_centered, store) if self.cfg.use_trace_activation else z_centered

        sim_loss_pn = self.sim_loss(store.trace_z.value if self.cfg.use_trace_activation else store.last_z.value, z_features)
        std_loss_pn = self.std_loss(store.var.value, z_features)
        lat_in_for_cov = self.prepare_lat_in_for_cov(z_features, store)
        cov_loss_pn  = self.cov_loss(z_features, lateral, store, lat_in=lat_in_for_cov)
        cov_target = self.cov_target(store.cov.value, z_features, lat_in_for_cov)
        lateral_loss_pn = self.lat_loss(cov_target, lateral)

        sim_loss = sim_loss_pn.mean(dim=0).sum()
        std_loss = std_loss_pn.mean(dim=0).sum()
        cov_loss = cov_loss_pn if cov_loss_pn.ndim == 0 else cov_loss_pn.mean(dim=0).sum()
        lat_loss = lateral_loss_pn.mean(dim=0).sum()

        vicreg_loss = self.cfg.sim_coeff * sim_loss + self.cfg.std_coeff * std_loss + self.cfg.cov_coeff * cov_loss
        lateral_loss = self.cfg.lat_coeff * lat_loss

        if self.cfg.use_trace_activation:
            store.trace_z.value = z_features[-1].detach()
        store.last_centered_z.value = z_features[-1].detach()
        self.rep_tracker.update(z_features)

        metrics = {
            'sim_loss': sim_loss.detach(),
            'std_loss': std_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'var_stat': store.var.value.mean(),
            'cov_stat': (store.cov.value**2).sum() / z.shape[1],
            'e_off': self.e_off_metric(z_features, store).detach(),
        }
        return vicreg_loss, lateral_loss, metrics
    
    def sim_loss(self, last_z, z_centered):
        # originally, we compute the MSE on z, but this is mathematically
        # equivalent to the MSE on z_centered
        if self.cfg.consider_last_batch_z: 
            assert not self.cfg.use_chunk_paritions, "consider_last_batch_z and use_chunk_paritions are mutually exclusive"
            z_centered = torch.concat([last_z.unsqueeze(0), z_centered], dim=0)

        if self.cfg.use_chunk_paritions:
            B, D = z_centered.shape
            zc = z_centered.view(B // self.chunk_size, self.chunk_size, D)
            # Chunk-local wrap-around: previous of first element is last element in same chunk.
            prev = torch.roll(zc, shifts=1, dims=1)
            if self.cfg.detach_previous:
                prev = prev.detach()
            cur = zc
            return self.cfg.sim_loss_fn(cur, prev).mean(dim=1)
        else:
            prev = z_centered[:-1]
            if self.cfg.detach_previous:
                prev = prev.detach()
            cur = z_centered[1:]
            return self.cfg.sim_loss_fn(cur, prev)

    def apply_trace(self, z_centered, store: MappingStore):
        decay = self.cfg.trace_decay
        if decay < 0.0 or decay > 1.0:
            raise ValueError(f"trace_decay must be in [0, 1], got {decay}")

        if self.cfg.use_chunk_paritions:
            B, D = z_centered.shape
            zc = z_centered.view(B // self.chunk_size, self.chunk_size, D)
            if self.chunk_size <= 1:
                return z_centered

            S = self.chunk_size
            x_det = zc.detach()
            # Chunk-local wrap-around state.
            prev = x_det[:, -1, :]
            W = self.ema_kernel(S, decay, z_centered.device, z_centered.dtype)
            ema_det = (1.0 - decay) * torch.einsum("tk,ckd->ctd", W, x_det)
            powers = (decay ** torch.arange(1, S + 1, device=z_centered.device, dtype=z_centered.dtype)).view(1, S, 1)
            ema_det = ema_det + powers * prev.unsqueeze(1)
            # Keep only local gradient wrt current sample, detach temporal path.
            trace = ema_det + (1.0 - decay) * (zc - x_det)
            return trace.reshape(B, D)

        B, D = z_centered.shape
        W = self.ema_kernel(B, decay, z_centered.device, z_centered.dtype)
        x_det = z_centered.detach()
        prev = store.trace_z.value.detach().view(1, D)
        ema_det = (1.0 - decay) * (W @ x_det)
        powers = (decay ** torch.arange(1, B + 1, device=z_centered.device, dtype=z_centered.dtype)).view(B, 1)
        ema_det = ema_det + powers * prev
        # Keep only local gradient wrt current sample, detach temporal path.
        return ema_det + (1.0 - decay) * (z_centered - x_det)

    def ema_kernel(self, n: int, decay: float, device, dtype):
        if n <= 0:
            return torch.empty((0, 0), device=device, dtype=dtype)
        idx = torch.arange(n, device=device)
        diff = (idx[:, None] - idx[None, :]).clamp_min(0)
        kernel = (decay ** diff).tril().to(dtype=dtype)
        return kernel

    def std_loss(self, var_stat, z_centered):
        var_gd = z_centered.pow(2)
        diff = self.variance_targets - var_stat
        std_loss_pn = -self.cfg.variance_hinge_fn(diff) * var_gd if not self.cfg.bidirectional_variance_loss else -diff * var_gd
        return std_loss_pn
    
    def prepare_lat_in_for_cov(self, z_centered: torch.Tensor, store: MappingStore):
        lat_in = z_centered.detach()
        if self.cfg.lateral_shift:
            lat_in = self.shift_lateral_input(lat_in, store)
        return lat_in

    def cov_target(self, cov_stat: torch.Tensor, z_centered: torch.Tensor, lat_in: torch.Tensor):
        if self.cfg.lateral_shift:
            # Match lateral training target to shifted usage: cross-cov(previous, current).
            return (lat_in.T @ z_centered.detach()) / max(1, z_centered.shape[0])
        return cov_stat

    def cov_loss(self, z_centered, lateral, store: MappingStore, lat_in=None):
        if self.cfg.use_cov_directly:
            z_flat = self.flatten_features(z_centered)
            cov = (z_flat.T @ z_flat) / max(1, (z_flat.shape[0] - 1))
            cov.diagonal().zero_()
            cov_loss = cov.pow_(2).sum()

            return  cov_loss
        else:
            if lat_in is None:
                lat_in = self.prepare_lat_in_for_cov(z_centered, store)
            return z_centered * (lateral(lat_in)).detach()

    def flatten_features(self, z_centered: torch.Tensor) -> torch.Tensor:
        if z_centered.ndim == 2:
            return z_centered
        if z_centered.ndim == 3:
            b, s, d = z_centered.shape
            return z_centered.reshape(b * s, d)
        raise ValueError(f"Unsupported activation rank for decorrelation: {z_centered.ndim}")

    def offdiag(self, m: torch.Tensor) -> torch.Tensor:
        return m - torch.diag_embed(torch.diagonal(m))

    def e_off_metric(self, z_centered: torch.Tensor, store: MappingStore):
        z_flat = self.flatten_features(z_centered)
        eps = float(self.cfg.decorrelation_eps)
        mu = store.mu.value
        var = store.var.value
        z_hat = (z_flat - mu) / torch.sqrt(var + eps)
        n = max(1, z_hat.shape[0])
        corr = (z_hat.T @ z_hat) / n
        off = self.offdiag(corr)
        return off.abs().mean()

    def shift_lateral_input(self, lat_in: torch.Tensor, store: MappingStore):
        shifted = torch.zeros_like(lat_in)
        if self.cfg.use_chunk_paritions:
            B, D = lat_in.shape
            lat_chunks = lat_in.view(B // self.chunk_size, self.chunk_size, D)
            shifted_chunks = shifted.view(B // self.chunk_size, self.chunk_size, D)
            shifted_chunks[:, 1:, :] = lat_chunks[:, :-1, :]
            # Chunk-local wrap-around.
            shifted_chunks[:, 0, :] = lat_chunks[:, -1, :]
            return shifted

        shifted[1:] = lat_in[:-1]
        shifted[0] = store.last_centered_z.value
        return shifted
    
    def lat_loss(self, cov_stat, lateral):
        cov_stat = cov_stat.clone()
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
        # Sequence mode already uses temporal adjacency inside each sample,
        # so chunk partition flags are ignored.

        B, S, D = z_centered.shape

        if self.cfg.consider_last_batch_z:
            last = last_z.view(1, D).unsqueeze(1).expand(B, 1, D)
            zc = torch.cat([last, z_centered], dim=1)
        else:
            zc = z_centered
        diff = zc[:, 1:, :] - zc[:, :-1, :]
        # average over sequence for compatibility with TCLoss
        return (diff ** 2).mean(dim=1)  

    def std_loss(self, var_stat, z_centered):
        std_loss_pn = super().std_loss(var_stat, z_centered)
        # Optional fix: avoid implicit scaling with sequence length S.
        # TRLoss.forward does mean(dim=0).sum() afterward.
        if self.cfg.sequence_std_mean_over_time and std_loss_pn.ndim == 3:
            # Keep expected magnitude comparable to legacy reduction by
            # compensating for averaging across time.
            s = z_centered.shape[1]
            return std_loss_pn.mean(dim=1) * s  # [B, D]
        return std_loss_pn

    def prepare_lat_in_for_cov(self, z_centered: torch.Tensor, store: MappingStore):
        lat_in = z_centered.detach()
        if self.cfg.lateral_shift:
            shifted = torch.zeros_like(lat_in)
            shifted[:, 1:, :] = lat_in[:, :-1, :]
            # Sequence-local wrap-around for the first step in each sample.
            shifted[:, 0, :] = lat_in[:, -1, :]
            lat_in = shifted
        return lat_in

    def cov_target(self, cov_stat: torch.Tensor, z_centered: torch.Tensor, lat_in: torch.Tensor):
        if self.cfg.lateral_shift:
            B, S, D = z_centered.shape
            z_flat = z_centered.detach().reshape(-1, D)
            lat_flat = lat_in.detach().reshape(-1, D)
            return (lat_flat.T @ z_flat) / max(1, z_flat.shape[0])
        return cov_stat

    def cov_loss(self, z_centered, lateral, store: MappingStore, lat_in=None):
        if self.cfg.use_cov_directly:
            return super().cov_loss(z_centered, lateral, store, lat_in=lat_in)

        B, S, D = z_centered.shape
        z_flat = z_centered.reshape(-1, D)
        if lat_in is None:
            lat_in = self.prepare_lat_in_for_cov(z_centered, store).reshape(-1, D)
        else:
            lat_in = lat_in.reshape(-1, D)
        lat_flat = lateral(lat_in).detach()
        lat = lat_flat.view(B, S, D)
        return z_centered * lat
