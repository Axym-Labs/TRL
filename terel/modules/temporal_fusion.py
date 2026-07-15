import torch
from torch import nn


class EMAFusion(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"temporal_fusion_alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        if y_seq.ndim != 3:
            raise ValueError(f"Expected (B, S, D), got {tuple(y_seq.shape)}")
        bsz, seq_len, rep_dim = y_seq.shape
        h_prev = y_seq.new_zeros((bsz, rep_dim))
        out = []
        for t in range(seq_len):
            h_prev = h_prev.detach()
            h_t = (1.0 - self.alpha) * y_seq[:, t, :] + self.alpha * h_prev
            out.append(h_t)
            h_prev = h_t
        return torch.stack(out, dim=1)


class ConcatLinearFusion(nn.Module):
    def __init__(self, rep_dim: int):
        super().__init__()
        self.lin = nn.Linear(rep_dim * 2, rep_dim)

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        if y_seq.ndim != 3:
            raise ValueError(f"Expected (B, S, D), got {tuple(y_seq.shape)}")
        bsz, seq_len, rep_dim = y_seq.shape
        h_prev = y_seq.new_zeros((bsz, rep_dim))
        out = []
        for t in range(seq_len):
            h_prev = h_prev.detach()
            fused_in = torch.cat([y_seq[:, t, :], h_prev], dim=1)
            h_t = self.lin(fused_in)
            out.append(h_t)
            h_prev = h_t
        return torch.stack(out, dim=1)


class ConcatModuleFusion(nn.Module):
    def __init__(self, rep_dim: int, module: nn.Module):
        super().__init__()
        self.module = module
        self.rep_dim = rep_dim

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        if y_seq.ndim != 3:
            raise ValueError(f"Expected (B, S, D), got {tuple(y_seq.shape)}")
        bsz, seq_len, rep_dim = y_seq.shape
        h_prev = y_seq.new_zeros((bsz, rep_dim))
        out = []
        for t in range(seq_len):
            h_prev = h_prev.detach()
            fused_in = torch.cat([y_seq[:, t, :], h_prev], dim=1)
            h_t = self.module(fused_in)
            out.append(h_t)
            h_prev = h_t
        return torch.stack(out, dim=1)


class ResidualGateFusion(nn.Module):
    def __init__(self, rep_dim: int):
        super().__init__()
        self.gate = nn.Linear(rep_dim * 2, rep_dim)

    def forward(self, y_seq: torch.Tensor) -> torch.Tensor:
        if y_seq.ndim != 3:
            raise ValueError(f"Expected (B, S, D), got {tuple(y_seq.shape)}")
        bsz, seq_len, rep_dim = y_seq.shape
        h_prev = y_seq.new_zeros((bsz, rep_dim))
        out = []
        for t in range(seq_len):
            h_prev = h_prev.detach()
            fused_in = torch.cat([y_seq[:, t, :], h_prev], dim=1)
            g_t = torch.sigmoid(self.gate(fused_in))
            h_t = y_seq[:, t, :] + g_t * h_prev
            out.append(h_t)
            h_prev = h_t
        return torch.stack(out, dim=1)


def build_temporal_fusion(mode: str, rep_dim: int, alpha: float, hidden_dim: int = 0):
    mode = mode.lower()
    if mode == "none":
        return None
    if mode == "ema":
        return EMAFusion(alpha=alpha)
    if mode == "concat_linear":
        return ConcatLinearFusion(rep_dim=rep_dim)
    if mode == "residual_gate":
        return ResidualGateFusion(rep_dim=rep_dim)
    if mode == "concat_mlp":
        h = hidden_dim if hidden_dim and hidden_dim > 0 else rep_dim
        mlp = nn.Sequential(
            nn.Linear(rep_dim * 2, h),
            nn.ReLU(),
            nn.Linear(h, rep_dim),
        )
        return ConcatModuleFusion(rep_dim=rep_dim, module=mlp)
    raise ValueError(f"Unknown temporal_fusion_mode: {mode}")
