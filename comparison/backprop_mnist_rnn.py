# train_rnn_mnist_row_predict.py
import os
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# ---------- config ----------
@dataclass(frozen=True)
class Config:
    data_path: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 2e-3
    max_epochs: int = 10
    project_name: str = "mnist-row-rnn"
    run_name: str = "rnn_28x128_128x64_pred64to28"
    seed: int = 42
    device: str = "auto"

# ---------- dataset (row-tokenized) ----------
class MNISTRowSequenceDataset(Dataset):
    """Returns tuple (X, Y) where
    X: (S-1, 28) float tensor (rows 0..26)
    Y: (S-1, 28) float tensor (rows 1..27)
    Values in [0,1]
    """
    def __init__(self, root, train=True, download=True, transform=None):
        self.ds = torchvision.datasets.MNIST(root, train=train, download=download, transform=transforms.ToTensor())
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img: (1,H,W), float tensor 0..1
        if self.transform is not None:
            img = self.transform(img)
        img = img.squeeze(0)  # (H,W) => (28,28)
        # rows as tokens (S=28)
        rows = img  # shape (28,28)
        X = rows[:-1, :].clone()  # (27,28)
        Y = rows[1:, :].clone()   # (27,28)
        return X, Y

# ---------- simple RNN layer (manual recurrence) ----------
class RNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, out_dim, bias=bias)
        self.hh = nn.Linear(out_dim, out_dim, bias=False)
        self.act = activation()
    def forward_step(self, x_t, h_prev):
        h_in = self.in_proj(x_t) + self.hh(h_prev)
        return self.act(h_in)
    def forward(self, x_seq, h0=None):
        # x_seq: (B, S, in_dim)
        if x_seq.dim() == 2:
            # single step
            if h0 is None:
                h0 = x_seq.new_zeros(x_seq.size(0), self.out_dim)
            return self.forward_step(x_seq, h0)
        B, S, D = x_seq.shape
        if h0 is None:
            h = x_seq.new_zeros(B, self.out_dim)
        else:
            h = h0
        outs = []
        for t in range(S):
            xt = x_seq[:, t, :]
            h = self.forward_step(xt, h)
            outs.append(h)
        return torch.stack(outs, dim=1)  # (B, S, out_dim)

# ---------- TREncoder (sequence-level recurrence across layers) ----------
class TREncoder(nn.Module):
    def __init__(self, layer_dims):
        super().__init
