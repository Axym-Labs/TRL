"""
Train a CIFAR-100 classifier using convolutional *architecture without parameter sharing*
(a.k.a. locally-connected layers) similar in spirit to Hinton's "convnets without weight sharing".
Still uses standard backprop. Uses PyTorch Lightning and a dataclass for config.

No argparse. Edit Config defaults at the bottom if you want to change hyperparams.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torch.optim as optim
from torch import Tensor
import random
import numpy as np


# ----------------------------
# Utility: compute conv output
# ----------------------------
def conv_output_dim(in_size: int, kernel: int, padding: int, stride: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1


# ----------------------------
# LocallyConnected2d
# ----------------------------
class LocallyConnected2d(nn.Module):
    """
    A 2D locally connected layer: like Conv2d but *no parameter sharing* across spatial locations.
    Implementation uses nn.Unfold and a per-output-location weight matrix.

    Params:
      in_channels, out_channels, kernel_size, input_size (H, W),
      stride, padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.input_size = input_size  # (H, W)

        # compute output spatial dimensions
        out_h = conv_output_dim(input_size[0], self.kernel_size, self.padding, self.stride)
        out_w = conv_output_dim(input_size[1], self.kernel_size, self.padding, self.stride)
        self.out_h = out_h
        self.out_w = out_w
        self.L = out_h * out_w  # number of distinct spatial locations

        # each spatial location has its own weight: (L, in_features, out_channels)
        in_features = in_channels * (self.kernel_size ** 2)
        self.in_features = in_features

        weight = torch.empty(self.L, in_features, out_channels)
        # bias per location/out_channel
        if bias:
            bias_p = torch.empty(self.L, out_channels)
        else:
            bias_p = None

        # register parameters
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(bias_p)
        else:
            self.register_parameter('bias', None)

        # unfold operator (no params) will be created at forward time
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-like init per location
        for i in range(self.L):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W) where H,W should match input_size set at init.
        returns: (B, out_channels, out_h, out_w)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"
        # unfold -> (B, in_features, L)
        unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
        patches = unfold(x)  # (B, in_features, L)
        # transpose -> (B, L, in_features)
        patches = patches.transpose(1, 2)

        # weight: (L, in_features, out_channels)
        # compute per-location matmul: result (B, L, out_channels)
        # using einsum: 'bli,lio->blo'
        out = torch.einsum('bli,lio->blo', patches, self.weight)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)  # (1, L, out_channels)

        # reshape -> (B, out_channels, out_h, out_w)
        out = out.transpose(1, 2).contiguous()  # (B, out_channels, L)
        out = out.view(B, self.out_channels, self.out_h, self.out_w)
        return out


# ----------------------------
# Small architecture builder
# ----------------------------
class LocallyConnectedCIFAR(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(False)
        self.cfg = cfg

        # Build architecture: three locally-connected layers with pooling in between.
        # We'll compute spatial sizes explicitly so LocallyConnected2d can be initialized correctly.

        in_h, in_w = cfg.input_size
        in_ch = 3

        # Layer 1
        k1 = cfg.kernel_sizes[0]
        p1 = cfg.paddings[0]
        s1 = cfg.strides[0]
        out_h1 = conv_output_dim(in_h, k1, p1, s1)
        out_w1 = conv_output_dim(in_w, k1, p1, s1)
        self.lc1 = LocallyConnected2d(in_channels=in_ch, out_channels=cfg.channels[0],
                                      kernel_size=k1, input_size=(in_h, in_w),
                                      stride=s1, padding=p1)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) if cfg.pool else nn.Identity()
        after_pool1_h = out_h1 // 2 if cfg.pool else out_h1
        after_pool1_w = out_w1 // 2 if cfg.pool else out_w1

        # Layer 2
        k2 = cfg.kernel_sizes[1]
        p2 = cfg.paddings[1]
        s2 = cfg.strides[1]
        out_h2 = conv_output_dim(after_pool1_h, k2, p2, s2)
        out_w2 = conv_output_dim(after_pool1_w, k2, p2, s2)
        self.lc2 = LocallyConnected2d(in_channels=cfg.channels[0], out_channels=cfg.channels[1],
                                      kernel_size=k2, input_size=(after_pool1_h, after_pool1_w),
                                      stride=s2, padding=p2)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) if cfg.pool else nn.Identity()
        after_pool2_h = out_h2 // 2 if cfg.pool else out_h2
        after_pool2_w = out_w2 // 2 if cfg.pool else out_w2

        # Layer 3
        k3 = cfg.kernel_sizes[2]
        p3 = cfg.paddings[2]
        s3 = cfg.strides[2]
        out_h3 = conv_output_dim(after_pool2_h, k3, p3, s3)
        out_w3 = conv_output_dim(after_pool2_w, k3, p3, s3)
        self.lc3 = LocallyConnected2d(in_channels=cfg.channels[1], out_channels=cfg.channels[2],
                                      kernel_size=k3, input_size=(after_pool2_h, after_pool2_w),
                                      stride=s3, padding=p3)
        self.act3 = nn.ReLU(inplace=True)

        # global average pooling -> head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.channels[2], cfg.num_classes)
        )

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # logging placeholders (Lightning will handle)
        self.train_acc = 0.0
        self.val_acc = 0.0

    def forward(self, x):
        x = self.lc1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.lc2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.lc3(x)
        x = self.act3(x)

        x = self.gap(x)  # (B, C, 1, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        # return {"optimizer": opt, "lr_scheduler": sched}
        return opt


# ----------------------------
# DataModule
# ----------------------------
class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # download only
        torchvision.datasets.CIFAR100(root=self.cfg.data_root, train=True, download=True)
        torchvision.datasets.CIFAR100(root=self.cfg.data_root, train=False, download=True)

    def setup(self, stage=None):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.train_set = torchvision.datasets.CIFAR100(root=self.cfg.data_root, train=True, transform=train_transforms)
        self.val_set = torchvision.datasets.CIFAR100(root=self.cfg.data_root, train=False, transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)


# ----------------------------
# Configuration dataclass
# ----------------------------
@dataclass
class Config:
    # data
    data_root: str = "./data"
    input_size: Tuple[int, int] = (32, 32)
    num_classes: int = 100
    batch_size: int = 128
    num_workers: int = 4

    # architecture: channels for three locally-connected layers
    channels: Sequence[int] = (32, 64, 128)

    # kernel/padding/stride for each locally-connected layer
    kernel_sizes: Sequence[int] = (3, 3, 3)
    paddings: Sequence[int] = (1, 1, 1)
    strides: Sequence[int] = (1, 1, 1)

    pool: bool = True
    dropout: float = 0.0 # 0.4

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0 # 1e-4
    epochs: int = 10

    # trainer
    gpus: int = 1  # set 0 to force CPU
    max_epochs: int = 30


# ----------------------------
# Helpers: deterministic seed
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # edit config here if you want different hyperparams
    cfg = Config()
    cfg.max_epochs = cfg.epochs

    seed_everything(42)

    # instantiate data and model
    dm = CIFAR100DataModule(cfg)
    model = LocallyConnectedCIFAR(cfg)

    # trainer: auto GPU if available (honors cfg.gpus)
    use_gpu = torch.cuda.is_available() and cfg.gpus > 0
    devices = 1 if use_gpu else None
    trainer = pl.Trainer(
        # accelerator='gpu' if use_gpu else 'cpu',
        # devices=devices,
        max_epochs=cfg.max_epochs,
        precision=16 if use_gpu else 32,  # mixed precision on GPU
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    # fit
    trainer.fit(model, datamodule=dm)
