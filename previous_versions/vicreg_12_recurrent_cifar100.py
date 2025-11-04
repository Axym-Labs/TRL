"""
Merged VICReg-like greedy pretraining with a locally-connected (no weight-sharing)
backbone fully integrated as the encoder: each backbone layer is trained locally
with TCLoss. Minimal changes to the training logic; mapping layers expose the
same API required by EncoderTrainer.
"""
from dataclasses import dataclass, asdict
from typing import Tuple, Iterator, Optional
import os
import random
from collections import defaultdict
import contextlib

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# ----------------------------
# Helper: conv output dim
# ----------------------------
def conv_output_dim(in_size: int, kernel: int, padding: int, stride: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1

# ----------------------------
# LocallyConnected2d (no-sharing conv)
# ----------------------------
from torch import Tensor
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 input_size: Tuple[int, int], stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size  # (H, W)

        out_h = conv_output_dim(input_size[0], self.kernel_size, self.padding, self.stride)
        out_w = conv_output_dim(input_size[1], self.kernel_size, self.padding, self.stride)
        self.out_h = out_h
        self.out_w = out_w
        self.L = out_h * out_w

        in_features = in_channels * (self.kernel_size ** 2)
        self.in_features = in_features

        weight = torch.empty(self.L, in_features, out_channels)
        if bias:
            bias_p = torch.empty(self.L, out_channels)
        else:
            bias_p = None

        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(bias_p)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.L):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"
        unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
        patches = unfold(x)                        # (B, in_features, L)
        patches = patches.transpose(1, 2)          # (B, L, in_features)
        out = torch.einsum('bli,lio->blo', patches, self.weight)  # (B, L, out_channels)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        out = out.transpose(1, 2).contiguous()     # (B, out_channels, L)
        out = out.view(B, self.out_channels, self.out_h, self.out_w)
        return out

# ----------------------------
# TCLoss, BatchNorm, NormalizedMapping (unchanged)
# ----------------------------
@dataclass(frozen=True)
class BatchNormConfig:
    momentum: float = 0.1
    eps: float = 1e-5
    use_mean: bool = True
    use_variance: bool = True
    scale_parameter: bool = True
    bias_parameter: bool = True

@dataclass(frozen=True)
class TCLossConfig:
    var_target_init: str = "ones"  # options: "ones", "rand"
    var_sample_factor: float = 1.0
    bidirectional_variance_loss: bool = False

    sim_coeff: float = 12.0
    std_coeff: float = 25.0
    cov_coeff: float = 12.0
    lat_coeff: float = 12.0

    cov_matrix_sparsity: float = 0.0

class TCLoss(nn.Module):
    def __init__(self, num_features: int, cfg: TCLossConfig):
        super().__init__()
        self.cfg = cfg

        if self.cfg.var_target_init == "rand":
            t = torch.rand(num_features) * self.cfg.var_sample_factor
        else:
            t = torch.ones(num_features) * self.cfg.var_sample_factor
        self.register_buffer("variance_targets", t)

        self.cov_matrix_mask = torch.rand((num_features, num_features)) <= self.cfg.cov_matrix_sparsity

    def forward(self, z: torch.Tensor, lateral: Optional[nn.Module] = None):
        var_stat, cov_stat, z_centered = self.compute_metrics(z)

        sim_loss_pn = self.sim_loss(z_centered)
        std_loss_pn = self.std_loss(var_stat, z_centered)
        cov_loss_pn  = self.cov_loss(z_centered, lateral)
        lateral_loss_pn = self.lat_loss(cov_stat, lateral)

        sim_loss = sim_loss_pn.mean(dim=0).sum()
        std_loss = std_loss_pn.mean(dim=0).sum()
        cov_loss = cov_loss_pn.mean(dim=0).sum()
        lat_loss = lateral_loss_pn.mean(dim=0).sum()

        vicreg_loss = self.cfg.sim_coeff * sim_loss + self.cfg.std_coeff * std_loss + self.cfg.cov_coeff * cov_loss
        lateral_loss = self.cfg.lat_coeff * lat_loss

        metrics = {
            'sim_loss': sim_loss.detach(),
            'std_loss': std_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'var_stat': var_stat.mean(),
            'cov_stat': (cov_stat**2).sum() / z.shape[1],
        }
        return vicreg_loss, lateral_loss, metrics

    def compute_metrics(self, z):
        mu = z.mean(dim=0).detach()
        z_centered = z - mu

        cov_stat = (z_centered.T @ z_centered) / z_centered.shape[0]
        cov_stat = cov_stat.detach()

        var_stat = torch.var(z, dim=0).detach()

        return var_stat, cov_stat, z_centered

    def sim_loss(self, z_centered):
        diff = z_centered[1:] - z_centered[:-1]
        diff = F.pad(diff, (0, 0, 0, 1), mode='constant', value=0.0)
        return (diff ** 2)

    def std_loss(self, var_stat, z_centered):
        var_gd = z_centered.pow(2)
        diff = self.variance_targets - var_stat
        std_loss_pn = -F.relu(diff) * var_gd if not self.cfg.bidirectional_variance_loss else -diff * var_gd
        return std_loss_pn

    def cov_loss(self, z_centered, lateral):
        if lateral is None:
            return torch.zeros_like(z_centered)
        return z_centered * (lateral(z_centered.detach())).detach()

    def lat_loss(self, cov_stat, lateral):
        if lateral is None:
            return torch.tensor(0.0, device=cov_stat.device)
        cov_stat = cov_stat.clone()
        cov_stat.fill_diagonal_(0.0)
        cov_stat[self.cov_matrix_mask] = 0.0
        lateral_loss_pn = F.mse_loss(lateral.weight, cov_stat, reduction="none")
        return lateral_loss_pn

class ConfigurableBatchNorm(nn.Module):
    def __init__(self, out_dim: int, bn_cfg: BatchNormConfig):
        super().__init__()
        self.bn_cfg = bn_cfg

        self.register_buffer('running_mean', torch.zeros(out_dim))
        self.register_buffer('running_var', torch.ones(out_dim))
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, out: torch.Tensor):
        out = self.bn_train(out) if self.training else self.bn_inference(out)

        if self.bn_cfg.scale_parameter:
            out = out * self.scale
        if self.bn_cfg.bias_parameter:
            out = out + self.bias

        return out

    def bn_train(self, out):
        out_norm, batch_mean, batch_var = self.bn_normalization(out)

        with torch.no_grad():
            m = self.bn_cfg.momentum
            if self.bn_cfg.use_mean:
                self.running_mean.mul_(1 - m).add_(batch_mean * m)
            if self.bn_cfg.use_variance:
                self.running_var.mul_(1 - m).add_(batch_var * m)

        return out_norm

    def bn_inference(self, out):
        out, *_ = self.bn_normalization(out, self.running_mean, self.running_var)
        return out

    def bn_normalization(self, out, mean=None, var=None):
        if self.bn_cfg.use_mean:
            mean = mean if mean is not None else out.mean(dim=0)
            out = out - mean
        if self.bn_cfg.use_variance:
            var = var if var is not None else out.var(dim=0, unbiased=False)
            out = out / torch.sqrt(var + self.bn_cfg.eps)
        return out, mean, var

# ----------------------------
# Mapping classes
# ----------------------------
class NormalizedLinearMapping(nn.Module):
    """
    Same behavior as original NormalizedMapping (linear layer used in original script).
    Forward: (N, D_in) -> (N, D_out)
    compute_loss: returns (acts_detached, loss, lat_loss, metrics)
    """
    def __init__(self, cfg, in_dim, out_dim, norm, layer_bias, act_fn):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=layer_bias)
        self.norm = norm
        self.act_fn = act_fn
        self.lat = nn.Linear(out_dim, out_dim, bias=False)
        self.criterion = TCLoss(out_dim, cfg.tcloss_config)

    def forward(self, x: torch.Tensor):
        out = self.lin(x)
        if self.norm is not None:
            out = self.norm(out)
        return self.act_fn(out)

    def compute_loss(self, x: torch.Tensor):
        acts_pre = self.forward(x)  # (N, out_dim)
        acts = self.act_fn(acts_pre)
        loss, lat_loss, metrics = self.criterion(acts, lateral=self.lat)
        return acts.detach(), loss, lat_loss, metrics

    def layer_lat_params(self):
        layer_params = list(self.lin.parameters())
        if self.norm is not None:
            layer_params += list(self.norm.parameters())
        lat_params = list(self.lat.parameters())
        return layer_params, lat_params

class NormalizedLocalMapping(nn.Module):
    """
    Wraps a LocallyConnected2d layer. The TCLoss is computed on channel-level
    pooled vectors (global average pool across H,W), producing (N, C_out) vectors
    for TCLoss. The mapping.forward returns the image-shaped output for the
    next layer. compute_loss returns (acts_for_next_layer.detached(), loss, lat_loss, metrics).
    """
    def __init__(self, cfg, in_channels, out_channels, kernel_size, input_size,
                 stride=1, padding=0, pool=True, layer_bias=True, act_fn=nn.ReLU, bn_cfg: Optional[BatchNormConfig]=None):
        super().__init__()
        self.lc = LocallyConnected2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, input_size=input_size,
                                     stride=stride, padding=padding, bias=layer_bias)
        self.pool = nn.MaxPool2d(2,2) if pool else nn.Identity()
        self.act_fn = act_fn()
        self.bn = ConfigurableBatchNorm(out_channels, bn_cfg) if bn_cfg is not None else None
        # lateral operates on channel-dim vectors (out_channels)
        self.lat = nn.Linear(out_channels, out_channels, bias=False)
        self.criterion = TCLoss(out_channels, cfg.tcloss_config)

    def forward(self, x: torch.Tensor):
        # x: (N, C_in, H, W)
        out = self.lc(x)                   # (N, C_out, H', W')
        out = self.act_fn(out)
        out = self.pool(out)
        return out                         # image-shaped tensor

    def compute_loss(self, x: torch.Tensor):
        # x is image-shaped input for this layer
        img_out = self.forward(x)          # (N, C_out, H', W')
        # compute channel-level pooled vectors for TCLoss
        ch_vec = F.adaptive_avg_pool2d(img_out, 1).view(img_out.size(0), -1)  # (N, C_out)
        if self.bn is not None:
            ch_vec = self.bn(ch_vec)
        acts = self.act_fn(ch_vec)
        # TCLoss expects (N, D)
        vic, lat_loss, metrics = self.criterion(acts, lateral=self.lat)
        return img_out.detach(), vic, lat_loss, metrics

    def layer_lat_params(self):
        layer_params = list(self.lc.parameters())
        if self.bn is not None:
            layer_params += list(self.bn.parameters())
        lat_params = list(self.lat.parameters())
        return layer_params, lat_params

class NormalizedProjectionMapping(nn.Module):
    """
    After the final conv layer: GAP -> dropout -> linear project to rep_dim.
    compute_loss computes TCLoss on the projected vector.
    """
    def __init__(self, cfg, in_channels, out_dim, dropout=0.4, bn_cfg: Optional[BatchNormConfig]=None, layer_bias=True, act_fn=nn.ReLU):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_channels, out_dim, bias=layer_bias)
        self.bn = ConfigurableBatchNorm(out_dim, bn_cfg) if bn_cfg is not None else None
        self.act_fn = act_fn()
        self.lat = nn.Linear(out_dim, out_dim, bias=False)
        self.criterion = TCLoss(out_dim, cfg.tcloss_config)

    def forward(self, x: torch.Tensor):
        # x: image (N, C_in, H, W)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.lin(x)
        if self.bn is not None:
            out = self.bn(out)
        return self.act_fn(out)

    def compute_loss(self, x: torch.Tensor):
        # x: image input to projection layer
        proj = self.forward(x)  # (N, out_dim)
        vic, lat_loss, metrics = self.criterion(proj, lateral=self.lat)
        return proj.detach(), vic, lat_loss, metrics

    def layer_lat_params(self):
        layer_params = list(self.lin.parameters())
        if self.bn is not None:
            layer_params += list(self.bn.parameters())
        lat_params = list(self.lat.parameters())
        return layer_params, lat_params

# ----------------------------
# Combined backbone encoder
# ----------------------------
@dataclass(frozen=True)
class EncoderConfig:
    layer_dims: Tuple[Tuple[int, int], ...] = ()
    layer_bias: bool = True
    recurrence_depth: int = 1
    activaton_fn: type[nn.Module] = nn.ReLU

@dataclass(frozen=True)
class Config:
    project_name: str = "experiments_cifar100"
    run_name: str = "vicreg_locally_connected_merged"
    data_path: str = "./data"
    seed: int = 42

    batch_size: int = 128  # must be multiple of chunk_size
    chunk_size: int = 8
    coherence: float = 1.0

    train_encoder_concurrently: bool = False
    epochs: int = 10
    classifier_epochs: int = 20

    lr: float = 5e-3

    num_workers: int = 4
    pin_memory: bool = True

    tcloss_config: TCLossConfig = TCLossConfig()
    batchnorm_config: BatchNormConfig|None = BatchNormConfig()

    # backbone specification (we will build encoder from this)
    input_size: Tuple[int,int] = (32,32)
    backbone_channels: Tuple[int,int,int] = (32,64,128)
    backbone_kernel_sizes: Tuple[int,int,int] = (3,3,3)
    backbone_paddings: Tuple[int,int,int] = (1,1,1)
    backbone_strides: Tuple[int,int,int] = (1,1,1)
    backbone_pool: bool = True
    backbone_dropout: float = 0.0 # 0.4
    backbone_out_dim: int = 128

    # encoder settings (unused for building backbone, kept for compatibility)
    encoders = [EncoderConfig()]

# ----------------------------
# CombinedTCEncoder: wraps mapping layers and provides same API as old TCEncoder
# ----------------------------
class CombinedTCEncoder(nn.Module):
    """
    Constructs an ordered list of mapping layers:
      [NormalizedLocalMapping(layer1), NormalizedLocalMapping(layer2), ..., NormalizedProjectionMapping]
    Presents the same enumeration API used by EncoderTrainer.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        channels = list(cfg.backbone_channels)
        ks = list(cfg.backbone_kernel_sizes)
        pads = list(cfg.backbone_paddings)
        strides = list(cfg.backbone_strides)
        pool = cfg.backbone_pool

        self.layers = nn.ModuleList()
        cur_h, cur_w = self.input_size
        in_ch = 3
        for i, out_ch in enumerate(channels):
            k = ks[i]
            p = pads[i]
            s = strides[i]
            lm = NormalizedLocalMapping(cfg, in_channels=in_ch, out_channels=out_ch,
                                        kernel_size=k, input_size=(cur_h, cur_w),
                                        stride=s, padding=p, pool=pool,
                                        layer_bias=True, act_fn=nn.ReLU, bn_cfg=cfg.batchnorm_config)
            self.layers.append(lm)
            # update spatial dims after conv and optional pool
            cur_h = conv_output_dim(cur_h, k, p, s)
            cur_w = conv_output_dim(cur_w, k, p, s)
            if pool:
                cur_h = cur_h // 2
                cur_w = cur_w // 2
            in_ch = out_ch

        # projection mapping
        proj = NormalizedProjectionMapping(cfg, in_channels=in_ch, out_dim=cfg.backbone_out_dim,
                                           dropout=cfg.backbone_dropout, bn_cfg=cfg.batchnorm_config,
                                           layer_bias=True, act_fn=nn.ReLU)
        self.layers.append(proj)

        self.recurrence_depth = 1
        self.rep_dim = cfg.backbone_out_dim

    @property
    def unique_layer_count(self):
        return len(self.layers)

    @property
    def pass_layer_count(self):
        return self.recurrence_depth * self.unique_layer_count

    def enumerate_unique_layers(self):
        for i, layer in enumerate(self.layers):
            yield (i, layer)

    def enumerate_pass_layers(self):
        for pass_i in range(self.recurrence_depth):
            for unique_i, layer in self.enumerate_unique_layers():
                layer_pass_i = pass_i * self.unique_layer_count + unique_i
                yield layer_pass_i, unique_i, layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        cur = x
        for _, _, layer in self.enumerate_pass_layers():
            cur = layer.forward(cur)  # for local mappings returns image; projection returns vector
        return cur

    def acts_before_layer(self, x: torch.Tensor, layer_idx, no_grad=True) -> torch.Tensor:
        cur = x
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for pass_i, _, layer in self.enumerate_pass_layers():
                if pass_i >= layer_idx:
                    break
                cur = layer.forward(cur)
        return cur

# ----------------------------
# EncoderTrainer (small change: accept externally provided encoder instance)
# ----------------------------
class EncoderTrainer(pl.LightningModule):
    def __init__(self, ident, cfg: Config, encoder_cfg: Optional[EncoderConfig]=None,
                 pre_model: Optional[nn.Module]=None, encoder: Optional[nn.Module]=None):
        super().__init__()
        self.ident = ident
        self.encoder_cfg = encoder_cfg
        self.train_concurrently = cfg.train_encoder_concurrently
        self.pre_model = pre_model
        # if an encoder instance is provided, use it; otherwise build via TCEncoder path
        if encoder is not None:
            self.encoder = encoder
        else:
            # fallback: old behavior (not used in our run)
            assert encoder_cfg is not None, "Encoder config required if encoder instance not provided"
            self.encoder = CombinedTCEncoder(cfg)
        self.rep_dim = self.encoder.rep_dim
        self.current_layer_idx = 0
        self.epochs_per_layer = cfg.epochs
        self.automatic_optimization = False
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_model is not None:
            x = self.pre_model(x)
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        inp, _labels = batch
        # in original script the dataset returned single images; here keep same
        if self.pre_model is not None:
            inp = self.pre_model(inp)

        # input shape depends on first layer: image-shaped (N,C,H,W)
        optimizers = self.optimizers()

        if self.train_concurrently:
            cur_inp = inp
            for pass_i, unique_i, layer in self.encoder.enumerate_pass_layers():
                optim_slice = optimizers[2 * unique_i : 2 * unique_i + 2]
                cur_inp, total_loss, lateral_loss, metrics = self._train_and_step_layer(cur_inp, unique_i, optim_slice)
                self._log_layer_metrics(pass_i, total_loss, lateral_loss, metrics, prog_bar=True)
        else:
            unique_layer_idx = self.current_layer_idx % self.encoder.unique_layer_count
            optim_slice = optimizers[2 * unique_layer_idx: 2 * unique_layer_idx + 2]
            inp_to_layer = self.encoder.acts_before_layer(inp, self.current_layer_idx)
            _, total_loss, lateral_loss, metrics = self._train_and_step_layer(inp_to_layer, unique_layer_idx, optim_slice)
            self._log_layer_metrics(self.current_layer_idx, total_loss, lateral_loss, metrics, prog_bar=True)

    def _train_and_step_layer(self, inp: torch.Tensor, layer_idx: int, optim_slice):
        enc_opt, lat_opt = optim_slice
        layer = self.encoder.layers[layer_idx]
        layer.train()

        # layer.compute_loss expects the input shape that forward expects (image or flattened)
        acts, total_loss, lateral_loss, metrics = layer.compute_loss(inp)

        enc_opt.zero_grad()
        self.manual_backward(total_loss)
        enc_opt.step()

        lat_opt.zero_grad()
        self.manual_backward(lateral_loss)
        lat_opt.step()

        return acts.detach(), total_loss, lateral_loss, metrics

    def configure_optimizers(self):
        optimizers = []
        for layer in self.encoder.layers:
            layer_params, lat_params = layer.layer_lat_params()
            optimizers.append(torch.optim.Adam(layer_params, lr=self.lr))
            optimizers.append(torch.optim.Adam(lat_params, lr=self.lr))
        return optimizers

    def on_train_epoch_end(self):
        if not self.train_concurrently:
            if (self.current_epoch + 1) % self.epochs_per_layer == 0 and \
               self.current_layer_idx < len(self.encoder.layers) - 1:
                self.current_layer_idx += 1
                print(f"--- Switching to train layer {self.current_layer_idx} ---")

    def _log_layer_metrics(self, layer_idx: int, total_loss: torch.Tensor, lateral_loss: torch.Tensor, metrics: dict, prog_bar: bool = False):
        self.log(f'{self.ident}_layer_{layer_idx}/vicreg_loss', total_loss, prog_bar=prog_bar)
        self.log_dict({f'{self.ident}_layer_{layer_idx}/{k}': v for k, v in metrics.items()})
        self.log(f'{self.ident}_layer_{layer_idx}/lateral_loss', lateral_loss.detach(), prog_bar=False)

# ----------------------------
# LinearClassifier unchanged (just increased to 100 classes)
# ----------------------------
class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder: nn.Module, cfg: Config, num_classes: int = 100):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.encoder.eval()
        self.classifier = nn.Linear(self.encoder.rep_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reps = self.encoder(x)
        return self.classifier(reps.detach())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)

# ----------------------------
# Dataloaders adapted for CIFAR-100 (kept original reordering logic)
# ----------------------------
def get_cifar_train_transform():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_cifar_val_transform():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

class DataReordering:
    @staticmethod
    def _group_indices_by_label(label_list: list[int]):
        labels = np.array(label_list)
        groups = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
        return groups

    @staticmethod
    def batch_coherence_reordering_fn(label_list: list[int], coherence: float, chunk_size: int):
        groups = DataReordering._group_indices_by_label(label_list)
        indices_reordered = []

        while True:
            labels_sufficient_data = [k for k, g in groups.items() if len(g) > chunk_size]
            labels_select = labels_sufficient_data if len(labels_sufficient_data) > 0 else list(groups.keys())

            if len(labels_select) == 0:
                break

            lens = [len(groups[label]) for label in labels_select]
            probs = [group_len / sum(lens) for group_len in lens]

            group_of_batch = random.choices(labels_select, probs, k=1)[0]
            indices_reordered += list(groups[group_of_batch][:chunk_size])

            groups[group_of_batch] = groups[group_of_batch][chunk_size:]
            if len(groups[group_of_batch]) == 0:
                del groups[group_of_batch]

        for _ in range(int((1 - coherence) * len(indices_reordered))):
            flip_idx1 = random.randint(0, len(indices_reordered) - 1)
            flip_idx2 = random.randint(0, len(indices_reordered) - 1)
            indices_reordered[flip_idx1], indices_reordered[flip_idx2] = (
                indices_reordered[flip_idx2], indices_reordered[flip_idx1]
            )

        return indices_reordered

class CoherentSampler(Sampler):
    def __init__(self, label_list: list[int], coherence: float, chunk_size: int, shuffle: bool = False):
        self.label_list = list(label_list)
        self.coherence = coherence
        self.chunk_size = chunk_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.label_list)))
        if self.shuffle:
            random.shuffle(order)

        labels_in_order = [self.label_list[i] for i in order]
        positions_reordered = DataReordering.batch_coherence_reordering_fn(labels_in_order, self.coherence, self.chunk_size)
        indices_reordered = [order[pos] for pos in positions_reordered]
        return iter(indices_reordered)

    def __len__(self) -> int:
        return len(self.label_list)

def build_dataloaders(cfg: Config):
    assert cfg.batch_size % cfg.chunk_size == 0, "batch_size must be a multiple of chunk_size"

    base_train = torchvision.datasets.CIFAR100(cfg.data_path, train=True, download=True, transform=get_cifar_train_transform())
    label_list = [base_train[i][1] for i in range(len(base_train))]

    sampler = CoherentSampler(label_list=label_list, coherence=cfg.coherence, chunk_size=cfg.chunk_size, shuffle=False)

    val_ds = torchvision.datasets.CIFAR100(cfg.data_path, train=False, download=True, transform=get_cifar_val_transform())

    train_loader = DataLoader(base_train, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader

# ----------------------------
# Run / orchestration
# ----------------------------
def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg)

    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
    wandb_logger.log_hyperparams(asdict(cfg))

    # Build the combined backbone encoder (every backbone layer is an encoder layer)
    combined_encoder = CombinedTCEncoder(cfg)

    # Single greedy training pass over the combined encoder
    pretrain_epochs = cfg.epochs  # you can adapt schedule: epochs * num_layers if needed
    pre_trainer = pl.Trainer(max_epochs=pretrain_epochs, accelerator="auto", devices=1,
                             logger=wandb_logger, enable_checkpointing=False)

    greedy_model = EncoderTrainer("e0", cfg, pre_model=None, encoder=combined_encoder)
    pre_trainer.fit(greedy_model, train_loader)

    # after greedy training, freeze encoder and train linear classifier on top
    frozen_encoder = greedy_model
    classifier = LinearClassifier(frozen_encoder, cfg, num_classes=100)
    classifier_trainer = pl.Trainer(max_epochs=cfg.classifier_epochs, accelerator="auto", devices=1,
                                    logger=wandb_logger, enable_checkpointing=False, num_sanity_val_steps=0)
    classifier_trainer.fit(classifier, train_loader)
    classifier_trainer.validate(classifier, dataloaders=val_loader)

    final_val_acc = classifier_trainer.callback_metrics.get('classifier_val_acc')
    if final_val_acc is not None:
        wandb_logger.experiment.summary["final_val_accuracy"] = final_val_acc.item()

    out_dir = "saved_models/vicreg_locally_connected_merged"
    os.makedirs(out_dir, exist_ok=True)

    encoder_path = os.path.join(out_dir, "vicreg_encoder.pth")
    hparams_path = os.path.join(out_dir, "vicreg_hparams.pth")
    classifier_path = os.path.join(out_dir, "vicreg_classifier.pth")

    # save state dict of the greedy_model (EncoderTrainer) which contains the encoder as .encoder
    torch.save(greedy_model.encoder.state_dict(), encoder_path)
    torch.save(asdict(cfg), hparams_path)
    torch.save(classifier.state_dict(), classifier_path)

    print(f"Saved encoder -> {encoder_path}")
    print(f"Saved hparams -> {hparams_path}")
    print(f"Saved classifier -> {classifier_path}")

cfg = Config()

if __name__ == "__main__":
    run(cfg)
