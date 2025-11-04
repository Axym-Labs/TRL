"""VICReg-like greedy MLP pretraining — batches composed of coherent same-class chunks.

Change in this version: the loss takes a single representation tensor `z` (interleaved pairs) and computes the invariance term as differences between consecutive samples: `diff = z[1:] - z[:-1]`.
Training now processes a single batch tensor at a time (standard training loop). The dataset returns a tensor of shape (2, C, H, W) per sample so the DataLoader collates to (B, 2, C, H, W), which we reshape to (2B, C, H, W) for the model.
"""
from dataclasses import dataclass, asdict
from typing import Tuple, Iterator
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


@dataclass(frozen=True)
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "vicreg_9_covar_coarse_inhib"
    data_path: str = "./data"
    seed: int = 42

    batch_size: int = 256  # must be multiple of chunk_size
    chunk_size: int = 32   # size of coherent same-class chunks
    coherence: float = 1.0  # 1.0 = perfect coherence, 0.0 = random reorder

    epochs_per_layer: int = 20
    classifier_epochs: int = 20

    lr: float = 1e-3
    bidirectional_variance_loss: bool = False
    var_sample_factor: float = 0.6 # change to 1.0
    var_target_init: str = "ones"  # options: "ones", "rand"
    invariance_l1: bool = False

    layer_dims: Tuple[Tuple[int, int], ...] = ((28 * 28, 512), (512, 256))

    num_workers: int = 4
    pin_memory: bool = True

    sim_coeff: float = 10.0
    std_coeff: float = 100.0
    cov_coeff: float = 6.0

    cov_matrix_sparsity: float = 0.0


def get_mnist_augment():
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_mnist_val_transform():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])


class SameClassPairDataset(Dataset):
    """Given a base dataset (no transform), returns a *stacked* pair tensor of different samples that share the same class.

    Returns: (pair_tensor, label) where pair_tensor has shape (2, C, H, W)
    """
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        # build class -> list of indices mapping
        self.class_to_indices = defaultdict(list)
        for idx in range(len(self.base)):
            _, label = self.base[idx]
            self.class_to_indices[int(label)].append(idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img1, label = self.base[idx]
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1, label


class DataReordering:
    @staticmethod
    def _group_indices_by_label(label_list: list[int]):
        labels = np.array(label_list)
        groups = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
        return groups

    @staticmethod
    def batch_coherence_reordering_fn(label_list: list[int], coherence: float, chunk_size: int):
        """Reorder indices such that blocks of `chunk_size` indices share the same label as much as possible.
        """
        groups = DataReordering._group_indices_by_label(label_list)
        indices_reordered = []

        while True:
            labels_sufficient_data = [k for k, g in groups.items() if len(g) > chunk_size]
            labels_select = labels_sufficient_data if len(labels_sufficient_data) > 0 else list(groups.keys())

            if len(labels_select) == 0:  # all groups exhausted
                break

            lens = [len(groups[label]) for label in labels_select]
            probs = [group_len / sum(lens) for group_len in lens]

            group_of_batch = random.choices(labels_select, probs, k=1)[0]
            indices_reordered += list(groups[group_of_batch][:chunk_size])

            groups[group_of_batch] = groups[group_of_batch][chunk_size:]
            if len(groups[group_of_batch]) == 0:
                del groups[group_of_batch]

        # then switch inplace for an according number of times
        for _ in range(int((1 - coherence) * len(indices_reordered))):
            flip_idx1 = random.randint(0, len(indices_reordered) - 1)
            flip_idx2 = random.randint(0, len(indices_reordered) - 1)
            indices_reordered[flip_idx1], indices_reordered[flip_idx2] = (
                indices_reordered[flip_idx2], indices_reordered[flip_idx1]
            )

        return indices_reordered


class CoherentSampler(Sampler):
    """Sampler that yields dataset indices reordered by `DataReordering.batch_coherence_reordering_fn`.

    It expects a `label_list` (one label per dataset index).
    """
    def __init__(self, label_list: list[int], coherence: float, chunk_size: int):
        self.label_list = list(label_list)
        self.coherence = coherence
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[int]:
        indices = DataReordering.batch_coherence_reordering_fn(self.label_list, self.coherence, self.chunk_size)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.label_list)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def effective_rank(A, eps=1e-6):
    s = torch.linalg.svdvals(A)
    s = s[s>eps]
    p = s / s.sum()
    H = - (p * torch.log(p)).sum()
    return torch.exp(H).item()


class RandomTargetVarianceLoss(nn.Module):
    """Loss that expects a single tensor `z` of shape (N, D).
    The forward() accepts an optional lateral head (nn.Module) that will be called
    with detached uncentered activations: predicted = lateral(z.detach()).
    The encoder receives -MSE(actual_centered, predicted_centered_detached).
    The lateral receives +MSE(predicted_unc, z.detach()).
    """
    def __init__(self, num_features: int, cfg: Config):
        super().__init__()
        self.sim_coeff = cfg.sim_coeff
        self.std_coeff = cfg.std_coeff
        self.cov_coeff = cfg.cov_coeff
        self.bidirectional = cfg.bidirectional_variance_loss
        if cfg.var_target_init == "rand":
            t = torch.rand(num_features) * cfg.var_sample_factor
        else:
            t = torch.ones(num_features) * cfg.var_sample_factor
        self.register_buffer("variance_targets", t)
        self.invariance_l1 = cfg.invariance_l1

        self.cov_matrix_sparisty = cfg.cov_matrix_sparsity
        self.cov_matrix_mask = None

    def forward(self, z: torch.Tensor, z_pre, inhib_det, lateral):
        """z: (N, D). lateral: module to predict z from z (will be called on z.detach())."""
        # detached mean used to center encoder input (encoder's loss uses centered actuals)
        mu = z.mean(dim=0).detach()
        z_centered = z - mu  # used in encoder cov loss

        # SIM loss (adjacent diffs) - per-sample/per-neuron
        sim_loss_pn = self.sim_loss(z)

        # STD loss (per-sample/per-neuron), returns var stat for diagnostics
        std_loss_pn, var_stat = self.std_loss(z, z_centered)
        z_cov_loss, lateral_loss_pn, lat_cov_loss, cov_stat = self.cov_loss(z, z_centered, z_pre, lateral)
        lat_scale_loss_pn = self.lat_scale_loss(z_pre, inhib_det)

        # Aggregate scalar losses (matching earlier API semantics)
        sim_loss = sim_loss_pn.mean(dim=0).sum()
        std_loss = std_loss_pn.mean(dim=0).sum()
        cov_loss = z_cov_loss.mean(dim=0).sum()
        vicreg_loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        lateral_loss = lateral_loss_pn.mean(dim=0).sum()
        lat_scale_loss = lat_scale_loss_pn.mean(dim=0).sum()

        total_loss = vicreg_loss + lateral_loss + lat_scale_loss

        loss_dict = {
            'sim': sim_loss.detach(),
            'std': std_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'var': var_stat.mean(),
            'vicreg': vicreg_loss.detach(),
            'cov': (cov_stat**2).sum() / z.shape[1],
            'lat': lateral_loss.detach(),
            'lat_cov': lat_cov_loss.detach().mean(dim=0).sum(),
            'lat_scale': lat_scale_loss.detach(),
        }
        return total_loss, loss_dict

    def sim_loss(self, z):
        diff = z[1:] - z[:-1]  # (N-1, D)
        diff = F.pad(diff, (0, 0, 0, 1), mode='constant', value=0.0)  # pad to (N,D)
        assert not self.invariance_l1
        return (diff ** 2)

    def std_loss(self, z, z_centered):
        var_stat = torch.var(z, dim=0).detach()
        var_gd = z_centered.pow(2)
        diff = self.variance_targets - var_stat
        if not self.bidirectional:
            std_loss_pn = -F.relu(diff) * var_gd
        else:
            std_loss_pn = -diff * var_gd
        return std_loss_pn, var_stat
    
    def cov_loss(self, z, z_centered, z_pre, lateral):
        if lateral is None:
            return torch.zeros_like(z_centered), torch.zeros_like(z_centered)
        
        cov_stat = (z_centered.T @ z_centered) / z.shape[0]
        cov_stat.fill_diagonal_(0.0)
        cov_stat = cov_stat.detach()

        if self.cov_matrix_mask is None:
            self.cov_matrix_mask = torch.rand_like(cov_stat) <= cfg.cov_matrix_sparsity

        cov_stat[self.cov_matrix_mask] = 0.0

        # original:
        lat_cov_loss = F.mse_loss(lateral.weight, cov_stat, reduction="none")
        lateral_loss_pn = lat_cov_loss

        # inhib = z_pre - z
        z_cov_loss = z * z_pre

        return z_cov_loss, lateral_loss_pn, lat_cov_loss, cov_stat
    
    def lat_scale_loss(self, z_pre, inhib_det):
        return F.mse_loss(inhib_det, z_pre.detach(), reduction="none")


class MappingLateralInhibition(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
        self.lat = nn.Linear(out_dim, out_dim)
        self.lat_scale = nn.Parameter(torch.nn.init.normal_(torch.empty(out_dim)))
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor):
        z_pre = self.lin(x)
        inhib_det = self.lat(z_pre.detach()).detach()
        inhib_det = inhib_det * self.lat_scale / self.out_dim
        z = z_pre - inhib_det
        return z, z_pre, inhib_det


class GreedyMLPEncoder(nn.Module):
    def __init__(self, layer_dims: Tuple[Tuple[int, int], ...]):
        super().__init__()
        self.rep_dim = layer_dims[-1][1]
        self.layers = nn.ModuleList()
        self.laterals = nn.ModuleList()
        for in_dim, out_dim in layer_dims:
            self.layers.append(MappingLateralInhibition(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x, *_ = layer(x)
        return x


class LayerWiseVICRegTrainer(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GreedyMLPEncoder(layer_dims=cfg.layer_dims)
        self.criterions = nn.ModuleList([
            RandomTargetVarianceLoss(out_dim, cfg)
            for _, out_dim in cfg.layer_dims
        ])
        self.current_layer_idx = 0
        self.epochs_per_layer = cfg.epochs_per_layer
        self.automatic_optimization = False
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        # there are 2 optimizers per layer (encoder, lateral) in the flat list
        optimizers = self.optimizers()
        enc_opt = optimizers[self.current_layer_idx]
        criterion = self.criterions[self.current_layer_idx]

        imgs, _labels = batch  # imgs: (B, C, H, W)
        inp = imgs.contiguous().view(imgs.size(0), -1)

        # pass through frozen layers
        with torch.no_grad():
            for i in range(self.current_layer_idx):
                self.encoder.layers[i].eval()
                inp, *_ = self.encoder.layers[i](inp)

        # forward through current layer
        current_layer = self.encoder.layers[self.current_layer_idx]
        current_layer.train()
        z, z_pre, inhib_det = current_layer(inp)  # (N, D)
        lateral = current_layer.lat

        # compute losses: total_loss for encoder, lateral_loss for lateral
        total_loss, loss_dict = criterion(z, z_pre, inhib_det, lateral)

        # --- encoder update (minimize total_loss) ---
        enc_opt.zero_grad()
        self.manual_backward(total_loss)
        enc_opt.step()

        # logging
        self.log(f'layer_{self.current_layer_idx}/total_loss', total_loss, prog_bar=True)
        self.log_dict({f'layer_{self.current_layer_idx}/{k}': v for k, v in loss_dict.items()})


    def configure_optimizers(self):
        optimizers = []
        for i, layer in enumerate(self.encoder.layers):
            # encoder layer optimizer
            enc_params = list(layer.parameters())
            optimizers.append(torch.optim.Adam(enc_params, lr=self.lr))
        return optimizers

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.epochs_per_layer == 0 and \
           self.current_layer_idx < len(self.encoder.layers) - 1:
            self.current_layer_idx += 1
            print(f"--- Switching to train layer {self.current_layer_idx} ---")


class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder: GreedyMLPEncoder, cfg: Config, num_classes: int = 10):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(self.encoder.rep_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = cfg.lr
        self.binarize_inputs = False

    def set_binarize_inputs(self, binarize: bool):
        self.binarize_inputs = binarize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reps = self.encoder(x)
        if self.binarize_inputs:
            reps = (reps > 0).float()
        return self.classifier(reps)

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


def build_dataloaders(cfg: Config):
    assert cfg.batch_size % cfg.chunk_size == 0, "batch_size must be a multiple of chunk_size"

    # For pretraining we need a base dataset without transforms so the wrapper can sample two imgs
    base_train = torchvision.datasets.MNIST(cfg.data_path, train=True, download=True, transform=get_mnist_augment())

    # extract labels for sampler
    label_list = [base_train[i][1] for i in range(len(base_train))]

    # sampler that enforces chunk coherence
    sampler = CoherentSampler(label_list=label_list, coherence=cfg.coherence, chunk_size=cfg.chunk_size)

    val_ds = torchvision.datasets.MNIST(cfg.data_path, train=False, download=True, transform=get_mnist_val_transform())

    train_loader = DataLoader(base_train, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader


def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg)

    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
    # log all hyperparameters centrally from Config
    wandb_logger.log_hyperparams(asdict(cfg))

    pretrain_epochs = len(cfg.layer_dims) * cfg.epochs_per_layer
    pre_trainer = pl.Trainer(max_epochs=pretrain_epochs, accelerator="auto", devices=1,
                             logger=wandb_logger, enable_checkpointing=False)

    greedy_model = LayerWiseVICRegTrainer(cfg)
    pre_trainer.fit(greedy_model, train_loader)

    frozen_encoder = greedy_model.encoder
    classifier = LinearClassifier(frozen_encoder, cfg)
    classifier_trainer = pl.Trainer(max_epochs=cfg.classifier_epochs, accelerator="auto", devices=1,
                                    logger=wandb_logger, enable_checkpointing=False, num_sanity_val_steps=0)
    classifier_trainer.fit(classifier, val_loader)
    classifier_trainer.validate(classifier, dataloaders=val_loader)

    final_val_acc = classifier_trainer.callback_metrics.get('classifier_val_acc')
    if final_val_acc is not None:
        wandb_logger.experiment.summary["final_val_accuracy"] = final_val_acc.item()

    # ---- SAVE MODELS TO the requested directory ----
    out_dir = "saved_models/vicreg_9_covar_coarse"
    os.makedirs(out_dir, exist_ok=True)

    encoder_path = os.path.join(out_dir, "vicreg_encoder.pth")
    hparams_path = os.path.join(out_dir, "vicreg_hparams.pth")
    classifier_path = os.path.join(out_dir, "vicreg_classifier.pth")

    torch.save(frozen_encoder.state_dict(), encoder_path)
    torch.save(asdict(cfg), hparams_path)
    torch.save(classifier.state_dict(), classifier_path)

    print(f"Saved encoder -> {encoder_path}")
    print(f"Saved hparams -> {hparams_path}")
    print(f"Saved classifier -> {classifier_path}")


cfg = Config()

if __name__ == "__main__":
    run(cfg)
