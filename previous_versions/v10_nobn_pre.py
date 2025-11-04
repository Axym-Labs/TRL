"""VICReg-like greedy MLP pretraining — batches composed of coherent same-class chunks.

Change in this version: the loss takes a single representation tensor `z` (interleaved pairs) and computes the invariance term as differences between consecutive samples: `diff = z[1:] - z[:-1]`.
Training now processes a single batch tensor at a time (standard training loop). The dataset returns a tensor of shape (2, C, H, W) per sample so the DataLoader collates to (B, 2, C, H, W), which we reshape to (2B, C, H, W) for the model.
"""
from dataclasses import dataclass, asdict
from typing import Tuple, Iterator
import os
import random
from collections import defaultdict
import contextlib

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

    sim_coeff: float = 10.0
    std_coeff: float = 25.0
    cov_coeff: float = 12.0
    lat_coeff: float = 12.0

    cov_matrix_sparsity: float = 0.0
    
@dataclass(frozen=True)
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "v10_nobn_pre"
    data_path: str = "./data"
    seed: int = 42

    batch_size: int = 64  # must be multiple of chunk_size
    chunk_size: int = 8 # size of coherent same-class chunks
    coherence: float = 1.0  # 1.0 = perfect coherence, 0.0 = random reorder

    train_encoder_concurrently: bool = True
    epochs: int = 40
    classifier_epochs: int = 30

    lr: float = 1e-3

    layer_dims: Tuple[Tuple[int, int], ...] = ((28 * 28, 512), (512, 256))

    num_workers: int = 4
    pin_memory: bool = True

    tcloss_config: TCLossConfig = TCLossConfig()
    batchnorm_config: BatchNormConfig|None = BatchNormConfig()

    encoder_layer_bias: bool = True


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
    """Sampler that yields dataset indices reordered by
    DataReordering.batch_coherence_reordering_fn while optionally shuffling
    the dataset order *without* breaking index<->label correspondence.
    """
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


class TCLoss(nn.Module):
    """Loss that expects a single tensor `z` of shape (N, D).
    The forward() accepts an optional lateral head (nn.Module) that will be called
    with detached uncentered activations: predicted = lateral(z.detach()).
    The encoder receives -MSE(actual_centered, predicted_centered_detached).
    The lateral receives +MSE(predicted_unc, z.detach()).
    """
    def __init__(self, num_features: int, cfg: TCLossConfig):
        super().__init__()
        self.cfg = cfg

        if self.cfg.var_target_init == "rand":
            t = torch.rand(num_features) * self.cfg.var_sample_factor
        else:
            t = torch.ones(num_features) * self.cfg.var_sample_factor
        self.register_buffer("variance_targets", t)

        self.cov_matrix_mask = torch.rand((num_features, num_features)) <= self.cfg.cov_matrix_sparsity

    def forward(self, z: torch.Tensor, lateral: nn.Module|None = None):
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
            'lat_loss': lat_loss.detach(),
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
        # originally, we compute the MSE on z, but this is mathematically
        # equivalent to the MSE on z_centered
        diff = z_centered[1:] - z_centered[:-1] # (N-1, D)
        diff = F.pad(diff, (0, 0, 0, 1), mode='constant', value=0.0) # pad to (N,D)
        return (diff ** 2)

    def std_loss(self, var_stat, z_centered):
        var_gd = z_centered.pow(2)
        diff = self.variance_targets - var_stat
        std_loss_pn = -F.relu(diff) * var_gd if not self.cfg.bidirectional_variance_loss else -diff * var_gd
        return std_loss_pn
    
    def cov_loss(self, z_centered, lateral):
        return z_centered * (lateral(z_centered.detach())).detach()
    
    def lat_loss(self, cov_stat, lateral):
        # no self-connections
        cov_stat.fill_diagonal_(0.0)
        # artificially sparsen with persistant mask
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


class NormalizedMapping(nn.Module):
    """
    Base layer, consisting of linear, custom batchnorm, relu and lateral layer
    """
    def __init__(self, cfg: Config, in_dim, out_dim, norm):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=cfg.encoder_layer_bias)
        self.norm = norm 
        self.relu = nn.ReLU()
        self.lat = nn.Linear(out_dim, out_dim, bias=False)
        self.criterion = TCLoss(out_dim, cfg.tcloss_config)

    def forward(self, x: torch.Tensor):
        out = self.lin(x)

        if self.norm is not None:
            out = self.norm(out)

        return self.relu(out)
    
    def compute_loss(self, x: torch.Tensor):
        acts_pre = self.forward(x)
        acts = self.relu(acts_pre)
        loss, *losses_metrics = self.criterion(acts, lateral=self.lat)

        # mean = acts_pre.mean(dim=0).detach()
        # loss = loss + 12.0 * (acts_pre * mean**2).mean(dim=0).sum()

        return (acts, loss, *losses_metrics)

    def layer_lat_params(self):
        layer_params = list(self.lin.parameters())
        if self.norm is not None:
            layer_params += self.norm.parameters()

        lat_params = self.lat.parameters()

        return layer_params, lat_params


class GreedyMLPEncoder(nn.Module):
    def __init__(self, cfg, layer_dims: Tuple[Tuple[int, int], ...]):
        super().__init__()
        self.rep_dim = layer_dims[-1][1]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in layer_dims:
            norm = ConfigurableBatchNorm(out_dim, cfg.batchnorm_config) if cfg.batchnorm_config is not None else None
            self.layers.append(NormalizedMapping(cfg, in_dim, out_dim, norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def acts_at_layer(self, x: torch.Tensor, layer_idx, no_grad=True) -> torch.Tensor:
        x = self.flatten(x)

        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            for i in range(layer_idx):
                self.layers[i].eval()
                x = self.layers[i](x)

        return x
        

class LayerWiseVICRegTrainer(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GreedyMLPEncoder(cfg, layer_dims=cfg.layer_dims)
        self.current_layer_idx = 0
        self.epochs_per_layer = cfg.epochs
        self.automatic_optimization = False
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        inp, _labels = batch
        inp = inp.contiguous().view(inp.size(0), -1)
        optimizers = self.optimizers()

        if self.cfg.train_encoder_concurrently:
            cur_inp = inp
            for i in range(len(self.encoder.layers)):
                optim_slice = optimizers[2 * i : 2 * i + 2]
                cur_inp, total_loss, lateral_loss, metrics = self._train_and_step_layer(cur_inp, i, optim_slice)
                self._log_layer_metrics(i, total_loss, lateral_loss, metrics, prog_bar=True)
        else:
            optim_slice = optimizers[2 * self.current_layer_idx : 2 * self.current_layer_idx + 2]
            inp = self.encoder.acts_at_layer(inp, self.current_layer_idx)
            _, total_loss, lateral_loss, metrics = self._train_and_step_layer(inp, self.current_layer_idx, optim_slice)
            self._log_layer_metrics(self.current_layer_idx, total_loss, lateral_loss, metrics, prog_bar=True)

    def _train_and_step_layer(self, inp: torch.Tensor, layer_idx: int, optim_slice):
        enc_opt, lat_opt = optim_slice
        layer = self.encoder.layers[layer_idx]
        layer.train()

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
        if not self.cfg.train_encoder_concurrently:
            if (self.current_epoch + 1) % self.epochs_per_layer == 0 and \
            self.current_layer_idx < len(self.encoder.layers) - 1:
                self.current_layer_idx += 1
                print(f"--- Switching to train layer {self.current_layer_idx} ---")

    def _log_layer_metrics(self, layer_idx: int, total_loss: torch.Tensor, lateral_loss: torch.Tensor, metrics: dict, prog_bar: bool = False):
        self.log(f'layer_{layer_idx}/vicreg_loss', total_loss, prog_bar=prog_bar)
        self.log_dict({f'layer_{layer_idx}/{k}': v for k, v in metrics.items()})
        self.log(f'layer_{layer_idx}/lateral_loss', lateral_loss.detach(), prog_bar=False)


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
    # augmenations only for the encoder are helpful, but dispensible
    # augmentations for both encoder and head or none are approximately equal in performance
    base_train = torchvision.datasets.MNIST(cfg.data_path, train=True, download=True, transform=get_mnist_val_transform())

    # extract labels for sampler
    label_list = [base_train[i][1] for i in range(len(base_train))]

    # sampler that enforces chunk coherence
    sampler = CoherentSampler(label_list=label_list, coherence=cfg.coherence, chunk_size=cfg.chunk_size, shuffle=False)

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

    pretrain_epochs = len(cfg.layer_dims) * cfg.epochs if not cfg.train_encoder_concurrently else cfg.epochs
    pre_trainer = pl.Trainer(max_epochs=pretrain_epochs, accelerator="auto", devices=1,
                             logger=wandb_logger, enable_checkpointing=False)

    greedy_model = LayerWiseVICRegTrainer(cfg)
    pre_trainer.fit(greedy_model, train_loader)

    frozen_encoder = greedy_model.encoder
    classifier = LinearClassifier(frozen_encoder, cfg)
    classifier_trainer = pl.Trainer(max_epochs=cfg.classifier_epochs, accelerator="auto", devices=1,
                                    logger=wandb_logger, enable_checkpointing=False, num_sanity_val_steps=0)
    classifier_trainer.fit(classifier, train_loader) # was: val_loader
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
