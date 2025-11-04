"""VICReg-like greedy MLP pretraining — pairs are now *two different samples of the same class* (not two augmented views).

Key changes from previous version:
- `SameClassPairDataset` wraps a base dataset (no transform) and returns a pair of images that share the same label.
- The augmentation pipeline is applied inside the pair dataset so both images get independent augmentations.
- Hyperparameters are still centralized in the frozen `Config` dataclass and logged to WandB.
"""
from dataclasses import dataclass, asdict
from typing import Tuple
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


@dataclass(frozen=True)
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "vicreg_mlp_same_class_pairs"
    data_path: str = "./data"
    seed: int = 42

    batch_size: int = 256
    epochs_per_layer: int = 40
    classifier_epochs: int = 20

    lr: float = 1e-3
    bidirectional_variance_loss: bool = False
    var_sample_factor: float = 1.0
    var_target_init: str = "ones"  # options: "ones", "rand"
    invariance_l1: bool = False

    layer_dims: Tuple[Tuple[int, int], ...] = ((28 * 28, 512), (512, 256))

    num_workers: int = 4
    pin_memory: bool = True


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
    """Given a base dataset (no transform), returns pairs of different samples that share the same class.

    Args:
        base_dataset: e.g. torchvision.datasets.MNIST(..., transform=None)
        transform: applied independently to both images (augmentations + ToTensor + Normalize)
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
        indices = self.class_to_indices[int(label)]
        if len(indices) == 1:
            idx2 = idx
        else:
            idx2 = idx
            # pick a different index with same label
            while idx2 == idx:
                idx2 = random.choice(indices)
        img2, _ = self.base[idx2]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Return in the same format as TwoCropsTransform did: ([x1, x2], label)
        return [img1, img2], label


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class RandomTargetVarianceLoss(nn.Module):
    def __init__(self, num_features: int, cfg: Config, sim_coeff: float = 25.0,
                 std_coeff: float = 25.0, cov_coeff: float = 500.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.bidirectional = cfg.bidirectional_variance_loss
        if cfg.var_target_init == "rand":
            t = torch.rand(num_features) * cfg.var_sample_factor
        else:
            t = torch.ones(num_features) * cfg.var_sample_factor
        self.register_buffer("variance_targets", t)
        self.invariance_l1 = cfg.invariance_l1

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        z_b = z_b.detach()
        if not self.invariance_l1:
            sim_loss_per_neuron = F.mse_loss(z_a, z_b, reduction='none').mean(dim=0)
        else:
            sim_loss_per_neuron = F.l1_loss(z_a, z_b, reduction='none').mean(dim=0)

        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        if self.bidirectional:
            std_loss_per_neuron = torch.abs(std_a - self.variance_targets)
        else:
            std_loss_per_neuron = F.relu(self.variance_targets - std_a)

        sim_loss_sum = sim_loss_per_neuron.sum()
        std_loss_sum = std_loss_per_neuron.sum()
        local_loss = (self.sim_coeff * sim_loss_sum +
                      self.std_coeff * std_loss_sum)

        z_a_centered = z_a - z_a.mean(dim=0)
        cov_a = (z_a_centered.T @ z_a_centered) / (z_a.shape[0] - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum() / z_a.shape[1]

        total_loss = local_loss + self.cov_coeff * cov_loss

        loss_dict = {'sim': sim_loss_sum.detach(),
                     'std': std_loss_sum.detach(),
                     'cov': cov_loss.detach()}

        return total_loss, loss_dict


class GreedyMLPEncoder(nn.Module):
    def __init__(self, layer_dims: Tuple[Tuple[int, int], ...]):
        super().__init__()
        self.rep_dim = layer_dims[-1][1]
        self.layers = nn.ModuleList()
        for in_dim, out_dim in layer_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
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
        opt = self.optimizers()[self.current_layer_idx]
        criterion = self.criterions[self.current_layer_idx]
        (x1, x2), _ = batch  # now x1 and x2 are independent samples sharing class label

        inp1 = x1.view(x1.size(0), -1)
        inp2 = x2.view(x2.size(0), -1)

        with torch.no_grad():
            for i in range(self.current_layer_idx):
                self.encoder.layers[i].eval()
                inp1 = self.encoder.layers[i](inp1)
                inp2 = self.encoder.layers[i](inp2)

        current_layer = self.encoder.layers[self.current_layer_idx]
        current_layer.train()
        y1 = current_layer(inp1)
        y2 = current_layer(inp2)

        loss, loss_dict = criterion(y1, y2)
        self.log(f'layer_{self.current_layer_idx}/vicreg_loss', loss, prog_bar=True)
        self.log_dict({f'layer_{self.current_layer_idx}/loss_{k}': v for k, v in loss_dict.items()})

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_optimizers(self):
        return [torch.optim.Adam(layer.parameters(), lr=self.lr) for layer in self.encoder.layers]

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
    # For pretraining we need a base dataset without transforms so the wrapper can sample two imgs
    base_train = torchvision.datasets.MNIST(cfg.data_path, train=True, download=True, transform=None)
    train_pairs = SameClassPairDataset(base_train, transform=get_mnist_augment())

    val_ds = torchvision.datasets.MNIST(cfg.data_path, train=False, download=True, transform=get_mnist_val_transform())

    train_loader = DataLoader(train_pairs, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader


def run(cfg: Config):
    random.seed(cfg.seed)
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

    os.makedirs("saved_models", exist_ok=True)
    torch.save(frozen_encoder.state_dict(), f"saved_models/vicreg_encoder_{cfg.bidirectional_variance_loss}.pth")
    torch.save(asdict(cfg), f"saved_models/vicreg_hparams_{cfg.bidirectional_variance_loss}.pth")
    torch.save(classifier.state_dict(), f"saved_models/vicreg_classifier_{cfg.bidirectional_variance_loss}.pth")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
