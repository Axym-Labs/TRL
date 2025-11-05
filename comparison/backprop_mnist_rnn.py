# train_bp_rnn_mnist.py
import os
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

@dataclass(frozen=True)
class Config:
    data_path: str = "./data"
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 1e-3
    max_epochs: int = 20
    project_name: str = "experiments_mnist"
    run_name: str = "bp_rnn"
    seed: int = 42
    device: str = "auto"

class MNISTRowSequenceDataset(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.ds = torchvision.datasets.MNIST(root, train=train, download=download, transform=transforms.ToTensor())
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]            # (1,28,28)
        img = img.squeeze(0)            # (28,28)
        X = img[:-1, :].float()         # (27,28) rows 0..26
        Y = img[1:, :].float()          # (27,28) rows 1..27
        return X, Y

class DeepRNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=28, hidden_size=128, batch_first=True, nonlinearity='tanh')
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64, batch_first=True, nonlinearity='tanh')
    def forward(self, x):  # x: (B, S, 28)
        out1, _ = self.rnn1(x)      # (B, S, 128)
        out2, _ = self.rnn2(out1)   # (B, S, 64)
        return out2                 # (B, S, 64)

class RowPredictorModule(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(dict(cfg.__dict__))
        self.cfg = cfg
        self.encoder = DeepRNNEncoder()
        self.pred = nn.Linear(64, 28)
        self.criterion = nn.MSELoss(reduction="mean")
        self.lr = cfg.lr
    def forward(self, x):
        h = self.encoder(x)            # (B, S, 64)
        preds = self.pred(h)           # (B, S, 28)
        return preds
    def training_step(self, batch, batch_idx):
        X, Y = batch                   # X,Y: (B, S, 28)
        preds = self.forward(X)
        loss = self.criterion(preds, Y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        preds = self.forward(X)
        loss = self.criterion(preds, Y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def build_loaders(cfg: Config):
    train_ds = MNISTRowSequenceDataset(cfg.data_path, train=True, download=True)
    val_ds = MNISTRowSequenceDataset(cfg.data_path, train=False, download=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader

def main():
    cfg = Config()
    pl.seed_everything(cfg.seed)
    train_loader, val_loader = build_loaders(cfg)

    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
    model = RowPredictorModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.device,
        # devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
