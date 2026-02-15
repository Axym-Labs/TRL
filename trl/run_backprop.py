from dataclasses import asdict
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import nn
import torch.nn.functional as F

from trl.config.config import Config
from trl.datasets.mnist import build_dataloaders


class BPMLP(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = []
        for enc in cfg.encoders:
            dims.extend(enc.layer_dims)
        in_dims = [d[0] for d in dims]
        out_dims = [d[1] for d in dims]
        if len(in_dims) == 0:
            raise ValueError("At least one hidden layer is required for backprop baseline.")

        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(in_dims, out_dims)])
        self.act = nn.ReLU()
        self.head_use_layers = cfg.head_use_layers
        rep_dim = out_dims[-1] if self.head_use_layers is None else sum(out_dims[i] for i in self.head_use_layers)
        self.head = nn.Linear(rep_dim, cfg.head_out_dim)
        self.lr = cfg.lr

    def forward(self, x):
        x = x.view(x.size(0), -1)
        acts = []
        for layer in self.layers:
            x = self.act(layer(x))
            acts.append(x)
        rep = acts[-1] if self.head_use_layers is None else torch.cat([acts[i] for i in self.head_use_layers], dim=-1)
        return self.head(rep)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log("bp_train_acc", acc, prog_bar=True)
        self.log("bp_train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log("bp_val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    cfg.setup_head_use_layers()
    train_loader, _, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)

    logger = None
    if cfg.logger == "wandb":
        logger = WandbLogger(project=cfg.project_name, name=f"{cfg.run_name} bp")
        logger.log_hyperparams(asdict(cfg))
    elif cfg.logger == "csv":
        logger = CSVLogger("lightning_logs/", name=f"{cfg.run_name}_bp")

    model = BPMLP(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs + cfg.head_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, dataloaders=val_loader)
    metric = trainer.callback_metrics.get("bp_val_acc")
    return float(metric.item()) if metric is not None else None
