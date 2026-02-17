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

def _build_from_encoder_optim(cfg: Config, params):
    optim_ctor = cfg.encoder_optim
    if optim_ctor is None:
        return None
    # Mirror TRL behavior: use the same optimizer constructor configured in cfg.encoder_optim.
    # Support plain classes (e.g. torch.optim.SGD) and partial callables.
    try:
        return optim_ctor(params, lr=cfg.lr)
    except TypeError:
        return optim_ctor(params)

def _build_bp_optimizer_legacy(cfg: Config, params):
    # Legacy path kept for reference; no longer used in execution.
    name = str(getattr(cfg, "bp_optimizer", "adam")).lower()
    momentum = float(getattr(cfg, "bp_momentum", 0.9))
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if name == "sgdm":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=momentum)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.0)
    raise ValueError(f"Unsupported bp_optimizer '{name}'. Use one of: adam, sgdm, sgd.")


def build_bp_optimizer(cfg: Config, params):
    opt = _build_from_encoder_optim(cfg, params)
    if opt is None:
        raise ValueError("Backprop requires cfg.encoder_optim to be set.")
    return opt


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
        self.log("train_acc", acc, prog_bar=True)
        self.log("classifier_train_acc", acc, prog_bar=True)
        self.log("train_loss", loss)
        self.log("classifier_train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        self.log("classifier_val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.layers.parameters(), "lr": self.lr},
            {"params": self.head.parameters(), "lr": self.lr},
        ]
        return build_bp_optimizer(self.cfg, param_groups)

class BPSeqMLPPredictor(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = []
        for enc in cfg.encoders:
            dims.extend(enc.layer_dims)
        if len(dims) == 0:
            raise ValueError("At least one hidden layer is required for sequence backprop baseline.")
        x_dim = cfg.head_out_dim
        first_in, _ = dims[0]
        hidden_dim = first_in - x_dim
        if hidden_dim <= 0:
            raise ValueError(
                f"Invalid sequence layout: first layer in_dim={first_in} must be larger than x_dim={x_dim} "
                f"to hold concatenated [x_t, hidden]."
            )

        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in dims])
        self.act = nn.ReLU()
        self.head = nn.Linear(dims[-1][1], cfg.head_out_dim)
        self.lr = cfg.lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        h = x.new_zeros((bsz, self.hidden_dim))
        outs = []
        for t in range(seq_len):
            cur = torch.cat([x[:, t, :], h], dim=1)
            for layer in self.layers:
                cur = self.act(layer(cur))
            h = cur
            outs.append(self.head(cur))
        return torch.stack(outs, dim=1)

    def training_step(self, batch, _batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_prediction_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_prediction_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.layers.parameters(), "lr": self.lr},
            {"params": self.head.parameters(), "lr": self.lr},
        ]
        return build_bp_optimizer(self.cfg, param_groups)


class BPSeqLinearPredictor(pl.LightningModule):
    """Linear recurrent predictor trained with BPTT (no gating, no nonlinearities)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = []
        for enc in cfg.encoders:
            dims.extend(enc.layer_dims)
        if len(dims) == 0:
            raise ValueError("At least one hidden layer is required for sequence backprop baseline.")
        x_dim = cfg.head_out_dim
        first_in, _ = dims[0]
        hidden_dim = first_in - x_dim
        if hidden_dim <= 0:
            raise ValueError(
                f"Invalid sequence layout: first layer in_dim={first_in} must be larger than x_dim={x_dim} "
                f"to hold concatenated [x_t, hidden]."
            )

        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in dims])
        self.head = nn.Linear(dims[-1][1], cfg.head_out_dim)
        self.lr = cfg.lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        h = x.new_zeros((bsz, self.hidden_dim))
        outs = []
        for t in range(seq_len):
            cur = torch.cat([x[:, t, :], h], dim=1)
            for layer in self.layers:
                cur = layer(cur)
            h = cur
            outs.append(self.head(cur))
        return torch.stack(outs, dim=1)

    def training_step(self, batch, _batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_prediction_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_prediction_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.layers.parameters(), "lr": self.lr},
            {"params": self.head.parameters(), "lr": self.lr},
        ]
        return build_bp_optimizer(self.cfg, param_groups)


def run(cfg: Config, return_metrics: bool = False):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    cfg.setup_head_use_layers()
    # Backprop baseline uses standard shuffled minibatches (no coherent chunk ordering).
    cfg.data_config.use_coherent_sampler = False
    if cfg.epochs != cfg.head_epochs and cfg.problem_type != "sequence":
        raise ValueError(
            f"Backprop baseline trains encoder and output head concurrently; expected epochs == head_epochs, "
            f"got epochs={cfg.epochs}, head_epochs={cfg.head_epochs}."
        )
    train_loader, _, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)

    logger = None
    if cfg.logger == "wandb":
        logger = WandbLogger(project=cfg.project_name, name=f"{cfg.run_name} bp")
        logger.log_hyperparams(asdict(cfg))
    elif cfg.logger == "csv":
        logger = CSVLogger("lightning_logs/", name=f"{cfg.run_name}_bp")

    if cfg.problem_type == "sequence":
        if cfg.head_task != "regression":
            raise ValueError("Sequence backprop baseline currently supports regression only.")
        model = BPSeqLinearPredictor(cfg) if cfg.bp_sequence_linear else BPSeqMLPPredictor(cfg)
    else:
        model = BPMLP(cfg)
    clip_norm = float(cfg.encoder_grad_clip_norm) if cfg.encoder_grad_clip_norm is not None else 0.0
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        gradient_clip_val=clip_norm if clip_norm > 0.0 else None,
        gradient_clip_algorithm="norm" if clip_norm > 0.0 else None,
    )
    trainer.fit(model, train_loader, val_loader)
    val_results = trainer.validate(model, dataloaders=val_loader)
    val_results = val_results[0] if val_results else {}
    metric_key = "val_loss" if cfg.problem_type == "sequence" else "val_acc"
    metric = trainer.callback_metrics.get(metric_key)
    primary = float(metric.item()) if metric is not None else None
    if primary is None:
        legacy_metric_key = "val_prediction_loss" if cfg.problem_type == "sequence" else "classifier_val_acc"
        if metric_key in val_results:
            primary = float(val_results[metric_key])
        elif legacy_metric_key in val_results:
            primary = float(val_results[legacy_metric_key])

    if return_metrics:
        metrics = {k: float(v) for k, v in val_results.items() if isinstance(v, (int, float))}
        return {
            "primary_metric_name": metric_key,
            "primary_metric": primary,
            "validation_metrics": metrics,
        }
    return primary
