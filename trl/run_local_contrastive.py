from dataclasses import asdict
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from trl.config.config import Config
from trl.datasets.mnist import build_dataloaders
from trl.modules.encoder import TREncoder
from trl.trainer.head import ClassifierHead


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected (B, D) activations, got {tuple(z.shape)}")

    # Keep cosine geometry by normalizing in-function.
    z = F.normalize(z, dim=1)
    device = z.device
    y = y.contiguous().view(-1, 1)
    mask_pos = torch.eq(y, y.T).float().to(device)
    batch_size = z.shape[0]

    pos_per_row = mask_pos.sum(1) - 1
    if pos_per_row.max() < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits = torch.matmul(z, z.T) / temperature
    logits_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()

    logits_max, _ = torch.max(
        logits * logits_mask + (1.0 - logits_mask) * -1e9, dim=1, keepdim=True
    )
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    exp_logits = torch.nan_to_num(exp_logits, nan=0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_mask = mask_pos * logits_mask
    denom_pos = pos_mask.sum(dim=1)
    valid = denom_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (denom_pos + 1e-12)
    return -mean_log_prob_pos[valid].mean()


class LocalContrastiveTrainer(pl.LightningModule):
    def __init__(self, ident: str, cfg: Config, encoder: TREncoder, pre_model=None):
        super().__init__()
        self.ident = ident
        self.cfg = cfg
        self.encoder = encoder
        self.pre_model = pre_model
        self.lr = cfg.lr
        self.temperature = 0.1

    def forward(self, x):
        if self.pre_model is not None:
            with torch.no_grad():
                x = self.pre_model(x)
            x = x.detach()
        return self.encoder(x)

    def local_layer_activations(self, x):
        cur = self.encoder.prepare_input(x)
        acts = []
        for _pass_i, _unique_i, layer in self.encoder.enumerate_pass_layers():
            z = layer(cur)
            acts.append(z)
            # Enforce locality: no cross-layer backprop in local-SupCon.
            cur = z.detach()
        return acts

    def training_step(self, batch, _batch_idx):
        x, y = batch
        if self.pre_model is not None:
            with torch.no_grad():
                x = self.pre_model(x)
            x = x.detach()
        acts = self.local_layer_activations(x)

        total = torch.tensor(0.0, device=x.device)
        for i, z in enumerate(acts):
            layer_loss = supervised_contrastive_loss(z, y, self.temperature)
            self.log(f"{self.ident}_layer_{i}/supcon", layer_loss, prog_bar=False)
            total = total + layer_loss

        self.log(f"{self.ident}/supcon_total", total, prog_bar=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.encoder.parameters(), lr=self.lr)


def run(cfg: Config, return_metrics: bool = False):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)
    # Local-SupCon baseline uses standard shuffled minibatches (no coherent chunk ordering).
    cfg.data_config.use_coherent_sampler = False

    train_loader, head_train_loader, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)

    trainer_logger = None
    if cfg.logger == "wandb":
        trainer_logger = WandbLogger(project=cfg.project_name, name=f"{cfg.run_name} local_supcon")
        trainer_logger.log_hyperparams(asdict(cfg))
    elif cfg.logger == "csv":
        trainer_logger = CSVLogger("lightning_logs/", name=f"{cfg.run_name}_local_supcon")

    pre_model = None
    for i, encoder_cfg in enumerate(cfg.encoders):
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices=1,
            logger=trainer_logger,
            enable_checkpointing=False,
        )
        encoder = TREncoder(cfg, encoder_cfg)
        module = LocalContrastiveTrainer(f"e{i}", cfg, encoder, pre_model=pre_model)
        trainer.fit(module, train_loader)
        pre_model = module

    classifier = ClassifierHead(pre_model, cfg, cfg.head_out_dim)
    classifier_trainer = pl.Trainer(
        max_epochs=cfg.head_epochs,
        accelerator="auto",
        devices=1,
        logger=trainer_logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    classifier_trainer.fit(classifier, head_train_loader)
    val_results = classifier_trainer.validate(classifier, dataloaders=val_loader)
    val_results = val_results[0] if val_results else {}

    callback_metric = classifier_trainer.callback_metrics.get("classifier_val_acc")
    final_val_acc = float(callback_metric.item()) if callback_metric is not None else None
    if final_val_acc is None and "classifier_val_acc" in val_results:
        final_val_acc = float(val_results["classifier_val_acc"])

    safe_run_name = cfg.run_name.replace(" ", "_")
    out_dir = os.path.join("saved_models", safe_run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(pre_model.state_dict(), os.path.join(out_dir, "local_supcon_encoder.pth"))
    torch.save(asdict(cfg), os.path.join(out_dir, "local_supcon_hparams.pth"))
    torch.save(classifier.state_dict(), os.path.join(out_dir, "local_supcon_classifier.pth"))

    if return_metrics:
        metrics = {k: float(v) for k, v in val_results.items() if isinstance(v, (int, float))}
        return {
            "primary_metric_name": "classifier_val_acc",
            "primary_metric": final_val_acc,
            "validation_metrics": metrics,
        }
    return final_val_acc
