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

    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature
    logits_max = sim.max(dim=1, keepdim=True).values
    logits = sim - logits_max.detach()

    B = y.shape[0]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye

    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=z.device)

    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
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
            x = self.pre_model(x)
        return self.encoder(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        if self.pre_model is not None:
            with torch.no_grad():
                x = self.pre_model(x)
        acts = self.encoder.gather_layer_activations(x, no_grad=False)

        total = torch.tensor(0.0, device=x.device)
        for i, z in enumerate(acts):
            layer_loss = supervised_contrastive_loss(z, y, self.temperature)
            self.log(f"{self.ident}_layer_{i}/supcon", layer_loss, prog_bar=False)
            total = total + layer_loss

        self.log(f"{self.ident}/supcon_total", total, prog_bar=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.encoder.parameters(), lr=self.lr)


def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

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
    classifier_trainer.validate(classifier, dataloaders=val_loader)

    final_val_acc = classifier_trainer.callback_metrics.get("classifier_val_acc")
    final_val_acc = float(final_val_acc.item()) if final_val_acc is not None else None

    safe_run_name = cfg.run_name.replace(" ", "_")
    out_dir = os.path.join("saved_models", safe_run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(pre_model.state_dict(), os.path.join(out_dir, "local_supcon_encoder.pth"))
    torch.save(asdict(cfg), os.path.join(out_dir, "local_supcon_hparams.pth"))
    torch.save(classifier.state_dict(), os.path.join(out_dir, "local_supcon_classifier.pth"))

    return final_val_acc
