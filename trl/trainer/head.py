
import torch
import pytorch_lightning as pl
from torch import nn as nn

from trl.config.config import Config
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer
from trl.modules.temporal_fusion import build_temporal_fusion
from trl.loss import TRSeqLoss
from trl.store import MappingStore


def _build_head_optimizer(cfg: Config, param_groups):
    optim_ctor = cfg.head_optim
    lr = float(cfg.head_lr if cfg.head_lr is not None else cfg.lr)
    try:
        return optim_ctor(param_groups, lr=lr)
    except TypeError:
        return optim_ctor(param_groups)


class ClassifierHead(pl.LightningModule):
    def __init__(self, encoder: EncoderTrainer, cfg: Config, out_dim: int = 10):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.encoder.eval()
        self.rep_dim = self.encoder.encoder.rep_dim if cfg.head_use_layers is None else \
            sum([self.encoder.encoder.layers[i].lin.out_features for i in cfg.head_use_layers])
        self.mapping = nn.Linear(self.rep_dim, out_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = float(cfg.head_lr if cfg.head_lr is not None else cfg.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.cfg.head_use_layers is not None:
                reps = self.encoder(x, gather_layer_activations=self.cfg.head_use_layers)
                reps = torch.cat(reps, dim=-1)
            else:
                reps = self.encoder(x)
        return self.mapping(reps.detach())
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log_metric(out, y, True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        self.log_metric(out, y, False)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.mapping.parameters(), "lr": self.lr},
        ]
        return _build_head_optimizer(self.cfg, param_groups)
    
    def log_metric(self, out, y, is_train: bool):
        acc = (out.argmax(dim=1) == y).float().mean()
        mode = "train" if is_train else "val"
        self.log(f'{mode}_acc', acc, prog_bar=True)
        self.log(f'classifier_{mode}_acc', acc, prog_bar=True)


class RegressorHead(ClassifierHead):
    def __init__(self, encoder: EncoderTrainer, cfg: Config, out_dim: int):
        super().__init__(encoder, cfg, out_dim)
        self.criterion = nn.MSELoss()

    def log_metric(self, out, y, is_train: bool):
        mode = "train" if is_train else "val"
        self.log(f"{mode}_loss", self.criterion(out, y))
        self.log(f"{mode}_prediction_loss", self.criterion(out, y))


class PredictorHead(RegressorHead):
    def __init__(self, encoder: SeqEncoderTrainer, cfg: Config, out_dim: int, **kwargs):
        # for sake of simple implementation
        assert cfg.data_config.batch_size >= 2

        super().__init__(encoder, cfg, out_dim, **kwargs)
        # Sequence path uses the final encoder representation tensor.
        # Gathered layer activations are not implemented for TRSeqEncoder.
        self.rep_dim = self.encoder.encoder.rep_dim
        self.mapping = nn.Linear(self.rep_dim, out_dim)
        self.temporal_fusion = build_temporal_fusion(
            mode=cfg.temporal_fusion_mode,
            rep_dim=self.rep_dim,
            alpha=cfg.temporal_fusion_alpha,
            hidden_dim=cfg.temporal_fusion_hidden_dim,
        )
        self.temporal_fusion_trl_coeff = cfg.temporal_fusion_trl_coeff
        self.fusion_criterion = None
        self.fusion_lat = None
        self.fusion_store = None
        if self.temporal_fusion is not None and self.temporal_fusion_trl_coeff > 0.0:
            self.fusion_criterion = TRSeqLoss(self.rep_dim, cfg.trloss_config, rep_tracker=None)
            self.fusion_lat = nn.Linear(self.rep_dim, self.rep_dim, bias=False)
            self.fusion_store = MappingStore(cfg.store_config, self.rep_dim, cfg.problem_type)

    def _encode_and_fuse(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reps = self.encoder(x)
        y_seq = reps.detach()
        return self.temporal_fusion(y_seq) if self.temporal_fusion is not None else y_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq = self._encode_and_fuse(x)
        return self.mapping(h_seq)

    def training_step(self, batch, batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        h_seq = self._encode_and_fuse(x)
        out = self.mapping(h_seq)
        pred_loss = self.criterion(out, y)
        loss = pred_loss
        if self.fusion_criterion is not None:
            self.fusion_store.update_post(h_seq)
            vicreg_loss, lateral_loss, _metrics = self.fusion_criterion(h_seq, lateral=self.fusion_lat, store=self.fusion_store)
            self.fusion_store.update_last_z(h_seq)
            fusion_trl_loss = vicreg_loss + lateral_loss
            self.log("train_fusion_trl_loss", fusion_trl_loss, prog_bar=True)
            loss = loss + self.temporal_fusion_trl_coeff * fusion_trl_loss
        self.log_metric(out, y, True)
        self.log("train_total_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        out = self(x)
        self.log_metric(out, y, False)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.mapping.parameters(), "lr": self.lr},
        ]
        if self.temporal_fusion is not None:
            param_groups.append({"params": self.temporal_fusion.parameters(), "lr": self.lr})
        if self.fusion_lat is not None:
            param_groups.append({"params": self.fusion_lat.parameters(), "lr": self.lr})
        return _build_head_optimizer(self.cfg, param_groups)


