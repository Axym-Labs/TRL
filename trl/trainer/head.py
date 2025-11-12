
import torch
import pytorch_lightning as pl
from torch import nn as nn

from trl.config.config import Config
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer

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
        self.lr = cfg.lr

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
        return torch.optim.Adam(self.mapping.parameters(), lr=self.lr)
    
    def log_metric(self, out, y, is_train: bool):
        acc = (out.argmax(dim=1) == y).float().mean()
        mode = "train" if is_train else "val"
        self.log(f'classifier_{mode}_acc', acc, prog_bar=True)


class RegressorHead(ClassifierHead):
    def __init__(self, encoder: EncoderTrainer, cfg: Config, out_dim: int):
        super().__init__(encoder, cfg, out_dim)
        self.criterion = nn.MSELoss()

    def log_metric(self, out, y, is_train: bool):
        mode = "train" if is_train else "val"
        self.log(f"{mode}_prediction_loss", self.criterion(out, y))


class PredictorHead(RegressorHead):
    def __init__(self, encoder: SeqEncoderTrainer, cfg: Config, out_dim: int, **kwargs):
        # for sake of simple implementation
        assert cfg.data_config.batch_size >= 2

        super().__init__(encoder, cfg, out_dim, **kwargs)

    def training_step(self, batch, batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        out = self(x)
        loss = self.criterion(out, y)
        self.log_metric(out, y, True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_orig, _ = batch
        x = x_orig[:, :-1, :]
        y = x_orig[:, 1:, :]
        out = self(x)
        self.log_metric(out, y, False)


