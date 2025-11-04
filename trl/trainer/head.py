
import torch
import pytorch_lightning as pl
from torch import nn as nn

from trl.config.config import Config
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer

class ClassifierHead(pl.LightningModule):
    def __init__(self, encoder: EncoderTrainer, cfg: Config, num_classes: int = 10):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.encoder.eval()
        self.mapping = nn.Linear(self.encoder.encoder.rep_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
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
    def __init__(self, encoder: SeqEncoderTrainer, cfg: Config, data_dim: int):
        super().__init__()

        self.cfg = cfg
        self.encoder = encoder
        self.encoder.eval()
        self.rep_dim = self.encoder.encoder.rep_dim
        self.mapping = nn.Linear(self.rep_dim, data_dim)
        self.criterion = nn.MSELoss()
        self.lr = cfg.lr

    def log_metric(self, out, y, is_train: bool):
        mode = "train" if is_train else "val"
        self.log(f"{mode}_prediction_loss", self.criterion(out, y))


class PredictorHead(RegressorHead):
    def __init__(self, encoder, cfg: Config, **kwargs):
        # for sake of simple implementation
        assert cfg.data_config.batch_size >= 2

        super().__init__(encoder, cfg, **kwargs)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x[:, :-1, :]
        y = x[:, 1:, :]
        out = self(x)
        loss = self.criterion(out, y)
        self.log_metric(out, y, True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x[:, :-1, :]
        y = x[:, 1:, :]
        out = self(x)
        self.log_metric(out, y, False)


