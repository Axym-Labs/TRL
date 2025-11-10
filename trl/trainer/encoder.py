import torch
import torch.nn as nn
import pytorch_lightning as pl

from trl.config.config import Config
from trl.modules.encoder import TREncoder


class EncoderTrainer(pl.LightningModule):
    def __init__(self, ident, cfg: Config, encoder: TREncoder, pre_model: nn.Module|None=None):
        super().__init__()
        self.ident = ident
        self.train_concurrently = cfg.train_encoder_concurrently
        self.encoder = encoder
        self.pre_model = pre_model
        self.current_layer_idx = 0
        self.epochs_per_layer = cfg.epochs
        self.automatic_optimization = False
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_model is not None:
            x = self.pre_model(x)
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        inp, _labels = batch
        if self.pre_model is not None:
            inp = self.pre_model(inp)
        prepared_inp = self.encoder.prepare_input(inp)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        total_loss = self._training_step_concurrent(prepared_inp) if self.train_concurrently else self._training_step_singlelayer(prepared_inp)
        self.manual_backward(total_loss)    
        optimizer.step()

    def _training_step_concurrent(self, prepared_inp: torch.Tensor):
        """Default concurrent training: run a pass through all layers"""
        cur_inp = prepared_inp
        # make sure total_loss is a tensor on the right device so we can add
        # per-layer losses without type errors
        device = prepared_inp.device if isinstance(prepared_inp, torch.Tensor) else next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        for pass_i, unique_i, _layer in self.encoder.enumerate_pass_layers():
            cur_inp, step_loss, lateral_loss, metrics = self._train_and_step_layer(cur_inp, unique_i)
            total_loss = total_loss + step_loss + lateral_loss
            self._log_layer_metrics(pass_i, total_loss, lateral_loss, metrics, prog_bar=True)
        return total_loss

    def _training_step_singlelayer(self, prepared_inp: torch.Tensor):
        """Train only the current layer (non-concurrent mode)."""
        unique_layer_idx = self.current_layer_idx % self.encoder.unique_layer_count
        inp_before = self.encoder.acts_before_layer(prepared_inp, self.current_layer_idx)
        _, total_loss, lateral_loss, metrics = self._train_and_step_layer(inp_before, unique_layer_idx)
        self._log_layer_metrics(self.current_layer_idx, total_loss, lateral_loss, metrics, prog_bar=True)
        return total_loss + lateral_loss

    def _train_and_step_layer(self, inp, layer_idx: int):
        layer = self.encoder.layers[layer_idx]
        layer.train()

        acts, total_loss, lateral_loss, metrics = layer.training_pass(inp)

        return acts.detach(), total_loss, lateral_loss, metrics

    def configure_optimizers(self):
        all_params = []
        for layer in self.encoder.layers:
            layer_params, lat_params = layer.layer_lat_params()
            all_params.extend(layer_params)
            all_params.extend(lat_params)
        return torch.optim.Adam(all_params, lr=self.lr)

    def on_train_epoch_end(self):
        for i, layer in  self.encoder.enumerate_unique_layers():
            self.log_dict({f"{self.ident}_layer_{i}/{k}": v for k, v in layer.criterion.epoch_metrics().items()})

        if not self.train_concurrently:
            if (self.current_epoch + 1) % self.epochs_per_layer == 0 and \
            self.current_layer_idx < len(self.encoder.layers) - 1:
                self.current_layer_idx += 1
                print(f"--- Switching to train layer {self.current_layer_idx} ---")

    def _log_layer_metrics(self, layer_idx: int, total_loss: torch.Tensor, lateral_loss: torch.Tensor, metrics: dict, prog_bar: bool = False):
        self.log(f'{self.ident}_layer_{layer_idx}/vicreg_loss', total_loss, prog_bar=prog_bar)
        self.log_dict({f'{self.ident}_layer_{layer_idx}/{k}': v for k, v in metrics.items()})
        self.log(f'{self.ident}_layer_{layer_idx}/lateral_loss', lateral_loss.detach(), prog_bar=False)


class SeqEncoderTrainer(EncoderTrainer):
    def _training_step_concurrent(self, prepared_inp):
        x, hidden = prepared_inp
        seq_len = x.size(1)
        device = x.device

        optimizer = self.optimizers()

        for t in range(seq_len):
            cur_inp = torch.cat([x[:, t, :], hidden], dim=1)
            total_loss_t = torch.tensor(0.0, device=device)
            for pass_i, unique_i, layer in self.encoder.enumerate_pass_layers():
                cur_inp, step_loss, lateral_loss, metrics = self._train_and_step_layer(cur_inp, unique_i)
                total_loss_t = total_loss_t + step_loss + lateral_loss
                self._log_layer_metrics(pass_i, step_loss, lateral_loss, metrics, prog_bar=True)

            optimizer.zero_grad()
            self.manual_backward(total_loss_t)
            optimizer.step()

            hidden = cur_inp

        return torch.tensor(0.0, device=device)

    def training_step(self, batch, batch_idx):
        if not self.train_concurrently:
            return super().training_step(batch, batch_idx)

        inp, _labels = batch
        if self.pre_model is not None:
            inp = self.pre_model(inp)
        prepared_inp = self.encoder.prepare_input(inp)
        self._training_step_concurrent(prepared_inp)
        return None

