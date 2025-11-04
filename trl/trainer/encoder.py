import torch
import torch.nn as nn
import pytorch_lightning as pl

from trl.config.config import Config, EncoderConfig
from trl.modules.encoder import TREncoder


class EncoderTrainer(pl.LightningModule):
    def __init__(self, ident, cfg: Config, encoder_cfg: EncoderConfig, pre_model: nn.Module|None=None):
        super().__init__()
        self.ident = ident
        self.encoder_cfg = encoder_cfg
        self.train_concurrently = cfg.train_encoder_concurrently
        self.pre_model = pre_model
        self.encoder = TREncoder(cfg, encoder_cfg, layer_dims=encoder_cfg.layer_dims)
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
        inp =  self.encoder.prepare_input(inp)

        optimizers = self.optimizers()

        if self.train_concurrently:
            cur_inp = inp
            for pass_i, unique_i, layer in self.encoder.enumerate_pass_layers():
                optim_slice = optimizers[2 * unique_i : 2 * unique_i + 2]
                cur_inp, total_loss, lateral_loss, metrics = self._train_and_step_layer(cur_inp, unique_i, optim_slice)
                self._log_layer_metrics(pass_i, total_loss, lateral_loss, metrics, prog_bar=True)
        else:
            unique_layer_idx = self.current_layer_idx % self.encoder.unique_layer_count
            optim_slice = optimizers[2 * unique_layer_idx: 2 * unique_layer_idx + 2]
            inp = self.encoder.acts_before_layer(inp, self.current_layer_idx)
            _, total_loss, lateral_loss, metrics = self._train_and_step_layer(inp, unique_layer_idx, optim_slice)
            self._log_layer_metrics(self.current_layer_idx, total_loss, lateral_loss, metrics, prog_bar=True)

    def _train_and_step_layer(self, inp: torch.Tensor, layer_idx: int, optim_slice):
        enc_opt, lat_opt = optim_slice
        layer = self.encoder.layers[layer_idx]
        layer.train()

        acts, total_loss, lateral_loss, metrics = layer.compute_loss(inp)

        enc_opt.zero_grad()
        self.manual_backward(total_loss)
        enc_opt.step()

        lat_opt.zero_grad()
        self.manual_backward(lateral_loss)
        lat_opt.step()

        return acts.detach(), total_loss, lateral_loss, metrics

    def configure_optimizers(self):
        optimizers = []
        for layer in self.encoder.layers:
            layer_params, lat_params = layer.layer_lat_params()
            optimizers.append(torch.optim.Adam(layer_params, lr=self.lr))
            optimizers.append(torch.optim.Adam(lat_params, lr=self.lr))
        return optimizers

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


class SeqEncoderTrainer(EncoderTrainer): ...

