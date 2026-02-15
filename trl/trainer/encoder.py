import torch
import torch.nn as nn
import pytorch_lightning as pl

from trl.config.config import Config
from trl.modules.encoder import TREncoder


class EncoderTrainer(pl.LightningModule):
    def __init__(self, ident, cfg: Config, encoder: TREncoder, pre_model: nn.Module|None=None):
        super().__init__()
        self.cfg = cfg
        self.ident = ident
        self.train_concurrently = cfg.train_encoder_concurrently
        self.encoder = encoder
        self.pre_model = pre_model
        self.current_layer_idx = 0
        self.epochs_per_layer = cfg.epochs
        self.optim_cls = cfg.encoder_optim
        self.automatic_optimization = False
        self.lr = cfg.lr

    def forward(self, x: torch.Tensor, gather_layer_activations: list|None = None) -> torch.Tensor:
        if self.pre_model is not None:
            with torch.no_grad():
                x = self.pre_model(x)
            x = x.detach()
        return self.encoder(x, gather_layer_activations=gather_layer_activations)

    def training_step(self, batch, batch_idx):
        inp, _labels = batch
        if self.pre_model is not None:
            with torch.no_grad():
                inp = self.pre_model(inp)
            inp = inp.detach()
        prepared_inp = self.encoder.prepare_input(inp)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        total_loss = self._training_step_concurrent(prepared_inp) if self.train_concurrently else self._training_step_singlelayer(prepared_inp)
        self.manual_backward(total_loss)
        if self.cfg.encoder_grad_clip_norm and self.cfg.encoder_grad_clip_norm > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.cfg.encoder_grad_clip_norm),
                gradient_clip_algorithm="norm",
            )
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
        layer_params = []
        lat_params = []
        for layer in self.encoder.layers:
            lp, ltp = layer.layer_lat_params()
            layer_params.extend(lp)
            lat_params.extend(ltp)

        lat_factor = float(self.cfg.encoder_lat_lr_factor)
        if lat_factor <= 0.0:
            raise ValueError(f"encoder_lat_lr_factor must be > 0, got {lat_factor}")
        if abs(lat_factor - 1.0) < 1e-12:
            return self.optim_cls(layer_params + lat_params, lr=self.lr)

        param_groups = [
            {"params": layer_params, "lr": self.lr},
            {"params": lat_params, "lr": self.lr * lat_factor},
        ]
        return self.optim_cls(param_groups, lr=self.lr)

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

        pass_buffers = [[] for _ in range(self.encoder.pass_layer_count)]
        pass_to_unique = {}

        for t in range(seq_len):
            # No temporal backpropagation: hidden state is detached at every step.
            cur_inp = torch.cat([x[:, t, :], hidden.detach()], dim=1)
            for pass_i, unique_i, layer in self.encoder.enumerate_pass_layers():
                z = layer(cur_inp)
                pass_buffers[pass_i].append(z)
                pass_to_unique[pass_i] = unique_i
                # No cross-layer backpropagation: next layer sees detached activations.
                cur_inp = z.detach()
            hidden = cur_inp

        total_loss = torch.tensor(0.0, device=device)
        for pass_i, z_list in enumerate(pass_buffers):
            unique_i = pass_to_unique[pass_i]
            layer = self.encoder.layers[unique_i]
            z_seq = torch.stack(z_list, dim=1)
            layer.store.update_post(z_seq)
            step_loss, lateral_loss, metrics = layer.criterion(z_seq, lateral=layer.lat, store=layer.store)
            layer.store.update_last_z(z_seq)
            total_loss = total_loss + step_loss + lateral_loss
            self._log_layer_metrics(pass_i, step_loss, lateral_loss, metrics, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

