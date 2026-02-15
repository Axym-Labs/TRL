"""VICReg-like greedy MLP pretraining — batches composed of coherent same-class chunks.

Change in this version: the loss takes a single representation tensor `z` (interleaved pairs) and computes the invariance term as differences between consecutive samples: `diff = z[1:] - z[:-1]`.
Training now processes a single batch tensor at a time (standard training loop). The dataset returns a tensor of shape (2, C, H, W) per sample so the DataLoader collates to (B, 2, C, H, W), which we reshape to (2B, C, H, W) for the model.
"""
from dataclasses import asdict
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from trl.config.config import Config
from trl.datasets.mnist import build_dataloaders
from trl.trainer.head import ClassifierHead, RegressorHead, PredictorHead
from trl.modules.encoder import TREncoder, TRSeqEncoder, TRSeqElementwiseEncoder
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer

def run(cfg: Config, return_metrics: bool = False):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    # for MNIST, the sequence task would be to predict the next rows from the previous
    train_loader, head_train_loader, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)

    trainer_logger = None
    if cfg.logger == "wandb":
        trainer_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
        # log all hyperparameters centrally from Config
        trainer_logger.log_hyperparams(asdict(cfg))
        trainer_logger.log_hyperparams({f"encoder_{i}": asdict(v) for i, v in enumerate(cfg.encoders)})
    elif cfg.logger == "csv":
        trainer_logger = CSVLogger("lightning_logs/", name=cfg.run_name)

    pre_model = None
    for i, encoder_cfg in enumerate(cfg.encoders):
        print(f"--- Now training encoder {i} ---")
        pretrain_epochs = len(encoder_cfg.layer_dims) * cfg.epochs * encoder_cfg.recurrence_depth if not cfg.train_encoder_concurrently else cfg.epochs
        pre_trainer = pl.Trainer(max_epochs=pretrain_epochs, accelerator="auto", devices=1,
                                logger=trainer_logger, enable_checkpointing=False)

        if cfg.problem_type == "sequence":
            if cfg.sequence_recurrent_encoder:
                encoder_cls = TRSeqEncoder
                encoder_trainer_cls = SeqEncoderTrainer
            else:
                encoder_cls = TRSeqElementwiseEncoder
                encoder_trainer_cls = EncoderTrainer
        else:
            encoder_cls = TREncoder
            encoder_trainer_cls = EncoderTrainer
        encoder = encoder_cls(cfg, encoder_cfg)
        encoder_trainer = encoder_trainer_cls(f"e{i}", cfg, encoder, pre_model=pre_model)
        pre_trainer.fit(encoder_trainer, train_loader)
        pre_model = encoder_trainer

    frozen_encoder = pre_model

    # if sequence problem: predict next item in sequence
    if cfg.head_task == "classification":
        assert cfg.problem_type != "sequence", "not implemented yet"
        head_cls = ClassifierHead
    elif cfg.head_task == "regression":
        head_cls = RegressorHead if cfg.problem_type != "sequence" else PredictorHead
    else:
        raise ValueError("Unsupported head_task")
    classifier = head_cls(frozen_encoder, cfg, cfg.head_out_dim)
    classifier_trainer = pl.Trainer(max_epochs=cfg.head_epochs, accelerator="auto", devices=1,
                                    logger=trainer_logger, enable_checkpointing=False, num_sanity_val_steps=0)
    classifier_trainer.fit(classifier, head_train_loader)
    val_results = classifier_trainer.validate(classifier, dataloaders=val_loader)
    val_results = val_results[0] if val_results else {}

    val_metric_key = "classifier_val_acc" if cfg.head_task == "classification" else "val_prediction_loss"
    callback_metric = classifier_trainer.callback_metrics.get(val_metric_key)
    final_val_metric = float(callback_metric.item()) if callback_metric is not None else None
    if final_val_metric is None and val_metric_key in val_results:
        final_val_metric = float(val_results[val_metric_key])
    if final_val_metric is not None and cfg.logger == "wandb":
        summary_key = "final_val_accuracy" if cfg.head_task == "classification" else "final_val_prediction_loss"
        trainer_logger.experiment.summary[summary_key] = final_val_metric

    # ---- SAVE MODELS TO the requested directory ----
    safe_run_name = cfg.run_name.replace(" ", "_")
    out_dir = os.path.join("saved_models", safe_run_name)
    os.makedirs(out_dir, exist_ok=True)

    encoder_path = os.path.join(out_dir, "vicreg_encoder.pth")
    hparams_path = os.path.join(out_dir, "vicreg_hparams.pth")
    classifier_path = os.path.join(out_dir, "vicreg_classifier.pth")

    torch.save(frozen_encoder.state_dict(), encoder_path)
    torch.save(asdict(cfg), hparams_path)
    torch.save(classifier.state_dict(), classifier_path)

    print(f"Saved encoder -> {encoder_path}")
    print(f"Saved hparams -> {hparams_path}")
    print(f"Saved classifier -> {classifier_path}")

    if return_metrics:
        metrics = {k: float(v) for k, v in val_results.items() if isinstance(v, (int, float))}
        return {
            "primary_metric_name": val_metric_key,
            "primary_metric": final_val_metric,
            "validation_metrics": metrics,
        }
    return final_val_metric
