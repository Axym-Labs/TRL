"""Diagnose validation-stream drift as a function of continuation step size."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from terel.resubmission.data import encoder_order, load_mnist_protocol
from terel.resubmission.evaluation import (
    classification_metrics,
    extract_online_representations,
    extract_representations,
    fit_linear_probe,
    representation_diagnostics,
)
from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    _optimizer,
    set_reproducible_seed,
)
from terel.resubmission.model import LayerLocalEncoder
from terel.resubmission.objective import LossCoefficients


def _model(config, input_dim: int, device: torch.device) -> LayerLocalEncoder:
    return LayerLocalEncoder(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        normalization=config.normalization,
        normalization_momentum=config.normalization_momentum,
        normalization_affine=config.normalization_affine,
        statistics_momentum=config.statistics_momentum,
        lateral_momentum=config.lateral_momentum,
    ).to(device)


def _load_configuration(path: Path) -> tuple[dict, EncoderExperimentConfig]:
    raw = yaml.safe_load(path.read_text())
    values = dict(raw["encoder"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    return raw, EncoderExperimentConfig(**values)


def _block_accuracies(logits, labels, order, block_size: int) -> list[float]:
    predictions = logits.argmax(dim=1)
    correct = (predictions[order] == labels[order]).to(torch.float32)
    return [
        float(correct[start : start + block_size].mean())
        for start in range(0, len(correct), block_size)
    ]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()

    audit = yaml.safe_load(arguments.config.read_text())
    if audit["evaluation_split"] != "validation":
        raise ValueError("the drift diagnostic must remain validation-only")
    base_path = Path(audit["base_config"])
    raw, encoder = _load_configuration(base_path)
    seed = int(raw["seed"])
    device = torch.device(arguments.device)
    splits = load_mnist_protocol(raw["data_root"], allow_download=False)
    checkpoint = torch.load(audit["checkpoint"], map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != seed:
        raise ValueError("checkpoint seed does not match the diagnostic configuration")

    set_reproducible_seed(seed)
    base_model = _model(encoder, splits.train.features.shape[1], device)
    base_model.load_state_dict(checkpoint["model"])
    train_representations = extract_representations(
        base_model,
        splits.train,
        batch_size=int(audit["probe"]["batch_size"]),
        device=device,
        use_all_layers=True,
    )
    offline_representations = extract_representations(
        base_model,
        splits.validation,
        batch_size=int(audit["probe"]["batch_size"]),
        device=device,
        use_all_layers=True,
    )
    probe, probe_training = fit_linear_probe(
        train_representations,
        splits.train.labels,
        num_classes=int(raw["num_classes"]),
        seed=seed + 10_000,
        epochs=int(audit["probe"]["epochs"]),
        batch_size=int(audit["probe"]["batch_size"]),
        optimizer_name=str(audit["probe"]["optimizer"]),
        learning_rate=float(audit["probe"]["learning_rate"]),
        weight_decay=float(audit["probe"]["weight_decay"]),
        device=device,
    )
    with torch.no_grad():
        offline_logits = probe(offline_representations.to(device)).cpu()
    offline_metrics = classification_metrics(
        offline_logits,
        splits.validation.labels,
        num_classes=int(raw["num_classes"]),
    )
    order, boundaries = encoder_order(
        splits.validation,
        order_mode=encoder.order_mode,
        seed=seed,
        chunk_size=encoder.chunk_size,
    )

    conditions = []
    for scale in audit["learning_rate_scales"]:
        scale = float(scale)
        model = _model(encoder, splits.train.features.shape[1], device)
        model.load_state_dict(checkpoint["model"])
        optimizer = _optimizer(
            encoder.optimizer,
            model.encoder_parameters(),
            learning_rate=scale * encoder.learning_rate,
            weight_decay=encoder.weight_decay,
            momentum=encoder.optimizer_momentum,
            beta1=encoder.optimizer_beta1,
            beta2=encoder.optimizer_beta2,
            epsilon=encoder.optimizer_epsilon,
        )
        representations, online_summary = extract_online_representations(
            model,
            optimizer,
            splits.validation.features,
            order=order,
            boundaries=boundaries,
            coefficients=LossCoefficients(
                similarity=encoder.similarity_coefficient,
                variance=encoder.variance_coefficient,
                covariance=encoder.covariance_coefficient,
            ),
            variance_target=encoder.variance_target,
            detach_previous=True,
            covariance_mode="residual_state",
            device=device,
            use_all_layers=True,
            residual_lateral_steps=encoder.residual_lateral_steps,
            residual_lateral_step_size=encoder.residual_lateral_step_size,
            residual_lateral_rule=encoder.residual_lateral_rule,
            residual_lateral_include_diagonal=(
                encoder.residual_lateral_include_diagonal
            ),
            residual_lateral_moment_normalization=(
                encoder.residual_lateral_moment_normalization
            ),
            residual_lateral_coefficient=encoder.residual_lateral_coefficient,
            residual_lateral_signal_offset=encoder.residual_lateral_signal_offset,
            postsynaptic_state_mode=encoder.postsynaptic_state_mode,
            lateral_matrix_mode=encoder.lateral_matrix_mode,
            combined_lateral_state_weight=encoder.combined_lateral_state_weight,
            temporal_term_enabled=encoder.temporal_term_enabled,
        )
        with torch.no_grad():
            logits = probe(representations.to(device)).cpu()
        metrics = classification_metrics(
            logits,
            splits.validation.labels,
            num_classes=int(raw["num_classes"]),
        )
        conditions.append(
            {
                "learning_rate_scale": scale,
                "learning_rate": scale * encoder.learning_rate,
                "metrics": metrics,
                "representation_diagnostics": representation_diagnostics(
                    representations, splits.validation.boundaries
                ),
                "stream_block_accuracy": _block_accuracies(
                    logits,
                    splits.validation.labels,
                    order,
                    int(audit["block_size"]),
                ),
                "online_summary": asdict(online_summary),
            }
        )

    _write_json(
        arguments.output,
        {
            "schema_version": 1,
            "role": "validation-only diagnostic",
            "seed": seed,
            "base_learning_rate": encoder.learning_rate,
            "checkpoint": str(Path(audit["checkpoint"]).resolve()),
            "same_trained_checkpoint": True,
            "same_fitted_probe": True,
            "offline_metrics": offline_metrics,
            "offline_representation_diagnostics": representation_diagnostics(
                offline_representations, splits.validation.boundaries
            ),
            "probe_training": asdict(probe_training),
            "conditions": conditions,
        },
    )


if __name__ == "__main__":
    main()
