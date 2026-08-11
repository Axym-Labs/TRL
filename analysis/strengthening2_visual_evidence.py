"""Generate frozen representation and neuron-state evidence for TeReL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from terel.resubmission.data import (
    DatasetSplits,
    TemporalTensorDataset,
    encoder_order,
    load_mnist_protocol,
)
from terel.resubmission.evaluation import (
    class_structure_diagnostics,
    extract_representations,
    representation_diagnostics,
)
from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    _optimizer,
    set_reproducible_seed,
)
from terel.resubmission.model import LayerLocalEncoder
from terel.resubmission.objective import LossCoefficients
from terel.resubmission.provenance import git_provenance
from terel.resubmission.training import local_train_step, train_local_encoder


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_capture24_cache(path: Path) -> DatasetSplits:
    values = np.load(path)

    def dataset(prefix: str) -> TemporalTensorDataset:
        return TemporalTensorDataset(
            features=torch.from_numpy(values[f"{prefix}_features"]).to(torch.float32),
            labels=torch.from_numpy(values[f"{prefix}_labels"]).to(torch.long),
            boundaries=torch.from_numpy(values[f"{prefix}_boundaries"]).to(torch.bool),
        )

    train = dataset("train")
    validation = dataset("validation")
    return DatasetSplits(
        train=train,
        validation=validation,
        test=validation,
        metadata={
            "dataset": "CAPTURE-24",
            "cache": str(path.resolve()),
            "train_participants": values["train_participants"].tolist(),
            "validation_participants": values["validation_participants"].tolist(),
        },
    )


def _load_configuration(path: Path) -> tuple[dict, EncoderExperimentConfig]:
    raw = yaml.safe_load(path.read_text())
    values = dict(raw["encoder"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    return raw, EncoderExperimentConfig(**values)


def _train(
    splits: DatasetSplits,
    config: EncoderExperimentConfig,
    *,
    seed: int,
    device: torch.device,
):
    if config.method != "terel_residual":
        raise ValueError("visual evidence requires the residual-state TeReL method")
    set_reproducible_seed(seed)
    model = LayerLocalEncoder(
        input_dim=splits.train.features.shape[1],
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        normalization=config.normalization,
        normalization_momentum=config.normalization_momentum,
        normalization_affine=config.normalization_affine,
        statistics_momentum=config.statistics_momentum,
        lateral_momentum=config.lateral_momentum,
    ).to(device)
    optimizer = _optimizer(
        config.optimizer,
        model.encoder_parameters(),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.optimizer_momentum,
        beta1=config.optimizer_beta1,
        beta2=config.optimizer_beta2,
        epsilon=config.optimizer_epsilon,
    )
    summary = train_local_encoder(
        model=model,
        optimizer=optimizer,
        dataset=splits.train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        order_mode=config.order_mode,
        order_seed=seed,
        chunk_size=config.chunk_size,
        coefficients=LossCoefficients(
            similarity=config.similarity_coefficient,
            variance=config.variance_coefficient,
            covariance=config.covariance_coefficient,
        ),
        variance_target=config.variance_target,
        detach_previous=True,
        covariance_mode="residual_state",
        device=device,
        training_mode=config.training_mode,
        augmentation=config.augmentation,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        residual_lateral_steps=config.residual_lateral_steps,
        residual_lateral_step_size=config.residual_lateral_step_size,
        residual_lateral_rule=config.residual_lateral_rule,
        residual_lateral_include_diagonal=config.residual_lateral_include_diagonal,
        residual_lateral_moment_normalization=(
            config.residual_lateral_moment_normalization
        ),
        residual_lateral_coefficient=config.residual_lateral_coefficient,
        residual_lateral_signal_offset=config.residual_lateral_signal_offset,
        postsynaptic_state_mode=config.postsynaptic_state_mode,
        lateral_matrix_mode=config.lateral_matrix_mode,
        combined_lateral_state_weight=config.combined_lateral_state_weight,
        temporal_term_enabled=config.temporal_term_enabled,
    )
    return model, optimizer, summary


def _collect_states(
    model,
    optimizer,
    dataset,
    config,
    *,
    seed: int,
    observations: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    order, boundaries = encoder_order(
        dataset,
        order_mode=config.order_mode,
        seed=seed,
        chunk_size=config.chunk_size,
    )
    limit = min(int(observations), len(order))
    if limit <= 0:
        raise ValueError("state evidence requires at least one observation")
    for state in model.states:
        state.reset_sequence()
    rows = []
    source_indices = []
    coefficients = LossCoefficients(
        similarity=config.similarity_coefficient,
        variance=config.variance_coefficient,
        covariance=config.covariance_coefficient,
    )
    for stream_index in range(limit):
        if bool(boundaries[stream_index]):
            for state in model.states:
                state.reset_sequence()
        source_index = int(order[stream_index])
        collector = []
        local_train_step(
            model=model,
            optimizer=optimizer,
            x=dataset.features[source_index].unsqueeze(0).to(device),
            boundaries=boundaries[stream_index].reshape(1).to(device),
            coefficients=coefficients,
            variance_target=config.variance_target,
            detach_previous=True,
            covariance_mode="residual_state",
            residual_lateral_steps=config.residual_lateral_steps,
            residual_lateral_step_size=config.residual_lateral_step_size,
            residual_lateral_rule=config.residual_lateral_rule,
            residual_lateral_include_diagonal=config.residual_lateral_include_diagonal,
            residual_lateral_moment_normalization=(
                config.residual_lateral_moment_normalization
            ),
            residual_lateral_coefficient=config.residual_lateral_coefficient,
            residual_lateral_signal_offset=config.residual_lateral_signal_offset,
            postsynaptic_state_mode=config.postsynaptic_state_mode,
            lateral_matrix_mode=config.lateral_matrix_mode,
            combined_lateral_state_weight=config.combined_lateral_state_weight,
            temporal_term_enabled=config.temporal_term_enabled,
            neuron_state_collector=collector,
        )
        rows.append(collector[-1][1][0].numpy())
        source_indices.append(source_index)
    return np.stack(rows), np.asarray(source_indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()

    raw, config = _load_configuration(arguments.config)
    seed = int(raw["seed"])
    if raw["dataset"] == "mnist":
        splits = load_mnist_protocol(raw["data_root"], allow_download=False)
    elif raw["dataset"] == "capture24-cache":
        splits = _load_capture24_cache(Path(raw["data_root"]))
    else:
        raise ValueError(f"unsupported dataset: {raw['dataset']}")

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)
    model, optimizer, training = _train(splits, config, seed=seed, device=device)
    representations = extract_representations(
        model,
        splits.validation,
        batch_size=int(raw.get("representation_batch_size", 2048)),
        device=device,
        use_all_layers=True,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "encoder_config": asdict(config),
            "seed": seed,
        },
        output / "checkpoint.pt",
    )
    state_values, state_source_indices = _collect_states(
        model,
        optimizer,
        splits.validation,
        config,
        seed=seed,
        observations=int(raw.get("state_observations", 2048)),
        device=device,
    )

    neuron_seed = int(raw["neuron_selection_seed"])
    neuron_count = int(raw.get("neuron_count", 4))
    neuron_indices = np.sort(
        np.random.default_rng(neuron_seed).choice(
            state_values.shape[1], size=neuron_count, replace=False
        )
    )
    geometry_seed = int(raw["geometry_selection_seed"])
    geometry_count = min(
        int(raw.get("geometry_observations", 3000)), len(representations)
    )
    geometry_indices = np.sort(
        np.random.default_rng(geometry_seed).choice(
            len(representations), size=geometry_count, replace=False
        )
    )
    labels = splits.validation.labels.detach().cpu()
    np.savez_compressed(
        output / "visual-evidence.npz",
        geometry_representations=representations[geometry_indices].numpy(),
        geometry_labels=labels[geometry_indices].numpy(),
        geometry_source_indices=geometry_indices,
        state_values=state_values[:, neuron_indices],
        state_source_indices=state_source_indices,
        state_labels=labels[state_source_indices].numpy(),
        neuron_indices=neuron_indices,
    )
    summary = {
        "dataset": splits.metadata,
        "seed": seed,
        "encoder": asdict(config),
        "training": asdict(training),
        "neuron_selection_seed": neuron_seed,
        "neuron_indices": neuron_indices.tolist(),
        "state_observations": len(state_values),
        "geometry_selection_seed": geometry_seed,
        "geometry_observations": geometry_count,
        "representation_diagnostics": representation_diagnostics(
            representations, splits.validation.boundaries
        ),
        "class_structure_diagnostics": class_structure_diagnostics(
            representations,
            labels,
            num_classes=int(raw["num_classes"]),
        ),
        "repository": git_provenance(Path(__file__).resolve().parents[1]),
    }
    _write_json(output / "visual-evidence.json", summary)


if __name__ == "__main__":
    main()
