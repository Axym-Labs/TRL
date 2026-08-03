from dataclasses import asdict, dataclass
import random
import time

import numpy as np
import torch

from .baselines import (
    BatchLinearSFA,
    IncrementalLinearSFA,
    SupervisedMLP,
    train_local_supervised_contrastive,
    train_supervised_mlp,
)
from .data import DatasetSplits, class_chunk_order
from .evaluation import (
    class_structure_diagnostics,
    classification_metrics,
    extract_representations,
    fit_linear_probe,
    representation_diagnostics,
)
from .model import LayerLocalEncoder
from .objective import LossCoefficients
from .provenance import TestGateError, assert_test_gate
from .training import train_local_encoder


@dataclass(frozen=True)
class EncoderExperimentConfig:
    method: str
    hidden_dims: tuple[int, ...]
    activation: str = "relu"
    epochs: int = 10
    batch_size: int = 256
    order_mode: str = "chronological"
    chunk_size: int = 16
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    similarity_coefficient: float = 1.0
    variance_coefficient: float = 2.5
    covariance_coefficient: float = 1.0
    variance_target: float = 1.0
    statistics_momentum: float = 0.9
    lateral_momentum: float = 0.99
    contrastive_temperature: float = 0.2
    sfa_components: int | None = None
    incsfa_whitening_dim: int | None = None
    incsfa_output_dim: int | None = None
    incsfa_learning_rate: float = 0.001


@dataclass(frozen=True)
class ProbeExperimentConfig:
    epochs: int = 30
    batch_size: int = 1024
    optimizer: str = "adamw"
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TestGateContext:
    manifest: dict
    protocol_path: str
    validation_ledger_path: str
    repository: str
    explicit_allow_test: bool


def set_reproducible_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer(name, parameters, *, learning_rate, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _parameter_bytes(model):
    return sum(_tensor_bytes(parameter) for parameter in model.parameters())


def _buffer_bytes(model):
    return sum(_tensor_bytes(buffer) for buffer in model.buffers())


def _optimizer_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += _tensor_bytes(value)
    return total


@torch.no_grad()
def _supervised_representations(model, dataset, *, batch_size, device):
    model.eval()
    batches = []
    for start in range(0, len(dataset), batch_size):
        batches.append(
            model.representations(dataset.features[start : start + batch_size].to(device))[-1]
            .detach()
            .cpu()
        )
    return torch.cat(batches)


def _sfa_order(dataset, config: EncoderExperimentConfig, seed: int):
    if config.order_mode == "chronological":
        return np.arange(len(dataset)), dataset.boundaries.detach().cpu().numpy()
    if config.order_mode == "class_chunks":
        return class_chunk_order(
            dataset.labels.detach().cpu().numpy(),
            chunk_size=config.chunk_size,
            seed=seed,
        )
    if config.order_mode == "shuffled":
        order = np.random.default_rng(seed).permutation(len(dataset))
        boundaries = np.zeros(len(dataset), dtype=bool)
        boundaries[0] = True
        return order, boundaries
    raise ValueError(f"Unknown SFA order mode: {config.order_mode}")


def run_representation_experiment(
    *,
    splits: DatasetSplits,
    dataset_name: str,
    num_classes: int,
    seed: int,
    encoder: EncoderExperimentConfig,
    probe: ProbeExperimentConfig,
    evaluation_split: str,
    device: torch.device,
    test_gate: TestGateContext | None = None,
):
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    if evaluation_split == "test":
        if test_gate is None:
            raise TestGateError("held-out evaluation requires a test gate context")
        assert_test_gate(
            test_gate.manifest,
            protocol_path=test_gate.protocol_path,
            validation_ledger_path=test_gate.validation_ledger_path,
            repository=test_gate.repository,
            explicit_allow_test=test_gate.explicit_allow_test,
        )

    set_reproducible_seed(seed)
    train_dataset = splits.train
    evaluation_dataset = splits.validation if evaluation_split == "validation" else splits.test
    input_dim = train_dataset.features.shape[1]
    method = encoder.method
    encoder_training = None
    probe_training = None
    optimizer = None

    if method in {"random", "terel_local", "terel_batch", "terel_direct", "terel_shift", "local_supcon"}:
        model = LayerLocalEncoder(
            input_dim=input_dim,
            hidden_dims=encoder.hidden_dims,
            activation=encoder.activation,
            statistics_momentum=encoder.statistics_momentum,
            lateral_momentum=encoder.lateral_momentum,
        ).to(device)
        if method != "random":
            optimizer = _optimizer(
                encoder.optimizer,
                model.encoder_parameters(),
                learning_rate=encoder.learning_rate,
                weight_decay=encoder.weight_decay,
            )
            if method == "local_supcon":
                encoder_training = train_local_supervised_contrastive(
                    model=model,
                    optimizer=optimizer,
                    dataset=train_dataset,
                    epochs=encoder.epochs,
                    batch_size=encoder.batch_size,
                    seed=seed,
                    chunk_size=encoder.chunk_size,
                    temperature=encoder.contrastive_temperature,
                    device=device,
                )
            else:
                detach_previous = method != "terel_batch"
                covariance_mode = {
                    "terel_local": "proxy",
                    "terel_batch": "proxy",
                    "terel_direct": "direct",
                    "terel_shift": "shifted_proxy",
                }[method]
                encoder_training = train_local_encoder(
                    model=model,
                    optimizer=optimizer,
                    dataset=train_dataset,
                    epochs=encoder.epochs,
                    batch_size=encoder.batch_size,
                    order_mode=encoder.order_mode,
                    order_seed=seed,
                    chunk_size=encoder.chunk_size,
                    coefficients=LossCoefficients(
                        similarity=encoder.similarity_coefficient,
                        variance=encoder.variance_coefficient,
                        covariance=encoder.covariance_coefficient,
                    ),
                    variance_target=encoder.variance_target,
                    detach_previous=detach_previous,
                    covariance_mode=covariance_mode,
                    device=device,
                )
        train_representations = extract_representations(
            model, train_dataset, batch_size=probe.batch_size, device=device
        )
        evaluation_representations = extract_representations(
            model, evaluation_dataset, batch_size=probe.batch_size, device=device
        )
        linear_probe, probe_training = fit_linear_probe(
            train_representations,
            train_dataset.labels,
            num_classes=num_classes,
            seed=seed + 10_000,
            epochs=probe.epochs,
            batch_size=probe.batch_size,
            optimizer_name=probe.optimizer,
            learning_rate=probe.learning_rate,
            weight_decay=probe.weight_decay,
            device=device,
        )
        with torch.no_grad():
            logits = linear_probe(evaluation_representations.to(device)).cpu()
        dynamic_state_bytes = _buffer_bytes(model) if method.startswith("terel_") else 0

    elif method == "bp":
        model = SupervisedMLP(
            input_dim=input_dim,
            hidden_dims=encoder.hidden_dims,
            output_dim=num_classes,
            activation=encoder.activation,
        ).to(device)
        optimizer = _optimizer(
            encoder.optimizer,
            model.parameters(),
            learning_rate=encoder.learning_rate,
            weight_decay=encoder.weight_decay,
        )
        encoder_training = train_supervised_mlp(
            model=model,
            optimizer=optimizer,
            dataset=train_dataset,
            epochs=encoder.epochs,
            batch_size=encoder.batch_size,
            seed=seed,
            device=device,
        )
        train_representations = _supervised_representations(
            model, train_dataset, batch_size=probe.batch_size, device=device
        )
        evaluation_representations = _supervised_representations(
            model, evaluation_dataset, batch_size=probe.batch_size, device=device
        )
        with torch.no_grad():
            logits = model(evaluation_dataset.features.to(device)).cpu()
        dynamic_state_bytes = 0

    elif method == "sfa":
        order, boundaries = _sfa_order(train_dataset, encoder, seed)
        components = encoder.sfa_components or min(encoder.hidden_dims[-1], input_dim)
        model = BatchLinearSFA(n_components=components)
        ordered = train_dataset.features[torch.as_tensor(order)].detach().cpu().numpy()
        model.fit(ordered, boundaries=boundaries)
        train_representations = torch.from_numpy(
            model.transform(train_dataset.features.detach().cpu().numpy())
        ).to(torch.float32)
        evaluation_representations = torch.from_numpy(
            model.transform(evaluation_dataset.features.detach().cpu().numpy())
        ).to(torch.float32)
        linear_probe, probe_training = fit_linear_probe(
            train_representations,
            train_dataset.labels,
            num_classes=num_classes,
            seed=seed + 10_000,
            epochs=probe.epochs,
            batch_size=probe.batch_size,
            optimizer_name=probe.optimizer,
            learning_rate=probe.learning_rate,
            weight_decay=probe.weight_decay,
            device=device,
        )
        with torch.no_grad():
            logits = linear_probe(evaluation_representations.to(device)).cpu()
        optimizer = None
        dynamic_state_bytes = 0
    elif method == "incsfa":
        order, boundaries = _sfa_order(train_dataset, encoder, seed)
        ordered = train_dataset.features[torch.as_tensor(order)].detach().cpu().numpy()
        whitening_dim = encoder.incsfa_whitening_dim or input_dim
        output_dim = encoder.incsfa_output_dim or min(encoder.hidden_dims[-1], whitening_dim)
        model = IncrementalLinearSFA(
            input_dim=input_dim,
            whitening_dim=whitening_dim,
            output_dim=output_dim,
            learning_rate=encoder.incsfa_learning_rate,
            seed=seed,
        )
        start_time = time.perf_counter()
        model.fit(ordered, boundaries=boundaries, epochs=encoder.epochs)
        encoder_training = {
            "epochs": int(encoder.epochs),
            "steps": int(encoder.epochs * len(train_dataset)),
            "examples": int(encoder.epochs * len(train_dataset)),
            "valid_temporal_pairs": int(model.derivative_pair_count_),
            "seconds": time.perf_counter() - start_time,
        }
        train_representations = torch.from_numpy(
            model.transform(train_dataset.features.detach().cpu().numpy())
        ).to(torch.float32)
        evaluation_representations = torch.from_numpy(
            model.transform(evaluation_dataset.features.detach().cpu().numpy())
        ).to(torch.float32)
        linear_probe, probe_training = fit_linear_probe(
            train_representations,
            train_dataset.labels,
            num_classes=num_classes,
            seed=seed + 10_000,
            epochs=probe.epochs,
            batch_size=probe.batch_size,
            optimizer_name=probe.optimizer,
            learning_rate=probe.learning_rate,
            weight_decay=probe.weight_decay,
            device=device,
        )
        with torch.no_grad():
            logits = linear_probe(evaluation_representations.to(device)).cpu()
        optimizer = None
        dynamic_state_bytes = model.dynamic_state_numel() * 8
    else:
        raise ValueError(f"Unknown experiment method: {method}")

    if isinstance(model, torch.nn.Module):
        parameter_bytes = _parameter_bytes(model)
    elif isinstance(model, IncrementalLinearSFA):
        parameter_bytes = int(model.components_.nbytes)
    else:
        parameter_bytes = int(model.components_.nbytes + model.mean_.nbytes)
    resource_accounting = {
        "parameter_bytes": int(parameter_bytes),
        "dynamic_state_bytes": int(dynamic_state_bytes),
        "optimizer_state_bytes": int(_optimizer_bytes(optimizer)) if optimizer is not None else 0,
        "encoder_batch_size": int(encoder.batch_size),
        "probe_batch_size": int(probe.batch_size),
    }
    return {
        "dataset": dataset_name,
        "dataset_metadata": splits.metadata,
        "method": method,
        "seed": int(seed),
        "evaluation_split": evaluation_split,
        "encoder_config": asdict(encoder),
        "probe_config": asdict(probe),
        "encoder_training": (
            asdict(encoder_training)
            if encoder_training is not None and not isinstance(encoder_training, dict)
            else encoder_training
        ),
        "probe_training": asdict(probe_training) if probe_training is not None else None,
        "metrics": classification_metrics(logits, evaluation_dataset.labels, num_classes=num_classes),
        "representation_diagnostics": representation_diagnostics(
            evaluation_representations, evaluation_dataset.boundaries
        ),
        "class_structure_diagnostics": class_structure_diagnostics(
            evaluation_representations,
            evaluation_dataset.labels,
            num_classes=num_classes,
        ),
        "resource_accounting": resource_accounting,
    }
