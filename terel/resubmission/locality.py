"""Run the predeclared bounded-history and memory execution audit."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .data import TemporalTensorDataset, load_mnist_protocol
from .evaluation import representation_diagnostics
from .experiments import EncoderExperimentConfig, set_reproducible_seed
from .model import LayerLocalEncoder
from .objective import LossCoefficients
from .provenance import canonical_sha256, git_provenance
from .training import train_local_encoder


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _optimizer_state_bytes(optimizer):
    return sum(
        _tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


@torch.no_grad()
def _represent(model, dataset, device):
    model.eval()
    outputs = []
    for start in range(0, len(dataset), 2048):
        outputs.append(model(dataset.features[start : start + 2048].to(device)).cpu())
    return torch.cat(outputs)


def _state_has_graph(model):
    return any(
        buffer.requires_grad or buffer.grad_fn is not None
        for state in model.states
        for buffer in state.buffers()
    )


def run_locality_audit(*, dataset, encoder, seed, device):
    variants = (
        ("detached-stream-b1", 1, True),
        ("detached-minibatch", encoder.batch_size, True),
        ("undetached-minibatch", encoder.batch_size, False),
    )
    results = {}
    for identifier, batch_size, detach_previous in variants:
        set_reproducible_seed(seed)
        model = LayerLocalEncoder(
            input_dim=dataset.features.shape[1],
            hidden_dims=encoder.hidden_dims,
            activation=encoder.activation,
            statistics_momentum=encoder.statistics_momentum,
            lateral_momentum=encoder.lateral_momentum,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.encoder_parameters(),
            lr=encoder.learning_rate,
            weight_decay=encoder.weight_decay,
        )
        state_before = sum(state.dynamic_state_numel() for state in model.states)
        training = train_local_encoder(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            epochs=encoder.epochs,
            batch_size=batch_size,
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
            covariance_mode="proxy",
            device=device,
        )
        state_after = sum(state.dynamic_state_numel() for state in model.states)
        representations = _represent(model, dataset, device)
        parameter_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
        results[identifier] = {
            "batch_size": int(batch_size),
            "temporal_reference_detached": bool(detach_previous),
            "training": asdict(training),
            "examples_per_second": float(training.examples / training.seconds),
            "dynamic_state_numel_before": int(state_before),
            "dynamic_state_numel_after": int(state_after),
            "dynamic_state_bytes": int(sum(_tensor_bytes(buffer) for state in model.states for buffer in state.buffers())),
            "parameter_bytes": int(parameter_bytes),
            "optimizer_state_bytes": int(_optimizer_state_bytes(optimizer)),
            "state_retains_autograd_graph": bool(_state_has_graph(model)),
            "representation_diagnostics": representation_diagnostics(
                representations, dataset.boundaries
            ),
        }
    return {
        "schema_version": 1,
        "seed": int(seed),
        "examples": int(len(dataset)),
        "encoder": asdict(encoder),
        "variants": results,
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260803)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    arguments = parser.parse_args(argv)

    manifest = json.loads(Path(arguments.manifest).read_text())
    configuration = manifest["configuration"]
    if manifest.get("configuration_sha256") != canonical_sha256(configuration):
        raise ValueError("manifest configuration checksum is invalid")
    provenance = git_provenance(arguments.repository)
    if provenance["tracked_dirty"] or provenance["code_commit"] != manifest.get("code_commit"):
        raise ValueError("locality audit requires the clean manifest code commit")
    mnist = configuration["datasets"]["mnist"]
    run = next(item for item in mnist["runs"] if item["id"] == "terel-local")
    encoder_values = dict(run["encoder"])
    encoder_values["hidden_dims"] = tuple(encoder_values["hidden_dims"])
    encoder = replace(
        EncoderExperimentConfig(**encoder_values),
        epochs=int(arguments.epochs),
    )
    train = load_mnist_protocol(mnist["data_root"], allow_download=False).train
    count = min(int(arguments.examples), len(train))
    if count <= 1:
        raise ValueError("locality audit requires at least two examples")
    dataset = TemporalTensorDataset(
        features=train.features[:count],
        labels=train.labels[:count],
        boundaries=train.boundaries[:count].clone(),
    )
    dataset.boundaries[0] = True
    audit = run_locality_audit(
        dataset=dataset,
        encoder=encoder,
        seed=arguments.seed,
        device=torch.device(arguments.device),
    )
    audit.update(
        {
            "manifest_configuration_sha256": manifest["configuration_sha256"],
            "manifest_code_commit": manifest["code_commit"],
            "manifest_path": str(Path(arguments.manifest).resolve()),
        }
    )
    _write_json(arguments.output, audit)


if __name__ == "__main__":
    main()
