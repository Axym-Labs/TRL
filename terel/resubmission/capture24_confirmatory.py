"""Freeze, prepare, and execute the CAPTURE-24 natural-order confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import DatasetSplits, TemporalTensorDataset, load_capture24_protocol
from .experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    TestGateContext,
    run_representation_experiment,
)
from .provenance import assert_test_gate, build_run_manifest


def _write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_protocol(path) -> dict:
    protocol = yaml.safe_load(Path(path).read_text())
    if protocol.get("evaluation_split") != "test":
        raise ValueError("CAPTURE-24 confirmation must target the test split")
    seeds = [int(seed) for seed in protocol.get("confirmatory_seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("CAPTURE-24 confirmation requires five distinct seeds")
    return protocol


def freeze_manifest(
    *, protocol_path, validation_ledger_path, repository, output_path
) -> dict:
    protocol = _load_protocol(protocol_path)
    ledger = json.loads(Path(validation_ledger_path).read_text())
    if not ledger.get("selection_complete"):
        raise ValueError("CAPTURE-24 validation selection is incomplete")
    if ledger.get("heldout_subjects_accessed"):
        raise ValueError("CAPTURE-24 held-out participants were accessed during selection")
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        configuration=protocol,
    )
    _write_json(output_path, manifest)
    return manifest


def _gate(
    *,
    manifest_path,
    protocol_path,
    validation_ledger_path,
    repository,
    allow_test,
) -> tuple[dict, dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    protocol = _load_protocol(protocol_path)
    assert_test_gate(
        manifest,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        explicit_allow_test=allow_test,
    )
    if manifest.get("configuration") != protocol:
        raise ValueError("manifest configuration differs from the frozen protocol")
    return manifest, protocol


def _save_cache(path, splits: DatasetSplits) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_features=splits.train.features.numpy(),
        train_labels=splits.train.labels.numpy(),
        train_boundaries=splits.train.boundaries.numpy(),
        validation_features=splits.validation.features.numpy(),
        validation_labels=splits.validation.labels.numpy(),
        validation_boundaries=splits.validation.boundaries.numpy(),
        test_features=splits.test.features.numpy(),
        test_labels=splits.test.labels.numpy(),
        test_boundaries=splits.test.boundaries.numpy(),
        metadata_json=json.dumps(splits.metadata, sort_keys=True),
    )


def _load_cache(path) -> DatasetSplits:
    values = np.load(path)

    def dataset(prefix):
        return TemporalTensorDataset(
            features=torch.from_numpy(values[f"{prefix}_features"]).to(torch.float32),
            labels=torch.from_numpy(values[f"{prefix}_labels"]).to(torch.long),
            boundaries=torch.from_numpy(values[f"{prefix}_boundaries"]).to(torch.bool),
        )

    metadata = json.loads(str(values["metadata_json"]))
    if not metadata.get("heldout_subjects_accessed"):
        raise ValueError("CAPTURE-24 confirmation cache does not contain held-out data")
    return DatasetSplits(
        train=dataset("train"),
        validation=dataset("validation"),
        test=dataset("test"),
        metadata=metadata,
    )


def prepare_cache(
    *,
    manifest_path,
    protocol_path,
    validation_ledger_path,
    repository,
    data_root,
    output_path,
    allow_test,
) -> None:
    _, protocol = _gate(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        allow_test=allow_test,
    )
    dataset = protocol["dataset"]
    splits = load_capture24_protocol(
        data_root,
        access_heldout=True,
        train_subjects=dataset["train_subjects"],
        validation_subjects=dataset["validation_subjects"],
        heldout_subjects=dataset["heldout_subjects"],
        window_seconds=int(dataset["window_seconds"]),
    )
    _save_cache(output_path, splits)


def _encoder(protocol: dict, condition: str) -> EncoderExperimentConfig:
    if condition == "random":
        return EncoderExperimentConfig(
            method="random",
            hidden_dims=tuple(protocol["encoder"]["hidden_dims"]),
            order_mode="chronological",
        )
    values = dict(protocol["encoder"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    values["order_mode"] = (
        "chronological"
        if condition == "chronological"
        else "within_stream_shuffled"
    )
    return EncoderExperimentConfig(**values)


def run_condition(
    *,
    manifest_path,
    protocol_path,
    validation_ledger_path,
    repository,
    cache_path,
    condition,
    seed,
    output_path,
    device,
    allow_test,
) -> dict:
    manifest, protocol = _gate(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        allow_test=allow_test,
    )
    if condition not in protocol["conditions"]:
        raise ValueError(f"condition '{condition}' is not frozen")
    if int(seed) not in protocol["confirmatory_seeds"]:
        raise ValueError(f"seed {seed} is not frozen")
    splits = _load_cache(cache_path)
    probe_values = dict(protocol["probe"])
    result = run_representation_experiment(
        splits=splits,
        dataset_name="CAPTURE-24 confirmatory",
        num_classes=int(protocol["num_classes"]),
        seed=int(seed),
        encoder=_encoder(protocol, condition),
        probe=ProbeExperimentConfig(**probe_values),
        evaluation_split="test",
        device=torch.device(device),
        test_gate=TestGateContext(
            manifest=manifest,
            protocol_path=str(protocol_path),
            validation_ledger_path=str(validation_ledger_path),
            repository=str(repository),
            explicit_allow_test=bool(allow_test),
        ),
    )
    result["condition"] = condition
    _write_json(output_path, result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--validation-ledger", required=True)
    freeze.add_argument("--repository", required=True)
    freeze.add_argument("--output", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--protocol", required=True)
    prepare.add_argument("--validation-ledger", required=True)
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--allow-test", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--validation-ledger", required=True)
    run.add_argument("--repository", required=True)
    run.add_argument("--cache", required=True)
    run.add_argument("--condition", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run.add_argument("--allow-test", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.command == "freeze":
        freeze_manifest(
            protocol_path=arguments.protocol,
            validation_ledger_path=arguments.validation_ledger,
            repository=arguments.repository,
            output_path=arguments.output,
        )
    elif arguments.command == "prepare":
        prepare_cache(
            manifest_path=arguments.manifest,
            protocol_path=arguments.protocol,
            validation_ledger_path=arguments.validation_ledger,
            repository=arguments.repository,
            data_root=arguments.data_root,
            output_path=arguments.output,
            allow_test=arguments.allow_test,
        )
    else:
        run_condition(
            manifest_path=arguments.manifest,
            protocol_path=arguments.protocol,
            validation_ledger_path=arguments.validation_ledger,
            repository=arguments.repository,
            cache_path=arguments.cache,
            condition=arguments.condition,
            seed=arguments.seed,
            output_path=arguments.output,
            device=arguments.device,
            allow_test=arguments.allow_test,
        )


if __name__ == "__main__":
    main()
