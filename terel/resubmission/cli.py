import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import yaml

from .data import load_mnist_protocol, load_pamap2_protocol
from .experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    TestGateContext,
    run_representation_experiment,
)


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    data_root: str
    num_classes: int
    evaluation_split: str
    seeds: tuple[int, ...]
    encoder: EncoderExperimentConfig
    probe: ProbeExperimentConfig

    def as_dictionary(self):
        return asdict(self)


def load_experiment_spec(path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a YAML mapping")
    encoder_raw = dict(raw.pop("encoder"))
    encoder_raw["hidden_dims"] = tuple(encoder_raw["hidden_dims"])
    probe_raw = dict(raw.pop("probe"))
    raw["seeds"] = tuple(int(seed) for seed in raw["seeds"])
    return ExperimentSpec(
        encoder=EncoderExperimentConfig(**encoder_raw),
        probe=ProbeExperimentConfig(**probe_raw),
        **raw,
    )


def _load_data(spec: ExperimentSpec):
    if spec.dataset == "mnist":
        return load_mnist_protocol(spec.data_root, allow_download=True)
    if spec.dataset == "pamap2":
        return load_pamap2_protocol(spec.data_root, stride=10, allow_download=True)
    raise ValueError(f"Unknown dataset: {spec.dataset}")


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_spec(spec: ExperimentSpec, *, output_directory, device, test_gate=None):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "experiment-spec.json", spec.as_dictionary())
    splits = _load_data(spec)
    results = []
    for seed in spec.seeds:
        result = run_representation_experiment(
            splits=splits,
            dataset_name=spec.dataset,
            num_classes=spec.num_classes,
            seed=seed,
            encoder=spec.encoder,
            probe=spec.probe,
            evaluation_split=spec.evaluation_split,
            device=device,
            test_gate=test_gate,
        )
        _write_json(output_directory / f"seed-{seed}.json", result)
        results.append(result)
    _write_json(output_directory / "results.json", results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run guarded TeReL resubmission experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--manifest")
    parser.add_argument("--protocol")
    parser.add_argument("--validation-ledger")
    parser.add_argument("--repository")
    parser.add_argument("--allow-test", action="store_true")
    arguments = parser.parse_args(argv)

    spec = load_experiment_spec(arguments.config)
    device_name = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    if device_name == "auto":
        device_name = "cpu"
    gate = None
    if spec.evaluation_split == "test":
        required = (
            arguments.manifest,
            arguments.protocol,
            arguments.validation_ledger,
            arguments.repository,
        )
        if not all(required):
            parser.error("test runs require --manifest, --protocol, --validation-ledger, and --repository")
        gate = TestGateContext(
            manifest=json.loads(Path(arguments.manifest).read_text()),
            protocol_path=arguments.protocol,
            validation_ledger_path=arguments.validation_ledger,
            repository=arguments.repository,
            explicit_allow_test=arguments.allow_test,
        )
    run_spec(spec, output_directory=arguments.output, device=torch.device(device_name), test_gate=gate)


if __name__ == "__main__":
    main()
