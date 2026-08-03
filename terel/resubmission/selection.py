import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_mnist_protocol, load_pamap2_protocol
from .experiments import EncoderExperimentConfig, ProbeExperimentConfig, run_representation_experiment
from .provenance import git_provenance, sha256_file


def load_selection_plan(path):
    plan = yaml.safe_load(Path(path).read_text())
    if not isinstance(plan, dict) or not isinstance(plan.get("datasets"), dict):
        raise ValueError("selection plan must contain a datasets mapping")
    for dataset_name, dataset in plan["datasets"].items():
        configurations = dataset.get("configurations", [])
        if len(configurations) > 12:
            raise ValueError(f"{dataset_name} exceeds the maximum of 12 validation configurations")
        identifiers = [configuration.get("id") for configuration in configurations]
        if None in identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{dataset_name} configuration ids must be present and unique")
    return plan


def validate_plan_protocol(plan, protocol_path):
    expected = plan.get("protocol_sha256")
    actual = sha256_file(protocol_path)
    if expected != actual:
        raise ValueError(f"selection plan protocol hash {expected} does not match {actual}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_dataset(name, config):
    if name == "mnist":
        return load_mnist_protocol(config["data_root"], allow_download=False)
    if name == "pamap2":
        return load_pamap2_protocol(config["data_root"], stride=10, allow_download=False)
    raise ValueError(f"unknown selection dataset: {name}")


def _metric(result, path):
    current = result
    for component in path.split("."):
        current = current[component]
    return float(current)


def evaluate_fidelity(seed_results, *, seeds, thresholds):
    if not thresholds:
        raise ValueError("selection requires non-collapse fidelity thresholds")
    by_seed = {}
    valid = True
    for seed, result in zip(seeds, seed_results, strict=True):
        diagnostics = result["representation_diagnostics"]
        values = {name: float(diagnostics[name]) for name in thresholds}
        values["passes"] = all(values[name] >= float(limit) for name, limit in thresholds.items())
        valid &= values["passes"]
        by_seed[str(seed)] = values
    return bool(valid), by_seed


def run_selection_plan(plan, *, output_directory, device, repository, protocol_path):
    validate_plan_protocol(plan, protocol_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        **git_provenance(repository),
        "protocol_sha256": sha256_file(protocol_path),
        "plan": plan,
    }
    _write_json(output_directory / "selection-provenance.json", provenance)

    seeds = tuple(int(seed) for seed in plan["seeds"])
    probe = ProbeExperimentConfig(**plan["probe"])
    ledger = {"selection_complete": False, "datasets": {}, "provenance": provenance}
    for dataset_name, dataset_config in plan["datasets"].items():
        splits = _load_dataset(dataset_name, dataset_config)
        metric_path = dataset_config["primary_metric"]
        fidelity_thresholds = {
            name: float(value)
            for name, value in dataset_config["fidelity_thresholds"].items()
        }
        dataset_records = []
        ledger["datasets"][dataset_name] = {
            "primary_metric": metric_path,
            "fidelity_thresholds": fidelity_thresholds,
            "records": dataset_records,
            "selected_configuration_id": None,
        }
        for raw_configuration in dataset_config["configurations"]:
            configuration = dict(dataset_config["encoder_base"])
            override = dict(raw_configuration)
            configuration_id = override.pop("id")
            configuration.update(override)
            configuration["hidden_dims"] = tuple(configuration["hidden_dims"])
            encoder = EncoderExperimentConfig(**configuration)
            seed_results = []
            config_directory = output_directory / dataset_name / configuration_id
            config_directory.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                output_path = config_directory / f"seed-{seed}.json"
                if output_path.exists():
                    result = json.loads(output_path.read_text())
                else:
                    result = run_representation_experiment(
                        splits=splits,
                        dataset_name=dataset_name,
                        num_classes=int(dataset_config["num_classes"]),
                        seed=seed,
                        encoder=encoder,
                        probe=probe,
                        evaluation_split="validation",
                        device=device,
                    )
                    result["configuration_id"] = configuration_id
                    _write_json(output_path, result)
                seed_results.append(result)
            values = [_metric(result, metric_path) for result in seed_results]
            variances = [
                float(result["representation_diagnostics"]["median_feature_variance"])
                for result in seed_results
            ]
            fidelity_valid, fidelity_by_seed = evaluate_fidelity(
                seed_results,
                seeds=seeds,
                thresholds=fidelity_thresholds,
            )
            dataset_records.append(
                {
                    "configuration_id": configuration_id,
                    "configuration": configuration,
                    "seed_values": dict(zip((str(seed) for seed in seeds), values, strict=True)),
                    "mean_validation_metric": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "median_feature_variance_by_seed": dict(
                        zip((str(seed) for seed in seeds), variances, strict=True)
                    ),
                    "fidelity_by_seed": fidelity_by_seed,
                    "fidelity_valid": fidelity_valid,
                }
            )
            _write_json(output_directory / "validation-ledger.partial.json", ledger)
        valid = [record for record in dataset_records if record["fidelity_valid"]]
        ranked = sorted(valid, key=lambda record: record["mean_validation_metric"], reverse=True)
        ledger["datasets"][dataset_name]["selected_configuration_id"] = (
            ranked[0]["configuration_id"] if ranked else None
        )
        _write_json(output_directory / "validation-ledger.partial.json", ledger)
    ledger["selection_complete"] = all(
        value["selected_configuration_id"] is not None for value in ledger["datasets"].values()
    )
    _write_json(output_directory / "validation-ledger.json", ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run bounded TeReL validation selection")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    arguments = parser.parse_args(argv)
    run_selection_plan(
        load_selection_plan(arguments.plan),
        output_directory=arguments.output,
        device=torch.device(arguments.device),
        repository=arguments.repository,
        protocol_path=arguments.protocol,
    )


if __name__ == "__main__":
    main()
