"""Bounded validation-only runner for corrected TeReL performance recovery."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .data import load_mnist_protocol
from .experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    run_representation_experiment,
)


def load_recovery_plan(path):
    plan = yaml.safe_load(Path(path).read_text())
    if not isinstance(plan, dict) or not isinstance(plan.get("candidates"), list):
        raise ValueError("recovery plan must contain a candidates list")
    identifiers = [candidate.get("id") for candidate in plan["candidates"]]
    if None in identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("recovery candidate ids must be present and unique")
    if len(identifiers) > int(plan.get("maximum_candidates", 24)):
        raise ValueError("recovery plan exceeds its maximum candidate budget")
    return plan


def resolve_candidate(plan, identifier):
    try:
        candidate = next(item for item in plan["candidates"] if item["id"] == identifier)
    except StopIteration as error:
        raise KeyError(f"unknown recovery candidate: {identifier}") from error
    encoder = dict(plan["encoder_base"])
    encoder.update(candidate.get("encoder", {}))
    encoder["hidden_dims"] = tuple(encoder["hidden_dims"])
    probe = dict(plan["probe_base"])
    probe.update(candidate.get("probe", {}))
    return candidate, EncoderExperimentConfig(**encoder), ProbeExperimentConfig(**probe)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_candidate(plan, *, identifier, seed, device, output_root):
    candidate, encoder, probe = resolve_candidate(plan, identifier)
    splits = load_mnist_protocol(plan["data_root"], allow_download=False)
    result = run_representation_experiment(
        splits=splits,
        dataset_name="mnist",
        num_classes=10,
        seed=int(seed),
        encoder=encoder,
        probe=probe,
        evaluation_split="validation",
        device=device,
    )
    result["candidate_id"] = identifier
    result["candidate_description"] = candidate.get("description", "")
    output_path = Path(output_root) / identifier / f"seed-{seed}.json"
    _write_json(output_path, result)
    summary = {
        "candidate": identifier,
        "seed": int(seed),
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "effective_rank": result["representation_diagnostics"]["effective_rank"],
        "median_feature_variance": result["representation_diagnostics"][
            "median_feature_variance"
        ],
        "nearest_centroid_accuracy": result["class_structure_diagnostics"][
            "nearest_centroid_accuracy"
        ],
        "encoder_seconds": (
            result["encoder_training"]["seconds"]
            if result["encoder_training"] is not None
            else 0.0
        ),
        "output": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--output", default="artifacts/recovery-v2")
    arguments = parser.parse_args(argv)
    run_candidate(
        load_recovery_plan(arguments.config),
        identifier=arguments.candidate,
        seed=arguments.seed,
        device=torch.device(arguments.device),
        output_root=arguments.output,
    )


if __name__ == "__main__":
    main()
