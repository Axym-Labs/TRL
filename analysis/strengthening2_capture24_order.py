"""Run one frozen CAPTURE-24 order condition on validation participants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from strengthening2_visual_evidence import _load_capture24_cache

from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    run_representation_experiment,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()

    raw = yaml.safe_load(arguments.config.read_text())
    if raw["evaluation_split"] != "validation":
        raise ValueError("the order audit must remain validation-only")
    if arguments.condition not in raw["conditions"]:
        raise ValueError("condition is not part of the frozen order audit")
    if arguments.seed not in raw["seeds"]:
        raise ValueError("seed is not part of the frozen order audit")

    splits = _load_capture24_cache(Path(raw["data_root"]))
    if arguments.condition == "random":
        encoder = EncoderExperimentConfig(
            method="random",
            hidden_dims=tuple(raw["encoder"]["hidden_dims"]),
            activation="relu",
            normalization="none",
        )
    else:
        values = dict(raw["encoder"])
        values["hidden_dims"] = tuple(values["hidden_dims"])
        values["order_mode"] = (
            "chronological"
            if arguments.condition == "chronological"
            else "within_stream_shuffled"
        )
        encoder = EncoderExperimentConfig(**values)

    result = run_representation_experiment(
        splits=splits,
        dataset_name="CAPTURE-24 validation",
        num_classes=int(raw["num_classes"]),
        seed=arguments.seed,
        encoder=encoder,
        probe=ProbeExperimentConfig(**raw["probe"]),
        evaluation_split="validation",
        device=torch.device(arguments.device),
    )
    result["condition"] = arguments.condition
    result["configuration"] = str(arguments.config.resolve())
    _write_json(arguments.output, result)


if __name__ == "__main__":
    main()
