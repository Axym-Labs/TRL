"""Freeze and execute the selected residual-state MNIST evaluation."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .confirmatory import run_confirmatory_manifest
from .provenance import build_run_manifest


CONFIRMATORY_SEEDS = (42, 43, 44, 45, 46)


def _write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_configuration(
    selection_plan: dict,
    validation_ledger: dict,
    *,
    confirmatory_seeds=CONFIRMATORY_SEEDS,
) -> dict:
    """Resolve the selected residual-state method into one immutable run matrix."""
    if not validation_ledger.get("selection_complete"):
        raise ValueError("validation selection is incomplete")
    seeds = [int(seed) for seed in confirmatory_seeds]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("confirmatory execution requires five distinct seeds")
    selected_id = validation_ledger.get("selected_configuration_id")
    selected = next(
        (
            candidate
            for candidate in selection_plan.get("candidates", [])
            if candidate.get("id") == selected_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected configuration is absent from the frozen plan")
    encoder = dict(selection_plan["encoder_base"])
    encoder.update(selected.get("encoder", {}))
    if encoder.get("method") != "terel_residual":
        raise ValueError("selected configuration is not samplewise TeReL")

    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "seeds": seeds,
        "selection_seeds": [int(seed) for seed in selection_plan["seeds"]],
        "probe": dict(selection_plan["probe_base"]),
        "selection_policy": (
            "One residual-state method was frozen from train-derived validation; "
            "no held-out alternatives are permitted."
        ),
        "datasets": {
            "mnist": {
                "data_root": selection_plan["data_root"],
                "num_classes": 10,
                "selected_configuration_id": selected_id,
                "runs": [{"id": selected_id, "encoder": encoder}],
            }
        },
    }


def freeze_manifest(
    *,
    selection_plan_path,
    validation_ledger_path,
    protocol_path,
    repository,
    output_path,
) -> dict:
    selection_plan = yaml.safe_load(Path(selection_plan_path).read_text())
    validation_ledger = json.loads(Path(validation_ledger_path).read_text())
    configuration = resolve_configuration(selection_plan, validation_ledger)
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        configuration=configuration,
    )
    _write_json(output_path, manifest)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--selection-plan", required=True)
    freeze.add_argument("--validation-ledger", required=True)
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--repository", required=True)
    freeze.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--validation-ledger", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--repository", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run.add_argument("--allow-test", action="store_true")
    run.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)

    if arguments.command == "freeze":
        freeze_manifest(
            selection_plan_path=arguments.selection_plan,
            validation_ledger_path=arguments.validation_ledger,
            protocol_path=arguments.protocol,
            repository=arguments.repository,
            output_path=arguments.output,
        )
        return

    manifest = json.loads(Path(arguments.manifest).read_text())
    run_confirmatory_manifest(
        manifest,
        manifest_path=arguments.manifest,
        output_directory=arguments.output,
        protocol_path=arguments.protocol,
        validation_ledger_path=arguments.validation_ledger,
        repository=arguments.repository,
        explicit_allow_test=arguments.allow_test,
        device=torch.device(arguments.device),
        dataset_filter="mnist",
        run_filter=None,
        seed_filter=arguments.seed,
    )


if __name__ == "__main__":
    main()
