"""Freeze and execute the held-out resubmission experiment matrix."""

import argparse
import json
from pathlib import Path

import torch

from .data import load_mnist_protocol, load_pamap2_protocol
from .experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    TestGateContext,
    run_representation_experiment,
)
from .provenance import build_run_manifest, canonical_sha256
from .selection import load_selection_plan, validate_plan_protocol


CONFIRMATORY_SEEDS = (1001, 1002, 1003, 1004, 1005)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _selected_encoder(selection_plan, validation_ledger, dataset_name):
    dataset_plan = selection_plan["datasets"][dataset_name]
    selected_id = validation_ledger["datasets"][dataset_name].get(
        "selected_configuration_id"
    )
    selected = next(
        (
            configuration
            for configuration in dataset_plan["configurations"]
            if configuration.get("id") == selected_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"no selected configuration for {dataset_name}")
    encoder = dict(dataset_plan["encoder_base"])
    encoder.update({key: value for key, value in selected.items() if key != "id"})
    return encoder, selected_id


def _run(identifier, encoder, **overrides):
    resolved = dict(encoder)
    resolved.update(overrides)
    return {"id": identifier, "encoder": resolved}


def resolve_confirmatory_configuration(
    selection_plan,
    validation_ledger,
    *,
    confirmatory_seeds=CONFIRMATORY_SEEDS,
):
    """Resolve selected TeReL settings into a fixed matched-control matrix."""
    if not validation_ledger.get("selection_complete"):
        raise ValueError("validation selection is incomplete")
    seeds = [int(seed) for seed in confirmatory_seeds]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("confirmatory execution requires five distinct seeds")

    mnist_encoder, mnist_selected = _selected_encoder(
        selection_plan, validation_ledger, "mnist"
    )
    pamap_encoder, pamap_selected = _selected_encoder(
        selection_plan, validation_ledger, "pamap2"
    )
    mnist_runs = [
        _run("terel-local", mnist_encoder),
        _run("random", mnist_encoder, method="random"),
        _run("local-supcon", mnist_encoder, method="local_supcon"),
        _run("bp", mnist_encoder, method="bp"),
        _run("direct-covariance", mnist_encoder, method="terel_direct"),
    ]
    pamap_runs = [
        _run("terel-ordered", pamap_encoder),
        _run("terel-shuffled", pamap_encoder, order_mode="shuffled"),
        _run("random", pamap_encoder, method="random"),
        _run("bp", pamap_encoder, method="bp"),
        _run("batch-sfa", pamap_encoder, method="sfa", sfa_components=52),
        _run(
            "incremental-sfa",
            pamap_encoder,
            method="incsfa",
            incsfa_whitening_dim=52,
            incsfa_output_dim=52,
            incsfa_learning_rate=0.001,
        ),
        _run("direct-covariance", pamap_encoder, method="terel_direct"),
    ]

    def dataset_entry(name, runs, selected_id):
        source = selection_plan["datasets"][name]
        return {
            "data_root": source["data_root"],
            "num_classes": int(source["num_classes"]),
            "selected_configuration_id": selected_id,
            "runs": runs,
        }

    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "seeds": seeds,
        "selection_seeds": [int(seed) for seed in selection_plan["seeds"]],
        "probe": dict(selection_plan["probe"]),
        "selection_policy": (
            "Controls inherit the selected TeReL architecture, optimizer, learning rate, "
            "batch size, and epoch budget; only method-defining fields change."
        ),
        "datasets": {
            "mnist": dataset_entry("mnist", mnist_runs, mnist_selected),
            "pamap2": dataset_entry("pamap2", pamap_runs, pamap_selected),
        },
    }


def freeze_confirmatory_manifest(
    *,
    selection_plan_path,
    validation_ledger_path,
    protocol_path,
    repository,
    output_path,
):
    plan = load_selection_plan(selection_plan_path)
    validate_plan_protocol(plan, protocol_path)
    ledger = json.loads(Path(validation_ledger_path).read_text())
    configuration = resolve_confirmatory_configuration(plan, ledger)
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


def _load_dataset(name, dataset):
    if name == "mnist":
        return load_mnist_protocol(dataset["data_root"], allow_download=False)
    if name == "pamap2":
        return load_pamap2_protocol(dataset["data_root"], stride=10, allow_download=False)
    raise ValueError(f"unknown confirmatory dataset: {name}")


def run_confirmatory_manifest(
    manifest,
    *,
    manifest_path,
    output_directory,
    protocol_path,
    validation_ledger_path,
    repository,
    explicit_allow_test,
    device,
    dataset_filter=None,
    run_filter=None,
    seed_filter=None,
):
    configuration = manifest["configuration"]
    if manifest.get("configuration_sha256") != canonical_sha256(configuration):
        raise ValueError("manifest configuration checksum is invalid")
    if configuration.get("evaluation_split") != "test":
        raise ValueError("confirmatory manifest does not request the test split")
    output_directory = Path(output_directory)
    probe = ProbeExperimentConfig(**configuration["probe"])
    gate = TestGateContext(
        manifest=manifest,
        protocol_path=str(protocol_path),
        validation_ledger_path=str(validation_ledger_path),
        repository=str(repository),
        explicit_allow_test=bool(explicit_allow_test),
    )
    completed = []
    for dataset_name, dataset in configuration["datasets"].items():
        if dataset_filter is not None and dataset_name != dataset_filter:
            continue
        splits = _load_dataset(dataset_name, dataset)
        for run in dataset["runs"]:
            if run_filter is not None and run["id"] != run_filter:
                continue
            encoder_values = dict(run["encoder"])
            encoder_values["hidden_dims"] = tuple(encoder_values["hidden_dims"])
            encoder = EncoderExperimentConfig(**encoder_values)
            for seed in configuration["seeds"]:
                if seed_filter is not None and int(seed) != int(seed_filter):
                    continue
                output_path = output_directory / dataset_name / run["id"] / f"seed-{seed}.json"
                if output_path.exists():
                    result = json.loads(output_path.read_text())
                    expected_identity = (
                        dataset_name,
                        run["id"],
                        int(seed),
                        manifest["configuration_sha256"],
                        manifest["code_commit"],
                    )
                    observed_identity = (
                        result.get("dataset"),
                        result.get("run_id"),
                        result.get("seed"),
                        result.get("manifest_configuration_sha256"),
                        result.get("manifest_code_commit"),
                    )
                    if observed_identity != expected_identity:
                        raise ValueError(f"stale or mismatched confirmatory output: {output_path}")
                else:
                    result = run_representation_experiment(
                        splits=splits,
                        dataset_name=dataset_name,
                        num_classes=int(dataset["num_classes"]),
                        seed=int(seed),
                        encoder=encoder,
                        probe=probe,
                        evaluation_split="test",
                        device=device,
                        test_gate=gate,
                    )
                    result.update(
                        {
                            "run_id": run["id"],
                            "manifest_configuration_sha256": manifest[
                                "configuration_sha256"
                            ],
                            "manifest_code_commit": manifest["code_commit"],
                            "manifest_path": str(Path(manifest_path).resolve()),
                        }
                    )
                    _write_json(output_path, result)
                completed.append(str(output_path.relative_to(output_directory)))
                _write_json(
                    output_directory / "confirmatory-ledger.partial.json",
                    {
                        "manifest_configuration_sha256": manifest[
                            "configuration_sha256"
                        ],
                        "completed": completed,
                    },
                )
    expected_paths = [
        output_directory / dataset_name / run["id"] / f"seed-{seed}.json"
        for dataset_name, dataset in configuration["datasets"].items()
        for run in dataset["runs"]
        for seed in configuration["seeds"]
    ]
    completed_paths = [path for path in expected_paths if path.exists()]
    ledger = {
        "manifest_configuration_sha256": manifest["configuration_sha256"],
        "completed": sorted(
            str(path.relative_to(output_directory)) for path in completed_paths
        ),
        "expected_runs": len(expected_paths),
        "confirmatory_complete": len(completed_paths) == len(expected_paths),
    }
    _write_json(output_directory / "confirmatory-ledger.json", ledger)
    return ledger


def main(argv=None):
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
    run.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    run.add_argument("--allow-test", action="store_true")
    run.add_argument("--dataset", choices=("mnist", "pamap2"))
    run.add_argument("--run-id")
    run.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    if arguments.command == "freeze":
        freeze_confirmatory_manifest(
            selection_plan_path=arguments.selection_plan,
            validation_ledger_path=arguments.validation_ledger,
            protocol_path=arguments.protocol,
            repository=arguments.repository,
            output_path=arguments.output,
        )
    else:
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
            dataset_filter=arguments.dataset,
            run_filter=arguments.run_id,
            seed_filter=arguments.seed,
        )


if __name__ == "__main__":
    main()
