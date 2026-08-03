"""Freeze a fully resolved v2 confirmatory matrix after validation selection."""

import argparse
import json
from pathlib import Path

import yaml

from .confirmatory import _write_json
from .provenance import build_run_manifest


def validate_confirmatory_matrix(matrix):
    if not isinstance(matrix, dict):
        raise ValueError("confirmatory matrix must be a mapping")
    seeds = [int(seed) for seed in matrix.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("confirmatory matrix requires five distinct seeds")
    if matrix.get("evaluation_split") != "test":
        raise ValueError("confirmatory matrix must request the test split")
    datasets = matrix.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("confirmatory matrix requires at least one dataset")
    for dataset_name, dataset in datasets.items():
        runs = dataset.get("runs", [])
        identifiers = [run.get("id") for run in runs]
        if not identifiers or None in identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{dataset_name} run ids must be present and unique")
        for run in runs:
            if not isinstance(run.get("encoder"), dict):
                raise ValueError(f"{dataset_name}/{run.get('id')} has no encoder mapping")
    resolved = dict(matrix)
    resolved["seeds"] = seeds
    return resolved


def load_confirmatory_matrix(path):
    return validate_confirmatory_matrix(yaml.safe_load(Path(path).read_text()))


def freeze_v2_manifest(
    *, matrix_path, protocol_path, validation_ledger_path, repository, output_path
):
    configuration = load_confirmatory_matrix(matrix_path)
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=protocol_path,
        validation_ledger_path=validation_ledger_path,
        repository=repository,
        configuration=configuration,
    )
    manifest["matrix_path"] = str(Path(matrix_path).resolve())
    _write_json(output_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--validation-ledger", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    manifest = freeze_v2_manifest(
        matrix_path=arguments.matrix,
        protocol_path=arguments.protocol,
        validation_ledger_path=arguments.validation_ledger,
        repository=arguments.repository,
        output_path=arguments.output,
    )
    print(json.dumps({"configuration_sha256": manifest["configuration_sha256"]}))


if __name__ == "__main__":
    main()
