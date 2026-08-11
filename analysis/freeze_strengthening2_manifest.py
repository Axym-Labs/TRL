"""Freeze a manifest from an explicit strengthening-2 configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from terel.resubmission.provenance import build_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--validation-ledger", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    ledger = json.loads(arguments.validation_ledger.read_text())
    if not ledger.get("selection_complete"):
        raise ValueError("validation selection is incomplete")
    configuration = yaml.safe_load(arguments.configuration.read_text())
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=arguments.configuration,
        validation_ledger_path=arguments.validation_ledger,
        repository=arguments.repository,
        configuration=configuration,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
