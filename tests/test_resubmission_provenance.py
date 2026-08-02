import json
import subprocess

import pytest

from terel.resubmission.provenance import (
    TestGateError,
    assert_test_gate,
    build_run_manifest,
    sha256_file,
)


def _git(directory, *args):
    return subprocess.run(
        ["git", *args],
        cwd=directory,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_test_gate_requires_explicit_flag_and_exact_frozen_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "TeReL tests")
    tracked = repo / "implementation.py"
    tracked.write_text("VERSION = 1\n")
    _git(repo, "add", "implementation.py")
    _git(repo, "commit", "-m", "frozen implementation")

    protocol = tmp_path / "protocol.md"
    protocol.write_text("protocol version 1\n")
    ledger = tmp_path / "validation.json"
    ledger.write_text(json.dumps({"selection_complete": True}))
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=protocol,
        validation_ledger_path=ledger,
        repository=repo,
        configuration={"method": "terel", "seed": 1001},
    )

    with pytest.raises(TestGateError, match="explicit"):
        assert_test_gate(
            manifest,
            protocol_path=protocol,
            validation_ledger_path=ledger,
            repository=repo,
            explicit_allow_test=False,
        )

    assert_test_gate(
        manifest,
        protocol_path=protocol,
        validation_ledger_path=ledger,
        repository=repo,
        explicit_allow_test=True,
    )

    protocol.write_text("silently changed protocol\n")
    with pytest.raises(TestGateError, match="protocol hash"):
        assert_test_gate(
            manifest,
            protocol_path=protocol,
            validation_ledger_path=ledger,
            repository=repo,
            explicit_allow_test=True,
        )


def test_test_gate_rejects_tracked_code_changes_after_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "TeReL tests")
    tracked = repo / "implementation.py"
    tracked.write_text("VERSION = 1\n")
    _git(repo, "add", "implementation.py")
    _git(repo, "commit", "-m", "frozen implementation")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("protocol version 1\n")
    ledger = tmp_path / "validation.json"
    ledger.write_text("{}\n")
    manifest = build_run_manifest(
        phase="confirmatory",
        frozen=True,
        selection_complete=True,
        protocol_path=protocol,
        validation_ledger_path=ledger,
        repository=repo,
        configuration={"method": "terel"},
    )

    tracked.write_text("VERSION = 2\n")
    with pytest.raises(TestGateError, match="tracked changes"):
        assert_test_gate(
            manifest,
            protocol_path=protocol,
            validation_ledger_path=ledger,
            repository=repo,
            explicit_allow_test=True,
        )

    assert manifest["protocol_sha256"] == sha256_file(protocol)
