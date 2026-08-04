import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import torch


class TestGateError(RuntimeError):
    __test__ = False

    pass


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repository, *arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def git_provenance(repository):
    return {
        "code_commit": _git(repository, "rev-parse", "HEAD"),
        "tracked_dirty": bool(_git(repository, "status", "--porcelain", "--untracked-files=no")),
    }


def environment_record():
    packages = {}
    for name in (
        "matplotlib",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    try:
        cudnn_version = torch.backends.cudnn.version()
        cudnn_error = None
    except RuntimeError as error:
        cudnn_version = None
        cudnn_error = f"{type(error).__name__}: {error}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
        "cudnn_error": cudnn_error,
        "cuda_devices": cuda_devices,
        "packages": packages,
    }


def build_run_manifest(
    *,
    phase: str,
    frozen: bool,
    selection_complete: bool,
    protocol_path,
    validation_ledger_path,
    repository,
    configuration: dict,
):
    provenance = git_provenance(repository)
    return {
        "schema_version": 1,
        "phase": phase,
        "frozen": bool(frozen),
        "selection_complete": bool(selection_complete),
        "protocol_sha256": sha256_file(protocol_path),
        "validation_ledger_sha256": sha256_file(validation_ledger_path),
        "configuration_sha256": canonical_sha256(configuration),
        "configuration": configuration,
        "environment": environment_record(),
        **provenance,
    }


def assert_test_gate(
    manifest: dict,
    *,
    protocol_path,
    validation_ledger_path,
    repository,
    explicit_allow_test: bool,
):
    """Refuse held-out evaluation unless the preregistered state is exact and clean."""
    if not explicit_allow_test:
        raise TestGateError("held-out evaluation requires the explicit allow-test flag")
    if manifest.get("phase") != "confirmatory":
        raise TestGateError("manifest phase is not confirmatory")
    if not manifest.get("frozen"):
        raise TestGateError("manifest is not frozen")
    if not manifest.get("selection_complete"):
        raise TestGateError("validation selection is incomplete")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise TestGateError("protocol hash does not match the frozen manifest")
    if manifest.get("validation_ledger_sha256") != sha256_file(validation_ledger_path):
        raise TestGateError("validation ledger hash does not match the frozen manifest")

    provenance = git_provenance(repository)
    if provenance["tracked_dirty"]:
        raise TestGateError("repository has tracked changes after the manifest was created")
    if manifest.get("code_commit") != provenance["code_commit"]:
        raise TestGateError("code commit does not match the frozen manifest")
    if manifest.get("tracked_dirty"):
        raise TestGateError("manifest was created from tracked changes")
