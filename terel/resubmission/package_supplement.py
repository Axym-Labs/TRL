"""Build an anonymized, portable source-and-results supplement archive."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_ARTIFACT_PATHS = (
    "canonical-online-analysis.json",
    "canonical-online-confirmatory-manifest.json",
    "canonical-online-confirmatory-results",
    "canonical-online-validation-results",
    "canonical-mechanism-validation.yaml",
    "canonical-mechanism-results",
    "batched-reference-analysis.json",
    "batched-reference-manifest.json",
    "batched-reference-results",
    "objective-mechanism-analysis.json",
    "objective-mechanism-protocol.md",
    "objective-mechanism-results",
    "local-comparator-analysis.json",
    "local-comparator-selection-analysis.json",
    "local-comparator-final-results",
    "local-comparator-validation-results",
    "normalization-control-analysis.json",
    "normalization-control-manifest.json",
    "normalization-control-results",
    "locality-audit-context.md",
    "locality-audit.json",
)

SOURCE_EXCLUDED_PREFIXES = (
    ".codex/",
    "analysis/",
    "analysis_outputs/",
    "artifacts/",
    "comparison/",
    "experiments_mnist/",
    "files/",
    "previous_versions/",
    "runs/",
    "scripts/",
)
SOURCE_EXCLUDED_FILES = {
    ".codex-autoresearch.md",
    "all_comparison_runs.py",
    "all_comparison_runs_out.log",
    "tests/test_legacy_regressions.py",
    "tests/test_resubmission_package_supplement.py",
    "tests/test_resubmission_analysis_v2.py",
    "tests/test_resubmission_confirmatory_v2.py",
    "tests/test_resubmission_mechanism_analysis_v2.py",
    "tests/test_resubmission_recovery.py",
    "terel/resubmission/analysis_v2.py",
    "terel/resubmission/confirmatory_v2.py",
    "terel/resubmission/mechanism_analysis_v2.py",
    "terel/resubmission/package_supplement.py",
    "terel/resubmission/recovery.py",
    "terel/resubmission/latest_review_analysis.py",
    "terel/resubmission/review_patch_analysis.py",
    "tests/test_resubmission_latest_review_analysis.py",
    "tests/test_resubmission_review_patch_analysis.py",
    "train.py",
}
SOURCE_INCLUDED_CONFIGS = {
    "configs/canonical-online-learning.yaml",
    "configs/canonical-online-mechanism.yaml",
    "configs/canonical-online-validation-ledger.json",
}
SOURCE_INCLUDED_DOCS = {"docs/canonical-online-protocol.md"}

PAPER_INCLUDED_FILES = {
    ".gitignore",
    "README.md",
    "axym-publication.sty",
    "main.pdf",
    "main.tex",
    "references.bib",
}
PAPER_INCLUDED_PREFIXES = ("figures/",)
PAPER_ARCHIVE_RENAMES = {
    "axym-publication.sty": "paper-template.sty",
}


def sanitize_text_artifact(text, *, replacements):
    """Apply deterministic longest-prefix path redactions to a text artifact."""
    for source in sorted(replacements, key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _tracked_files(repository):
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode()) for item in completed.stdout.split(b"\0") if item]


def _supplement_source_files(repository):
    """Return final source while omitting tracked development-diary material."""
    selected = []
    for relative in _tracked_files(repository):
        normalized = relative.as_posix()
        if normalized in SOURCE_EXCLUDED_FILES:
            continue
        if normalized.startswith(SOURCE_EXCLUDED_PREFIXES):
            continue
        if normalized.startswith("configs/") and normalized not in SOURCE_INCLUDED_CONFIGS:
            continue
        if normalized.startswith("docs/") and normalized not in SOURCE_INCLUDED_DOCS:
            continue
        selected.append(relative)
    return selected


def _supplement_paper_files(repository):
    """Return the current manuscript while omitting unreferenced old material."""
    selected = []
    for relative in _tracked_files(repository):
        normalized = relative.as_posix()
        if (
            normalized in PAPER_INCLUDED_FILES
            or normalized.startswith(PAPER_INCLUDED_PREFIXES)
            or (
                normalized.startswith("generated_")
                and normalized.endswith(".tex")
            )
        ):
            selected.append(relative)
    return selected


def _artifact_files(artifact_root, requested_paths):
    files = []
    missing = []
    for relative in requested_paths:
        source = artifact_root / relative
        if not source.exists():
            missing.append(relative)
        elif source.is_dir():
            files.extend(path for path in sorted(source.rglob("*")) if path.is_file())
        else:
            files.append(source)
    return files, missing


def _sanitized_payload(path, replacements):
    original = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return original, original, False
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        return original, original, False
    sanitized = sanitize_text_artifact(text, replacements=replacements).encode("utf-8")
    return original, sanitized, sanitized != original


def _write_zip_member(archive, name, payload):
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 4, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def build_supplement_archive(
    *,
    repository,
    artifact_root,
    output_path,
    artifact_paths=DEFAULT_ARTIFACT_PATHS,
    paper_repository=None,
    private_roots=(),
):
    repository = Path(repository).resolve()
    artifact_root = Path(artifact_root).resolve()
    output_path = Path(output_path).resolve()
    paper_repository = (
        Path(paper_repository).resolve() if paper_repository is not None else None
    )
    workspace_root = repository.parents[1]
    replacements = {
        "paper-internal/resubmission-1/artifacts/confirmatory-manifest-v2.json": (
            "artifacts/batched-reference-manifest.json"
        ),
        "./configs/resubmission/confirmatory-matrix-v2.yaml": (
            "artifacts/batched-reference-manifest.json"
        ),
        "axym-publication": "paper-template",
        "Axym publication": "included publication",
        "Axym": "Paper",
        str(workspace_root / "data" / "mnist"): "data/mnist",
        str(workspace_root / "data" / "pamap2"): "data/pamap2",
        str(artifact_root): "artifacts",
        str(repository): ".",
        str(workspace_root): "<WORKSPACE>",
    }
    if paper_repository is not None:
        replacements[str(paper_repository)] = "paper"
    for private_root in private_roots:
        replacements[str(Path(private_root).resolve())] = "artifacts"
    artifact_files, missing = _artifact_files(artifact_root, artifact_paths)
    payloads = []
    manifest_entries = []

    def add_file(source, archive_relative, source_kind):
        original, packaged, redacted = _sanitized_payload(source, replacements)
        archive_name = f"TeReL-supplement/{archive_relative.as_posix()}"
        payloads.append((archive_name, packaged))
        manifest_entries.append(
            {
                "path": archive_relative.as_posix(),
                "source_kind": source_kind,
                "source_sha256": _sha256(original),
                "packaged_sha256": _sha256(packaged),
                "path_redacted": redacted,
            }
        )

    for relative in _supplement_source_files(repository):
        add_file(repository / relative, Path("source") / relative, "tracked_source")
    if paper_repository is not None:
        for relative in _supplement_paper_files(paper_repository):
            archive_relative = Path(PAPER_ARCHIVE_RENAMES.get(relative.as_posix(), relative))
            add_file(
                paper_repository / relative,
                Path("paper") / archive_relative,
                "tracked_paper_source",
            )
    for source in artifact_files:
        add_file(
            source,
            Path("artifacts") / source.relative_to(artifact_root),
            "executed_artifact",
        )

    manifest = {
        "schema_version": 1,
        "description": (
            "Anonymized portable TeReL supplement. Path-only redactions are "
            "identified by paired source and packaged SHA-256 values."
        ),
        "missing_optional_artifact_paths": missing,
        "files": sorted(manifest_entries, key=lambda item: item["path"]),
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name, payload in sorted(payloads):
            _write_zip_member(archive, name, payload)
        _write_zip_member(
            archive,
            "TeReL-supplement/redaction-manifest.json",
            manifest_payload,
        )
    temporary.replace(output_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper-repository")
    parser.add_argument(
        "--private-root",
        action="append",
        default=[],
        help="Private source prefix to rewrite as 'artifacts' inside text files",
    )
    arguments = parser.parse_args(argv)
    manifest = build_supplement_archive(
        repository=arguments.repository,
        artifact_root=arguments.artifact_root,
        output_path=arguments.output,
        paper_repository=arguments.paper_repository,
        private_roots=arguments.private_root,
    )
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "missing_optional_artifact_paths": manifest[
                    "missing_optional_artifact_paths"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
