"""Build an anonymized, portable source-and-results supplement archive."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_ARTIFACT_PATHS = (
    "confirmatory-analysis-v2.json",
    "confirmatory-manifest-v2.json",
    "confirmatory-protocol-v2.md",
    "confirmatory-results-v2",
    "confirmatory-results",
    "locality-audit-context.md",
    "locality-audit.json",
    "latest-review-analysis-v4.json",
    "latest-review-confirmatory-manifest-v4.json",
    "latest-review-confirmatory-results-v4",
    "latest-review-control-protocol-v4.md",
    "mechanism-audit-analysis-v2.json",
    "mechanism-audit-analysis-v2-regenerated.json",
    "mechanism-audit-protocol-v2.md",
    "mechanism-audit-results-v2",
    "recovery-results-v2",
    "review-patch-analysis-v3.json",
    "review-patch-confirmatory-analysis-v3.json",
    "review-patch-confirmatory-manifest-v3.json",
    "review-patch-confirmatory-results-v3",
    "review-patch-validation-v3",
    "selection-results-final",
)


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
):
    repository = Path(repository).resolve()
    artifact_root = Path(artifact_root).resolve()
    output_path = Path(output_path).resolve()
    paper_repository = (
        Path(paper_repository).resolve() if paper_repository is not None else None
    )
    workspace_root = repository.parents[1]
    replacements = {
        str(workspace_root / "data" / "mnist"): "data/mnist",
        str(workspace_root / "data" / "pamap2"): "data/pamap2",
        str(artifact_root): "artifacts",
        str(repository): ".",
        str(workspace_root): "<WORKSPACE>",
    }
    if paper_repository is not None:
        replacements[str(paper_repository)] = "paper"
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

    for relative in _tracked_files(repository):
        add_file(repository / relative, Path("source") / relative, "tracked_source")
    if paper_repository is not None:
        for relative in _tracked_files(paper_repository):
            add_file(
                paper_repository / relative,
                Path("paper") / relative,
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
    arguments = parser.parse_args(argv)
    manifest = build_supplement_archive(
        repository=arguments.repository,
        artifact_root=arguments.artifact_root,
        output_path=arguments.output,
        paper_repository=arguments.paper_repository,
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
