import json
import subprocess
import zipfile

from terel.resubmission.package_supplement import (
    build_supplement_archive,
    sanitize_text_artifact,
)


def test_supplement_sanitizer_removes_local_paths_without_changing_numbers():
    original = json.dumps(
        {
            "data_root": "/private/reviewer-workspace/data/mnist",
            "manifest_path": "/private/reviewer-workspace/manifest.json",
            "accuracy": 0.973,
            "seed": 1101,
        }
    )

    sanitized = sanitize_text_artifact(
        original,
        replacements={
            "/private/reviewer-workspace/data/mnist": "data/mnist",
            "/private/reviewer-workspace": "<WORKSPACE>",
        },
    )
    parsed = json.loads(sanitized)

    assert "/private/reviewer-workspace" not in sanitized
    assert parsed["data_root"] == "data/mnist"
    assert parsed["accuracy"] == 0.973
    assert parsed["seed"] == 1101


def test_supplement_archive_includes_tracked_code_paper_and_artifacts(tmp_path):
    repository = tmp_path / "workspace" / "code"
    paper = tmp_path / "workspace" / "paper"
    artifacts = tmp_path / "private-artifacts"
    private_source = tmp_path / "internal" / "task-arc"
    for tracked_root, relative in ((repository, "module.py"), (paper, "main.tex")):
        tracked_root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=tracked_root, check=True)
        (tracked_root / relative).write_text(f"source={tracked_root}\n")
        subprocess.run(["git", "add", relative], cwd=tracked_root, check=True)
    artifacts.mkdir()
    (artifacts / "result.json").write_text(
        json.dumps({"path": str(private_source / "runs"), "accuracy": 0.97})
    )
    output = tmp_path / "supplement.zip"

    manifest = build_supplement_archive(
        repository=repository,
        paper_repository=paper,
        artifact_root=artifacts,
        artifact_paths=("result.json",),
        output_path=output,
        private_roots=(private_source,),
    )

    assert not manifest["missing_optional_artifact_paths"]
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "TeReL-supplement/source/module.py" in names
        assert "TeReL-supplement/paper/main.tex" in names
        packaged = archive.read("TeReL-supplement/artifacts/result.json").decode()
    assert str(private_source) not in packaged
    assert json.loads(packaged)["path"] == "artifacts/runs"
    assert json.loads(packaged)["accuracy"] == 0.97


def test_supplement_archive_excludes_tracked_development_diary(tmp_path):
    repository = tmp_path / "workspace" / "code"
    artifacts = tmp_path / "artifacts"
    repository.mkdir(parents=True)
    artifacts.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = {
        "terel/module.py": "final source\n",
        "analysis_outputs/diagnostic.json": "{}\n",
        "artifacts/recovery/seed.json": "{}\n",
        "previous_versions/old.py": "old source\n",
        ".codex-autoresearch.md": "private diary\n",
        "configs/resubmission/recovery-v2.yaml": "old config\n",
        "configs/resubmission/residual-state-validation.yaml": "final config\n",
        "files/main.pdf": "old paper\n",
        "tests/test_legacy_regressions.py": "old regression\n",
    }
    for relative, payload in tracked.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    (artifacts / "result.json").write_text("{}\n")
    output = tmp_path / "supplement.zip"

    build_supplement_archive(
        repository=repository,
        artifact_root=artifacts,
        artifact_paths=("result.json",),
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "TeReL-supplement/source/terel/module.py" in names
    assert not any("analysis_outputs" in name for name in names)
    assert not any("artifacts/recovery" in name for name in names)
    assert not any("previous_versions" in name for name in names)
    assert not any(".codex" in name for name in names)
    assert not any("recovery-v2.yaml" in name for name in names)
    assert not any("files/main.pdf" in name for name in names)
    assert not any("test_legacy_regressions.py" in name for name in names)
    assert (
        "TeReL-supplement/source/configs/resubmission/residual-state-validation.yaml"
        in names
    )


def test_default_supplement_includes_final_scientific_records(tmp_path):
    repository = tmp_path / "workspace" / "code"
    artifacts = tmp_path / "artifacts"
    repository.mkdir(parents=True)
    artifacts.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    for filename in (
        "residual-state-final-analysis.json",
        "residual-state-final-manifest.json",
        "residual-state-validation-config.yaml",
        "objective-mechanism-analysis.json",
    ):
        (artifacts / filename).write_text("{}\n")
    result = artifacts / "residual-state-final-results" / "mnist"
    result.mkdir(parents=True)
    (result / "seed.json").write_text("{}\n")
    output = tmp_path / "supplement.zip"

    build_supplement_archive(
        repository=repository,
        artifact_root=artifacts,
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "TeReL-supplement/artifacts/residual-state-final-analysis.json" in names
    assert "TeReL-supplement/artifacts/residual-state-final-manifest.json" in names
    assert "TeReL-supplement/artifacts/residual-state-validation-config.yaml" in names
    assert (
        "TeReL-supplement/artifacts/objective-mechanism-analysis.json"
        in names
    )
    assert (
        "TeReL-supplement/artifacts/residual-state-final-results/mnist/seed.json"
        in names
    )
