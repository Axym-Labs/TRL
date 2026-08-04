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
    for tracked_root, relative in ((repository, "module.py"), (paper, "main.tex")):
        tracked_root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=tracked_root, check=True)
        (tracked_root / relative).write_text(f"source={tracked_root}\n")
        subprocess.run(["git", "add", relative], cwd=tracked_root, check=True)
    artifacts.mkdir()
    (artifacts / "result.json").write_text(
        json.dumps({"path": str(repository), "accuracy": 0.97})
    )
    output = tmp_path / "supplement.zip"

    manifest = build_supplement_archive(
        repository=repository,
        paper_repository=paper,
        artifact_root=artifacts,
        artifact_paths=("result.json",),
        output_path=output,
    )

    assert not manifest["missing_optional_artifact_paths"]
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "TeReL-supplement/source/module.py" in names
        assert "TeReL-supplement/paper/main.tex" in names
        packaged = archive.read("TeReL-supplement/artifacts/result.json").decode()
    assert str(repository) not in packaged
    assert json.loads(packaged)["accuracy"] == 0.97
