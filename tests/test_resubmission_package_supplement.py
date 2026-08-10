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
    (paper / "appendix").mkdir()
    (paper / "appendix" / "old.tex").write_text("development diary\n")
    (paper / "figures").mkdir()
    (paper / "figures" / "result.py").write_text("# final figure\n")
    subprocess.run(["git", "add", "."], cwd=paper, check=True)
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
        assert "TeReL-supplement/paper/figures/result.py" in names
        assert "TeReL-supplement/paper/appendix/old.tex" not in names
        packaged = archive.read("TeReL-supplement/artifacts/result.json").decode()
    assert str(private_source) not in packaged
    assert json.loads(packaged)["path"] == "artifacts/runs"
    assert json.loads(packaged)["accuracy"] == 0.97


def test_supplement_archive_anonymizes_paper_template_identifiers(tmp_path):
    repository = tmp_path / "workspace" / "code"
    paper = tmp_path / "workspace" / "paper"
    artifacts = tmp_path / "artifacts"
    for tracked_root in (repository, paper):
        tracked_root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=tracked_root, check=True)
    (repository / "module.py").write_text("# source\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    (paper / "main.tex").write_text("\\usepackage{axym-publication}\n")
    (paper / "axym-publication.sty").write_text(
        "\\ProvidesPackage{axym-publication}\n\\definecolor{AxymInk}{HTML}{111827}\n"
    )
    (paper / "README.md").write_text("Uses the Axym publication template.\n")
    subprocess.run(["git", "add", "."], cwd=paper, check=True)
    artifacts.mkdir()
    output = tmp_path / "supplement.zip"

    build_supplement_archive(
        repository=repository,
        paper_repository=paper,
        artifact_root=artifacts,
        artifact_paths=(),
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        paper_text = "\n".join(
            archive.read(name).decode()
            for name in names
            if name.startswith("TeReL-supplement/paper/")
        )
    assert "TeReL-supplement/paper/paper-template.sty" in names
    assert not any("axym" in name.lower() for name in names)
    assert "Axym" not in paper_text
    assert "axym-publication" not in paper_text
    assert "\\usepackage{paper-template}" in paper_text


def test_supplement_archive_uses_stable_comparator_provenance_names(tmp_path):
    repository = tmp_path / "workspace" / "TeReL"
    artifacts = tmp_path / "artifacts"
    repository.mkdir(parents=True)
    artifacts.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "module.py").write_text("# source\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    old_manifest = (
        tmp_path
        / "workspace/terel-paper-internal/resubmission-1/artifacts/"
        "review-patch-confirmatory-manifest-v3.json"
    )
    old_matrix = repository / "configs/resubmission/review-patch-confirmatory-matrix-v3.yaml"
    (artifacts / "result.json").write_text(
        json.dumps(
            {
                "manifest_path": str(old_manifest),
                "matrix_path": str(old_matrix),
                "selection_policy": (
                    "This control was fixed without changing or rerunning the v2 "
                    "headline matrix."
                ),
            }
        )
    )
    output = tmp_path / "supplement.zip"

    build_supplement_archive(
        repository=repository,
        artifact_root=artifacts,
        artifact_paths=("result.json",),
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        packaged = archive.read("TeReL-supplement/artifacts/result.json").decode()
    parsed = json.loads(packaged)
    assert parsed["manifest_path"] == "artifacts/local-comparator-manifest.json"
    assert parsed["matrix_path"] == "artifacts/local-comparator-manifest.json"
    assert "v2" not in parsed["selection_policy"]
    assert "final comparison matrix" in parsed["selection_policy"]


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
        "configs/resubmission/discarded-config.yaml": "old config\n",
        "configs/resubmission/residual-state-validation.yaml": "final config\n",
        "configs/canonical-online-learning.yaml": "canonical config\n",
        "configs/canonical-online-validation-ledger.json": "{}\n",
        "configs/canonical-online-mechanism.yaml": "mechanism config\n",
        "docs/canonical-online-protocol.md": "canonical protocol\n",
        "files/main.pdf": "old paper\n",
        "tests/test_legacy_regressions.py": "old regression\n",
        "tests/test_resubmission_package_supplement.py": "release tooling test\n",
        "terel/resubmission/package_supplement.py": "release tooling\n",
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
    assert not any("discarded-config.yaml" in name for name in names)
    assert not any("files/main.pdf" in name for name in names)
    assert not any("test_legacy_regressions.py" in name for name in names)
    assert not any("package_supplement.py" in name for name in names)
    assert not any("residual-state-validation.yaml" in name for name in names)
    assert "TeReL-supplement/source/configs/canonical-online-learning.yaml" in names
    assert (
        "TeReL-supplement/source/configs/canonical-online-validation-ledger.json"
        in names
    )
    assert "TeReL-supplement/source/configs/canonical-online-mechanism.yaml" in names
    assert "TeReL-supplement/source/docs/canonical-online-protocol.md" in names


def test_default_supplement_includes_final_scientific_records(tmp_path):
    repository = tmp_path / "workspace" / "code"
    artifacts = tmp_path / "artifacts"
    repository.mkdir(parents=True)
    artifacts.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    for filename in (
        "canonical-online-analysis.json",
        "canonical-online-confirmatory-manifest.json",
        "canonical-mechanism-validation.yaml",
        "objective-mechanism-analysis.json",
    ):
        (artifacts / filename).write_text("{}\n")
    result = artifacts / "canonical-online-confirmatory-results" / "mnist"
    result.mkdir(parents=True)
    (result / "seed.json").write_text("{}\n")
    validation = artifacts / "canonical-online-validation-results" / "inhibition"
    validation.mkdir(parents=True)
    (validation / "seed.json").write_text("{}\n")
    output = tmp_path / "supplement.zip"

    build_supplement_archive(
        repository=repository,
        artifact_root=artifacts,
        output_path=output,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert (
        "TeReL-supplement/artifacts/canonical-online-analysis.json" in names
    )
    assert (
        "TeReL-supplement/artifacts/canonical-online-confirmatory-manifest.json"
        in names
    )
    assert "TeReL-supplement/artifacts/canonical-mechanism-validation.yaml" in names
    assert (
        "TeReL-supplement/artifacts/objective-mechanism-analysis.json"
        in names
    )
    assert (
        "TeReL-supplement/artifacts/canonical-online-confirmatory-results/mnist/seed.json"
        in names
    )
    assert (
        "TeReL-supplement/artifacts/canonical-online-validation-results/inhibition/seed.json"
        in names
    )
