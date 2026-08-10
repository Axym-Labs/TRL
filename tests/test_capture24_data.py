from pathlib import Path

import pandas as pd
import torch

from terel.resubmission import data


def _write_subject(path: Path, *, annotation: str) -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5s"),
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 3.0, 4.0, 5.0],
            "z": [3.0, 4.0, 5.0, 6.0],
            "annotation": [annotation] * 4,
        }
    )
    frame.to_csv(path, index=False)


def test_capture24_protocol_hides_heldout_until_explicit_access(tmp_path):
    pd.DataFrame(
        {
            "annotation": ["sleep-code", "walk-code"],
            "label:Willetts2018": ["sleep", "walking"],
        }
    ).to_csv(tmp_path / "annotation-label-dictionary.csv", index=False)
    _write_subject(tmp_path / "P001.csv.gz", annotation="sleep-code")
    _write_subject(tmp_path / "P009.csv.gz", annotation="walk-code")
    (tmp_path / "P102.csv.gz").write_bytes(b"malformed heldout data")

    development = data.load_capture24_protocol(
        tmp_path,
        train_subjects=(1,),
        validation_subjects=(9,),
        heldout_subjects=(102,),
        access_heldout=False,
    )

    assert len(development.train) == 2
    assert len(development.validation) == 2
    assert development.test is development.validation
    assert torch.equal(development.train.labels, torch.tensor([3, 3]))
    assert torch.equal(development.validation.labels, torch.tensor([5, 5]))
    assert development.metadata["heldout_subjects_accessed"] is False
    assert "P102.csv.gz" not in development.metadata["source_sha256"]

    _write_subject(tmp_path / "P102.csv.gz", annotation="sleep-code")
    confirmatory = data.load_capture24_protocol(
        tmp_path,
        train_subjects=(1,),
        validation_subjects=(9,),
        heldout_subjects=(102,),
        access_heldout=True,
    )

    assert len(confirmatory.test) == 2
    assert confirmatory.metadata["heldout_subjects_accessed"] is True
    assert confirmatory.metadata["test_subject_rows"] == {102: 2}
    assert "P102.csv.gz" in confirmatory.metadata["source_sha256"]
