from pathlib import Path

import torch

from terel.resubmission import data


def _write_subject(path: Path, subject: int, activities: tuple[str, ...]) -> None:
    rows = []
    timestamp = 0
    for activity in activities:
        for index in range(4):
            rows.append(
                f"{subject},{activity},{timestamp},{index + 1}.0,{index + 2}.0,"
                f"{index + 3}.0;\n"
            )
            timestamp += 50
        timestamp = 0
    path.write_text("".join(rows))


def test_wisdm_development_protocol_windows_records_and_hides_heldout(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_subject(raw / "data_1600_accel_watch.txt", 1600, ("A", "B"))
    _write_subject(raw / "data_1631_accel_watch.txt", 1631, ("A", "B"))
    (raw / "data_1641_accel_watch.txt").write_text("malformed heldout data")

    splits = data.load_wisdm_development_protocol(
        raw,
        train_subjects=(1600,),
        validation_subjects=(1631,),
        heldout_subjects=(1641,),
        window_samples=4,
    )

    assert len(splits.train) == 2
    assert len(splits.validation) == 2
    assert splits.train.features.shape == (2, 16)
    assert torch.equal(splits.train.boundaries, torch.tensor([True, True]))
    assert torch.equal(splits.train.labels, torch.tensor([0, 1]))
    assert splits.metadata["train_subject_rows"] == {1600: 2}
    assert splits.metadata["validation_subject_rows"] == {1631: 2}
    assert splits.metadata["heldout_subjects_accessed"] is False
    assert splits.test is splits.validation
