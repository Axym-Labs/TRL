from pathlib import Path

import pandas as pd
import torch

from terel.resubmission import data

ACTIVITY_COLUMNS = (
    "label:LYING_DOWN",
    "label:SITTING",
    "label:OR_standing",
    "label:FIX_walking",
    "label:FIX_running",
    "label:BICYCLING",
)


def _write_subject(root: Path, subject: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    for column in ACTIVITY_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    frame.to_csv(root / f"{subject}.features_labels.csv.gz", index=False)


def _write_fold(root: Path, fold: int, split: str, subjects: tuple[str, ...]) -> None:
    (root / f"fold_{fold}_{split}_android_uuids.txt").write_text(
        "".join(f"{subject}\n" for subject in subjects)
    )
    (root / f"fold_{fold}_{split}_iphone_uuids.txt").write_text("")


def test_extrasensory_development_protocol_preserves_gaps_and_hides_fold_zero(tmp_path):
    """Reading held-out users or joining a long timestamp gap must fail this test."""
    features = tmp_path / "features"
    folds = tmp_path / "folds"
    features.mkdir()
    folds.mkdir()
    _write_subject(
        features,
        "train-user",
        [
            {"timestamp": 0, "raw_acc:a": 1.0, "raw_acc:b": 2.0, "label:SITTING": 1.0},
            {"timestamp": 60, "raw_acc:a": 2.0, "raw_acc:b": 4.0},
            {
                "timestamp": 200,
                "raw_acc:a": 3.0,
                "raw_acc:b": 6.0,
                "label:FIX_walking": 1.0,
            },
        ],
    )
    _write_subject(
        features,
        "validation-user",
        [
            {
                "timestamp": 0,
                "raw_acc:a": 2.0,
                "raw_acc:b": 4.0,
                "label:LYING_DOWN": 1.0,
            },
            {
                "timestamp": 60,
                "raw_acc:a": 3.0,
                "raw_acc:b": 6.0,
                "label:OR_standing": 1.0,
            },
        ],
    )
    # A malformed held-out file proves that development loading never opens it.
    (features / "heldout-user.features_labels.csv.gz").write_bytes(b"not a gzip file")
    _write_fold(folds, 0, "test", ("heldout-user",))
    _write_fold(folds, 1, "test", ("validation-user",))

    splits = data.load_extrasensory_development_protocol(features, folds_root=folds)

    assert len(splits.train) == 3
    assert len(splits.validation) == 2
    assert torch.equal(splits.train.boundaries, torch.tensor([True, False, True]))
    assert torch.equal(splits.train.labels, torch.tensor([1, -1, 3]))
    assert torch.equal(splits.validation.labels, torch.tensor([0, 2]))
    assert splits.metadata["train_subjects"] == ["train-user"]
    assert splits.metadata["validation_subjects"] == ["validation-user"]
    assert splits.metadata["train_subject_rows"] == {"train-user": 3}
    assert splits.metadata["validation_subject_rows"] == {"validation-user": 2}
    assert splits.metadata["heldout_subjects"] == ["heldout-user"]
    assert splits.metadata["heldout_subjects_accessed"] is False
    assert splits.test is splits.validation
