import json

import numpy as np
import torch

from terel.resubmission.capture24_confirmatory import _load_cache, _save_cache
from terel.resubmission.data import DatasetSplits, TemporalTensorDataset


def _dataset(value: float) -> TemporalTensorDataset:
    return TemporalTensorDataset(
        features=torch.tensor([[value, value + 1]], dtype=torch.float32),
        labels=torch.tensor([0]),
        boundaries=torch.tensor([True]),
    )


def test_capture24_confirmation_cache_preserves_split_roles(tmp_path):
    source = DatasetSplits(
        train=_dataset(1.0),
        validation=_dataset(2.0),
        test=_dataset(3.0),
        metadata={
            "dataset": "CAPTURE-24",
            "heldout_subjects_accessed": True,
            "test_subject_rows": {102: 1},
        },
    )
    path = tmp_path / "confirmation.npz"

    _save_cache(path, source)
    loaded = _load_cache(path)

    assert torch.equal(loaded.train.features, source.train.features)
    assert torch.equal(loaded.validation.features, source.validation.features)
    assert torch.equal(loaded.test.features, source.test.features)
    assert loaded.metadata["heldout_subjects_accessed"] is True
    assert loaded.metadata["test_subject_rows"] == {"102": 1}
    with np.load(path) as values:
        assert json.loads(str(values["metadata_json"]))["dataset"] == "CAPTURE-24"
