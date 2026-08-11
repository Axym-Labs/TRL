import importlib.util
from pathlib import Path

import pytest
import torch


def _module():
    path = Path(__file__).parents[1] / "analysis" / "strengthening2_online_drift.py"
    spec = importlib.util.spec_from_file_location("strengthening2_online_drift", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_accuracies_follow_stream_order():
    module = _module()
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 0])
    order = torch.tensor([1, 2, 0, 3])

    values = module._block_accuracies(logits, labels, order, block_size=2)

    assert values == pytest.approx([0.5, 1.0])
