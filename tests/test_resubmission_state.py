import torch

from terel.resubmission.state import TeReLState


def test_streaming_state_update_does_not_retain_autograd_graph():
    """Storing a live activation graph would violate bounded-history locality."""
    state = TeReLState(features=3, statistics_momentum=0.9, lateral_momentum=0.95)
    z = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    state.update(z)

    for tensor in (state.mean, state.variance, state.lateral, state.previous, state.previous_centered):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_dynamic_state_size_is_constant_in_stream_length():
    """The state footprint may depend on width, never on samples already seen."""
    state = TeReLState(features=3, statistics_momentum=0.9, lateral_momentum=0.95)
    expected_numel = 3 + 3 + 9 + 3 + 3 + 1

    before = state.dynamic_state_numel()
    for _ in range(20):
        state.update(torch.randn(1, 3, requires_grad=True))
    after = state.dynamic_state_numel()

    assert before == expected_numel
    assert after == expected_numel

