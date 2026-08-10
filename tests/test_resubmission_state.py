import torch

from terel.resubmission.state import TeReLState


def test_streaming_state_update_does_not_retain_autograd_graph():
    """Storing a live activation graph would violate bounded-history locality."""
    state = TeReLState(features=3, statistics_momentum=0.9, lateral_momentum=0.95)
    z = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    state.update(z)

    assert state.previous_centered is None
    for tensor in (state.mean, state.variance, state.lateral, state.previous):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_dynamic_state_size_is_constant_in_stream_length():
    """The state footprint may depend on width, never on samples already seen."""
    state = TeReLState(features=3, statistics_momentum=0.9, lateral_momentum=0.95)
    expected_numel = 3 + 3 + 9 + 3 + 1

    before = state.dynamic_state_numel()
    for _ in range(20):
        state.update(torch.randn(1, 3, requires_grad=True))
    after = state.dynamic_state_numel()

    assert before == expected_numel
    assert after == expected_numel


def test_reset_sequence_clears_only_temporal_predecessors():
    """A new stream keeps learned statistics but cannot inherit a predecessor."""
    state = TeReLState(features=3, statistics_momentum=0.9, lateral_momentum=0.95)
    state.update(torch.tensor([[1.0, 2.0, 3.0]]))
    state.ensure_previous_centered().fill_(4.0)
    state.ensure_previous_neuron_state().fill_(5.0)
    state.ensure_residual_lateral().fill_(6.0)
    learned = {
        name: getattr(state, name).clone()
        for name in ("mean", "variance", "lateral", "residual_lateral")
    }

    state.reset_sequence()

    assert not bool(state.has_previous)
    assert torch.count_nonzero(state.previous) == 0
    assert torch.count_nonzero(state.previous_centered) == 0
    assert torch.count_nonzero(state.previous_neuron_state) == 0
    for name, value in learned.items():
        assert torch.equal(getattr(state, name), value)
