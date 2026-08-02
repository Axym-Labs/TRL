import torch

from terel.resubmission.objective import (
    LossCoefficients,
    detached_terel_loss,
    direct_offdiagonal_covariance_loss,
    lateral_proxy_error_bound,
    terel_loss,
    temporal_references,
)
from terel.resubmission.state import TeReLState


def test_variance_term_pushes_low_variance_activation_away_from_mean():
    """A positive variance coefficient must oppose collapse, not reinforce it."""
    z = torch.tensor([[0.5]], requires_grad=True)
    loss, _ = detached_terel_loss(
        z=z,
        previous=torch.zeros_like(z),
        mean=torch.zeros(1),
        variance=torch.zeros(1),
        lateral=torch.zeros(1, 1),
        pair_valid=torch.tensor([False]),
        coefficients=LossCoefficients(similarity=0.0, variance=1.0, covariance=0.0),
        variance_target=1.0,
    )

    loss.backward()

    assert torch.allclose(loss.detach(), torch.tensor(-0.25))
    assert torch.allclose(z.grad, torch.tensor([[-1.0]]))


def test_detached_objective_does_not_backpropagate_into_previous_activation():
    """Reintroducing a temporal graph must make this locality test fail."""
    z = torch.tensor([[2.0]], requires_grad=True)
    previous = torch.tensor([[1.0]], requires_grad=True)
    loss, _ = detached_terel_loss(
        z=z,
        previous=previous,
        mean=torch.zeros(1),
        variance=torch.ones(1),
        lateral=torch.zeros(1, 1),
        pair_valid=torch.tensor([True]),
        coefficients=LossCoefficients(similarity=1.0, variance=0.0, covariance=0.0),
        variance_target=1.0,
    )

    loss.backward()

    assert torch.allclose(z.grad, torch.tensor([[2.0]]))
    assert previous.grad is None


def test_temporal_references_respect_stream_boundaries():
    """A temporal loss must never join independent sequences or class chunks."""
    state = TeReLState(features=1, statistics_momentum=0.9, lateral_momentum=0.9)
    state.update(torch.tensor([[9.0]]))
    z = torch.tensor([[1.0], [2.0], [5.0], [6.0]], requires_grad=True)
    boundaries = torch.tensor([False, False, True, False])

    previous, valid = temporal_references(z, state=state, boundaries=boundaries, detach=True)

    assert torch.equal(previous, torch.tensor([[9.0], [1.0], [2.0], [5.0]]))
    assert torch.equal(valid, torch.tensor([True, True, False, True]))
    assert previous.requires_grad is False


def test_undetached_pair_gradient_contains_both_temporal_factors():
    """The batched variant must expose the previous-sample derivative explicitly."""
    z = torch.tensor([[1.0], [3.0]], requires_grad=True)
    state = TeReLState(features=1, statistics_momentum=0.9, lateral_momentum=0.9)
    previous, valid = temporal_references(
        z,
        state=state,
        boundaries=torch.tensor([True, False]),
        detach=False,
    )

    loss, _ = terel_loss(
        z=z,
        previous=previous,
        mean=torch.zeros(1),
        variance=torch.ones(1),
        lateral=torch.zeros(1, 1),
        pair_valid=valid,
        coefficients=LossCoefficients(similarity=1.0, variance=0.0, covariance=0.0),
        variance_target=1.0,
        detach_previous=False,
    )
    loss.backward()

    assert torch.allclose(loss.detach(), torch.tensor(4.0))
    assert torch.allclose(z.grad, torch.tensor([[-4.0], [4.0]]))


def test_shifted_lateral_proxy_obeys_operator_norm_error_bound():
    """The shifted approximation claim must be a checkable inequality."""
    lateral = torch.tensor([[1.0, 0.2], [0.0, 0.5]])
    target = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
    current = torch.tensor([2.0, -1.0])
    previous = torch.tensor([1.5, -0.5])

    actual, bound = lateral_proxy_error_bound(
        lateral=lateral,
        target=target,
        current=current,
        reference=previous,
    )

    assert actual <= bound + 1e-6
    assert actual > 0.0


def test_direct_covariance_control_is_zero_only_for_decorrelated_fixture():
    """The direct control must measure off-diagonal energy, not total variance."""
    correlated = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    decorrelated = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])

    correlated_loss = direct_offdiagonal_covariance_loss(correlated, mean=torch.zeros(2))
    decorrelated_loss = direct_offdiagonal_covariance_loss(decorrelated, mean=torch.zeros(2))

    assert torch.allclose(correlated_loss, torch.tensor(1.0))
    assert torch.allclose(decorrelated_loss, torch.tensor(0.0))


def test_covariance_proxy_can_use_an_explicit_shifted_reference():
    """The asynchronous ablation must use the previous centered activation, not the current one."""
    z = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss, _ = terel_loss(
        z=z,
        previous=torch.zeros_like(z),
        mean=torch.zeros(2),
        variance=torch.ones(2),
        lateral=torch.eye(2),
        lateral_reference=torch.tensor([[3.0, 4.0]]),
        pair_valid=torch.tensor([False]),
        coefficients=LossCoefficients(similarity=0.0, variance=0.0, covariance=1.0),
        variance_target=1.0,
        detach_previous=True,
    )

    assert torch.allclose(loss, torch.tensor(5.5))

