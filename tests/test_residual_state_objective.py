import torch

from terel.resubmission import objective
from terel.resubmission.state import TeReLState


def test_residual_state_separates_causal_scalars_from_auxiliary_parameters():
    """Resource accounting must not call learned lateral matrices temporal state."""
    width = 5
    state = TeReLState(width, statistics_momentum=0.9, lateral_momentum=0.99)
    state.ensure_residual_lateral()

    assert state.previous_centered is None
    assert state.causal_dynamic_state_numel() == 3 * width + 1
    assert state.auxiliary_parameter_numel() == 2 * width * width


def test_shifted_proxy_allocates_centered_predecessor_only_when_requested():
    """The accepted zero-offset path must not pay for a shifted-proxy-only vector."""
    state = TeReLState(3, statistics_momentum=0.9, lateral_momentum=0.99)

    centered = state.ensure_previous_centered()
    assert centered.shape == (3,)
    state.update(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(state.previous_centered, torch.tensor([[1.0, 2.0, 3.0]])[0])
    assert state.causal_dynamic_state_numel() == 4 * 3 + 1


def test_regularized_target_residual_reproduces_detached_activation_gradient():
    """Dropping a term, mask, or reduction must break target-gradient equivalence."""
    target_builder = getattr(objective, "regularized_target_residual", None)
    assert callable(target_builder), "regularized target construction is not implemented"

    z = torch.tensor(
        [[2.0, -1.0], [1.0, 3.0], [4.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    previous = torch.tensor([[1.0, -2.0], [9.0, 9.0], [2.0, 1.0]], dtype=torch.float64)
    mean = torch.tensor([1.0, 0.0], dtype=torch.float64)
    variance = torch.tensor([0.25, 1.5], dtype=torch.float64)
    lateral = torch.tensor([[0.0, 0.5], [0.25, 0.0]], dtype=torch.float64)
    lateral_reference = torch.tensor(
        [[2.0, -1.0], [1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64
    )
    pair_valid = torch.tensor([True, False, True])
    coefficients = objective.LossCoefficients(
        similarity=2.0,
        variance=1.0,
        covariance=4.0,
    )

    target, residual = target_builder(
        z=z,
        previous=previous,
        mean=mean,
        variance=variance,
        lateral=lateral,
        pair_valid=pair_valid,
        coefficients=coefficients,
        variance_target=1.0,
        lateral_reference=lateral_reference,
    )

    expected_residual = torch.tensor(
        [[0.625, 2.0], [1.0, 0.25], [3.375, -1.75]], dtype=torch.float64
    )
    expected_target = torch.tensor(
        [[1.375, -3.0], [0.0, 2.75], [0.625, 1.75]], dtype=torch.float64
    )
    assert target.requires_grad is False
    assert residual.requires_grad is False
    assert torch.allclose(residual, expected_residual)
    assert torch.allclose(target, expected_target)
    assert torch.allclose(z.detach() - target, residual)

    original, _ = objective.detached_terel_loss(
        z=z,
        previous=previous,
        mean=mean,
        variance=variance,
        lateral=lateral,
        pair_valid=pair_valid,
        coefficients=coefficients,
        variance_target=1.0,
        lateral_reference=lateral_reference,
    )
    original_gradient = torch.autograd.grad(original, z)[0]

    surrogate = (
        coefficients.similarity
        / (z.shape[0] * z.shape[1])
        * (z - target).square().sum()
    )
    surrogate_gradient = torch.autograd.grad(surrogate, z)[0]

    assert torch.allclose(original_gradient, surrogate_gradient, atol=1e-12, rtol=1e-12)


def test_preactivation_residual_state_gives_the_exact_weight_outer_product():
    """Omitting the local activation derivative must break the synaptic factorization."""
    target_builder = getattr(objective, "regularized_target_residual", None)
    assert callable(target_builder), "regularized target construction is not implemented"

    inputs = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor(
        [[1.0, 0.5], [-1.0, 0.25]], dtype=torch.float64, requires_grad=True
    )
    preactivation = inputs @ weights.T
    activation = torch.nn.functional.leaky_relu(preactivation, negative_slope=0.1)
    previous = torch.tensor([[1.5, -0.075]], dtype=torch.float64)
    coefficients = objective.LossCoefficients(
        similarity=1.0,
        variance=0.0,
        covariance=0.0,
    )

    loss, _ = objective.detached_terel_loss(
        z=activation,
        previous=previous,
        mean=torch.zeros(2, dtype=torch.float64),
        variance=torch.ones(2, dtype=torch.float64),
        lateral=torch.zeros(2, 2, dtype=torch.float64),
        pair_valid=torch.tensor([True]),
        coefficients=coefficients,
        variance_target=1.0,
    )
    actual_weight_gradient = torch.autograd.grad(loss, weights)[0]

    _, activation_residual = target_builder(
        z=activation,
        previous=previous,
        mean=torch.zeros(2, dtype=torch.float64),
        variance=torch.ones(2, dtype=torch.float64),
        lateral=torch.zeros(2, 2, dtype=torch.float64),
        pair_valid=torch.tensor([True]),
        coefficients=coefficients,
        variance_target=1.0,
    )
    activation_slope = torch.tensor([[1.0, 0.1]], dtype=torch.float64)
    neuron_state = activation_residual * activation_slope
    expected_neuron_state = torch.tensor([[1.0, -0.01]], dtype=torch.float64)
    expected_weight_gradient = torch.tensor(
        [[2.0, 1.0], [-0.02, -0.01]], dtype=torch.float64
    )

    assert torch.allclose(neuron_state, expected_neuron_state, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        neuron_state.T @ inputs,
        expected_weight_gradient,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        actual_weight_gradient,
        expected_weight_gradient,
        atol=1e-12,
        rtol=1e-12,
    )


def test_residual_lateral_equilibrium_inhibits_a_correlated_state():
    """Using the wrong lateral sign must amplify rather than suppress coactivity."""
    equilibrium = getattr(objective, "residual_lateral_equilibrium", None)
    assert callable(equilibrium), "residual-state lateral equilibrium is not implemented"

    base_state = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    inhibitory = torch.ones(2, 2, dtype=torch.float64)

    state = equilibrium(
        base_state=base_state,
        lateral=inhibitory,
        coefficient=1.0,
    )

    assert torch.allclose(
        state,
        torch.tensor([[1.0 / 3.0, 1.0 / 3.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert abs(float(state[0, 0] * state[0, 1])) < 1.0


def test_local_residual_dynamics_converges_to_the_equilibrium_reference():
    """A local dynamics implementation must not silently solve a different system."""
    equilibrium = getattr(objective, "residual_lateral_equilibrium", None)
    dynamics = getattr(objective, "residual_lateral_dynamics", None)
    assert callable(equilibrium), "residual-state lateral equilibrium is not implemented"
    assert callable(dynamics), "local residual-state dynamics is not implemented"

    base_state = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    inhibitory = torch.ones(2, 2, dtype=torch.float64)
    expected = equilibrium(
        base_state=base_state,
        lateral=inhibitory,
        coefficient=1.0,
    )

    actual = dynamics(
        base_state=base_state,
        lateral=inhibitory,
        coefficient=1.0,
        steps=80,
        step_size=0.2,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_zero_residual_lateral_steps_is_the_uninhibited_boundary():
    base = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    lateral = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    actual = objective.residual_lateral_dynamics(
        base_state=base,
        lateral=lateral,
        coefficient=1000.0,
        steps=0,
        step_size=0.1,
    )

    assert torch.equal(actual, base)


def test_residual_lateral_moment_uses_the_same_detached_neuron_states():
    """Updating from activations instead of residual states must change this moment."""
    moment_builder = getattr(objective, "residual_lateral_moment", None)
    assert callable(moment_builder), "residual-state lateral moment is not implemented"

    neuron_states = torch.tensor(
        [[2.0, -1.0], [-2.0, 1.0]], dtype=torch.float64, requires_grad=True
    )

    moment = moment_builder(neuron_states)

    assert moment.requires_grad is False
    assert torch.allclose(
        moment,
        torch.tensor([[4.0, -2.0], [-2.0, 1.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
