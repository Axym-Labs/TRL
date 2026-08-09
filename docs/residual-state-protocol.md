# Residual-state TeReL-S protocol

## Claim

Residual-state TeReL-S learns a two-hidden-layer representation from a
label-constructed MNIST stream while retaining detached, fixed-size temporal
state and no error path between learned layers. Its feedforward gradient uses
the neuron's preactivation residual state; inhibitory lateral plasticity uses
pairs of those same states.

## Data and roles

- Stratify the 60,000-example MNIST training split into 50,000 fitting and
  10,000 validation examples with split seed 1701.
- Use the official held-out split only for the frozen final evaluation.
- Construct fitting order from randomly permuted same-class chunks of length
  16. The labels determine order and do not enter the encoder objective.
- Fit and compare exactly two validation configurations: residual-state
  TeReL-S and its otherwise identical zero-coupling reference.
- Freeze the selected configuration before final evaluation. Permit exactly
  one method and final seeds 42, 43, 44, 45, and 46.

## Selected encoder

- Dimensions 784→512→256; LeakyReLU slope 0.01; identity normalization.
- Joint local training, batch size 1, two fitting-set presentations.
- AdamW, learning rate 6e-6, zero weight decay.
- Soft-SFA coefficients (similarity, variance, covariance) = (1, 2.5, 1),
  variance target 1, and state momenta 0.99.
- Residual-state coupling 1000; four Euler settling steps of size 0.1.
- Full residual second moment, normalized by feature count, including the
  diagonal.

## Probe and reporting

- Concatenate both hidden layers and fit the same linear probe for every run:
  AdamW for 60 epochs, batch size 2048, learning rate 0.003, weight decay 1e-4.
- Report raw accuracy, mean, sample standard deviation, 95% Student-t interval,
  effective rank, feature variance, state RMS before and after inhibition,
  parameter/state sizes, optimizer updates, and CPU time.
- The final result cannot select a replacement configuration. Direct-covariance,
  objective-component, Local SupCon, normalization, and TeReL-batched evidence
  retain their separate frozen protocols.
