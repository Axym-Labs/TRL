# Canonical samplewise TeReL protocol

## Method

The encoder has dimensions 784→512→256 with LeakyReLU activations. It trains
both layers jointly, processes one observation per update, and detaches the
activation passed between learned layers. Each layer keeps the preceding
activation, running mean, running variance, representation second moment, and
residual-state second moment. A sequence-boundary flag disables the temporal
term at every chunk start.

Training uses plain SGD with learning rate 0.01, no momentum, no weight decay,
and no optimizer state. The soft-SFA coefficients are `(1, 2.5, 1)`, the
variance target is 1, and all running-statistic momenta are 0.99. Residual-state
inhibition uses coefficient 1000 and one lateral pass of size 0.1. The matched
mechanism control omits that pass, giving the exact detached objective gradient.

## MNIST protocol

The 60,000-example training partition is stratified into 50,000 fitting and
10,000 validation examples. At each of two encoder presentations, fitting
examples are randomly arranged into same-class chunks of length 16. Labels
determine adjacency but do not enter the encoder objective. A common linear
probe reads the concatenation of both hidden layers.

Selection uses only the validation partition and the three declared seeds in
`configs/canonical-online-validation-ledger.json`. The selected configuration
is frozen before final evaluation. Final records use the five predetermined
seeds in `configs/canonical-online-learning.yaml`; no final result selects
among alternative configurations.

## Resource accounting

For a layer of width `D`, causal state comprises `3D` scalars and one layer
flag. The two `D × D` learned lateral operators are auxiliary parameters, not
causal state. Optimizer state, temporary activations, and inference parameters
are reported separately.
