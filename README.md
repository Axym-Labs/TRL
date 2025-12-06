
# Temporal Regularized Learning

Temporal Regularized Learning (TRL) is a highly local and self-supervised prodecure that optimizes
each neuron individually. We adapt the self-supervised loss formulation of VICReg, consisting
of variance, invariance and covariance to input streams with sequential coherence and for online-
compatibility. It removes the need for biphasic updates, negatives or inner-loop convergence, given
three scalar memory units per neuron and an auxiliary lateral network. Knowledge about downstream
tasks can be injected through the sequence ordering, allowing for supervised training. We present
TRL and its simplified variant, TRL-S. Experiments on MNIST show TRL is competetive with
backpropagation, Forward-Forward and Equilibrium Propagation, while TRL-S achieves similar
performance despite its simplified setup. We show TRL creates neurons with specialized receptive
fields at the first layer. In later layers, some neurons specialize by activating only for some types of
input.

- Repository of the paper "**Temporal Regularized Learning: Self-Supervised learning local in space and time**"
- Train a TRL model with `train.py`.
- `trl/config/configurations.py` contains functions that modify the setup, e.g. modify the model architecture.
- Link to the paper: https://axym.org/files/TRL.pdf
