
# Temporal Regularized Learning: Self-Supervised Learning Local In Space And Time

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17840254.svg)](https://doi.org/10.5281/zenodo.17840254)
[![ResearchGate](https://img.shields.io/badge/Read%20on-ResearchGate-00cc66.svg)](https://www.researchgate.net/publication/398409613_Temporal_Regularized_Learning_Self-supervised_learning_local_in_space_and_time)


**[Paper PDF](https://axym.org/files/TRL.pdf)**

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

Cite the paper:

```
@misc{Wiest2025,
  author       = {Wiest, Davide},
  title        = {{Temporal Regularized Learning: Self-supervised learning local in space and time}},
  publisher    = {Zenodo},
  year         = {2025},
  doi          = {10.5281/zenodo.17840254},
  url          = {https://doi.org/10.5281/zenodo.17840254}
}
```


### Quickstart
- Train a TRL model with `train.py`.
- `trl/config/configurations.py` contains functions that modify the setup, e.g. modify the model architecture.
- The `previous_versions` folder has a README with short explanations of what changed in each version.
