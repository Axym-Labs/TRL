# Temporal Regularized Learning (TeReL)

TeReL learns deep slow features one observation at a time. A soft Slow Feature
Analysis objective defines a detached regularized target. Each neuron tracks
its preactivation residual to that target; presynaptic activity and the
postsynaptic neuron state give the exact feedforward gradient before lateral
inhibition. One pass through a learned inhibitory operator modifies this state,
and pairs of the same states drive an anti-Hebbian lateral update. No error
crosses a learned layer or persists through time.

On label-ordered MNIST, TeReL reaches 95.84 ± 0.07% accuracy after two data
presentations using plain SGD, batch size one, and one lateral matrix-vector
pass. The pass improves an otherwise identical no-inhibition reference by 1.42
points on validation (95% Student-t interval [1.32, 1.52]). TeReL-Offline, a
less constrained minibatch reference with short temporal graphs and BatchNorm,
reaches 97.30 ± 0.07%. The paper treats label-derived adjacency as controlled
temporal supervision, not as unlabeled self-supervision.

Exact execution identifiers and checksums belong in
[`ARTIFACT_README.md`](ARTIFACT_README.md).

## Quick start

Python 3.12 and all direct dependencies are pinned in `pyproject.toml` and
`uv.lock`.

```bash
uv sync --extra test
uv run pytest -q
```

The tests cover target-gradient equivalence, the preactivation outer product,
the inhibitory sign and one-pass dynamics, same-state lateral learning,
cross-layer and temporal detachment, resource accounting, data roles,
baselines, and frozen-protocol integrity.

## Canonical samplewise protocol

Set `data_root` in `configs/canonical-online-learning.yaml`, then run a declared
validation candidate with the recovery entrypoint. The selected configuration
is `terel-online`; `configs/canonical-online-mechanism.yaml` defines its matched
no-inhibition control.

```bash
uv run python -m terel.resubmission.recovery \
  --config configs/canonical-online-learning.yaml \
  --candidate terel-online \
  --seed 101 \
  --output artifacts/canonical-online-validation \
  --device cuda
```

The selection ledger records the fixed validation comparison and resource
accounting. A confirmatory manifest can be frozen only from a clean repository:

```bash
uv run python -m terel.resubmission.residual_confirmatory freeze \
  --selection-plan configs/canonical-online-learning.yaml \
  --validation-ledger configs/canonical-online-validation-ledger.json \
  --protocol docs/canonical-online-protocol.md \
  --repository . \
  --output artifacts/canonical-online-confirmatory-manifest.json
```

Combine final records with the matched validation mechanism records using:

```bash
uv run python -m terel.resubmission.canonical_online_analysis \
  --final artifacts/canonical-online-confirmatory-results/mnist/terel-online \
  --inhibited-validation artifacts/canonical-online-validation-results/inhibition \
  --reference-validation artifacts/canonical-online-validation-results/no-inhibition \
  --output artifacts/canonical-online-analysis.json
```

## Resource boundary

At width `D`, the canonical path stores three causal scalars per neuron: the
preceding activation, running mean, and running variance. One validity flag is
stored per layer. The representation and residual-state operators are two
`D × D` auxiliary parameter matrices; they are weights, not temporal state.
For the 784→512→256 encoder this is 2,306 causal scalars, 655,360 auxiliary
parameters, and 533,248 feedforward parameters. Plain SGD adds no optimizer
state.

Locality permits dense same-layer communication. It does not imply neuron
independence, biological realism, low energy, or efficient dense hardware.
TeReL-Offline is a performance reference rather than a matched locality
control because it also changes batching, normalization, schedule, and
optimization.

## Supporting evidence

The repository retains protocols for TeReL-Offline, Local SupCon,
normalization-matched random features, direct covariance, objective ablations,
classical SFA, and incremental SFA. Their stable identifiers and
analysis commands are documented in `ARTIFACT_README.md`; the manuscript uses
scientific names rather than internal run labels.

## Repository layout

```text
terel/resubmission/                 method, baselines, analyses, and protocol gates
configs/                            canonical and supporting configurations
docs/                               scientific protocol records
tests/                              mathematical and implementation-fidelity tests
```
