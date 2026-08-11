# Temporal Regularized Learning (TeReL)

TeReL learns deep slow features one observation at a time. At each layer, a
soft Slow Feature Analysis objective defines a detached activation target. Its
exact ReLU preactivation residual becomes a signed postsynaptic neuron state.
The feedforward gradient then factors into this state and the presynaptic
activity; one same-layer matrix supplies an explicit correction and receives
an anti-Hebbian state--state contribution. No error signal crosses a learned
layer or persists through time.

On a class-chunked MNIST stream, the reported `784 → 512 → 256` encoder reaches
**96.39 ± 0.19%** held-out linear-probe accuracy with plain SGD, batch size
one, and two presentations of the encoder-training observations. Random
features reach 95.14 ± 0.27%, Layer-local SupCon reaches 96.95 ± 0.13%, and
the spatially and temporally relaxed TeReL-Offline reference reaches
98.39 ± 0.04%. Continued unlabeled updates with the validation-selected step
reach 96.43 ± 0.15% using the same fitted probe.

The experiment uses class labels only to construct temporal adjacency; labels
do not enter TeReL's target or synaptic updates. The paper therefore treats
this order as a controlled temporal relation, not as evidence for arbitrary
natural streams.

## Install and verify

Python 3.12 and direct dependencies are pinned in `pyproject.toml` and
`uv.lock`.

```bash
uv sync --extra test
uv run pytest -q
```

The tests cover target--gradient equivalence, the neuron-local outer product,
the anti-Hebbian sign, the single explicit correction, detachment across
layers and observations, one-matrix resource accounting, evaluation roles,
and manifest integrity.

## Method and resource boundary

For each neuron, TeReL retains the preceding activation `p`, running mean `μ`,
and running variance `v`. A layer of width `D` therefore has `3D` causal
scalars and one sequence-validity bit. It also has one `D × D` matrix of
auxiliary lateral parameters, distinct from both causal state and feedforward
weights. Plain SGD adds no optimizer state. The same-layer correction is one
matrix--vector operation, not iterative settling.

For the reported encoder, this gives 2,306 causal scalars, 327,680 auxiliary
lateral parameters, and 533,248 feedforward parameters. Dense same-layer
communication remains a real cost; locality here describes credit assignment,
not neuron independence or biological realism.

## Reproduce the reported analysis

The complete comparison is declared in
[`configs/strengthening2-final-matrix.yaml`](configs/strengthening2-final-matrix.yaml),
and continued online evaluation is declared in
[`configs/strengthening2-online-continuation.yaml`](configs/strengthening2-online-continuation.yaml).
Held-out execution is guarded by a clean-repository manifest and an explicit
authorization flag. Given the archived records, regenerate all reported
summaries with:

```bash
uv run python analysis/strengthening2_final_results.py \
  --results artifacts/final-results \
  --failure-ledger artifacts/final-results/failures.json \
  --online-continuation-results \
    artifacts/online-continuation-final-results/mnist/terel-online-scaled \
  --output artifacts/strengthening2-final-analysis-mnist.json
```

Execution commits, manifest hashes, and record digests are kept in
[`ARTIFACT_README.md`](ARTIFACT_README.md), rather than in the manuscript.

## Repository layout

```text
terel/resubmission/   method, baselines, evaluation, and protocol gates
analysis/             final analysis and figure-data utilities
configs/              selected protocols and supporting controls
tests/                mathematical and implementation-fidelity tests
```

The manuscript is available from the DOI listed on its first page. Please cite
that version when using the method or results.
