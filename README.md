# Temporal Regularized Learning (TeReL)

TeReL trains each layer of a nonlinear encoder from temporal coherence,
anti-collapse variance expansion, and a same-layer decorrelation signal. The
corrected formulation is a deep local, soft-constraint form of Slow Feature
Analysis: gradients do not cross layers or preceding steps, while detached
running statistics and a dense lateral operator provide bounded-history state.

This repository contains the leakage-controlled implementation and experiment
pipeline for the revised paper. The revision explicitly treats label-ordered
MNIST as label-assisted and tests natural, label-free encoder pretraining on
chronological PAMAP2 streams.

> **Correction notice.** Results from the originally cited TeReL experiment
> commits are not evidence for the revised claims. An optimizer-registration
> defect left later encoder layers unstepped, one legacy BP dispatcher selected
> the wrong model, and the original split reused held-out data during model
> development. The corrected pipeline retires all affected numbers and refuses
> held-out evaluation until a validation ledger, protocol hash, clean code
> commit, and explicit test flag agree.

## Quick start

Python 3.12 and all direct dependencies are pinned in `pyproject.toml` and
`uv.lock`. Each run also enables deterministic PyTorch algorithms and records
the Python, package, CUDA, cuDNN, CPU, and GPU environment in its frozen
manifest.

```bash
uv sync --extra test
uv run pytest -q
```

The fidelity suite covers the objective signs and detachments, every-layer
updates, fixed state size, split isolation, boundary-aware SFA controls, the
IncSFA port, BP construction, and the held-out gate.

## Reproduce the corrected experiments

1. Download MNIST under a local data root and place the extracted PAMAP2
   archive under another. Update only the two `data_root` values in
   `configs/resubmission/selection-plan.yaml` for your machine.
2. Run the bounded validation sweep. It uses seeds 101, 202, and 303 and never
   evaluates the official MNIST test split or PAMAP2 subject 8.

```bash
uv run python -m terel.resubmission.selection \
  --plan configs/resubmission/selection-plan.yaml \
  --output artifacts/selection-results \
  --repository . \
  --protocol /path/to/confirmatory-protocol.md \
  --device cuda
```

3. After the selection ledger is complete and the worktree is committed and
   clean, freeze the exact five-seed test matrix.

```bash
uv run python -m terel.resubmission.confirmatory freeze \
  --selection-plan configs/resubmission/selection-plan.yaml \
  --validation-ledger artifacts/selection-results/validation-ledger.json \
  --protocol /path/to/confirmatory-protocol.md \
  --repository . \
  --output artifacts/confirmatory-manifest.json
```

4. Open the held-out gate explicitly. Runs are written per seed and resume
   safely when the same manifest is used again.

```bash
uv run python -m terel.resubmission.confirmatory run \
  --manifest artifacts/confirmatory-manifest.json \
  --validation-ledger artifacts/selection-results/validation-ledger.json \
  --protocol /path/to/confirmatory-protocol.md \
  --repository . \
  --output artifacts/confirmatory-results \
  --device cuda \
  --allow-test
```

The frozen matrix compares corrected TeReL with the identical random encoder,
local supervised contrastive learning, supervised BP, and direct covariance on
MNIST. On PAMAP2 it additionally pairs chronological and shuffled TeReL and
includes batch SFA and IncSFA.

### Frozen revised result

The executed matrix used code commit
`abfa0ec1df78204b66d0c34141f5deb6063a572a`, manifest SHA-256
`38ca928e0884a8bc40a5c7ffefc46313fc96c170661aea63f93bf9496b2b1af0`,
and configuration SHA-256
`c2520fa7e5751ca879da6683e6c193bfa5a800f59ec685e82199d892bbce8c42`.
All 60 planned runs completed before analysis.

On MNIST, TeReL reaches 0.886 mean accuracy versus 0.890 for the matched
random encoder; the paired difference is -0.0042 with 95% bootstrap interval
[-0.0092, -0.0003]. Its nearest-centroid accuracy nevertheless rises from
0.760 to 0.852, showing a repeatable change in class geometry. Direct
in-batch covariance reaches 0.916 and exposes the cost of the tracked lateral
proxy. On subject-disjoint PAMAP2, chronological TeReL reaches 0.307 macro-F1
versus 0.312 after shuffling; the paired interval [-0.0625, 0.0530] does not
support a natural-order advantage. These held-out results narrowed the claims
and did not trigger further tuning.

## Analyze results and audit locality

```bash
uv run python -m terel.resubmission.analysis \
  --results artifacts/confirmatory-results \
  --analysis-output artifacts/confirmatory-analysis.json \
  --results-tex artifacts/generated_results.tex \
  --appendix-tex artifacts/generated_appendix_results.tex

uv run python -m terel.resubmission.locality \
  --manifest artifacts/confirmatory-manifest.json \
  --repository . \
  --output artifacts/locality-audit.json \
  --device cuda
```

The analysis preserves every raw seed and reports the mean, sample standard
deviation, 10,000-resample percentile interval, and paired primary effects.
The locality audit compares batch-size-one detached execution with detached
and undetached minibatches, including throughput, peak memory, parameter,
optimizer, and dynamic-state bytes.

## Method and claim boundary

- TeReL's temporal, variance, and decorrelation structure is inherited at the
  objective level from SFA and related regularized self-supervised methods.
- The algorithmic contribution is the particular deep local parameterization,
  detached population/temporal state, and tracked same-layer lateral proxy.
- Local credit assignment does not imply sparse communication: a width-`D`
  layer stores `D² + 4D + 1` dynamic state elements.
- Label-derived MNIST ordering is supervision through the data order. The
  self-supervised temporal-order contrast is the label-free PAMAP2 encoder run.
- Hardware relevance is a motivation; this repository does not claim measured
  energy efficiency or biological plausibility.

## Repository layout

```text
terel/resubmission/                 corrected objectives, models, baselines, and gates
configs/resubmission/               frozen validation plan
tests/test_resubmission_*.py        scientific and implementation-fidelity tests
terel/ and previous_versions/       legacy exploratory code retained for provenance
```

The historical Zenodo record is available at
[doi:10.5281/zenodo.18673107](https://doi.org/10.5281/zenodo.18673107). Cite the
revised paper once its new archival record is available; the historical record
should not be used as the source of corrected numerical results.
