# Temporal Regularized Learning (TeReL)

TeReL trains each layer of a nonlinear encoder from temporal coherence,
anti-collapse variance expansion, and a same-layer decorrelation signal. The
corrected formulation is a deep local, soft-constraint form of Slow Feature
Analysis. Canonical TeReL is greedy and layer-local, retaining short gradients
between adjacent examples inside a chunk. TeReL-S also detaches time; running
statistics and a dense lateral operator then provide bounded-history state.

This repository contains the validated implementation and frozen experiment
pipeline for the revised paper. Label-ordered MNIST is explicitly treated as
controlled temporal supervision; chronological PAMAP2 is a secondary
natural-order stress test.

> **Correction and recovery notice.** An optimizer-registration defect left
> later encoder layers unstepped in one historical path, and the first revision
> protocol simultaneously changed the intended schedule, temporal gradient,
> normalization, lateral timescale, and readout. The v2 recovery reinstates the
> intended protocol, verifies every layer update, and freezes results behind a
> hashed manifest. The official MNIST test set was historically observed, so
> v2 is reported as a sequential confirmation rather than a pristine first use.

## Quick start

Python 3.12 and all direct dependencies are pinned in `pyproject.toml` and
`uv.lock`. Each run also enables deterministic PyTorch algorithms and records
the Python, package, CUDA, cuDNN, CPU, and GPU environment in its frozen
manifest.

```bash
uv sync --extra test
uv run pytest -q
```

The fidelity suite covers objective signs and detachments, every-layer updates,
matched readouts, greedy scheduling, fixed streaming state, samplewise graph
release, split isolation, BP construction, and the frozen test gate.

## Reproduce the corrected experiments

1. Download MNIST and update `data_root` in the v2 plans for your machine.
2. Run candidates from the bounded validation-only recovery plans with seeds
   101, 202, and 303. For example:

```bash
uv run python -m terel.resubmission.recovery \
  --config configs/resubmission/recovery-v2.yaml \
  --candidate canonical-recovered-bn \
  --seed 101 \
  --output artifacts/recovery-v2 \
  --device cuda
```

The complete candidate registry is in `recovery-v2.yaml`; the samplewise study
uses `streaming-recovery-v2.yaml`. The frozen choices and all validation
records are summarized in `validation-ledger-v2.json`. The post-confirmation
one-factor controls use `mechanism-audit-v2.yaml` with the same recovery runner.

3. From a clean commit, freeze the exact matrix using the v2 protocol and
validation ledger. Then run it with the explicit test flag:

```bash
uv run python -m terel.resubmission.confirmatory_v2 \
  --matrix configs/resubmission/confirmatory-matrix-v2.yaml \
  --validation-ledger configs/resubmission/validation-ledger-v2.json \
  --protocol /path/to/confirmatory-protocol-v2.md \
  --repository . \
  --output artifacts/confirmatory-manifest-v2.json
uv run python -m terel.resubmission.confirmatory run \
  --manifest artifacts/confirmatory-manifest-v2.json \
  --validation-ledger configs/resubmission/validation-ledger-v2.json \
  --protocol /path/to/confirmatory-protocol-v2.md \
  --repository . \
  --output artifacts/confirmatory-results-v2 \
  --device cuda \
  --allow-test
```

Runs are written atomically per seed and resume safely with the same manifest.
The frozen matrix compares canonical TeReL, a final-layer readout, TeReL-S,
an unnormalized random encoder, and data-presentation-matched BP. A later
one-row manifest adds the normalization-matched random control described below.

### Frozen revised result

The executed matrix used code commit
`02afd90cf6927a588aa424d61cb86c6876b25c17` and configuration SHA-256
`3e83ea66c03a621bbdc6f1a16a143c6dbf50643ae2b5213a1d78afba68ed0a6b`.
All 25 planned runs completed before analysis.

Canonical TeReL reaches 97.30 ± 0.07% accuracy versus 98.34 ± 0.08% for
data-presentation-matched BP and 95.13 ± 0.27% for the original
no-normalization random encoder. The paired gaps are -1.04 points to BP and
+2.17 points to that random reference. The last layer retains 97.14%,
and TeReL-S reaches 96.29%. The tuned batch-size-one TeReL-S configuration
reaches 95.14 ± 0.04% validation accuracy with effective rank 112.8, fixed
state, and no retained temporal graph.

A separately frozen validation-only audit changes one factor at a time under
the recovered canonical protocol. Removing temporal coherence loses 8.18
points; shuffling away class persistence loses 7.55. Removing variance
expansion collapses median variance to `7.7e-5`, while removing decorrelation
leaves variance high but collapses effective rank to 2.8. These controls are
mechanism evidence, not post-test model selection.

The bounded review patch adds the missing matched label-aware comparator without
changing the v2 matrix. A six-candidate validation grid selected Local SupCon
at learning rate `1e-3` and temperature `0.1`; its frozen five-seed test mean is
96.98 ± 0.10%, versus 97.30 ± 0.07% for canonical TeReL (paired difference
0.32 points, 95% Student-t interval [0.13, 0.52]). A separate validation audit
measures about 0.76 cosine alignment between TeReL's lagged and same-batch
lateral directions; the exact direct-covariance control reaches 96.93%.

The latest bounded review patch adds one normalization-matched random row
without rerunning the v2 matrix. Its hidden weights, biases, and BatchNorm
affine parameters remain at their seed-specific initialization. One
no-gradient pass over the 50,000-example training subset calibrates only the
BatchNorm running statistics before the unchanged all-layer probe. Across the
same five test seeds, this control reaches 95.35 ± 0.19%; canonical TeReL is
1.95 points higher with a paired 95% Student-t interval [1.64, 2.25]. The
calibration treatment, five seeds, and stopping rule are frozen in
`latest-review-confirmatory-matrix-v4.yaml` and
`latest-review-validation-ledger-v4.json`.

## Analyze results and audit locality

```bash
uv run python -m terel.resubmission.analysis_v2 \
  --results artifacts/confirmatory-results-v2 \
  --streaming-results artifacts/streaming-recovery-v2 \
  --analysis-output artifacts/confirmatory-analysis-v2.json \
  --results-tex artifacts/generated_results_v2.tex \
  --appendix-tex artifacts/generated_appendix_results_v2.tex

uv run python -m terel.resubmission.mechanism_analysis_v2 \
  --reference-results artifacts/recovery-v2 \
  --audit-results artifacts/mechanism-audit-results-v2 \
  --analysis-output artifacts/mechanism-audit-analysis-v2-regenerated.json \
  --results-tex artifacts/generated-mechanism-results-v2.tex

uv run python -m terel.resubmission.latest_review_analysis \
  --control-results artifacts/latest-review-confirmatory-results-v4 \
  --reference-results artifacts/confirmatory-results-v2 \
  --analysis-output artifacts/latest-review-analysis-v4.json \
  --results-tex artifacts/generated-latest-review-v4.tex
```

The analysis preserves every raw seed and reports the mean, sample standard
deviation, 10,000-resample percentile interval, and paired primary effects.
It also records representation geometry, parameters, optimizer and dynamic
state, peak memory, encoder examples, optimizer steps, wall time, and a
declared operation proxy.
The original frozen mechanism JSON is retained for execution provenance; the
`-regenerated` JSON is the byte-exact output of the reporting source supplied
in the anonymous archive and produces the same numerical mechanism table.

## Method and claim boundary

- TeReL's temporal, variance, and decorrelation structure is inherited at the
  objective level from SFA and related regularized self-supervised methods.
- The algorithmic contribution is the particular deep local parameterization,
  detached population state, canonical within-chunk temporal signal, and
  tracked same-layer lateral proxy. TeReL-S additionally detaches time.
- Local credit assignment does not imply sparse communication: a width-`D`
  layer stores `D² + 4D + 1` dynamic state elements.
- Label-derived MNIST ordering is supervision through the data order. PAMAP2 is
  a secondary natural-order stress test whose activity classes are withheld
  from the encoder loss, although annotations determine transition removal and
  boundaries; it is not the paper's headline.
- Samplewise TeReL-S demonstrates fixed history and no temporal graph. Hardware
  relevance remains a perspective; no energy-efficiency result is claimed.

## Repository layout

```text
terel/resubmission/                 corrected objectives, models, baselines, and gates
configs/resubmission/               frozen validation plan
tests/test_resubmission_*.py        scientific and implementation-fidelity tests
terel/ and previous_versions/       legacy exploratory code retained for provenance
```

For anonymous review, use the portable supplement generated by
`terel.resubmission.package_supplement`; it contains the code, frozen plans,
raw records, analyses, and a redaction/checksum manifest without author links.

```bash
uv run python -m terel.resubmission.package_supplement \
  --repository . \
  --paper-repository /path/to/anonymous-paper-source \
  --artifact-root /path/to/frozen-artifacts \
  --output TeReL-anonymous-supplement.zip
```
