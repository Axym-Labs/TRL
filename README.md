# Temporal Regularized Learning (TeReL)

TeReL-S learns deep slow features with credit assignment local in space and
time. A soft Slow Feature Analysis objective defines a detached regularized
target. Each neuron tracks its preactivation residual to that target; the
uncoupled state's outer product with presynaptic activity is the exact
feedforward gradient. Learned inhibition settles the state used for the
feedforward update and learns lateral connections from pairs of the same
states. No error crosses a learned layer or persists through time.

The samplewise method is the center of the paper. TeReL-batched is a less
restrictive performance reference that keeps short temporal graphs and uses
BatchNorm. Label-ordered MNIST is treated as controlled temporal supervision,
not unlabeled self-supervision. Chronological PAMAP2 is a secondary stress test
whose result is inconclusive.

On final MNIST evaluation, residual-state TeReL-S reaches 95.50 ± 0.19%
accuracy after two data presentations. On the train-derived validation split,
it improves an otherwise identical zero-coupling samplewise reference by 3.01
points, with 95% Student-t interval [2.71, 3.31], while preserving effective
rank. TeReL-batched reaches 97.30 ± 0.07%; its matched backpropagation and Local
SupCon references reach 98.34 ± 0.08% and 96.98 ± 0.10%.

Exact execution identifiers and checksums are in
[`ARTIFACT_README.md`](ARTIFACT_README.md).

## Quick start

Python 3.12 and all direct dependencies are pinned in `pyproject.toml` and
`uv.lock`.

```bash
uv sync --extra test
uv run pytest -q
```

The test suite checks target-gradient equivalence, the preactivation outer
product, inhibitory sign and dynamics, use of the same neuron states at lateral
synapses, cross-layer and temporal detachment, dynamic-state size, data roles,
baselines, and frozen-protocol integrity.

## Reproduce residual-state TeReL-S

Update `data_root` in
`configs/resubmission/residual-state-validation.yaml` for your MNIST location.
The validation plan contains exactly the selected method and its matched
zero-coupling reference.

```bash
uv run python -m terel.resubmission.recovery \
  --config configs/resubmission/residual-state-validation.yaml \
  --candidate terel-s-residual \
  --seed 101 \
  --output artifacts/residual-state-validation \
  --device cpu
```

The final manifest is built only from a clean source tree, the complete
validation ledger, and the protocol document.

```bash
uv run python -m terel.resubmission.residual_confirmatory freeze \
  --selection-plan configs/resubmission/residual-state-validation.yaml \
  --validation-ledger configs/resubmission/residual-state-validation-ledger.json \
  --protocol docs/residual-state-protocol.md \
  --repository . \
  --output artifacts/residual-state-confirmatory-manifest.json

uv run python -m terel.resubmission.residual_confirmatory run \
  --manifest artifacts/residual-state-confirmatory-manifest.json \
  --validation-ledger configs/resubmission/residual-state-validation-ledger.json \
  --protocol docs/residual-state-protocol.md \
  --repository . \
  --output artifacts/residual-state-confirmatory \
  --device cpu \
  --allow-test
```

Runs are written atomically and resume against the same manifest. Analyze the
completed records and generate the manuscript tables with:

```bash
uv run python -m terel.resubmission.residual_state_analysis \
  --results artifacts/residual-state-confirmatory \
  --validation-ledger configs/resubmission/residual-state-validation-ledger.json \
  --analysis-output artifacts/residual-state-analysis.json \
  --results-tex artifacts/generated-residual-results.tex \
  --appendix-tex artifacts/generated-residual-appendix.tex
```

## Method boundary

- TeReL-S uses one observation per forward/backward call and retains
  `2D² + 4D + 1` detached state scalars at width `D`. Two dense lateral matrices
  remain an explicit training cost.
- The representation matrix constructs the decorrelation part of the target.
  A separate residual-state matrix acts through inhibitory settling dynamics.
- The feedforward statistic has Hebbian two-factor form at the gradient level;
  the descent sign remains explicit, and the experiments pass the gradient to
  AdamW.
- Locality permits dense same-layer communication. It does not imply neuron
  independence, biological realism, low energy, or efficient present-day
  hardware.
- The TeReL-S/TeReL-batched accuracy difference is descriptive because their
  batch size, normalization, schedule, and optimizer granularity differ.

## Supporting evidence

The repository also retains the frozen TeReL-batched, Local SupCon,
normalization-matched random, direct-covariance, objective-ablation, classical
SFA, IncSFA, and PAMAP2 protocols. Their stable artifact identifiers and
analysis commands are documented in `ARTIFACT_README.md`; manuscript prose uses
scientific names only.

## Repository layout

```text
terel/resubmission/                 method, baselines, analyses, and protocol gates
configs/resubmission/               validation and final configurations
docs/                               scientific protocol records
tests/                              mathematical and implementation-fidelity tests
```

The anonymous supplement is created with
`terel.resubmission.package_supplement`; it contains code, paper source, frozen
plans, ledgers, raw records, generated analyses, and a redaction/checksum
manifest without dataset files. Its source filter excludes tracked diagnostic
outputs, discarded configurations, previous versions, and internal work logs.

```bash
uv run python -m terel.resubmission.package_supplement \
  --repository . \
  --paper-repository /path/to/terel-paper \
  --artifact-root /path/to/final-supplement \
  --private-root /path/to/private-source-records \
  --output /path/to/TeReL-anonymous-supplement.zip
```
