# Anonymous TeReL supplement

This archive contains the final implementation, paper source, frozen plans,
selection ledgers, raw records, generated analyses, and fidelity tests. It does
not contain datasets, discarded configurations, failed runs, or a development
chronology.

## Setup

From the extracted `TeReL-supplement/` directory, use Python 3.12 and the
locked environment:

```bash
uv sync --project source --extra test
uv run --project source pytest -q source/tests
```

Place MNIST under `data/mnist` and PAMAP2 under `data/pamap2`, or change only
the portable `data_root` field before reproducing a run. The executed records
may contain path-only redactions. `redaction-manifest.json` records the source
and packaged SHA-256 values for every file; redaction changes no metric,
configuration value, or seed.

## Evidence map

- `artifacts/residual-state-final-*` contains the frozen TeReL-S manifest,
  five final records, and their analysis.
- `artifacts/residual-state-validation-*` contains the two-method selection
  plan, resolved ledger, and matched residual-state comparison.
- `artifacts/batched-reference-*` contains TeReL-batched, backpropagation, and
  random-reference records.
- `artifacts/objective-mechanism-*` contains the soft-SFA component ablations.
- `artifacts/local-comparator-*` contains Local SupCon selection and final
  records together with the lagged/direct-covariance control.
- `artifacts/normalization-control-*` contains the BatchNorm-calibrated random
  reference.
- `artifacts/natural-stream-*` contains the PAMAP2 selection and final stress
  test.
- `paper/` contains the manuscript, generated tables, and figure sources.

## Reanalyze TeReL-S

```bash
uv run --project source python -m terel.resubmission.residual_state_analysis \
  --results artifacts/residual-state-final-results \
  --validation-ledger source/configs/resubmission/residual-state-validation-ledger.json \
  --analysis-output /tmp/residual-state-analysis.json \
  --results-tex /tmp/generated-residual-results.tex \
  --appendix-tex /tmp/generated-residual-appendix.tex
```

The execution commits and configuration hashes stored in the manifests are the
run identities. The top-level redaction manifest is the byte-level audit trail
for packaged files.
