# Anonymous TeReL supplement

This supplement contains the implementation, frozen plans, validation ledgers,
raw per-seed records, analysis outputs, manuscript source, and fidelity tests
used by the revised paper. Dataset files are not redistributed.

## Setup

Use Python 3.12 and install the locked environment:

```bash
uv sync --extra test
uv run pytest -q
```

Place MNIST under `data/mnist` and PAMAP2 under `data/pamap2`, or update only
the portable `data_root` fields before reproducing a run. The original executed
records may contain path-only redactions. `redaction-manifest.json` gives both
the SHA-256 of each source artifact before redaction and the packaged SHA-256;
no metric or configuration number is altered.

## Evidence map

- `artifacts/confirmatory-results-v2/`: the five-seed corrected MNIST matrix.
- `artifacts/mechanism-audit-results-v2/`: validation-only one-factor controls.
- `artifacts/review-patch-validation-v3/`: Local SupCon selection and the
  lagged/direct lateral controls requested in the latest review.
- `artifacts/review-patch-confirmatory-results-v3/`: the selected Local SupCon
  comparator on the five frozen test seeds.
- `artifacts/confirmatory-results/`: the earlier frozen natural-order PAMAP2
  stress test, retained as secondary evidence.
- `source/configs/resubmission/` and `source/docs/`: resolved matrices,
  validation ledgers, and protocols.

The code commit and configuration hashes stored in each confirmatory manifest
are the execution identities. Path redaction does not change those embedded
identities; use the redaction manifest when checking packaged file bytes.

## Reanalysis

```bash
uv run python -m terel.resubmission.analysis_v2 \
  --results artifacts/confirmatory-results-v2 \
  --streaming-results artifacts/recovery-results-v2 \
  --analysis-output /tmp/confirmatory-analysis-v2.json \
  --results-tex /tmp/generated-results-v2.tex \
  --appendix-tex /tmp/generated-appendix-v2.tex

uv run python -m terel.resubmission.review_patch_analysis \
  --validation-results artifacts/review-patch-validation-v3 \
  --output /tmp/review-patch-analysis-v3.json \
  --confirmatory-results artifacts/review-patch-confirmatory-results-v3 \
  --reference-results artifacts/confirmatory-results-v2 \
  --confirmatory-output /tmp/review-patch-confirmatory-analysis-v3.json \
  --results-tex /tmp/generated-review-patch-v3.tex \
  --appendix-tex /tmp/generated-review-patch-appendix-v3.tex
```
