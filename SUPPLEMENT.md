# Anonymous TeReL supplement

This archive contains the final implementation, manuscript source, selected
protocols, raw result records, analyses, and fidelity tests needed to audit the
reported claims.

## Setup

From the extracted `TeReL-supplement/` directory, use Python 3.12 and the
locked environment:

```bash
uv sync --project source --extra test
uv run --project source pytest -q source/tests
```

Place MNIST under `data/mnist`, or change only the portable `data_root` field
before reproducing a run. Executed records may contain path-only redactions;
paper source also has identifying template names replaced.
`redaction-manifest.json` gives the source and packaged SHA-256 values of every
file. Redaction changes no metric, configuration value, or random seed.

## Evidence map

- `artifacts/canonical-online-confirmatory-*` contains the selected samplewise
  manifest, final records, and final execution ledger.
- `artifacts/canonical-online-analysis.json` combines the final summary with the
  matched validation effect and exact resource accounting.
- `artifacts/canonical-mechanism-*` contains the matched no-inhibition
  comparison.
- `artifacts/batched-reference-*` contains TeReL-Offline, backpropagation, and
  random-reference records.
- `artifacts/objective-mechanism-*` contains the soft-SFA component ablations.
- `artifacts/local-comparator-*` contains Local SupCon and the
  lagged/direct-covariance comparison.
- `artifacts/normalization-control-*` contains the BatchNorm-calibrated random
  reference.
- `source/configs/canonical-online-*` and
  `source/docs/canonical-online-protocol.md` specify the canonical method,
  selection record, mechanism boundary, and resource accounting.
- `paper/` contains only the current manuscript, generated tables, and figure
  sources.

The execution commits and configuration hashes inside the manifests identify
the runs. The top-level redaction manifest is the byte-level audit trail for
the package.
