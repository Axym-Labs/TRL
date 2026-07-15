# TeReL Ablation Run Specs (from `terel_ablation_results.csv`)

Source file: `analysis_outputs/terel_ablation_results.csv`
Columns available: `variant`, `val_acc`, `val_error`

## Naming legend

- `terel`: TeReL setup (uses chunk-aware policy where applicable).
- `terels`: TeReL-S setup (no chunk-aware policy where applicable).
- `baseline`: default setup for that family.
- `last_layer_head`: head uses only last hidden layer.
- `trace`: trace activation enabled.
- `trace_fast`: trace enabled with faster decay variant.
- `lateral_shift`: lateral temporal shift enabled.
- `shiftcov` or `lateral_shift_cov_target`: covariance target uses shifted target.
- `fast_cov`: faster covariance-related setting used in that experiment.

## Per-run details

| Variant | Family | Key changes vs family baseline | val_acc | val_error |
|---|---|---|---:|---:|
| `terel_baseline` | TeReL | None (reference TeReL setup) | 0.9691 | 0.0309 |
| `terel_last_layer_head` | TeReL | `last_layer_head` | 0.9633 | 0.0367 |
| `terel_trace` | TeReL | `trace` | 0.9542 | 0.0458 |
| `terel_lateral_shift` | TeReL | `lateral_shift` | 0.9165 | 0.0835 |
| `terel_trace_fast` | TeReL | `trace_fast` | 0.9711 | 0.0289 |
| `terel_lateral_shift_cov_target` | TeReL | `lateral_shift` + `shifted covariance target` | 0.9518 | 0.0482 |
| `terel_trace_lateral_shift_fast_cov` | TeReL | `trace_fast` + `lateral_shift` + `fast_cov` | 0.9336 | 0.0664 |
| `terel_trace_lateral_shift_last_layer` | TeReL | `trace` + `lateral_shift` + `last_layer_head` | 0.9120 | 0.0880 |
| `terels_baseline` | TeReL-S | None (reference TeReL-S setup) | 0.9658 | 0.0342 |
| `terels_shift_shiftcov_last_layer` | TeReL-S | `lateral_shift` + `shifted covariance target` + `last_layer_head` | 0.9289 | 0.0711 |
| `terels_tracefast_shift_shiftcov_last_layer` | TeReL-S | `trace_fast` + `lateral_shift` + `shifted covariance target` + `last_layer_head` | 0.9204 | 0.0796 |

## Quick read

- Best run in this file: `terel_trace_fast` (`val_acc=0.9711`).
- Strong baselines: `terel_baseline` and `terels_baseline`.
- Largest drops are in runs that combine `lateral_shift` with additional constraints (trace and/or last-layer head).
