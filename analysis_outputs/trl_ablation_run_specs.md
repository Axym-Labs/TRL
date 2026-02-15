# TRL Ablation Run Specs (from `trl_ablation_results.csv`)

Source file: `analysis_outputs/trl_ablation_results.csv`  
Columns available: `variant`, `val_acc`, `val_error`

## Naming legend

- `trl`: TRL setup (uses chunk-aware policy where applicable).
- `trls`: TRL-S setup (no chunk-aware policy where applicable).
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
| `trl_baseline` | TRL | None (reference TRL setup) | 0.9691 | 0.0309 |
| `trl_last_layer_head` | TRL | `last_layer_head` | 0.9633 | 0.0367 |
| `trl_trace` | TRL | `trace` | 0.9542 | 0.0458 |
| `trl_lateral_shift` | TRL | `lateral_shift` | 0.9165 | 0.0835 |
| `trl_trace_fast` | TRL | `trace_fast` | 0.9711 | 0.0289 |
| `trl_lateral_shift_cov_target` | TRL | `lateral_shift` + `shifted covariance target` | 0.9518 | 0.0482 |
| `trl_trace_lateral_shift_fast_cov` | TRL | `trace_fast` + `lateral_shift` + `fast_cov` | 0.9336 | 0.0664 |
| `trl_trace_lateral_shift_last_layer` | TRL | `trace` + `lateral_shift` + `last_layer_head` | 0.9120 | 0.0880 |
| `trls_baseline` | TRL-S | None (reference TRL-S setup) | 0.9658 | 0.0342 |
| `trls_shift_shiftcov_last_layer` | TRL-S | `lateral_shift` + `shifted covariance target` + `last_layer_head` | 0.9289 | 0.0711 |
| `trls_tracefast_shift_shiftcov_last_layer` | TRL-S | `trace_fast` + `lateral_shift` + `shifted covariance target` + `last_layer_head` | 0.9204 | 0.0796 |

## Quick read

- Best run in this file: `trl_trace_fast` (`val_acc=0.9711`).
- Strong baselines: `trl_baseline` and `trls_baseline`.
- Largest drops are in runs that combine `lateral_shift` with additional constraints (trace and/or last-layer head).
