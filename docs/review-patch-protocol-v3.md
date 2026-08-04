# Frozen protocol: new-review evidence patch v3

This protocol adds only the evidence requested by the independent post-revision
review. It does not retune TeReL or replace the five-seed v2 confirmatory
matrix.

## Data and split roles

- MNIST uses the existing deterministic split: 50,000 official-training
  examples for encoder/probe fitting, 10,000 official-training examples for
  validation, and the official 10,000-example split for sequential
  confirmation.
- Validation seeds are 101, 202, and 303. The later confirmatory comparator
  uses seeds 1101, 1202, 1303, 1404, and 1505.
- No candidate in this patch may use official-test metrics for selection.

## Local supervised-contrastive comparison

All six candidates use the canonical 784--512--256 architecture, LeakyReLU,
batch normalization, the all-layer linear probe, batch size 256, 60 complete
training-set presentations, AdamW, and no augmentation. The bounded grid varies
only encoder learning rate in {1e-4, 3e-4, 1e-3} and supervised-contrastive
temperature in {0.1, 0.2}. Each minibatch is class-chunk ordered and the loss
uses all same-label positives in that minibatch, which gives the comparator at
least the class-relation information available to TeReL through adjacency.

Select the candidate with the highest mean validation accuracy over the three
seeds. Break an exact tie by lower sample standard deviation, then lower
learning rate, then higher temperature. Freeze the selected configuration
before running the five test seeds. Report all validation candidates and all
confirmatory seeds.

## Lateral controls

The `lagged-proxy-audit` candidate is exactly the recovered canonical TeReL
configuration with reporting-only diagnostics enabled. For every nonzero
minibatch direction and layer it measures cosine alignment, relative L2 error,
and norm ratio between the operator available before minibatch k and the
off-diagonal operator constructed directly from minibatch k. These diagnostics
must not enter the loss or selection.

The `direct-covariance-matched` candidate keeps canonical undetached temporal
pairs, greedy 30-epoch-per-layer training, normalization, architecture,
readout, and data presentations. It replaces only the lagged lateral proxy by
the differentiable same-minibatch squared off-diagonal covariance penalty.
Its covariance coefficient is 0.25 because that penalty's activation gradient
has the factor four stated in the paper, whereas the proxy coefficient is 1.

Both lateral controls are repeated on validation seeds 101, 202, and 303. They
are diagnostic controls, not a new model-selection search.

## Stopping and reporting

- Run every predeclared validation candidate exactly once per validation seed.
- Do not extend the grid after observing results.
- Freeze a clean execution commit, a validation ledger containing every
  candidate record, and a one-run confirmatory manifest before test execution.
- Preserve raw JSON records, confusion matrices, resource accounting, and
  reporting scripts in the anonymized supplement.
