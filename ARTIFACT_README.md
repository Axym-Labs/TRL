# TeReL artifact provenance

This document holds the execution identifiers, checksums, and internal artifact
filenames that are intentionally omitted from the scientific manuscript. The
manuscript states the final method and evaluation protocol; this record makes
the supplied results auditable byte for byte.

## Environment and verification

Python 3.12 and direct dependencies are pinned in `pyproject.toml` and
`uv.lock`. Runs enable deterministic PyTorch and cuDNN behavior and the
prescribed cuBLAS workspace configuration.

```bash
uv sync --extra test
uv run pytest -q
```

## Primary MNIST evidence

```text
execution commit  02afd90cf6927a588aa424d61cb86c6876b25c17
manifest           9a1f8b8e5b75293efcbe5b13a767e74f57370f91658c73deb51e808e6c1c9828
configuration      3e83ea66c03a621bbdc6f1a16a143c6dbf50643ae2b5213a1d78afba68ed0a6b
protocol           faab529f7dab0d2940a8e7f6dd662c482b9da64bcf661e78029378c9fb0a68da
validation ledger  11faf7a030f9f76fd2174225ca8ba806d9439e0bbf63924b01361909ee3305cb
result records     727a0e9c335d5281cd3037ef8f64ee68ba927e9e77e4c0741312d0e4a7dcde88
analysis           edfbe6443f161491795f9eae8f0c01011df249f591c672c7325f261fb25f21c0
```

## Mechanism evidence

```text
execution commit       9532b3476c36e981dd5164c2c43affd4e791c9c6
configuration          c061c4f761e2037265a2c2b45033d05bc11c62c3470b6a3390756a6067c80d28
protocol               505db43ba8cb06d5175beb163f0828c58b58afcb9b66a95217e15b8abf1cd41e
result records         53c67637a43b4f92227db1a19a144a049b934f34ca1ac5506223b4971fcee6bc
execution analysis     9f7378302e47219ef14e0d7a30d20a8f37417a5595ae8a94e502944b1bf43d61
regenerated analysis   95102e0d4858188f56daf638b979bb805341837639519bc93178cfc68fb91d7d
```

The execution analysis is retained as an immutable run artifact. Regenerating
with the supplied reporting source produces the second JSON, identical paper
numbers, and additional Student-t metadata.

## Local SupCon and direct-covariance evidence

```text
execution commit  e738ec33792cff2f40add7de862622bd0f60661e
reporting commit  64cd2d82b9f1fe869dc1847a0ce4f8e5cd738f37
configuration     d17267673e4eb9263cee080acfb549820835701d01e3eed5eab0e22e302afba2
protocol          c36f86cf795d6fd83ef7e7fdf5f1672b6756d8b66ed2473feaf97749cfbcf829
validation ledger 79208773d0a12606b9211050f9a3649df3898263c2844adc29ce19ddf31d0b8b
manifest          dc021f1038d67add3bb7142ce25deb27a2fb94c6c43dcf5509b565a850c837fa
record digest     6f60879f23f927297892883195d8e2204adb6406b1ab736a71404799888e6654
analysis          6cc83cd265c3389ec82f9491b70148490d9ad8d5f6d1fafd3bdc89c3c424b0cb
```

## Normalization-matched random evidence

```text
execution commit  7e1631fc7cedd02cc870e4f1aa4007182970ae9d
reporting commit  888c073bd499df3130f96e8365cb1f0510fbea8d
configuration     01a03e1bb173ac5d8d03e8a1b2bf4a25abf3aab81620b58693f781558860de2f
protocol          d2c209bda037329bfbc61f40bdd9ae5405cad34ba420fedf34e0ee2e81d495dd
validation ledger 8714d2624d30ef8d5c185977813fbb922f35a54a2071ddebe77c0c8597bce62d
manifest          48ca2bfd2e2306955696b808c48f49d2cfb225eeed6367bc0e20a186e7fe1ce8
record digest     edbdf926351a3c84c028cdb11a364dba957412fbf05c21695902ae9d1901f585
analysis          fc98d9073064ae78e7e8bd93be34b5484620d7d4735c29ffdd9cfc8af20ac9eb
```

## Analysis commands

The source archive retains the executed `v2`/`v3`/`v4` filenames as stable
artifact identifiers.

```bash
uv run python -m terel.resubmission.confirmatory run \
  --manifest artifacts/confirmatory-manifest-v2.json \
  --validation-ledger configs/resubmission/validation-ledger-v2.json \
  --protocol /path/to/confirmatory-protocol-v2.md \
  --repository . \
  --output artifacts/confirmatory-results-v2 \
  --device cuda \
  --allow-test

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

uv run python -m terel.resubmission.review_patch_analysis \
  --validation-results artifacts/review-patch-validation-v3 \
  --confirmatory-results artifacts/review-patch-confirmatory-results-v3 \
  --reference-results artifacts/confirmatory-results-v2 \
  --output artifacts/review-patch-validation-analysis-v3.json \
  --confirmatory-output artifacts/review-patch-confirmatory-analysis-v3.json

uv run python -m terel.resubmission.latest_review_analysis \
  --control-results artifacts/latest-review-confirmatory-results-v4 \
  --reference-results artifacts/confirmatory-results-v2 \
  --analysis-output artifacts/latest-review-analysis-v4.json
```

## PAMAP2 stress-test evidence

```text
execution commit  abfa0ec1df78204b66d0c34141f5deb6063a572a
manifest          38ca928e0884a8bc40a5c7ffefc46313fc96c170661aea63f93bf9496b2b1af0
result records    ad817a220fae65f9ebceb5941f967ec5bff2015c10847d1f8b00f54032ce7213
```

Absolute source paths are redacted in the anonymous supplement and paired with
source and packaged checksums. Dataset files are not redistributed.
