# TeReL artifact provenance

The manuscript contains the final method, protocol, and scientific results.
This file contains immutable execution identifiers and checksums needed to
audit those results. Python 3.12 and direct dependencies are fixed by
`pyproject.toml` and `uv.lock`; runs enable deterministic PyTorch and cuDNN
behavior and the prescribed cuBLAS workspace configuration.

```bash
uv sync --extra test
uv run pytest -q
```

## Canonical samplewise TeReL

```text
execution commit   44ddf2ac737ef19530b457a34fe0ccd0c5547017
manifest           a697595d4f2a842e2f79db0bfe1dcf09ba251e4c003e97ef0095230471f75283
configuration      5b080d918c4050506d80d416bcbd4441bce3d9f1f988e7eb9bedee03a1e5937c
protocol source    fd7d698c6946289558ff85509e1098a24b078c3f351df468d21d82a989cd2476
validation ledger  7ecd9eeb72d38d7c6952746e696c18c191f7293aa6ce7816dfdb07477797d8cb
result records     eca357dd694a57be70477e6ccaab6f1cb4aaeadb0690a23afbc693c8594a2239
```

The record digest concatenates the five final files in lexical seed order. The
manifest contains the selected encoder and probe configuration, environment
record, and predetermined final runs. The matched no-inhibition comparison is
validation-only and is specified by `configs/canonical-online-mechanism.yaml`.

## TeReL-Offline and backpropagation references

```text
execution commit   02afd90cf6927a588aa424d61cb86c6876b25c17
manifest           9a1f8b8e5b75293efcbe5b13a767e74f57370f91658c73deb51e808e6c1c9828
configuration      3e83ea66c03a621bbdc6f1a16a143c6dbf50643ae2b5213a1d78afba68ed0a6b
protocol           faab529f7dab0d2940a8e7f6dd662c482b9da64bcf661e78029378c9fb0a68da
validation ledger  11faf7a030f9f76fd2174225ca8ba806d9439e0bbf63924b01361909ee3305cb
result records     727a0e9c335d5281cd3037ef8f64ee68ba927e9e77e4c0741312d0e4a7dcde88
analysis           d9b1b88f4acb24c2758cbe9b4396030352cd22f8df893a761c8c92458e6ab302
```

## Objective mechanisms

```text
execution commit       9532b3476c36e981dd5164c2c43affd4e791c9c6
configuration          c061c4f761e2037265a2c2b45033d05bc11c62c3470b6a3390756a6067c80d28
protocol               505db43ba8cb06d5175beb163f0828c58b58afcb9b66a95217e15b8abf1cd41e
result records         53c67637a43b4f92227db1a19a144a049b934f34ca1ac5506223b4971fcee6bc
analysis               95102e0d4858188f56daf638b979bb805341837639519bc93178cfc68fb91d7d
```

## Local SupCon and direct covariance

```text
execution commit   e738ec33792cff2f40add7de862622bd0f60661e
reporting commit   64cd2d82b9f1fe869dc1847a0ce4f8e5cd738f37
configuration      d17267673e4eb9263cee080acfb549820835701d01e3eed5eab0e22e302afba2
protocol           c36f86cf795d6fd83ef7e7fdf5f1672b6756d8b66ed2473feaf97749cfbcf829
validation ledger  79208773d0a12606b9211050f9a3649df3898263c2844adc29ce19ddf31d0b8b
manifest           dc021f1038d67add3bb7142ce25deb27a2fb94c6c43dcf5509b565a850c837fa
record digest      6f60879f23f927297892883195d8e2204adb6406b1ab736a71404799888e6654
analysis           6cc83cd265c3389ec82f9491b70148490d9ad8d5f6d1fafd3bdc89c3c424b0cb
```

## Normalization-matched random features

```text
execution commit   7e1631fc7cedd02cc870e4f1aa4007182970ae9d
reporting commit   888c073bd499df3130f96e8365cb1f0510fbea8d
configuration      01a03e1bb173ac5d8d03e8a1b2bf4a25abf3aab81620b58693f781558860de2f
protocol           d2c209bda037329bfbc61f40bdd9ae5405cad34ba420fedf34e0ee2e81d495dd
validation ledger  8714d2624d30ef8d5c185977813fbb922f35a54a2071ddebe77c0c8597bce62d
manifest           48ca2bfd2e2306955696b808c48f49d2cfb225eeed6367bc0e20a186e7fe1ce8
record digest      edbdf926351a3c84c028cdb11a364b049b934f34ca1ac5506223b4971fcee6bc
analysis           fc98d9073064ae78e7e8bd93be34b5484620d7d4735c29ffdd9cfc8af20ac9eb
```

## Anonymous supplement

The package contains only the canonical samplewise records and the current
scientific comparators. It excludes discarded configurations, partial ledgers,
old figures, development analyses, and natural-stream premise tests.

```bash
uv run python -m terel.resubmission.package_supplement \
  --repository . \
  --paper-repository /path/to/terel-paper \
  --artifact-root /path/to/final-artifacts \
  --private-root /path/to/private-source-records \
  --output /path/to/TeReL-anonymous-supplement.zip
```

The archive's `redaction-manifest.json` records source and packaged SHA-256
values for every file and marks deterministic path redactions.
