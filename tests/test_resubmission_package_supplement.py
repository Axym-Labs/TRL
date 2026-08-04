import json

from terel.resubmission.package_supplement import sanitize_text_artifact


def test_supplement_sanitizer_removes_local_paths_without_changing_numbers():
    original = json.dumps(
        {
            "data_root": "/home/reviewer/main/data/mnist",
            "manifest_path": "/home/reviewer/main/private/manifest.json",
            "accuracy": 0.973,
            "seed": 1101,
        }
    )

    sanitized = sanitize_text_artifact(
        original,
        replacements={
            "/home/reviewer/main/data/mnist": "data/mnist",
            "/home/reviewer/main": "<WORKSPACE>",
        },
    )
    parsed = json.loads(sanitized)

    assert "/home/" not in sanitized
    assert parsed["data_root"] == "data/mnist"
    assert parsed["accuracy"] == 0.973
    assert parsed["seed"] == 1101
