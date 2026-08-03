import pytest

from terel.resubmission.confirmatory_v2 import validate_confirmatory_matrix


def _matrix():
    return {
        "schema_version": 2,
        "evaluation_split": "test",
        "seeds": [401, 502, 603, 704, 805],
        "probe": {"readout": "all"},
        "datasets": {
            "mnist": {
                "data_root": "/data/mnist",
                "num_classes": 10,
                "runs": [
                    {
                        "id": "terel-all",
                        "encoder": {
                            "method": "terel_batch",
                            "hidden_dims": [512, 256],
                        },
                    },
                    {
                        "id": "terel-last",
                        "encoder": {
                            "method": "terel_batch",
                            "hidden_dims": [512, 256],
                        },
                        "probe": {"readout": "last"},
                    },
                ],
            }
        },
    }


def test_v2_matrix_requires_five_seeds_and_unique_run_ids():
    matrix = validate_confirmatory_matrix(_matrix())

    assert matrix["evaluation_split"] == "test"
    assert matrix["datasets"]["mnist"]["runs"][1]["probe"]["readout"] == "last"


def test_v2_matrix_rejects_a_tuned_test_seed_count():
    matrix = _matrix()
    matrix["seeds"] = [401]

    with pytest.raises(ValueError, match="five distinct"):
        validate_confirmatory_matrix(matrix)
