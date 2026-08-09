import pytest

from terel.resubmission import residual_confirmatory


def _plan():
    return {
        "seeds": [101, 102, 103],
        "data_root": "/data/mnist",
        "encoder_base": {
            "hidden_dims": [512, 256],
            "activation": "leaky_relu",
            "normalization": "none",
            "epochs": 2,
            "batch_size": 1,
        },
        "probe_base": {"epochs": 60, "batch_size": 2048, "readout": "all"},
        "candidates": [
            {"id": "terel-s-reference", "encoder": {"method": "terel_local"}},
            {
                "id": "terel-s-residual",
                "encoder": {
                    "method": "terel_residual",
                    "residual_lateral_rule": "dual_inhibitory",
                    "residual_lateral_coefficient": 1000.0,
                },
            },
        ],
    }


def test_residual_confirmatory_configuration_freezes_only_the_selected_method():
    configuration = residual_confirmatory.resolve_configuration(
        _plan(),
        {
            "selection_complete": True,
            "selected_configuration_id": "terel-s-residual",
        },
        confirmatory_seeds=(42, 43, 44, 45, 46),
    )

    assert configuration["evaluation_split"] == "test"
    assert configuration["seeds"] == [42, 43, 44, 45, 46]
    assert configuration["selection_seeds"] == [101, 102, 103]
    assert len(configuration["datasets"]["mnist"]["runs"]) == 1
    run = configuration["datasets"]["mnist"]["runs"][0]
    assert run["id"] == "terel-s-residual"
    assert run["encoder"]["method"] == "terel_residual"
    assert run["encoder"]["residual_lateral_coefficient"] == 1000.0


def test_residual_confirmatory_configuration_rejects_unfinished_selection():
    with pytest.raises(ValueError, match="selection is incomplete"):
        residual_confirmatory.resolve_configuration(
            _plan(),
            {
                "selection_complete": False,
                "selected_configuration_id": "terel-s-residual",
            },
        )
