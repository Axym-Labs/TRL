import pytest

from terel.resubmission.confirmatory import _probe_for_run, resolve_confirmatory_configuration


def _selection_inputs():
    plan = {
        "protocol_sha256": "protocol-hash",
        "seeds": [101, 202, 303],
        "probe": {
            "epochs": 30,
            "batch_size": 2048,
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
        },
        "datasets": {
            "mnist": {
                "data_root": "/data/mnist",
                "num_classes": 10,
                "encoder_base": {
                    "method": "terel_local",
                    "hidden_dims": [512, 256],
                    "epochs": 10,
                    "batch_size": 256,
                    "order_mode": "class_chunks",
                },
                "configurations": [
                    {
                        "id": "selected",
                        "learning_rate": 0.001,
                        "variance_coefficient": 5.0,
                        "covariance_coefficient": 1.0,
                    }
                ],
            },
            "pamap2": {
                "data_root": "/data/pamap2",
                "num_classes": 12,
                "encoder_base": {
                    "method": "terel_local",
                    "hidden_dims": [512, 256],
                    "epochs": 10,
                    "batch_size": 512,
                    "order_mode": "chronological",
                },
                "configurations": [
                    {
                        "id": "selected",
                        "learning_rate": 0.0003,
                        "variance_coefficient": 2.5,
                        "covariance_coefficient": 0.5,
                    }
                ],
            },
        },
    }
    ledger = {
        "selection_complete": True,
        "datasets": {
            "mnist": {"selected_configuration_id": "selected"},
            "pamap2": {"selected_configuration_id": "selected"},
        },
    }
    return plan, ledger


def test_confirmatory_configuration_resolves_selection_and_fixed_controls():
    plan, ledger = _selection_inputs()

    configuration = resolve_confirmatory_configuration(
        plan,
        ledger,
        confirmatory_seeds=(1001, 1002, 1003, 1004, 1005),
    )

    assert configuration["seeds"] == [1001, 1002, 1003, 1004, 1005]
    mnist = {run["id"]: run for run in configuration["datasets"]["mnist"]["runs"]}
    pamap2 = {run["id"]: run for run in configuration["datasets"]["pamap2"]["runs"]}
    assert mnist["terel-local"]["encoder"]["learning_rate"] == 0.001
    assert mnist["random"]["encoder"]["method"] == "random"
    assert mnist["local-supcon"]["encoder"]["method"] == "local_supcon"
    assert mnist["bp"]["encoder"]["method"] == "bp"
    assert pamap2["terel-shuffled"]["encoder"]["order_mode"] == "shuffled"
    assert pamap2["batch-sfa"]["encoder"]["method"] == "sfa"
    assert pamap2["incremental-sfa"]["encoder"]["method"] == "incsfa"
    assert pamap2["incremental-sfa"]["encoder"]["epochs"] == 1
    assert pamap2["incremental-sfa"]["encoder"]["incsfa_output_dim"] == 52
    assert configuration["selection_seeds"] == [101, 202, 303]


def test_confirmatory_configuration_refuses_incomplete_selection():
    plan, ledger = _selection_inputs()
    ledger["selection_complete"] = False

    with pytest.raises(ValueError, match="incomplete"):
        resolve_confirmatory_configuration(
            plan,
            ledger,
            confirmatory_seeds=(1001, 1002, 1003, 1004, 1005),
        )


def test_confirmatory_run_can_freeze_a_secondary_readout_without_changing_encoder():
    global_probe = {
        "epochs": 60,
        "batch_size": 2048,
        "optimizer": "adamw",
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "readout": "all",
    }
    run = {"id": "terel-last", "encoder": {}, "probe": {"readout": "last"}}

    probe = _probe_for_run(global_probe, run)

    assert probe.readout == "last"
    assert probe.epochs == 60
