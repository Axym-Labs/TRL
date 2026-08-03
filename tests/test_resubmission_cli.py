import json

from terel.resubmission.cli import load_experiment_spec


def test_yaml_experiment_spec_round_trips_frozen_method_and_probe_fields(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(
        """
dataset: mnist
data_root: /datasets/mnist
num_classes: 10
evaluation_split: validation
seeds: [101, 202, 303]
encoder:
  method: terel_local
  hidden_dims: [512, 256]
  epochs: 10
  batch_size: 256
  order_mode: class_chunks
  statistics_momentum: 0.9
  lateral_momentum: 0.99
probe:
  epochs: 30
  batch_size: 1024
  optimizer: adamw
  learning_rate: 0.003
  weight_decay: 0.0001
  readout: all
"""
    )

    spec = load_experiment_spec(config)

    assert spec.dataset == "mnist"
    assert spec.seeds == (101, 202, 303)
    assert spec.encoder.hidden_dims == (512, 256)
    assert spec.encoder.method == "terel_local"
    assert spec.encoder.statistics_momentum == 0.9
    assert spec.probe.learning_rate == 0.003
    assert spec.probe.readout == "all"
    json.dumps(spec.as_dictionary())
