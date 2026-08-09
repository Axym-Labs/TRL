from terel.resubmission.canonical_validation import (
    load_validation_plan,
    resolve_candidate,
)


def test_validation_plan_resolves_one_attributable_override(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text(
        """
maximum_candidates: 2
data_root: /tmp/mnist
encoder_base:
  method: terel_local
  hidden_dims: [4, 2]
probe_base:
  readout: last
candidates:
  - id: baseline
  - id: all
    probe: {readout: all}
"""
    )

    plan = load_validation_plan(path)
    _, encoder, probe = resolve_candidate(plan, "all")

    assert encoder.hidden_dims == (4, 2)
    assert probe.readout == "all"
