from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _exporter_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/export_stage16d_reward_v3_representative_traces.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_reward_v3_trace_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _freeflight_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/classify_stage16d_reward_v3_freeflight.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_reward_v3_freeflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay_aggregate_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/aggregate_stage16d_reward_v3_replay_validation.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_reward_v3_replay_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contact_summary_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/summarize_stage16d_reward_v3_contact.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_reward_v3_contact_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reward_v3_representative_trace_keeps_exact_pair_force_and_contact_reward() -> None:
    frames, replicas = 321, 20
    pair_force = np.zeros((frames, replicas, 5, 3), dtype=np.float32)
    pair_force[:, 3, 0, 0] = 2.0
    expected = np.zeros((frames, replicas, 5), dtype=bool)
    expected[:, :, 0] = True
    arrays = {
        "replica_object_twist": np.zeros((frames, replicas, 6), dtype=np.float32),
        "object_twist_reference": np.zeros((frames, 6), dtype=np.float32),
        "replica_reference_contact_mask": expected,
        "replica_actual_contact_mask": expected.copy(),
        "replica_fingertip_object_pair_force_world": pair_force,
        "replica_fingertip_object_pair_force_valid": np.ones((frames, replicas), dtype=bool),
        "replica_contact_reward": np.full((frames, replicas), 0.5, dtype=np.float32),
        "fingertip_link_names": np.asarray(
            [
                "r_thumb_distal",
                "r_index_finger_distal",
                "r_middle_finger_distal",
                "r_ring_finger_distal",
                "r_pinky_distal",
            ]
        ),
    }

    selected = _exporter_module()._selected_trace(
        arrays,
        selected={"replica": 3, "seed": 42},
        role="best_progress",
        source_sha256="a" * 64,
    )

    assert selected["selected_fingertip_object_pair_force_world"].shape == (321, 5, 3)
    assert np.all(selected["selected_fingertip_object_pair_force_world"][:, 0, 0] == 2.0)
    assert np.all(selected["selected_S_contact_n"] == 2.0)
    assert np.all(selected["selected_r_contact"] == 0.5)
    assert int(selected["selected_replica"].item()) == 3


def test_freeflight_analysis_requires_persistent_loss_then_pairforce_recontact() -> None:
    module = _freeflight_module()
    frames, replicas = 321, 20
    force = np.zeros((frames, replicas, 5, 3), dtype=np.float32)
    force[3, 0, 0, 0] = 1.0
    actual = np.zeros((frames, replicas, 5), dtype=bool)
    actual[1:4, 0, 0] = True
    actual[8:, 0, 0] = True
    expected = np.zeros((frames, replicas, 5), dtype=bool)
    expected[:, :, 0] = True
    valid = np.ones((frames, replicas), dtype=bool)
    valid[0] = False
    events = module._events(
        force=force,
        valid=valid,
        actual=actual,
        pose=np.zeros((frames, replicas, 7), dtype=np.float32),
        twist=np.zeros((frames, replicas, 6), dtype=np.float32),
        expected=expected,
    )
    assert len(events) == 1
    assert events[0]["loss_duration_control_steps"] == 4
    assert events[0]["recontact_frame"] == 8
    assert module._classification(events, []) == "FREE_FLIGHT_RECATCH_RESOLVED"


def test_replay_aggregate_rejects_legacy_or_incomplete_v3_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        replica_fingertip_object_pair_force_world=np.zeros((321, 20, 5, 3), dtype=np.float32),
        replica_fingertip_object_pair_force_valid=np.ones((321, 20), dtype=bool),
        replica_reference_contact_mask=np.zeros((321, 20, 5), dtype=bool),
        replica_actual_contact_mask=np.zeros((321, 20, 5), dtype=bool),
        replica_contact_reward=np.zeros((321, 20), dtype=np.float32),
        reward_v3_samples=np.asarray(4_194_304, dtype=np.int64),
        trace_type=np.asarray("stage16d_ppo26d"),
        requested_clip=np.asarray("hocap_170105"),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        '{"status":"STAGE16D_PPO26D_REPLAY_VALIDATED","headless":true,'
        '"object":"hocap_170105","trace":"' + str(trace) + '",'
        '"finite":true,"frame_count":321}',
        encoding="utf-8",
    )
    evidence = _replay_aggregate_module()._headless_evidence("hocap_170105", receipt, trace)
    assert evidence["trace"]["reward_v3_samples"] == 4_194_304


def test_force_farming_ratio_is_unavailable_when_v3_has_no_actual_contact() -> None:
    module = _contact_summary_module()
    assert module._p95_ratio(None, 1.0) is None
    assert module._p95_ratio(2.0, None) is None
    assert module._p95_ratio(2.0, 0.0) is None
    assert module._p95_ratio(2.0, 0.5) == 4.0
