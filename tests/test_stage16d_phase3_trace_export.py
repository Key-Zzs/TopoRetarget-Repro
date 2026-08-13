from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _exporter_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/export_stage16d_phase3_representative_traces.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_phase3_trace_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representative_trace_export_uses_formal_replicas_and_v2_twist_terms(
    tmp_path: Path,
) -> None:
    frames, replicas = 321, 20
    source = tmp_path / "formal_trace.npz"
    reference = np.zeros((frames, 6), dtype=np.float32)
    twists = np.zeros((frames, replicas, 6), dtype=np.float32)
    twists[:, 1, 0] = 0.075
    object_pose = np.zeros((frames, replicas, 7), dtype=np.float32)
    object_pose[..., 3] = 1.0
    hand_pose = np.zeros((frames, replicas, 1, 7), dtype=np.float32)
    hand_pose[..., 3] = 1.0
    np.savez_compressed(
        source,
        reference_kinematics_version=np.asarray(2, dtype=np.int64),
        object_twist_reference=reference,
        replica_object_twist=twists,
        replica_object_pose=object_pose,
        replica_hand_collision_body_pose=hand_pose,
        replica_contact_pair_presence=np.zeros((frames, replicas, 1), dtype=bool),
        replica_action=np.zeros((frames, replicas, 26), dtype=np.float32),
    )
    evaluation = {
        "seed_set": {"identifier": "formal_holdout_seed_set_v1"},
        "frame_zero": [
            {
                "seed": 100 + index,
                "object_tracking_error_m": {"final": float(index)},
            }
            for index in range(replicas)
        ],
    }
    qualification_rows = []
    for index in range(replicas):
        success = index not in {1, 2, 3}
        qualification_rows.append(
            {
                "seed": 100 + index,
                "complete_trajectory": success,
                "terminal_contact_pass": success,
                "terminal_stability_pass": success,
                "inter_finger_penetration_pass": True,
                "contact_causality_pass": True,
                "contact_topology_pass": True,
                "semantic_progress": 1.0 if index == 4 else 0.1,
            }
        )
    evaluation_path = tmp_path / "evaluation.json"
    qualification_path = tmp_path / "qualification.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    qualification_path.write_text(
        json.dumps(
            {"kind": "formal", "trace": str(source.resolve()), "episodes": qualification_rows}
        ),
        encoding="utf-8",
    )

    result = _exporter_module().export_representative_traces(
        evaluation_path=evaluation_path,
        qualification_path=qualification_path,
        source_trace_path=source,
        best_output=tmp_path / "best_trace.npz",
        failure_output=tmp_path / "failure_trace.npz",
        manifest_output=tmp_path / "manifest.json",
    )

    assert result["selected"]["best_progress"]["replica"] == 4
    # Failed replicas have final errors 1, 2, and 3: the median is replica 2.
    assert result["selected"]["representative_failure"]["replica"] == 2
    with np.load(tmp_path / "best_trace.npz", allow_pickle=False) as archive:
        assert str(archive["representative_trace_role"].item()) == "best_progress"
        assert int(archive["selected_replica"].item()) == 4
        assert int(archive["reference_kinematics_version"].item()) == 2
    with np.load(tmp_path / "failure_trace.npz", allow_pickle=False) as archive:
        assert str(archive["representative_trace_role"].item()) == "representative_failure"
        assert int(archive["selected_replica"].item()) == 2
        assert "selected_reward_obj_vel" in archive.files
        assert "selected_reward_obj_ang_vel" in archive.files
