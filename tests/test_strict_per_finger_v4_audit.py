from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_audit_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/audit_stage16d_strict_per_finger_v4.py"
    )
    spec = importlib.util.spec_from_file_location("strict_v4_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("strict V4 audit module cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body_names() -> tuple[str, ...]:
    return (
        "r_wrist",
        "r_index_finger_proximal",
        "r_index_finger_middle",
        "r_index_finger_aux",
        "r_index_finger_distal",
        "r_middle_finger_proximal",
        "r_middle_finger_middle",
        "r_middle_finger_aux",
        "r_middle_finger_distal",
        "r_pinky_proximal",
        "r_pinky_middle",
        "r_pinky_aux",
        "r_pinky_distal",
        "r_ring_finger_proximal",
        "r_ring_finger_middle",
        "r_ring_finger_aux",
        "r_ring_finger_distal",
        "r_thumb_proximal",
        "r_thumb_middle",
        "r_thumb_aux",
        "r_thumb_distal",
    )


def test_strict_v4_audit_requires_own_tip_and_records_substitution(tmp_path: Path) -> None:
    audit = _load_audit_module()
    mask = np.zeros((321, 5), dtype=bool)
    mask[1, 0] = True
    mask[2:5, 1] = True
    source = tmp_path / "mask.npz"
    np.savez_compressed(
        source,
        strict_source_contact_mask=mask,
        finger_names=np.asarray(("thumb", "index", "middle", "ring", "pinky")),
        control_index=np.arange(321),
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "status": "STRICT_V4_CONTACT_CONTRACT_FROZEN",
                "frozen_parameters": {
                    "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
                    "source_required_classes": [
                        "SOURCE_CONTACT_CONFIRMED",
                        "SOURCE_CONTACT_PERSISTENT",
                    ],
                    "numerical_floor_n": 1.0e-4,
                },
            }
        ),
        encoding="utf-8",
    )
    source_trace = np.broadcast_to(mask[:, None], (321, 20, 5)).copy()
    valid = np.ones((321, 20), dtype=bool)
    valid[0] = False
    hand_force = np.zeros((321, 20, 21, 3), dtype=np.float32)
    hand_force[1, 0, 20, 0] = 1.0  # Required thumb receives its own named tip.
    hand_force[2:5, 0, 1, 0] = 1.0  # Required index has same-finger non-tip only.
    hand_presence = np.linalg.norm(hand_force, axis=-1) > 1.0e-4
    tip_force = hand_force[:, :, (20, 4, 8, 16, 12)]
    tip_presence = hand_presence[:, :, (20, 4, 8, 16, 12)]
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        replica_source_contact_mask=source_trace,
        replica_tip_pair_presence=tip_presence,
        replica_tip_pair_force_world=tip_force,
        replica_hand_object_pair_force_world=hand_force,
        replica_hand_object_pair_presence=hand_presence,
        replica_hand_object_pair_force_valid=valid,
        replica_object_pose=np.zeros((321, 20, 7), dtype=np.float32),
        replica_object_twist=np.zeros((321, 20, 6), dtype=np.float32),
        replica_reward_total=np.zeros((321, 20), dtype=np.float32),
        hand_body_names=np.asarray(_body_names()),
        fingertip_link_names=np.asarray(
            (
                "r_thumb_distal",
                "r_index_finger_distal",
                "r_middle_finger_distal",
                "r_ring_finger_distal",
                "r_pinky_distal",
            )
        ),
        fingertip_force_sensor_indices=np.asarray((20, 4, 8, 16, 12), dtype=np.int64),
        replica_r_contact_v4=np.zeros((321, 20), dtype=np.float32),
        replica_per_finger_contact_reward=np.zeros((321, 20, 5), dtype=np.float32),
    )
    result = audit.audit(
        clip="hocap_170105",
        trace_path=trace,
        source_mask_path=source,
        contract_path=contract,
    )
    by_finger = {row["finger"]: row for row in result["per_finger"]}
    assert by_finger["thumb"]["source_tip_recall"] == 1.0 / 20.0
    assert by_finger["index"]["source_tip_recall"] == 0.0
    assert by_finger["index"]["same_finger_non_tip_substitution_fraction"] == 1.0 / 20.0
    assert result["aggregate"]["satisfied_source_tip_samples"] == 1
    assert result["aggregate"]["valid_source_tip_samples"] == 80
