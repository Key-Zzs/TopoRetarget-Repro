from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.evaluation.qualify_physical_functionality_full_cycle import qualify


def test_missing_destination_signals_are_not_fabricated(tmp_path: Path) -> None:
    count = 12
    pose = np.zeros((count, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    wrist = pose.copy()
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    np.savez(
        trace_root / "episode_00.npz",
        object_pose=pose,
        wrist_pose=wrist,
        action=np.zeros((count, 26), dtype=np.float64),
        tip_pair_presence=np.zeros((count, 5), dtype=bool),
        hand_object_pair_presence=np.zeros((count, 21), dtype=bool),
        table_object_contact=np.ones(count, dtype=bool),
        table_object_contact_valid=np.ones(count, dtype=bool),
        fingertip_object_pair_force_valid=np.ones(count, dtype=bool),
    )
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            {
                "schema_version": "HardeningV2RuntimeEventsV1",
                "contact": 1,
                "pickup": 3,
                "place": 8,
                "release": 9,
            }
        ),
        encoding="utf-8",
    )
    result = qualify(
        trace_root=trace_root,
        output=tmp_path / "output",
        runtime_events=events,
        destination_signal_root=None,
        geometry_safe=True,
    )
    assert result["source_table_contact_substituted_for_destination_support"] is False
    assert result["destination_signals_available_for_all_traces"] is False
    assert result["phase_status_counts"]["PF_pick"]["FAIL"] == 1
    assert result["phase_status_counts"]["PF_transport"]["NOT_REACHED"] == 1
