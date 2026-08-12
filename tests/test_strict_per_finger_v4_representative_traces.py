from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/export_stage16d_strict_per_finger_v4_representative_traces.py"
    )
    spec = importlib.util.spec_from_file_location("strict_v4_representative", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("strict V4 representative module cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_hand_flight_trace_selection_carries_the_frozen_episode_seed() -> None:
    module = _module()
    qualification = {
        "episodes": [
            {"replica": replica, "seed": 1000 + replica, "semantic_progress": 0.1}
            for replica in range(20)
        ]
    }
    suite_rows = [
        {
            "replica": str(replica),
            "qualified_success": "False",
            "physics_success": "False",
            "E_t_terminal_cm": "1.0",
        }
        for replica in range(20)
    ]
    audit = {
        "per_replica": [
            {
                "source_tip_recall": 0.0,
                "persistent_source_tip_recall": 0.0,
            }
            for _ in range(20)
        ],
        "no_tip_no_hand_flight_events": [
            {
                "event_type": "NO_HAND_OBJECT_CONTACT_FLIGHT",
                "replica": 7,
                "duration_control_steps": 5,
            }
        ],
    }

    result = module._select(
        qualification=qualification,
        suite_rows=suite_rows,
        audit=audit,
    )

    flight = result["representative_no_hand_flight_recontact"]
    assert flight is not None
    assert flight["replica"] == 7
    assert flight["seed"] == 1007
