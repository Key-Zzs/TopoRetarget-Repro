"""Pure-contract coverage for the Stage16 controlled-hand gravity diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAGNOSE = _load(
    "stage16_hand_gravity_root_cause",
    "scripts/evaluation/diagnose_stage16_hand_gravity_root_cause.py",
)
INSPECT = _load(
    "stage16_hand_gravity_inspector",
    "scripts/rl/isaaclab/inspect_stage16_hand_gravity.py",
)


def _pose(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    values = np.zeros((2, 7), dtype=np.float64)
    values[:, 3:] = quaternion
    return values


def test_reference_command_actual_metrics_preserve_tracking_attribution() -> None:
    command = _pose((1.0, 0.0, 0.0, 0.0))
    actual = command.copy()
    actual[1, 3:] = (0.0, 0.0, 0.0, 1.0)
    trace = {
        "embedded_reference_wrist_pose": command,
        "wrist_target_pose": command,
        "wrist_pose": actual,
        "embedded_reference_finger_q": np.zeros((2, 20)),
        "finger_target_q": np.zeros((2, 20)),
        "finger_q": np.ones((2, 20)),
        "virtual_wrist_target_q": np.zeros((2, 6)),
        "virtual_wrist_q": np.zeros((2, 6)),
        "actuator_effort": np.zeros((2, 26)),
        "reference_index": np.asarray((0, 320)),
        "phase": np.asarray((0, 6)),
    }
    metrics, rows, joints = DIAGNOSE._trace_metrics(trace)
    assert metrics["wrist_orientation_deg"]["reference_to_command"]["mean"] == 0.0
    assert metrics["wrist_orientation_deg"]["command_to_actual"]["mean"] == 90.0
    assert metrics["actual_tracking_primary"] is True
    assert metrics["reference_phase"] == {
        "index_start": 0,
        "index_end": 320,
        "unique_indices": 2,
        "unique_phase_codes": [0, 6],
    }
    assert rows[0]["semantic_phase_name"] == "PRE_CONTACT"
    assert rows[1]["semantic_phase_name"] == "TERMINAL"
    assert [row["Joint"] for row in joints] == [
        "virtual_prismatic_x",
        "virtual_prismatic_y",
        "virtual_prismatic_z",
        "virtual_revolute_x",
        "virtual_revolute_y",
        "virtual_revolute_z",
    ]


def test_per_body_gravity_interpretation_handles_schema_defaults() -> None:
    class Attr:
        def __init__(self, valid: bool, value: bool = False) -> None:
            self.valid = valid
            self.value = value

        def IsValid(self) -> bool:
            return self.valid

        def Get(self) -> bool:
            return self.value

    class API:
        def __init__(self, prim: Attr) -> None:
            self.prim = prim

        def GetDisableGravityAttr(self) -> Attr:
            return self.prim

    class Physx:
        PhysxRigidBodyAPI = API

    assert INSPECT._gravity_disabled(Attr(False), Physx) is None
    assert INSPECT._gravity_disabled(Attr(True, False), Physx) is False
    assert INSPECT._gravity_disabled(Attr(True, True), Physx) is True


def test_gravity_on_mode_is_diagnostic_only_and_cannot_replace_normal_evaluation() -> None:
    evaluator = (REPO_ROOT / "scripts/rl/isaaclab/evaluate_physical_hoi.py").read_text(
        encoding="utf-8"
    )
    inspector = (REPO_ROOT / "scripts/rl/isaaclab/inspect_stage16_hand_gravity.py").read_text(
        encoding="utf-8"
    )
    assert 'choices=("current_off", "ablation_on")' in evaluator
    assert "HAND_GRAVITY_ABLATION_REQUIRES_FULL_TRAJECTORY_TABLE" in evaluator
    assert "diagnostic-only" in evaluator
    assert "world_gravity_mps2" in inspector
    assert "ON_ALL_HAND_AND_VIRTUAL_BODIES" in inspector
    assert "world gravity = 0" not in inspector.lower()
