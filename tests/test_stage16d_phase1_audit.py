"""Pure-contract coverage for Stage 16-D Phase 1 audit helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE1 = _load("stage16d_phase1_audit", "scripts/evaluation/audit_stage16d_phase1.py")
RSI = _load("stage16d_rsi_audit", "scripts/rl/isaaclab/audit_stage16d_rsi_state_quality.py")
COUNTERFACTUAL = _load(
    "stage16d_counterfactual_audit",
    "scripts/rl/isaaclab/audit_stage16d_terminal_counterfactuals.py",
)


def _reference(path: Path, *, twist_scale: float = 1.0) -> None:
    times = np.arange(321, dtype=np.float64) * 0.05
    angle = times * 0.2
    quaternion = np.stack(
        (np.cos(angle * 0.5), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle * 0.5)),
        axis=-1,
    )
    position = np.stack((times * 0.1, np.zeros_like(times), np.zeros_like(times)), axis=-1)
    twist = np.tile(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2]), (321, 1)) * twist_scale
    np.savez(
        path,
        timestamps=times,
        object_pose_translation_world_ref=position,
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=twist,
        metadata=np.asarray(json.dumps({"quaternion_convention": "wxyz"})),
    )


def test_reference_finite_difference_checks_linear_and_world_angular_twist(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    invalid = tmp_path / "invalid.npz"
    _reference(valid)
    _reference(invalid, twist_scale=0.0)
    valid_result = PHASE1._reference_audit(valid)
    invalid_result = PHASE1._reference_audit(invalid)
    assert valid_result["reference_twist_valid"] is True
    assert valid_result["factor8_velocity_scaling"] == "stored source twist divided by 8"
    assert valid_result["finite_difference"]["angular_world_error_radps"]["p95"] < 1.0e-9
    assert invalid_result["status"] == "REFERENCE_TWIST_CONTRACT_INVALID"


def test_rsi_strata_cover_all_required_phases_and_classification_boundaries() -> None:
    topology = {
        "source_onset_window": {"start": 60, "end": 80},
        "final_hold_window": {"start": 210, "end": 230},
    }
    indices = RSI._indices(topology)
    phases = {RSI._phase(index, topology) for index in indices}
    assert len(indices) <= 64
    assert {
        "pre_contact",
        "near_contact",
        "contact_onset",
        "persistent_contact",
        "manipulation",
        "terminal",
    } <= phases
    assert RSI._phase(320, topology) == "terminal"


def test_terminal_residual_and_support_classification_are_explicit() -> None:
    actual = np.array([[0.2, -0.1, 0.0, 0.4, 0.2, -0.3]])
    reference = np.array([[0.1, 0.1, 0.0, 0.1, 0.2, 0.0]])
    np.testing.assert_allclose(
        PHASE1._twist_residual(actual, reference),
        [[0.1, -0.2, 0.0, 0.3, 0.0, -0.3]],
    )
    assert (
        PHASE1._classify_source_support(
            explicit_support=False, inferred_support=True, explicit_absence=False
        )
        == "SUPPORT_INFERRED"
    )
    assert (
        PHASE1._classify_source_support(
            explicit_support=False, inferred_support=False, explicit_absence=False
        )
        == "SUPPORT_UNKNOWN"
    )


def test_rsi_uniform_histogram_and_per_clip_phase_counts_cover_10k_resets() -> None:
    topology = {
        "source_onset_window": {"start": 60, "end": 80},
        "final_hold_window": {"start": 210, "end": 230},
    }
    audit, rows = PHASE1._rsi_audit({"hocap_example": topology})
    dynamic = audit["dynamic_sampling"]
    assert dynamic["sample_count"] == 10_000
    assert sum(row["count"] for row in rows) == 10_000
    phases = dynamic["phase_counts_by_clip"]["hocap_example"]["counts"]
    assert sum(phases.values()) == 10_000
    assert set(phases) == {
        "pre_contact",
        "near_contact",
        "contact_onset",
        "persistent_contact",
        "manipulation",
        "terminal",
    }


def test_reset_only_write_report_is_fail_closed() -> None:
    assert RSI._reset_only_write_report(0, 0)["pass"] is True
    try:
        RSI._reset_only_write_report(1, 0)
    except RuntimeError as error:
        assert "prohibited rollout state writes" in str(error)
    else:
        raise AssertionError("rollout object writes must fail the RSI diagnostic")


def test_gravity_counterfactual_changes_only_gravity_flags() -> None:
    cfg = SimpleNamespace(
        sim=SimpleNamespace(gravity=(0.0, 0.0, 0.0), untouched="keep"),
        object_170105=SimpleNamespace(
            spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True))
        ),
        object_170650=SimpleNamespace(
            spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True))
        ),
        controller="unchanged",
    )
    COUNTERFACTUAL._apply_gravity_only(cfg)
    assert cfg.sim.gravity == (0.0, 0.0, -9.81)
    assert cfg.object_170105.spawn.rigid_props.disable_gravity is False
    assert cfg.object_170650.spawn.rigid_props.disable_gravity is False
    assert cfg.sim.untouched == "keep"
    assert cfg.controller == "unchanged"


def test_rotation_error_uses_raw_so3_geodesic_degrees() -> None:
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    half_turn_z = np.array([[0.0, 0.0, 0.0, 1.0]])
    error = COUNTERFACTUAL._rotation_error_deg(identity, half_turn_z)
    assert error[0] == 180.0
