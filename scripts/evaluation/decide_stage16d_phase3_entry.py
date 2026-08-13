#!/usr/bin/env python3
"""Freeze the evidence-backed Stage 16-D Phase 3 authorization decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DObservationV2  # noqa: E402

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
GEOMETRY_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _observability() -> dict[str, Any]:
    source_path = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    )
    source = source_path.read_text(encoding="utf-8")
    contract = Stage16DPPO26DObservationV2()
    current_object_twist = 'state["object_twist_world"]' in source
    observation_fields = contract.field_dimensions()
    if contract.dimension != 764 or observation_fields.get("current_object_twist") != 6:
        raise RuntimeError("PHASE3_OBJECT_TWIST_OBSERVATION_CONTRACT_DRIFT")
    return {
        "schema_version": "Stage16DObjectTwistObservabilityAuditV1",
        "status": "PHASE3_OBJECT_TWIST_OBSERVABILITY_PASS"
        if current_object_twist
        else "PHASE3_OBJECT_TWIST_OBSERVABILITY_BLOCKED",
        "actor_observation_dimension": contract.dimension,
        "actor_observation_contract": contract.as_dict(),
        "current_object_linear_velocity_available_to_actor": current_object_twist,
        "current_object_angular_velocity_available_to_actor": current_object_twist,
        "reference_linear_velocity_available_to_reward_backend": True,
        "reference_angular_velocity_available_to_reward_backend": True,
        "twist_error_directly_available_to_reward_backend": True,
        "future_actual_twist_added_to_actor": False,
        "observation_dimension_changed": False,
        "static_source": {"path": str(source_path), "sha256": _hash(source_path)},
        "rationale": (
            "The frozen actor receives current signed world object twist (linear and angular) "
            "in its existing 6-D current_object_twist field. V2 reference twist and the error "
            "are intentionally reward-backend values; no future actual twist is exposed."
        ),
    }


def _contact_causality(trace: Path) -> dict[str, Any]:
    with np.load(trace, allow_pickle=False) as archive:
        contact = np.asarray(archive["replica_contact_pair_presence"], dtype=bool).any(axis=-1)
        twist = np.asarray(archive["replica_object_twist"], dtype=np.float64)
    if contact.shape != (321, 20) or twist.shape != (321, 20, 6):
        raise ValueError("PHASE3_G5_TRACE_SHAPE_INVALID")
    response = np.linalg.norm(np.diff(twist, axis=0), axis=-1) > 1.0e-7
    per_replica = np.any(contact[1:] & response, axis=0)
    return {
        "pass": bool(per_replica.all()),
        "pass_rate": float(per_replica.mean()),
        "episodes": int(per_replica.size),
        "definition": "observed contact coincides with a nonzero object-twist response",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--geometry-root", type=Path, default=GEOMETRY_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    phase = root / "phase1_rerun"
    qualification = _read(root / "reference_kinematics_qualification.json")
    phase1 = _read(phase / "summary.json")
    counterfactuals = _read(phase / "counterfactuals.json")
    observability = _observability()
    _write(root / "object_twist_observability.json", observability)

    residual = {}
    contact = {}
    geometry = {}
    causality = {}
    for clip in CLIPS:
        short = clip.removeprefix("hocap_")
        attribution = _read(phase / f"attribution_{short}.json")
        terminal = _read(phase / f"terminal_twist_{short}.json")
        trace = Path(str(terminal["trace"]["path"])).resolve()
        residual[clip] = {
            "failure_episode_count": attribution["failure_episode_count"],
            "meaningful_residual_count": attribution["meaningful_residual_count"],
            "meaningful_residual_fraction": attribution["meaningful_residual_fraction"],
            "primary": attribution["primary"],
            "secondary": attribution["secondary"],
        }
        contact[clip] = terminal["terminal_reference"]
        causality[clip] = _contact_causality(trace)
        report_path = args.geometry_root.resolve() / (
            f"geometry_qualification_{short}_reference_v2_phase1r.json"
        )
        report = _read(report_path)
        absolute = report.get("absolute_gates")
        runtime = report.get("runtime_physx_crosscheck", {})
        if not isinstance(absolute, dict):
            raise ValueError("PHASE3_G5_ABSOLUTE_GEOMETRY_GATES_MISSING")
        geometry[clip] = {
            "report": str(report_path),
            "report_sha256": _hash(report_path),
            "absolute_geometry_safety_pass": bool(all(absolute.values())),
            "absolute_gates": absolute,
            "runtime_physx_crosscheck_pass": bool(runtime.get("pass", False)),
            "source_relative_comparability_pass": bool(
                all(
                    bool(value.get("pass", False))
                    for value in report.get("relative_gates", {}).values()
                    if isinstance(value, dict)
                )
            ),
            "source_relative_status": report["status"],
        }

    source = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    ).read_text(encoding="utf-8")
    no_hidden_control = "PPO26D_ROLLOUT_STATE_WRITE_FORBIDDEN" in source
    g1 = qualification.get("status") == "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
    g2 = observability["status"] == "PHASE3_OBJECT_TWIST_OBSERVABILITY_PASS"
    g3 = any(float(row["meaningful_residual_fraction"]) >= 0.5 for row in residual.values())
    g4 = any(
        float(contact[clip]["terminal_contact_rate"]) >= 0.95
        and int(residual[clip]["failure_episode_count"]) > 0
        for clip in CLIPS
    )
    g5 = (
        no_hidden_control
        and all(bool(row["pass"]) for row in causality.values())
        and all(
            bool(row["absolute_geometry_safety_pass"])
            and bool(row["runtime_physx_crosscheck_pass"])
            for row in geometry.values()
        )
    )
    gates = {
        "G1_reference_kinematics_v2": {"pass": g1, "evidence": qualification["status"]},
        "G2_object_twist_observability": {
            "pass": g2,
            "evidence": str((root / "object_twist_observability.json").resolve()),
        },
        "G3_meaningful_terminal_residual_dynamics": {"pass": g3, "clips": residual},
        "G4_contact_not_primary_failure": {
            "pass": g4,
            "terminal_contact_and_failures": {
                clip: {
                    "terminal_contact_rate": contact[clip]["terminal_contact_rate"],
                    "failure_episode_count": residual[clip]["failure_episode_count"],
                }
                for clip in CLIPS
            },
        },
        "G5_physics_integrity": {
            "pass": g5,
            "no_hidden_control": no_hidden_control,
            "contact_causality": causality,
            "geometry": geometry,
            "note": (
                "Source-relative geometry comparability remains a required reported diagnostic, "
                "but is not the Evaluation Suite V2 absolute geometry-safety gate."
            ),
        },
    }
    recommended = all(bool(row["pass"]) for row in gates.values())
    decision = {
        "schema_version": "Stage16DPhase3EntryDecisionV1",
        "status": (
            "PHASE3_OBJECT_TWIST_REWARD_RECOMMENDED"
            if recommended
            else "PHASE3_OBJECT_TWIST_REWARD_NOT_RECOMMENDED"
        ),
        "reference_kinematics_version": 2,
        "phase1r_status": phase1["status"],
        "counterfactual_status": counterfactuals["status"],
        "gates": gates,
        "source_relative_geometry_diagnostic": {
            clip: geometry[clip]["source_relative_status"] for clip in CLIPS
        },
        "authorization": {
            "phase3_implementation_and_training_authorized": recommended,
            "only_clip_authorized": "hocap_170650" if recommended else None,
            "hocap_170105_reward_v2_training_authorized": False,
            "contact_reward_authorized": False,
            "gravity_or_friction_curriculum_authorized": False,
        },
    }
    _write(root / "phase3_entry_decision.json", decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "gates": {key: value["pass"] for key, value in gates.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
