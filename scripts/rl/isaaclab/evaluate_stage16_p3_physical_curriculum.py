#!/usr/bin/env python3
"""Evaluate one P3/P4 physical PPO checkpoint from frozen contact-ready resets.

This driver reuses the Stage16-D PPO evaluator's checkpoint loader, simulator
rollout, GPU trace capture, and collision-body reconstruction.  It only adds
the P3/P4 requirements that the legacy frame-zero evaluator cannot express:
pre-registered contact-ready `(seed, reset_index)` pairs, stage-matching
gravity/friction, and variable-length traces beginning from a safe RSI state.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_stage16d_ppo26d import (  # noqa: E402
    checkpoint_hash,
    model_from_checkpoint,
    run_episode,
)

from toporetarget.evaluation import (  # noqa: E402
    PhysicsEpisodeEvidence,
    aggregate_rollouts,
    hand_metric_series,
    object_metric_series,
    trajectory_success,
)
from toporetarget.evaluation.full_hand_contact import hand_body_manifest  # noqa: E402
from toporetarget.rl.geometry_audit.exact_evaluator import (
    evaluate_runtime_proxy_state,  # noqa: E402
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES as FK_HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.gravity_friction_curriculum import (  # noqa: E402
    INITIAL_SAFE_BANKS,
    load_gravity_friction_curriculum,
)
from toporetarget.rl.physical_evaluation import (  # noqa: E402
    FINGERS,
    P4_QUALIFICATION_SCHEMA,
    PHYSICAL_EVALUATION_SCHEMA,
    contact_metrics,
    flight_metrics,
    load_contact_ready_evaluation_pairs,
    physical_failure_status,
    twist_metrics,
    validate_pair_set_against_safe_indices,
)
from toporetarget.rl.physical_p3 import PHYSICAL_PPO_CHECKPOINT_SCHEMA  # noqa: E402
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode  # noqa: E402
from toporetarget.rl.rsi.contact_ready_v2 import load_safe_bank  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_SAFE_BANK_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2/p1"
DEFAULT_CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
DEFAULT_PAIR_CONTRACT = (
    REPO_ROOT / "configs/rl/stage16/stage16_p3_p4_contact_ready_evaluation_pairs_v1.yaml"
)
DEFAULT_GEOMETRY_MANIFEST = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)
DEFAULT_V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
DEFAULT_V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
DEFAULT_GATES = DEFAULT_V4_ROOT / "frozen_evaluation_gates.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("PHYSICAL_EVALUATION_CSV_ROWS_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(output: Path, *, phase: str) -> None:
    """Leave a durable phase receipt when Isaac closes before reporting an error."""
    _write_json(output / "evaluation_progress.json", {"phase": phase})


def _path_receipt(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PHYSICAL_EVALUATION_REQUIRED_INPUT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _mode_paths(mode: ContactRewardMode) -> tuple[Path, Path]:
    if mode is ContactRewardMode.AGGREGATE_V3:
        return (
            DEFAULT_V3_ROOT / "contact_reward_contract.json",
            REPO_ROOT / ".local/reports/stage16d_reward_v3_contact",
        )
    return DEFAULT_V4_ROOT / "strict_v4_contract.json", DEFAULT_V4_ROOT


def _stage_output(root: Path, *, mode: ContactRewardMode, clip: str, stage: str, kind: str) -> Path:
    if kind == "formal":
        return root / "p4" / clip
    if stage in {"C0", "C1", "C2"}:
        mode_directory = "v3" if mode is ContactRewardMode.AGGREGATE_V3 else "v4"
        return root / "physical_pilot" / mode_directory / clip / stage.lower() / "dev"
    return root / "selected_mode" / clip / stage.lower() / "dev"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument(
        "--contact-mode", choices=tuple(mode.value for mode in ContactRewardMode), required=True
    )
    parser.add_argument("--stage", choices=("C0", "C1", "C2", "C3", "C4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--safe-bank-root", type=Path, default=DEFAULT_SAFE_BANK_ROOT)
    parser.add_argument("--curriculum-contract", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--pair-contract", type=Path, default=DEFAULT_PAIR_CONTRACT)
    parser.add_argument("--geometry-manifest", type=Path, default=DEFAULT_GEOMETRY_MANIFEST)
    parser.add_argument("--frozen-gates", type=Path, default=DEFAULT_GATES)
    return parser


def _read_mapping(path: Path, *, marker: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{marker}_JSON_OBJECT_REQUIRED")
    return value


def _load_gate(path: Path, *, clip: str) -> dict[str, Any]:
    frozen = _read_mapping(path.resolve(), marker="PHYSICAL_EVALUATION_FROZEN_GATES")
    if frozen.get("status") != "STRICT_V4_EVALUATION_GATES_FROZEN":
        raise ValueError("PHYSICAL_EVALUATION_FROZEN_GATES_STATUS_INVALID")
    gates = frozen.get("task_gates", {}).get("clips", {})
    gate = gates.get(clip) if isinstance(gates, dict) else None
    if not isinstance(gate, dict) or gate.get("schema_version") != "PhysicsConsistentTaskGateV1":
        raise ValueError("PHYSICAL_EVALUATION_TASK_GATE_INVALID")
    return gate


def _checkpoint_contract(
    payload: Mapping[str, Any], *, clip: str, mode: ContactRewardMode, stage: str
) -> None:
    if (
        payload.get("schema_version") != PHYSICAL_PPO_CHECKPOINT_SCHEMA
        or payload.get("clip") != clip
        or payload.get("selected_contact_mode") != mode.value
        or payload.get("curriculum_stage") != stage
    ):
        raise ValueError("PHYSICAL_EVALUATION_CHECKPOINT_CONTRACT_MISMATCH")
    state = payload.get("curriculum_state")
    if not isinstance(state, Mapping) or (
        state.get("curriculum_stage") != stage
        or state.get("selected_contact_mode") != mode.value
        or tuple(state.get("allowed_reset_banks", ())) != INITIAL_SAFE_BANKS
    ):
        raise ValueError("PHYSICAL_EVALUATION_CHECKPOINT_CURRICULUM_STATE_INVALID")


def _environment_contract(
    report: Mapping[str, Any], *, clip: str, stage_physics: Mapping[str, Any]
) -> None:
    ppo = report.get("ppo26d")
    physics = report.get("gravity_friction_curriculum")
    if not isinstance(ppo, Mapping) or not isinstance(physics, Mapping):
        raise ValueError("PHYSICAL_EVALUATION_ENVIRONMENT_CONTRACT_MISSING")
    expected = {
        "fixed_clip": clip,
        "active_clip_ids": [clip],
        "object_rollout_state_writes": 0,
        "wrist_root_state_writes_during_step": 0,
    }
    observed = {key: ppo.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"PHYSICAL_EVALUATION_CAUSAL_ENVIRONMENT_DRIFT:{observed}")
    physics_expected = {
        "stage": stage_physics["curriculum_stage"],
        "gravity_scale": stage_physics["gravity_scale"],
        "friction_scale": stage_physics["friction_scale"],
        "support": "none",
        "external_guidance": False,
        "frame_zero_full_gravity_authorized": False,
    }
    mismatch = {
        key: {"expected": value, "observed": physics.get(key)}
        for key, value in physics_expected.items()
        if physics.get(key) != value
    }
    if mismatch:
        raise ValueError(f"PHYSICAL_EVALUATION_STAGE_PHYSICS_DRIFT:{mismatch}")


def _reconstruct_hand(trace: dict[str, np.ndarray]) -> np.ndarray:
    return reconstruct_hand_collision_body_pose(
        trace["wrist_pose"], trace["finger_q"], repo_root=REPO_ROOT
    ).astype(np.float32)


def _valid_rows(trace: Mapping[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(trace["fingertip_object_pair_force_valid"], dtype=bool)
    hand_valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    if valid.ndim != 1 or not np.array_equal(valid, hand_valid) or valid[0] or not valid[1:].all():
        raise ValueError("PHYSICAL_EVALUATION_PAIR_FORCE_VALIDITY_INVALID")
    return valid


def _reference_contact(
    trace: Mapping[str, np.ndarray], mode: ContactRewardMode
) -> tuple[np.ndarray, np.ndarray]:
    if mode is ContactRewardMode.AGGREGATE_V3:
        expected = np.asarray(trace["reference_contact_mask"], dtype=bool)
        actual = np.asarray(trace["actual_contact_mask"], dtype=bool)
    else:
        expected = np.asarray(trace["source_contact_mask"], dtype=bool)
        actual = np.asarray(trace["tip_pair_presence"], dtype=bool)
    if expected.shape != actual.shape or expected.shape[1:] != (len(FINGERS),):
        raise ValueError("PHYSICAL_EVALUATION_TIP_CONTACT_TRACE_INVALID")
    return expected, actual


def _inter_finger_penetration(hand_pose: np.ndarray) -> np.ndarray:
    from toporetarget.rl.physics_retargeting.self_collision import (
        InterFingerCapsulePenetrationV1,
        load_self_collision_contract,
    )

    contract = load_self_collision_contract(
        REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml", repo_root=REPO_ROOT
    )
    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        REPO_ROOT / contract.runtime_collision_manifest_path,
        expected_body_names=FK_HAND_COLLISION_BODY_NAMES,
        radius_scale=contract.capsule_radius_scale,
        device="cpu",
    )
    with torch.no_grad():
        values = metric.evaluate(torch.as_tensor(hand_pose, dtype=torch.float32))[
            "maximum_penetration_m"
        ]
    return np.asarray(values.numpy(), dtype=np.float64)


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("PHYSICAL_EVALUATION_NONFINITE_AGGREGATE")
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _mean_optional(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return None if not numeric else float(np.mean(numeric))


def _episode_status(
    *,
    trace: dict[str, np.ndarray],
    episode: Mapping[str, Any],
    gate: Mapping[str, Any],
    environment: Mapping[str, Any],
    geometry: Mapping[str, Any],
    inter_finger: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    valid = _valid_rows(trace)
    hand_contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    terminal_count = min(int(gate["terminal_window_control_steps"]), int(valid.sum()))
    terminal_indices = np.flatnonzero(valid)[-terminal_count:]
    terminal_contact = bool(episode["terminal_contact"])
    actual_twist = np.asarray(trace["object_twist"], dtype=np.float64)
    terminal_linear = np.linalg.norm(actual_twist[terminal_indices, :3], axis=-1)
    terminal_angular = np.linalg.norm(actual_twist[terminal_indices, 3:], axis=-1)
    terminal_limits_linear = np.where(
        hand_contact[terminal_indices],
        float(gate["terminal_linear_speed_mps"]),
        float(gate["terminal_free_object_linear_speed_mps"]),
    )
    terminal_limits_angular = np.where(
        hand_contact[terminal_indices],
        float(gate["terminal_angular_speed_radps"]),
        float(gate["terminal_free_object_angular_speed_radps"]),
    )
    terminal_stability = bool(
        terminal_contact
        and np.all(terminal_linear <= terminal_limits_linear)
        and np.all(terminal_angular <= terminal_limits_angular)
    )
    twist_delta = np.diff(actual_twist, axis=0)
    contact_causality = bool(
        np.any(hand_contact[1:] & (np.linalg.norm(twist_delta, axis=-1) > 1.0e-7))
    )
    action = np.asarray(trace["action"], dtype=np.float64)
    finite = all(
        np.isfinite(np.asarray(trace[name])).all()
        for name in ("object_pose", "object_twist", "wrist_pose", "finger_q", "action")
    )
    action_bounds = bool(
        np.isfinite(action).all() and np.all(np.abs(action) <= gate["action_limit"])
    )
    absolute_geometry = bool(
        float(geometry["max_penetration_m"]) < float(gate["catastrophic_penetration_m"])
        and float(geometry["p95_penetration_m"]) <= float(gate["p95_penetration_m"])
    )
    inter_pass = bool(inter_finger.max(initial=0.0) <= gate["maximum_inter_finger_penetration_m"])
    ppo = environment["ppo26d"]
    causality = {
        "external_guidance": False,
        "support": "none",
        "hidden_attachment": False,
        "hidden_force_or_attachment": bool(ppo["hidden_force_or_attachment"]),
        "object_rollout_state_write_count": int(ppo["object_rollout_state_writes"]),
        "wrist_root_rollout_write_count": int(ppo["wrist_root_state_writes_during_step"]),
        "frame_zero_full_gravity": False,
    }
    evidence = PhysicsEpisodeEvidence(
        terminal_contact_pass=terminal_contact,
        terminal_stability_pass=terminal_stability,
        contact_causality_pass=contact_causality,
        inter_finger_penetration_pass=inter_pass,
        absolute_hand_object_penetration_pass=absolute_geometry,
        action_bounds_pass=action_bounds,
        no_hidden_force=not causality["hidden_force_or_attachment"],
        no_object_rollout_state_write=causality["object_rollout_state_write_count"] == 0,
        no_wrist_root_teleport=causality["wrist_root_rollout_write_count"] == 0,
    )
    reason = int(episode["termination_reason"])
    failure = physical_failure_status(
        termination_reason=reason,
        finite=finite,
        absolute_geometry_pass=absolute_geometry,
        inter_finger_pass=inter_pass,
        max_penetration_m=float(geometry["max_penetration_m"]),
        catastrophic_penetration_m=float(gate["catastrophic_penetration_m"]),
    )
    return {"evidence": evidence, "causality": causality, "failure": failure}, {
        "terminal_stability": terminal_stability,
        "contact_causality": contact_causality,
        "finite": finite,
    }


def _evaluate_episode(
    *,
    index: int,
    pair: Any,
    env: Any,
    trainer: Any,
    clip: str,
    mode: ContactRewardMode,
    gate: Mapping[str, Any],
    environment: Mapping[str, Any],
    geometry_manifest: Path,
    output: Path,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    curriculum_receipt: Mapping[str, str],
    stage: str,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    env.cfg.evaluation_reset_reference_indices = (pair.reset_index,)
    episode = run_episode(
        env,
        trainer,
        capture=True,
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
        expected_clip=clip,
        seed=pair.seed,
    )
    if int(episode["start_reference_index"]) != pair.reset_index:
        raise RuntimeError("PHYSICAL_EVALUATION_RESET_PAIR_NOT_APPLIED")
    trace = episode.pop("trace")
    if not isinstance(trace, dict):
        raise RuntimeError("PHYSICAL_EVALUATION_TRACE_MISSING")
    trace["hand_collision_body_names"] = np.asarray(FK_HAND_COLLISION_BODY_NAMES)
    trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
    valid = _valid_rows(trace)
    expected, actual = _reference_contact(trace, mode)
    geometry, raw_geometry = evaluate_runtime_proxy_state(
        manifest_path=geometry_manifest,
        clip=clip,
        object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
        hand_collision_body_pose=np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[
            :, None
        ],
        hand_collision_body_names=tuple(str(item) for item in trace["hand_collision_body_names"]),
    )
    inter_finger = _inter_finger_penetration(trace["hand_collision_body_pose"])
    status, diagnostics = _episode_status(
        trace=trace,
        episode=episode,
        gate=gate,
        environment=environment,
        geometry=geometry,
        inter_finger=inter_finger,
    )
    metrics = object_metric_series(
        np.asarray(trace["object_pose"], dtype=np.float64),
        np.asarray(trace["object_reference"], dtype=np.float64),
    )
    metrics.update(
        hand_metric_series(
            np.asarray(trace["hand_collision_body_pose"], dtype=np.float64),
            [str(name) for name in trace["hand_collision_body_names"].tolist()],
            np.asarray(trace["tracked_link_reference"], dtype=np.float64),
            [str(name) for name in env.reference_bank.tracked_link_names],
        )
    )
    success = trajectory_success(
        metrics,
        complete=bool(episode["reached_final_reference"]),
        physics=status["evidence"],
    )
    contact, per_finger = contact_metrics(expected=expected, actual=actual, valid=valid)
    flight = flight_metrics(
        tip_contact=actual.any(axis=-1),
        hand_contact=np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1),
        valid=valid,
        object_pose=np.asarray(trace["object_pose"], dtype=np.float64),
        object_twist=np.asarray(trace["object_twist"], dtype=np.float64),
    )
    twist = twist_metrics(
        actual=np.asarray(trace["object_twist"], dtype=np.float64),
        reference=np.asarray(trace["object_twist_reference"], dtype=np.float64),
        valid=valid,
        terminal_steps=int(gate["terminal_window_control_steps"]),
    )
    trace_path = output / "traces" / f"episode_{index:03d}.npz"
    raw_path = output / "geometry" / f"episode_{index:03d}_pairs.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        **trace,
        trace_type=np.asarray("stage16_p3_p4_contact_ready_physical"),
        clip=np.asarray(clip),
        checkpoint_path=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(checkpoint_hash(checkpoint)),
        selected_contact_mode=np.asarray(mode.value),
        curriculum_stage=np.asarray(stage),
        policy_training_samples=np.asarray(int(checkpoint_payload["policy_training_samples"])),
        physical_cumulative_samples=np.asarray(
            int(checkpoint_payload["physical_cumulative_samples"])
        ),
        reference_hash=np.asarray(json.dumps(checkpoint_payload["reference_hash"], sort_keys=True)),
        curriculum_contract_sha256=np.asarray(curriculum_receipt["sha256"]),
        action_contract=np.asarray("26D_reference_residual"),
    )
    np.savez_compressed(raw_path, **raw_geometry)
    detailed = {
        "schema_version": PHYSICAL_EVALUATION_SCHEMA,
        "episode": index,
        "seed": pair.seed,
        "reset_index": pair.reset_index,
        "trace": _path_receipt(trace_path),
        "geometry_raw": _path_receipt(raw_path),
        "rollout": episode,
        "evaluation_suite_v2": success,
        "interaction": contact,
        "per_finger": per_finger,
        "flight": flight,
        "twist": twist,
        "penetration": {
            "hand_object": geometry,
            "inter_finger_max_penetration_m": float(inter_finger.max(initial=0.0)),
        },
        "causality": status["causality"],
        "physical_failure": status["failure"],
        "diagnostics": diagnostics,
    }
    _write_json(output / "episodes" / f"episode_{index:03d}.json", detailed)
    row = {
        "episode": index,
        "seed": pair.seed,
        "reset_index": pair.reset_index,
        "steps": int(episode["steps"]),
        "reached_reference_end": bool(episode["reached_final_reference"]),
        "E_r_mean_deg": float(success["E_r_mean_deg"]),
        "E_t_mean_cm": float(success["E_t_mean_cm"]),
        "E_j_mean_cm": float(success["E_j_mean_cm"]),
        "E_ft_mean_cm": float(success["E_ft_mean_cm"]),
        "kinematic_success": bool(success["kinematic_success"]),
        "physics_success": bool(success["physics_success"]),
        "qualified_success": bool(success["qualified_success"]),
        "terminal_contact": bool(status["evidence"].terminal_contact_pass),
        "terminal_stability": bool(status["evidence"].terminal_stability_pass),
        "contact_causality": bool(status["evidence"].contact_causality_pass),
        "source_tip_recall": contact["source_tip_recall"],
        "source_persistent_tip_recall": contact["source_persistent_tip_recall"],
        "cross_finger_compensation": contact["cross_finger_compensation"],
        "persistent_cross_finger_compensation": contact["persistent_cross_finger_compensation"],
        "fully_missing_source_contact": contact["fully_missing_source_contact"],
        "source_contact_full_coverage": contact["source_contact_full_coverage"],
        "no_tip_contact_fraction": flight["no_tip_contact_fraction"],
        "no_hand_object_contact_fraction": flight["no_hand_object_contact_fraction"],
        "flight_event_count": int(flight["flight_event_count"]),
        "longest_flight_gap": int(flight["longest_flight_gap"]),
        "recontact_count": int(flight["recontact_count"]),
        "Delta_v_mean_mps": float(twist["Delta_v_mps"]["mean"]),
        "Delta_v_p95_mps": float(twist["Delta_v_mps"]["p95"]),
        "Delta_v_terminal_mps": float(twist["Delta_v_mps"]["terminal"]),
        "Delta_omega_mean_radps": float(twist["Delta_omega_radps"]["mean"]),
        "Delta_omega_p95_radps": float(twist["Delta_omega_radps"]["p95"]),
        "Delta_omega_terminal_radps": float(twist["Delta_omega_radps"]["terminal"]),
        "hand_object_max_penetration_mm": float(geometry["max_penetration_m"]) * 1000.0,
        "hand_object_p95_penetration_mm": float(geometry["p95_penetration_m"]) * 1000.0,
        "active_p95_penetration_mm": float(geometry["active_p95_penetration_m"]) * 1000.0,
        "interfinger_max_penetration_mm": float(inter_finger.max(initial=0.0)) * 1000.0,
        "absolute_geometry_pass": bool(status["evidence"].absolute_hand_object_penetration_pass),
        "object_drop": bool(status["failure"]["object_drop"]),
        "joint_limit": bool(status["failure"]["joint_limit"]),
        "catastrophic_contact": bool(status["failure"]["catastrophic_contact"]),
        "finite": bool(status["failure"]["finite"]),
        "trace": str(trace_path.resolve()),
    }
    return row, [{"episode": index, **value} for value in per_finger], detailed


def _aggregate(
    *,
    rows: list[dict[str, object]],
    per_finger_rows: list[dict[str, object]],
    details: list[dict[str, object]],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    suite = {
        "schema_version": "TopoRetargetEvaluationSuiteV2ResultV1",
        "aggregate": aggregate_rollouts(rows),
    }
    interaction_keys = (
        "source_tip_recall",
        "source_persistent_tip_recall",
        "cross_finger_compensation",
        "persistent_cross_finger_compensation",
        "fully_missing_source_contact",
        "source_contact_full_coverage",
    )
    interaction = {
        "schema_version": "Stage16P3P4InteractionMetricsV1",
        "aggregate": {key: _mean_optional([row[key] for row in rows]) for key in interaction_keys},
        "per_finger": [
            {
                "finger": finger,
                **{
                    key: _mean_optional(
                        [row[key] for row in per_finger_rows if row["finger"] == finger]
                    )
                    for key in (
                        "source_tip_recall",
                        "persistent_source_tip_recall",
                        "cross_finger_compensation",
                        "persistent_cross_finger_compensation",
                    )
                },
            }
            for finger in FINGERS
        ],
    }
    flight_events = [
        {"episode": detail["episode"], **event}
        for detail in details
        for event in detail["flight"]["events"]
    ]
    flight = {
        "schema_version": "Stage16P3P4FlightMetricsV1",
        "no_tip_contact_fraction": _mean_optional([row["no_tip_contact_fraction"] for row in rows]),
        "no_hand_object_contact_fraction": _mean_optional(
            [row["no_hand_object_contact_fraction"] for row in rows]
        ),
        "flight_event_count": len(flight_events),
        "longest_flight_gap": max((int(row["longest_flight_gap"]) for row in rows), default=0),
        "mean_flight_gap": _mean_optional(
            [detail["flight"]["mean_flight_gap"] for detail in details]
        ),
        "recontact_count": sum(int(row["recontact_count"]) for row in rows),
        "events": flight_events,
    }
    twist = {
        "schema_version": "Stage16P3P4TwistMetricsV1",
        "aggregation": "equal_weight_per_episode",
        "Delta_v_mps": {
            key: _mean_optional([detail["twist"]["Delta_v_mps"][key] for detail in details])
            for key in ("mean", "p95", "terminal")
        },
        "Delta_omega_radps": {
            key: _mean_optional([detail["twist"]["Delta_omega_radps"][key] for detail in details])
            for key in ("mean", "p95", "terminal")
        },
        "terminal_abs_v_mps": _mean_optional(
            [detail["twist"]["terminal_abs_v_mps"] for detail in details]
        ),
        "terminal_abs_omega_radps": _mean_optional(
            [detail["twist"]["terminal_abs_omega_radps"] for detail in details]
        ),
        "terminal_stability_rate": float(np.mean([row["terminal_stability"] for row in rows])),
    }
    penetration = {
        "schema_version": "Stage16P3P4ExactGeometryMetricsV1",
        "geometry_contract": "RuntimeCollisionProxyPenetrationV1",
        "aggregation": "max_over_episodes_for_max; equal_weight_per_episode_for_p95",
        "hand_object_max_penetration_m": max(
            float(detail["penetration"]["hand_object"]["max_penetration_m"]) for detail in details
        ),
        "hand_object_p95_penetration_m": _mean_optional(
            [detail["penetration"]["hand_object"]["p95_penetration_m"] for detail in details]
        ),
        "active_p95_penetration_m": _mean_optional(
            [detail["penetration"]["hand_object"]["active_p95_penetration_m"] for detail in details]
        ),
        "interfinger_max_penetration_m": max(
            float(detail["penetration"]["inter_finger_max_penetration_m"]) for detail in details
        ),
        "absolute_geometry_pass": all(bool(row["absolute_geometry_pass"]) for row in rows),
    }
    return suite, interaction, flight, twist, penetration


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.kind == "formal" and args.stage != "C4":
        raise ValueError("P4_FORMAL_REQUIRES_C4_FULL_GRAVITY_CHECKPOINT")
    mode = ContactRewardMode.parse(args.contact_mode)
    checkpoint = args.checkpoint.resolve()
    curriculum = load_gravity_friction_curriculum(args.curriculum_contract.resolve())
    stage_physics = curriculum.physics(args.stage)
    if args.kind == "formal" and (
        stage_physics["gravity_scale"] != 1.0 or stage_physics["friction_scale"] != 1.0
    ):
        raise ValueError("P4_FORMAL_REQUIRES_FULL_GRAVITY_NOMINAL_FRICTION")
    pair_sets = load_contact_ready_evaluation_pairs(args.pair_contract.resolve())
    pair_set = pair_sets[args.clip][args.kind]
    safe_bank_path = args.safe_bank_root.resolve() / (
        f"safe_bank_{args.clip.removeprefix('hocap_')}.npz"
    )
    safe_bank = load_safe_bank(safe_bank_path)
    validate_pair_set_against_safe_indices(pair_set, safe_indices=safe_bank["runtime_index"])
    output = _stage_output(
        args.output_root.resolve(), mode=mode, clip=args.clip, stage=args.stage, kind=args.kind
    )
    if output.exists():
        existing_names = {entry.name for entry in output.iterdir()}
        recovery_receipts = {
            name == "evaluation_progress.json" or name.startswith("evaluation_failure")
            for name in existing_names
        }
        if existing_names and not all(recovery_receipts):
            raise FileExistsError(f"PHYSICAL_EVALUATION_OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=True)
    contact_contract, contact_mask_root = _mode_paths(mode)
    receipts = {
        "checkpoint": _path_receipt(checkpoint),
        "curriculum_contract": _path_receipt(args.curriculum_contract),
        "pair_contract": _path_receipt(args.pair_contract),
        "safe_bank": _path_receipt(safe_bank_path),
        "geometry_manifest": _path_receipt(args.geometry_manifest),
        "frozen_gates": _path_receipt(args.frozen_gates),
        "contact_contract": _path_receipt(contact_contract),
    }
    gate = _load_gate(args.frozen_gates.resolve(), clip=args.clip)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    _write_progress(output, phase="isaac_app_starting")
    app = AppLauncher(headless=True).app
    env = None
    phase = "isaac_app_started"
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        phase = "environment_configuring"
        _write_progress(output, phase=phase)
        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=1, clip=args.clip, rsi=True, critical_dr=False
        )
        ppo_cfg.configure_stage16d_contact_reward(
            cfg,
            mode=mode,
            reference_root=args.reference_root.resolve(),
            contact_reward_contract=contact_contract.resolve(),
            contact_mask_root=contact_mask_root.resolve(),
        )
        ppo_cfg.configure_stage16_contact_ready_rsi_v2(cfg, safe_bank_path=safe_bank_path)
        ppo_cfg.configure_stage16_p3_p4_curriculum(
            cfg, curriculum_contract_path=args.curriculum_contract.resolve(), stage=args.stage
        )
        phase = "environment_constructing"
        _write_progress(output, phase=phase)
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        phase = "environment_ready"
        _write_progress(output, phase=phase)
        trainer, checkpoint_payload = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip=args.clip
        )
        phase = "checkpoint_loaded"
        _write_progress(output, phase=phase)
        _checkpoint_contract(checkpoint_payload, clip=args.clip, mode=mode, stage=args.stage)
        environment = env.contract_report()
        _environment_contract(environment, clip=args.clip, stage_physics=stage_physics)
        rows: list[dict[str, object]] = []
        per_finger_rows: list[dict[str, object]] = []
        details: list[dict[str, object]] = []
        for index, pair in enumerate(pair_set.pairs):
            phase = f"episode_{index:03d}_running"
            _write_progress(output, phase=phase)
            row, per_finger, detailed = _evaluate_episode(
                index=index,
                pair=pair,
                env=env,
                trainer=trainer,
                clip=args.clip,
                mode=mode,
                gate=gate,
                environment=env.contract_report(),
                geometry_manifest=args.geometry_manifest.resolve(),
                output=output,
                checkpoint=checkpoint,
                checkpoint_payload=checkpoint_payload,
                curriculum_receipt=receipts["curriculum_contract"],
                stage=args.stage,
            )
            rows.append(row)
            per_finger_rows.extend(per_finger)
            details.append(detailed)
            _write_progress(output, phase=f"episode_{index:03d}_complete")
        phase = "aggregating"
        _write_progress(output, phase=phase)
        suite, interaction, flight, twist, penetration = _aggregate(
            rows=rows, per_finger_rows=per_finger_rows, details=details
        )
        final_environment = env.contract_report()
        _environment_contract(final_environment, clip=args.clip, stage_physics=stage_physics)
        status = (
            "P4_CONTACT_READY_FULL_GRAVITY_FORMAL20_COMPLETE"
            if args.kind == "formal"
            else "P3_CONTACT_READY_DEVELOPMENT_EVALUATION_COMPLETE"
        )
        qualification = {
            "schema_version": P4_QUALIFICATION_SCHEMA
            if args.kind == "formal"
            else PHYSICAL_EVALUATION_SCHEMA,
            "status": status,
            "kind": args.kind,
            "clip": args.clip,
            "contact_mode": mode.value,
            "curriculum_stage": args.stage,
            "curriculum_physics": stage_physics,
            "checkpoint": receipts["checkpoint"],
            "policy_training_samples": int(checkpoint_payload["policy_training_samples"]),
            "physical_cumulative_samples": int(checkpoint_payload["physical_cumulative_samples"]),
            "reset_domain": "contact_ready_rsi_v2",
            "pair_set": pair_set.as_dict(),
            "inputs": receipts,
            "physics_contract": final_environment,
            "physics_contract_sha256": hashlib.sha256(
                json.dumps(final_environment, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "causal_contract": {
                "gravity_scale": stage_physics["gravity_scale"],
                "friction_scale": stage_physics["friction_scale"],
                "support": "none",
                "external_guidance": False,
                "frame_zero_full_gravity": False,
                "rollout_object_state_writes": 0,
                "rollout_wrist_root_writes": 0,
            },
            "episodes": rows,
            "evaluation_suite_v2": suite,
            "interaction": interaction,
            "flight": flight,
            "twist": twist,
            "penetration": penetration,
            "physical_failure": {
                key: int(sum(bool(row[key]) for row in rows))
                for key in ("object_drop", "joint_limit", "catastrophic_contact")
            },
            "finite_episode_count": int(sum(bool(row["finite"]) for row in rows)),
        }
        _write_csv(output / "per_episode.csv", rows)
        _write_csv(output / "per_finger.csv", per_finger_rows)
        _write_json(output / "evaluation_suite_v2.json", suite)
        _write_json(output / "interaction.json", interaction)
        _write_json(output / "flight.json", flight)
        _write_json(output / "twist.json", twist)
        _write_json(output / "penetration.json", penetration)
        _write_json(output / "qualification.json", qualification)
        _write_json(output / "evaluation.json", qualification)
        _write_json(
            output / "replay_contract.json",
            {
                "schema_version": "Stage16P3P4ReplayContractV1",
                "trace_schema": "stage16_p3_p4_contact_ready_physical",
                "viewer": str(
                    REPO_ROOT / "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py"
                ),
                "full_hand_manifest": hand_body_manifest(
                    tuple(FK_HAND_COLLISION_BODY_NAMES), repo_root=REPO_ROOT
                ),
            },
        )
        _write_progress(output, phase="complete")
        print(json.dumps({"status": status, "output": str(output.resolve())}))
        return 0
    except BaseException as error:
        failure_path = output / "evaluation_failure.json"
        attempt = 1
        while failure_path.exists():
            attempt += 1
            failure_path = output / f"evaluation_failure_attempt_{attempt:03d}.json"
        _write_json(
            failure_path,
            {
                "schema_version": "Stage16P3P4PhysicalEvaluationFailureV1",
                "status": "PHYSICAL_EVALUATION_FAILED",
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
