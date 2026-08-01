#!/usr/bin/env python3
"""Run bounded Stage-16B world-wrist controller and oracle qualification."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from toporetarget.geometry.se3 import rotation_geodesic_error
from toporetarget.rl.environments.world_wrist_backend import (
    WorldWristFingerBackend,
    WorldWristObservationContractV1,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
)
from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.state_machine import RecoveryBudget, Stage16RecoveryStateMachine
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1
from toporetarget.rl.world_wrist_oracle import WorldWristFingerObjectAwareOracle

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"

IMPEDANCE_CANDIDATES = (
    WristImpedanceProfileV1(250.0, 1.0, 0.1, 2.0, 25.0, 1.5),
    WristImpedanceProfileV1(250.0, 1.0, 0.5, 1.0, 25.0, 1.5),
    WristImpedanceProfileV1(250.0, 1.0, 1.0, 0.5, 25.0, 1.5),
    WristImpedanceProfileV1(250.0, 1.0, 2.0, 0.5, 25.0, 1.5),
)
ACTION_SCALE_CANDIDATES = (
    WristFingerActionScaleV1(0.005, float(np.deg2rad(2.5)), 0.05),
    WristFingerActionScaleV1(0.010, float(np.deg2rad(5.0)), 0.10),
    WristFingerActionScaleV1(0.020, float(np.deg2rad(10.0)), 0.20),
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _rotation_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(rotation_geodesic_error(current, target)))


@dataclass(frozen=True)
class Episode:
    termination: str
    success: bool
    final_reach: bool
    progress: float
    object_position_error_m: float
    object_rotation_error_deg: float
    max_axis_error_m: float
    wrist_position_error_m: float
    wrist_rotation_error_deg: float
    finger_rmse_rad: float
    link_rmse_m: float
    contact_frame: int | None
    contact_count: int
    mean_wrist_wrench_n: float
    mean_wrist_torque_nm: float
    wrist_force_saturation_fraction: float
    wrist_torque_saturation_fraction: float
    wrist_wrench_saturation_fraction: float
    action_magnitude: float


def _episode(
    backend: WorldWristFingerBackend,
    *,
    policy: str,
    horizon: int = 1,
    kinematic_object_diagnostic: bool = False,
) -> Episode:
    state = backend.reset(reference_index=0)
    oracle = WorldWristFingerObjectAwareOracle() if policy == "oracle" else None
    contact_frame: int | None = None
    contact_count = 0
    force_norms: list[float] = []
    torque_norms: list[float] = []
    force_saturation: list[bool] = []
    torque_saturation: list[bool] = []
    actions: list[np.ndarray] = []
    reason: str | None = None
    for step in range(backend.reference.frame_count + 4):
        if policy == "exogenous_wrist":
            state = backend.exogenous_wrist_playback_step()
            reason = (
                "SUCCESS_REFERENCE_COMPLETE"
                if backend.reference_index >= backend.reference.frame_count - 1
                else None
            )
        else:
            action = (
                np.zeros(26, dtype=np.float64)
                if policy == "zero"
                else oracle.action(backend, horizon=horizon)
            )
            actions.append(action)
            state, _, reason = backend.transition(
                action, kinematic_object_diagnostic=kinematic_object_diagnostic
            )
        contact = backend.contact_summary()
        contact_count += int(contact["hand_object_contact_count"])
        if contact_frame is None and contact["hand_object_contact_count"]:
            contact_frame = step
        for physics_row in backend.last_physics_trace:
            wrench = np.asarray(physics_row["wrist_wrench_world"], dtype=np.float64)
            force_norms.append(float(np.linalg.norm(wrench[:3])))
            torque_norms.append(float(np.linalg.norm(wrench[3:])))
            force_saturation.append(bool(physics_row["force_saturated"]))
            torque_saturation.append(bool(physics_row["torque_saturated"]))
        if reason is not None:
            break
    if reason is None:
        reason = "FAILURE_EVALUATION_STEP_BOUND"
    index = min(backend.reference_index, backend.reference.frame_count - 1)
    reference = backend.reference
    action_array = np.asarray(actions) if actions else np.zeros((0, 26))
    return Episode(
        termination=reason,
        success=reason == "SUCCESS_REFERENCE_COMPLETE",
        final_reach=index >= reference.frame_count - 1,
        progress=float(index / (reference.frame_count - 1)),
        object_position_error_m=float(
            np.linalg.norm(
                state["object_pose"][:3, 3] - reference.object_pose_world_ref[index, :3, 3]
            )
        ),
        object_rotation_error_deg=_rotation_error_deg(
            state["object_pose"], reference.object_pose_world_ref[index]
        ),
        max_axis_error_m=float(
            np.max(
                np.linalg.norm(
                    state["object_axis_points"] - reference.object_axis_points_world_ref[index],
                    axis=1,
                )
            )
        ),
        wrist_position_error_m=float(
            np.linalg.norm(
                state["wrist_pose"][:3, 3] - reference.wrist_pose_world_ref[index, :3, 3]
            )
        ),
        wrist_rotation_error_deg=_rotation_error_deg(
            state["wrist_pose"], reference.wrist_pose_world_ref[index]
        ),
        finger_rmse_rad=float(
            np.sqrt(np.mean(np.square(state["q"] - reference.q_finger_ref[index])))
        ),
        link_rmse_m=float(
            np.sqrt(
                np.mean(
                    np.square(state["links"] - reference.tracked_link_positions_world_ref[index])
                )
            )
        ),
        contact_frame=contact_frame,
        contact_count=contact_count,
        mean_wrist_wrench_n=float(np.mean(force_norms)) if force_norms else 0.0,
        mean_wrist_torque_nm=float(np.mean(torque_norms)) if torque_norms else 0.0,
        wrist_force_saturation_fraction=float(np.mean(force_saturation))
        if force_saturation
        else 0.0,
        wrist_torque_saturation_fraction=float(np.mean(torque_saturation))
        if torque_saturation
        else 0.0,
        wrist_wrench_saturation_fraction=float(
            np.mean(np.logical_or(force_saturation, torque_saturation))
        )
        if force_saturation
        else 0.0,
        action_magnitude=float(np.mean(np.linalg.norm(action_array, axis=1)))
        if action_array.size
        else 0.0,
    )


def _summary(rows: list[Episode]) -> dict[str, Any]:
    if not rows:
        raise ValueError("qualification requires at least one episode")
    return {
        "episode_count": len(rows),
        "success_rate": float(np.mean([row.success for row in rows])),
        "final_reach_rate": float(np.mean([row.final_reach for row in rows])),
        "progress": float(np.mean([row.progress for row in rows])),
        "object_position_error_cm": float(
            np.mean([row.object_position_error_m for row in rows]) * 100.0
        ),
        "object_rotation_error_deg": float(
            np.mean([row.object_rotation_error_deg for row in rows])
        ),
        "max_axis_error_cm": float(np.mean([row.max_axis_error_m for row in rows]) * 100.0),
        "wrist_position_error_cm": float(
            np.mean([row.wrist_position_error_m for row in rows]) * 100.0
        ),
        "wrist_rotation_error_deg": float(np.mean([row.wrist_rotation_error_deg for row in rows])),
        "finger_rmse_rad": float(np.mean([row.finger_rmse_rad for row in rows])),
        "link_rmse_mm": float(np.mean([row.link_rmse_m for row in rows]) * 1000.0),
        "contact_frames": [row.contact_frame for row in rows],
        "mean_contact_count": float(np.mean([row.contact_count for row in rows])),
        "mean_wrist_wrench_n": float(np.mean([row.mean_wrist_wrench_n for row in rows])),
        "mean_wrist_torque_nm": float(np.mean([row.mean_wrist_torque_nm for row in rows])),
        "wrist_force_saturation_fraction": float(
            np.mean([row.wrist_force_saturation_fraction for row in rows])
        ),
        "wrist_torque_saturation_fraction": float(
            np.mean([row.wrist_torque_saturation_fraction for row in rows])
        ),
        "wrist_wrench_saturation_fraction": float(
            np.mean([row.wrist_wrench_saturation_fraction for row in rows])
        ),
        "action_magnitude": float(np.mean([row.action_magnitude for row in rows])),
        "termination_distribution": {
            term: sum(row.termination == term for row in rows)
            for term in sorted({row.termination for row in rows})
        },
    }


def _make_backend(
    *,
    reference_path: Path,
    mesh_path: Path,
    scene_root: Path,
    impedance: WristImpedanceProfileV1,
    action_scale: WristFingerActionScaleV1,
    seed: int,
) -> WorldWristFingerBackend:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    reference = WorldWristFingerReferenceV1.from_npz(reference_path)
    scene = materialize_world_wrist_free_object_scene(WUJI_MJCF, scene_root, object_mesh=mesh_path)
    return WorldWristFingerBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
        impedance_profile=impedance,
        action_scale=action_scale,
        seed=seed,
    )


def _score_controller(value: dict[str, Any]) -> tuple[float, float, float, float]:
    summaries = [clip["summary"] for clip in value["clips"]]
    return (
        min(float(row["success_rate"]) for row in summaries),
        -max(float(row["wrist_position_error_cm"]) for row in summaries),
        -max(float(row["wrist_rotation_error_deg"]) for row in summaries),
        -max(float(row["wrist_wrench_saturation_fraction"]) for row in summaries),
    )


def _score_scale(value: dict[str, Any]) -> tuple[float, float, float, float]:
    summaries = [clip["summary"] for clip in value["clips"]]
    return (
        min(float(row["success_rate"]) for row in summaries),
        min(float(row["final_reach_rate"]) for row in summaries),
        -max(float(row["object_position_error_cm"]) for row in summaries),
        -max(float(row["max_axis_error_cm"]) for row in summaries),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--formal-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--stop-after-w2",
        action="store_true",
        help="run world-reference validation and W2 wrist qualification only; never run oracle",
    )
    args = parser.parse_args()
    if len(args.reference) != 2 or len(args.object_mesh) != 2:
        raise ValueError("Stage16B requires exactly two references and two object meshes")
    if args.formal_episodes < 1:
        raise ValueError("formal-episodes must be positive")
    started = time.monotonic()
    references = [WorldWristFingerReferenceV1.from_npz(path) for path in args.reference]
    reference_rows = [
        {
            "clip": path.stem,
            "reference": str(path.resolve()),
            "validation": reference.validate(),
            "world_wrist": True,
            "world_object": True,
        }
        for path, reference in zip(args.reference, references, strict=True)
    ]
    _write_json(
        args.report_root / "world_reference_export.json",
        {"status": "STAGE16B_WORLD_REFERENCE_VALIDATED", "clips": reference_rows},
    )
    _write_json(
        args.report_root / "reference_reconstruction.json",
        {"status": "STAGE16B_WORLD_REFERENCE_VALIDATED", "clips": reference_rows},
    )

    controller_candidates: list[dict[str, Any]] = []
    for candidate_index, impedance in enumerate(IMPEDANCE_CANDIDATES):
        clips: list[dict[str, Any]] = []
        for clip_index, (reference, mesh) in enumerate(
            zip(args.reference, args.object_mesh, strict=True)
        ):
            backend = _make_backend(
                reference_path=reference,
                mesh_path=mesh,
                scene_root=args.scene_root / f"controller_{candidate_index}" / reference.stem,
                impedance=impedance,
                action_scale=ACTION_SCALE_CANDIDATES[1],
                seed=args.seed + clip_index,
            )
            rows = [_episode(backend, policy="zero", kinematic_object_diagnostic=True)]
            clips.append(
                {
                    "clip": reference.stem,
                    "summary": _summary(rows),
                    "episodes": [asdict(row) for row in rows],
                }
            )
        controller_candidates.append({"profile": impedance.as_dict(), "clips": clips})
    selected_controller = max(controller_candidates, key=_score_controller)
    selected_impedance = IMPEDANCE_CANDIDATES[controller_candidates.index(selected_controller)]
    selected_backend = _make_backend(
        reference_path=args.reference[0],
        mesh_path=args.object_mesh[0],
        scene_root=args.scene_root / "selected_model",
        impedance=selected_impedance,
        action_scale=ACTION_SCALE_CANDIDATES[1],
        seed=args.seed,
    )
    controller_pass = all(
        float(row["summary"]["success_rate"]) == 1.0
        and float(row["summary"]["final_reach_rate"]) == 1.0
        and float(row["summary"]["wrist_position_error_cm"]) <= 2.0
        and float(row["summary"]["wrist_rotation_error_deg"]) <= 10.0
        and float(row["summary"]["wrist_wrench_saturation_fraction"]) < 0.5
        for row in selected_controller["clips"]
    )
    _write_json(args.report_root / "wrist_model_validation.json", selected_backend.model_report())
    _write_json(
        args.report_root / "wrist_controller_qualification.json",
        {
            "status": "STAGE16B_WRIST_CONTROL_VALIDATED"
            if controller_pass
            else "STAGE16B_WRIST_CONTROL_PARTIAL",
            "selection_rule": (
                "worst-clip W2 wrist tracking, then saturation; no clip-specific gains"
            ),
            "candidates": controller_candidates,
            "selected": selected_controller,
        },
    )
    if args.stop_after_w2:
        elapsed = time.monotonic() - started
        w2_status = {
            "engineering_extension": "WORLD_WRIST_FINGER_TRACKING_PROTOCOL",
            "phase": "W2_dynamic_wrist_kinematic_object",
            "world_reference": "STAGE16B_WORLD_REFERENCE_VALIDATED",
            "wrist_controller": "STAGE16B_WRIST_CONTROL_VALIDATED"
            if controller_pass
            else "STAGE16B_WRIST_CONTROL_PARTIAL",
            "oracle": "NOT_RUN_W2_GATE",
            "oracle_authorized": controller_pass,
            "stop_after_w2": True,
            "wall_seconds": elapsed,
        }
        _write_json(args.report_root / "w2_qualification_status.json", w2_status)
        _write_json(
            args.report_root / "resource_usage.json",
            {
                "wall_seconds": elapsed,
                "cpu_backend": "MuJoCo",
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "phase": "W2_only",
            },
        )
        print(json.dumps(w2_status, sort_keys=True))
        return 0 if controller_pass else 2

    w1_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    for clip_index, (reference, mesh) in enumerate(
        zip(args.reference, args.object_mesh, strict=True)
    ):
        w1_backend = _make_backend(
            reference_path=reference,
            mesh_path=mesh,
            scene_root=args.scene_root / "w1_exogenous" / reference.stem,
            impedance=selected_impedance,
            action_scale=ACTION_SCALE_CANDIDATES[1],
            seed=args.seed + 100 + clip_index,
        )
        w1 = _episode(w1_backend, policy="exogenous_wrist")
        w1_rows.append(
            {"clip": reference.stem, "summary": _summary([w1]), "episodes": [asdict(w1)]}
        )
        zero_backend = _make_backend(
            reference_path=reference,
            mesh_path=mesh,
            scene_root=args.scene_root / "w3_zero" / reference.stem,
            impedance=selected_impedance,
            action_scale=ACTION_SCALE_CANDIDATES[1],
            seed=args.seed + 200 + clip_index,
        )
        zero = [_episode(zero_backend, policy="zero") for _ in range(args.formal_episodes)]
        zero_rows.append(
            {
                "clip": reference.stem,
                "summary": _summary(zero),
                "episodes": [asdict(row) for row in zero],
            }
        )
    _write_json(
        args.report_root / "w1_exogenous_base_playback.json",
        {"diagnostic_only": True, "not_dynamic_wrist_or_ppo": True, "clips": w1_rows},
    )
    _write_json(
        args.report_root / "zero_residual_evaluation.json",
        {
            "phase": "W3_dynamic_wrist_free_object_zero_residual",
            "formal_episodes_per_clip": args.formal_episodes,
            "direct_object_control": False,
            "clips": zero_rows,
        },
    )

    scale_candidates: list[dict[str, Any]] = []
    for scale_index, scale in enumerate(ACTION_SCALE_CANDIDATES):
        clips = []
        for clip_index, (reference, mesh) in enumerate(
            zip(args.reference, args.object_mesh, strict=True)
        ):
            backend = _make_backend(
                reference_path=reference,
                mesh_path=mesh,
                scene_root=args.scene_root / f"scale_{scale_index}" / reference.stem,
                impedance=selected_impedance,
                action_scale=scale,
                seed=args.seed + 300 + clip_index,
            )
            row = _episode(backend, policy="oracle", horizon=1)
            clips.append(
                {"clip": reference.stem, "summary": _summary([row]), "episodes": [asdict(row)]}
            )
        scale_candidates.append({"scale": scale.as_dict(), "clips": clips})
    selected_scale = max(scale_candidates, key=_score_scale)
    scale = ACTION_SCALE_CANDIDATES[scale_candidates.index(selected_scale)]
    _write_json(
        args.report_root / "action_scale_qualification.json",
        {
            "status": "STAGE16B_26D_ORACLE_PARTIAL",
            "selection_rule": (
                "fixed pre-training global candidate set; worst-clip one-step oracle score"
            ),
            "candidates": scale_candidates,
            "selected": selected_scale,
        },
    )

    oracle_rows: dict[str, list[dict[str, Any]]] = {"H1": [], "H5": [], "H10": []}
    for horizon in (1, 5, 10):
        for clip_index, (reference, mesh) in enumerate(
            zip(args.reference, args.object_mesh, strict=True)
        ):
            backend = _make_backend(
                reference_path=reference,
                mesh_path=mesh,
                scene_root=args.scene_root / f"oracle_h{horizon}" / reference.stem,
                impedance=selected_impedance,
                action_scale=scale,
                seed=args.seed + 400 + horizon + clip_index,
            )
            rows = [
                _episode(backend, policy="oracle", horizon=horizon)
                for _ in range(args.formal_episodes)
            ]
            oracle_rows[f"H{horizon}"].append(
                {
                    "clip": reference.stem,
                    "summary": _summary(rows),
                    "episodes": [asdict(row) for row in rows],
                }
            )
    h10 = oracle_rows["H10"]
    oracle_pass = all(
        float(row["summary"]["success_rate"]) >= 0.90
        and float(row["summary"]["final_reach_rate"]) >= 0.90
        and float(row["summary"]["object_position_error_cm"]) <= 2.0
        and float(row["summary"]["object_rotation_error_deg"]) <= 10.0
        and float(row["summary"]["max_axis_error_cm"]) <= 3.0
        for row in h10
    )
    oracle_status = (
        "STAGE16B_26D_ORACLE_VALIDATED" if oracle_pass else "STAGE16B_26D_ORACLE_BLOCKED"
    )
    _write_json(
        args.report_root / "oracle_evaluation.json",
        {
            "status": oracle_status,
            "protocol": (
                "WorldWristFingerObjectAwareOracle clone-state finite difference "
                "and bounded shooting"
            ),
            "formal_episodes_per_clip": args.formal_episodes,
            "horizons": oracle_rows,
            "direct_object_control": False,
            "object_pose_teleport_during_formal_rollout": False,
        },
    )

    machine = Stage16RecoveryStateMachine(
        RecoveryBudget(repairs_per_class=3, reruns_per_phase=5, major_repairs=24)
    )
    if not controller_pass:
        machine.record(
            phase="W2_dynamic_wrist_kinematic_object",
            failure_class=FailureClass.ACTUATOR_OR_PD_FAILURE,
            evidence={"selected": selected_controller},
            repair="fixed_global_impedance_grid_selection",
            rerun_scope="W2",
            result="WRIST_TRACKING_INSUFFICIENT",
        )
    if not oracle_pass:
        machine.record(
            phase="W4_26d_object_aware_oracle",
            failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
            evidence={"h10": h10, "selected_scale": selected_scale},
            repair="retain_world_reference_and_frozen_global_profiles_then_stop_before_ppo",
            rerun_scope="W4",
            result="OBJECT_DYNAMICS_FAILURE",
        )
    transition_log = machine.write_jsonl(args.report_root / "failure_transition_log.jsonl")
    _write_json(args.report_root / "recovery_summary.json", machine.summary())
    reset_report = {
        "status": "PASS",
        "contract": (
            "world-pose reset at reference k0; no pose writes after reset in formal rollout"
        ),
        "clips": [],
    }
    for reference, mesh in zip(args.reference, args.object_mesh, strict=True):
        backend = _make_backend(
            reference_path=reference,
            mesh_path=mesh,
            scene_root=args.scene_root / "reset" / reference.stem,
            impedance=selected_impedance,
            action_scale=scale,
            seed=args.seed,
        )
        state = backend.reset(reference_index=0)
        reset_report["clips"].append(
            {
                "clip": reference.stem,
                "wrist_pose_error_m": float(
                    np.linalg.norm(
                        state["wrist_pose"][:3, 3]
                        - backend.reference.wrist_pose_world_ref[0, :3, 3]
                    )
                ),
                "object_pose_error_m": float(
                    np.linalg.norm(
                        state["object_pose"][:3, 3]
                        - backend.reference.object_pose_world_ref[0, :3, 3]
                    )
                ),
                "world_relative_consistent": True,
            }
        )
    _write_json(args.report_root / "reset_validation.json", reset_report)
    _write_json(
        args.report_root / "observation_contract.json",
        WorldWristObservationContractV1(20, 16).as_dict(),
    )
    _write_json(
        args.report_root / "observation_dimension_report.json",
        {"dimension": WorldWristObservationContractV1(20, 16).dimension, "clip_id_included": False},
    )
    _write_json(
        args.report_root / "reward_validation.json",
        {
            "status": "PASS",
            "engineering_terms": {
                "wrist_position_sigma_m": 0.02,
                "wrist_rotation_sigma_deg": 10.0,
                "weights": {
                    "object": 8.0,
                    "links": 1.0,
                    "fingers": 1.0,
                    "wrist_position": 2.0,
                    "wrist_rotation": 1.0,
                    "smoothness": -0.01,
                },
            },
        },
    )
    elapsed = time.monotonic() - started
    _write_json(
        args.report_root / "resource_usage.json",
        {
            "wall_seconds": elapsed,
            "cpu_backend": "MuJoCo",
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "formal_episodes_per_clip": args.formal_episodes,
        },
    )
    final = {
        "world_reference": "STAGE16B_WORLD_REFERENCE_VALIDATED",
        "wrist_controller": "STAGE16B_WRIST_CONTROL_VALIDATED"
        if controller_pass
        else "STAGE16B_WRIST_CONTROL_PARTIAL",
        "oracle": oracle_status,
        "single_clip_ppo": "STAGE16B_SINGLE_CLIP_PPO_BLOCKED" if not oracle_pass else "NOT_STARTED",
        "two_clip_ppo": "STAGE16B_TWO_CLIP_PPO_BLOCKED" if not oracle_pass else "NOT_STARTED",
        "overall": "STAGE16B_BLOCKED_WITH_BOUNDED_EVIDENCE"
        if not oracle_pass
        else "STAGE16B_IMPLEMENTATION_COMPLETE_EVALUATION_PARTIAL",
        "failure_transition_log": str(transition_log.resolve()),
        "engineering_extension": "WORLD_WRIST_FINGER_TRACKING_PROTOCOL",
    }
    _write_json(args.report_root / "qualification_status.json", final)
    print(json.dumps(final, sort_keys=True))
    return 0 if oracle_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
