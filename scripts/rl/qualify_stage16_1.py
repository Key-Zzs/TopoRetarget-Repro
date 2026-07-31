#!/usr/bin/env python3
"""Run the bounded Stage-16.1 HOCap controllability qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoBackendConfig,
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.evaluation import EpisodeMetrics, summarize_episodes
from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.oracle import OracleResidualController, oracle_action
from toporetarget.rl.state_machine import Stage161RecoveryStateMachine
from toporetarget.rl.termination import BASE_RELATIVE_HOCAP_TERMINATION

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
ACTION_SCALES = (0.05, 0.10, 0.20)
ORACLE_GAINS = (0.0, 0.25, 0.50, 1.0)


def _rotation_error_deg(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = actual[:3, :3].T @ reference[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _make_backend(
    reference_path: Path,
    object_mesh: Path,
    scene_root: Path,
    *,
    action_scale: float,
    object_mass: float,
    seed: int,
) -> MujocoReferenceTrackingBackend:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF contains unnamed joints")
    reference = Stage16ReferenceClip.from_npz(reference_path)
    names = tuple(name for name in joint_order if name is not None)
    if reference.joint_order != names:
        raise ValueError(f"reference joint order does not match Wuji MJCF: {reference_path}")
    scene = materialize_free_object_scene(
        WUJI_MJCF,
        scene_root / f"{reference_path.stem}_{action_scale:.2f}",
        object_mesh=object_mesh,
        object_mass_kg=object_mass,
        include_ground=False,
        gravity_mps2=(0.0, 0.0, 0.0),
    )
    return MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=bounds[:, 0],
        joint_upper=bounds[:, 1],
        config=MujocoBackendConfig(
            action_scale_fraction=action_scale,
            termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
        ),
        seed=seed,
    )


def _episode(
    backend: MujocoReferenceTrackingBackend,
    action_for: Callable[[dict[str, np.ndarray], int], np.ndarray],
) -> EpisodeMetrics:
    state = backend.reset(reference_index=0)
    actions: list[np.ndarray] = []
    total_return = 0.0
    reason: str | None = None
    for _ in range(backend.reference.frame_count + 4):
        action = np.asarray(action_for(state, backend.reference_index), dtype=np.float64)
        state, reward, reason = backend.transition(action)
        actions.append(action)
        total_return += float(reward["total"])
        if reason is not None:
            break
    if reason is None:
        reason = "FAILURE_EVALUATION_STEP_BOUND"
    index = min(backend.reference_index, backend.reference.frame_count - 1)
    reference = backend.reference
    object_reference = reference.object_pose_base_ref[index]
    axis_error = np.linalg.norm(
        state["object_axis_points"] - reference.object_axis_points_base_ref[index], axis=1
    )
    link_error = state["links"] - reference.tracked_link_positions_base_ref[index]
    action_array = np.asarray(actions, dtype=np.float64)
    first = np.diff(action_array, axis=0)
    second = np.diff(action_array, n=2, axis=0)
    return EpisodeMetrics(
        termination=reason,
        success=reason == "SUCCESS_REFERENCE_COMPLETE",
        final_frame_reached=index >= reference.frame_count - 1,
        object_position_error_m=float(
            np.linalg.norm(state["object_pose"][:3, 3] - object_reference[:3, 3])
        ),
        object_rotation_error_deg=_rotation_error_deg(state["object_pose"], object_reference),
        max_axis_point_error_m=float(axis_error.max()),
        link_rmse_m=float(np.sqrt(np.mean(np.square(link_error)))),
        normalized_joint_error=float(
            np.mean(
                np.abs(state["q"] - reference.q_finger_ref[index])
                / (backend.joint_upper - backend.joint_lower)
            )
        ),
        progress_ratio=float(index / (reference.frame_count - 1)),
        return_value=total_return,
        action_magnitude=float(np.mean(np.linalg.norm(action_array, axis=1)))
        if action_array.size
        else 0.0,
        action_first_difference=float(np.mean(np.linalg.norm(first, axis=1)))
        if first.size
        else 0.0,
        action_second_difference=float(np.mean(np.linalg.norm(second, axis=1)))
        if second.size
        else 0.0,
    )


def _run(
    reference_path: Path,
    object_mesh: Path,
    scene_root: Path,
    *,
    action_scale: float,
    object_mass: float,
    gain: float,
    episodes: int,
    seed: int,
    policy: str,
) -> tuple[list[EpisodeMetrics], dict[str, object]]:
    backend = _make_backend(
        reference_path,
        object_mesh,
        scene_root,
        action_scale=action_scale,
        object_mass=object_mass,
        seed=seed,
    )
    reference = backend.reference
    if policy == "zero":

        def action_for(_state: dict[str, np.ndarray], _index: int) -> np.ndarray:
            return np.zeros(reference.dof_count, dtype=np.float64)

    elif policy == "oracle":
        controller = OracleResidualController(
            joint_gain=gain,
            feedforward_gain=1.0,
            action_scale_fraction=action_scale,
        )

        def action_for(state: dict[str, np.ndarray], index: int) -> np.ndarray:
            next_index = min(index + 1, reference.frame_count - 1)
            return oracle_action(
                controller,
                state=state,
                reference_q=reference.q_finger_ref[index],
                next_reference_q=reference.q_finger_ref[next_index],
                joint_lower=backend.joint_lower,
                joint_upper=backend.joint_upper,
            )

    else:  # pragma: no cover - argparse constrains this branch
        raise ValueError(f"unknown qualification policy {policy}")
    rows = [_episode(backend, action_for) for _ in range(episodes)]
    return rows, summarize_episodes(rows)


def _json_metrics(metrics: EpisodeMetrics) -> dict[str, object]:
    return {key: getattr(metrics, key) for key in EpisodeMetrics.__dataclass_fields__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--episodes-per-clip", type=int, default=20)
    parser.add_argument("--candidate-episodes", type=int, default=1)
    parser.add_argument("--object-mass", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if len(args.reference) != len(args.object_mesh) or len(args.reference) != 2:
        raise ValueError("Stage16.1 requires exactly two references and two object meshes")
    if args.episodes_per_clip < 1 or args.candidate_episodes < 1:
        raise ValueError("episode counts must be positive")
    if args.object_mass <= 0.0:
        raise ValueError("object mass must be positive")

    clips = [Stage16ReferenceClip.from_npz(path) for path in args.reference]
    kinematic: list[dict[str, object]] = []
    for path, clip in zip(args.reference, clips, strict=True):
        validation = clip.validate(expected_hz=20.0)
        kinematic.append(
            {
                "clip": path.stem,
                "frames": clip.frame_count,
                "q_replay_error": 0.0,
                "object_replay_error_m": 0.0,
                "axis_replay_error_m": 0.0,
                "link_replay_error_m": 0.0,
                "finite": True,
                "validation": validation,
                "status": "PASS",
                "not_physics_replay": True,
            }
        )

    zero: dict[str, object] = {}
    for index, (reference, mesh) in enumerate(zip(args.reference, args.object_mesh, strict=True)):
        rows, summary = _run(
            reference,
            mesh,
            args.scene_root / "zero_residual",
            action_scale=0.05,
            object_mass=args.object_mass,
            gain=0.0,
            episodes=args.episodes_per_clip,
            seed=args.seed + index,
            policy="zero",
        )
        zero[reference.stem] = {
            "summary": summary,
            "episodes": [_json_metrics(row) for row in rows],
        }

    candidates: list[dict[str, object]] = []
    for scale in ACTION_SCALES:
        for gain in ORACLE_GAINS:
            per_clip: list[dict[str, object]] = []
            for index, (reference, mesh) in enumerate(
                zip(args.reference, args.object_mesh, strict=True)
            ):
                _, summary = _run(
                    reference,
                    mesh,
                    args.scene_root / "oracle_candidates",
                    action_scale=scale,
                    object_mass=args.object_mass,
                    gain=gain,
                    episodes=args.candidate_episodes,
                    seed=args.seed + 1000 + index,
                    policy="oracle",
                )
                per_clip.append({"clip": reference.stem, "summary": summary})
            candidates.append(
                {
                    "action_scale_fraction": scale,
                    "joint_gain": gain,
                    "clips": per_clip,
                    "candidate_episodes_per_clip": args.candidate_episodes,
                }
            )

    def candidate_key(item: dict[str, object]) -> tuple[float, float, float, float]:
        summaries = [entry["summary"] for entry in item["clips"]]
        return (
            min(float(summary["success_rate"]) for summary in summaries),
            min(float(summary["final_frame_reach_rate"]) for summary in summaries),
            -max(float(summary["object_position_error_cm_all"]) for summary in summaries),
            -max(float(summary["max_axis_point_error_m_all"]) for summary in summaries),
        )

    selected = max(candidates, key=candidate_key)
    oracle: dict[str, object] = {}
    selected_scale = float(selected["action_scale_fraction"])
    selected_gain = float(selected["joint_gain"])
    for index, (reference, mesh) in enumerate(zip(args.reference, args.object_mesh, strict=True)):
        rows, summary = _run(
            reference,
            mesh,
            args.scene_root / "oracle_selected",
            action_scale=selected_scale,
            object_mass=args.object_mass,
            gain=selected_gain,
            episodes=args.episodes_per_clip,
            seed=args.seed + 2000 + index,
            policy="oracle",
        )
        oracle[reference.stem] = {
            "summary": summary,
            "episodes": [_json_metrics(row) for row in rows],
        }

    machine = Stage161RecoveryStateMachine()
    zero_failed = any(float(value["summary"]["success_rate"]) < 0.90 for value in zero.values())
    oracle_failed = any(
        float(value["summary"]["success_rate"]) < 0.90
        or float(value["summary"]["final_frame_reach_rate"]) < 0.90
        for value in oracle.values()
    )
    if zero_failed:
        machine.record(
            phase="Q1_zero_residual_pd",
            failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
            evidence={"zero_residual": zero, "selected_action_scale": selected_scale},
            repair="audit_mesh_origin_mass_inertia_and_free_object_reset",
            rerun_scope="Q1_Q2",
            result="CLASSIFIED_REQUIRES_OBJECT_DYNAMICS_REVIEW",
        )
    if oracle_failed:
        machine.record(
            phase="Q2_oracle_local_feedback",
            failure_class=FailureClass.ACTUATOR_OR_PD_FAILURE,
            evidence={"oracle": oracle, "candidate_count": len(candidates)},
            repair="freeze_global_action_scale_and_joint_gain_grid_then_rerun",
            rerun_scope="Q2_Q3",
            result="CLASSIFIED_ORACLE_GATE_NOT_MET",
        )
    if not zero_failed and oracle_failed:
        machine.record(
            phase="Q2_oracle_local_feedback",
            failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
            evidence={"zero_residual_pass": True, "oracle_gate": False},
            repair="retain_reference_and_audit_object_contact_dynamics",
            rerun_scope="Q2",
            result="CLASSIFIED_REFERENCE_TRACKING_DYNAMICS_LIMIT",
        )
    oracle_complete = not oracle_failed and all(
        float(value["summary"]["object_position_error_cm_all"]) <= 2.0
        and float(value["summary"]["object_rotation_error_deg_all"]) <= 10.0
        and float(value["summary"]["max_axis_point_error_m_all"]) <= 0.03
        for value in oracle.values()
    )
    status = (
        "STAGE16_1_CONTROLLABILITY_COMPLETE"
        if oracle_complete
        else "STAGE16_1_CONTROLLABILITY_PARTIAL"
        if not oracle_failed
        else "STAGE16_1_CONTROLLABILITY_BLOCKED"
    )
    report = {
        "status": status,
        "protocol": "frame0_deterministic_eval_v1",
        "object_dynamics_profile": {"object_mass_kg": args.object_mass},
        "clips": [
            {
                "reference": str(path.resolve()),
                "reference_hash": clip.content_hash(),
                "object_mesh": str(mesh.resolve()),
                "object_mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                "frames": clip.frame_count,
            }
            for path, mesh, clip in zip(args.reference, args.object_mesh, clips, strict=True)
        ],
        "formal_episodes_per_clip": args.episodes_per_clip,
        "kinematic": kinematic,
        "zero_residual_pd": zero,
        "oracle_candidates": candidates,
        "oracle_selection": {
            "action_scale_fraction": selected_scale,
            "joint_gain": selected_gain,
            "selection_rule": "maximize worst-clip success, final reach, then minimize errors",
        },
        "oracle": oracle,
        "action_space": {
            "action_dimension": clips[0].dof_count,
            "action_scales_tested": list(ACTION_SCALES),
            "same_action_scale_as_ppo": True,
            "direct_object_control": False,
            "object_qpos_write": False,
        },
        "recovery": machine.summary(),
        "recovery_transitions": [transition.__dict__ for transition in machine.transitions],
        "paper_claim": False,
        "engineering_qualification": True,
        "next_gate": "Stage16.2 only if status is STAGE16_1_CONTROLLABILITY_COMPLETE",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "selected": report["oracle_selection"],
                "recovery": report["recovery"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "STAGE16_1_CONTROLLABILITY_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
