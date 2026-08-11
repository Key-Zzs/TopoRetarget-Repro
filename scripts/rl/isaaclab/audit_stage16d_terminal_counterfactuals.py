#!/usr/bin/env python3
"""Run bounded, action-preserving Stage 16-D terminal-dynamics counterfactuals.

Each invocation owns one fresh Isaac process and writes its receipt before
process shutdown. Isaac Sim 5.1 can terminate the host process from
``app.close()``; after environment cleanup this script deliberately lets its
fresh subprocess exit normally instead of calling ``app.close()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: np.ndarray, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} expected {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    array = _finite(values, name="summary")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("summary requires a non-empty vector")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "final": float(array[-1]),
    }


def _rotation_error_deg(actual_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    actual = _finite(actual_wxyz, name="actual quaternion")
    reference = _finite(reference_wxyz, name="reference quaternion")
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    dot = np.abs((actual * reference).sum(axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _selection(
    trace_path: Path, qualification_path: Path
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    qualification = _read_json(qualification_path)
    with np.load(trace_path, allow_pickle=False) as trace:
        pose = _finite(trace["replica_object_pose"], name="replica_object_pose", shape=(321, 20, 7))
        twist = _finite(
            trace["replica_object_twist"], name="replica_object_twist", shape=(321, 20, 6)
        )
        actions = _finite(trace["replica_action"], name="replica_action", shape=(321, 20, 26))
        contact = np.asarray(trace["replica_contact_pair_presence"], dtype=bool).any(axis=-1)
    episodes = qualification["episodes"]
    if len(episodes) != 20:
        raise ValueError("counterfactual requires all 20 frozen formal episodes")
    terminal_score = np.linalg.norm(twist[-100:, :, :3], axis=-1).mean(axis=0) + np.linalg.norm(
        twist[-100:, :, 3:], axis=-1
    ).mean(axis=0)
    stable = [int(row["replica"]) for row in episodes if bool(row["terminal_stability_pass"])]
    failed = [int(row["replica"]) for row in episodes if not bool(row["terminal_stability_pass"])]
    stable_ish = (
        min(stable, key=lambda replica: terminal_score[replica])
        if stable
        else int(np.argmin(terminal_score))
    )
    failed_representative = max(failed, key=lambda replica: terminal_score[replica])
    if failed_representative == stable_ish:
        alternatives = [replica for replica in failed if replica != stable_ish]
        if not alternatives:
            raise ValueError("cannot select distinct stable-ish and failed formal replicas")
        failed_representative = max(alternatives, key=lambda replica: terminal_score[replica])
    rows: list[dict[str, Any]] = []
    for label, replica in (("stable_ish", stable_ish), ("failed", failed_representative)):
        episode = episodes[replica]
        contact_frames = np.flatnonzero(contact[:, replica])
        if not contact_frames.size:
            raise ValueError(
                f"formal replica {replica} has no contact for free-drift initialization"
            )
        rows.append(
            {
                "label": label,
                "replica": replica,
                "seed": int(episode["seed"]),
                "formal_terminal_stability": bool(episode["terminal_stability_pass"]),
                "terminal_score": float(terminal_score[replica]),
                "last_contact_frame": int(contact_frames[-1]),
                "action_sha256": hashlib.sha256(
                    actions[:, replica].astype(np.float32).tobytes()
                ).hexdigest(),
            }
        )
    return qualification, pose, twist, actions, contact, rows


def _frozen_nominal(
    *,
    pose: np.ndarray,
    twist: np.ndarray,
    contact: np.ndarray,
    reference_pose: np.ndarray,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for selection in rows:
        replica = int(selection["replica"])
        actual = pose[:, replica]
        linear = np.linalg.norm(twist[:, replica, :3], axis=-1)
        angular = np.linalg.norm(twist[:, replica, 3:], axis=-1)
        translation = np.linalg.norm(actual[:, :3] - reference_pose[:, :3], axis=-1) * 100.0
        rotation = _rotation_error_deg(actual[:, 3:], reference_pose[:, 3:])
        timeline = [
            {
                "frame": int(frame),
                "time_s": float(frame * 0.05),
                "object_z_m": float(actual[frame, 2]),
                "linear_speed_mps": float(linear[frame]),
                "angular_speed_radps": float(angular[frame]),
                "contact": bool(contact[frame, replica]),
                "tracking_translation_cm": float(translation[frame]),
                "tracking_rotation_deg": float(rotation[frame]),
            }
            for frame in range(321)
        ]
        output.append(
            {
                **selection,
                "timeline": timeline,
                "summary": {
                    "object_z_m": _summary(actual[:, 2]),
                    "linear_speed_mps": _summary(linear),
                    "angular_speed_radps": _summary(angular),
                    "tracking_translation_cm": _summary(translation),
                    "tracking_rotation_deg": _summary(rotation),
                    "contact_steps": int(contact[:, replica].sum()),
                },
            }
        )
    return output


def _app() -> Any:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True).app


def _apply_gravity_only(cfg: Any) -> None:
    """Apply the single permitted CF1/CF2/CF3 physics change in place."""

    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.object_170105.spawn.rigid_props.disable_gravity = False
    cfg.object_170650.spawn.rigid_props.disable_gravity = False


def _configure_env(*, clip: str, count: int, gravity: bool) -> tuple[Any, Any]:
    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )

    cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo_cfg.configure_stage16d_ppo26d(cfg, num_envs=count, clip=clip, rsi=False, critical_dr=False)
    if gravity:
        _apply_gravity_only(cfg)
    return cfg, IsaacPPO26DReferenceTrackingEnv(cfg)


def _active_object(env: Any, clip: str) -> Any:
    return env._object_170105 if clip == "hocap_170105" else env._object_170650


def _close(env: Any | None) -> None:
    if env is not None:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def _gravity_replay(
    args: argparse.Namespace,
    *,
    selections: list[dict[str, Any]],
    actions: np.ndarray,
    reference_pose: np.ndarray,
) -> dict[str, Any]:
    _app()
    env: Any | None = None
    try:
        import torch

        cfg, env = _configure_env(clip=args.clip, count=len(selections), gravity=True)
        env.reset(seed=20260811)
        selection_replicas = [int(row["replica"]) for row in selections]
        selected_actions = torch.as_tensor(
            actions[:, selection_replicas], device=env.device, dtype=torch.float32
        )
        records: list[list[dict[str, Any]]] = [[] for _ in selections]
        active = np.ones(len(selections), dtype=bool)
        for control_step in range(selected_actions.shape[0]):
            before = env._state()
            before_position = before["object_position_scene"].detach().cpu().numpy()
            before_quaternion = before["object_quaternion_wxyz"].detach().cpu().numpy()
            _, _, terminated, timed_out, extras = env.step(selected_actions[control_step])
            ended = (terminated | timed_out).detach().cpu().numpy().astype(bool)
            state = env._state()
            position = state["object_position_scene"].detach().cpu().numpy()
            quaternion = state["object_quaternion_wxyz"].detach().cpu().numpy()
            contact = extras["ppo26d"]["contact_any"].detach().cpu().numpy().astype(bool)
            linear_speed = extras["ppo26d"]["object_linear_speed_mps"].detach().cpu().numpy()
            angular_speed = extras["ppo26d"]["object_angular_speed_radps"].detach().cpu().numpy()
            reason_code = extras["ppo26d"]["primary_reason_code"].detach().cpu().numpy()
            reason_labels = extras["ppo26d"]["termination_reasons"]
            reference_index = min(control_step + 1, reference_pose.shape[0] - 1)
            visible_position = np.where(ended[:, None], before_position, position)
            visible_quaternion = np.where(ended[:, None], before_quaternion, quaternion)
            translation = (
                np.linalg.norm(visible_position - reference_pose[reference_index, :3], axis=-1)
                * 100.0
            )
            rotation = _rotation_error_deg(visible_quaternion, reference_pose[reference_index, 3:])
            for index in range(len(selections)):
                if not active[index]:
                    continue
                records[index].append(
                    {
                        "frame": reference_index,
                        "time_s": float(reference_index * 0.05),
                        "object_z_m": float(visible_position[index, 2]),
                        "linear_speed_mps": float(linear_speed[index]),
                        "angular_speed_radps": float(angular_speed[index]),
                        "contact": bool(contact[index]),
                        "tracking_translation_cm": float(translation[index]),
                        "tracking_rotation_deg": float(rotation[index]),
                        "terminated": bool(terminated[index]),
                        "timed_out": bool(timed_out[index]),
                        "termination_reason": str(reason_labels[int(reason_code[index])]),
                        "post_step_pose_available": bool(not ended[index]),
                    }
                )
            active &= ~ended
            if not active.any():
                break
        trajectories = []
        for selection, timeline in zip(selections, records, strict=True):
            trajectories.append(
                {
                    **selection,
                    "timeline": timeline,
                    "summary": {
                        key: _summary(np.asarray([row[key] for row in timeline], dtype=np.float64))
                        for key in (
                            "object_z_m",
                            "linear_speed_mps",
                            "angular_speed_radps",
                            "tracking_translation_cm",
                            "tracking_rotation_deg",
                        )
                    }
                    | {
                        "contact_steps": sum(row["contact"] for row in timeline),
                        "completed_saved_action_steps": len(timeline),
                        "ended_early": bool(
                            timeline[-1]["terminated"] or timeline[-1]["timed_out"]
                        ),
                        "final_termination_reason": timeline[-1]["termination_reason"],
                    },
                }
            )
        return {
            "schema_version": "Stage16DTerminalCounterfactualV1",
            "case": "CF1_SAME_ACTION_GRAVITY_ONLY",
            "clip": args.clip,
            "gravity": [0.0, 0.0, -9.81],
            "action_steps": int(selected_actions.shape[0]),
            "same_saved_26d_actions": True,
            "physics_contract": {
                "gravity_changed_only": True,
                "damping_changed": False,
                "controller_changed": False,
                "contact_geometry_changed": False,
                "policy_retrained": False,
            },
            "trajectories": trajectories,
        }
    finally:
        _close(env)


def _free_drift(
    args: argparse.Namespace,
    *,
    selections: list[dict[str, Any]],
    pose: np.ndarray,
    twist: np.ndarray,
    reference_pose: np.ndarray,
    reference_twist: np.ndarray,
) -> dict[str, Any]:
    _app()
    env: Any | None = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.scene_frame import scene_to_global

        initial_kind = args.initial_state
        if initial_kind == "last_contact":
            frames = [int(row["last_contact_frame"]) for row in selections]
            initial_pose = np.stack(
                [
                    pose[frame, int(row["replica"])]
                    for frame, row in zip(frames, selections, strict=True)
                ]
            )
            initial_twist = np.stack(
                [
                    twist[frame, int(row["replica"])]
                    for frame, row in zip(frames, selections, strict=True)
                ]
            )
        elif initial_kind == "reference_terminal":
            frames = [320 for _ in selections]
            initial_pose = np.repeat(reference_pose[-1:, :], len(selections), axis=0)
            initial_twist = np.repeat(reference_twist[-1:, :], len(selections), axis=0)
        else:
            raise ValueError(f"unknown initial state: {initial_kind}")
        cfg, env = _configure_env(clip=args.clip, count=len(selections), gravity=args.gravity)
        env.reset(seed=20260811)
        object_state = torch.as_tensor(
            np.concatenate((initial_pose, initial_twist), axis=-1),
            device=env.device,
            dtype=torch.float32,
        )
        object_state[:, :3] = scene_to_global(object_state[:, :3], env.scene.env_origins)
        _active_object(env, args.clip).write_root_state_to_sim(
            object_state, env_ids=torch.arange(len(selections), device=env.device, dtype=torch.long)
        )
        initial_position = initial_pose[:, :3].copy()
        timeline: list[list[dict[str, Any]]] = [
            [
                {
                    "physics_step": 0,
                    "time_s": 0.0,
                    "object_z_m": float(initial_position[index, 2]),
                    "linear_speed_mps": float(np.linalg.norm(initial_twist[index, :3])),
                    "angular_speed_radps": float(np.linalg.norm(initial_twist[index, 3:])),
                }
            ]
            for index in range(len(selections))
        ]
        for physics_step in range(1, 121):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            state = env._state()
            position = state["object_position_scene"].detach().cpu().numpy()
            current_twist = state["object_twist_world"].detach().cpu().numpy()
            for index in range(len(selections)):
                timeline[index].append(
                    {
                        "physics_step": physics_step,
                        "time_s": float(physics_step * env.physics_dt),
                        "object_z_m": float(position[index, 2]),
                        "linear_speed_mps": float(np.linalg.norm(current_twist[index, :3])),
                        "angular_speed_radps": float(np.linalg.norm(current_twist[index, 3:])),
                    }
                )
        trajectories = []
        for selection, rows in zip(selections, timeline, strict=True):
            z = np.asarray([row["object_z_m"] for row in rows], dtype=np.float64)
            linear = np.asarray([row["linear_speed_mps"] for row in rows], dtype=np.float64)
            angular = np.asarray([row["angular_speed_radps"] for row in rows], dtype=np.float64)
            trajectories.append(
                {
                    **selection,
                    "initial_reference_frame": frames[len(trajectories)],
                    "timeline": rows,
                    "summary": {
                        "object_z_m": _summary(z),
                        "linear_speed_mps": _summary(linear),
                        "angular_speed_radps": _summary(angular),
                        "vertical_displacement_m": float(z[-1] - z[0]),
                    },
                }
            )
        return {
            "schema_version": "Stage16DTerminalCounterfactualV1",
            "case": "CF2_TERMINAL_FREE_DRIFT"
            if initial_kind == "last_contact"
            else "CF3_REFERENCE_TERMINAL_STATE_FREE_DRIFT",
            "clip": args.clip,
            "gravity": [0.0, 0.0, -9.81] if args.gravity else [0.0, 0.0, 0.0],
            "initial_state": initial_kind,
            "physics_steps": 120,
            "duration_s": 1.0,
            "damping": "unchanged_nominal_zero",
            "reset_time_object_state_initialization_calls": 1,
            "post_initialization_object_state_write_calls": 0,
            "guidance": False,
            "attachment": False,
            "trajectories": trajectories,
        }
    finally:
        _close(env)


def _write_result(args: argparse.Namespace, value: dict[str, Any]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case": value["case"], "output": str(args.output.resolve())}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=("gravity_replay", "free_drift"), required=True)
    parser.add_argument("--initial-state", choices=("last_contact", "reference_terminal"))
    parser.add_argument("--gravity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.mode == "gravity_replay" and not args.gravity:
        raise ValueError("CF1 gravity replay requires --gravity")
    if args.mode == "free_drift" and args.initial_state is None:
        raise ValueError("free drift requires --initial-state")
    qualification, pose, twist, actions, contact, selections = _selection(
        args.trace.resolve(), args.qualification.resolve()
    )
    with np.load(args.reference.resolve(), allow_pickle=False) as archive:
        reference_pose = np.concatenate(
            (
                _finite(archive["object_pose_translation_world_ref"], name="reference position"),
                _finite(
                    archive["object_pose_quaternion_world_ref_wxyz"], name="reference quaternion"
                ),
            ),
            axis=-1,
        )
        reference_twist = _finite(archive["object_twist_world_ref"], name="reference twist")
    if reference_pose.shape != (321, 7) or reference_twist.shape != (321, 6):
        raise ValueError("counterfactual requires 321-sample factor-8 reference")
    common = {
        "frozen_inputs": {
            "trace": str(args.trace.resolve()),
            "trace_sha256": _sha256(args.trace.resolve()),
            "qualification": str(args.qualification.resolve()),
            "qualification_sha256": _sha256(args.qualification.resolve()),
            "reference": str(args.reference.resolve()),
            "reference_sha256": _sha256(args.reference.resolve()),
        },
        "formal_qualification_status": qualification["status"],
        "selected_representatives": selections,
    }
    if args.mode == "gravity_replay":
        result = _gravity_replay(
            args, selections=selections, actions=actions, reference_pose=reference_pose
        )
    else:
        result = _free_drift(
            args,
            selections=selections,
            pose=pose,
            twist=twist,
            reference_pose=reference_pose,
            reference_twist=reference_twist,
        )
    _write_result(args, {**result, **common})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
