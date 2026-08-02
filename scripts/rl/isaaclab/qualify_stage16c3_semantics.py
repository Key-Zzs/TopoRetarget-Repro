#!/usr/bin/env python3
"""Run the bounded Stage 16-C.3 semantic qualification on real PhysX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--wrist-controller-mode",
        choices=("wrist_impedance_v1", "computed_wrench_v2", "effective_dynamics_v3"),
        default="effective_dynamics_v3",
    )
    parser.add_argument("--force-limit-n", type=float)
    parser.add_argument("--torque-limit-nm", type=float)
    parser.add_argument("--translation-position-gain-s2", type=float)
    parser.add_argument("--rotation-position-gain-s2", type=float)
    parser.add_argument("--v1-translation-stiffness-npm", type=float)
    parser.add_argument("--v1-rotation-stiffness-nmprad", type=float)
    parser.add_argument("--v1-translation-damping-ratio", type=float)
    parser.add_argument("--v1-rotation-damping-ratio", type=float)
    parser.add_argument("--v1-force-limit-n", type=float)
    parser.add_argument("--v1-torque-limit-nm", type=float)
    parser.add_argument("--collect-wrist-diagnostics", action="store_true")
    parser.add_argument("--collect-contact-telemetry", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c2_c5_isaaclab/c3_semantic_qualification.json",
    )
    return parser.parse_args()


def tensor_value(value: Any) -> float:
    return float(value.detach().cpu().item())


def run_clip(
    env, torch, *, clip_index: int, diagnostic_kinematic_object: bool, steps: int
) -> dict[str, Any]:
    env.cfg.diagnostic_kinematic_object = diagnostic_kinematic_object
    env._clip_index[:] = clip_index
    env._reference_index.zero_()
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    action = torch.zeros((env.num_envs, 26), device=env.device)
    maxima = {
        "wrist_position_m": 0.0,
        "wrist_rotation_deg": 0.0,
        "object_position_m": 0.0,
        "object_axis_m": 0.0,
        "object_rotation_deg": 0.0,
        "force_saturation_ratio": 0.0,
        "torque_saturation_ratio": 0.0,
    }
    squared_errors = {"wrist_position_m": 0.0, "wrist_rotation_deg": 0.0}
    failures: dict[str, int] = {}
    completed = 0
    for _ in range(steps):
        _, _, terminated, truncated, extras = env.step(action)
        values = extras["stage16"]
        maxima["wrist_position_m"] = max(
            maxima["wrist_position_m"], tensor_value(values["wrist_position_error_m"].amax())
        )
        maxima["wrist_rotation_deg"] = max(
            maxima["wrist_rotation_deg"],
            tensor_value(values["wrist_orientation_error_rad"].amax()) * 180.0 / 3.141592653589793,
        )
        squared_errors["wrist_position_m"] += tensor_value(
            values["wrist_position_error_m"].square().mean()
        )
        squared_errors["wrist_rotation_deg"] += (
            tensor_value(values["wrist_orientation_error_rad"].square().mean())
            * (180.0 / 3.141592653589793) ** 2
        )
        maxima["force_saturation_ratio"] = max(
            maxima["force_saturation_ratio"],
            tensor_value(values["force_saturation_ratio"].amax()),
        )
        maxima["torque_saturation_ratio"] = max(
            maxima["torque_saturation_ratio"],
            tensor_value(values["torque_saturation_ratio"].amax()),
        )
        maxima["object_position_m"] = max(
            maxima["object_position_m"], tensor_value(values["object_position_error_m"].amax())
        )
        maxima["object_axis_m"] = max(
            maxima["object_axis_m"], tensor_value(values["object_axis_error_m"].amax())
        )
        maxima["object_rotation_deg"] = max(
            maxima["object_rotation_deg"],
            tensor_value(values["object_orientation_error_rad"].amax()) * 180.0 / 3.141592653589793,
        )
        completed += int((terminated | truncated).sum().item())
        for code in torch.unique(values["primary_reason_code"]).tolist():
            if code:
                label = values["termination_reasons"][int(code)]
                failures[label] = failures.get(label, 0) + int(
                    (values["primary_reason_code"] == code).sum().item()
                )
    return {
        "clip": env.reference_bank.clip_ids[clip_index],
        "mode": "dynamic_wrist_finger_kinematic_object"
        if diagnostic_kinematic_object
        else "dynamic_wrist_finger_free_object_zero_residual",
        "steps": steps,
        "completed_episodes": completed,
        "maxima": maxima,
        "rmse": {key: (value / steps) ** 0.5 for key, value in squared_errors.items()},
        "terminal_reasons": failures,
        "finite": bool(torch.isfinite(env._get_observations()["policy"]).all()),
    }


def run_basis_actions(env, torch) -> dict[str, Any]:
    """Verify every 26-D basis reaches its intended frozen target slice."""

    env.cfg.diagnostic_kinematic_object = False
    env._clip_index.zero_()
    env._reference_index.zero_()
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    results: list[dict[str, Any]] = []
    for dimension in range(26):
        positive = torch.zeros((1, 26), device=env.device)
        negative = torch.zeros_like(positive)
        positive[:, dimension] = 0.25
        negative[:, dimension] = -0.25
        env._pre_physics_step(positive)
        positive_finger = env._joint_target_isaac.clone()
        positive_wrist = env._wrist_target_position.clone()
        env._pre_physics_step(negative)
        negative_finger = env._joint_target_isaac.clone()
        negative_wrist = env._wrist_target_position.clone()
        if dimension < 3:
            response = tensor_value((positive_wrist - negative_wrist).norm())
        elif dimension < 6:
            response = tensor_value(env._wrist_target_quaternion[:, 1:].abs().sum())
        else:
            isaac_index = int(env.action_adapter.isaac_from_canonical[dimension - 6].item())
            response = tensor_value(
                (positive_finger[:, isaac_index] - negative_finger[:, isaac_index]).abs()
            )
        results.append({"dimension": dimension, "response": response, "pass": response > 0.0})
    return {
        "basis_count": len(results),
        "all_pass": all(item["pass"] for item in results),
        "results": results,
        "object_direct_response": False,
        "scope": "target/action mapping semantics; dynamic contact response is assessed separately",
    }


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.balanced_clip_assignment = False
        cfg.wrist_controller_mode = args.wrist_controller_mode
        cfg.collect_wrist_diagnostics = args.collect_wrist_diagnostics
        cfg.collect_contact_telemetry = args.collect_contact_telemetry
        if args.force_limit_n is not None:
            cfg.wrist_force_limit_n = args.force_limit_n
        if args.torque_limit_nm is not None:
            cfg.wrist_torque_limit_nm = args.torque_limit_nm
        if args.translation_position_gain_s2 is not None:
            cfg.wrist_translation_position_gain_s2 = args.translation_position_gain_s2
        if args.rotation_position_gain_s2 is not None:
            cfg.wrist_rotation_position_gain_s2 = args.rotation_position_gain_s2
        if args.v1_translation_stiffness_npm is not None:
            cfg.wrist_v1_translation_stiffness_npm = args.v1_translation_stiffness_npm
        if args.v1_rotation_stiffness_nmprad is not None:
            cfg.wrist_v1_rotation_stiffness_nmprad = args.v1_rotation_stiffness_nmprad
        if args.v1_translation_damping_ratio is not None:
            cfg.wrist_v1_translation_damping_ratio = args.v1_translation_damping_ratio
        if args.v1_rotation_damping_ratio is not None:
            cfg.wrist_v1_rotation_damping_ratio = args.v1_rotation_damping_ratio
        if args.v1_force_limit_n is not None:
            cfg.wrist_v1_force_limit_n = args.v1_force_limit_n
        if args.v1_torque_limit_nm is not None:
            cfg.wrist_v1_torque_limit_nm = args.v1_torque_limit_nm
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260802)
        kinematic = []
        free = []
        for clip_index in range(2):
            kinematic.append(
                run_clip(
                    env,
                    torch,
                    clip_index=clip_index,
                    diagnostic_kinematic_object=True,
                    steps=args.steps,
                )
            )
            free.append(
                run_clip(
                    env,
                    torch,
                    clip_index=clip_index,
                    diagnostic_kinematic_object=False,
                    steps=args.steps,
                )
            )
        basis = run_basis_actions(env, torch)
        contract = env.contract_report()
        kinematic_pass = all(
            result["finite"]
            and result["maxima"]["wrist_position_m"] <= 0.02
            and result["maxima"]["wrist_rotation_deg"] <= 10.0
            for result in kinematic
        )
        free_finite = all(result["finite"] for result in free)
        contact_trace_available = args.collect_contact_telemetry and bool(
            env.contact_substep_records
        )
        c3_status = (
            "STAGE16C3_WRIST_DYNAMIC_TRACKING_BLOCKED"
            if not kinematic_pass
            else (
                "STAGE16C3_CONTACT_CAUSALITY_BLOCKED"
                if not contact_trace_available
                else "STAGE16C3_SEMANTIC_QUALIFICATION_PENDING_ANALYSIS"
            )
        )
        c3_blocker = (
            "C3_WRIST_DYNAMIC_TRACKING_EXCEEDS_2CM_10DEG"
            if not kinematic_pass
            else (
                "C3_CONTACT_DRIVEN_RESPONSE_EVIDENCE_NOT_YET_IMPLEMENTED"
                if not contact_trace_available
                else "C3_CONTACT_TRACE_REQUIRES_CAUSALITY_AUDIT"
            )
        )
        result = {
            "status": c3_status,
            "blocker": c3_blocker,
            "reference_artifact_contract": {
                "status": "VALIDATED",
                "clips": list(env.reference_bank.clip_ids),
                "frames": env.reference_bank.frame_count,
                "scene_frame_contract": contract["scene_frame"],
                "note": (
                    "C3-1 uses a kinematic-object diagnostic, but the wrist remains dynamic; "
                    "this validates artifact/frame conventions rather than claiming a full "
                    "kinematic wrist replay pass."
                ),
            },
            "dynamic_wrist_finger_kinematic_object": {
                "status": "PASS" if kinematic_pass else "FAIL",
                "results": kinematic,
            },
            "free_object_zero_residual": {
                "status": "FINITE_BUT_CONTACT_EVIDENCE_INCOMPLETE" if free_finite else "FAIL",
                "results": free,
                "formal_object_state_writes": contract["object_rollout_state_writes"],
                "wrist_root_state_writes": contract["wrist_root_state_writes_during_step"],
            },
            "basis_actions": basis,
            "mujoco_trace_replay": {
                "status": "NOT_RUN_GATE_BLOCKED_BY_C3",
                "reason": (
                    "PhysX action-trace replay is not evaluated as semantic acceptance "
                    "until the C.3 wrist and contact gates pass"
                ),
            },
            "contact": {
                "status": "CAPTURED_NOT_YET_ANALYZED"
                if contact_trace_available
                else "NOT_CAPTURED",
                "reason": (
                    "All-hand contact telemetry was written and awaits causal analysis."
                    if contact_trace_available
                    else (
                        "Run with --collect-contact-telemetry for substep pair/force/"
                        "impulse capture."
                    )
                ),
                "sensor_contract": env.contact_sensor_contract(),
                "hand_collision_inventory": env.hand_collision_inventory(),
                "substep_record_count": len(env.contact_substep_records),
            },
            "wrist_controller": {
                "mode": cfg.wrist_controller_mode,
                "force_limit_n": (
                    env.wrist_controller.profile.force_limit_n
                    if cfg.wrist_controller_mode == "wrist_impedance_v1"
                    else (
                        env.wrist_controller_v2.profile.force_limit_n
                        if cfg.wrist_controller_mode == "computed_wrench_v2"
                        else env.wrist_controller_v3.profile.force_limit_n
                    )
                ),
                "torque_limit_nm": (
                    env.wrist_controller.profile.torque_limit_nm
                    if cfg.wrist_controller_mode == "wrist_impedance_v1"
                    else (
                        env.wrist_controller_v2.profile.torque_limit_nm
                        if cfg.wrist_controller_mode == "computed_wrench_v2"
                        else env.wrist_controller_v3.profile.torque_limit_nm
                    )
                ),
                "translation_position_gain_s2": cfg.wrist_translation_position_gain_s2,
                "rotation_position_gain_s2": cfg.wrist_rotation_position_gain_s2,
                "v1_profile": {
                    "translation_stiffness_npm": (
                        env.wrist_controller.profile.translation_stiffness_npm
                    ),
                    "translation_damping_ratio": (
                        env.wrist_controller.profile.translation_damping_ratio
                    ),
                    "rotation_stiffness_nmprad": (
                        env.wrist_controller.profile.rotation_stiffness_nmprad
                    ),
                    "rotation_damping_ratio": (env.wrist_controller.profile.rotation_damping_ratio),
                },
                "substep_diagnostic_record_count": len(env.wrist_diagnostic_records),
            },
            "contract": contract,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.collect_wrist_diagnostics:
            wrist_path = args.output.with_name(f"{args.output.stem}_wrist_substeps.json")
            wrist_path.write_text(
                json.dumps(env.wrist_diagnostic_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result["wrist_controller"]["substep_diagnostics_path"] = str(wrist_path)
        if args.collect_contact_telemetry:
            contact_path = args.output.with_name(f"{args.output.stem}_contact_substeps.jsonl")
            contact_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in env.contact_substep_records
                ),
                encoding="utf-8",
            )
            result["contact"]["substep_trace_path"] = str(contact_path)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        # This runner is a qualification collector. A contact trace still
        # requires the independent causal audit before C.3 can authorize C.4.
        return 1
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
