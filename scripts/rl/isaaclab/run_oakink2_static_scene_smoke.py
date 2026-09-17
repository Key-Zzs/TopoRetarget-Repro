#!/usr/bin/env python3
"""Run serialized HOCap regression or OakInk2 static-scene infrastructure smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hocap-regression", "oakink2-smoke"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hocap-clip", choices=("hocap_170105", "hocap_170650"))
    parser.add_argument("--item-index", type=int)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _finite_tensor(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all())


def _one_rollout(env: Any, *, clip: str, steps: int, seed: int) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    print(f"PROGRESS clip={clip} phase=reset_start", file=sys.stderr, flush=True)
    observation, _ = env.reset(seed=seed)
    print(f"PROGRESS clip={clip} phase=reset_done", file=sys.stderr, flush=True)
    initial = env._state()
    initial_object = initial["object_position_scene"].detach().clone()
    initial_quaternion = initial["object_quaternion_wxyz"].detach().clone()
    finite = all(_finite_tensor(value) for value in initial.values() if hasattr(value, "dtype"))
    support_contact = False
    controller_started = False
    reward_finite = True
    for step in range(steps):
        action = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
        observation, reward, _terminated, _truncated, _extras = env.step(action)
        controller_started = True
        state = env._state()
        finite &= all(_finite_tensor(value) for value in state.values() if hasattr(value, "dtype"))
        finite &= all(
            _finite_tensor(value) for value in observation.values() if hasattr(value, "dtype")
        )
        reward_finite &= _finite_tensor(reward)
        sensor = env.scene[env._table_support_sensor_name()]
        force = sensor.data.force_matrix_w
        support_contact |= bool(torch.linalg.vector_norm(force, dim=-1).max().item() > 1.0e-4)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"PROGRESS clip={clip} phase=rollout step={step + 1}/{steps} "
                f"elapsed_s={time.monotonic() - started:.3f}",
                file=sys.stderr,
                flush=True,
            )
    terminal = env._state()
    translation = torch.linalg.vector_norm(
        terminal["object_position_scene"] - initial_object, dim=-1
    )
    dot = torch.abs((terminal["object_quaternion_wxyz"] * initial_quaternion).sum(dim=-1))
    rotation = 2.0 * torch.acos(torch.clamp(dot, max=1.0))
    writes = env.rollout_state_write_report()
    contract = env.contract_report()
    physics = contract["gravity_friction_curriculum"]
    collision = physics["support_collision_contract"]
    gravity = list(physics["gravity_world_mps2"])
    checks = {
        "scene_construction": True,
        "assets_load": True,
        "physx_starts": True,
        "controller_starts": controller_started,
        "simulation_finite": finite and reward_finite,
        "collision_active": collision["object_support_collision"] is True,
        "support_exists": physics["table_actor_active"] is True,
        "gravity_contract": gravity == [0.0, 0.0, -9.81],
        "no_nan_inf": finite and reward_finite,
        "no_illegal_object_write": int(writes["object_rollout_state_writes"]) == 0,
        "no_hidden_force": physics["external_guidance"] is False,
        "no_wrist_root_teleport": int(writes["wrist_root_state_writes_during_step"]) == 0,
    }
    return {
        "clip": clip,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "steps": steps,
        "seed": seed,
        "contract": contract,
        "state_write_report": writes,
        "scientific_physical_outcome": {
            "support_contact_observed": support_contact,
            "object_translation_drift_m_max": float(translation.max().item()),
            "object_rotation_drift_rad_max": float(rotation.max().item()),
            "classification_role": "SCIENTIFIC_OUTCOME_NOT_INFRASTRUCTURE_GATE",
        },
    }


def _oakink2(root: Path, *, steps: int, item_index: int) -> list[dict[str, Any]]:
    from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
    from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

    payload = json.loads(root.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    if item_index < 0 or item_index >= len(payload["anchors"]):
        raise ValueError(f"OAKINK2_STATIC_SCENE_ITEM_INDEX_INVALID:{item_index}")
    for index, item in ((item_index, payload["anchors"][item_index]),):
        env = None
        try:
            print(
                f"PROGRESS clip={item['clip_id']} phase=construct_start",
                file=sys.stderr,
                flush=True,
            )
            env = _make_table_env(
                clip=item["clip_id"],
                num_envs=1,
                start_index=0,
                mode=ContactRewardMode.STRICT_PER_FINGER_V4,
                stage="C4",
                training_rsi=False,
                reward_aggregation_mode="grouped_multiplicative_v1",
                rse_enabled=True,
                full_horizon_evaluation=False,
                reference_path=Path(item["reference"]),
                object_usd_path=Path(item["object_usd"]),
                support_proxy_path=Path(item["support_proxy"]),
                support_asset_path=Path(item["support_asset"]),
                contact_contract_path=Path(item["contact_contract"]),
                contact_mask_root=Path(item["contact_mask_root"]),
                reference_distance_root=Path(item["reference_distance_root"]),
                object_mesh_root=Path(item["object_mesh_root"]),
                continuous_virtual_wrist_angles=True,
                source_controller_admission_v2=True,
                static_reference_adapter=True,
            )
            print(
                f"PROGRESS clip={item['clip_id']} phase=construct_done",
                file=sys.stderr,
                flush=True,
            )
            row = _one_rollout(env, clip=item["clip_id"], steps=steps, seed=20260917 + index)
            row["anchor_id"] = item["anchor_id"]
        except BaseException as exc:
            row = {
                "anchor_id": item["anchor_id"],
                "clip": item["clip_id"],
                "status": "FAIL",
                "infrastructure_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        finally:
            if env is not None:
                env.close()
        rows.append(row)
    return rows


def _hocap(*, steps: int, clip: str) -> list[dict[str, Any]]:
    from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _load_start, _make_table_env

    rows: list[dict[str, Any]] = []
    for index, selected_clip in ((0 if clip == "hocap_170105" else 1, clip),):
        env = None
        try:
            start = _load_start(selected_clip)
            print(
                f"PROGRESS clip={selected_clip} phase=construct_start",
                file=sys.stderr,
                flush=True,
            )
            env = _make_table_env(
                clip=selected_clip,
                num_envs=1,
                start_index=int(start["start_index"]),
                stage="C4",
            )
            print(
                f"PROGRESS clip={selected_clip} phase=construct_done",
                file=sys.stderr,
                flush=True,
            )
            row = _one_rollout(env, clip=selected_clip, steps=steps, seed=20260915 + index)
        except BaseException as exc:
            row = {
                "clip": selected_clip,
                "status": "FAIL",
                "infrastructure_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        finally:
            if env is not None:
                env.close()
        rows.append(row)
    return rows


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.steps != 40:
        raise ValueError("STATIC_SCENE_SMOKE_REQUIRES_EULA_AND_FROZEN_40_STEPS")
    if args.mode == "oakink2-smoke" and args.manifest is None:
        raise ValueError("OAKINK2_STATIC_SCENE_MANIFEST_REQUIRED")
    if args.mode == "oakink2-smoke" and args.item_index is None:
        raise ValueError("OAKINK2_STATIC_SCENE_ITEM_INDEX_REQUIRED")
    if args.mode == "hocap-regression" and args.manifest is not None:
        raise ValueError("HOCAP_REGRESSION_DOES_NOT_ACCEPT_OAKINK2_MANIFEST")
    if args.mode == "hocap-regression" and args.hocap_clip is None:
        raise ValueError("HOCAP_REGRESSION_CLIP_REQUIRED")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        rows = (
            _hocap(steps=args.steps, clip=args.hocap_clip)
            if args.mode == "hocap-regression"
            else _oakink2(
                args.manifest.resolve(),  # type: ignore[union-attr]
                steps=args.steps,
                item_index=args.item_index,  # type: ignore[arg-type]
            )
        )
        payload = {
            "schema_version": (
                "ExistingSupportPhysicalizationHOCapRegressionV1"
                if args.mode == "hocap-regression"
                else "OakInk2StaticReferenceHoldSmokeV1"
            ),
            "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
            "mode": args.mode,
            "max_gpu_jobs": 1,
            "rows": rows,
        }
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 2
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
