#!/usr/bin/env python3
"""Evaluate a matched full-gravity object-only support counterfactual in PhysX.

The command is deliberately separate from the RL environment.  It spawns the
frozen object once with its frame-zero reset pose, optionally spawns the
generated finite support actor, then records only simulator telemetry.  No
object pose is written after reset and no hand, policy, reward, or guidance
path is present in this scene.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--case", choices=("with_support", "without_support"), required=True)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references",
    )
    parser.add_argument(
        "--object-usd",
        type=Path,
        default=REPO_ROOT / ".local/generated_assets/isaaclab",
    )
    parser.add_argument(
        "--support-asset",
        type=Path,
        default=None,
        help="Generated support_proxy.usda; required for --case with_support.",
    )
    parser.add_argument(
        "--proxy-json",
        type=Path,
        default=None,
        help="Inference table_proxy.json; required for --case with_support.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output receipt. Defaults to .local/reports/stage16_support_reconstruction/physics/.",
    )
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _one_vector(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.size == 0:
        return np.zeros(3, dtype=np.float64)
    reshaped = np.asarray(array, dtype=np.float64).reshape(-1, 3)
    return reshaped.sum(axis=0)


def _contact_telemetry(sensor: Any, support_normal: np.ndarray) -> tuple[np.ndarray, bool, int]:
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError("SUPPORT_CONTACT_FORCE_MATRIX_UNAVAILABLE")
    filtered_force = _as_numpy(force_matrix)[0]
    filtered_vectors = np.asarray(filtered_force, dtype=np.float64).reshape(-1, 3)
    magnitudes = np.linalg.norm(filtered_vectors, axis=1)
    active = magnitudes > 1.0e-4
    if sensor.data.net_forces_w is not None:
        net_force = _one_vector(sensor.data.net_forces_w[0])
    else:
        net_force = filtered_vectors.sum(axis=0)
    # The object-centric sensor reports the force on the object.  Retain the
    # signed world vector for audit, while the backend reducer projects it
    # onto the inferred support normal.
    if not np.isfinite(net_force).all():
        raise RuntimeError("SUPPORT_CONTACT_FORCE_NONFINITE")
    del support_normal
    return net_force, bool(active.any()), int(active.sum())


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_path = args.reference / f"{args.clip}.world_wrist.stage16.npz"
    object_usd = args.object_usd / args.clip / f"{args.clip}.usda"
    if not reference_path.is_file():
        raise FileNotFoundError(f"REFERENCE_NOT_FOUND:{reference_path}")
    if not object_usd.is_file():
        raise FileNotFoundError(f"OBJECT_USD_NOT_FOUND:{object_usd}")
    with np.load(reference_path, allow_pickle=False) as archive:
        translation = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
    if translation.shape[0] < 1 or quaternion.shape[0] < 1:
        raise ValueError("REFERENCE_FRAME_ZERO_MISSING")
    if args.case == "with_support":
        if args.support_asset is None or not args.support_asset.is_file():
            raise FileNotFoundError("SUPPORT_ASSET_REQUIRED")
        if args.proxy_json is None or not args.proxy_json.is_file():
            raise FileNotFoundError("SUPPORT_PROXY_JSON_REQUIRED")
        proxy = json.loads(args.proxy_json.read_text(encoding="utf-8"))
        normal = np.asarray(proxy["plane_normal"], dtype=np.float64)
        normal /= np.linalg.norm(normal)
    else:
        proxy = {"plane_normal": [0.0, 0.0, 1.0], "plane_offset": 0.0}
        normal = np.asarray(proxy["plane_normal"], dtype=np.float64)
    return (
        translation[0],
        quaternion[0],
        {
            "reference_path": str(reference_path.resolve()),
            "object_usd": str(object_usd.resolve()),
            "support_asset": (
                str(args.support_asset.resolve()) if args.support_asset is not None else None
            ),
            "proxy": proxy,
            "support_normal": normal,
        },
    )


def _run_simulation(
    *,
    args: argparse.Namespace,
    initial_translation: np.ndarray,
    initial_quaternion: np.ndarray,
    inputs: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    print("STAGE16_SUPPORT_MARKER app_launch", flush=True)
    app = AppLauncher(headless=True).app
    print("STAGE16_SUPPORT_MARKER app_started", flush=True)
    simulation = None
    try:
        # Remaining Isaac imports happen only after AppLauncher owns the
        # SimulationApp, matching the repository's Isaac entry-point contract.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg

        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=1,
            device="cuda:0",
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
                gpu_max_rigid_contact_count=2**22,
                gpu_max_rigid_patch_count=2**20,
            ),
        )
        simulation = sim_utils.SimulationContext(sim_cfg)
        print("STAGE16_SUPPORT_MARKER simulation_context_created", flush=True)
        object_path = Path(inputs["object_usd"])
        object_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(object_path),
                copy_from_source=False,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(value) for value in initial_translation),
                rot=tuple(float(value) for value in initial_quaternion),
            ),
        )
        support_cfg = None
        sensor_cfg = None
        if args.case == "with_support":
            proxy = inputs["proxy"]
            support_normal = np.asarray(inputs["support_normal"], dtype=np.float64)
            support_top_pose = np.asarray(proxy["table_pose"], dtype=np.float64)
            support_center = support_top_pose[:3] - (
                0.5 * float(proxy["table_thickness"]) * support_normal
            )
            support_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SupportProxy",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(inputs["support_asset"]),
                    copy_from_source=False,
                    activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(value) for value in support_center),
                    rot=tuple(float(value) for value in support_top_pose[3:7]),
                ),
            )
            sensor_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                force_threshold=1.0e-4,
                max_contact_data_count_per_prim=64,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/SupportProxy"],
            )

        from isaaclab.utils import configclass

        @configclass
        class SupportSceneCfg(InteractiveSceneCfg):
            object: RigidObjectCfg = object_cfg
            support: RigidObjectCfg | None = support_cfg
            contact_sensor: ContactSensorCfg | None = sensor_cfg

        scene_cfg = SupportSceneCfg(
            num_envs=1,
            env_spacing=1.0,
            replicate_physics=True,
            clone_in_fabric=False,
            lazy_sensor_update=False,
        )
        scene = InteractiveScene(scene_cfg)
        print("STAGE16_SUPPORT_MARKER scene_created", flush=True)
        simulation.reset()
        print("STAGE16_SUPPORT_MARKER simulation_reset", flush=True)
        scene.update(sim_cfg.dt)
        obj = scene["object"]
        print("STAGE16_SUPPORT_MARKER object_resolved", flush=True)
        runtime_mass = float(_as_numpy(obj.data.default_mass[0]).reshape(-1)[0])
        support_runtime_state = None
        if args.case == "with_support":
            support_runtime_state = _as_numpy(scene["support"].data.root_state_w[0]).reshape(-1)
            print(
                f"STAGE16_SUPPORT_MARKER support_state {support_runtime_state.tolist()}",
                flush=True,
            )
        sensor = scene["contact_sensor"] if args.case == "with_support" else None
        print("STAGE16_SUPPORT_MARKER sensor_resolved", flush=True)
        support_normal = np.asarray(inputs["support_normal"], dtype=np.float64)
        rows: list[dict[str, Any]] = []
        started = time.monotonic()
        for step in range(args.steps):
            if step == 0:
                print("STAGE16_SUPPORT_MARKER first_step_begin", flush=True)
            scene.write_data_to_sim()
            if step == 0:
                print("STAGE16_SUPPORT_MARKER data_written", flush=True)
            simulation.step(render=False)
            if step == 0:
                print("STAGE16_SUPPORT_MARKER physics_stepped", flush=True)
            scene.update(sim_cfg.dt)
            if step == 0:
                print("STAGE16_SUPPORT_MARKER scene_updated", flush=True)
            root_state = _as_numpy(obj.data.root_state_w[0]).reshape(-1)
            if root_state.shape != (13,) or not np.isfinite(root_state).all():
                raise RuntimeError("SUPPORT_OBJECT_ROOT_STATE_NONFINITE_OR_WRONG_SHAPE")
            position = root_state[:3]
            quaternion = root_state[3:7]
            linear = root_state[7:10]
            angular = root_state[10:13]
            if sensor is not None:
                support_force, support_contact, contact_count = _contact_telemetry(
                    sensor, support_normal
                )
            else:
                support_force = np.zeros(3, dtype=np.float64)
                support_contact = False
                contact_count = 0
            rows.append(
                {
                    "step": step,
                    "time_s": float((step + 1) * sim_cfg.dt),
                    "position_world_m": position.tolist(),
                    "orientation_world_wxyz": quaternion.tolist(),
                    "linear_velocity_world_mps": linear.tolist(),
                    "angular_velocity_world_radps": angular.tolist(),
                    "support_force_world_n": support_force.tolist(),
                    "support_contact": support_contact,
                    "support_contact_count": contact_count,
                    "external_guidance": False,
                    "object_state_writes": 0,
                    "hidden_attachment": False,
                    "kinematic_support_force": False,
                }
            )
        print("STAGE16_SUPPORT_MARKER stepping_complete", flush=True)
        result = {
            "schema_version": "Stage16SupportPhysXReceiptV1",
            "status": "CAPTURED_PHYSX_TELEMETRY",
            "clip": args.clip,
            "case": args.case,
            "steps": args.steps,
            "dt_s": float(sim_cfg.dt),
            "full_gravity": True,
            "device": "cuda:0",
            "wall_time_s": time.monotonic() - started,
            "mass_kg": runtime_mass,
            "initial_translation_world_m": initial_translation.tolist(),
            "initial_quaternion_world_wxyz": initial_quaternion.tolist(),
            "support_runtime_root_state_initial": (
                support_runtime_state.tolist() if support_runtime_state is not None else None
            ),
            "inputs": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in inputs.items()
            },
            "summary": {
                "status": "PENDING_BACKEND_REDUCTION",
                "reducer": "toporetarget.physics.support.physics_validation",
            },
            "telemetry": rows,
            "causality": {
                "object_state_writes_after_reset": 0,
                "external_guidance": False,
                "hidden_attachment": False,
                "kinematic_support_force": False,
                "hand_present": False,
                "policy_present": False,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {key: result[key] for key in ("status", "clip", "case", "steps")},
                sort_keys=True,
            ),
            flush=True,
        )
        return result
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        app.close(wait_for_replicator=False)


def main() -> int:
    args = _parse_args()
    if not args.clip or any(token in args.clip for token in ("/", "\\", "..")):
        raise SystemExit("INDEPENDENT_SUPPORT_CLIP_ID_INVALID")
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    initial_translation, initial_quaternion, inputs = _load_inputs(args)
    output = args.output or (
        REPO_ROOT
        / ".local/reports/stage16_support_reconstruction/physics"
        / args.clip
        / f"{args.case}.json"
    )
    result = _run_simulation(
        args=args,
        initial_translation=initial_translation,
        initial_quaternion=initial_quaternion,
        inputs=inputs,
        output=output,
    )
    return 0 if result["status"] == "CAPTURED_PHYSX_TELEMETRY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
