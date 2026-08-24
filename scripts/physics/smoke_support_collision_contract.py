#!/usr/bin/env python3
"""Exercise SupportCollisionContractV1 in a two-environment PhysX smoke.

Environment 0 keeps explicit hand/support collision enabled.  Environment 1
authors only the inferred hand/support filtered pair.  Both environments keep
object/support collision enabled.  The receipt is a runtime contract smoke,
not a PF/DF scientific evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.support import (  # noqa: E402
    SupportType,
    apply_hand_support_pair_filter,
    support_collision_policy,
)
from toporetarget.runtime import validate_gpu_preflight_receipt  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-preflight-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().numpy().reshape(-1)]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 120:
        raise ValueError("SUPPORT_COLLISION_SMOKE_AT_LEAST_120_STEPS_REQUIRED")
    if not args.accept_eula:
        raise ValueError("SUPPORT_COLLISION_SMOKE_REQUIRES_EULA")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    gpu_receipt = validate_gpu_preflight_receipt(args.gpu_preflight_receipt)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    simulation = None
    result: dict[str, Any] | None = None
    try:
        import isaaclab.sim as sim_utils
        import omni.usd
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        support_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/SupportProxy",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 1.0, 0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.85)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        object_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectProbe",
            spawn=sim_utils.CuboidCfg(
                size=(0.16, 0.16, 0.16),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.0, 0.75)),
        )
        hand_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HandProbe",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 0.10, 0.10),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.0, 0.65),
                lin_vel=(0.0, 0.0, -1.0),
            ),
        )

        @configclass
        class CollisionSceneCfg(InteractiveSceneCfg):
            support: AssetBaseCfg = support_cfg
            object_probe: RigidObjectCfg = object_cfg
            hand_probe: RigidObjectCfg = hand_cfg

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
            ),
        )
        simulation = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(
            CollisionSceneCfg(
                num_envs=2,
                env_spacing=2.0,
                # The two environments intentionally have different pairwise
                # collision policies; PhysX replication would copy env_0's
                # explicit policy into env_1 before its filtered pair is read.
                replicate_physics=False,
                clone_in_fabric=False,
            )
        )
        stage = omni.usd.get_context().get_stage()
        explicit_policy = apply_hand_support_pair_filter(
            stage,
            hand_prim_paths=("/World/envs/env_0/HandProbe",),
            support_prim_paths=("/World/envs/env_0/SupportProxy",),
            support_type=SupportType.SOURCE_EXPLICIT_SUPPORT,
        )
        inferred_policy = apply_hand_support_pair_filter(
            stage,
            hand_prim_paths=("/World/envs/env_1/HandProbe",),
            support_prim_paths=("/World/envs/env_1/SupportProxy",),
            support_type=SupportType.INFERRED_PLANAR_SUPPORT,
        )
        simulation.reset()
        for name in ("object_probe", "hand_probe"):
            entity = scene[name]
            initial_state = entity.data.default_root_state.clone()
            initial_state[:, :3] += scene.env_origins
            entity.write_root_state_to_sim(initial_state)
            entity.reset()
        scene.write_data_to_sim()
        scene.update(sim_cfg.dt)
        object_rows: list[list[list[float]]] = []
        hand_rows: list[list[list[float]]] = []
        for _ in range(args.steps):
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(sim_cfg.dt)
            object_rows.append(
                [_as_list(scene["object_probe"].data.root_state_w[index]) for index in range(2)]
            )
            hand_rows.append(
                [_as_list(scene["hand_probe"].data.root_state_w[index]) for index in range(2)]
            )

        object_final = [object_rows[-1][index] for index in range(2)]
        hand_final = [hand_rows[-1][index] for index in range(2)]
        object_rest = [abs(row[2] - 0.18) <= 0.03 and abs(row[9]) <= 0.08 for row in object_final]
        explicit_hand_collision = hand_final[0][2] >= 0.13 and abs(hand_final[0][9]) <= 0.10
        inferred_hand_crossed = hand_final[1][2] < -0.20
        inferred_crossing_velocity_errors = [
            abs(frame[1][9] + 1.0) for frame in hand_rows if frame[1][2] <= 0.20
        ]
        inferred_velocity_error = (
            max(inferred_crossing_velocity_errors)
            if inferred_crossing_velocity_errors
            else float("inf")
        )
        inferred_no_impulse = inferred_hand_crossed and inferred_velocity_error <= 0.05
        checks = {
            "cuda_simulation_device": str(simulation.device) == "cuda:0",
            "object_rests_and_collides_explicit": object_rest[0],
            "object_rests_and_collides_inferred": object_rest[1],
            "inferred_hand_crosses_support": inferred_hand_crossed,
            "inferred_hand_crossing_no_impulse": inferred_no_impulse,
            "explicit_hand_support_collision_active": explicit_hand_collision,
            "inferred_filter_is_pairwise": (
                inferred_policy["status"] == "PAIRWISE_HAND_SUPPORT_FILTER_AUTHORED"
                and inferred_policy["global_support_collision_disabled"] is False
            ),
        }
        passed = all(checks.values())
        result = {
            "schema_version": "SupportCollisionContractV1SmokeReceipt",
            "status": "PASS" if passed else "FAIL",
            "timestamp": _utc(),
            "runtime_authority": "PHYSX_GPU_SMOKE_NOT_PF_DF_SCIENTIFIC_ACCEPTANCE",
            "gpu_preflight": {
                "path": str(args.gpu_preflight_receipt.resolve()),
                "status": gpu_receipt["status"],
                "host": gpu_receipt["host"],
            },
            "steps": args.steps,
            "dt": sim_cfg.dt,
            "collision_matrix": {
                "explicit": support_collision_policy(SupportType.SOURCE_EXPLICIT_SUPPORT),
                "inferred": support_collision_policy(SupportType.INFERRED_PLANAR_SUPPORT),
            },
            "authored_policy": {"explicit": explicit_policy, "inferred": inferred_policy},
            "checks": checks,
            "telemetry": {
                "object_final_root_state_w": object_final,
                "hand_final_root_state_w": hand_final,
                "inferred_crossing_max_vertical_velocity_error_mps": inferred_velocity_error,
            },
        }
        _write(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if passed else 2
    except BaseException as error:
        failure = {
            "schema_version": "SupportCollisionContractV1SmokeReceipt",
            "status": "FAIL",
            "timestamp": _utc(),
            "reason": f"{type(error).__name__}:{error}",
            "traceback": traceback.format_exc(),
        }
        _write(args.output.resolve(), failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
