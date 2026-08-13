#!/usr/bin/env python3
"""Render real offscreen Stage 16-C.1 asset qualification frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (  # noqa: E402
    load_asset_migration_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
    )
    parser.add_argument("--object", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--include-hand-poses", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c1_asset_migration/visual",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_asset_migration_config(args.config)
    cfg.validate(REPO_ROOT)
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True, enable_cameras=True).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import omni.replicator.core as rep
        import omni.usd
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors.camera import Camera, CameraCfg
        from PIL import Image
        from pxr import Gf, UsdGeom

        args.output_dir.mkdir(parents=True, exist_ok=True)
        wuji_usd = (
            REPO_ROOT / cfg.output_root / "wuji_hand2_beta1/configuration/wujihand2_physics.usd"
        ).resolve()
        object_usd = (REPO_ROOT / cfg.output_root / args.object / f"{args.object}.usda").resolve()
        if not wuji_usd.is_file() or not object_usd.is_file():
            raise FileNotFoundError("generated Stage 16-C.1 USD assets are missing")

        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            render_interval=1,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Stage16C1Light", light_cfg)

        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(wuji_usd),
                copy_from_source=False,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=False,
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=list(cfg.wuji.joint_order),
                    stiffness=cfg.wuji.drive_stiffness,
                    damping=cfg.wuji.drive_damping,
                    effort_limit_sim=0.6,
                    velocity_limit_sim=12.0,
                )
            },
        )
        scene_cfg.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(object_usd),
                copy_from_source=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, 0.35)),
        )
        scene = InteractiveScene(scene_cfg)
        camera = Camera(
            CameraCfg(
                prim_path="/World/Stage16C1Camera",
                height=480,
                width=640,
                update_period=0.0,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=32.0,
                    focus_distance=0.5,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 10.0),
                ),
            )
        )
        sim.reset()
        robot = scene["robot"]
        obj = scene["object"]
        expected = list(cfg.wuji.joint_order)
        lookup = {name: index for index, name in enumerate(robot.joint_names)}
        indices = torch.tensor([lookup[name] for name in expected], device=sim.device)
        stage = omni.usd.get_context().get_stage()
        robot_prim = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
        collision_path = f"/World/envs/env_0/Object/{args.object}/Collision/convex_hull_v1"
        collision_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(collision_path))
        if not collision_mesh:
            raise RuntimeError(f"HEADLESS_RENDER_FAILURE: missing {collision_path}")

        rendered: list[dict[str, object]] = []

        def step(count: int) -> None:
            for _ in range(count):
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(sim_cfg.dt)
                camera.update(sim_cfg.dt)

        def set_view(eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
            camera.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device=sim.device),
                torch.tensor([target], dtype=torch.float32, device=sim.device),
            )

        def capture(name: str) -> None:
            step(8)
            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            path = args.output_dir / name
            Image.fromarray(rgb).save(path)
            rendered.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "shape": list(rgb.shape),
                    "minimum": int(rgb.min()),
                    "maximum": int(rgb.max()),
                    "mean": float(rgb.mean()),
                    "nonblank": bool(rgb.max() > rgb.min()),
                }
            )

        set_view((0.45, 0.42, 0.62), (0.0, 0.0, 0.4))
        if args.include_hand_poses:
            capture("wuji_default_pose.png")
            semantic_targets = torch.tensor(
                [[0.35, -0.15, 0.45, 0.4] * 5], dtype=torch.float32, device=sim.device
            )
            targets = torch.zeros_like(robot.data.joint_pos)
            targets[:, indices] = semantic_targets
            robot.set_joint_position_target(targets)
            step(80)
            capture("wuji_joint_step_poses.png")

        robot_prim.MakeInvisible()
        object_state = obj.data.default_root_state.clone()
        object_state[:, :3] = torch.tensor([[0.0, 0.0, 0.35]], device=sim.device)
        obj.write_root_state_to_sim(object_state)
        collision_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        collision_mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.05, 0.05)])
        collision_mesh.CreateDisplayOpacityAttr([0.35])
        set_view((0.30, 0.28, 0.48), (0.0, 0.0, 0.35))
        capture(f"{args.object}_visual_collision_overlay.png")

        collision_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        robot_prim.MakeVisible()
        robot_root_state = robot.data.default_root_state.clone()
        robot_root_state[:, :3] += scene.env_origins
        robot.write_root_state_to_sim(robot_root_state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        contact_state = obj.data.default_root_state.clone()
        contact_state[:, :3] = torch.tensor([[0.09, 0.0, 0.35]], device=sim.device)
        obj.write_root_state_to_sim(contact_state)
        contact_targets = torch.zeros_like(robot.data.joint_pos)
        contact_targets[:, indices] = 0.35
        robot.set_joint_position_target(contact_targets)
        wrist_ids, _ = robot.find_bodies("r_wrist")
        force = torch.zeros((1, 1, 3), device=sim.device)
        force[..., 0] = 0.5
        robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=wrist_ids)
        step(35)
        set_view((0.42, 0.38, 0.56), (0.06, 0.0, 0.38))
        capture(f"{args.object}_hand_object_contact.png")

        write_json(
            args.output_dir / f"{args.object}_render.json",
            {
                "status": "PASS" if all(item["nonblank"] for item in rendered) else "FAIL",
                "object": args.object,
                "device": str(sim.device),
                "offscreen": True,
                "frames": rendered,
            },
        )
        print(json.dumps(rendered, sort_keys=True))
        rep.vp_manager.destroy_hydra_textures("Replicator")
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
