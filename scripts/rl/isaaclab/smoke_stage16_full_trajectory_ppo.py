#!/usr/bin/env python3
"""Run one real table-supported full-trajectory Stage16 V3 PPO smoke.

This is intentionally separate from the historical ContactReady RSI runner.
Every environment starts from the single receipt-selected PRE_CONTACT frame;
the inferred table is present for the complete reference horizon.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
_ISAACLAB_SOURCE = REPO_ROOT / ".local/external/IsaacLab/source/isaaclab"
sys.path.insert(0, str(_ISAACLAB_SOURCE))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.full_trajectory_episode_start import validate_full_trajectory_start

START_ROOT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart/episode_start"
SUPPORT_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction/inference"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=START_ROOT.parent / "ppo_smoke")
    return parser


def _load_start(clip: str) -> dict[str, Any]:
    path = START_ROOT / f"{clip}.json"
    payload = validate_full_trajectory_start(
        json.loads(path.read_text(encoding="utf-8")), clip=clip
    )
    reference = REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz"
    table = SUPPORT_ROOT / clip / "table_proxy.json"
    if payload["reference_hash"] != _sha256(reference):
        raise RuntimeError("FULL_TRAJECTORY_SMOKE_REFERENCE_HASH_MISMATCH")
    if payload["support_contract_hash"] != _sha256(table):
        raise RuntimeError("FULL_TRAJECTORY_SMOKE_SUPPORT_HASH_MISMATCH")
    return payload


def _make_table_env(
    *,
    clip: str,
    num_envs: int,
    start_index: int,
    mode: Any = None,
    stage: str = "C0",
) -> Any:
    """Construct the production PPO environment with fixed finite supports."""

    import isaaclab.sim as sim_utils
    import numpy as np
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass

    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )
    from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

    def support_cfg(support_clip: str, name: str) -> RigidObjectCfg:
        proxy_path = SUPPORT_ROOT / support_clip / "table_proxy.json"
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        normal = np.asarray(proxy["plane_normal"], dtype=np.float64)
        center = np.asarray(proxy["table_pose"][:3], dtype=np.float64) - (
            0.5 * float(proxy["table_thickness"]) * normal
        )
        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(
                    REPO_ROOT / ".local/support_assets/hocap" / support_clip / "support_proxy.usda"
                ),
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
                pos=tuple(float(value) for value in center),
                rot=tuple(float(value) for value in proxy["table_pose"][3:7]),
            ),
        )

    def support_sensor(object_name: str, support_name: str) -> ContactSensorCfg:
        return ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
            update_period=0.0,
            history_length=1,
            track_pose=True,
            track_contact_points=False,
            track_friction_forces=False,
            force_threshold=1.0e-4,
            max_contact_data_count_per_prim=64,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{support_name}"],
        )

    @configclass
    class TableSceneCfg(InteractiveSceneCfg):
        support_170105: RigidObjectCfg | None = None
        support_170650: RigidObjectCfg | None = None
        object_170105_support_contact: ContactSensorCfg | None = None
        object_170650_support_contact: ContactSensorCfg | None = None

    class TableSupportedEnv(IsaacPPO26DReferenceTrackingEnv):
        def _setup_scene(self) -> None:
            self._robot = Articulation(self.cfg.robot)
            self._object_170105 = RigidObject(self.cfg.object_170105)
            self._object_170650 = RigidObject(self.cfg.object_170650)
            self._support_170105 = RigidObject(self.cfg.scene.support_170105)
            self._support_170650 = RigidObject(self.cfg.scene.support_170650)
            self.scene.articulations["robot"] = self._robot
            self.scene.rigid_objects["object_170105"] = self._object_170105
            self.scene.rigid_objects["object_170650"] = self._object_170650
            self.scene.rigid_objects["support_170105"] = self._support_170105
            self.scene.rigid_objects["support_170650"] = self._support_170650
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[])
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light.func("/World/Light", light)

        def _reset_idx(self, env_ids: Any) -> None:
            super()._reset_idx(env_ids)
            # Valid only for this receipt-selected stable table support start.
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            state_105 = self._object_170105.data.root_state_w[ids].clone()
            state_650 = self._object_170650.data.root_state_w[ids].clone()
            state_105[:, 7:] = 0.0
            state_650[:, 7:] = 0.0
            self._object_170105.write_root_state_to_sim(state_105, env_ids=ids)
            self._object_170650.write_root_state_to_sim(state_650, env_ids=ids)

        def contract_report(self) -> dict[str, object]:
            report = super().contract_report()
            physical = report["gravity_friction_curriculum"]
            assert isinstance(physical, dict)
            physical["support"] = "finite_inferred_table_proxy_v1"
            physical["table_actor_active"] = True
            physical["mid_trajectory_rsi"] = "disabled"
            physical["table_resting_reset_semantics"] = "TABLE_RESTING_RESET_SEMANTICS_V1"
            return report

    cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo_cfg.configure_stage16d_ppo26d(
        cfg, num_envs=num_envs, clip=clip, rsi=False, critical_dr=False
    )
    selected_mode = ContactRewardMode.AGGREGATE_V3 if mode is None else mode
    ppo_cfg.configure_stage16d_contact_reward(
        cfg,
        mode=selected_mode,
        reference_root=REFERENCE_ROOT,
        contact_reward_contract=(
            REPO_ROOT
            / ".local/reports/stage16d_reward_v3_pairforce_unblock"
            / "contact_reward_contract.json"
        ),
        contact_mask_root=REPO_ROOT / ".local/reports/stage16d_reward_v3_contact",
    )
    ppo_cfg.configure_stage16_p3_p4_curriculum(
        cfg,
        curriculum_contract_path=(
            REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
        ),
        stage=stage,
    )
    cfg.scene = TableSceneCfg(
        num_envs=num_envs,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=False,
        lazy_sensor_update=True,
        support_170105=support_cfg("hocap_170105", "Support170105"),
        support_170650=support_cfg("hocap_170650", "Support170650"),
        object_170105_support_contact=support_sensor("Object170105", "Support170105"),
        object_170650_support_contact=support_sensor("Object170650", "Support170650"),
    )
    cfg.stage16d_fixed_clip = clip
    cfg.evaluation_reset_reference_indices = (start_index,) * num_envs
    cfg.stage16_support_mode = "finite_inferred_table_proxy_v1"
    cfg.stage16_external_guidance = False
    return TableSupportedEnv(cfg)


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.num_envs <= 0 or args.updates <= 0:
        raise ValueError("FULL_TRAJECTORY_SMOKE_BUDGET_INVALID")
    if (args.num_envs * 40) % 32:
        raise ValueError("FULL_TRAJECTORY_SMOKE_MINIBATCH_ALIGNMENT_INVALID")
    start = _load_start(args.clip)
    output = args.output_root.resolve() / args.clip
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    env = None
    try:
        import torch

        from scripts.rl.isaaclab.train_stage16_p3_physical_curriculum import (
            _restore_zero_g_checkpoint,
            _selected_zero_g_checkpoint,
        )
        from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        env = _make_table_env(
            clip=args.clip, num_envs=args.num_envs, start_index=int(start["start_index"])
        )
        env.reset(seed=20260814)
        contract = env.contract_report()
        physics = contract["gravity_friction_curriculum"]
        assert isinstance(physics, dict)
        if (
            physics.get("support") != "finite_inferred_table_proxy_v1"
            or physics.get("table_actor_active") is not True
            or physics.get("mid_trajectory_rsi") != "disabled"
        ):
            raise RuntimeError("FULL_TRAJECTORY_SMOKE_ENVIRONMENT_CONTRACT_INVALID")
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        selected = _selected_zero_g_checkpoint(ContactRewardMode.AGGREGATE_V3, args.clip)
        initialization = _restore_zero_g_checkpoint(
            trainer,
            checkpoint=Path(str(selected["selected_checkpoint"])),
            clip=args.clip,
            mode=ContactRewardMode.AGGREGATE_V3,
        )
        before = torch.cat([value.detach().flatten() for value in trainer.model.parameters()])
        metrics = [trainer.collect_and_update(env) for _ in range(args.updates)]
        after = torch.cat([value.detach().flatten() for value in trainer.model.parameters()])
        update_norm = float(torch.linalg.vector_norm(after - before).item())
        writes = env.rollout_state_write_report()
        sensor = env.scene[
            "object_170105_support_contact"
            if args.clip == "hocap_170105"
            else "object_170650_support_contact"
        ]
        support_force = sensor.data.force_matrix_w
        support_contact_observed = bool(
            torch.linalg.vector_norm(support_force, dim=-1).max().item() > 1.0e-4
        )
        residual_action_abs_mean = float(env._actions.abs().mean().item())
        samples = sum(int(item["samples"]) for item in metrics)
        finite = all(bool(item["finite"]) for item in metrics)
        passed = bool(
            finite
            and update_norm > 0.0
            and samples == args.num_envs * args.updates * 40
            and int(writes["object_rollout_state_writes"]) == 0
            and int(writes["wrist_root_state_writes_during_step"]) == 0
            and residual_action_abs_mean > 0.0
        )
        result = {
            "schema_version": "Stage16FullTrajectoryPpoSmokeV1",
            "status": "PASS" if passed else "FAIL",
            "clip": args.clip,
            "reward_mode": "aggregate_v3",
            "stage": "C0",
            "samples": samples,
            "optimizer_updates": args.updates,
            "start": start,
            "environment": contract,
            "initialization": {**initialization, "selection": selected},
            "finite": finite,
            "actor_parameter_update_l2": update_norm,
            "ppo": [item["ppo"] for item in metrics],
            "residual_action_abs_mean": residual_action_abs_mean,
            "table_contact_observed": support_contact_observed,
            "rollout_writes": writes,
        }
        _write(output / "smoke.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0 if passed else 2
    except BaseException as error:
        _write(
            output / "smoke_failure.json",
            {
                "schema_version": "Stage16FullTrajectoryPpoSmokeFailureV1",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
