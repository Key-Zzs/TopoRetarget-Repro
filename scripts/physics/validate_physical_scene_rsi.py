#!/usr/bin/env python3
"""P3-B.6 physical scene, reference validity, and RSI requalification.

The offline phase is the authority for all 321 reference rows.  The optional
Isaac phases add real PhysX reset and joint replay evidence using the same
finite support actor, nominal gravity/friction, zero residual actions, and no
rollout object or wrist-root state writes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.physical_scene_rsi import (  # noqa: E402
    CLIPS,
    build_physical_reference_validity_mask,
    evaluate_physical_pose_geometry,
    load_table_proxy,
    sha256_file,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_p3b6_scene_rsi_requalification"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
SOURCE_CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
GEOMETRY_MANIFEST = (
    REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/"
    "runtime_collision_geometry_manifest.json"
)
SUPPORT_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction/inference"
SUPPORT_ASSET_ROOT = REPO_ROOT / ".local/support_assets/hocap"
P3B5_ROOT = REPO_ROOT / ".local/reports/stage16_p3b5_geometry_attribution"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("offline", "dynamic", "joint", "all"), default="all")
    parser.add_argument("--clip", choices=CLIPS, default=None)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--dynamic-max-states", type=int, default=0)
    parser.add_argument("--dynamic-start", type=int, default=0)
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--joint-start", type=int, default=None)
    parser.add_argument("--joint-replicas", type=int, default=4)
    return parser.parse_args()


def _write_parquet(path: Path, rows: dict[str, np.ndarray]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("PHYSICAL_VALIDITY_PARQUET_WRITER_UNAVAILABLE") from exc
    columns: dict[str, Any] = {}
    for name, value in rows.items():
        array = np.asarray(value)
        if array.ndim == 1:
            columns[name] = array.tolist()
        else:
            columns[name] = [item.tolist() for item in array]
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path)


def _plot_clip(
    *,
    clip: str,
    rows: dict[str, np.ndarray],
    output: Path,
    geometry_manifest: Path,
    repo_root: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from toporetarget.physics.support.runtime_support import table_top_corners
    from toporetarget.physics.support.types import (
        FinitePlanarSupportProxy,
        NominalSupportMaterialV1,
    )
    from toporetarget.rl.geometry_audit.runtime_geometry import load_runtime_geometry_manifest
    from toporetarget.rl.geometry_audit.transforms import transform_points

    hand_proxies, object_map = load_runtime_geometry_manifest(geometry_manifest)
    object_proxies = object_map[clip]
    wrist = np.asarray(rows["reference_wrist_pose"], dtype=np.float64)
    finger_q = np.asarray(rows["reference_q_finger"], dtype=np.float64)
    hand = reconstruct_hand_collision_body_pose(wrist, finger_q, repo_root=repo_root)
    object_pose = np.asarray(rows["reference_object_pose"], dtype=np.float64)
    proxy_payload = load_table_proxy(SUPPORT_ROOT / clip / "table_proxy.json")
    table = FinitePlanarSupportProxy(
        table_pose=tuple(proxy_payload["table_pose"]),
        table_extent=tuple(proxy_payload["table_extent"]),
        table_thickness=float(proxy_payload["table_thickness"]),
        plane_normal=tuple(proxy_payload["plane_normal"]),
        plane_offset=float(proxy_payload["plane_offset"]),
        material=NominalSupportMaterialV1(),
    )
    indices = [0, int(np.flatnonzero(rows["overall_reference_geometry_valid"])[-1])]
    invalid = np.flatnonzero(~rows["overall_reference_geometry_valid"])
    if invalid.size:
        indices.append(int(invalid[0]))
    indices = list(dict.fromkeys(indices))
    for index in indices:
        fig = plt.figure(figsize=(8, 7))
        axis = fig.add_subplot(111, projection="3d")
        for body_index, proxy in enumerate(hand_proxies):
            points = transform_points(proxy.scaled_vertices, proxy.local_pose_xyz_wxyz)
            points = transform_points(points, hand[index, body_index])
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="tab:blue", alpha=0.35)
        for proxy in object_proxies:
            points = transform_points(proxy.scaled_vertices, proxy.local_pose_xyz_wxyz)
            points = transform_points(points, object_pose[index])
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c="tab:orange", alpha=0.65)
        corners = table_top_corners(table)
        corners = np.vstack((corners, corners[0]))
        axis.plot(corners[:, 0], corners[:, 1], corners[:, 2], c="tab:green", linewidth=2.0)
        axis.set_title(
            f"{clip} frame {index} | {rows['support_state'][index]} | "
            f"valid={bool(rows['overall_reference_geometry_valid'][index])}"
        )
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.set_zlabel("world z (m)")
        axis.view_init(elev=24, azim=-65)
        fig.tight_layout()
        fig.savefig(output / f"{clip}_frame_{index:03d}.png", dpi=150)
        plt.close(fig)


def run_offline() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if not GEOMETRY_MANIFEST.is_file():
        raise FileNotFoundError(f"PHYSICAL_GEOMETRY_MANIFEST_MISSING:{GEOMETRY_MANIFEST}")
    frozen = {
        "schema_version": "PhysicalSceneRSIFrozenInputsV1",
        "branch": os.popen("git branch --show-current", "r").read().strip(),
        "head": os.popen("git rev-parse HEAD", "r").read().strip(),
        "geometry_manifest": _receipt(GEOMETRY_MANIFEST),
        "p3b5_geometry_contract": _receipt(P3B5_ROOT / "geometry_contract.json"),
        "p3b5_decision_contract": _receipt(P3B5_ROOT / "decision_contract.json"),
        "clips": {},
        "forbidden": [
            "reference_ghost_as_formal_geometry",
            "tracked_points_as_hand_geometry",
            "moving_or_disabling_table",
            "guidance_or_attachment",
            "rollout_object_state_write",
            "rollout_wrist_root_write",
            "geometry_threshold_mutation",
            "old_rsi_blacklist",
        ],
    }
    _write_json(REPORT_ROOT / "frozen_inputs.json", frozen)
    summaries: dict[str, Any] = {}
    for clip in CLIPS:
        reference = REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz"
        source = SOURCE_CONTACT_ROOT / clip / "source_contact_evidence_runtime.npz"
        table = SUPPORT_ROOT / clip / "table_proxy.json"
        rows, safe_bank, summary = build_physical_reference_validity_mask(
            clip=clip,
            reference_path=reference,
            source_contact_evidence_path=source,
            geometry_manifest_path=GEOMETRY_MANIFEST,
            table_proxy_path=table,
            repo_root=REPO_ROOT,
        )
        clip_root = REPORT_ROOT / clip
        clip_root.mkdir(parents=True, exist_ok=True)
        _write_parquet(clip_root / "physical_reference_validity_mask.parquet", rows)
        np.savez_compressed(clip_root / "physical_reference_validity_mask.npz", **rows)
        np.savez_compressed(clip_root / "physical_safe_rsi_bank.npz", **safe_bank)
        _write_json(clip_root / "trajectory_geometry_qualification.json", summary["trajectory"])
        _write_json(clip_root / "coverage_gate.json", summary["physical_safe_rsi"])
        _plot_clip(
            clip=clip,
            rows=rows,
            output=clip_root / "screenshots",
            geometry_manifest=GEOMETRY_MANIFEST,
            repo_root=REPO_ROOT,
        )
        frozen["clips"][clip] = {
            "reference": _receipt(reference),
            "source_contact_evidence": _receipt(source),
            "table_proxy": _receipt(table),
            "support_asset": _receipt(SUPPORT_ASSET_ROOT / clip / "support_proxy.usda"),
        }
        summaries[clip] = summary
    _write_json(REPORT_ROOT / "frozen_inputs.json", frozen)
    _write_json(
        REPORT_ROOT / "offline_summary.json",
        {
            clip: summary["trajectory"] | {"coverage": summary["physical_safe_rsi"]}
            for clip, summary in summaries.items()
        },
    )
    return summaries


def _state_pose(env: Any, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = env.scene.env_origins
    wrist_position = state["wrist_position_scene"] + origins
    object_position = state["object_position_scene"] + origins
    wrist_pose = np.concatenate(
        [
            wrist_position.detach().cpu().numpy(),
            state["wrist_quaternion_wxyz"].detach().cpu().numpy(),
        ],
        axis=1,
    )
    object_pose = np.concatenate(
        [
            object_position.detach().cpu().numpy(),
            state["object_quaternion_wxyz"].detach().cpu().numpy(),
        ],
        axis=1,
    )
    finger_q = state["finger_q"].detach().cpu().numpy()
    return wrist_pose, object_pose, finger_q


def _support_contact(sensor: Any) -> np.ndarray:
    forces = sensor.data.force_matrix_w
    if forces is None:
        raise RuntimeError("PHYSICAL_SUPPORT_SENSOR_FORCE_MATRIX_UNAVAILABLE")
    values = forces.detach()
    while values.ndim > 2:
        values = values.sum(dim=-2)
    return (values.norm(dim=-1) > 1.0e-4).cpu().numpy().astype(bool)


def _make_physical_env(*, clip: str, count: int, start_indices: tuple[int, ...]) -> tuple[Any, Any]:
    """Create the reference env with two explicit finite support actors."""

    import isaaclab.sim as sim_utils
    import torch
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

    @configclass
    class PhysicalSceneCfg(InteractiveSceneCfg):
        support_170105: RigidObjectCfg | None = None
        support_170650: RigidObjectCfg | None = None
        object_170105_support_contact: ContactSensorCfg | None = None
        object_170650_support_contact: ContactSensorCfg | None = None

    def support_cfg(support_clip: str, name: str) -> RigidObjectCfg:
        proxy = load_table_proxy(SUPPORT_ROOT / support_clip / "table_proxy.json")
        normal = np.asarray(proxy["plane_normal"], dtype=np.float64)
        center = np.asarray(proxy["table_pose"][:3], dtype=np.float64) - (
            0.5 * float(proxy["table_thickness"]) * normal
        )
        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(SUPPORT_ASSET_ROOT / support_clip / "support_proxy.usda"),
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

    class PhysicalSupportPPOEnv(IsaacPPO26DReferenceTrackingEnv):
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

    cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo_cfg.configure_stage16d_ppo26d(cfg, num_envs=count, clip=clip, rsi=False, critical_dr=False)
    ppo_cfg.configure_stage16d_reference_kinematics_v2(cfg, reference_root=REFERENCE_ROOT)
    cfg.scene = PhysicalSceneCfg(
        num_envs=count,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=False,
        lazy_sensor_update=True,
        support_170105=support_cfg("hocap_170105", "Support170105"),
        support_170650=support_cfg("hocap_170650", "Support170650"),
        object_170105_support_contact=support_sensor("Object170105", "Support170105"),
        object_170650_support_contact=support_sensor("Object170650", "Support170650"),
    )
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.sim.physics_material.static_friction = 0.8
    cfg.sim.physics_material.dynamic_friction = 0.6
    cfg.sim.physics_material.restitution = 0.0
    cfg.object_170105.spawn.rigid_props.disable_gravity = clip != "hocap_170105"
    cfg.object_170650.spawn.rigid_props.disable_gravity = clip != "hocap_170650"
    cfg.stage16d_fixed_clip = clip
    cfg.reset_reference_index = "frame0"
    cfg.evaluation_reset_reference_indices = start_indices
    cfg.stage16_support_mode = "finite_inferred_table_proxy_v1"
    cfg.stage16_external_guidance = False
    cfg.scene.lazy_sensor_update = True
    return PhysicalSupportPPOEnv(cfg), torch


def run_dynamic(
    *,
    summaries: dict[str, Any],
    requested_clip: str,
    max_states: int,
    state_offset: int,
    steps: int,
) -> dict[str, Any]:
    if not summaries:
        raise RuntimeError("PHYSICAL_DYNAMIC_REQUIRES_OFFLINE_SUMMARIES")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    try:
        app = AppLauncher(headless=True).app
    except SystemExit as exc:
        raise RuntimeError(f"PHYSICAL_DYNAMIC_APP_LAUNCH_SYSTEM_EXIT:{exc.code}") from exc
    outputs: dict[str, Any] = {}
    try:
        for clip in (requested_clip,):
            bank_path = REPORT_ROOT / clip / "physical_safe_rsi_bank.npz"
            with np.load(bank_path, allow_pickle=False) as archive:
                states = np.asarray(archive["runtime_index"], dtype=np.int64)
                sem = np.asarray(archive["semantic_class"]).astype("U24")
                support = np.asarray(archive["support_state"]).astype("U24")
            if max_states > 0:
                states = states[state_offset : state_offset + max_states]
                sem = sem[state_offset : state_offset + max_states]
                support = support[state_offset : state_offset + max_states]
            else:
                states = states[state_offset:]
                sem = sem[state_offset:]
                support = support[state_offset:]
            if not len(states):
                outputs[clip] = {"status": "NO_OFFLINE_GEOMETRY_SAFE_STATES"}
                continue
            start_indices = tuple(int(index) for index in states for _ in range(4))
            env, torch = _make_physical_env(
                clip=clip, count=len(start_indices), start_indices=start_indices
            )
            try:
                env.reset(seed=20260814)
                initial = _state_pose(env, env._state())
                contact_steps = np.full(len(start_indices), -1, dtype=np.int64)
                support_contact_steps = np.zeros(len(start_indices), dtype=np.int64)
                max_displacement_before_hand_contact = np.zeros(
                    len(start_indices), dtype=np.float64
                )
                max_downward_before_hand_contact = np.zeros(len(start_indices), dtype=np.float64)
                terminated_flags = np.zeros(len(start_indices), dtype=bool)
                timed_out_flags = np.zeros(len(start_indices), dtype=bool)
                first_termination_step = np.full(len(start_indices), -1, dtype=np.int64)
                first_termination_reason = np.full(len(start_indices), "", dtype="U64")
                for step in range(steps):
                    _, _, terminated, timed_out, extras = env.step(
                        torch.zeros((len(start_indices), 26), device=env.device)
                    )
                    state = env._state()
                    _, current_object, _ = _state_pose(env, state)
                    displacement = np.asarray(
                        [
                            math.sqrt(
                                sum(
                                    float(delta) * float(delta)
                                    for delta in current_object[index, :3] - initial[1][index, :3]
                                )
                            )
                            for index in range(len(start_indices))
                        ],
                        dtype=np.float64,
                    )
                    downward = np.asarray(
                        [
                            max(0.0, float(initial[1][index, 2] - current_object[index, 2]))
                            for index in range(len(start_indices))
                        ],
                        dtype=np.float64,
                    )
                    for index in range(len(start_indices)):
                        if contact_steps[index] < 0:
                            max_displacement_before_hand_contact[index] = max(
                                max_displacement_before_hand_contact[index], displacement[index]
                            )
                            max_downward_before_hand_contact[index] = max(
                                max_downward_before_hand_contact[index], downward[index]
                            )
                    hand_contact = (
                        extras["ppo26d"]["contact_any"].detach().cpu().numpy().astype(bool)
                    )
                    support_sensor = env.scene[
                        "object_170105_support_contact"
                        if clip == "hocap_170105"
                        else "object_170650_support_contact"
                    ]
                    support_contact = _support_contact(support_sensor)
                    newly = (contact_steps < 0) & hand_contact
                    contact_steps[newly] = step
                    support_contact_steps += support_contact.astype(np.int64)
                    terminated_np = terminated.detach().cpu().numpy().astype(bool)
                    timed_out_np = timed_out.detach().cpu().numpy().astype(bool)
                    reason_codes = (
                        extras["ppo26d"]["primary_reason_code"].detach().cpu().numpy().astype(int)
                    )
                    reason_labels = tuple(extras["ppo26d"]["termination_reasons"])
                    first_event = (first_termination_step < 0) & (terminated_np | timed_out_np)
                    first_termination_step[first_event] = step
                    first_termination_reason[first_event] = np.asarray(
                        [reason_labels[code] for code in reason_codes[first_event]], dtype="U64"
                    )
                    terminated_flags |= terminated_np
                    timed_out_flags |= timed_out_np
                writes = env.rollout_state_write_report()
                rows: list[dict[str, Any]] = []
                for state_index, frame in enumerate(states):
                    for replica in range(4):
                        index = state_index * 4 + replica
                        displacement = float(max_displacement_before_hand_contact[index])
                        downward = float(max_downward_before_hand_contact[index])
                        persist = (
                            int(max(0, steps - int(contact_steps[index])))
                            if contact_steps[index] >= 0
                            else 0
                        )
                        source_class = str(sem[state_index])
                        if source_class == "PRE_CONTACT":
                            safe = (
                                not terminated_flags[index]
                                and not timed_out_flags[index]
                                and support_contact_steps[index] >= steps
                                and displacement <= 0.01
                                and downward <= 0.01
                            )
                        else:
                            safe = (
                                not terminated_flags[index]
                                and not timed_out_flags[index]
                                and contact_steps[index] >= 0
                                and persist >= 3
                                and displacement <= 0.01
                                and downward <= 0.01
                            )
                        rows.append(
                            {
                                "runtime_index": int(frame),
                                "replica": replica,
                                "semantic_class": source_class,
                                "support_state": str(support[state_index]),
                                "first_hand_contact_step": int(contact_steps[index]),
                                "support_contact_steps": int(support_contact_steps[index]),
                                "object_displacement_before_hand_contact_m": displacement,
                                "object_downward_displacement_before_hand_contact_m": downward,
                                "contact_persistence_control_steps": persist,
                                "terminated": bool(terminated_flags[index]),
                                "timed_out": bool(timed_out_flags[index]),
                                "terminated_or_timed_out": bool(
                                    terminated_flags[index] or timed_out_flags[index]
                                ),
                                "first_termination_step": int(first_termination_step[index]),
                                "first_termination_reason": str(first_termination_reason[index]),
                                "gravity_safe": bool(safe),
                            }
                        )
                safe_by_state = {
                    int(frame): all(
                        row["gravity_safe"] for row in rows if row["runtime_index"] == int(frame)
                    )
                    for frame in states
                }
                output = {
                    "schema_version": "PhysicalSafeRSIDynamicResetQualificationV1",
                    "clip": clip,
                    "state_count": len(states),
                    "replicas_per_state": 4,
                    "control_steps": steps,
                    "executed_control_steps": steps,
                    "gravity_world_mps2": [0.0, 0.0, -9.81],
                    "support_mode": "finite_inferred_table_proxy_v1",
                    "external_guidance": False,
                    "rollout_object_state_writes": int(writes["object_rollout_state_writes"]),
                    "rollout_wrist_root_state_writes": int(
                        writes["wrist_root_state_writes_during_step"]
                    ),
                    "all_replicas_write_gate_pass": bool(
                        writes["object_rollout_state_writes"] == 0
                        and writes["wrist_root_state_writes_during_step"] == 0
                    ),
                    "safe_state_count": int(sum(safe_by_state.values())),
                    "safe_state_indices": [
                        index for index, value in safe_by_state.items() if value
                    ],
                    "terminated_replica_count": int(terminated_flags.sum()),
                    "timed_out_replica_count": int(timed_out_flags.sum()),
                    "rows": rows,
                }
                suffix = f"{state_offset:04d}_{state_offset + len(states) - 1:04d}"
                _write_json(
                    REPORT_ROOT / clip / f"dynamic_reset_qualification_{suffix}.json", output
                )
                np.savez_compressed(
                    REPORT_ROOT / clip / f"dynamic_safe_reset_indices_{suffix}.npz",
                    runtime_index=np.asarray(output["safe_state_indices"], dtype=np.int64),
                )
                outputs[clip] = merge_dynamic_reports(clip)
            finally:
                env.close()
    finally:
        app.close()
    return outputs


def merge_dynamic_reports(clip: str) -> dict[str, Any]:
    """Merge chunked dynamic evidence and materialize the dynamic-safe bank."""

    reports = sorted((REPORT_ROOT / clip).glob("dynamic_reset_qualification_*.json"))
    if not reports:
        raise RuntimeError(f"PHYSICAL_DYNAMIC_CHUNKS_MISSING:{clip}")
    rows: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    payloads: list[tuple[Path, dict[str, Any], set[int]]] = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_rows = payload.get("rows", [])
        payloads.append((path, payload, {int(row["runtime_index"]) for row in payload_rows}))
    selected_payloads: list[tuple[Path, dict[str, Any], set[int]]] = []
    covered: set[int] = set()
    ignored_overlaps: list[str] = []
    # A one-state smoke run may overlap a later full chunk.  Prefer the chunk
    # covering more reference states so the aggregate remains one row per
    # state/replica while retaining the smoke receipt on disk.
    for path, payload, indices in sorted(
        payloads, key=lambda item: (len(item[2]), item[0].name), reverse=True
    ):
        if indices & covered:
            ignored_overlaps.append(str(path.resolve()))
            continue
        selected_payloads.append((path, payload, indices))
        covered.update(indices)
    for path, payload, _indices in sorted(selected_payloads, key=lambda item: item[0].name):
        rows.extend(payload.get("rows", []))
        chunk_summaries.append(
            {
                "path": str(path.resolve()),
                "state_count": payload.get("state_count"),
                "safe_state_count": payload.get("safe_state_count"),
                "rollout_object_state_writes": payload.get("rollout_object_state_writes"),
                "rollout_wrist_root_state_writes": payload.get("rollout_wrist_root_state_writes"),
            }
        )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["runtime_index"]), []).append(row)
    safe_indices = sorted(
        index
        for index, state_rows in grouped.items()
        if len(state_rows) == 4 and all(bool(row["gravity_safe"]) for row in state_rows)
    )
    with np.load(REPORT_ROOT / clip / "physical_safe_rsi_bank.npz", allow_pickle=False) as archive:
        offline = {name: np.asarray(archive[name]) for name in archive.files}
    keep = np.isin(np.asarray(offline["runtime_index"], dtype=np.int64), safe_indices)
    dynamic_bank = {
        name: value[keep] if value.ndim > 0 and len(value) == len(keep) else value
        for name, value in offline.items()
    }
    dynamic_bank["dynamic_reset_qualified"] = np.ones(int(keep.sum()), dtype=bool)
    np.savez_compressed(REPORT_ROOT / clip / "physical_safe_rsi_bank_dynamic.npz", **dynamic_bank)
    aggregate = {
        "schema_version": "PhysicalSafeRSIDynamicResetAggregateV1",
        "clip": clip,
        "chunk_count": len(chunk_summaries),
        "chunks": chunk_summaries,
        "ignored_overlapping_chunks": ignored_overlaps,
        "dynamic_rows": len(rows),
        "dynamic_state_count": len(grouped),
        "dynamic_safe_state_count": len(safe_indices),
        "dynamic_safe_state_indices": safe_indices,
        "all_replicas_write_gate_pass": all(
            int(chunk["rollout_object_state_writes"]) == 0
            and int(chunk["rollout_wrist_root_state_writes"]) == 0
            for chunk in chunk_summaries
        ),
        "external_guidance": False,
        "support_mode": "finite_inferred_table_proxy_v1",
        "gravity_world_mps2": [0.0, 0.0, -9.81],
    }
    _write_json(REPORT_ROOT / clip / "dynamic_reset_qualification.json", aggregate)
    return aggregate


def run_joint(
    *, start_indices: dict[str, int], requested_clip: str, replicas: int
) -> dict[str, Any]:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    try:
        app = AppLauncher(headless=True).app
    except SystemExit as exc:
        raise RuntimeError(f"PHYSICAL_JOINT_APP_LAUNCH_SYSTEM_EXIT:{exc.code}") from exc
    result: dict[str, Any] = {}
    try:
        for clip in (requested_clip,):
            start = int(start_indices[clip])
            if start < 0 or start >= 321:
                raise ValueError("PHYSICAL_JOINT_START_OUT_OF_RANGE")
            starts = (start,) * replicas
            env, torch = _make_physical_env(clip=clip, count=replicas, start_indices=starts)
            try:
                env.reset(seed=20260814)
                wrist_rows: list[np.ndarray] = []
                object_rows: list[np.ndarray] = []
                finger_rows: list[np.ndarray] = []
                twist_rows: list[np.ndarray] = []
                action_rows: list[np.ndarray] = []
                wrist, obj, finger = _state_pose(env, env._state())
                wrist_rows.append(wrist)
                object_rows.append(obj)
                finger_rows.append(finger)
                twist_rows.append(env._state()["object_twist_world"].detach().cpu().numpy())
                action_rows.append(np.zeros((replicas, 26), dtype=np.float32))
                termination_step: int | None = None
                termination_reason: str | None = None
                for step in range(321 - start - 1):
                    _, _, terminated, timed_out, extras = env.step(
                        torch.zeros((replicas, 26), device=env.device)
                    )
                    state = env._state()
                    wrist, obj, finger = _state_pose(env, state)
                    wrist_rows.append(wrist)
                    object_rows.append(obj)
                    finger_rows.append(finger)
                    twist_rows.append(state["object_twist_world"].detach().cpu().numpy())
                    action_rows.append(np.zeros((replicas, 26), dtype=np.float32))
                    if bool((terminated | timed_out).any()):
                        termination_step = step + 1
                        reason_codes = (
                            extras["ppo26d"]["primary_reason_code"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        )
                        reason_labels = tuple(extras["ppo26d"]["termination_reasons"])
                        termination_reason = ",".join(
                            sorted({reason_labels[code] for code in reason_codes})
                        )
                        break
                writes = env.rollout_state_write_report()
                trace = {
                    "wrist_pose": np.asarray(wrist_rows),
                    "object_pose": np.asarray(object_rows),
                    "finger_q": np.asarray(finger_rows),
                    "object_twist_world": np.asarray(twist_rows),
                    "action": np.asarray(action_rows),
                    "reference_start_index": np.asarray(start, dtype=np.int64),
                    "reference_kinematics_version": np.asarray(2, dtype=np.int64),
                }
                np.savez_compressed(REPORT_ROOT / clip / "joint_zero_replay_trace.npz", **trace)
                geometry = evaluate_physical_pose_geometry(
                    clip=clip,
                    wrist_pose=trace["wrist_pose"],
                    finger_q=trace["finger_q"],
                    object_pose=trace["object_pose"],
                    geometry_manifest_path=GEOMETRY_MANIFEST,
                    table_proxy_path=SUPPORT_ROOT / clip / "table_proxy.json",
                    repo_root=REPO_ROOT,
                )
                geometry_path = REPORT_ROOT / clip / "joint_zero_replay_geometry.npz"
                np.savez_compressed(geometry_path, **geometry)
                geometry_summary = {
                    "trace_frames": int(trace["wrist_pose"].shape[0]),
                    "hand_object_max_m": float(np.max(geometry["hand_object_max_penetration_m"])),
                    "hand_table_max_m": float(np.max(geometry["hand_table_max_penetration_m"])),
                    "object_table_max_m": float(np.max(geometry["object_table_max_penetration_m"])),
                    "inter_finger_max_m": float(np.max(geometry["inter_finger_max_penetration_m"])),
                    "geometry_gate_pass": bool(
                        np.max(geometry["hand_object_max_penetration_m"]) < 0.010
                        and np.max(geometry["hand_table_max_penetration_m"]) <= 0.002
                        and np.max(geometry["object_table_max_penetration_m"]) <= 0.002
                        and np.max(geometry["inter_finger_max_penetration_m"]) <= 0.003
                    ),
                    "geometry_path": str(geometry_path.resolve()),
                }
                output = {
                    "schema_version": "PhysicalJointZeroReplayQualificationV1",
                    "clip": clip,
                    "start_reference_index": start,
                    "replicas": replicas,
                    "executed_control_steps": int(len(wrist_rows) - 1),
                    "expected_control_steps": 321 - start - 1,
                    "termination_step": termination_step,
                    "full_frame_zero_replay_authorized": bool(
                        start == 0
                        and termination_step is None
                        and len(wrist_rows) - 1 == 321 - start - 1
                    ),
                    "full_frame_zero_replay_status": (
                        "AUTHORIZED_FROM_EARLIEST_PHYSICALLY_VALID_PRE_CONTACT"
                        if (
                            start == 0
                            and termination_step is None
                            and len(wrist_rows) - 1 == 321 - start - 1
                        )
                        else "FULL_FRAME_ZERO_REPLAY_NOT_AUTHORIZED"
                    ),
                    "termination_reason": termination_reason,
                    "external_guidance": False,
                    "rollout_object_state_writes": int(writes["object_rollout_state_writes"]),
                    "rollout_wrist_root_state_writes": int(
                        writes["wrist_root_state_writes_during_step"]
                    ),
                    "zero_action_max_abs": float(np.max(np.abs(trace["action"]))),
                    "trace_path": str(
                        (REPORT_ROOT / clip / "joint_zero_replay_trace.npz").resolve()
                    ),
                    "geometry": geometry_summary,
                }
                _write_json(REPORT_ROOT / clip / "joint_zero_replay_qualification.json", output)
                result[clip] = output
            finally:
                env.close()
    finally:
        app.close()
    return result


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    summaries: dict[str, Any] = {}
    if args.phase in {"offline", "all", "dynamic", "joint"}:
        offline_path = REPORT_ROOT / "offline_summary.json"
        if args.phase == "offline" or not offline_path.is_file():
            summaries = run_offline()
        else:
            payload = json.loads(offline_path.read_text(encoding="utf-8"))
            summaries = {
                clip: {"trajectory": value, "physical_safe_rsi": value.get("coverage", {})}
                for clip, value in payload.items()
            }
    dynamic: dict[str, Any] = {}
    joint: dict[str, Any] = {}
    if args.phase in {"dynamic", "all"}:
        if args.clip is None:
            raise ValueError("PHYSICAL_DYNAMIC_REQUIRES_ONE_CLIP_PER_ISAAC_PROCESS")
        dynamic = run_dynamic(
            summaries=summaries,
            requested_clip=args.clip,
            max_states=args.dynamic_max_states,
            state_offset=args.dynamic_start,
            steps=args.dynamic_steps,
        )
    if args.phase in {"joint", "all"}:
        if args.clip is None:
            raise ValueError("PHYSICAL_JOINT_REQUIRES_ONE_CLIP_PER_ISAAC_PROCESS")
        starts: dict[str, int] = {}
        for clip in (args.clip,):
            with np.load(
                REPORT_ROOT / clip / "physical_reference_validity_mask.npz", allow_pickle=False
            ) as archive:
                valid = np.asarray(archive["overall_reference_geometry_valid"], dtype=bool)
                semantic = np.asarray(archive["semantic_class"]).astype("U24")
            candidates = np.flatnonzero(valid & (semantic == "PRE_CONTACT"))
            starts[clip] = int(candidates[0]) if candidates.size else int(np.flatnonzero(valid)[0])
        if args.joint_start is not None:
            starts = {args.clip: int(args.joint_start)}
        joint = run_joint(
            start_indices=starts,
            requested_clip=args.clip,
            replicas=args.joint_replicas,
        )
    final = {
        "schema_version": "PhysicalSceneRSIRequalificationV1",
        "offline": {clip: summary["trajectory"] for clip, summary in summaries.items()},
        "dynamic": {
            clip: {key: value for key, value in report.items() if key not in {"rows"}}
            for clip, report in dynamic.items()
        },
        "joint": joint,
        "wall_time_s": time.monotonic() - started,
        "ppo_started": False,
    }
    _write_json(REPORT_ROOT / "final_summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
