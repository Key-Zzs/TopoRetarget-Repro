#!/usr/bin/env python3
"""Inspect Stage-16B world-wrist policies in MuJoCo or a truthful fallback."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from toporetarget.rl.environments.world_wrist_backend import (
    WorldWristFingerBackend,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1, quaternion_wxyz_from_matrix
from toporetarget.rl.world_wrist_oracle import (
    AdaptiveMultiHorizonContactOracle,
    WorldWristFingerObjectAwareOracle,
)

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
OVERLAY_NAMES = (
    "show_reference_wrist",
    "show_reference_ghost",
    "show_world_frame",
    "show_wrist_frame",
    "show_axis_points",
    "show_tracked_links",
    "show_contacts",
    "show_contact_forces",
    "show_wrist_wrench",
    "show_base_target",
)


class AdaptiveOracleVisualizationPolicy:
    """Run the shared adaptive oracle or replay one of its immutable traces."""

    def __init__(
        self,
        backend: WorldWristFingerBackend,
        *,
        clip_name: str,
        action_trace: Path | None,
        selection_trace: Path | None,
    ) -> None:
        self.backend = backend
        self.oracle = None if action_trace is not None else AdaptiveMultiHorizonContactOracle()
        self.actions = (
            None
            if action_trace is None
            else np.asarray(np.load(action_trace, allow_pickle=False)["actions"], dtype=np.float64)
        )
        self.selection_rows: list[dict[str, Any]] = []
        if selection_trace is not None:
            self.selection_rows = [
                row
                for line in selection_trace.read_text(encoding="utf-8").splitlines()
                if (row := json.loads(line)).get("clip", clip_name) == clip_name
            ]
        self.index = 0
        self.last_metadata: dict[str, Any] | None = None
        self.horizon_counts: dict[str, int] = {}
        self.trace_source = "live_adaptive_oracle" if action_trace is None else "action_only_replay"

    @staticmethod
    def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
        selected_horizon = int(row["selected_requested_horizon"])
        selected = next(
            candidate
            for candidate in row.get("candidates", [])
            if int(candidate["requested_horizon"]) == selected_horizon
        )
        diagnostics = selected["diagnostics"]
        return {
            "remaining": int(row["remaining"]),
            "selected_horizon": selected_horizon,
            "effective_horizon": int(row["selected_effective_horizon"]),
            "candidate_gate_values": {
                f"H{int(candidate['requested_horizon'])}": float(
                    candidate["diagnostics"]["predicted_gate_violation"]
                )
                for candidate in row.get("candidates", [])
            },
            "gate_violation": float(diagnostics["predicted_gate_violation"]),
            "gate_margin": float(diagnostics["minimum_gate_margin"]),
            "predicted_termination": diagnostics["predicted_termination"],
            "selection_reason": row["reason"],
        }

    def __call__(self, _state: dict[str, np.ndarray]) -> np.ndarray:
        if self.actions is not None:
            if self.index >= len(self.actions):
                raise ValueError("adaptive oracle action trace ended before the rollout")
            action = self.actions[self.index].copy()
            self.last_metadata = (
                self._metadata_from_row(self.selection_rows[self.index])
                if self.index < len(self.selection_rows)
                else None
            )
        else:
            assert self.oracle is not None
            action = self.oracle.action(self.backend)
            self.last_metadata = self._metadata_from_row(self.oracle.selection_trace[-1])
        if self.last_metadata is not None:
            key = f"H{self.last_metadata['selected_horizon']}"
            self.horizon_counts[key] = self.horizon_counts.get(key, 0) + 1
        self.index += 1
        return action


def _profile(
    controller_path: Path | None, action_scale_path: Path | None
) -> tuple[WristImpedanceProfileV1, WristFingerActionScaleV1]:
    impedance = WristImpedanceProfileV1()
    scale = WristFingerActionScaleV1()
    if controller_path is not None:
        report = json.loads(controller_path.read_text(encoding="utf-8"))
        selected = report["selected"]["profile"]
        impedance = WristImpedanceProfileV1(
            translation_stiffness_npm=float(selected["translation_stiffness_npm"]),
            translation_damping_ratio=float(selected["translation_damping_ratio"]),
            rotation_stiffness_nmprad=float(selected["rotation_stiffness_nmprad"]),
            rotation_damping_ratio=float(selected["rotation_damping_ratio"]),
            force_limit_n=float(selected["force_limit_n"]),
            torque_limit_nm=float(selected["torque_limit_nm"]),
            feedforward_twist_gain=float(selected["feedforward_twist_gain"]),
        )
    if action_scale_path is not None:
        report = json.loads(action_scale_path.read_text(encoding="utf-8"))
        selected = report["selected"]["scale"]
        scale = WristFingerActionScaleV1(
            translation_m=float(selected["translation_m"]),
            rotation_rad=float(selected["rotation_rad"]),
            finger_joint_range_fraction=float(selected["finger_joint_range_fraction"]),
        )
    return impedance, scale


def _backend(args: argparse.Namespace) -> WorldWristFingerBackend:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    reference = WorldWristFingerReferenceV1.from_npz(args.reference)
    impedance, scale = _profile(args.controller_report, args.action_scale_report)
    scene = materialize_world_wrist_free_object_scene(
        WUJI_MJCF, args.scene_root / args.reference.stem, object_mesh=args.object_mesh
    )
    return WorldWristFingerBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
        impedance_profile=impedance,
        action_scale=scale,
        seed=args.seed,
    )


def _policy(args: argparse.Namespace, backend: WorldWristFingerBackend) -> Any:
    if args.policy == "zero":
        return lambda _state: np.zeros(26, dtype=np.float64)
    if args.policy == "oracle":
        oracle = WorldWristFingerObjectAwareOracle()
        return lambda _state: oracle.action(backend, horizon=args.oracle_horizon)
    if args.policy == "adaptive-oracle":
        return AdaptiveOracleVisualizationPolicy(
            backend,
            clip_name=args.reference.stem,
            action_trace=args.adaptive_action_trace,
            selection_trace=args.adaptive_selection_trace,
        )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --policy ppo")
    trainer = PPOTrainer(backend.observation().size, 26, config=PPOConfig(), device=args.device)
    payload = load_checkpoint(args.checkpoint, map_location=trainer.device)
    if payload.get("action_dim") != 26:
        raise ValueError("checkpoint is not a 26-D Stage-16B policy")
    trainer.model.load_state_dict(payload["model"])
    trainer.normalizer.load_state_dict(payload["normalizer"])
    trainer.freeze_observation_normalizer()

    def policy(state: dict[str, np.ndarray]) -> np.ndarray:
        observation = torch.as_tensor(backend.observation(state), dtype=torch.float32)[None]
        with torch.no_grad():
            action, _, _ = trainer.act(observation, deterministic=args.deterministic)
        return np.clip(action[0].cpu().numpy(), -1.0, 1.0)

    return policy


def _summary(
    backend: WorldWristFingerBackend,
    frames: list[np.ndarray],
    reason: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = backend._state()  # noqa: SLF001 - rendering summary reads current simulator state
    index = backend.reference_index
    return {
        "policy": args.policy,
        "mode": args.mode,
        "frames": len(frames),
        "termination": reason,
        "reference_index": index,
        "wrist_position_error_m": float(
            np.linalg.norm(
                state["wrist_pose"][:3, 3] - backend.reference.wrist_pose_world_ref[index, :3, 3]
            )
        ),
        "object_position_error_m": float(
            np.linalg.norm(
                state["object_pose"][:3, 3] - backend.reference.object_pose_world_ref[index, :3, 3]
            )
        ),
        "contacts": backend.contact_summary(),
        "last_wrist_wrench": (
            None
            if backend.last_control is None
            else backend.last_control.applied_wrench_world.tolist()
        ),
        "overlays_requested": {name: bool(getattr(args, name)) for name in OVERLAY_NAMES},
        "diagnostic_hud_requested": {
            "show_selected_horizon": args.show_selected_horizon,
            "show_gate_margins": args.show_gate_margins,
        },
        "non_claim": "visual inspection only; world-wrist oracle is not PPO success",
    }


def _camera(backend: WorldWristFingerBackend, args: argparse.Namespace) -> Any:
    """Return either a named model camera or a deterministic scene camera."""

    if args.camera != "auto":
        camera_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if camera_id < 0:
            raise ValueError(f"unknown MuJoCo camera: {args.camera}")
        return camera_id
    state = backend._state()  # noqa: SLF001 - camera follows the visible task workspace
    reference = backend.reference
    target = reference.wrist_pose_world_ref[backend.reference_index, :3, 3]
    points = np.vstack(
        (
            state["wrist_pose"][:3, 3],
            state["object_pose"][:3, 3],
            target,
            reference.object_pose_world_ref[backend.reference_index, :3, 3],
        )
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = points.mean(axis=0)
    camera.distance = max(0.24, 1.4 * float(np.max(np.ptp(points, axis=0))) + 0.16)
    camera.azimuth = 135.0
    camera.elevation = -25.0
    return camera


def _write_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    """Write a compact frame sheet without depending on an implicit GUI backend."""

    if not images:
        return
    selected = images
    if len(images) > 12:
        selected = [images[index] for index in np.linspace(0, len(images) - 1, 12, dtype=int)]
    columns = min(4, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    height, width, channels = images[0].shape
    sheet = np.zeros((rows * height, columns * width, channels), dtype=images[0].dtype)
    for index, image in enumerate(selected):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    import imageio.v3 as iio

    iio.imwrite(output, sheet)


def _scene_geom(scene: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_marker(
    scene: mujoco.MjvScene,
    *,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> bool:
    geom = _scene_geom(scene)
    if geom is None:
        return False
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    return True


def _add_connector(
    scene: mujoco.MjvScene,
    *,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: tuple[float, float, float, float],
    arrow: bool = False,
) -> bool:
    geom = _scene_geom(scene)
    if geom is None:
        return False
    geom_type = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    return True


def _add_frame(
    scene: mujoco.MjvScene,
    pose: np.ndarray,
    *,
    length: float,
    alpha: float = 1.0,
) -> int:
    origin = np.asarray(pose[:3, 3], dtype=np.float64)
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    colors = ((1.0, 0.15, 0.15, alpha), (0.15, 1.0, 0.15, alpha), (0.2, 0.45, 1.0, alpha))
    return sum(
        _add_connector(
            scene,
            start=origin,
            end=origin + length * rotation[:, axis],
            width=0.004,
            rgba=colors[axis],
            arrow=True,
        )
        for axis in range(3)
    )


def _ghost_data(backend: WorldWristFingerBackend, index: int) -> mujoco.MjData:
    data = mujoco.MjData(backend.model)
    reference = backend.reference
    wrist_pose = reference.wrist_pose_world_ref[index]
    object_pose = reference.object_pose_world_ref[index]
    data.qpos[backend.wrist_qpos_address : backend.wrist_qpos_address + 7] = np.concatenate(
        (wrist_pose[:3, 3], quaternion_wxyz_from_matrix(wrist_pose[:3, :3]))
    )
    data.qpos[backend.finger_qpos_addresses] = reference.q_finger_ref[index]
    data.qpos[backend.object_qpos_address : backend.object_qpos_address + 7] = np.concatenate(
        (object_pose[:3, 3], quaternion_wxyz_from_matrix(object_pose[:3, :3]))
    )
    mujoco.mj_forward(backend.model, data)
    return data


def _add_overlays(
    scene: mujoco.MjvScene,
    backend: WorldWristFingerBackend,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Append actual MuJoCo geoms for every requested diagnostic overlay."""

    state = backend._state()  # noqa: SLF001 - visualization-only simulator read
    reference = backend.reference
    index = backend.reference_index
    counts = {name: 0 for name in OVERLAY_NAMES}

    if args.show_reference_ghost:
        first = scene.ngeom
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultPerturb(perturb)
        mujoco.mjv_addGeoms(
            backend.model,
            _ghost_data(backend, index),
            option,
            perturb,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        object_geom_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_GEOM, "stage16b_object_geom"
        )
        for geom_index in range(first, scene.ngeom):
            geom = scene.geoms[geom_index]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == object_geom_id:
                geom.rgba[3] = 0.0
                continue
            geom.rgba[:] = np.asarray([0.0, 0.85, 1.0, 0.28], dtype=np.float32)
            geom.transparent = 1
            geom.emission = 0.25
            counts["show_reference_ghost"] += 1

    if args.show_world_frame:
        counts["show_world_frame"] += _add_frame(scene, np.eye(4), length=0.08)
    if args.show_wrist_frame:
        counts["show_wrist_frame"] += _add_frame(scene, state["wrist_pose"], length=0.05)
    if args.show_reference_wrist:
        counts["show_reference_wrist"] += _add_frame(
            scene, reference.wrist_pose_world_ref[index], length=0.055, alpha=0.65
        )
    if args.show_base_target:
        counts["show_base_target"] += _add_connector(
            scene,
            start=state["wrist_pose"][:3, 3],
            end=reference.wrist_pose_world_ref[index, :3, 3],
            width=3.0,
            rgba=(1.0, 0.2, 1.0, 1.0),
        )
    if args.show_axis_points:
        for point in state["object_axis_points"]:
            counts["show_axis_points"] += _add_marker(
                scene, position=point, radius=0.004, rgba=(1.0, 0.45, 0.05, 1.0)
            )
        for point in reference.object_axis_points_world_ref[index]:
            counts["show_axis_points"] += _add_marker(
                scene, position=point, radius=0.003, rgba=(1.0, 1.0, 0.1, 0.75)
            )
    if args.show_tracked_links:
        for current, target in zip(
            state["links"], reference.tracked_link_positions_world_ref[index], strict=True
        ):
            counts["show_tracked_links"] += _add_marker(
                scene, position=current, radius=0.0025, rgba=(0.2, 1.0, 0.25, 0.9)
            )
            counts["show_tracked_links"] += _add_marker(
                scene, position=target, radius=0.002, rgba=(0.0, 0.9, 1.0, 0.75)
            )
            counts["show_tracked_links"] += _add_connector(
                scene,
                start=current,
                end=target,
                width=1.5,
                rgba=(0.7, 0.9, 1.0, 0.55),
            )
    if args.show_contacts or args.show_contact_forces:
        for contact_index in range(min(backend.data.ncon, 64)):
            contact = backend.data.contact[contact_index]
            if args.show_contacts:
                counts["show_contacts"] += _add_marker(
                    scene,
                    position=contact.pos,
                    radius=0.0035,
                    rgba=(1.0, 0.1, 0.1, 1.0),
                )
            if args.show_contact_forces:
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(backend.model, backend.data, contact_index, contact_force)
                force_world = contact.frame.reshape(3, 3).T @ contact_force[:3]
                norm = float(np.linalg.norm(force_world))
                if norm > 1e-9:
                    end = contact.pos + min(0.06, 0.002 * norm) * force_world / norm
                    counts["show_contact_forces"] += _add_connector(
                        scene,
                        start=contact.pos,
                        end=end,
                        width=0.004,
                        rgba=(1.0, 0.15, 0.05, 0.9),
                        arrow=True,
                    )
    if args.show_wrist_wrench and backend.last_control is not None:
        origin = state["wrist_pose"][:3, 3]
        wrench = backend.last_control.applied_wrench_world
        for vector, scale, color in (
            (wrench[:3], 0.003, (1.0, 0.2, 1.0, 0.95)),
            (wrench[3:], 0.04, (0.2, 0.8, 1.0, 0.95)),
        ):
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                counts["show_wrist_wrench"] += _add_connector(
                    scene,
                    start=origin,
                    end=origin + min(0.09, scale * norm) * vector / norm,
                    width=0.005,
                    rgba=color,
                    arrow=True,
                )
    return counts


def _annotate_frame(
    frame: np.ndarray,
    backend: WorldWristFingerBackend,
    *,
    policy_name: str,
    policy_metadata: dict[str, Any] | None,
    action: np.ndarray | None,
    reward: dict[str, float] | None,
    reason: str | None,
    show_selected_horizon: bool,
    show_gate_margins: bool,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    state = backend._state()  # noqa: SLF001 - visualization-only simulator read
    index = backend.reference_index
    reference = backend.reference
    position_error_cm = 100.0 * float(
        np.linalg.norm(state["wrist_pose"][:3, 3] - reference.wrist_pose_world_ref[index, :3, 3])
    )
    relative_rotation = (
        state["wrist_pose"][:3, :3].T @ reference.wrist_pose_world_ref[index, :3, :3]
    )
    rotation_error_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)))
    )
    object_error_cm = 100.0 * float(
        np.linalg.norm(state["object_pose"][:3, 3] - reference.object_pose_world_ref[index, :3, 3])
    )
    lines = [
        f"policy={policy_name}  frame={index}/{reference.frame_count - 1}",
        f"wrist error: {position_error_cm:.3f} cm  {rotation_error_deg:.3f} deg",
        f"object error: {object_error_cm:.3f} cm  contacts={backend.data.ncon}",
    ]
    if policy_metadata is not None and show_selected_horizon:
        lines.append(
            "adaptive: "
            f"H{policy_metadata['selected_horizon']}->H{policy_metadata['effective_horizon']}  "
            f"remaining={policy_metadata['remaining']}"
        )
        lines.append(f"reason={policy_metadata['selection_reason'].split(':', maxsplit=1)[0]}")
    if policy_metadata is not None and show_gate_margins:
        candidate_values = "  ".join(
            f"{name}={value:.3f}"
            for name, value in policy_metadata["candidate_gate_values"].items()
        )
        lines.append(
            f"candidate gates: {candidate_values}  "
            f"selected margin={policy_metadata['gate_margin']:.3f}"
        )
        if policy_metadata.get("predicted_termination") is not None:
            lines.append(f"predicted={policy_metadata['predicted_termination']}")
    if policy_name == "ppo" and action is not None:
        wrist, finger = action[:6], action[6:]
        lines.append(
            f"action: wrist|max|={np.max(np.abs(wrist)):.3f}  "
            f"finger|max|={np.max(np.abs(finger)):.3f}  "
            f"saturated={np.mean(np.abs(action) >= 0.999):.1%}"
        )
        if backend.last_control is not None:
            wrench = backend.last_control.applied_wrench_world
            lines.append(
                f"wrench: force={np.linalg.norm(wrench[:3]):.3f} N  "
                f"torque={np.linalg.norm(wrench[3:]):.3f} Nm"
            )
    if reward is not None:
        lines.append(
            "reward: "
            + "  ".join(
                f"{name}={reward[name]:.3f}"
                for name in (
                    "total",
                    "object",
                    "tracked_links",
                    "finger_joints",
                    "wrist_position",
                    "wrist_rotation",
                    "smoothness",
                )
            )
        )
    if reason is not None:
        lines.append(f"termination={reason}")
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    box_height = 18 * len(lines) + 12
    draw.rectangle((8, 8, min(frame.shape[1] - 8, 632), 8 + box_height), fill=(0, 0, 0, 175))
    for line_index, line in enumerate(lines):
        draw.text((16, 14 + 18 * line_index), line, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _headless(args: argparse.Namespace, backend: WorldWristFingerBackend, policy: Any) -> int:
    import imageio.v3 as iio

    args.output_frames.mkdir(parents=True, exist_ok=True)
    backend.reset(reference_index=args.start_frame)
    images: list[np.ndarray] = []
    trace: list[dict[str, float]] = []
    reason: str | None = None
    renderer: mujoco.Renderer | None = None
    renderer_error: Exception | None = None
    overlay_counts = {name: 0 for name in OVERLAY_NAMES}
    try:
        renderer = mujoco.Renderer(backend.model, height=args.height, width=args.width)
        renderer_status = "mujoco_offscreen"
    except Exception as exc:  # noqa: BLE001 - renderer is an optional platform capability
        renderer_error = exc
        renderer_status = "PASS_WITH_LIMITATION_NUMERICAL_FALLBACK"
    camera = _camera(backend, args)

    def render_frame(step: int, action: np.ndarray | None, reward: dict[str, float] | None) -> None:
        if renderer is None:
            return
        renderer.update_scene(backend.data, camera=camera)
        counts = _add_overlays(renderer.scene, backend, args)
        for name, count in counts.items():
            overlay_counts[name] += count
        frame = _annotate_frame(
            renderer.render().copy(),
            backend,
            policy_name=args.policy,
            policy_metadata=getattr(policy, "last_metadata", None),
            action=action,
            reward=reward,
            reason=reason,
            show_selected_horizon=args.show_selected_horizon,
            show_gate_margins=args.show_gate_margins,
        )
        images.append(frame)
        iio.imwrite(args.output_frames / f"frame_{step:04d}.png", frame)

    render_frame(0, None, None)
    for step in range(args.max_steps):
        state = backend._state()  # noqa: SLF001 - renderer samples simulator state
        action = np.asarray(policy(state), dtype=np.float64)
        state, reward, reason = backend.transition(
            action, kinematic_object_diagnostic=args.kinematic_object_diagnostic
        )
        index = backend.reference_index
        trace.append(
            {
                "step": float(step),
                "reference_index": float(index),
                "object_position_error_m": float(
                    np.linalg.norm(
                        state["object_pose"][:3, 3]
                        - backend.reference.object_pose_world_ref[index, :3, 3]
                    )
                ),
                "wrist_position_error_m": float(
                    np.linalg.norm(
                        state["wrist_pose"][:3, 3]
                        - backend.reference.wrist_pose_world_ref[index, :3, 3]
                    )
                ),
                "reward": float(reward["total"]),
                "wrench_norm_n": float(
                    np.linalg.norm(backend.last_control.applied_wrench_world[:3])
                    if backend.last_control is not None
                    else 0.0
                ),
            }
        )
        render_frame(step + 1, action, reward)
        if reason is not None:
            break
    if renderer is not None:
        renderer.close()
    if renderer_error is not None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        values = np.asarray(
            [
                [
                    item["object_position_error_m"],
                    item["wrist_position_error_m"],
                    item["reward"],
                    item["wrench_norm_n"],
                ]
                for item in trace
            ]
        )
        steps = np.arange(values.shape[0])
        axes[0].plot(steps, values[:, 0], label="object position error (m)")
        axes[0].plot(steps, values[:, 1], label="wrist position error (m)")
        axes[0].axhline(0.05, color="tab:red", linestyle="--", label="formal object gate")
        axes[0].legend(loc="best")
        axes[0].set_title("Stage16B numerical renderer fallback — no geometry claim")
        axes[1].plot(steps, values[:, 2], label="reward")
        axes[1].plot(steps, values[:, 3], label="wrist wrench norm (N)")
        axes[1].legend(loc="best")
        axes[1].set_xlabel("control step")
        figure.text(
            0.01,
            0.01,
            f"{type(renderer_error).__name__}: {renderer_error}",
            fontsize=7,
            family="monospace",
        )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        figure.savefig(args.output_frames / "renderer_fallback.png", dpi=150)
        plt.close(figure)
    if args.output_video is not None and images:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.output_video, np.asarray(images), fps=args.output_fps)
    _write_contact_sheet(images, args.output_frames / "contact_sheet.png")
    result = _summary(backend, images, reason, args) | {
        "renderer": renderer_status,
        "simulated_steps": len(trace),
        "numerical_trace": trace,
        "output_fps": args.output_fps,
        "duration_seconds": len(images) / args.output_fps if images else 0.0,
        "overlays_rendered": overlay_counts,
        "hud_rendered": bool(images),
        "camera_fixed_for_rollout": True,
        "kinematic_object_diagnostic": args.kinematic_object_diagnostic,
        "renderer_error": None
        if renderer_error is None
        else f"{type(renderer_error).__name__}: {renderer_error}",
        "adaptive_oracle": (
            None
            if not isinstance(policy, AdaptiveOracleVisualizationPolicy)
            else {
                "trace_source": policy.trace_source,
                "selected_horizon_counts": policy.horizon_counts,
                "selection_trace_attached": bool(policy.selection_rows),
                "non_claim": "action-only replay is oracle evidence, never PPO evaluation",
            }
        ),
    }
    (args.output_frames / "visualization_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: result[key] for key in ("renderer", "termination", "frames")}, sort_keys=True
        )
    )
    return 0


def _interactive(args: argparse.Namespace, backend: WorldWristFingerBackend, policy: Any) -> int:
    try:
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover - platform-specific viewer
        raise RuntimeError("interactive mode requires mujoco.viewer") from exc
    backend.reset(reference_index=args.start_frame)
    reason: str | None = None
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        camera = _camera(backend, args)
        if isinstance(camera, mujoco.MjvCamera):
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
        for _ in range(args.max_steps):
            if not viewer.is_running():
                break
            started = time.monotonic()
            state = backend._state()  # noqa: SLF001 - viewer samples simulator state
            action = np.asarray(policy(state), dtype=np.float64)
            _, reward, reason = backend.transition(
                action, kinematic_object_diagnostic=args.kinematic_object_diagnostic
            )
            viewer.user_scn.ngeom = 0
            _add_overlays(viewer.user_scn, backend, args)
            metadata = getattr(policy, "last_metadata", None)
            if metadata is not None and hasattr(viewer, "add_overlay"):
                title = f"adaptive oracle | frame {backend.reference_index}"
                details: list[str] = []
                if args.show_selected_horizon:
                    details.append(
                        f"H{metadata['selected_horizon']} -> H{metadata['effective_horizon']} | "
                        f"remaining {metadata['remaining']}"
                    )
                    details.append(metadata["selection_reason"].split(":", maxsplit=1)[0])
                if args.show_gate_margins:
                    details.append(
                        " | ".join(
                            f"{name}={value:.3f}"
                            for name, value in metadata["candidate_gate_values"].items()
                        )
                    )
                details.append(f"reward={reward['total']:.3f}")
                viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    title,
                    "\n".join(details),
                )
            viewer.sync()
            if reason is not None:
                break
            remaining = 1.0 / args.playback_fps - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        if not args.close_on_complete:
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
    print(json.dumps(_summary(backend, [], reason, args), sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--object-mesh", required=True, type=Path)
    parser.add_argument("--controller-report", type=Path)
    parser.add_argument("--action-scale-report", type=Path)
    parser.add_argument(
        "--scene-root", type=Path, default=Path(".local/visualize_stage16_world_wrist")
    )
    parser.add_argument(
        "--policy", choices=("zero", "oracle", "adaptive-oracle", "ppo"), default="zero"
    )
    parser.add_argument(
        "--adaptive-action-trace",
        type=Path,
        help="immutable adaptive-oracle action NPZ to replay instead of re-optimizing",
    )
    parser.add_argument(
        "--adaptive-selection-trace",
        type=Path,
        help="JSONL selection evidence used to annotate adaptive-oracle replay frames",
    )
    parser.add_argument(
        "--kinematic-object-diagnostic",
        action="store_true",
        help=(
            "render W2 with reference-driven object pose; never describe this as "
            "free-object control"
        ),
    )
    parser.add_argument("--mode", choices=("interactive", "headless"), default="interactive")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--show-reference-wrist", action="store_true")
    parser.add_argument("--show-reference-ghost", action="store_true")
    parser.add_argument("--show-world-frame", action="store_true")
    parser.add_argument("--show-wrist-frame", action="store_true")
    parser.add_argument("--show-axis-points", action="store_true")
    parser.add_argument("--show-tracked-links", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument("--show-contact-forces", action="store_true")
    parser.add_argument("--show-wrist-wrench", action="store_true")
    parser.add_argument("--show-base-target", action="store_true")
    parser.add_argument("--show-selected-horizon", action="store_true")
    parser.add_argument("--show-gate-margins", action="store_true")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--playback-fps", type=float, default=20.0)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=20.0,
        help="encoded MP4 frame rate; use 5 for a 4x slow-motion 20 Hz rollout",
    )
    parser.add_argument("--close-on-complete", action="store_true")
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-frames", type=Path)
    parser.add_argument("--oracle-horizon", choices=(1, 5, 10), type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "headless" and args.output_frames is None:
        parser.error("--output-frames is required with --mode headless")
    if args.policy == "ppo" and args.checkpoint is None:
        parser.error("--checkpoint is required with --policy ppo")
    if args.adaptive_action_trace is not None and args.policy != "adaptive-oracle":
        parser.error("--adaptive-action-trace requires --policy adaptive-oracle")
    if args.adaptive_selection_trace is not None and args.policy != "adaptive-oracle":
        parser.error("--adaptive-selection-trace requires --policy adaptive-oracle")
    if args.max_steps < 1 or args.playback_fps <= 0.0 or args.output_fps <= 0.0:
        parser.error("max-steps, playback-fps, and output-fps must be positive")
    reference = WorldWristFingerReferenceV1.from_npz(args.reference)
    if not 0 <= args.start_frame < reference.frame_count:
        parser.error("--start-frame is outside reference frame range")
    backend = _backend(args)
    policy = _policy(args, backend)
    return (
        _headless(args, backend, policy)
        if args.mode == "headless"
        else _interactive(args, backend, policy)
    )


if __name__ == "__main__":
    raise SystemExit(main())
