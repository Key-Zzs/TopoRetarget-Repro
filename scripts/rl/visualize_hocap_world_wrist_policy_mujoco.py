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
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1
from toporetarget.rl.world_wrist_oracle import WorldWristFingerObjectAwareOracle

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def _profile(path: Path | None) -> tuple[WristImpedanceProfileV1, WristFingerActionScaleV1]:
    if path is None:
        return WristImpedanceProfileV1(), WristFingerActionScaleV1()
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report["selected"]["profile"]
    return (
        WristImpedanceProfileV1(
            translation_stiffness_npm=float(selected["translation_stiffness_npm"]),
            translation_damping_ratio=float(selected["translation_damping_ratio"]),
            rotation_stiffness_nmprad=float(selected["rotation_stiffness_nmprad"]),
            rotation_damping_ratio=float(selected["rotation_damping_ratio"]),
            force_limit_n=float(selected["force_limit_n"]),
            torque_limit_nm=float(selected["torque_limit_nm"]),
            feedforward_twist_gain=float(selected["feedforward_twist_gain"]),
        ),
        WristFingerActionScaleV1(),
    )


def _backend(args: argparse.Namespace) -> WorldWristFingerBackend:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    reference = WorldWristFingerReferenceV1.from_npz(args.reference)
    impedance, scale = _profile(args.controller_report)
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
        "overlays_requested": {
            name: bool(getattr(args, name))
            for name in (
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
    camera.distance = max(0.28, 3.0 * float(np.max(np.ptp(points, axis=0))) + 0.22)
    camera.azimuth = 135.0
    camera.elevation = -25.0
    return camera


def _write_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    """Write a compact frame sheet without depending on an implicit GUI backend."""

    if not images:
        return
    columns = min(4, len(images))
    rows = int(np.ceil(len(images) / columns))
    height, width, channels = images[0].shape
    sheet = np.zeros((rows * height, columns * width, channels), dtype=images[0].dtype)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    import imageio.v3 as iio

    iio.imwrite(output, sheet)


def _headless(args: argparse.Namespace, backend: WorldWristFingerBackend, policy: Any) -> int:
    import imageio.v3 as iio

    args.output_frames.mkdir(parents=True, exist_ok=True)
    backend.reset(reference_index=args.start_frame)
    images: list[np.ndarray] = []
    trace: list[dict[str, float]] = []
    reason: str | None = None
    renderer: mujoco.Renderer | None = None
    renderer_error: Exception | None = None
    try:
        renderer = mujoco.Renderer(backend.model, height=args.height, width=args.width)
        renderer_status = "mujoco_offscreen"
    except Exception as exc:  # noqa: BLE001 - renderer is an optional platform capability
        renderer_error = exc
        renderer_status = "PASS_WITH_LIMITATION_NUMERICAL_FALLBACK"
    for step in range(args.max_steps):
        state = backend._state()  # noqa: SLF001 - renderer samples simulator state
        action = policy(state)
        state, reward, reason = backend.transition(action)
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
        if renderer is not None:
            renderer.update_scene(backend.data, camera=_camera(backend, args))
            frame = renderer.render().copy()
            images.append(frame)
            iio.imwrite(args.output_frames / f"frame_{step:04d}.png", frame)
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
        iio.imwrite(args.output_video, np.asarray(images), fps=20)
    _write_contact_sheet(images, args.output_frames / "contact_sheet.png")
    result = _summary(backend, images, reason, args) | {
        "renderer": renderer_status,
        "simulated_steps": len(trace),
        "numerical_trace": trace,
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
            _, _, reason = backend.transition(policy(state))
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
    parser.add_argument(
        "--scene-root", type=Path, default=Path(".local/visualize_stage16_world_wrist")
    )
    parser.add_argument("--policy", choices=("zero", "oracle", "ppo"), default="zero")
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
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--playback-fps", type=float, default=20.0)
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
    if args.max_steps < 1 or args.playback_fps <= 0.0:
        parser.error("max-steps and playback-fps must be positive")
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
