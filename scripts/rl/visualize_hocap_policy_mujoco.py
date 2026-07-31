#!/usr/bin/env python3
"""Interactive or headless MuJoCo inspection for one Stage-16 HOCap clip.

The renderer is an inspection aid.  It does not turn kinematic replay or an
oracle controller into a PPO result, and it reports that distinction in the
sidecar review JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoBackendConfig,
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.oracle import OracleResidualController, oracle_action
from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.trainer import PPOTrainer
from toporetarget.rl.randomization import DomainRandomizationConfig
from toporetarget.rl.termination import BASE_RELATIVE_HOCAP_TERMINATION
from toporetarget.rl.visualization import write_dashboard

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def _backend(args: argparse.Namespace) -> MujocoReferenceTrackingBackend:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF contains unnamed joints")
    reference = Stage16ReferenceClip.from_npz(args.reference)
    if reference.joint_order != tuple(name for name in joint_order if name is not None):
        raise ValueError("reference joint order does not match Wuji MJCF")
    scene = materialize_free_object_scene(
        WUJI_MJCF,
        args.scene_root,
        object_mesh=args.object_mesh,
        include_ground=False,
        gravity_mps2=(0.0, 0.0, 0.0),
    )
    return MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=bounds[:, 0],
        joint_upper=bounds[:, 1],
        config=MujocoBackendConfig(
            action_scale_fraction=args.action_scale_fraction,
            termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
        ),
        randomization=DomainRandomizationConfig(enabled=args.domain_randomization),
        seed=args.seed,
    )


def _policy(args: argparse.Namespace, backend: MujocoReferenceTrackingBackend) -> Any:
    if args.policy == "zero":

        def zero(_state: dict[str, np.ndarray], _index: int) -> np.ndarray:
            return np.zeros(backend.reference.dof_count, dtype=np.float64)

        return zero
    if args.policy == "oracle":
        controller = OracleResidualController(
            joint_gain=args.oracle_gain,
            action_scale_fraction=args.action_scale_fraction,
        )

        def oracle(state: dict[str, np.ndarray], index: int) -> np.ndarray:
            next_index = min(index + 1, backend.reference.frame_count - 1)
            return oracle_action(
                controller,
                state=state,
                reference_q=backend.reference.q_finger_ref[index],
                next_reference_q=backend.reference.q_finger_ref[next_index],
                joint_lower=backend.joint_lower,
                joint_upper=backend.joint_upper,
            )

        return oracle
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --policy checkpoint")
    observation = backend.observation(backend.reset(reference_index=args.start_frame))
    trainer = PPOTrainer(observation.size, backend.reference.dof_count, device=args.device)
    payload = load_checkpoint(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(payload["model"])
    trainer.normalizer.load_state_dict(payload["normalizer"])
    trainer.freeze_observation_normalizer()

    def checkpoint(state: dict[str, np.ndarray], _index: int) -> np.ndarray:
        nonlocal observation
        observation = backend.observation(state)
        import torch

        with torch.no_grad():
            action, _, _ = trainer.act(
                torch.as_tensor(observation[None], dtype=torch.float32, device=trainer.device),
                deterministic=args.deterministic,
            )
        return torch.clamp(action[0], -1.0, 1.0).cpu().numpy()

    return checkpoint


def _annotate(
    rgb: np.ndarray,
    *,
    frame: int,
    state: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return rgb
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    reference = args._reference
    target = min(frame, reference.frame_count - 1)
    position_error = float(
        np.linalg.norm(state["object_pose"][:3, 3] - reference.object_pose_base_ref[target][:3, 3])
    )
    axis_error = float(
        np.max(
            np.linalg.norm(
                state["object_axis_points"] - reference.object_axis_points_base_ref[target],
                axis=-1,
            )
        )
    )
    lines = [
        f"Stage16 inspection | policy={args.policy} | frame={target}/{reference.frame_count - 1}",
        "current object = orange mesh; reference ghost = numeric target overlay",
        f"progress={target / max(reference.frame_count - 1, 1):.3f}  "
        f"object_pos_error={position_error * 100:.2f} cm",
        f"max_axis_error={axis_error * 100:.2f} cm  termination={args._termination or 'running'}",
        f"show_axis_points={args.show_axis_points}  "
        f"show_tracked_links={args.show_tracked_links}  show_contacts={args.show_contacts}",
    ]
    draw.rectangle((0, 0, image.width, 76), fill=(15, 15, 15))
    for index, line in enumerate(lines):
        draw.text((8, 5 + index * 14), line, fill=(240, 240, 240))
    return np.asarray(image)


def _contact_sheet(frames: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in frames]
    if not images:
        return
    thumb_w = 320
    thumb_h = int(images[0].height * thumb_w / images[0].width)
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + 20)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_h), frames[index].name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _numeric_fallback(records: list[dict[str, object]], output: Path) -> Path:
    """Create a real headless diagnostic image when GL is unavailable."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [float(row["step"]) for row in records if "reference_index" in row]
    indices = [float(row["reference_index"]) for row in records if "reference_index" in row]
    returns = [float(row["reward_total"]) for row in records if "reward_total" in row]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(steps, indices, label="reference index")
    axes[0].set_ylabel("reference frame")
    axes[0].legend()
    axes[1].plot(steps, returns, label="reward total", color="tab:orange")
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("reward")
    axes[1].legend()
    figure.suptitle("Stage 16 numerical visualization fallback (MuJoCo GL unavailable)")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def _headless(
    args: argparse.Namespace,
    backend: MujocoReferenceTrackingBackend,
    policy: Any,
) -> int:
    frame_dir = args.output_frames
    if frame_dir is None:
        raise ValueError("headless mode requires --output-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    state = backend.reset(reference_index=args.start_frame)
    rendered: list[Path] = []
    records: list[dict[str, object]] = []
    reason: str | None = None
    for step in range(args.max_steps):
        action = np.asarray(policy(state, backend.reference_index), dtype=np.float64)
        state, reward, reason = backend.transition(action)
        keyframes = {
            0,
            max(args.max_steps // 4, 1),
            max(args.max_steps // 2, 1),
            max(args.max_steps * 3 // 4, 1),
        }
        if step in keyframes or reason:
            try:
                rgb = backend.render_rgb(width=args.width, height=args.height)
                args._termination = reason
                path = frame_dir / f"frame_{step:04d}.png"
                from PIL import Image

                Image.fromarray(
                    _annotate(rgb, frame=backend.reference_index, state=state, args=args)
                ).save(path)
                rendered.append(path)
            except Exception as exc:  # renderer fallback is part of the CLI contract
                records.append({"step": step, "renderer_error": f"{type(exc).__name__}: {exc}"})
        records.append(
            {
                "step": step,
                "reference_index": backend.reference_index,
                "action_abs_max": float(np.max(np.abs(action))),
                "reward_total": float(reward["total"]),
                "termination": reason,
            }
        )
        if reason is not None:
            break
    sheet = frame_dir / "contact_sheet.png"
    _contact_sheet(rendered, sheet)
    numerical_fallback = None
    dashboard = None
    if not rendered:
        numerical_fallback = _numeric_fallback(records, frame_dir / "numerical_fallback.png")
        dashboard = write_dashboard(
            frame_dir / "dashboard.html",
            {
                "policy": args.policy,
                "reference": str(args.reference.resolve()),
                "renderer": "unavailable",
                "records": records,
                "fallback": str(numerical_fallback.resolve()),
            },
        )
    if args.output_video is not None and rendered:
        try:
            import imageio.v2 as imageio
            from PIL import Image

            with imageio.get_writer(args.output_video, fps=20) as writer:
                for path in rendered:
                    writer.append_data(np.asarray(Image.open(path)))
        except Exception as exc:
            records.append({"video_error": f"{type(exc).__name__}: {exc}"})
    review = {
        "status": "PASS_WITH_LIMITATION" if rendered or numerical_fallback else "FAIL_RENDERER",
        "policy": args.policy,
        "renderer": "mujoco_offscreen",
        "reference": str(args.reference.resolve()),
        "object_mesh": str(args.object_mesh.resolve()),
        "records": records,
        "frames": [str(path.resolve()) for path in rendered],
        "contact_sheet": str(sheet.resolve()),
        "numerical_fallback": str(numerical_fallback.resolve()) if numerical_fallback else None,
        "dashboard": str(dashboard.resolve()) if dashboard else None,
        "renderer_failure_isolation": (
            "MuJoCo offscreen renderer failed; numerical fallback retained"
        ),
        "non_claim": "inspection visual only; oracle/kinematic replay is not PPO success",
    }
    review_path = frame_dir / "visual_review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": review["status"], "frames": len(rendered), "review": str(review_path)}
        )
    )
    return 0 if rendered or numerical_fallback else 2


def _interactive(
    args: argparse.Namespace,
    backend: MujocoReferenceTrackingBackend,
    policy: Any,
) -> int:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("interactive mode requires mujoco.viewer") from exc
    state = backend.reset(reference_index=args.start_frame)
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        for _ in range(args.max_steps):
            if not viewer.is_running():
                break
            action = policy(state, backend.reference_index)
            state, _, reason = backend.transition(action)
            viewer.sync()
            if reason is not None:
                break
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--object-mesh", required=True, type=Path)
    parser.add_argument("--scene-root", type=Path, default=Path(".local/visualize_stage16"))
    parser.add_argument("--policy", choices=("checkpoint", "oracle", "zero"), default="oracle")
    parser.add_argument("--mode", choices=("interactive", "headless"), default="interactive")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--show-reference-ghost", action="store_true")
    parser.add_argument("--show-axis-points", action="store_true")
    parser.add_argument("--show-tracked-links", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument("--camera", default="default")
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-frames", type=Path)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--action-scale-fraction", type=float, default=0.05)
    parser.add_argument("--oracle-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.mode == "headless" and args.output_frames is None:
        parser.error("--output-frames is required with --mode headless")
    if args.policy == "checkpoint" and args.checkpoint is None:
        parser.error("--checkpoint is required with --policy checkpoint")
    if not 0 <= args.start_frame < 41:
        parser.error("--start-frame must be within the 41-frame HOCap reference")
    args._reference = Stage16ReferenceClip.from_npz(args.reference)
    args._termination = None
    backend = _backend(args)
    policy = _policy(args, backend)
    if args.mode == "headless":
        return _headless(args, backend, policy)
    return _interactive(args, backend, policy)


if __name__ == "__main__":
    raise SystemExit(main())
