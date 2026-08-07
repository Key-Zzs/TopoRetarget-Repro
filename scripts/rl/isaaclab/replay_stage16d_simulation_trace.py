#!/usr/bin/env python3
"""Replay a recorded Stage 16-D simulation trace as exact collision proxies in IsaacLab."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    ConvexProxyGeometry,
    load_runtime_geometry_manifest,
)
from toporetarget.rl.geometry_audit.simulation_trace_replay import (  # noqa: E402
    Stage16DSimulationTraceReplay,
    infer_object_id,
    load_reference_object_pose,
    load_stage16d_simulation_trace,
)
from toporetarget.rl.geometry_audit.transforms import transform_points  # noqa: E402

DEFAULT_MANIFEST = (
    REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/"
    "runtime_collision_geometry_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Recorded simulation .npz")
    parser.add_argument(
        "--geometry",
        type=Path,
        help="Optional matching *_geometry.npz for penetration coloring and metrics",
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        help="Optional qualification JSON; corrected traces auto-detect a matching sibling",
    )
    parser.add_argument(
        "--source-trace",
        type=Path,
        help="Optional source/reference NPZ rendered as a translucent object ghost",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--object", choices=("hocap_170105", "hocap_170650"))
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive; defaults to the trace end")
    parser.add_argument("--frame", type=int, help="Show one frame and hold the window open")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--loop", action="store_true", help="Replay until the IsaacLab window closes"
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=None,
        help="Bound loop count, primarily for automated/headless validation",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Bound static-frame hold; GUI defaults to hold until close",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    return parser.parse_args()


def _safe_prim_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _finger_color(body_name: str) -> tuple[float, float, float]:
    if "thumb" in body_name:
        return (0.15, 0.80, 0.90)
    if "index" in body_name:
        return (0.20, 0.45, 0.95)
    if "middle" in body_name:
        return (0.25, 0.80, 0.35)
    if "ring" in body_name:
        return (1.00, 0.58, 0.15)
    if "pinky" in body_name:
        return (0.72, 0.32, 0.90)
    return (0.55, 0.60, 0.68)


@dataclass
class ProxyPrim:
    translate_op: object
    orient_op: object
    color_attr: object
    base_color: tuple[float, float, float]


def _validate_args(args: argparse.Namespace) -> None:
    if not args.accept_eula:
        raise ValueError("explicit --accept-eula is required for this licensed runtime process")
    if args.fps <= 0.0 or args.speed <= 0.0:
        raise ValueError("--fps and --speed must be positive")
    if args.max_loops is not None and args.max_loops <= 0:
        raise ValueError("--max-loops must be positive")
    if args.hold_seconds is not None and args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be non-negative")
    frame_range_requested = args.loop or args.end_frame is not None or args.start_frame != 0
    if args.frame is not None and frame_range_requested:
        raise ValueError("--frame cannot be combined with --loop/--start-frame/--end-frame")
    if args.headless and args.frame is not None and args.hold_seconds is None:
        args.hold_seconds = 0.0


def _status_line(trace: Stage16DSimulationTraceReplay, frame: int, replica: int) -> str:
    row = trace.diagnostics(frame, replica)
    penetration = (
        "n/a" if row.worst_penetration_m is None else f"{row.worst_penetration_m * 1000.0:.3f} mm"
    )
    groups = ",".join(row.contact_groups) or "none"
    return (
        f"kind={trace.trace_kind} qualification={trace.qualification_status} "
        f"frame={frame:03d}/{trace.frame_count - 1:03d} replica={replica} "
        f"contacts={row.contact_body_count} groups={groups} force={row.contact_force_norm_n:.3f} N "
        f"object_v={row.object_linear_speed_mps:.3f} m/s "
        f"object_w={row.object_angular_speed_radps:.3f} rad/s "
        f"penetration={penetration} finite={row.finite} reason={row.reason_code}"
    )


def main() -> int:
    args = parse_args()
    _validate_args(args)
    object_id = args.object or infer_object_id(args.trace)
    hand_proxies, object_proxy_map = load_runtime_geometry_manifest(args.manifest)
    if object_id not in object_proxy_map:
        raise ValueError(f"object {object_id} is not present in {args.manifest}")
    if len(object_proxy_map[object_id]) != 1:
        count = len(object_proxy_map[object_id])
        raise ValueError(f"replay requires exactly one object proxy, got {count}")
    trace = load_stage16d_simulation_trace(
        args.trace,
        geometry_path=args.geometry,
        expected_body_names=[proxy.body_name for proxy in hand_proxies],
        qualification_path=args.qualification,
    )
    source_object_pose = (
        load_reference_object_pose(args.source_trace, expected_frames=trace.frame_count)
        if args.source_trace is not None
        else None
    )
    trace.validate_replica(args.replica)
    if args.frame is not None:
        if not 0 <= args.frame < trace.frame_count:
            raise ValueError(f"--frame must be in [0, {trace.frame_count - 1}]")
        frames = range(args.frame, args.frame + 1)
    else:
        frames = trace.frame_indices(args.start_frame, args.end_frame)

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=args.headless).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import omni.usd
        from pxr import Gf, UsdGeom

        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            render_interval=1,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.88, 1.0))
        light_cfg.func("/World/ReplayLight", light_cfg)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/Replay")

        def create_proxy(
            proxy: ConvexProxyGeometry,
            path: str,
            color: tuple[float, float, float],
            opacity: float,
        ) -> ProxyPrim:
            parent = UsdGeom.Xform.Define(stage, path)
            translate = parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "replay")
            orient = parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "replay")
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/CollisionProxy")
            vertices = transform_points(proxy.scaled_vertices, proxy.local_pose_xyz_wxyz)
            mesh.CreatePointsAttr([Gf.Vec3f(*row) for row in vertices])
            mesh.CreateFaceVertexCountsAttr([3] * len(proxy.faces))
            mesh.CreateFaceVertexIndicesAttr(proxy.faces.reshape(-1).astype(int).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            color_attr = mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            mesh.CreateDisplayOpacityAttr([opacity])
            return ProxyPrim(translate, orient, color_attr, color)

        hand_prims = [
            create_proxy(
                proxy,
                f"/World/Replay/Hand/Body_{index:02d}_{_safe_prim_name(proxy.body_name)}",
                _finger_color(proxy.body_name),
                0.82,
            )
            for index, proxy in enumerate(hand_proxies)
        ]
        object_prim = create_proxy(
            object_proxy_map[object_id][0],
            f"/World/Replay/Object/{object_id}",
            (0.92, 0.82, 0.20),
            0.72,
        )
        source_object_prim = None
        if source_object_pose is not None:
            source_object_prim = create_proxy(
                object_proxy_map[object_id][0],
                f"/World/Replay/SourceObject/{object_id}",
                (0.20, 0.88, 1.00),
                0.25,
            )

        def set_pose(prim: ProxyPrim, pose: np.ndarray) -> None:
            position = pose[:3].astype(float)
            quaternion = pose[3:7].astype(float)
            quaternion /= np.linalg.norm(quaternion)
            prim.translate_op.Set(Gf.Vec3d(*position))
            prim.orient_op.Set(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:])))

        def show_frame(frame: int) -> None:
            replica = args.replica
            for body, prim in enumerate(hand_prims):
                set_pose(prim, trace.hand_collision_body_pose[frame, replica, body])
                color = (
                    (1.0, 0.08, 0.05)
                    if trace.contact_pair_presence[frame, replica, body]
                    else prim.base_color
                )
                prim.color_attr.Set([Gf.Vec3f(*color)])
            set_pose(object_prim, trace.object_pose[frame, replica])
            if source_object_prim is not None and source_object_pose is not None:
                set_pose(source_object_prim, source_object_pose[frame])
            object_color = object_prim.base_color
            if trace.frame_worst_penetration_m is not None:
                depth = float(trace.frame_worst_penetration_m[frame, replica])
                if depth > 0.003:
                    object_color = (1.0, 0.05, 0.05)
                elif depth > 0.0:
                    object_color = (1.0, 0.42, 0.05)
            object_prim.color_attr.Set([Gf.Vec3f(*object_color)])
            sim.render()
            print("\r" + _status_line(trace, frame, replica), end="", flush=True)

        center, radius = trace.camera_bounds(args.replica, frames)
        if source_object_pose is not None:
            reference_positions = source_object_pose[np.asarray(list(frames)), :3]
            low = np.minimum(center - radius, reference_positions.min(axis=0))
            high = np.maximum(center + radius, reference_positions.max(axis=0))
            center = (low + high) * 0.5
            radius = max(float(np.linalg.norm(high - low)), 0.12)
        eye = center + np.asarray((1.35, 1.15, 0.85)) * radius
        sim.set_camera_view(eye.tolist(), center.tolist())
        sim.reset()

        qualification_metrics = trace.qualification_metrics or {}
        print(
            f"REPLAY_INPUT kind={trace.trace_kind} frames={trace.frame_count} "
            f"qualification={trace.qualification_status} metrics={qualification_metrics} "
            f"source_ghost={'enabled' if source_object_pose is not None else 'disabled'}",
            flush=True,
        )

        if args.frame is not None:
            show_frame(args.frame)
            started = time.monotonic()
            while simulation_app.is_running():
                hold_expired = (
                    args.hold_seconds is not None
                    and time.monotonic() - started >= args.hold_seconds
                )
                if hold_expired:
                    break
                sim.render()
                time.sleep(0.01)
        else:
            loop_count = 0
            keep_replaying = True
            frame_period = 1.0 / (args.fps * args.speed)
            while simulation_app.is_running() and keep_replaying:
                for frame in frames:
                    if not simulation_app.is_running():
                        break
                    started = time.monotonic()
                    show_frame(frame)
                    remaining = frame_period - (time.monotonic() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                loop_count += 1
                keep_replaying = args.loop and (
                    args.max_loops is None or loop_count < args.max_loops
                )
        print()
        print(
            f"REPLAY_COMPLETE object={object_id} trace={args.trace.resolve()} "
            f"replica={args.replica} frames={frames.start}:{frames.stop} "
            f"qualification={trace.qualification_status}",
            flush=True,
        )
        return 0
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
