#!/usr/bin/env python3
"""Replay a recorded physical-HOI trace as exact collision proxies in IsaacLab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.raw_mocap_overlay import (  # noqa: E402
    FINGER_ORDER,
    RawMocapOverlay,
    RawMocapOverlayUnavailable,
    decimate_visual_mesh,
    resolve_raw_mocap_overlay,
)
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    ConvexProxyGeometry,
    load_runtime_geometry_manifest,
)
from toporetarget.rl.geometry_audit.simulation_trace_replay import (  # noqa: E402
    Stage16DSimulationTraceReplay,
    infer_object_id,
    load_factor8_hocap_reference_object_pose,
    load_stage16d_simulation_trace,
)
from toporetarget.rl.geometry_audit.transforms import transform_points  # noqa: E402

DEFAULT_MANIFEST = (
    REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/"
    "runtime_collision_geometry_manifest.json"
)
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
INFERRED_TABLE_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction/inference"


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
        "--reference",
        type=Path,
        help=(
            "Frozen uniformly sampled HO-Cap Stage 16 reference; inferred from --object by default"
        ),
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-ghost",
        dest="reference_ghost",
        action="store_true",
        default=True,
        help="Show the geometric-retarget reference ghost (default)",
    )
    reference_group.add_argument(
        "--no-reference-ghost",
        dest="reference_ghost",
        action="store_false",
        help="Disable the geometric-retarget reference ghost (backward compatible)",
    )
    parser.add_argument(
        "--no-reference-object",
        action="store_true",
        help="Keep the geometric-retarget hand ghost while hiding its object",
    )
    mocap_group = parser.add_mutually_exclusive_group()
    mocap_group.add_argument(
        "--mocap-ghost",
        dest="mocap_ghost",
        action="store_true",
        default=True,
        help="Show original HOCap MANO and object visual ghosts (default)",
    )
    mocap_group.add_argument(
        "--no-mocap-ghost",
        dest="mocap_ghost",
        action="store_false",
        help="Hide original HOCap MANO, object, and fingertips",
    )
    mocap_object_group = parser.add_mutually_exclusive_group()
    mocap_object_group.add_argument(
        "--mocap-object", dest="mocap_object", action="store_true", default=True
    )
    mocap_object_group.add_argument("--no-mocap-object", dest="mocap_object", action="store_false")
    mocap_mesh_group = parser.add_mutually_exclusive_group()
    mocap_mesh_group.add_argument(
        "--mocap-object-low-poly",
        dest="mocap_object_max_faces",
        action="store_const",
        const=2000,
        help="Display raw object using the deterministic 2,000-face visual mesh",
    )
    mocap_mesh_group.add_argument(
        "--mocap-object-max-faces",
        type=int,
        help="Maximum faces for the deterministic raw-object display mesh (minimum: 4)",
    )
    mocap_tip_group = parser.add_mutually_exclusive_group()
    mocap_tip_group.add_argument(
        "--mocap-fingertips", dest="mocap_fingertips", action="store_true", default=True
    )
    mocap_tip_group.add_argument(
        "--no-mocap-fingertips", dest="mocap_fingertips", action="store_false"
    )
    parser.add_argument(
        "--require-mocap-ghost",
        action="store_true",
        help="Fail instead of replaying actual-only when raw provenance/assets are unavailable",
    )
    parser.add_argument(
        "--mocap-similarity-output",
        type=Path,
        help="Optional JSON for RAW_MOCAP_VS_ACTUAL object/wrist/fingertip diagnostics",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--object")
    parser.add_argument(
        "--support-proxy",
        type=Path,
        help="Direct table_proxy.json for an independent trace with support telemetry.",
    )
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
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="Optional JSON receipt written after a complete replay pass",
    )
    parser.add_argument("--accept-eula", action="store_true")
    return parser.parse_args()


def _safe_prim_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _inferred_table_proxy(clip: str, path: Path | None = None) -> dict[str, object]:
    """Load the frozen finite support used by C4 without changing replay physics."""

    selected = (
        path.resolve() if path is not None else INFERRED_TABLE_ROOT / clip / "table_proxy.json"
    )
    payload = json.loads(selected.read_text(encoding="utf-8"))
    required = {"table_pose", "table_extent", "table_thickness", "plane_normal"}
    if not required.issubset(payload):
        raise ValueError("C4_REPLAY_INFERRED_TABLE_CONTRACT_INVALID")
    return payload


def _trace_has_inferred_table_contact(trace_path: Path) -> bool:
    with np.load(trace_path, allow_pickle=False) as archive:
        return "table_object_contact" in archive.files


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


@dataclass
class VisualMeshPrim:
    points_attr: object
    translate_op: object | None = None
    orient_op: object | None = None


def _validate_args(args: argparse.Namespace) -> None:
    if not args.accept_eula:
        raise ValueError("explicit --accept-eula is required for this licensed runtime process")
    if args.fps <= 0.0 or args.speed <= 0.0:
        raise ValueError("--fps and --speed must be positive")
    if args.max_loops is not None and args.max_loops <= 0:
        raise ValueError("--max-loops must be positive")
    if args.hold_seconds is not None and args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be non-negative")
    if args.mocap_object_max_faces is not None and args.mocap_object_max_faces < 4:
        raise ValueError("--mocap-object-max-faces must be at least 4")
    if args.object is not None and (
        not args.object or any(token in args.object for token in ("/", "\\", ".."))
    ):
        raise ValueError("INDEPENDENT_PHYSICAL_REPLAY_OBJECT_ID_INVALID")
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
    inter_finger = (
        "n/a"
        if row.inter_finger_penetration_m is None
        else f"{row.inter_finger_penetration_m * 1000.0:.3f} mm"
    )
    groups = ",".join(row.contact_groups) or "none"
    return (
        f"kind={trace.trace_kind} qualification={trace.qualification_status} "
        f"frame={frame:03d}/{trace.frame_count - 1:03d} replica={replica} "
        f"contacts={row.contact_body_count} groups={groups} force={row.contact_force_norm_n:.3f} N "
        f"object_v={row.object_linear_speed_mps:.3f} m/s "
        f"object_w={row.object_angular_speed_radps:.3f} rad/s "
        f"penetration={penetration} inter_finger={inter_finger} "
        f"finite={row.finite} reason={row.reason_code}"
    )


def _write_validation_receipt(
    path: Path,
    *,
    args: argparse.Namespace,
    trace: Stage16DSimulationTraceReplay,
    frames: range,
    reference_ghost: str,
    ppo_metadata: dict[str, object],
    inferred_table_rendered: bool,
    mocap_overlay: RawMocapOverlay | None,
    mocap_unavailable: str | None,
) -> None:
    """Materialize replay evidence without changing legacy trace playback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    transform_digest = hashlib.sha256()
    for value in (trace.object_pose, trace.hand_collision_body_pose):
        array = np.ascontiguousarray(value)
        transform_digest.update(str(array.dtype).encode("ascii"))
        transform_digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        transform_digest.update(array.tobytes())
    payload = {
        "schema_version": "Stage16DPPO26DReplayValidationV1",
        "status": "STAGE16D_PPO26D_REPLAY_VALIDATED",
        "headless": args.headless,
        "trace": str(args.trace.resolve()),
        "object": args.object,
        "replica": args.replica,
        "frame_start": frames.start,
        "frame_end_exclusive": frames.stop,
        "frame_count": len(frames),
        "trace_kind": trace.trace_kind,
        "qualification_status": trace.qualification_status,
        "hand_collision_proxy_count": len(trace.hand_collision_body_names),
        "reference_ghost": reference_ghost,
        "inferred_table_rendered": inferred_table_rendered,
        "raw_mocap_ghost": "AVAILABLE" if mocap_overlay is not None else mocap_unavailable,
        "raw_mocap_object_visible": bool(args.mocap_ghost and args.mocap_object),
        "raw_mocap_fingertips_visible": bool(args.mocap_ghost and args.mocap_fingertips),
        "ppo_action_contract": ppo_metadata.get("action_contract"),
        "finite": all(trace.diagnostics(frame, args.replica).finite for frame in frames),
        "actual_replay_transforms_sha256": transform_digest.hexdigest(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ppo26d_embedded_reference(
    trace_path: Path, *, expected_frames: int
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Read optional PPO metadata without changing legacy trace interpretation."""

    with np.load(trace_path, allow_pickle=False) as archive:
        if "embedded_reference_object_pose" not in archive.files:
            return None, {}
        reference = np.asarray(archive["embedded_reference_object_pose"], dtype=np.float64)
        if reference.shape != (expected_frames, 7) or not np.isfinite(reference).all():
            raise ValueError("embedded_reference_object_pose must be finite [frames, 7]")
        metadata = {
            name: str(archive[name].item())
            for name in (
                "trace_type",
                "clip",
                "requested_clip",
                "action_contract",
                "checkpoint_path",
                "cumulative_training_samples",
                "selected_num_envs",
            )
            if name in archive.files
        }
        if "reward_total" in archive.files:
            reward_total = np.asarray(archive["reward_total"], dtype=np.float64)
            if reward_total.shape != (expected_frames,) or not np.isfinite(reward_total).all():
                raise ValueError("PPO reward_total must be finite [frames]")
            metadata["reward_total"] = reward_total
        if "clip_index" in archive.files:
            clip_index = np.asarray(archive["clip_index"], dtype=np.int64)
            if clip_index.shape != (expected_frames,):
                raise ValueError("PPO clip_index must have shape [frames]")
            metadata["clip_index"] = clip_index
    if metadata.get("trace_type") != "stage16d_ppo26d":
        raise ValueError("embedded PPO reference requires trace_type=stage16d_ppo26d")
    if metadata.get("action_contract") != "26D_reference_residual":
        raise ValueError("embedded PPO reference requires 26D reference-residual metadata")
    return reference, metadata


def _load_retimed_reference_links(
    reference_path: Path, *, expected_frames: int
) -> np.ndarray | None:
    """Display the existing geometric hand as link-point ghosts, never physics."""

    with np.load(reference_path, allow_pickle=False) as source:
        if "tracked_link_positions_world_ref" not in source.files:
            return None
        values = np.asarray(source["tracked_link_positions_world_ref"], dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 3:
        raise ValueError("reference tracked_link_positions_world_ref is invalid")
    intervals = values.shape[0] - 1
    if (expected_frames - 1) % intervals:
        raise ValueError("reference link ghost cannot identify the runtime retiming")
    coordinate = np.arange(expected_frames, dtype=np.float64) / ((expected_frames - 1) / intervals)
    lower = np.minimum(np.floor(coordinate).astype(np.int64), intervals - 1)
    alpha = np.clip(coordinate - lower, 0.0, 1.0)[:, None, None]
    result = (1.0 - alpha) * values[lower] + alpha * values[lower + 1]
    result[-1] = values[-1]
    return result


def _trace_actual_tip_positions(trace: Stage16DSimulationTraceReplay, replica: int) -> np.ndarray:
    """Use recorded distal-body origins as the replay's robot fingertip support points."""

    names = tuple(trace.hand_collision_body_names)
    expected = {
        "thumb": "r_thumb_distal",
        "index": "r_index_finger_distal",
        "middle": "r_middle_finger_distal",
        "ring": "r_ring_finger_distal",
        "pinky": "r_pinky_distal",
    }
    try:
        indices = [names.index(expected[finger]) for finger in FINGER_ORDER]
    except ValueError as exc:
        raise ValueError("RAW_MOCAP_ACTUAL_FINGERTIP_PROXY_MISSING") from exc
    return np.asarray(trace.hand_collision_body_pose[:, replica, indices, :3], dtype=np.float64)


def _rotation_error_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> np.ndarray:
    dot = np.abs(
        np.sum(np.asarray(first_wxyz)[..., 3:] * np.asarray(second_wxyz)[..., 3:], axis=-1)
    )
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _write_mocap_similarity(
    path: Path, *, trace: Stage16DSimulationTraceReplay, replica: int, overlay: RawMocapOverlay
) -> None:
    """Persist morphology-aware, object-local raw-MANO versus actual diagnostics."""

    actual_object = np.asarray(trace.object_pose[:, replica], dtype=np.float64)
    raw_object = overlay.raw_object_pose_world_wxyz
    actual_tips = _trace_actual_tip_positions(trace, replica)
    raw_tips = overlay.raw_mano_fingertips_world
    from toporetarget.rl.geometry_audit.raw_mocap_overlay import pose_wxyz_to_matrix

    def local(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        matrix = pose_wxyz_to_matrix(pose)
        return np.einsum("tji,tfj->tfi", matrix[:, :3, :3], points - matrix[:, None, :3, 3])

    raw_local = local(raw_tips, raw_object)
    actual_local = local(actual_tips, actual_object)
    distances = np.linalg.norm(raw_local - actual_local, axis=-1)
    payload = {
        "schema_version": "Stage16RawMocapSimilarityV1",
        "comparison": "RAW_MOCAP_VS_ACTUAL",
        "robot_fingertip_definition": (
            "recorded distal collision-body origins; no MANO-to-robot vertex comparison"
        ),
        "object_translation_difference_m": {
            "mean": float(np.linalg.norm(raw_object[:, :3] - actual_object[:, :3], axis=-1).mean()),
            "max": float(np.linalg.norm(raw_object[:, :3] - actual_object[:, :3], axis=-1).max()),
        },
        "object_rotation_difference_rad": {
            "mean": float(_rotation_error_rad(raw_object, actual_object).mean()),
            "max": float(_rotation_error_rad(raw_object, actual_object).max()),
        },
        "per_finger_object_local_distance_m": {
            finger: {
                "mean": float(distances[:, index].mean()),
                "max": float(distances[:, index].max()),
            }
            for index, finger in enumerate(FINGER_ORDER)
        },
        "coordinate_alignment": overlay.coordinate_alignment,
        "time_alignment": overlay.time_alignment,
        "source_provenance": overlay.source_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    embedded_reference, ppo_metadata = _ppo26d_embedded_reference(
        args.trace, expected_frames=trace.frame_count
    )
    if ppo_metadata:
        trace_clip = ppo_metadata.get("clip")
        if trace_clip != object_id:
            raise ValueError(
                f"PPO trace clip {trace_clip!r} does not match replay object {object_id!r}"
            )
        expected_clip_index = {"hocap_170105": 0, "hocap_170650": 1}.get(object_id, 0)
        clip_index = np.asarray(ppo_metadata.get("clip_index"), dtype=np.int64)
        if clip_index.shape != (trace.frame_count,) or not np.all(
            clip_index == expected_clip_index
        ):
            raise ValueError(
                f"PPO trace clip_index does not match its declared clip: clip={trace_clip!r}"
            )
    reference_path = args.reference or (
        DEFAULT_REFERENCE_ROOT / f"{object_id}.world_wrist.stage16.npz"
    )
    reference_object_pose = None
    reference_link_positions = None
    if args.reference_ghost:
        reference_object_pose = (
            embedded_reference
            if embedded_reference is not None
            else load_factor8_hocap_reference_object_pose(
                reference_path, expected_frames=trace.frame_count, time_scale=8
            )
        )
        if embedded_reference is None:
            reference_link_positions = _load_retimed_reference_links(
                reference_path, expected_frames=trace.frame_count
            )
    mocap_overlay = None
    mocap_unavailable = None
    if args.mocap_ghost:
        try:
            mocap_overlay = resolve_raw_mocap_overlay(
                trace_path=args.trace,
                frame_count=trace.frame_count,
                clip=object_id,
                reference_path=reference_path,
            )
        except RawMocapOverlayUnavailable as exc:
            mocap_unavailable = str(exc)
            if args.require_mocap_ghost:
                raise
    trace.validate_replica(args.replica)
    inferred_table_rendered = _trace_has_inferred_table_contact(args.trace)
    if ppo_metadata:
        print(
            "PPO26D metadata "
            f"samples={ppo_metadata.get('cumulative_training_samples')} "
            f"envs={ppo_metadata.get('selected_num_envs')} "
            f"checkpoint={ppo_metadata.get('checkpoint_path')}",
            flush=True,
        )
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
    ghost_ui = None
    keyboard_input = None
    keyboard = None
    keyboard_subscription = None
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
        reference_layer = UsdGeom.Xform.Define(stage, "/World/Replay/Reference")
        mocap_layer = UsdGeom.Xform.Define(stage, "/World/Replay/Mocap")

        if inferred_table_rendered:
            table = _inferred_table_proxy(object_id, args.support_proxy)
            pose = np.asarray(table["table_pose"], dtype=np.float64)
            normal = np.asarray(table["plane_normal"], dtype=np.float64)
            extent = np.asarray(table["table_extent"], dtype=np.float64)
            thickness = float(table["table_thickness"])
            if pose.shape != (7,) or normal.shape != (3,) or extent.shape != (2,):
                raise ValueError("C4_REPLAY_INFERRED_TABLE_SHAPE_INVALID")
            table_parent = UsdGeom.Xform.Define(stage, "/World/Replay/InferredTable")
            table_parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "replay").Set(
                Gf.Vec3d(*(pose[:3] - 0.5 * thickness * normal))
            )
            table_parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "replay").Set(
                Gf.Quatd(float(pose[3]), Gf.Vec3d(*pose[4:]))
            )
            table_parent.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "replay").Set(
                Gf.Vec3d(float(extent[0]), float(extent[1]), thickness)
            )
            table_cube = UsdGeom.Cube.Define(stage, "/World/Replay/InferredTable/Proxy")
            table_cube.CreateSizeAttr(1.0)
            table_cube.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.38, 0.44)])
            table_cube.CreateDisplayOpacityAttr([0.32])

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

        def create_visual_mesh(
            vertices: np.ndarray,
            faces: np.ndarray,
            path: str,
            color: tuple[float, float, float],
            opacity: float,
            *,
            dynamic_vertices: bool,
        ) -> VisualMeshPrim:
            """Create a USD-only mesh: no rigid body, collider, or contact API."""

            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(
                [Gf.Vec3f(*row) for row in np.asarray(vertices, dtype=np.float64)]
            )
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int64).reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            mesh.CreateDisplayOpacityAttr([opacity])
            return VisualMeshPrim(
                mesh.GetPointsAttr() if dynamic_vertices else mesh.GetPointsAttr()
            )

        def create_marker(
            path: str,
            color: tuple[float, float, float],
            radius: float,
            *,
            opacity: float = 0.9,
        ) -> object:
            marker = UsdGeom.Sphere.Define(stage, path)
            marker.CreateRadiusAttr(radius)
            marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            marker.CreateDisplayOpacityAttr([opacity])
            return marker.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "replay")

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
        reference_object_prim = None
        if reference_object_pose is not None and not args.no_reference_object:
            reference_object_prim = create_proxy(
                object_proxy_map[object_id][0],
                f"/World/Replay/Reference/Object/{object_id}",
                (0.20, 0.88, 1.00),
                0.25,
            )
        reference_link_markers: list[object] = []
        if reference_link_positions is not None:
            reference_link_markers = [
                create_marker(
                    f"/World/Replay/Reference/Link_{index:02d}", (0.20, 0.88, 1.00), 0.006
                )
                for index in range(reference_link_positions.shape[1])
            ]
        mocap_mano_prim = None
        mocap_object_prim = None
        mocap_tip_markers: list[object] = []
        mocap_object_vertices = None
        mocap_object_faces = None
        if mocap_overlay is not None:
            mocap_mano_prim = create_visual_mesh(
                mocap_overlay.raw_mano_vertices_world[0],
                mocap_overlay.raw_mano_faces,
                "/World/Replay/Mocap/MANO",
                (0.94, 0.24, 0.68),
                0.18,
                dynamic_vertices=True,
            )
            if args.mocap_object:
                mocap_object_vertices = mocap_overlay.raw_object_vertices_local
                mocap_object_faces = mocap_overlay.raw_object_faces
                if args.mocap_object_max_faces is not None:
                    mocap_object_vertices, mocap_object_faces = decimate_visual_mesh(
                        mocap_object_vertices,
                        mocap_object_faces,
                        max_faces=args.mocap_object_max_faces,
                    )
                mocap_object_prim = create_visual_mesh(
                    mocap_object_vertices,
                    mocap_object_faces,
                    f"/World/Replay/Mocap/Object/{object_id}",
                    (0.88, 0.22, 0.78),
                    0.20,
                    dynamic_vertices=False,
                )
            if args.mocap_fingertips:
                mocap_tip_markers = [
                    create_marker(
                        f"/World/Replay/Mocap/Fingertips/{finger}",
                        (1.0, 0.12, 0.74),
                        0.008,
                        opacity=0.65,
                    )
                    for finger in FINGER_ORDER
                ]

        reference_layer_available = bool(reference_object_prim or reference_link_markers)
        mocap_layer_available = bool(mocap_mano_prim or mocap_object_prim or mocap_tip_markers)
        layer_visible = {
            "reference": reference_layer_available,
            "mocap": mocap_layer_available,
        }

        def set_layer_visibility(layer_name: str, prim: object, visible: bool) -> None:
            """Toggle a whole visual layer without mutating replay or physics state."""

            visibility = UsdGeom.Imageable(prim).GetVisibilityAttr()
            visibility.Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
            layer_visible[layer_name] = visible
            print(
                f"GHOST_VISIBILITY layer={layer_name} visible={visible}",
                flush=True,
            )

        set_layer_visibility("reference", reference_layer.GetPrim(), layer_visible["reference"])
        set_layer_visibility("mocap", mocap_layer.GetPrim(), layer_visible["mocap"])

        if not args.headless:
            import carb
            import omni.appwindow
            import omni.ui as ui

            mocap_model = ui.SimpleBoolModel(default_value=layer_visible["mocap"])
            reference_model = ui.SimpleBoolModel(default_value=layer_visible["reference"])
            mocap_model.add_value_changed_fn(
                lambda model: set_layer_visibility(
                    "mocap", mocap_layer.GetPrim(), model.get_value_as_bool()
                )
            )
            reference_model.add_value_changed_fn(
                lambda model: set_layer_visibility(
                    "reference", reference_layer.GetPrim(), model.get_value_as_bool()
                )
            )
            ghost_ui = ui.Window("Replay Ghost Visibility", width=310, height=105)
            with ghost_ui.frame:
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Visual-only live controls (no replay restart)")
                    with ui.HStack(height=22):
                        ui.CheckBox(model=mocap_model, width=18, enabled=mocap_layer_available)
                        ui.Label("Raw MOCAP ghost  [M]")
                    with ui.HStack(height=22):
                        ui.CheckBox(
                            model=reference_model,
                            width=18,
                            enabled=reference_layer_available,
                        )
                        ui.Label("Retarget reference ghost  [R]")

            keyboard_input = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()

            def on_keyboard_event(event: object, *_args: object, **_kwargs: object) -> bool:
                if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                    return True
                if event.input == carb.input.KeyboardInput.M and mocap_layer_available:
                    mocap_model.set_value(not mocap_model.get_value_as_bool())
                elif event.input == carb.input.KeyboardInput.R and reference_layer_available:
                    reference_model.set_value(not reference_model.get_value_as_bool())
                return True

            keyboard_subscription = keyboard_input.subscribe_to_keyboard_events(
                keyboard, on_keyboard_event
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
            if (
                layer_visible["reference"]
                and reference_object_prim is not None
                and reference_object_pose is not None
            ):
                set_pose(reference_object_prim, reference_object_pose[frame])
            if layer_visible["reference"]:
                for marker, position in zip(
                    reference_link_markers,
                    reference_link_positions[frame] if reference_link_positions is not None else (),
                    strict=True,
                ):
                    marker.Set(Gf.Vec3d(*position))
            if layer_visible["mocap"] and mocap_overlay is not None and mocap_mano_prim is not None:
                mocap_mano_prim.points_attr.Set(
                    [Gf.Vec3f(*row) for row in mocap_overlay.raw_mano_vertices_world[frame]]
                )
            if (
                layer_visible["mocap"]
                and mocap_overlay is not None
                and mocap_object_prim is not None
                and mocap_object_vertices is not None
            ):
                pose = mocap_overlay.raw_object_pose_world_wxyz[frame]
                matrix = np.eye(4, dtype=np.float64)
                quaternion = pose[3:] / np.linalg.norm(pose[3:])
                from scipy.spatial.transform import Rotation

                matrix[:3, :3] = Rotation.from_quat(
                    np.r_[quaternion[1:], quaternion[0]]
                ).as_matrix()
                matrix[:3, 3] = pose[:3]
                world_vertices = mocap_object_vertices @ matrix[:3, :3].T + matrix[:3, 3]
                mocap_object_prim.points_attr.Set([Gf.Vec3f(*row) for row in world_vertices])
            if layer_visible["mocap"] and mocap_overlay is not None:
                for marker, position in zip(
                    mocap_tip_markers, mocap_overlay.raw_mano_fingertips_world[frame], strict=True
                ):
                    marker.Set(Gf.Vec3d(*position))
            object_color = object_prim.base_color
            if trace.frame_worst_penetration_m is not None:
                depth = float(trace.frame_worst_penetration_m[frame, replica])
                if depth > 0.003:
                    object_color = (1.0, 0.05, 0.05)
                elif depth > 0.0:
                    object_color = (1.0, 0.42, 0.05)
            object_prim.color_attr.Set([Gf.Vec3f(*object_color)])
            sim.render()
            line = _status_line(trace, frame, replica)
            if ppo_metadata:
                action = np.asarray(trace.policy_action(frame, args.replica))
                if action.shape == (26,):
                    line += (
                        f" policy_action_norm={np.linalg.norm(action):.4f}"
                        f" wrist_residual_norm={np.linalg.norm(action[:6]):.4f}"
                        f" finger_residual_norm={np.linalg.norm(action[6:]):.4f}"
                        " reference_frame=embedded_ppo"
                    )
                reward_total = ppo_metadata.get("reward_total")
                if isinstance(reward_total, np.ndarray):
                    line += f" reward={float(reward_total[frame]):.4f}"
            print("\r" + line, end="", flush=True)

        center, radius = trace.camera_bounds(args.replica, frames)
        if reference_object_pose is not None:
            reference_positions = reference_object_pose[np.asarray(list(frames)), :3]
            low = np.minimum(center - radius, reference_positions.min(axis=0))
            high = np.maximum(center + radius, reference_positions.max(axis=0))
            center = (low + high) * 0.5
            radius = max(float(np.linalg.norm(high - low)), 0.12)
        if mocap_overlay is not None:
            selected = np.asarray(list(frames), dtype=np.int64)
            mocap_positions = np.concatenate(
                (
                    mocap_overlay.raw_mano_vertices_world[selected].reshape(-1, 3),
                    mocap_overlay.raw_object_pose_world_wxyz[selected, :3],
                ),
                axis=0,
            )
            low = np.minimum(center - radius, mocap_positions.min(axis=0))
            high = np.maximum(center + radius, mocap_positions.max(axis=0))
            center = (low + high) * 0.5
            radius = max(float(np.linalg.norm(high - low)), 0.12)
        eye = center + np.asarray((1.35, 1.15, 0.85)) * radius
        sim.set_camera_view(eye.tolist(), center.tolist())
        sim.reset()

        qualification_metrics = trace.qualification_metrics or {}
        reference_label = (
            "embedded_ppo_reference"
            if embedded_reference is not None
            else str(reference_path.resolve())
            if reference_object_pose is not None
            else "none"
        )
        ghost_kind = (
            "embedded_ppo_reference"
            if ppo_metadata and args.reference_ghost
            else "factor8_hocap_reference"
            if reference_object_pose is not None
            else "disabled"
        )
        print(
            f"REPLAY_INPUT kind={trace.trace_kind} frames={trace.frame_count} "
            f"qualification={trace.qualification_status} metrics={qualification_metrics} "
            "reference_ghost="
            f"{ghost_kind} "
            f"inferred_table_rendered={inferred_table_rendered} "
            f"reference={reference_label}",
            flush=True,
        )
        if mocap_overlay is not None:
            alignment_status = mocap_overlay.coordinate_alignment["status"]
            time_status = mocap_overlay.time_alignment["status"]
            print(
                "RAW_MOCAP_GHOST=AVAILABLE "
                f"clip={mocap_overlay.clip} alignment={alignment_status} "
                f"time={time_status} "
                f"object_visible={args.mocap_object} fingertips_visible={args.mocap_fingertips} "
                f"object_display_faces="
                f"{0 if mocap_object_faces is None else len(mocap_object_faces)}/"
                f"{len(mocap_overlay.raw_object_faces)}",
                flush=True,
            )
        elif args.mocap_ghost:
            print(f"RAW_MOCAP_GHOST_UNAVAILABLE={mocap_unavailable}", flush=True)

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
        if args.validation_output is not None:
            _write_validation_receipt(
                args.validation_output,
                args=args,
                trace=trace,
                frames=frames,
                reference_ghost=ghost_kind,
                ppo_metadata=ppo_metadata,
                inferred_table_rendered=inferred_table_rendered,
                mocap_overlay=mocap_overlay,
                mocap_unavailable=mocap_unavailable,
            )
        if args.mocap_similarity_output is not None:
            if mocap_overlay is None:
                raise RuntimeError(f"RAW_MOCAP_SIMILARITY_UNAVAILABLE:{mocap_unavailable}")
            _write_mocap_similarity(
                args.mocap_similarity_output,
                trace=trace,
                replica=args.replica,
                overlay=mocap_overlay,
            )
        print(
            f"REPLAY_COMPLETE object={object_id} trace={args.trace.resolve()} "
            f"replica={args.replica} frames={frames.start}:{frames.stop} "
            f"qualification={trace.qualification_status}",
            flush=True,
        )
        return 0
    finally:
        if (
            keyboard_input is not None
            and keyboard is not None
            and keyboard_subscription is not None
        ):
            keyboard_input.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        ghost_ui = None
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
