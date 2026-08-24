#!/usr/bin/env python3
"""Prepare and visualize finite support for one frozen physical-HOI clip.

This command is intentionally useful in headless mode.  It writes only under
``.local`` by default and never edits source data, PPO artifacts, or runtime
object state.  ``--static`` and ``--replay`` are geometry visualizations; a
PhysX result is accepted only through an explicit telemetry receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.support import (  # noqa: E402
    SupportResolutionMode,
    evidence_from_sequence_directory,
    qualify_geometry,
    resolve_support,
    transform_mesh_trajectory,
    validate_and_finalize_resolution,
    validate_hand_table_geometry,
    validate_object_table_geometry,
    write_finite_planar_support_usda,
)
from toporetarget.physics.support.types import jsonable  # noqa: E402
from toporetarget.physics.support_contract import sha256_file  # noqa: E402
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    load_runtime_geometry_manifest,
)
from toporetarget.rl.geometry_audit.transforms import (  # noqa: E402
    compose_poses,
    transform_points,
)

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
DEFAULT_OBJECT_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/objects"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction"
DEFAULT_SOURCE_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/subject_1")
DEFAULT_SUPPORT_ASSET_ROOT = REPO_ROOT / ".local/support_assets/hocap"


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                if len(values) >= 4:
                    vertices.append([float(values[1]), float(values[2]), float(values[3])])
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError(f"OBJECT_OBJ_HAS_NO_VERTICES:{path}")
    return result


def _collision_vertices(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import trimesh

        loaded = trimesh.load(path, force="mesh", process=False)
        hull = loaded.convex_hull
        vertices = np.asarray(hull.vertices, dtype=np.float64)
        return vertices, {
            "source": "trimesh_convex_hull_v1",
            "source_mesh": str(path.resolve()),
            "source_mesh_sha256": sha256_file(path),
            "vertex_count": int(len(vertices)),
            "runtime_asset_equivalence": (
                "matches convex-hull family; verify against Isaac USD manifest"
            ),
        }
    except Exception as error:  # pragma: no cover - optional dependency fallback
        visual = _load_obj_vertices(path)
        bounds = np.column_stack((visual.min(axis=0), visual.max(axis=0)))
        corners = np.asarray(
            [[bounds[axis, bit >> axis & 1] for axis in range(3)] for bit in range(8)],
            dtype=np.float64,
        )
        return corners, {
            "source": "visual_aabb_fallback_not_runtime_qualified",
            "error": f"{type(error).__name__}: {error}",
            "vertex_count": int(len(corners)),
        }


def _hand_collision_points_world(
    reference: dict[str, np.ndarray], geometry_manifest: Path
) -> np.ndarray:
    wrist = np.concatenate(
        (
            np.asarray(reference["wrist_pose_translation_world_ref"], dtype=np.float64),
            np.asarray(reference["wrist_pose_quaternion_world_ref_wxyz"], dtype=np.float64),
        ),
        axis=1,
    )
    qpos = np.asarray(reference["q_finger_ref"], dtype=np.float64)
    body_pose = reconstruct_hand_collision_body_pose(wrist, qpos, repo_root=REPO_ROOT)
    proxies, _objects = load_runtime_geometry_manifest(geometry_manifest)
    if tuple(proxy.body_name for proxy in proxies) != HAND_COLLISION_BODY_NAMES:
        raise ValueError("INDEPENDENT_SUPPORT_HAND_GEOMETRY_BODY_ORDER_INVALID")
    frames: list[np.ndarray] = []
    for frame in range(len(body_pose)):
        points = [
            transform_points(
                proxy.scaled_vertices,
                compose_poses(body_pose[frame, index], proxy.local_pose_xyz_wxyz),
            )
            for index, proxy in enumerate(proxies)
        ]
        frames.append(np.concatenate(points, axis=0))
    result = np.stack(frames)
    if not np.isfinite(result).all():
        raise ValueError("INDEPENDENT_SUPPORT_HAND_GEOMETRY_NONFINITE")
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_evidence(clip: str, source_root: Path, source_sequence_dir: Path | None) -> object:
    source_dir = (
        source_sequence_dir.resolve()
        if source_sequence_dir is not None
        else source_root / f"20231025_{clip.removeprefix('hocap_')}"
    )
    try:
        return evidence_from_sequence_directory(source_dir)
    except FileNotFoundError as error:
        return {
            "explicit": False,
            "recovered": False,
            "provenance": {
                "sequence_dir": str(source_dir),
                "audit_status": "SOURCE_SEQUENCE_NOT_MOUNTED",
                "error": str(error),
            },
        }


def _native_contact_mask(path: Path | None, *, frame_count: int) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        if "combined_support_exclusion_mask" in archive.files:
            combined = np.asarray(archive["combined_support_exclusion_mask"], dtype=bool)
            if combined.shape != (frame_count,):
                raise ValueError("INDEPENDENT_SUPPORT_COMBINED_CONTACT_SHAPE_INVALID")
            return combined
        required = {"confirmed_contact", "probable_contact", "transition"}
        if not required.issubset(archive.files):
            raise ValueError("INDEPENDENT_SUPPORT_NATIVE_CONTACT_KEYS_MISSING")
        confirmed = np.asarray(archive["confirmed_contact"], dtype=bool)
        probable = np.asarray(archive["probable_contact"], dtype=bool)
        transition = np.asarray(archive["transition"], dtype=bool)
    if confirmed.shape != probable.shape or confirmed.shape != transition.shape:
        raise ValueError("INDEPENDENT_SUPPORT_NATIVE_CONTACT_SHAPE_INVALID")
    if confirmed.ndim != 2:
        raise ValueError("INDEPENDENT_SUPPORT_NATIVE_CONTACT_SHAPE_INVALID")
    if confirmed.shape[0] != frame_count:
        raise ValueError("INDEPENDENT_SUPPORT_NATIVE_CONTACT_FRAME_COUNT_MISMATCH")
    # A transition frame is deliberately excluded from a plane-fit interval:
    # it is a contact-boundary observation, not clean pre-contact evidence.
    return np.any(confirmed | probable | transition, axis=1)


def _plot_clip(
    *,
    clip: str,
    reference: dict[str, np.ndarray],
    visual_world: np.ndarray,
    collision_world: np.ndarray,
    result: object,
    destination: Path,
    replay: bool,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional visualization dependency
        raise RuntimeError("SUPPORT_VISUALIZATION_REQUIRES_MATPLOTLIB") from error
    proxy = result.table_proxy
    if proxy is None:
        return []
    from toporetarget.physics.support.runtime_support import table_top_corners

    corners = table_top_corners(proxy)
    frame_indices = [
        proxy_index for proxy_index in (0, result.support_interval.end_frame_exclusive - 1)
    ]
    frame_indices.extend([min(len(visual_world) - 1, index) for index in (19, 27, 35, 40)])
    if not replay:
        frame_indices = [0]
    frame_indices = list(
        dict.fromkeys(index for index in frame_indices if 0 <= index < len(visual_world))
    )
    written: list[str] = []
    for frame in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        object_points = visual_world[frame]
        collision_points = collision_world[frame]
        axes[0].scatter(
            object_points[:, 0], object_points[:, 2], s=1, alpha=0.22, label="object visual"
        )
        axes[0].scatter(
            collision_points[:, 0], collision_points[:, 2], s=6, alpha=0.6, label="collision proxy"
        )
        axes[0].plot(
            np.r_[corners[:, 0], corners[0, 0]],
            np.r_[corners[:, 2], corners[0, 2]],
            "b-",
            label="table top",
        )
        axes[0].set_xlabel("world x (m)")
        axes[0].set_ylabel("world z (m)")
        axes[0].set_title(f"{clip} frame {frame}: side")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.2)
        axes[1].scatter(
            object_points[:, 0], object_points[:, 1], s=1, alpha=0.22, label="object visual"
        )
        axes[1].plot(
            np.r_[corners[:, 0], corners[0, 0]],
            np.r_[corners[:, 1], corners[0, 1]],
            "b-",
            label="table extent",
        )
        axes[1].set_xlabel("world x (m)")
        axes[1].set_ylabel("world y (m)")
        axes[1].set_title("support footprint")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.2)
        path = destination / f"{clip}_frame_{frame:03d}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def _resolve_clip(
    *,
    clip: str,
    mode: SupportResolutionMode,
    reference_root: Path,
    object_root: Path,
    source_root: Path,
    source_sequence_dir: Path | None,
    support_asset_root: Path,
    output_root: Path,
    static: bool,
    replay: bool,
    runtime_geometry_manifest: Path | None,
    source_contact_native: Path | None,
) -> dict[str, object]:
    reference_path = reference_root / f"{clip}.world_wrist.stage16.npz"
    object_path = object_root / f"{clip}.obj"
    with np.load(reference_path, allow_pickle=False) as loaded:
        reference = {key: loaded[key] for key in loaded.files}
    visual = _load_obj_vertices(object_path)
    collision, collision_evidence = _collision_vertices(object_path)
    source = _source_evidence(clip, source_root, source_sequence_dir)
    contact_mask = _native_contact_mask(
        source_contact_native, frame_count=len(reference["timestamps"])
    )
    result = resolve_support(
        dataset="hocap",
        sequence=clip,
        object_visual_vertices_local=visual,
        object_collision_vertices_local=collision,
        object_pose_translation_world=reference["object_pose_translation_world_ref"],
        object_pose_quaternion_world_wxyz=reference["object_pose_quaternion_world_ref_wxyz"],
        timestamps=reference["timestamps"],
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support=source,
        contact_mask=contact_mask,
        object_twist_world=reference.get("object_twist_world_ref"),
        mode=mode,
        source_reference_kind="stage16_retargeted_runtime_object_pose",
    )
    clip_root = output_root / "inference" / clip
    _write_json(
        output_root / "source_evidence" / f"{clip}.json",
        source.as_dict() if hasattr(source, "as_dict") else source,
    )
    _write_json(
        clip_root / "stable_interval.json",
        result.stable_interval.as_dict() if result.stable_interval else {},
    )
    _write_json(
        clip_root / "plane_fit.json", result.plane_fit.as_dict() if result.plane_fit else {}
    )
    _write_json(
        clip_root / "support_patch.json", result.plane_fit.as_dict() if result.plane_fit else {}
    )
    _write_json(
        clip_root / "table_proxy.json", result.table_proxy.as_dict() if result.table_proxy else {}
    )
    _write_json(clip_root / "collision_mesh_evidence.json", collision_evidence)
    if result.table_proxy is not None and result.support_interval is not None:
        support_asset_root = support_asset_root / clip
        support_asset = write_finite_planar_support_usda(
            result.table_proxy, support_asset_root / "support_proxy.usda"
        )
        _write_json(
            support_asset_root / "provenance.json",
            {
                "schema_version": "Stage16SupportAssetProvenanceV1",
                "asset": str(support_asset.resolve()),
                "support_type": result.support_type.value,
                "support_inferred": result.support_inferred,
                "support_resolution_hashes": result.hashes,
            },
        )
        visual_world = transform_mesh_trajectory(
            visual,
            reference["object_pose_translation_world_ref"],
            reference["object_pose_quaternion_world_ref_wxyz"],
        )
        collision_world = transform_mesh_trajectory(
            collision,
            reference["object_pose_translation_world_ref"],
            reference["object_pose_quaternion_world_ref_wxyz"],
        )
        interval = (
            result.support_interval.start_frame,
            min(len(visual_world), result.support_interval.end_frame_exclusive + 4),
        )
        object_geometry = validate_object_table_geometry(
            visual_vertices_local=visual,
            collision_vertices_local=collision,
            object_translation_world=reference["object_pose_translation_world_ref"],
            object_quaternion_world_wxyz=reference["object_pose_quaternion_world_ref_wxyz"],
            plane_normal=result.plane_fit.plane_normal,
            plane_offset=result.plane_fit.plane_offset,
            table_extent=result.table_proxy.table_extent,
            table_pose=result.table_proxy.table_pose,
            relevant_interval=interval,
        )
        # Inferred support restores object support only.  Hand/proxy geometry is
        # retained as a diagnostic even when full runtime collision proxies are
        # available; pairwise runtime filtering keeps it out of PF acceptance.
        tracked_links = reference.get("tracked_link_positions_world_ref")
        link_diagnostic: dict[str, object] = {"status": "NOT_AVAILABLE"}
        if tracked_links is not None:
            link_signed = (
                np.einsum("tvi,i->tv", tracked_links, np.asarray(result.plane_fit.plane_normal))
                - result.plane_fit.plane_offset
            )
            link_diagnostic = {
                "status": "DIAGNOSTIC_ONLY",
                "source": "tracked_16_link_points_only",
                "minimum_signed_distance_m": float(np.min(link_signed)),
                "minimum_by_frame_m": np.min(link_signed, axis=1).tolist(),
            }
        hand_points = (
            None
            if runtime_geometry_manifest is None
            else _hand_collision_points_world(reference, runtime_geometry_manifest)
        )
        hand_geometry = validate_hand_table_geometry(
            hand_points_world=hand_points,
            plane_normal=result.plane_fit.plane_normal,
            plane_offset=result.plane_fit.plane_offset,
            table_extent=result.table_proxy.table_extent,
            table_pose=result.table_proxy.table_pose,
            source=(
                "tracked_16_link_points_only; full hand mesh not supplied"
                if hand_points is None
                else "authored_runtime_hand_convex_proxies_on_retargeted_reference_fk"
            ),
        )
        hand_geometry["link_point_diagnostic"] = link_diagnostic
        geometry = qualify_geometry(
            object_table=object_geometry,
            hand_table=hand_geometry,
            visual_collision_consistent=abs(result.plane_fit.delta_support_geometry) <= 0.005,
            hand_table_is_diagnostic_only=True,
        )
        _write_json(clip_root / "object_table_geometry.json", object_geometry)
        _write_json(clip_root / "hand_table_geometry.json", hand_geometry)
        _write_json(clip_root / "geometry_validation.json", geometry.as_dict())
        screenshot_root = output_root / "screenshots"
        screenshot_root.mkdir(parents=True, exist_ok=True)
        if static or replay:
            _plot_clip(
                clip=clip,
                reference=reference,
                visual_world=visual_world,
                collision_world=collision_world,
                result=result,
                destination=screenshot_root,
                replay=replay,
            )
        result = validate_and_finalize_resolution(
            result, geometry=geometry, transfer_status="NOT_RUN"
        )
    else:
        geometry = None
        _write_json(
            clip_root / "geometry_validation.json",
            {"status": "DEFERRED", "reason": result.diagnostics},
        )
    _write_json(clip_root / "support_resolution.json", result.as_dict())
    _write_json(
        clip_root / "static_support_test.json",
        {"status": "NOT_RUN", "reason": "requires explicit Isaac PhysX telemetry receipt"},
    )
    _write_json(
        clip_root / "no_support_test.json",
        {"status": "NOT_RUN", "reason": "requires matched full-gravity Isaac PhysX counterfactual"},
    )
    _write_json(
        clip_root / "ab_comparison.json",
        {"status": "NOT_RUN", "reason": "requires matched full-gravity Isaac PhysX counterfactual"},
    )
    _write_json(
        clip_root / "support_transfer.json",
        {
            "status": "DEFERRED",
            "reason": (
                "requires runtime reference-following PhysX receipt and independent "
                "hand-object geometry clearance"
            ),
        },
    )
    return result.as_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hocap", choices=("hocap",))
    parser.add_argument("--sequence")
    parser.add_argument(
        "--support", default="auto", choices=[item.value for item in SupportResolutionMode]
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-sequence-dir",
        type=Path,
        help="Exact raw sequence directory for an independent clip.",
    )
    parser.add_argument("--support-asset-root", type=Path, default=DEFAULT_SUPPORT_ASSET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--runtime-geometry-manifest",
        type=Path,
        help="Frozen authored hand collision proxies for formal hand/table clearance.",
    )
    parser.add_argument(
        "--source-contact-native",
        type=Path,
        help="Hash-validated native source-contact authority supplied by the production runner.",
    )
    parser.add_argument("--static", action="store_true", help="write a static support overlay")
    parser.add_argument("--replay", action="store_true", help="write representative replay frames")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sequence is not None and (
        not args.sequence or any(token in args.sequence for token in ("/", "\\", ".."))
    ):
        raise ValueError("INDEPENDENT_SUPPORT_CLIP_ID_INVALID")
    if args.source_sequence_dir is not None and args.sequence is None:
        raise ValueError("--source-sequence-dir requires --sequence")
    clips = (args.sequence,) if args.sequence else CLIPS
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        rows.append(
            _resolve_clip(
                clip=clip,
                mode=SupportResolutionMode(args.support),
                reference_root=args.reference_root.resolve(),
                object_root=args.object_root.resolve(),
                source_root=args.source_root.resolve(),
                source_sequence_dir=args.source_sequence_dir,
                support_asset_root=args.support_asset_root.resolve(),
                output_root=output_root,
                static=args.static,
                replay=args.replay,
                runtime_geometry_manifest=(
                    None
                    if args.runtime_geometry_manifest is None
                    else args.runtime_geometry_manifest.resolve()
                ),
                source_contact_native=(
                    None
                    if args.source_contact_native is None
                    else args.source_contact_native.resolve()
                ),
            )
        )
    summary = {
        "schema_version": "Stage16SupportResolutionRunV1",
        "dataset": args.dataset,
        "clips": rows,
        "support_mode": args.support,
        "physics_backend": "NOT_RUN; explicit PhysX receipt required",
        "p3_status": "P3_RESTART_BLOCKED_BY_HAND_OBJECT_RESET_GEOMETRY",
    }
    _write_json(output_root / "final_summary.json", summary)
    (output_root / "visualization_commands.md").write_text(
        "# Stage 16 support reconstruction commands\n\n"
        "Geometry/static overlays:\n\n"
        "```bash\n"
        "PYTHONPATH=src python scripts/physics/prepare_physical_support.py "
        "--support auto --static --replay\n"
        "```\n\n"
        "Full-gravity object-only PhysX for each clip:\n\n"
        "```bash\n"
        "conda run --no-capture-output -n toporetarget-isaaclab "
        "env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src "
        "python scripts/physics/evaluate_physical_support.py "
        "--clip hocap_170105 --case with_support --steps 360 --accept-eula "
        "--support-asset .local/support_assets/hocap/hocap_170105/support_proxy.usda "
        "--proxy-json .local/reports/stage16_support_reconstruction/inference/"
        "hocap_170105/table_proxy.json\n"
        "conda run --no-capture-output -n toporetarget-isaaclab "
        "env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src "
        "python scripts/physics/evaluate_physical_support.py "
        "--clip hocap_170105 --case without_support --steps 360 --accept-eula\n"
        "PYTHONPATH=src python scripts/physics/summarize_physical_support.py\n"
        "```\n\n"
        "The same commands apply to hocap_170650 after changing clip and paths. "
        "See docs/physics/SUPPORT_RESOLUTION.md for the full contract.\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
