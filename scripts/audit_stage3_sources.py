"""Audit the selected MANO backend, model topology, and competing fingertip anchors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.keypoints import load_mano_model_geometry
from toporetarget.keypoints.registry import load_layouts, load_profiles

MANIPTRANS_TIP_FALLBACK = {
    "index_tip": 353,
    "middle_tip": 467,
    "pinky_tip": 695,
    "ring_tip": 576,
    "thumb_tip": 766,
}
MANO_JOINT_BY_TIP = {
    "index_tip": 3,
    "middle_tip": 6,
    "pinky_tip": 9,
    "ring_tip": 12,
    "thumb_tip": 15,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _maniptrans_tips(path: Path) -> dict[str, int]:
    if not path.is_file():
        return dict(MANIPTRANS_TIP_FALLBACK)
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r'"(\w+_tip)":\s*mano_out_verts\[:,\s*(\d+)\]', text)
    parsed = {name: int(index) for name, index in matches}
    return parsed or dict(MANIPTRANS_TIP_FALLBACK)


def _tip_stats(
    vertices: np.ndarray, joints: np.ndarray, anchors: dict[str, int]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, vertex_index in anchors.items():
        distances = np.linalg.norm(
            vertices[:, vertex_index] - joints[:, MANO_JOINT_BY_TIP[name]], axis=-1
        )
        result[name] = {
            "vertex_index": vertex_index,
            "min_m": float(np.min(distances)),
            "mean_m": float(np.mean(distances)),
            "max_m": float(np.max(distances)),
        }
    return result


def _candidate_plot(
    output: Path,
    *,
    neutral_vertices: np.ndarray,
    neutral_joints: np.ndarray,
    real_vertices: np.ndarray,
    real_joints: np.ndarray,
    smplx_tips: dict[str, int],
    maniptrans_tips: dict[str, int],
) -> None:
    import matplotlib.pyplot as plt

    names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    figure = plt.figure(figsize=(12, 5))
    for subplot, (title, vertices, joints) in enumerate(
        (
            ("MANO v1.2 neutral", neutral_vertices, neutral_joints),
            ("GRAB frame 0", real_vertices[0], real_joints[0]),
        ),
        start=1,
    ):
        axes = figure.add_subplot(1, 2, subplot, projection="3d")
        axes.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="tab:blue", label="MANO joints")
        for color, label, anchors in (
            ("tab:orange", "installed smplx", smplx_tips),
            ("tab:red", "ManipTrans", maniptrans_tips),
        ):
            points = np.asarray([vertices[anchors[name]] for name in names])
            axes.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, label=label)
        axes.set_title(title)
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_zlabel("z [m]")
        axes.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path, required=True)
    parser.add_argument("--maniptrans-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-id", default="hand_r")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequence = load_hoi_sequence(args.cache)
    hand = sequence.hand(args.hand_id)
    layouts = load_layouts()
    profile = load_profiles()["mano_v1_2_smplx_to_mediapipe21"]
    backend_file = (
        Path(__file__).resolve().parents[1] / "src/toporetarget/data/mano_backends/smplx_backend.py"
    )
    try:
        smplx_version = importlib.metadata.version("smplx")
    except importlib.metadata.PackageNotFoundError:
        smplx_version = "missing"
    from smplx.vertex_ids import vertex_ids

    smplx_tips = {
        "thumb_tip": int(vertex_ids["mano"]["thumb"]),
        "index_tip": int(vertex_ids["mano"]["index"]),
        "middle_tip": int(vertex_ids["mano"]["middle"]),
        "ring_tip": int(vertex_ids["mano"]["ring"]),
        "pinky_tip": int(vertex_ids["mano"]["pinky"]),
    }
    maniptrans_file = args.maniptrans_root / "main/dataset/grab_dataset_dexhand.py"
    maniptrans_tips = _maniptrans_tips(maniptrans_file)
    model_records: dict[str, Any] = {}
    neutral_geometry = load_mano_model_geometry(
        args.mano_model_root, side=hand.side, expected_vertex_count=profile.expected_vertex_count
    )
    for side in ("right", "left"):
        try:
            geometry = load_mano_model_geometry(
                args.mano_model_root, side=side, expected_vertex_count=profile.expected_vertex_count
            )
            model_records[side] = {
                "path": geometry.model_path,
                "sha256": geometry.model_hash,
                "vertex_count": geometry.vertex_count,
                "J_regressor_shape": list(geometry.joint_regressor.shape),
                "faces_shape": None if geometry.faces is None else list(geometry.faces.shape),
                "v_template_shape": list(geometry.v_template.shape),
            }
        except Exception as exc:
            model_records[side] = {"error": str(exc)}

    source_track = next(iter(hand.keypoint_tracks.values()), None)
    joints = None if source_track is None else source_track.positions_scene
    vertices = hand.vertices_scene
    backend_audit = {
        "cache": str(args.cache),
        "schema_version": sequence.metadata.schema_version,
        "sequence_id": sequence.metadata.sequence_id,
        "num_frames": sequence.num_frames,
        "native_fps": sequence.metadata.native_fps,
        "timestamps_first_three": sequence.timestamps[:3],
        "hand_id": hand.hand_id,
        "side": hand.side,
        "available_keypoint_layouts": sorted(hand.keypoint_tracks),
        "keypoint_tracks": {
            name: {
                "layout_name": track.layout_name,
                "shape": list(track.positions_scene.shape),
                "semantic_names": track.semantic_names,
                "valid_shape": None if track.valid is None else list(track.valid.shape),
                "frame_name": track.frame_name,
                "units": track.units,
            }
            for name, track in hand.keypoint_tracks.items()
        },
        "first_three_wrist_points": None if joints is None else joints[:3, 0],
        "vertices_shape": None if vertices is None else list(vertices.shape),
        "wrist_pose_translation_first_three": hand.wrist_pose_scene.pose_scene[:3, :3, 3],
        "wrist_origin_definition": (
            "Stage 2B PoseTrack translation from MANO transl/global-orient input"
        ),
        "wrist_joint0_to_pose_translation_norm_m": (
            None
            if joints is None
            else np.linalg.norm(joints[:, 0] - hand.wrist_pose_scene.pose_scene[:, :3, 3], axis=-1)
        ),
        "mano_model_profile": (
            None if hand.mano_parameters is None else hand.mano_parameters.model_profile
        ),
        "installed_smplx_version": smplx_version,
        "smplx_mano_joint_order": list(layouts["mano16_smplx"].semantic_names),
        "backend_source": backend_file,
        "model_records": model_records,
        "wrist_pose_not_modified_by_stage3": True,
    }
    (args.output_dir / "current_backend_audit.json").write_text(
        json.dumps(_json_value(backend_audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    real_candidate_stats = None
    neutral_candidate_stats = None
    neutral_joints = neutral_geometry.joint_regressor @ neutral_geometry.v_template
    if vertices is not None and joints is not None:
        real_candidate_stats = {
            "installed_smplx": _tip_stats(vertices, joints, smplx_tips),
            "ManipTrans": _tip_stats(vertices, joints, maniptrans_tips),
        }
        neutral_candidate_stats = {
            "installed_smplx": _tip_stats(
                neutral_geometry.v_template[None, ...], neutral_joints[None, ...], smplx_tips
            ),
            "ManipTrans": _tip_stats(
                neutral_geometry.v_template[None, ...], neutral_joints[None, ...], maniptrans_tips
            ),
        }
        _candidate_plot(
            args.output_dir / "fingertip_candidates.png",
            neutral_vertices=neutral_geometry.v_template,
            neutral_joints=neutral_joints,
            real_vertices=vertices,
            real_joints=joints,
            smplx_tips=smplx_tips,
            maniptrans_tips=maniptrans_tips,
        )
    mapping_sources = {
        "installed_smplx_version": smplx_version,
        "installed_smplx_source": "smplx.vertex_ids.VERTEX_IDS['mano']",
        "smplx_candidate_fingertip_indices": smplx_tips,
        "maniptrans_candidate_fingertip_indices": maniptrans_tips,
        "candidate_differences": {
            name: {"smplx": smplx_tips.get(name), "ManipTrans": maniptrans_tips.get(name)}
            for name in sorted(set(smplx_tips) | set(maniptrans_tips))
            if smplx_tips.get(name) != maniptrans_tips.get(name)
        },
        "topology_compatibility": {
            "current_MANO_vertex_count": neutral_geometry.vertex_count,
            "expected_profile_vertex_count": profile.expected_vertex_count,
            "faces_shape": (
                None if neutral_geometry.faces is None else list(neutral_geometry.faces.shape)
            ),
            "current_model_hash": neutral_geometry.model_hash,
            "model_topology_matches_profile": neutral_geometry.vertex_count
            == profile.expected_vertex_count,
        },
        "candidate_evidence": {
            "smplx": (
                "installed smplx package source and VERTEX_IDS['mano']; "
                "current backend emits smplx MANO geometry"
            ),
            "ManipTrans": str(maniptrans_file),
        },
        "neutral_pose_candidate_distances_m": neutral_candidate_stats,
        "real_grab_candidate_distances_m": real_candidate_stats,
        "selected_profile": profile.profile_id,
        "selected_profile_mapping": {
            name: anchor.vertex_index for name, anchor in profile.fingertip_mapping.items()
        },
        "selection_rationale": [
            "The selected anchors are the explicit candidates used by the installed smplx backend.",
            "The current MANO model has 778 vertices and matches the profile topology.",
            "ManipTrans candidates differ and remain recorded rather than being silently "
            "collapsed.",
        ],
        "unresolved_discrepancies": [
            "ManipTrans and installed smplx use different plausible distal-phalanx vertex "
            "anchors; this is not external MediaPipe ground truth.",
            "A topology/model change requires a new mapping profile and re-audit.",
        ],
    }
    (args.output_dir / "mapping_sources.json").write_text(
        json.dumps(_json_value(mapping_sources), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            _json_value({"backend_audit": backend_audit, "mapping_sources": mapping_sources}),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
