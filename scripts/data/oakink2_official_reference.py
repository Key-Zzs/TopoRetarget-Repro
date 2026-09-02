#!/usr/bin/env python3
"""Export exact-frame MANO geometry through OakInk2's official loader path.

This module intentionally does not import the TopoRetarget OakInk2 adapter.
It is the independent ``OfficialOakInk2MANOReferenceV1`` side of the O1R
comparison and must run in an environment containing the official
``oakink2_toolkit`` and ``manotorch`` packages.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from oakink2_toolkit.dataset import OakInk2__Dataset


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _find_primitive_identifier(complex_task: Any, primitive_key: str) -> str:
    expected = ast.literal_eval(primitive_key)
    matches = [
        identifier
        for identifier, frame_range_def in complex_task.exec_range_map.items()
        if frame_range_def == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(f"OFFICIAL_PRIMITIVE_KEY_RESOLUTION_FAILED:{primitive_key}:{matches}")
    return str(matches[0])


def _prepare_mano_root(model_path: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="oakink2_o1r_official_mano_")
    root = Path(temporary.name)
    models = root / "models"
    models.mkdir()
    os.symlink(model_path.resolve(), models / "MANO_RIGHT.pkl")
    return temporary, root


def export_reference(
    dataset_prefix: Path,
    fixed_review_set_path: Path,
    mano_model: Path,
    output_root: Path,
) -> dict[str, Any]:
    fixed = json.loads(fixed_review_set_path.read_text(encoding="utf-8"))
    episodes = fixed.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 2:
        raise RuntimeError("OFFICIAL_FIXED_REVIEW_SET_MUST_HAVE_TWO_EPISODES")
    if not mano_model.is_file():
        raise RuntimeError(f"OFFICIAL_MANO_MODEL_MISSING:{mano_model}")

    temporary, mano_root = _prepare_mano_root(mano_model)
    try:
        layer = ManoLayer(
            mano_assets_root=str(mano_root),
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to(torch.device("cpu"))
        layer.eval()
        closed_faces = layer.get_mano_closed_faces().detach().cpu().numpy().astype(np.int64)
        dataset = OakInk2__Dataset(dataset_prefix=str(dataset_prefix), return_instantiated=True)
        episode_receipts: list[dict[str, Any]] = []
        for episode in episodes:
            label = str(episode["review"])
            sequence_token = str(episode["sequence_id"])
            sequence = sequence_token.replace("++", "/", 1)
            requested_frames = [int(value) for value in episode["sampled_mocap_frames"]]
            complex_task = dataset.load_complex_task(sequence)
            identifier = _find_primitive_identifier(complex_task, str(episode["primitive_key"]))
            primitive = dataset.load_primitive_task(complex_task, identifier)
            annotation_path = Path(dataset.anno_prefix) / f"{complex_task.seq_token}.pkl"
            with annotation_path.open("rb") as handle:
                annotation = pickle.load(handle)
            mocap_ids = [int(value) for value in annotation["mocap_frame_id_list"]]
            position_by_mocap = {frame: index for index, frame in enumerate(mocap_ids)}
            if len(position_by_mocap) != len(mocap_ids):
                raise RuntimeError(f"OFFICIAL_DUPLICATE_MOCAP_FRAME_ID:{sequence}")
            raw_mano = annotation["raw_mano"]
            raw_smplx = annotation.get("raw_smplx", {})
            image_ids = {int(value) for value in annotation.get("frame_id_list", [])}
            target = str(episode["target_object"])
            object_track = annotation["obj_transf"].get(target)
            if not isinstance(object_track, dict):
                raise RuntimeError(f"OFFICIAL_TARGET_OBJECT_TRACK_MISSING:{target}")

            frame_receipts: list[dict[str, Any]] = []
            for frame in requested_frames:
                if frame not in position_by_mocap:
                    raise RuntimeError(f"SOURCE_FRAME_MISSING:{sequence}:{frame}")
                if frame not in raw_mano or frame not in object_track:
                    raise RuntimeError(f"SOURCE_FRAME_MISSING:{sequence}:{frame}")
                start, end = primitive.frame_range
                offset = frame - int(start)
                if not 0 <= offset < int(end) - int(start):
                    raise RuntimeError(
                        f"OFFICIAL_FRAME_OUTSIDE_PRIMITIVE:{frame}:{primitive.frame_range}"
                    )
                if not bool(primitive.rh_in_range_mask[offset].item()):
                    raise RuntimeError(f"OFFICIAL_FRAME_OUTSIDE_RIGHT_INTERVAL:{frame}")

                loader_pose = primitive.rh_param["pose_coeffs"][offset : offset + 1]
                loader_betas = primitive.rh_param["betas"][offset : offset + 1]
                loader_tsl = primitive.rh_param["tsl"][offset : offset + 1]
                direct = raw_mano[frame]
                direct_pose = _array(direct["rh__pose_coeffs"])
                direct_betas = _array(direct["rh__betas"])
                direct_tsl = _array(direct["rh__tsl"])
                if not (
                    np.array_equal(_array(loader_pose), direct_pose)
                    and np.array_equal(_array(loader_betas), direct_betas)
                    and np.array_equal(_array(loader_tsl), direct_tsl)
                ):
                    raise RuntimeError(f"OFFICIAL_LOADER_RAW_FRAME_BINDING_MISMATCH:{frame}")

                with torch.no_grad():
                    mano = layer(pose_coeffs=loader_pose, betas=loader_betas)
                vertices = mano.verts + loader_tsl.unsqueeze(1)
                joints = mano.joints + loader_tsl.unsqueeze(1)
                frame_root = output_root / label / f"frame_{frame}"
                frame_root.mkdir(parents=True, exist_ok=True)
                np.save(frame_root / "official_mano_vertices.npy", _array(vertices)[0])
                np.save(frame_root / "official_mano_joints.npy", _array(joints)[0])
                np.save(frame_root / "official_mano_faces.npy", closed_faces)
                np.savez(
                    frame_root / "official_raw_params.npz",
                    pose_coeffs=direct_pose,
                    betas=direct_betas,
                    tsl=direct_tsl,
                )
                receipt = {
                    "authority": "OfficialOakInk2MANOReferenceV1",
                    "sequence_id": sequence,
                    "sequence_token": sequence_token,
                    "primitive_identifier": identifier,
                    "primitive_interval": list(map(int, primitive.frame_range)),
                    "right_hand_interval": list(map(int, primitive.frame_range_rh)),
                    "requested_mocap_frame_id": frame,
                    "mocap_position": position_by_mocap[frame],
                    "mocap_position_equals_frame_id": position_by_mocap[frame] == frame,
                    "is_image_frame_id": frame in image_ids,
                    "raw_mano_key": frame,
                    "raw_smplx_key": frame if frame in raw_smplx else None,
                    "object_transform_key": frame,
                    "official_loader_selected_frame": frame,
                    "official_loader_matches_direct_raw_params": True,
                    "mano_model_path": str(mano_model.resolve()),
                    "mano_model_sha256": _sha256(mano_model),
                    "vertices_shape": list(vertices.shape[1:]),
                    "joints_shape": list(joints.shape[1:]),
                    "closed_faces_shape": list(closed_faces.shape),
                    "status": "OFFICIAL_EXACT_FRAME_EXPORTED",
                }
                _write_json(frame_root / "receipt.json", receipt)
                frame_receipts.append(receipt)
            episode_receipts.append(
                {
                    "review": label,
                    "record_id": episode["record_id"],
                    "sequence_id": sequence,
                    "sequence_token": sequence_token,
                    "primitive_identifier": identifier,
                    "sampled_mocap_frames": requested_frames,
                    "frames": frame_receipts,
                }
            )
    finally:
        temporary.cleanup()

    result = {
        "authority": "OfficialOakInk2MANOReferenceV1",
        "official_adapter_imported": False,
        "dataset_prefix": str(dataset_prefix.resolve()),
        "mano_model_path": str(mano_model.resolve()),
        "mano_model_sha256": _sha256(mano_model),
        "manolayer": {
            "rot_mode": "quat",
            "side": "right",
            "center_idx": 0,
            "use_pca": False,
            "flat_hand_mean": True,
            "translation": "mano_out vertices/joints plus rh__tsl",
            "faces": "get_mano_closed_faces",
        },
        "episodes": episode_receipts,
        "status": "OFFICIAL_REFERENCE_EXPORT_COMPLETE",
    }
    _write_json(output_root / "official_export_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-prefix", type=Path, required=True)
    parser.add_argument("--fixed-review-set", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_reference(
                args.dataset_prefix,
                args.fixed_review_set,
                args.mano_model,
                args.output_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
