from pathlib import Path

import numpy as np
import pytest

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.readers.grab import GrabParseError, load_ply_mesh, read_grab_npz


def _write_ascii_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n"
        "1 0 0\n"
        "0 1 0\n"
        "3 0 1 2\n",
        encoding="ascii",
    )


def _write_fixture(root: Path, *, include_left: bool = True) -> Path:
    (root / "grab" / "s1").mkdir(parents=True)
    (root / "tools" / "object_meshes").mkdir(parents=True)
    (root / "tools" / "subject_meshes").mkdir(parents=True)
    _write_ascii_ply(root / "tools" / "object_meshes" / "object.ply")
    _write_ascii_ply(root / "tools" / "subject_meshes" / "hand.ply")
    params = {
        "global_orient": np.zeros((4, 3), dtype=np.float32),
        "hand_pose": np.zeros((4, 24), dtype=np.float32),
        "transl": np.zeros((4, 3), dtype=np.float32),
        "fullpose": np.zeros((4, 45), dtype=np.float32),
    }
    payload = {
        "gender": "female",
        "sbj_id": "s1",
        "framerate": 120.0,
        "obj_name": "object",
        "n_frames": 4,
        "n_comps": 24,
        "motion_intent": "pass",
        "body": {},
        "rhand": {"params": params, "vtemp": "tools/subject_meshes/hand.ply"},
        "object": {
            "params": {"global_orient": np.zeros((4, 3)), "transl": np.zeros((4, 3))},
            "object_mesh": "tools/object_meshes/object.ply",
        },
    }
    if include_left:
        payload["lhand"] = {"params": params, "vtemp": "tools/subject_meshes/hand.ply"}
    path = root / "grab" / "s1" / "demo.npz"
    np.savez(path, **payload)
    return path


def test_nested_grab_fields_and_ply_path_resolution(tmp_path) -> None:
    sequence_path = _write_fixture(tmp_path / "GRAB")
    record = read_grab_npz(sequence_path)
    assert record.object_name == "object"
    assert record.hand("right").params["fullpose"].shape == (4, 45)
    vertices, faces = load_ply_mesh(tmp_path / "GRAB" / "tools" / "object_meshes" / "object.ply")
    assert vertices.shape == (3, 3)
    assert faces.tolist() == [[0, 1, 2]]
    clip = record.clip(FrameRange(1, 3))
    assert clip.num_frames == 2
    assert clip.start_frame == 1


def test_missing_hand_and_invalid_path_are_actionable(tmp_path) -> None:
    sequence_path = _write_fixture(tmp_path / "GRAB", include_left=False)
    record = read_grab_npz(sequence_path)
    with pytest.raises(GrabParseError, match="no left hand"):
        record.hand("left")
    with pytest.raises(GrabParseError, match="existing .npz"):
        read_grab_npz(tmp_path / "missing.npz")
