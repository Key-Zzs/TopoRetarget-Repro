from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from toporetarget.viz.oakink2_html_viewer import (
    OakInk2HTMLViewerV2Data,
    TrustedHTMLViewerData,
    payload_for,
    vertex_normals,
)


def camera_presets() -> dict[str, dict[str, list[float]]]:
    matrix = np.eye(4, dtype=np.float32).T.reshape(-1).tolist()
    return {
        name: {
            "anatomy_camera_model_matrix": matrix,
            "anatomy_projection_matrix": matrix,
        }
        for name in ("FRONT", "OBLIQUE", "SIDE")
    }


def test_trusted_viewer_payload_is_precomputed_geometry_only() -> None:
    vertices = np.tile(
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        (1, 260, 1),
    )[:, :778]
    # A minimally valid 778-vertex topology keeps the serialization contract
    # independent of MANO reconstruction details.
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    data = TrustedHTMLViewerData(
        frames=np.array([4279]),
        hand_vertices_world=vertices,
        hand_vertices_anatomy=vertices,
        hand_faces_closed=faces,
        hand_faces_open=faces,
        hand_joints_world=np.zeros((1, 21, 3), dtype=np.float32),
        hand_joints_anatomy=np.zeros((1, 21, 3), dtype=np.float32),
        object_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        object_faces=faces,
        object_transforms=np.eye(4, dtype=np.float32)[None],
        primary_frame=4279,
        record={"target_object": "C10001", "object_centroid": [0.0, 0.0, 0.0]},
        camera_presets=camera_presets(),
    )

    payload = payload_for(data)

    assert payload["schemaVersion"] == "OakInk2HTMLViewerV2"
    assert payload["sceneFrame"] == "SCENE_WORLD_MANO_ROOT_RELATIVE"
    assert payload["contracts"]["orbitMutates"] == "CAMERA_VIEW_MATRIX_ONLY"
    assert payload["frames"] == [4279]
    assert "pose" not in payload
    assert "betas" not in payload
    assert payload["handShape"] == [1, 778, 3]


def test_object_model_is_precomputed_in_the_hand_root_frame() -> None:
    anatomy = np.zeros((1, 778, 3), dtype=np.float32)
    world = anatomy + np.array([[[2.0, 3.0, 4.0]]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    transform = np.eye(4, dtype=np.float32)[None]
    transform[0, :3, 3] = [7.0, 11.0, 13.0]
    data = TrustedHTMLViewerData(
        frames=np.array([4279]),
        hand_vertices_world=world,
        hand_vertices_anatomy=anatomy,
        hand_faces_closed=faces,
        hand_faces_open=faces,
        hand_joints_world=np.zeros((1, 21, 3), dtype=np.float32),
        hand_joints_anatomy=np.zeros((1, 21, 3), dtype=np.float32),
        object_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        object_faces=faces,
        object_transforms=transform,
        primary_frame=4279,
        record={"target_object": "C10001", "object_centroid": [0.0, 0.0, 0.0]},
        camera_presets=camera_presets(),
    )

    encoded = payload_for(data)["objectModelsScene"]
    models = np.frombuffer(base64.b64decode(encoded), dtype=np.float32).reshape(1, 4, 4)

    assert np.allclose(models[0, :3, 3], [5.0, 8.0, 9.0])


def test_v2_viewer_has_scene_controls_and_camera_only_orbit() -> None:
    source = (Path(__file__).parents[2] / "src/toporetarget/viz/oakink2_html_viewer.py").read_text(
        encoding="utf-8"
    )

    for control in (
        "HAND ONLY",
        "HAND + OBJECT",
        "SKELETON ONLY",
        "HAND + SKELETON + OBJECT",
    ):
        assert control in source
    assert "SOURCE VIEW" not in source
    assert "CANONICAL VIEW" not in source
    assert "gl.DEPTH_TEST" in source
    assert "OBLIQUE" in source and "FRONT" in source and "SIDE" in source
    assert "class OakInk2HTMLViewerV2Data" in source
    assert "ViewerCameraStateV1" in source
    assert "function cameraMatrices()" in source
    assert "function freeView()" not in source
    assert "const I=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])" in source
    assert "objectModelsScene" in source
    assert "RESET CAMERA" in source
    assert 'canvas.addEventListener("wheel"' in source
    assert "gl.readPixels" in source


def test_compatibility_name_points_to_v2_data_contract() -> None:
    assert TrustedHTMLViewerData is OakInk2HTMLViewerV2Data


def test_vertex_normals_are_finite_for_precomputed_frame_stack() -> None:
    vertices = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2,
        dtype=np.float64,
    )
    normals = vertex_normals(vertices, np.array([[0, 1, 2]], dtype=np.int64))
    assert np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0)
