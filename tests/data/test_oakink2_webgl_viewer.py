from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[2] / "scripts" / "data" / "run_oakink2_o0_o4.py"
    spec = importlib.util.spec_from_file_location("run_oakink2_o0_o4", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_webgl_viewer_normals_are_unit_and_finite() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    normals = _module().vertex_normals(vertices, faces)

    assert normals.shape == vertices.shape
    assert np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_viewer_declares_depth_and_normal_rendering() -> None:
    source = (Path(__file__).parents[2] / "scripts" / "data" / "run_oakink2_o0_o4.py").read_text(
        encoding="utf-8"
    )

    assert "local_webgl_depth_normal_v1" in source
    assert "gl.DEPTH_TEST" in source
    assert "gl.CULL_FACE" in source
    assert "vNormal" in source
