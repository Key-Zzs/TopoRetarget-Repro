from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh


def _module():
    path = Path(__file__).parents[2] / "scripts/data/run_oakink2_o1r2.py"
    spec = importlib.util.spec_from_file_location("run_oakink2_o1r2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_rank_has_midrank_and_outlier_controls() -> None:
    module = _module()

    assert module.percentile_rank(np.array([0.0, 1.0, 2.0]), 1.0) == 50.0
    assert module.percentile_rank(np.arange(100.0), 99.0) > 99.0


def test_ablation_view_transform_is_finite_and_root_translation_is_explicit() -> None:
    module = _module()
    point = np.array([[0.0, 0.0, 0.0]])

    for view in module.VIEW_ROTATIONS:
        transform = module.canonical_view_transform(view)
        transformed = module.transform_points(transform, point)
        assert np.isfinite(transform).all()
        assert np.allclose(transformed[0, 2], 0.45)


def test_skeleton_has_one_root_five_distinct_acyclic_chains() -> None:
    module = _module()
    joints = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
    result = module.skeleton_diagnostics(joints)

    assert result["joint_count"] == 21
    assert result["root_count"] == 1
    assert result["five_finger_chains_valid"] is True
    assert result["duplicate_finger_chains"] is False
    assert result["parent_cycles"] is False


def test_closed_mesh_metrics_and_export_round_trip(tmp_path: Path) -> None:
    module = _module()
    mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.03])
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[5] = [0.04, 0.05, 0]
    joints[9] = [0.01, 0.06, 0]
    joints[13] = [-0.01, 0.06, 0]
    joints[17] = [-0.04, 0.05, 0]
    for chain in module.FINGER_CHAINS.values():
        for offset, index in enumerate(chain[1:], 1):
            joints[index, 1] += 0.02 * offset

    metrics = module.mesh_metrics(mesh.vertices, mesh.faces, joints)
    receipt = module.export_round_trip(tmp_path, mesh.vertices, mesh.faces)

    assert metrics["vertices_finite"] is True
    assert metrics["face_indices_valid"] is True
    assert metrics["watertight"] is True
    assert metrics["palm_thickness"]["palm_thickness_m"] > 0
    assert receipt["obj"]["vertex_max_error_m"] < 1e-6
    assert receipt["ply"]["vertex_max_error_m"] < 1e-6
    assert receipt["obj"]["topology_parity"] is True
    assert receipt["ply"]["topology_parity"] is True


def test_renderer_independence_is_structural() -> None:
    source = (Path(__file__).parents[2] / "scripts/data/run_oakink2_o1r2.py").read_text(
        encoding="utf-8"
    )
    independent = source.split("def render_independent", 1)[1].split("def render_official", 1)[0]

    assert "PyMultiObjRenderer" not in independent
    assert "pyrender.OffscreenRenderer" in independent
    assert "custom HTML" in source
    assert '"O3_RERUN": "NO"' in source
    assert '"MANIFEST_V3_CREATED": "NO"' in source
