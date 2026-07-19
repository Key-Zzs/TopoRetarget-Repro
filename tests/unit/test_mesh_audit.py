import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh


def _cube() -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def test_closed_cube_is_reliably_watertight() -> None:
    vertices, faces = _cube()
    report = audit_mesh(vertices, faces)
    assert report.watertight
    assert report.boundary_edge_count == 0
    assert report.non_manifold_edge_count == 0
    assert report.sign_reliability == "reliable_watertight"
    assert report.signed_volume is not None


def test_open_and_degenerate_meshes_are_not_repaired() -> None:
    vertices, faces = _cube()
    open_report = audit_mesh(vertices, faces[:-2])
    assert open_report.sign_reliability == "open_surface"
    degenerate_faces = np.vstack((faces, [0, 0, 0]))
    degenerate_report = audit_mesh(vertices, degenerate_faces)
    assert degenerate_report.near_zero_area_faces >= 1
    assert degenerate_report.face_count == len(degenerate_faces)


def test_duplicate_and_unreferenced_vertices_are_reported() -> None:
    vertices, faces = _cube()
    vertices = np.vstack((vertices, vertices[0], [10.0, 10.0, 10.0]))
    report = audit_mesh(vertices, faces)
    assert report.duplicate_vertices == 1
    assert report.unreferenced_vertices >= 2
