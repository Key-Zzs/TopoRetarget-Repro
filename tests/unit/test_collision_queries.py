import numpy as np

from toporetarget.geometry.collision_queries import probe_robot_surface
from toporetarget.geometry.robot_surface import RobotSurfaceSampleSet, RobotSurfaceSamplingProfile
from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh


def test_collision_probe_reports_pointwise_penetration_without_optimization() -> None:
    profile = RobotSurfaceSamplingProfile(
        "test",
        "1",
        1,
        "area_uniform_triangles",
        0,
        "engineering",
        "not_paper_specified",
        False,
        False,
    )
    points = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64)
    samples = RobotSurfaceSampleSet(
        "synthetic",
        "right",
        profile,
        np.array(["g", "g"]),
        np.array(["link", "link"]),
        np.array(["box", "box"]),
        np.array([0, 1]),
        points,
        np.zeros_like(points),
        points,
        np.zeros_like(points),
        points,
        np.zeros_like(points),
        [{"geometry_id": "g"}],
        {},
    )
    vertices, faces = make_synthetic_mesh("cube")
    result = probe_robot_surface(
        samples, ReferenceSignedDistanceBackend(vertices, faces), np.eye(4)
    )
    assert result["penetration_depth"][0] > 0
    assert result["penetration_depth"][1] == 0
    assert result["final_query_set"] is False
    assert result["optimization"] is False
