from toporetarget.data.synthetic import make_synthetic_sequence
from toporetarget.geometry.object_geometry import scene_samples_for_frames
from toporetarget.geometry.signed_distance.validation import validate_analytic_shape
from toporetarget.geometry.surface_sampling import load_surface_profile, sample_mesh_surface


def test_synthetic_stage6_dataflow_and_analytic_validation() -> None:
    sequence = make_synthetic_sequence(num_frames=3)
    track = sequence.rigid_objects[0]
    samples = sample_mesh_surface(
        track.mesh.vertices_local,
        track.mesh.faces,
        load_surface_profile("paper_strict_area_uniform"),
    )
    points, normals = scene_samples_for_frames(track, samples, [0, 1, 2])
    assert points.shape == (3, 50, 3)
    assert normals.shape == (3, 50, 3)
    assert validate_analytic_shape("sphere")["status"] == "pass"
    assert validate_analytic_shape("cube")["status"] == "pass"
