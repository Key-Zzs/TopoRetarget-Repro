import os
from pathlib import Path

import pytest

from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.object_geometry import object_track_from_canonical, sample_object_track
from toporetarget.geometry.surface_sampling import load_surface_profile

pytestmark = pytest.mark.skipif(
    os.environ.get("TOPORETARGET_RUN_REAL_GRAB_TESTS") != "1",
    reason="requires bounded local GRAB canonical cache",
)


def test_bounded_grab_primary_object_stage6() -> None:
    canonical = Path(
        os.environ.get(
            "TOPORETARGET_STAGE6_CANONICAL",
            ".local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr",
        )
    )
    track = object_track_from_canonical(canonical, "cubemedium")
    audit = audit_mesh(track.mesh.vertices_local, track.mesh.faces)
    samples = sample_object_track(track, load_surface_profile("paper_strict_area_uniform"))
    assert samples.count == 50
    assert audit.mesh_hash == samples.mesh_array_hash
    assert audit.sign_reliability in {
        "reliable_watertight",
        "open_surface",
        "non_manifold",
        "watertight_inconsistent_winding",
    }
