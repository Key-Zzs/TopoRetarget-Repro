from types import SimpleNamespace

import numpy as np
import pytest

from toporetarget.adapters.datasets.hocap_primary_object import (
    HOCapPrimaryObjectError,
    primary_object_from_authority,
    resolve_hocap_primary_object,
)
from toporetarget.rl.independent_physical_refinement import stable_hash


def _sequence(*, runner_distance: float = 0.11) -> SimpleNamespace:
    frame_count = 10
    hand = np.zeros((frame_count, 21, 3), dtype=np.float64)
    hand[..., 2] = 0.01
    vertices = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    def obj(object_id: str, plane_z: float) -> SimpleNamespace:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
        poses[:, 2, 3] = plane_z
        return SimpleNamespace(
            object_id=object_id,
            mesh=SimpleNamespace(
                vertices_local=vertices,
                faces=faces,
                mesh_hash=f"hash-{object_id}",
            ),
            pose_scene=SimpleNamespace(pose_scene=poses),
        )

    return SimpleNamespace(
        metadata=SimpleNamespace(dataset_name="hocap", sequence_id="subject_1/demo"),
        hands=[
            SimpleNamespace(
                keypoint_tracks={
                    "mediapipe21": SimpleNamespace(positions_scene=hand),
                }
            )
        ],
        rigid_objects=[obj("G01_2", 0.0), obj("G01_1", 0.01 - runner_distance)],
    )


def test_raw_surface_resolver_selects_unique_candidate() -> None:
    result = resolve_hocap_primary_object(_sequence())

    assert result["status"] == "RESOLVED"
    assert result["primary_object_id"] == "G01_2"
    assert result["outcome_inputs_used"] is False
    assert result["winner_runner_up_margin_m"] > 0.01


def test_raw_surface_resolver_refuses_small_margin() -> None:
    result = resolve_hocap_primary_object(_sequence(runner_distance=0.015))

    assert result["status"] == "UNRESOLVED"
    assert result["primary_object_id"] is None
    assert "PRIMARY_OBJECT_CANDIDATE_MARGIN_TOO_SMALL" in result["failure_reasons"]


def test_authority_lookup_detects_object_set_drift() -> None:
    core = {
        "schema_version": "HOCapPrimaryObjectAuthorityV1",
        "mappings": [
            {
                "status": "RESOLVED",
                "sequence": "subject_1/demo",
                "primary_object_id": "G01_2",
                "available_object_ids": ["G01_1", "G01_2"],
            }
        ],
    }
    authority = {**core, "authority_sha256": stable_hash(core)}

    assert (
        primary_object_from_authority(
            authority,
            sequence="subject_1/demo",
            available_object_ids=["G01_1", "G01_2"],
        )
        == "G01_2"
    )
    with pytest.raises(HOCapPrimaryObjectError, match="OBJECT_SET_DRIFT"):
        primary_object_from_authority(
            authority,
            sequence="subject_1/demo",
            available_object_ids=["G01_2", "G01_1"],
        )
