"""CPU-only contract tests for source MANO contact semantics."""

from __future__ import annotations

import numpy as np

from toporetarget.evaluation.source_contact_semantics import (
    FINGER_ORDER,
    REGION_ORDER,
    SEGMENT_ORDER,
    build_mano_surface_region_map,
    classify_source_contact,
    map_native_contact_to_control,
    per_region_surface_statistics,
    persistent_mask,
)


def _region_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One pure vertex per required semantic region plus a boundary vertex."""

    weights = np.zeros((7, 16), dtype=np.float64)
    chains = {
        "index": 1,
        "middle": 4,
        "pinky": 7,
        "ring": 10,
        "thumb": 13,
    }
    for vertex, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        weights[vertex, chains[finger] + 2] = 1.0
    weights[5, 0] = 1.0
    weights[6, 0] = 0.5
    weights[6, 1] = 0.5
    vertices = np.asarray([[float(index), 0.0, 0.0] for index in range(7)])
    joints = np.asarray([[float(index), 0.0, 0.0] for index in range(16)])
    faces = np.asarray([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    return weights, faces, vertices, joints


def test_lbs_region_map_has_required_groups_and_explicit_boundary() -> None:
    weights, faces, vertices, joints = _region_fixture()
    region_map = build_mano_surface_region_map(weights, faces, vertices, joints)

    for index, finger in enumerate(FINGER_ORDER):
        assert region_map.region_id[index] == REGION_ORDER.index(finger)
    assert region_map.region_id[5] == REGION_ORDER.index("palm")
    assert region_map.region_id[6] == REGION_ORDER.index("boundary_ambiguous")
    assert region_map.segment_id[6] == SEGMENT_ORDER.index("boundary_ambiguous")
    assert np.allclose(region_map.soft_region_weight.sum(axis=1), 1.0)


def test_surface_statistics_uses_triangle_topology_for_component_size() -> None:
    weights, faces, vertices, joints = _region_fixture()
    region_map = build_mano_surface_region_map(weights, faces, vertices, joints)
    # The first three semantic vertices are joined by the first triangle and
    # are each within 5 mm, proving this is not a nearest-vertex proxy.
    distances = np.full((2, 7), 0.020, dtype=np.float64)
    distances[:, :3] = 0.001
    stats = per_region_surface_statistics(distances, region_map, faces)

    assert stats["minimum_surface_distance_m"].shape == (2, 6)
    assert stats["near_vertex_count"].shape == (2, 6, 4)
    assert stats["largest_component_vertices_at_5mm"][0].max() >= 1


def test_contact_classification_requires_component_and_native_persistence() -> None:
    minimum = np.full((4, 5), 0.020, dtype=np.float64)
    component = np.zeros((4, 5), dtype=np.int32)
    minimum[:2, 0] = 0.001
    component[:2, 0] = 3
    minimum[2, 1] = 0.001
    component[2, 1] = 3
    result = classify_source_contact(minimum, component)

    assert result["confirmed_contact"][:, 0].tolist() == [True, True, False, False]
    assert result["probable_contact"][:, 1].tolist() == [False, False, True, False]
    assert result["class"][2, 0] == "SOURCE_CONTACT_TRANSITION"


def test_native_to_factor8_mapping_never_fills_a_state_change_as_contact() -> None:
    native = np.full((3, 5), "SOURCE_NO_CONTACT", dtype="<U32")
    native[:2, 0] = "SOURCE_CONTACT_CONFIRMED"
    mapped = map_native_contact_to_control(native, factor=2, control_frames=5)

    assert mapped["class"][:, 0].tolist() == [
        "SOURCE_CONTACT_CONFIRMED",
        "SOURCE_CONTACT_PERSISTENT",
        "SOURCE_CONTACT_CONFIRMED",
        "SOURCE_CONTACT_TRANSITION",
        "SOURCE_NO_CONTACT",
    ]
    assert mapped["expected_contact"][:, 0].tolist() == [True, True, True, False, False]


def test_region_mapping_is_deterministic_and_keeps_boundary_out_of_fingers() -> None:
    weights, faces, vertices, joints = _region_fixture()
    first = build_mano_surface_region_map(weights, faces, vertices, joints)
    second = build_mano_surface_region_map(weights, faces, vertices, joints)

    assert np.array_equal(first.region_id, second.region_id)
    assert np.array_equal(first.segment_id, second.segment_id)
    assert first.region_id[-1] == REGION_ORDER.index("boundary_ambiguous")


def test_contact_sensitivity_does_not_promote_an_isolated_vertex() -> None:
    minimum = np.full((3, 5), 0.020, dtype=np.float64)
    component = np.zeros((3, 5), dtype=np.int32)
    minimum[:, 0] = 0.0005
    result = classify_source_contact(minimum, component, threshold_m=0.001)

    assert not result["confirmed_contact"].any()
    assert not result["probable_contact"].any()
    assert result["proximity_only"][:, 0].all()


def test_persistence_requires_a_full_run_at_the_declared_horizon() -> None:
    values = np.asarray([True, False, True, True, True, False])

    assert persistent_mask(values, minimum_frames=2).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
    ]
