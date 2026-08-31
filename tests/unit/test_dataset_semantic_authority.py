from __future__ import annotations

from toporetarget.semantic import (
    AuthorityStatus,
    CanonicalHOIRecordV1,
    DatasetSemanticAuthorityV1,
    ObjectAssetBindingV1,
    TargetObjectAuthorityV1,
    canonical_hash,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "subject": "subject_1",
        "raw_sequence": "subject_1/sequence",
        "episode_id": "episode-00",
        "active_hand": "right",
        "target_object": "A",
        "start_frame": 0,
        "end_frame": 100,
        "approach_frame": 10,
        "contact_frame": 20,
        "pickup_frame": 35,
        "transport_frame": 40,
        "place_frame": 70,
        "release_frame": 80,
        "retreat_frame": 90,
        "episode_type": "SINGLE_HAND_PICK_PLACE",
        "complete": True,
        "physicalization_v1_eligible": True,
        "source_support_metadata": {},
        "provenance": {
            "raw_mano": {"sha256": "hand"},
            "raw_object": {"sha256": "pose", "official_object_index": 0},
            "object_mesh": {"sha256": "mesh", "path": "missing.obj"},
        },
    }
    row.update(overrides)
    return row


def test_canonical_record_hash_is_deterministic() -> None:
    first = CanonicalHOIRecordV1.from_episode_row(_row(), object_ids=["A", "B"])
    second = CanonicalHOIRecordV1.from_episode_row(_row(), object_ids=["A", "B"])
    assert first.as_dict() == second.as_dict()
    assert first.canonical_record_sha256 == canonical_hash(first.as_dict(include_hash=False))
    assert first.target_object_track_id == "hocap_object_track:0"


def test_target_authority_keeps_ranking_and_clear_winner() -> None:
    result = TargetObjectAuthorityV1.rank_candidates(
        [
            {
                "object_id": "A",
                "contact_frame": 20,
                "release_frame": 80,
                "complete": True,
                "physicalization_v1_eligible": True,
            },
            {
                "object_id": "B",
                "contact_frame": None,
                "release_frame": None,
                "complete": False,
                "physicalization_v1_eligible": False,
            },
        ],
        focus=_row(),
        official_target="A",
    )
    assert result["status"] == AuthorityStatus.TARGET_OBJECT_PASS.value
    assert result["selected_object_id"] == "A"
    assert len(result["candidates"]) == 2
    assert result["top1_top2_margin"] > 0.15


def test_target_authority_quarantines_ambiguous_candidates() -> None:
    result = TargetObjectAuthorityV1.rank_candidates(
        [
            {
                "object_id": "A",
                "contact_frame": 20,
                "release_frame": 80,
                "complete": True,
                "physicalization_v1_eligible": True,
            },
            {
                "object_id": "B",
                "contact_frame": 20,
                "release_frame": 80,
                "complete": True,
                "physicalization_v1_eligible": True,
            },
        ],
        focus=_row(),
    )
    assert result["status"] == AuthorityStatus.TARGET_OBJECT_AMBIGUOUS.value


def test_binding_is_independent_from_selection() -> None:
    record = CanonicalHOIRecordV1.from_episode_row(_row())
    valid = ObjectAssetBindingV1.validate(
        record,
        {
            "episode": {"object_id": "A", "mesh_sha256": "mesh"},
            "asset": {"object_id": "A", "mesh_sha256": "mesh"},
        },
    )
    invalid = ObjectAssetBindingV1.validate(
        record,
        {
            "episode": {"object_id": "A", "mesh_sha256": "mesh"},
            "asset": {"object_id": "B", "mesh_sha256": "mesh"},
        },
    )
    assert valid["status"] == "PASS"
    assert invalid["status"] == AuthorityStatus.OBJECT_ASSET_BINDING_FAIL.value


def test_preflight_blocks_bimanual_and_accepts_complete_single_hand() -> None:
    authority = DatasetSemanticAuthorityV1()
    row = _row()
    record, binding, result = authority.preflight(
        row,
        object_ids=["A"],
        candidates=[
            {
                "object_id": "A",
                "contact_frame": 20,
                "release_frame": 80,
                "complete": True,
                "physicalization_v1_eligible": True,
            }
        ],
    )
    assert binding["status"] == "PASS"
    assert result["status"] == AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value
    assert record.canonical_record_sha256
    bimanual = dict(row, active_hand="both", episode_type="BIMANUAL_SAME_OBJECT")
    _, _, blocked = authority.preflight(
        bimanual,
        object_ids=["A"],
        candidates=[
            {
                "object_id": "A",
                "contact_frame": 20,
                "release_frame": 80,
                "complete": True,
                "physicalization_v1_eligible": True,
            }
        ],
    )
    assert blocked["status"] == AuthorityStatus.SEMANTIC_PREFLIGHT_FAIL.value
