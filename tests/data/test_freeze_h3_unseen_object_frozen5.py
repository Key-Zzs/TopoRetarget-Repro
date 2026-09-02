from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data.freeze_h3_unseen_object_frozen5 import (  # noqa: E402
    SELECTION_SEED,
    SPLIT_LABEL,
    _eligible,
    _select,
    _sha256,
    _stable_hash,
    main,
)


def _episode(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "active_hand": "right",
        "physicalization_v1_eligible": True,
        "complete": True,
        "episode_type": "SINGLE_HAND_PICK_PLACE",
        "other_hand_same_target": False,
        "handover": False,
        "overlapping_other_hand_other_object": False,
        "return_semantics": "RETURN_TO_NON_INTERACTING_IDLE",
        "approach_frame": 1,
        "contact_frame": 2,
        "pickup_frame": 3,
        "transport_frame": 4,
        "place_frame": 5,
        "release_frame": 6,
        "retreat_frame": 7,
    }
    value.update(overrides)
    return value


def test_h3d_episode_eligibility_is_fail_closed() -> None:
    assert _eligible(_episode()) == (True, [])
    passed, reasons = _eligible(_episode(other_hand_same_target=True))
    assert not passed
    assert reasons == ["SAME_OBJECT_BIMANUAL_FORBIDDEN"]
    passed, reasons = _eligible(_episode(release_frame=None))
    assert not passed
    assert "LIFECYCLE_FRAMES_REQUIRED" in reasons


def test_selection_prefers_subject_and_prefix_diversity_without_replacement() -> None:
    rows = [
        {
            "episode_id": "e0",
            "target_object": "G09_1",
            "raw_sequence": "s0",
            "subject": "subject_1",
            "object_prefix": "G09",
        },
        {
            "episode_id": "e1",
            "target_object": "G09_2",
            "raw_sequence": "s1",
            "subject": "subject_2",
            "object_prefix": "G09",
        },
        {
            "episode_id": "e2",
            "target_object": "G11_1",
            "raw_sequence": "s2",
            "subject": "subject_2",
            "object_prefix": "G11",
        },
        {
            "episode_id": "e3",
            "target_object": "G19_1",
            "raw_sequence": "s3",
            "subject": "subject_3",
            "object_prefix": "G19",
        },
    ]
    selected = _select(rows, count=3)
    assert [row["episode_id"] for row in selected] == ["e0", "e2", "e3"]
    assert len({row["target_object"] for row in selected}) == 3
    assert len({row["raw_sequence"] for row in selected}) == 3


def test_selection_contract_constants_and_hash_are_deterministic() -> None:
    assert SELECTION_SEED == 20260826
    assert SPLIT_LABEL == "UNSEEN_OBJECT_INSTANCE_HELDOUT"
    assert _stable_hash({"b": 2, "a": 1}) == _stable_hash({"a": 1, "b": 2})


def test_development_exclusions_are_unique_evidence_referenced_and_not_p6_metadata_only() -> None:
    path = REPO_ROOT / "configs/contracts/h3_development_object_exclusions_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = payload["entries"]
    object_ids = [row["object_id"] for row in entries]
    assert len(object_ids) == len(set(object_ids))
    assert {
        "G10_1",
        "G10_2",
        "G10_3",
        "G10_4",
        "G04_1",
        "G04_2",
        "G04_3",
        "G04_4",
        "G16_3",
        "G09_4",
        "G22_3",
        "G16_2",
        "G06_4",
        "G21_1",
        "G19_1",
        "G02_1",
        "G09_1",
        "G18_1",
        "G15_1",
        "G05_1",
    } == set(object_ids)
    assert {"G11_1", "G21_2", "G21_4"}.isdisjoint(object_ids)
    for row in entries:
        evidence_path = Path(row["evidence_path"])
        assert not evidence_path.is_absolute()
        assert ".." not in evidence_path.parts
        if evidence_path.parts[:2] == (".local", "reports"):
            continue
        assert (REPO_ROOT / evidence_path).is_file()


def test_full_freeze_is_deterministic_and_object_disjoint(tmp_path, monkeypatch) -> None:
    episodes = []
    for index in range(6):
        object_id = f"G{index:02d}_1"
        mesh = tmp_path / object_id / "textured_mesh.obj"
        mesh.parent.mkdir()
        mesh.write_text(
            f"v 0 0 0\nv {index + 1} 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )
        digest = _sha256(mesh)
        provenance = {
            "object_mesh": {"path": str(mesh), "sha256": digest},
            "raw_mano": {"path": str(tmp_path / "poses_m.npy"), "sha256": "1" * 64},
            "raw_object": {"path": str(tmp_path / "poses_o.npy"), "sha256": "2" * 64},
            "meta": {"path": str(tmp_path / "meta.yaml"), "sha256": "3" * 64},
            "mano_calibration": {
                "path": str(tmp_path / "subject.yaml"),
                "sha256": "4" * 64,
            },
        }
        episodes.append(
            {
                **_episode(),
                "episode_id": f"episode_{index}",
                "raw_sequence": f"subject_{index}/sequence_{index}",
                "subject": f"subject_{index}",
                "target_object": object_id,
                "start_frame": 0,
                "end_frame": 10,
                "duration_frames": 10,
                "provenance": provenance,
            }
        )
    episode_index = tmp_path / "episodes.json"
    episode_index.write_text(json.dumps(episodes), encoding="utf-8")
    exclusions = tmp_path / "exclusions.yaml"
    exclusions.write_text(
        yaml.safe_dump(
            {
                "schema_version": "H3DevelopmentObjectExclusionsV1",
                "entries": [
                    {
                        "object_id": "G00_1",
                        "exposure_class": "UNIT_TEST",
                        "evidence_scope": "unit",
                        "evidence_path": "README.md",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocol_value = {
        "schema_version": "H3PhysicalizationProtocolV1",
        "unit": True,
        "freeze": {"H3_EXECUTION_HEAD": "a" * 40},
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value), encoding="utf-8")
    protocol_hash = tmp_path / "protocol_hash.txt"
    protocol_hash.write_text(_stable_hash(protocol_value) + "\n", encoding="utf-8")
    old_p6_value = {"schema_version": "HistoricalP6Unit", "status": "FROZEN_NOT_EXECUTED"}
    old_p6_value["manifest_sha256"] = _stable_hash(old_p6_value)
    old_p6 = tmp_path / "old_p6.json"
    old_p6.write_text(json.dumps(old_p6_value), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_h3_unseen_object_frozen5.py",
            "--episode-index",
            str(episode_index),
            "--development-exclusions",
            str(exclusions),
            "--h3-protocol",
            str(protocol),
            "--h3-protocol-hash",
            str(protocol_hash),
            "--old-p6-manifest",
            str(old_p6),
            "--output-root",
            str(output),
        ],
    )
    assert main() == 0
    manifest = json.loads(
        (output / "unseen_object_frozen5_manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads((output / "selection_receipt.json").read_text(encoding="utf-8"))
    assert manifest["split_type"] == SPLIT_LABEL
    assert manifest["held_out_count"] == 5
    assert "G00_1" not in {row["object_id"] for row in manifest["clips"]}
    assert receipt["object_id_overlap_with_development"] == 0
    assert receipt["mesh_sha256_overlap_with_development"] == 0
    assert receipt["downstream_outcomes_used"] is False
    authority = json.loads((output / "episode_object_authority.json").read_text(encoding="utf-8"))
    authority_core = dict(authority)
    authority_hash = authority_core.pop("authority_sha256")
    assert _stable_hash(authority_core) == authority_hash
    assert manifest["primary_object_authority_sha256"] == authority_hash
