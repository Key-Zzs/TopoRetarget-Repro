from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from toporetarget.rl.independent_physical_refinement import (
    BatchContractError,
    HOCapCandidate,
    append_stage_receipt,
    assert_frozen_manifest,
    assert_independent_lineages,
    freeze_method_contract,
    freeze_selection,
    scan_hocap_candidates,
    select_held_out_candidates,
    validate_authority_manifest,
)


def _candidate(index: int, *, development: bool = False) -> HOCapCandidate:
    suffix = "170105" if development else f"{220000 + index}"
    return HOCapCandidate(
        clip_id=f"hocap_{suffix}",
        sequence=f"subject_{index}/20231025_{suffix}",
        subject=f"subject_{index}",
        object_ids=(f"G{index:02d}_1",),
        raw_path=f"/raw/{index}",
        raw_frames=60,
        raw_fps=30.0,
        raw_hashes={"meta.yaml": str(index)},
        eligible=True,
        reasons=(),
    )


def test_selection_is_deterministic_and_excludes_development() -> None:
    rows = [_candidate(1, development=True), *[_candidate(item) for item in range(2, 9)]]

    first = select_held_out_candidates(rows)
    second = select_held_out_candidates(reversed(rows))

    assert [item.clip_id for item in first] == [item.clip_id for item in second]
    assert len(first) == 5
    assert "hocap_170105" not in {item.clip_id for item in first}


def test_multi_object_selection_requires_explicit_primary_authority(tmp_path: Path) -> None:
    rows = [
        HOCapCandidate(
            **{
                **_candidate(index).__dict__,
                "object_ids": (f"G{index:02d}_1", f"G{index:02d}_2"),
            }
        )
        for index in range(1, 6)
    ]

    with pytest.raises(BatchContractError, match="PRIMARY_OBJECT_AUTHORITY_REQUIRED"):
        freeze_selection(candidates=rows, root=tmp_path)


def test_freeze_manifest_hash_rejects_outcome_mutation(tmp_path: Path) -> None:
    manifest = freeze_selection(
        candidates=[_candidate(item) for item in range(1, 7)], root=tmp_path
    )

    assert_frozen_manifest(manifest)
    manifest["clips"][0]["exclusion_audit"]["outcome_observed"] = True
    with pytest.raises(BatchContractError, match="HELD_OUT_MANIFEST_HASH_DRIFT"):
        assert_frozen_manifest(manifest)


def test_method_contract_uses_dynamic_rsi_domain(tmp_path: Path) -> None:
    contract, digest = freeze_method_contract(tmp_path)

    assert digest
    assert contract["ppo"]["max_updates"] == 15
    assert contract["rsi"]["training"] == "uniform_runtime_reference_valid_index_domain"


def test_authority_validation_is_per_clip_and_fail_closed() -> None:
    result = validate_authority_manifest(
        {"authorities": {"retarget": {"supported_clips": ["hocap_a"]}}},
        ["hocap_a", "hocap_b"],
    )

    assert result["valid"] is False
    assert result["unsupported"]["retarget"] == ["hocap_b"]
    assert "source_policy" in result["unsupported"]


def test_lineages_cannot_share_storage_or_rng() -> None:
    alpha = {
        "lineage": {
            "actor_root": "a",
            "critic_root": "c1",
            "optimizer_root": "o1",
            "normalizer_root": "n1",
            "rng_seed": 1,
        }
    }
    beta = {
        "lineage": {
            "actor_root": "b",
            "critic_root": "c2",
            "optimizer_root": "o2",
            "normalizer_root": "n2",
            "rng_seed": 2,
        }
    }
    assert_independent_lineages([alpha, beta])
    beta["lineage"]["optimizer_root"] = "o1"
    with pytest.raises(BatchContractError, match="optimizer_root"):
        assert_independent_lineages([alpha, beta])


def test_terminal_state_cannot_resume_into_ppo() -> None:
    state = {"state": "ACCEPTED_FROZEN", "stages": []}
    with pytest.raises(BatchContractError, match="TERMINAL_CLIP_CANNOT_ADVANCE"):
        append_stage_receipt(
            state,
            stage="ppo",
            status="PPO_RUNNING",
            started_utc="2026-08-22T00:00:00Z",
            ended_utc="2026-08-22T00:00:01Z",
            wall_seconds=1.0,
        )


def test_raw_scan_uses_only_required_static_modalities(tmp_path: Path) -> None:
    raw = tmp_path / "HOCap"
    sequence = raw / "data/subject_2/20231022_203100"
    sequence.mkdir(parents=True)
    mesh = raw / "data/models/G10_1"
    mesh.mkdir(parents=True)
    (mesh / "textured_mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (sequence / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "subject_id": "subject_2",
                "object_ids": ["G10_1"],
                "mano_sides": ["right"],
                "num_frames": 3,
            }
        ),
        encoding="utf-8",
    )
    np.save(sequence / "poses_m.npy", np.zeros((1, 3, 51), dtype=np.float32))
    np.save(sequence / "poses_o.npy", np.zeros((1, 3, 7), dtype=np.float32))

    rows = scan_hocap_candidates(raw)

    assert len(rows) == 1
    assert rows[0].eligible is True
    assert rows[0].clip_id == "hocap_203100"
