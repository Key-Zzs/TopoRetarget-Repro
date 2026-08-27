from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data import freeze_h3_hardening_regression_manifest as freeze  # noqa: E402


def _row(episode_id: str, index: int) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "raw_sequence": f"subject_{index}/sequence_{index}",
        "subject": f"subject_{index}",
        "active_hand": "right",
        "target_object": f"G{index:02d}_1",
        "start_frame": 0,
        "end_frame": 10,
        "duration_frames": 10,
        "physicalization_v1_eligible": True,
        "complete": True,
        "episode_type": "SINGLE_HAND_PICK_PLACE",
        "other_hand_same_target": False,
        "provenance": {
            "raw_mano": {"path": f"/raw/{index}/poses_m.npy", "sha256": "1" * 64},
            "raw_object": {"path": f"/raw/{index}/poses_o.npy", "sha256": "2" * 64},
            "meta": {"path": f"/raw/{index}/meta.yaml", "sha256": "3" * 64},
            "object_mesh": {"path": f"/mesh/{index}.obj", "sha256": "4" * 64},
            "mano_calibration": {"path": f"/mano/{index}.yaml", "sha256": "5" * 64},
        },
    }


def test_h3c_manifest_is_regression_not_heldout(tmp_path, monkeypatch) -> None:
    rows = [_row(episode, index) for index, episode in enumerate(freeze.EXPECTED_EPISODES)]
    episode_index = tmp_path / "episodes.json"
    episode_index.write_text(json.dumps(rows), encoding="utf-8")
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps(
            {
                "schema_version": "PipelineHardeningSetManifestV1",
                "episodes": [{"episode_id": episode} for episode in freeze.EXPECTED_EPISODES],
            }
        ),
        encoding="utf-8",
    )
    head = "a" * 40
    protocol_value = {
        "schema_version": "H3PhysicalizationProtocolV1",
        "freeze": {"H3_EXECUTION_HEAD": head},
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value), encoding="utf-8")
    protocol_hash = tmp_path / "protocol_hash.txt"
    protocol_hash.write_text(freeze._stable_hash(protocol_value) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(freeze, "_git_head", lambda: head)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_h3_hardening_regression_manifest.py",
            "--episode-index",
            str(episode_index),
            "--old-hardening-manifest",
            str(old),
            "--h3-protocol",
            str(protocol),
            "--h3-protocol-hash",
            str(protocol_hash),
            "--output-root",
            str(output),
        ],
    )
    assert freeze.main() == 0
    manifest = json.loads(
        (output / "hardening_regression_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_role"] == "PIPELINE_HARDENING_SET_V1"
    assert manifest["held_out"] is False
    assert manifest["held_out_rate_denominator"] is False
    assert manifest["fresh_raw_to_final_execution_required"] is True
    assert [row["episode_id"] for row in manifest["clips"]] == list(freeze.EXPECTED_EPISODES)
    core = dict(manifest)
    embedded = core.pop("manifest_sha256")
    assert freeze._stable_hash(core) == embedded
    authority = json.loads((output / "episode_object_authority.json").read_text(encoding="utf-8"))
    authority_core = dict(authority)
    authority_hash = authority_core.pop("authority_sha256")
    assert freeze._stable_hash(authority_core) == authority_hash
    assert manifest["primary_object_authority_sha256"] == authority_hash
