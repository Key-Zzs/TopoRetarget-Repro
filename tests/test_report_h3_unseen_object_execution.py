from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.report_h3_unseen_object_execution import aggregate


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(path: Path) -> dict[str, object]:
    clips = [
        {"episode_id": f"episode_{index}", "primary_object_id": f"G{index:02d}_1"}
        for index in range(5)
    ]
    value: dict[str, object] = {
        "schema_version": "H3UnseenObjectFrozen5ManifestV1",
        "HELD_OUT_SET_FROZEN": "YES",
        "held_out_count": 5,
        "h3_protocol_hash": "b" * 64,
        "clips": clips,
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write(path, value)
    return value


def test_readiness_no_emits_blocked_receipt_without_outcomes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    readiness = tmp_path / "readiness.json"
    _write(readiness, {"H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION": "NO"})
    result = aggregate(manifest_path, readiness, tmp_path / "report")
    assert result["status"] == "EXECUTION_BLOCKED"
    assert result["claim"] == "H3D_EXECUTION_BLOCKED_BY_PIPELINE_READINESS"
    assert result["downstream_outcomes_observed"] is False


def test_ready_execution_reports_scientific_rates(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path)
    readiness = tmp_path / "readiness.json"
    _write(readiness, {"H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION": "YES"})
    report = tmp_path / "report"
    for index, clip in enumerate(manifest["clips"]):
        status = (
            "ACCEPTED_FROZEN_FULL_CYCLE"
            if index == 0
            else ("ACCEPTED_AFTER_REFINEMENT_FULL_CYCLE" if index == 1 else "PPO_BUDGET_EXHAUSTED")
        )
        episode = clip["episode_id"]
        row = {
            "schema_version": "H3UnseenObjectEpisodeResultV1",
            "episode": episode,
            "object_id": clip["primary_object_id"],
            "held_out": True,
            "method_contract_hash": "b" * 64,
            "per_episode_tuning": False,
            "retarget": "PASS",
            "source_route": "ZERO_RESIDUAL",
            "source_executable": "PASS",
            "source_fidelity": "DEGRADED",
            "support": "PASS",
            "frozen_pf": "10/10" if index == 0 else "0/10",
            "ppo_updates": 0 if index == 0 else 15,
            "pf_pick": "PASS",
            "pf_transport": "PASS" if index < 2 else "FAIL",
            "pf_place": "PASS" if index < 2 else "NOT_REACHED",
            "pf_release": "PASS" if index < 2 else "NOT_REACHED",
            "pf_retreat": "PASS" if index < 2 else "NOT_REACHED",
            "pf_full": "PASS" if index < 2 else "FAIL",
            "df_pose": "PASS" if index < 2 else "FAIL",
            "df_linear": "PASS",
            "df_angular": "PASS",
            "status": status,
            "accepted_frozen": index == 0,
            "initial_physical_failure": index != 0,
            "recovered_by_ppo": index == 1,
            "accepted_final": index < 2,
            "lineage": {
                "actor_root": f"actor_{index}",
                "critic_root": f"critic_{index}",
                "optimizer_root": f"optimizer_{index}",
                "normalizer_root": f"normalizer_{index}",
                "rng_seed": index,
            },
            "replay_commands": {
                "full": f"replay full {index}",
                "pick_lift": f"replay pick {index}",
                "place_release": f"replay place {index}",
            },
            "timing_seconds": {"total": 1.0},
            "failure_taxonomy": [],
        }
        _write(report / "per_episode" / episode / "final_status.json", row)
    decision = aggregate(manifest_path, readiness, report)
    metrics = decision["metrics"]
    assert metrics["SR_frozen"] == 0.2
    assert metrics["RR"] == 0.25
    assert metrics["SR_final"] == 0.4
    assert metrics["UTR"] == 0.0
    assert metrics["claim"] == "UNSEEN_OBJECT_GENERALIZATION_MIXED"
    assert metrics["shared_policy_zero_shot_claim"] is False
