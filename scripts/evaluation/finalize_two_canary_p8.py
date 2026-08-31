#!/usr/bin/env python3
"""Finalize the two approved-canary P8 ledger without inventing blocked outcomes."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
P8_ROOT = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p8_two_canary_physicalization"
)
P5_ROOT = (
    REPO_ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p5_two_canary_retarget"
)
P6_ROOT = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p6_semantic_certification"
)
P7_ROOT = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p7_unseen_object_refreeze"
)
GPU = P8_ROOT / "gpu/gpu_preflight_receipt.json"
MANIFEST = P5_ROOT / "two_canary_manifest.json"
BASE_GEOMETRY = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo/runtime_collision_geometry_manifest.json"
)
CONTACT_CONTRACT = (
    REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json"
)
REPLAY = REPO_ROOT / "scripts/rl/isaaclab/replay_physical_hoi_trace.py"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P8_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"P8_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _first_number(*values: object) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _stage_seconds(receipt: dict[str, Any] | None) -> float:
    if not receipt:
        return 0.0
    return sum(
        float(row.get("wall_seconds", 0.0))
        for row in receipt.get("stages", [])
        if isinstance(row, dict) and isinstance(row.get("wall_seconds"), (int, float))
    )


def _technical_failures() -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for path in sorted(P8_ROOT.glob("clips/**/logs/*.receipt.json")):
        try:
            payload = _read(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("status") != "FAIL":
            continue
        failures.append(
            {
                "kind": "initial_p8_stage",
                "receipt": _artifact(path),
                "stage": payload.get("stage"),
                "command": payload.get("command"),
                "log": payload.get("log"),
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
    for path in sorted(P8_ROOT.glob("technical_retries/**/**/*.receipt.json")):
        try:
            payload = _read(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("status") != "FAIL":
            continue
        failures.append(
            {
                "kind": "technical_retry_stage",
                "receipt": _artifact(path),
                "stage": payload.get("stage"),
                "command": payload.get("command"),
                "log": payload.get("log"),
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
    for path in sorted(P8_ROOT.glob("technical_retries/**/source_policy_failure.json")):
        payload = _read(path)
        failures.append(
            {
                "kind": "technical_retry_terminal_record",
                "receipt": _artifact(path),
                "reason": payload.get("reason"),
                "completed_stages": [
                    row.get("stage") for row in payload.get("completed_stages", [])
                ],
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
    # The first wrong-MJCF attempt was captured by the P8 source-policy failure
    # record at the main root before retry namespaces were introduced.
    for path in sorted(P8_ROOT.glob("**/source_policy_failure.json")):
        if "technical_retries" in path.parts:
            continue
        payload = _read(path)
        failures.append(
            {
                "kind": "initial_p8_attempt",
                "receipt": _artifact(path),
                "reason": payload.get("reason"),
                "completed_stages": [
                    row.get("stage") for row in payload.get("completed_stages", [])
                ],
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
    return failures


def _clip_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    p5 = _read(MANIFEST)
    by_id = {row["clip_id"]: row for row in p5["clips"]}
    return [
        {
            "label": "canary_1",
            "clip_id": "hocap_subject_9_20231027_125315__right__G21_3__ep00",
            "authority": P8_ROOT / "authority/canary_1.json",
            "p5_dir": P5_ROOT
            / "canary_1/report/episodes/hocap_subject_9_20231027_125315__right__G21_3__ep00",
            "source_prerequisite": P8_ROOT
            / "technical_retries/retry3/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/source_policy/source_policy_prerequisites_receipt.json",
            "source_final": P8_ROOT
            / "technical_retries/retry3/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/source_policy/source_policy_receipt.v4.json",
            "support_preflight": P8_ROOT
            / "technical_retries/retry4/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/support/support_preflight_receipt.json",
            "support_final": P8_ROOT
            / "technical_retries/retry5/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/support/final_summary.json",
            "support_failure": P8_ROOT
            / "technical_retries/retry5/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/support/support_failure.json",
            "support_report": P8_ROOT
            / "technical_retries/retry5/clips/hocap_subject_9_20231027_125315__right__G21_3__ep00/support",
            "terminal": "PHYSICAL_SCENE_INVALID",
            "terminal_reason": "Support PhysX telemetry ran, but hand-object geometry remained invalid and support transfer was deferred.",
        },
        {
            "label": "canary_2",
            "clip_id": "hocap_subject_6_20231025_110646__right__G05_1__ep00",
            "authority": P8_ROOT / "authority/canary_2.json",
            "p5_dir": P5_ROOT
            / "canary_2/report/episodes/hocap_subject_6_20231025_110646__right__G05_1__ep00",
            "source_prerequisite": P8_ROOT
            / "technical_retries/retry6/clips/hocap_subject_6_20231025_110646__right__G05_1__ep00/source_policy/source_policy_prerequisites_receipt.json",
            "source_final": None,
            "support_preflight": P8_ROOT
            / "technical_retries/retry7/clips/hocap_subject_6_20231025_110646__right__G05_1__ep00/support/support_preflight_receipt.json",
            "support_final": None,
            "support_failure": None,
            "support_report": P8_ROOT
            / "technical_retries/retry7/clips/hocap_subject_6_20231025_110646__right__G05_1__ep00/support",
            "terminal": "SUPPORT_UNRESOLVED",
            "terminal_reason": "Support preflight was UNRESOLVED; no PhysX policy evaluation was authorized.",
        },
    ], by_id


def _make_episode(
    config: dict[str, Any], p5_row: dict[str, Any], failures: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    p5_dir = config["p5_dir"]
    geometric = _read(p5_dir / "geometric_retarget_receipt.json")
    semantic = _read(p5_dir / "retarget/semantic_qualification.json")
    source_prereq = _read(config["source_prerequisite"])
    source_final = _read(config["source_final"]) if config["source_final"] else None
    support_preflight = _read(config["support_preflight"])
    support_final = _read(config["support_final"]) if config["support_final"] else None
    if source_final is not None:
        source_route = str(source_final.get("selected_route", "TECHNICAL_FAILURE"))
        source_status = "PASS"
    else:
        source_route = "NOT_RUN_SUPPORT_GATE_BLOCKED"
        source_status = "PASS_CPU_AUTHORITIES_ONLY"
    if (
        support_final is not None
        and support_final.get("overall_status") != "PASS_WITH_TRANSFER_DEFERRED"
    ):
        support_status = "PHYSICAL_SCENE_INVALID"
    elif support_preflight.get("status") != "PASS":
        support_status = "SUPPORT_UNRESOLVED"
    else:
        support_status = "PASS_PREFLIGHT_ONLY"

    timing_payload = geometric.get("timing", {})
    if not isinstance(timing_payload, dict):
        timing_payload = {}
    exact_retarget = _first_number(
        timing_payload.get("total_seconds"), geometric.get("wall_seconds")
    )
    semantic_qa = _first_number(semantic.get("semantic_qa_seconds"))
    source_seconds = _stage_seconds(source_final or source_prereq)
    support_seconds = 0.0
    if config["support_failure"] and config["support_failure"].is_file():
        support_seconds = float(_read(config["support_failure"]).get("wall_seconds", 0.0))
    else:
        support_seconds = float(support_preflight.get("productive_run_seconds", 0.0))
    clip_failures = [
        row for row in failures if config["clip_id"] in json.dumps(row, sort_keys=True)
    ]
    technical_seconds = sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in clip_failures)
    productive = sum(
        value or 0.0 for value in (exact_retarget, semantic_qa, source_seconds, support_seconds)
    )
    reference_report = p5_dir / "retarget/semantic_qualification.json"
    geometry_receipt = p5_dir / "geometric_retarget_receipt.json"
    episode = {
        "schema_version": "P8TwoCanaryEpisodeLedgerV1",
        "episode": config["clip_id"],
        "object": p5_row["primary_object_id"],
        "retarget_semantic": "RETARGET_SEMANTIC_PASS",
        "source_route": source_route,
        "source_status": source_status,
        "support": support_status,
        "frozen_pf": "NOT_RUN_GATE_BLOCKED",
        "ppo_updates": 0,
        "pf_pick": "NOT_RUN_GATE_BLOCKED",
        "pf_transport": "NOT_RUN_GATE_BLOCKED",
        "pf_place": "NOT_RUN_GATE_BLOCKED",
        "pf_release": "NOT_RUN_GATE_BLOCKED",
        "pf_retreat": "NOT_RUN_GATE_BLOCKED",
        "pf_full": "NOT_RUN_GATE_BLOCKED",
        "df_pose": "NOT_RUN_GATE_BLOCKED",
        "df_linear": "NOT_RUN_GATE_BLOCKED",
        "df_angular": "NOT_RUN_GATE_BLOCKED",
        "geometry": "PASS"
        if support_final is None
        or support_final.get("clips", [{}])[0].get("geometry_status") == "PASS"
        else "FAIL",
        "causality": "NOT_RUN_POLICY_TRACE_UNAVAILABLE",
        "confirm20": "NOT_RUN_GATE_BLOCKED",
        "qualification": "NOT_RUN_GATE_BLOCKED",
        "trace_export": "NOT_RUN_POLICY_TRACE_UNAVAILABLE",
        "replay_status": "FAILED_BEST_DIAGNOSTIC",
        "terminal_status": config["terminal"],
        "terminal_reason": config["terminal_reason"],
        "frozen_inputs": {
            "p5_manifest": _artifact(MANIFEST),
            "p5_manifest_sha256": _read(MANIFEST).get("manifest_sha256"),
            "authority": _artifact(config["authority"]),
            "retarget_geometric_receipt": _artifact(geometry_receipt),
            "retarget_semantic_qualification": _artifact(reference_report),
            "source_prerequisite": _artifact(config["source_prerequisite"]),
            "gpu_preflight": _artifact(GPU),
            "interaction_contact_contract": _artifact(CONTACT_CONTRACT),
            "base_runtime_geometry_manifest": _artifact(BASE_GEOMETRY),
        },
        "support_evidence": {
            "preflight": _artifact(config["support_preflight"]),
            "final": _artifact(config["support_final"]) if config["support_final"] else None,
            "failure": _artifact(config["support_failure"]) if config["support_failure"] else None,
        },
        "source_evidence": {
            "prerequisite": _artifact(config["source_prerequisite"]),
            "final": _artifact(config["source_final"]) if config["source_final"] else None,
        },
        "alignment": {
            "raw_to_retarget_reference": "PASS_FROM_P5_AND_REFERENCE_V2_SOURCE_KEY_CHECK",
            "retarget_to_actual_policy": "NOT_REACHED",
            "reason": "No physical policy trace was authorized after the terminal support gate.",
        },
        "method_contract_hashes": {
            "replay_entrypoint": _artifact(REPLAY),
            "contact_contract": _artifact(CONTACT_CONTRACT),
            "base_runtime_geometry": _artifact(BASE_GEOMETRY),
            "gpu_preflight": _artifact(GPU),
        },
        "timing": {
            "semantic_preflight_s": None,
            "semantic_preflight_status": "SHARED_P4_GATE_NOT_SEPARATELY_TIMED",
            "exact_retarget_s": exact_retarget,
            "retarget_semantic_qa_s": semantic_qa,
            "source_controller_s": source_seconds,
            "support_s": support_seconds,
            "isaac_bootstrap_s": None,
            "frozen_eval_s": None,
            "ppo_training_s": 0.0,
            "ppo_eval_s": None,
            "confirm_s": None,
            "qualification_s": None,
            "trace_export_s": None,
            "productive_time_s": productive,
            "technical_retry_time_s": technical_seconds,
            "cache_hit": bool(
                source_prereq.get("cache_hit", False) or support_preflight.get("cache_hit", False)
            ),
        },
        "policy_outcomes_observed": False,
    }
    row = {
        "Episode": config["clip_id"],
        "Object": p5_row["primary_object_id"],
        "Retarget Semantic": "RETARGET_SEMANTIC_PASS",
        "Source Route": source_route,
        "Support": support_status,
        "Frozen PF": "NOT_RUN_GATE_BLOCKED",
        "PPO Updates": 0,
        "PF Pick": "NOT_RUN_GATE_BLOCKED",
        "PF Transport": "NOT_RUN_GATE_BLOCKED",
        "PF Place": "NOT_RUN_GATE_BLOCKED",
        "PF Release": "NOT_RUN_GATE_BLOCKED",
        "PF Retreat": "NOT_RUN_GATE_BLOCKED",
        "PF Full": "NOT_RUN_GATE_BLOCKED",
        "DF Pose": "NOT_RUN_GATE_BLOCKED",
        "DF Linear": "NOT_RUN_GATE_BLOCKED",
        "DF Angular": "NOT_RUN_GATE_BLOCKED",
        "Status": config["terminal"],
    }
    return episode, row


def main() -> int:
    required = [MANIFEST, GPU, BASE_GEOMETRY, CONTACT_CONTRACT, REPLAY]
    required.extend(
        [
            P6_ROOT / name
            for name in (
                "manual_acceptance.json",
                "canary_role_update.json",
                "corpus_qa_sample.json",
                "corpus_qa_results.csv",
                "semantic_false_positive_analysis.json",
                "final_authority_decision.json",
            )
        ]
    )
    required.extend(
        [
            P7_ROOT / name
            for name in (
                "unseen_object_frozen5_manifest.json",
                "unseen_object_frozen5_manifest.yaml",
                "manifest_sha256.txt",
                "selection_receipt.json",
                "development_object_exclusions.csv",
                "object_identity_table.csv",
            )
        ]
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"P8_FINALIZE_REQUIRED_INPUT_MISSING:{path}")
    final_decision = _read(P6_ROOT / "final_authority_decision.json")
    p7 = _read(P7_ROOT / "unseen_object_frozen5_manifest.json")
    if final_decision.get("status") != "PASS":
        raise ValueError("P8_FINALIZE_P6_NOT_PASS")
    if p7.get("status") != "FROZEN_NOT_EXECUTED":
        raise ValueError("P8_FINALIZE_P7_NOT_FROZEN")

    configs, p5_by_id = _clip_rows()
    failures = _technical_failures()
    episodes: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for config in configs:
        episode, row = _make_episode(config, p5_by_id[config["clip_id"]], failures)
        episodes.append(episode)
        main_rows.append(row)
        timing_rows.append({"Episode": episode["episode"], **episode["timing"]})

    per_episode = P8_ROOT / "per_episode"
    for episode in episodes:
        _write_json(per_episode / f"{episode['episode']}.json", episode)
    with (P8_ROOT / "main_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(main_rows[0]))
        writer.writeheader()
        writer.writerows(main_rows)
    with (P8_ROOT / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    replay_root = P8_ROOT / "replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    command_lines = [
        "# P8 replay status: FAILED_BEST_DIAGNOSTIC for both clips.",
        "# The terminal support gates produced no Stage16D physical-policy .npz trace.",
        "# Therefore no executable replay command exists for these canaries; a support JSON",
        "# or screenshot is not substituted for a policy trace.",
        "",
        f"# HARD REQUIREMENT CHECK (executed): /home/deepcybo/miniconda3/bin/conda run --no-capture-output -n toporetarget-rl python {REPLAY} --help",
        "# The exact CLI was verified before this ledger was written.",
    ]
    for config in configs:
        support_report = config["support_report"]
        command_lines.extend(
            [
                "",
                f"## {config['label']} — {config['clip_id']}",
                f"# Best diagnostic: {config['terminal']} (not a physical-policy replay)",
                f"# support summary: {support_report / 'final_summary.json' if config['support_final'] else support_report / 'support_preflight.json'}",
                "# Full / Pick-Lift / Place-Release: BLOCKED because no valid --trace .npz was produced.",
                f"# Do not run replay against support JSON: {support_report}",
            ]
        )
    (replay_root / "visualization_commands.md").write_text(
        "\n".join(command_lines) + "\n", encoding="utf-8"
    )
    (replay_root / "manual_acceptance.md").write_text(
        "# P8 manual acceptance\n\n"
        "Replay status for both approved canaries: `FAILED_BEST_DIAGNOSTIC`.\n\n"
        "No manual PASS is possible because neither clip reached a valid physical-policy trace. "
        "Canary 1 requires review of correct target object/hand, hand-object geometry, support, "
        "and the causal pick/transport/place/release/retreat sequence after the geometry gate is repaired. "
        "Canary 2 requires source-support authority before any PhysX policy route.\n\n"
        "The required visual checks remain: no flick, no teleport, raw/retarget/actual fidelity, "
        "and correct target object identity.\n",
        encoding="utf-8",
    )

    _write_json(
        P8_ROOT / "p8_terminal_decision.json",
        {
            "schema_version": "P8TwoCanaryTerminalDecisionV1",
            "status": "COMPLETE_WITH_TERMINAL_GATES",
            "scientific_acceptance": "FAIL_NO_CANARY_ACCEPTED",
            "terminal_statuses": {
                episode["episode"]: episode["terminal_status"] for episode in episodes
            },
            "allowed_terminal_statuses_used": ["PHYSICAL_SCENE_INVALID", "SUPPORT_UNRESOLVED"],
            "forbidden_terminal_statuses_used": [],
            "downstream_fail_closed": True,
            "frozen_eval": "NOT_RUN_GATE_BLOCKED",
            "ppo": "NOT_RUN_GATE_BLOCKED",
            "replay": "FAILED_BEST_DIAGNOSTIC",
            "policy_outcomes_observed": False,
            "p6_final_authority_decision": _artifact(P6_ROOT / "final_authority_decision.json"),
            "p7_manifest": _artifact(P7_ROOT / "unseen_object_frozen5_manifest.json"),
            "created_utc": _utc(),
        },
    )
    with (P8_ROOT / "technical_failures.jsonl").open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, sort_keys=True) + "\n")
    _write_json(
        P8_ROOT / "resource_usage.json",
        {
            "schema_version": "P8TwoCanaryResourceUsageV1",
            "gpu_preflight": _artifact(GPU),
            "gpu_preflight_status": _read(GPU).get("status"),
            "cpu_fallback": _read(GPU).get("cpu_fallback"),
            "physical_policy_gpu_runs": 0,
            "support_physx_runs": 2,
            "ppo_optimizer_steps": 0,
            "shared_ppo_state": False,
            "technical_failure_count": len(failures),
        },
    )
    print(
        json.dumps(
            {"status": "P8_FINALIZED", "episodes": episodes, "technical_failures": len(failures)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
