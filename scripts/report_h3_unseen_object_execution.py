#!/usr/bin/env python3
"""Aggregate H3-D unseen-object Frozen5 execution or emit its blocked receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {
    "ACCEPTED_FROZEN_FULL_CYCLE",
    "ACCEPTED_AFTER_REFINEMENT_FULL_CYCLE",
    "ACCEPTED_PICK_ONLY",
    "PPO_BUDGET_EXHAUSTED",
    "SOURCE_CONTROLLER_TRUE_HARD_FAILURE",
    "SUPPORT_UNRESOLVED",
    "RETARGET_FAILED_AFTER_VALID_INPUT",
    "PF_PASS_DF_FAIL",
    "TECHNICAL_FAILURE",
}

MAIN_COLUMNS = (
    "episode",
    "object_id",
    "retarget",
    "source_route",
    "source_executable",
    "source_fidelity",
    "support",
    "frozen_pf",
    "ppo_updates",
    "pf_pick",
    "pf_transport",
    "pf_place",
    "pf_release",
    "pf_retreat",
    "pf_full",
    "df_pose",
    "df_linear",
    "df_angular",
    "status",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--h3c-readiness", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H3D_JSON_OBJECT_REQUIRED:{path}")
    return value


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H3D_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3D_REPORT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3D_REPORT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _passed(value: object) -> bool:
    if value is True:
        return True
    text = str(value).upper()
    return text in {"PASS", "TRUE", "10/10", "20/20"} or text.startswith("PASS_")


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected = manifest.get("manifest_sha256")
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    clips = core.get("clips")
    if not (
        core.get("schema_version") == "H3UnseenObjectFrozen5ManifestV1"
        and core.get("HELD_OUT_SET_FROZEN") == "YES"
        and core.get("held_out_count") == 5
        and isinstance(expected, str)
        and _stable_hash(core) == expected
        and isinstance(clips, list)
        and len(clips) == 5
        and isinstance(core.get("h3_protocol_hash"), str)
        and len(core["h3_protocol_hash"]) == 64
    ):
        raise ValueError("H3D_UNSEEN_OBJECT_MANIFEST_INVALID")
    ids = [str(row.get("episode_id")) for row in clips if isinstance(row, dict)]
    objects = [str(row.get("primary_object_id")) for row in clips if isinstance(row, dict)]
    if len(set(ids)) != 5 or len(set(objects)) != 5:
        raise ValueError("H3D_UNSEEN_OBJECT_MANIFEST_IDENTITY_DRIFT")
    return [dict(row) for row in clips]


def aggregate(manifest_path: Path, readiness_path: Path, report_root: Path) -> dict[str, Any]:
    manifest = _json(manifest_path.resolve())
    clips = _validate_manifest(manifest)
    readiness = _json(readiness_path.resolve())
    ready = readiness.get("H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION") == "YES"
    output = report_root.resolve()
    authorities = {
        "manifest": _artifact(manifest_path.resolve()),
        "h3c_readiness": _artifact(readiness_path.resolve()),
    }
    if not ready:
        blocked = {
            "schema_version": "H3UnseenObjectExecutionBlockedReceiptV1",
            "status": "EXECUTION_BLOCKED",
            "claim": "H3D_EXECUTION_BLOCKED_BY_PIPELINE_READINESS",
            "reason": "H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION=NO",
            "downstream_outcomes_observed": False,
            "ppo_updates": 0,
            "episode_ids": [row["episode_id"] for row in clips],
            "object_ids": [row["primary_object_id"] for row in clips],
            "authorities": authorities,
        }
        _write_new(output / "execution_blocked_receipt.json", blocked)
        return blocked

    rows: list[dict[str, Any]] = []
    result_artifacts: list[dict[str, str]] = []
    for clip in clips:
        episode = str(clip["episode_id"])
        path = output / "per_episode" / episode / "final_status.json"
        row = _json(path)
        if not (
            row.get("schema_version") == "H3UnseenObjectEpisodeResultV1"
            and row.get("episode") == episode
            and row.get("object_id") == clip.get("primary_object_id")
            and row.get("status") in TERMINAL_STATUSES
            and row.get("held_out") is True
            and row.get("method_contract_hash") == manifest.get("h3_protocol_hash")
            and row.get("per_episode_tuning") is False
        ):
            raise ValueError(f"H3D_EPISODE_RESULT_INVALID:{episode}")
        missing = sorted(set(MAIN_COLUMNS) - row.keys())
        if missing:
            raise ValueError(f"H3D_MAIN_METRICS_MISSING:{episode}:{missing}")
        if not isinstance(row.get("timing_seconds"), dict) or not isinstance(
            row.get("failure_taxonomy"), list
        ):
            raise ValueError(f"H3D_DIAGNOSTICS_MISSING:{episode}")
        if not all(
            isinstance(row.get(field), bool)
            for field in (
                "accepted_frozen",
                "accepted_final",
                "initial_physical_failure",
                "recovered_by_ppo",
            )
        ):
            raise ValueError(f"H3D_ACCEPTANCE_FLAGS_MISSING:{episode}")
        if row["accepted_frozen"] and row["initial_physical_failure"]:
            raise ValueError(f"H3D_FROZEN_ACCEPTANCE_CONTRADICTION:{episode}")
        if row["recovered_by_ppo"] and not (
            row["initial_physical_failure"] and row["accepted_final"]
        ):
            raise ValueError(f"H3D_RECOVERY_CONTRADICTION:{episode}")
        lineage = row.get("lineage")
        replay = row.get("replay_commands")
        if (
            not isinstance(lineage, dict)
            or not isinstance(replay, dict)
            or set(replay)
            != {
                "full",
                "pick_lift",
                "place_release",
            }
        ):
            raise ValueError(f"H3D_LINEAGE_OR_REPLAY_MISSING:{episode}")
        rows.append(row)
        result_artifacts.append(_artifact(path))

    lineage_fields = ("actor_root", "critic_root", "optimizer_root", "normalizer_root", "rng_seed")
    for field in lineage_fields:
        values = [row["lineage"].get(field) for row in rows]
        if any(value is None for value in values) or len(set(values)) != 5:
            raise ValueError(f"H3D_INDEPENDENT_LINEAGE_DRIFT:{field}")

    _write_csv(
        output / "main_metrics.csv",
        [{field: row[field] for field in MAIN_COLUMNS} for row in rows],
        MAIN_COLUMNS,
    )
    timing_rows = [
        {"episode": row["episode"], "phase": phase, "seconds": seconds}
        for row in rows
        for phase, seconds in sorted(row["timing_seconds"].items())
    ]
    _write_csv(output / "timing.csv", timing_rows, ("episode", "phase", "seconds"))
    failure_rows = [
        {
            "episode": row["episode"],
            "classification": failure.get("classification", "UNCLASSIFIED"),
            "stage": failure.get("stage", "UNKNOWN"),
            "resolved": bool(failure.get("resolved", False)),
            "details": str(failure.get("details", "")),
        }
        for row in rows
        for failure in row["failure_taxonomy"]
        if isinstance(failure, dict)
    ]
    _write_csv(
        output / "failure_taxonomy.csv",
        failure_rows,
        ("episode", "classification", "stage", "resolved", "details"),
    )
    accepted_frozen = sum(bool(row["accepted_frozen"]) for row in rows)
    recovered = sum(bool(row["recovered_by_ppo"]) for row in rows)
    initial_failures = sum(bool(row["initial_physical_failure"]) for row in rows)
    accepted_final = sum(bool(row["accepted_final"]) for row in rows)
    if initial_failures != 5 - accepted_frozen:
        raise ValueError("H3D_INITIAL_FAILURE_DENOMINATOR_DRIFT")
    if accepted_final != accepted_frozen + recovered:
        raise ValueError("H3D_FINAL_ACCEPTANCE_DECOMPOSITION_DRIFT")
    pf_pick_success = sum(_passed(row["pf_pick"]) for row in rows)
    pf_full_success = sum(_passed(row["pf_full"]) for row in rows)
    df_pose_success = sum(_passed(row["df_pose"]) for row in rows)
    df_linear_success = sum(_passed(row["df_linear"]) for row in rows)
    df_angular_success = sum(_passed(row["df_angular"]) for row in rows)
    claim = (
        "UNSEEN_OBJECT_INSTANCE_METHOD_GENERALIZATION_SUPPORTED"
        if accepted_final == 5
        else (
            "UNSEEN_OBJECT_GENERALIZATION_NOT_SUPPORTED"
            if accepted_final == 0
            else "UNSEEN_OBJECT_GENERALIZATION_MIXED"
        )
    )
    metrics = {
        "schema_version": "H3UnseenObjectGeneralizationMetricsV1",
        "status": "COMPLETE",
        "denominator": 5,
        "N_accepted_frozen": accepted_frozen,
        "N_initial_failures": initial_failures,
        "N_recovered_by_PPO": recovered,
        "N_final_accepted": accepted_final,
        "SR_frozen": accepted_frozen / 5.0,
        "RR": None if initial_failures == 0 else recovered / initial_failures,
        "SR_final": accepted_final / 5.0,
        "UTR": 0.0,
        "PF_pick_success": pf_pick_success,
        "PF_pick_success_rate": pf_pick_success / 5.0,
        "PF_full_cycle_success": pf_full_success,
        "PF_full_cycle_success_rate": pf_full_success / 5.0,
        "DF_pose_success": df_pose_success,
        "DF_pose_success_rate": df_pose_success / 5.0,
        "DF_linear_success": df_linear_success,
        "DF_linear_success_rate": df_linear_success / 5.0,
        "DF_angular_success": df_angular_success,
        "DF_angular_success_rate": df_angular_success / 5.0,
        "claim": claim,
        "claim_scope": (
            "method-level generalization to objects not used during method development; "
            "episode-specific retarget references, source controllers, and PPO lineages"
        ),
        "shared_policy_zero_shot_claim": False,
        "SHARED_POLICY_ZERO_SHOT_CLAIM": "NO",
        "INDEPENDENT_PPO_PER_EPISODE": "YES",
        "independent_lineage_fields": list(lineage_fields),
        "authorities": authorities,
        "episode_results": result_artifacts,
    }
    _write_new(output / "generalization_metrics.json", metrics)
    decision = {
        "schema_version": "H3UnseenObjectExecutionDecisionV1",
        "status": "COMPLETE",
        "H3D_EXECUTION": "COMPLETE",
        "claim": claim,
        "metrics": metrics,
        "terminal_statuses": {row["episode"]: row["status"] for row in rows},
    }
    _write_new(output / "final_decision.json", decision)
    return decision


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            aggregate(args.manifest, args.h3c_readiness, args.report_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
