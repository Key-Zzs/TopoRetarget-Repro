#!/usr/bin/env python3
"""Freeze and run independent held-out raw-to-physical refinement batches.

The runner composes only declared production authorities.  It is intentionally
unable to promote a development-only physical CLI to held-out usage: a missing
authority creates durable ``PIPELINE_INVALID`` receipts with zero PPO updates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    append_stage_receipt,
    assert_frozen_manifest,
    atomic_write_json,
    atomic_write_text,
    freeze_method_contract,
    freeze_selection,
    scan_hocap_candidates,
    stable_hash,
    validate_authority_manifest,
    write_capability_gap_receipts,
    write_clip_state,
)

DEFAULT_RAW_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/HOCap")
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/independent_multiclip_hocap_pilot"
DEFAULT_RUN_ROOT = REPO_ROOT / ".local/runs/independent_multiclip_hocap_pilot"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate-config", "execute"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--authority-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _current_authority_manifest() -> dict[str, Any]:
    """Describe current production CLIs from their real public boundaries."""

    supported = ["hocap_170105", "hocap_170650"]
    return {
        "schema_version": "IndependentPhysicalRefinementAuthorityAuditV1",
        "source": "current_repository_cli_boundaries",
        "authorities": {
            "retarget": {"supported_clips": []},
            "source_policy": {"supported_clips": supported},
            "support": {"supported_clips": supported},
            "frozen_evaluation": {"supported_clips": supported},
            "physical_refinement": {"supported_clips": supported},
            "qualification": {"supported_clips": supported},
            "trace_export": {"supported_clips": supported},
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchContractError(f"AUTHORITY_MANIFEST_INVALID:{path}")
    return value


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _template(value: str, *, clip: dict[str, Any], report_root: Path, run_root: Path) -> str:
    return value.format(
        clip_id=clip["clip_id"],
        sequence=clip["sequence"],
        report_root=str(report_root),
        run_root=str(run_root),
        clip_report_root=str(report_root / "clips" / clip["clip_id"]),
        clip_run_root=str(run_root / clip["clip_id"]),
    )


def _run_authority(
    *,
    name: str,
    entry: dict[str, Any],
    clip: dict[str, Any],
    report_root: Path,
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call one declared authority and require its durable, hashable receipt."""

    command_template = entry.get("command")
    receipt_template = entry.get("receipt")
    if not isinstance(command_template, list) or not all(
        isinstance(item, str) for item in command_template
    ):
        raise BatchContractError(f"AUTHORITY_COMMAND_MISSING:{name}")
    if not isinstance(receipt_template, str):
        raise BatchContractError(f"AUTHORITY_RECEIPT_MISSING:{name}")
    command = [
        _template(item, clip=clip, report_root=report_root, run_root=run_root)
        for item in command_template
    ]
    receipt_path = Path(
        _template(receipt_template, clip=clip, report_root=report_root, run_root=run_root)
    )
    started = _utc_now()
    start_tick = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - start_tick
    ended = _utc_now()
    driver = {
        "schema_version": "IndependentPhysicalRefinementAuthorityDriverV1",
        "authority": name,
        "clip_id": clip["clip_id"],
        "command": command,
        "exit_code": result.returncode,
        "start_utc": started,
        "end_utc": ended,
        "wall_seconds": wall_seconds,
        "stdout_tail": result.stdout[-12000:],
        "stderr_tail": result.stderr[-12000:],
        "receipt_expected": str(receipt_path),
    }
    atomic_write_json(report_root / "clips" / clip["clip_id"] / name / "driver.json", driver)
    if result.returncode != 0:
        raise BatchContractError(
            f"AUTHORITY_COMMAND_FAILED:{name}:{clip['clip_id']}:{result.returncode}"
        )
    if not receipt_path.is_file():
        raise BatchContractError(f"AUTHORITY_RECEIPT_NOT_WRITTEN:{name}:{clip['clip_id']}")
    receipt = _load_manifest(receipt_path)
    if receipt.get("status") not in {"COMPLETE", "ACCEPTED", "PASS"}:
        raise BatchContractError(f"AUTHORITY_RECEIPT_NOT_COMPLETE:{name}:{clip['clip_id']}")
    return receipt, {
        "started_utc": started,
        "ended_utc": ended,
        "wall_seconds": wall_seconds,
        "exit_code": result.returncode,
        "output_hashes": {"authority_receipt": stable_hash(receipt)},
    }


def _execute_declared_authorities(
    *,
    manifest: dict[str, Any],
    method_hash: str,
    authority: dict[str, Any],
    report_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Run one independent, sequential lineage per clip from declared commands."""

    authorities = authority["authorities"]
    final_receipts: list[str] = []
    for clip in manifest["clips"]:
        state_path = report_root / "clips" / clip["clip_id"] / "final_receipt.json"
        if state_path.is_file():
            state = _load_manifest(state_path)
            completed_stages = {
                str(item.get("stage"))
                for item in state.get("stages", [])
                if isinstance(item, dict)
                and (
                    item.get("status") in {"COMPLETE", "ACCEPTED"}
                    or str(item.get("status", "")).endswith("_DONE")
                )
            }
            if (
                state.get("state")
                in {
                    "ACCEPTED_FROZEN",
                    "ACCEPTED_AFTER_REFINEMENT",
                    "PPO_BUDGET_EXHAUSTED",
                }
                and "trace_export" in completed_stages
            ):
                final_receipts.append(str(state_path.resolve()))
                continue
        else:
            from toporetarget.rl.independent_physical_refinement import initial_clip_state

            state = initial_clip_state(clip=clip, method_contract_hash=method_hash)
            completed_stages: set[str] = set()
        for name in ("retarget", "source_policy", "support", "frozen_evaluation"):
            if name in completed_stages:
                continue
            receipt, timing = _run_authority(
                name=name,
                entry=dict(authorities[name]),
                clip=clip,
                report_root=report_root,
                run_root=run_root,
            )
            state = append_stage_receipt(
                state,
                stage=name,
                status=f"{name.upper()}_DONE",
                input_hashes=state["raw_hashes"],
                cache_hit=bool(receipt.get("cache_hit", False)),
                retry_count=int(receipt.get("retry_count", 0)),
                productive_run_seconds=float(
                    receipt.get("productive_run_seconds", timing["wall_seconds"])
                ),
                technical_retry_seconds=float(receipt.get("technical_retry_seconds", 0.0)),
                **timing,
            )
            state.setdefault("authority_outputs", {})[name] = receipt
            write_clip_state(report_root, state)
        frozen = state.get("authority_outputs", {}).get("frozen_evaluation")
        if not isinstance(frozen, dict):
            raise BatchContractError(f"FROZEN_EVALUATION_RECEIPT_MISSING:{clip['clip_id']}")
        if bool(frozen.get("accepted", False)):
            status = "ACCEPTED_FROZEN"
            state = append_stage_receipt(
                state,
                stage="evaluate_first_decision",
                status=status,
                started_utc=_utc_now(),
                ended_utc=_utc_now(),
                wall_seconds=0.0,
                input_hashes=state["raw_hashes"],
            )
            state["PPO_UPDATES"] = 0
        else:
            receipt, timing = _run_authority(
                name="physical_refinement",
                entry=dict(authorities["physical_refinement"]),
                clip=clip,
                report_root=report_root,
                run_root=run_root,
            )
            updates = int(receipt.get("ppo_updates", -1))
            progression = receipt.get("eval10_by_update")
            if (
                updates < 0
                or updates > 15
                or not isinstance(progression, list)
                or len(progression) != updates
            ):
                raise BatchContractError(
                    f"PPO_AUTHORITY_BUDGET_OR_EVAL_CONTRACT_INVALID:{clip['clip_id']}"
                )
            status = (
                "ACCEPTED_AFTER_REFINEMENT"
                if receipt.get("accepted") is True
                else "PPO_BUDGET_EXHAUSTED"
            )
            state = append_stage_receipt(
                state,
                stage="physical_refinement",
                status=status,
                input_hashes=state["raw_hashes"],
                cache_hit=bool(receipt.get("cache_hit", False)),
                retry_count=int(receipt.get("retry_count", 0)),
                productive_run_seconds=float(
                    receipt.get("productive_run_seconds", timing["wall_seconds"])
                ),
                technical_retry_seconds=float(receipt.get("technical_retry_seconds", 0.0)),
                **timing,
            )
            state.setdefault("authority_outputs", {})["physical_refinement"] = receipt
            state["PPO_UPDATES"] = updates
            state["ppo_progression"] = progression
        write_clip_state(report_root, state)
        for name in ("qualification", "trace_export"):
            if name in completed_stages:
                continue
            receipt, timing = _run_authority(
                name=name,
                entry=dict(authorities[name]),
                clip=clip,
                report_root=report_root,
                run_root=run_root,
            )
            # A terminal state may receive export/qualification metadata, but never more PPO.
            state["stages"].append(
                {
                    "stage": name,
                    "status": "COMPLETE",
                    "input_hashes": state["raw_hashes"],
                    "cache_hit": bool(receipt.get("cache_hit", False)),
                    "retry_count": int(receipt.get("retry_count", 0)),
                    "productive_run_seconds": float(
                        receipt.get("productive_run_seconds", timing["wall_seconds"])
                    ),
                    "technical_retry_seconds": float(receipt.get("technical_retry_seconds", 0.0)),
                    "start_utc": timing["started_utc"],
                    "end_utc": timing["ended_utc"],
                    **timing,
                }
            )
            state.setdefault("authority_outputs", {})[name] = receipt
            write_clip_state(report_root, state)
        final_receipts.append(str(state_path.resolve()))
    return {"status": "COMPLETE", "final_receipts": final_receipts, "max_concurrent_clips": 1}


def _write_preflight_block_summary(
    *, root: Path, manifest: dict[str, Any], method_hash: str, batch: dict[str, Any]
) -> None:
    """Publish explicit NOT_RUN summaries when execution stops before a rollout."""

    clips = manifest["clips"]
    aggregate = root / "aggregate"
    analysis = root / "analysis"
    replay = root / "replay"
    for directory in (aggregate, analysis, replay, root / "tests", root / "orchestrator"):
        directory.mkdir(parents=True, exist_ok=True)
    with (aggregate / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["clip_id", "frozen_pf_v2", "ppo_updates", "final_pf_v2", "final_status"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "clip_id": clip["clip_id"],
                "frozen_pf_v2": "NOT_RUN",
                "ppo_updates": 0,
                "final_pf_v2": "NOT_RUN",
                "final_status": "PIPELINE_INVALID",
            }
            for clip in clips
        )
    with (aggregate / "timing_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "stage", "seconds", "status"])
        writer.writeheader()
        writer.writerows(
            {
                "clip_id": clip["clip_id"],
                "stage": "raw_to_final_simulation",
                "seconds": "NOT_RUN",
                "status": "PIPELINE_INVALID_BEFORE_RETARGET",
            }
            for clip in clips
        )
    with (aggregate / "failure_taxonomy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "category", "secondary_reason"])
        writer.writeheader()
        writer.writerows(
            {
                "clip_id": clip["clip_id"],
                "category": "PIPELINE_INVALID",
                "secondary_reason": "PRODUCTION_AUTHORITY_DOES_NOT_SUPPORT_HELD_OUT_CLIP",
            }
            for clip in clips
        )
    with (aggregate / "method_contract_hashes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "method_contract_hash"])
        writer.writeheader()
        writer.writerows(
            {"clip_id": clip["clip_id"], "method_contract_hash": method_hash} for clip in clips
        )
    atomic_write_json(
        analysis / "success_rates.json",
        {
            "SR_frozen": "NOT_RUN",
            "RecoveryRate": "NOT_RUN",
            "SR_final": "NOT_RUN",
            "UTR": "NOT_RUN",
            "denominator": len(clips),
            "reason": batch["reason"],
        },
    )
    atomic_write_json(
        analysis / "compute_cost.json",
        {"total_ppo_updates": 0, "total_samples": 0, "status": "NOT_RUN"},
    )
    atomic_write_json(
        analysis / "bottleneck_analysis.json",
        {
            "PRIMARY_BOTTLENECK": "NOT_RUN",
            "status": "NOT_IDENTIFIABLE",
            "reason": "no raw-to-simulation execution was authorized",
        },
    )
    atomic_write_text(
        analysis / "bottleneck_analysis.md",
        "# Bottleneck analysis\n\n"
        "NOT_IDENTIFIABLE: the physical authority preflight stopped before retargeting.\n",
    )
    atomic_write_text(
        replay / "visualization_commands.md",
        "# Replay commands\n\n"
        "NOT_RUN for every held-out clip: no physical trace exists because authority preflight "
        "failed.\n",
    )
    atomic_write_text(
        root / "orchestrator/state_machine.md",
        "# State machine\n\n"
        "`SELECTED -> RAW_VALIDATED -> RETARGETED -> SOURCE_POLICY_READY -> SUPPORT_READY "
        "-> FROZEN_EVAL_DONE -> ACCEPTED_FROZEN | PPO_CANDIDATE "
        "-> ACCEPTED_AFTER_REFINEMENT | PPO_BUDGET_EXHAUSTED`.\n\n"
        "A missing authority enters `PIPELINE_INVALID` before retarget/PPO.\n",
    )
    atomic_write_json(
        root / "orchestrator/resume_test.json", {"status": "NOT_RUN", "reason": batch["reason"]}
    )
    summary = {
        "schema_version": "IndependentPhysicalRefinementPilotSummaryV1",
        "status": "PIPELINE_INVALID",
        "method_contract_hash": method_hash,
        "held_out_count": len(clips),
        "raw_to_final_pipeline_run": False,
        "ppo_updates": 0,
        "summary_metrics": "NOT_RUN",
        "reason": batch["reason"],
        "authority_preflight": batch["authority_preflight"],
    }
    atomic_write_json(root / "final_summary.json", summary)
    atomic_write_text(
        root / "final_summary.md",
        "# Independent Multi-Clip HOCap Physical Refinement\n\n"
        "`PIPELINE_INVALID`: current production authority does not support the frozen held-out "
        "clips. No retarget, simulation, trace, or PPO update was run; success rates and timing "
        "are `NOT_RUN`.\n",
    )
    atomic_write_text(
        root / "handoff.md",
        "# Independent Multi-Clip HOCap Physical Refinement Handoff\n\n"
        "Selection is frozen, but `RAW_TO_FINAL_PIPELINE_RUN=NO` because every required physical "
        "authority is development-only. `PPO_UPDATES=0`; no held-out result, replay command, "
        "success rate, or bottleneck claim is available.\n",
    )


def _write_contract(
    root: Path, manifest: dict[str, Any], method_hash: str, authority: dict[str, Any]
) -> None:
    contracts = root / "contracts"
    pipeline = {
        "schema_version": "SingleClipPhysicalPipelineContractV1",
        "raw_input": "HOCap poses_m.npy + poses_o.npy + meta.yaml + canonical object mesh",
        "stages": [
            "raw_validation",
            "geometric_retarget",
            "source_policy_preparation",
            "support_preparation",
            "frozen_eval10",
            "frozen_confirm20_if_candidate",
            "failure_only_independent_ppo",
            "confirm20_and_stop",
            "final_trace_export",
        ],
        "authority_manifest_sha256": validate_authority_manifest(
            authority, [item["clip_id"] for item in manifest["clips"]]
        )["authority_manifest_sha256"],
        "method_contract_hash": method_hash,
        "cache_policy": "immutable_global_assets_may_be_reused_and_must_be_receipted",
        "unsupported_current_authority": (
            "no raw-held-out source-policy and physical runner authority"
        ),
    }
    atomic_write_json(contracts / "single_clip_pipeline_contract.json", pipeline)


def _run(args: argparse.Namespace) -> int:
    root = args.report_root.resolve()
    manifest_path = root / "selection/held_out_5_manifest.json"
    if args.mode == "prepare":
        if manifest_path.exists() and not args.resume:
            raise BatchContractError("SELECTION_NAMESPACE_EXISTS_USE_RESUME")
        candidates = scan_hocap_candidates(args.raw_root.resolve())
        manifest = freeze_selection(candidates=candidates, root=root)
        _, method_hash = freeze_method_contract(root)
    else:
        if not manifest_path.is_file():
            raise BatchContractError("HELD_OUT_MANIFEST_MISSING_RUN_PREPARE_FIRST")
        manifest = _load_manifest(manifest_path)
        assert_frozen_manifest(manifest)
        method_hash = (
            (root / "contracts/method_contract_hash.txt").read_text(encoding="utf-8").strip()
        )
    authority = (
        _load_manifest(args.authority_manifest)
        if args.authority_manifest
        else _current_authority_manifest()
    )
    _write_contract(root, manifest, method_hash, authority)
    check = validate_authority_manifest(authority, [item["clip_id"] for item in manifest["clips"]])
    atomic_write_json(root / "orchestrator/authority_preflight.json", check)
    if args.mode == "validate-config":
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0 if check["valid"] else 2
    if args.mode == "execute":
        if not check["valid"]:
            paths = write_capability_gap_receipts(
                root=root, manifest=manifest, method_contract_hash=method_hash, authority=authority
            )
            batch = {
                "status": "PIPELINE_INVALID",
                "reason": "PRODUCTION_AUTHORITY_DOES_NOT_SUPPORT_FROZEN_HELD_OUT_SET",
                "clip_receipts": [str(path.resolve()) for path in paths],
                "ppo_optimizer_steps": 0,
                "authority_preflight": check,
            }
            atomic_write_json(root / "orchestrator/batch_receipt.json", batch)
            _write_preflight_block_summary(
                root=root, manifest=manifest, method_hash=method_hash, batch=batch
            )
            return 2
        batch = _execute_declared_authorities(
            manifest=manifest,
            method_hash=method_hash,
            authority=authority,
            report_root=root,
            run_root=args.run_root.resolve(),
        )
        atomic_write_json(root / "orchestrator/batch_receipt.json", batch)
        return 0
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "method_contract_hash": method_hash,
                **check,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except BatchContractError as error:
        print(f"INDEPENDENT_PHYSICAL_REFINEMENT_BATCH_ERROR:{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
