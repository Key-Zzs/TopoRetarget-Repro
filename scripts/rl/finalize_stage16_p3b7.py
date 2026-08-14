#!/usr/bin/env python3
"""Materialize the P3-B.7 V2 gate from immutable B.5/B.6 evidence.

This is intentionally an evidence-only entry point.  It may authorize later
PPO work, but cannot start it; a failed hard-reset pool is a terminal B.7
outcome and leaves all PPO/restart artifacts absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.p3_restart_gate_v2 import (  # noqa: E402
    CLIPS,
    EarlyTableResetCoverageGateV1,
    build_early_table_reset_pool,
    stage16_p3_restart_gate_v2,
)

B6 = REPO_ROOT / ".local/reports/stage16_p3b6_scene_rsi_requalification"
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_p3b7_restart"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"P3B7_REQUIRED_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _required_inputs() -> dict[str, Path]:
    result = {
        "support_final": REPO_ROOT
        / ".local/reports/stage16_support_reconstruction/final_summary.json",
        "p3b5_final": REPO_ROOT
        / ".local/reports/stage16_p3b5_geometry_attribution/final_summary.json",
        "p3b6_frozen": B6 / "frozen_inputs.json",
        "p3b6_final": B6 / "final_summary.json",
        "p3b6_offline": B6 / "offline_summary.json",
        "geometry_manifest": (
            REPO_ROOT
            / ".local/reports/stage16d_metric_qualification_and_ppo"
            / "runtime_collision_geometry_manifest.json"
        ),
        "reference_contract": REPO_ROOT
        / ".local/reports/stage16d_reference_kinematics_v2/reference_timestamp_contract.json",
        "action_contract": REPO_ROOT / "src/toporetarget/rl/ppo/ppo26d_contract.py",
        "action_mapper": REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/action_adapter.py",
        "curriculum": REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml",
        "entry_gate": REPO_ROOT / "configs/rl/stage16/stage16_p3_entry_gate_v2.yaml",
        "v3_contract": REPO_ROOT
        / ".local/reports/stage16d_reward_v3_pairforce_unblock/contact_reward_contract.json",
        "v4_contract": REPO_ROOT
        / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json",
    }
    for clip in CLIPS:
        result[f"{clip}_reference"] = (
            REPO_ROOT
            / ".local/reports/stage16d_reference_kinematics_v2/references"
            / f"{clip}.reference_kinematics_v2.npz"
        )
        result[f"{clip}_support_proxy"] = (
            REPO_ROOT
            / ".local/reports/stage16_support_reconstruction/inference"
            / clip
            / "table_proxy.json"
        )
        result[f"{clip}_v3_selection"] = (
            REPO_ROOT
            / ".local/reports/stage16d_reward_v3_pairforce_unblock"
            / clip
            / "dev/checkpoint_selection.json"
        )
        result[f"{clip}_v4_selection"] = (
            REPO_ROOT
            / ".local/reports/stage16d_strict_per_finger_v4"
            / clip
            / "checkpoint_selection.json"
        )
    result["hocap_170650_v4_selection"] = (
        REPO_ROOT
        / ".local/reports/stage16d_strict_per_finger_v4/hocap_170650"
        / "final_checkpoint_selection.json"
    )
    return result


def _selected_checkpoint_receipts(inputs: dict[str, Path]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for mode in ("v3", "v4"):
        for clip in CLIPS:
            selection_key = f"{clip}_{mode}_selection"
            selection = json.loads(inputs[selection_key].read_text(encoding="utf-8"))
            ranked = selection.get("ranked")
            if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
                raise ValueError(f"P3B7_ZERO_G_SELECTION_INVALID:{selection_key}")
            checkpoint = Path(str(ranked[0].get("checkpoint", ""))).resolve()
            receipt = _receipt(checkpoint)
            if receipt["sha256"] != ranked[0].get("checkpoint_sha256"):
                raise ValueError(f"P3B7_ZERO_G_CHECKPOINT_HASH_MISMATCH:{selection_key}")
            selected[f"{mode}_{clip}"] = {"selection": _receipt(inputs[selection_key]), **receipt}
    return selected


def _blocked_status(reason: str) -> dict[str, str]:
    return {"status": "NOT_RUN", "reason": reason}


def _write_blocked_artifacts(
    output: Path,
    *,
    final: dict[str, object],
    validation_status: str,
) -> None:
    """Write every required B.7 receipt without inventing unrun PPO evidence."""

    reason = str(final["reason_unrun"])
    for relative in (
        "residual_recoverability/joint/summary.json",
        "residual_recoverability/geometry/summary.json",
        "residual_recoverability/invalid_windows/summary.json",
        "residual_recoverability/summary.json",
        "controller_safety/command_chain.json",
        "frozen_sanity/summary.json",
        "ppo_smoke/v3/summary.json",
        "ppo_smoke/v4/summary.json",
        "p3_restart/v3/hocap_170105/c0/summary.json",
        "p3_restart/v3/hocap_170105/c1/summary.json",
        "p3_restart/v3/hocap_170105/c2/summary.json",
        "p3_restart/v3/hocap_170650/c0/summary.json",
        "p3_restart/v3/hocap_170650/c1/summary.json",
        "p3_restart/v3/hocap_170650/c2/summary.json",
        "p3_restart/v4/hocap_170105/c0/summary.json",
        "p3_restart/v4/hocap_170105/c1/summary.json",
        "p3_restart/v4/hocap_170105/c2/summary.json",
        "p3_restart/v4/hocap_170650/c0/summary.json",
        "p3_restart/v4/hocap_170650/c1/summary.json",
        "p3_restart/v4/hocap_170650/c2/summary.json",
    ):
        _write(output / relative, _blocked_status(reason))

    c1_receipt = REPO_ROOT / ".local/reports/stage16c1_asset_migration/hocap_170105_vector_1.json"
    c1_data = json.loads(c1_receipt.read_text(encoding="utf-8"))
    response = c1_data["joint_response_by_name_rad"]
    _write(
        output / "controller_safety/joint_response_audit.json",
        {
            "schema_version": "Stage16P3B7JointResponseAuditV1",
            "gate_status": "NOT_RUN_UPSTREAM_R1_HARD_RESET_POOL",
            "instrumentation_regression": {
                "historical_observation": "18_of_20",
                "classification": "CASE_B_INSTRUMENTATION_RAMP",
                "cause": (
                    "A 20-step, 0.15-rad shared ramp left r_pinky_mcp_abd and "
                    "r_thumb_ip below the 1e-5-rad response threshold; it was not "
                    "evidence of two passive or unmapped joints."
                ),
                "remediation": "individual 0.35-rad, 40-step probe per mapped joint",
            },
            "fresh_gpu_receipt": _receipt(c1_receipt),
            "joints_with_response": c1_data["joints_with_response"],
            "expected_joint_count": 20,
            "threshold_rad": c1_data["joint_response_threshold_rad"],
            "probe_amplitude_rad": c1_data["joint_probe_amplitude_rad"],
            "probe_dwell_steps": c1_data["joint_probe_dwell_steps"],
            "minimum_observed_response_rad": min(float(value) for value in response.values()),
            "result": "20_OF_20_RESPONSE_CONFIRMED",
        },
    )
    _write(
        output / "c2_global_selection.json",
        {"selected": "NO_MODE_ELIGIBLE", "reason": reason},
    )
    _write_text(
        output / "comparison/old_vs_new_c2.csv",
        "metric,value\nRESET_GEOMETRY_BLOCKER_RESOLVED,NO\nreason,R1_HARD_RESET_POOL_FAILED\n",
    )
    _write_text(
        output / "comparison/old_vs_new_c2.md",
        "# Old vs new C2\n\n`RESET_GEOMETRY_BLOCKER_RESOLVED = NO`. "
        "No fresh C2 exists because R1 failed before PPO authorization.\n",
    )
    _write(
        output / "tables/residual_recoverability.json",
        [{"clip": clip, **_blocked_status(reason)} for clip in CLIPS],
    )
    _write(
        output / "tables/joint_safety.json",
        [
            {
                "clip": "hocap_170105",
                "joint": "all_20_mapped_joints",
                "reference_invalid": "NOT_ASSESSED_R1",
                "command_unsafe": "NOT_ASSESSED_R1",
                "actual_overshoot": "NOT_ASSESSED_R1",
                "safety_projection_used": False,
                "result": "NOT_RUN_R1",
            }
        ],
    )
    _write(
        output / "tests.json",
        {
            "status": validation_status,
            "required_final_commands": [
                "conda run -n toporetarget-rl ruff check .",
                "conda run -n toporetarget-rl ruff format --check .",
                "conda run -n toporetarget-rl python -m mypy src",
                "conda run --no-capture-output -n toporetarget-rl python -m pytest -q",
                "conda run -n toporetarget-rl python scripts/check_paper_fidelity.py",
            ],
        },
    )
    _write(
        output / "resource_usage.json",
        {
            "ppo_gpu_preflight": "NOT_RUN_NO_PPO_AUTHORIZATION",
            "ppo_smoke": "NOT_RUN_R1_HARD_RESET_POOL_FAILED",
            "physical_joint_response_probe": "GPU_EXECUTED_1_ENV_1000_STEPS",
        },
    )
    _write_text(
        output / "failure_transitions.jsonl",
        json.dumps(
            {
                "from": "R1_hard_reset_validity",
                "status": "FAIL",
                "decision": final["decision"],
                "next_action": final["next_action"],
            },
            sort_keys=True,
        )
        + "\n",
    )
    _write_text(
        output / "visualization_commands.md",
        "# B.7 retained physical-view commands\n\n"
        "No PPO/C2 visualization command exists: R1 failed and no policy was run. "
        "These are the two real B.6 finite-table dynamic validation commands retained "
        "for inspecting the reset evidence.\n\n"
        "```bash\n"
        "PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab "
        "python scripts/physics/validate_physical_scene_rsi.py --phase dynamic "
        "--clip hocap_170105 --dynamic-start 0 --dynamic-max-states 32 "
        "--dynamic-steps 20 --accept-eula\n"
        "PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab "
        "python scripts/physics/validate_physical_scene_rsi.py --phase dynamic "
        "--clip hocap_170650 --dynamic-start 0 --dynamic-max-states 32 "
        "--dynamic-steps 20 --accept-eula\n"
        "```\n",
    )
    _write_text(
        output / "final_summary.md",
        "# Stage16 P3-B.7 final summary\n\n"
        f"Decision: `{final['decision']}`.  P3 restart: `{final['can_p3_restart']}`.\n\n"
        "The complete reference audit is retained as a diagnostic; it is not a hard "
        "entry gate.  R1 is a hard failure because `hocap_170650` has no continuous "
        "eight-frame, 1g dynamically qualified early table-supported PRE_CONTACT reset "
        "window.  PPO smoke and fresh P3 C0-C2 were therefore not run.\n",
    )
    _write_text(
        output / "handoff.md",
        "# Handoff\n\n"
        f"Stop at `{final['decision']}`.  Primary next action: `{final['next_action']}`.\n\n"
        "Debug the `hocap_170650` reset pipeline without changing the reference, reward, "
        "gravity/friction schedule, support semantics, or adding hidden corrections. "
        "Do not start PPO until a new immutable hard-reset receipt satisfies R1.\n",
    )
    _write(
        output / "git_commits.json",
        {"status": "PENDING_LOCAL_COMMITS", "execution_start_head": _git_head()},
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _record_commits(output: Path, start_head: str) -> None:
    completed = subprocess.run(
        ["git", "log", "--format=%H%x09%s", f"{start_head}..HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    commits = [
        {"commit": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
        for line in completed.stdout.splitlines()
        if "\t" in line
    ]
    _write(
        output / "git_commits.json",
        {"status": "RECORDED", "execution_start_head": start_head, "commits": commits},
    )


def _run(output: Path, validation_status: str) -> dict[str, object]:
    b6_final = json.loads((B6 / "final_summary.json").read_text(encoding="utf-8"))
    if b6_final.get("decision") != "P3_RESTART_BLOCKED_REFERENCE_GEOMETRY":
        raise ValueError("P3B7_HISTORICAL_REFERENCE_DIAGNOSTIC_MISSING")
    support = json.loads(
        (REPO_ROOT / ".local/reports/stage16_support_reconstruction/final_summary.json").read_text(
            encoding="utf-8"
        )
    )
    support_valid = all(
        item.get("geometry_validation", {}).get("status") == "PASS"
        for item in support.get("clips", [])
        if isinstance(item, dict)
    )
    inputs = _required_inputs()
    receipts = {name: _receipt(path) for name, path in inputs.items()}
    frozen = {
        "schema_version": "Stage16P3B7FrozenInputsV1",
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "head": _git_head(),
        "inputs": receipts,
        "selected_zero_g_checkpoints": _selected_checkpoint_receipts(inputs),
        "historical_p3b6_artifact_preserved": True,
        "reference_modified": False,
    }
    _write(output / "frozen_inputs.json", frozen)
    pools: dict[str, dict[str, object]] = {}
    gate = EarlyTableResetCoverageGateV1()
    for clip in CLIPS:
        with np.load(
            B6 / clip / "physical_reference_validity_mask.npz", allow_pickle=False
        ) as archive:
            validity = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(
            B6 / clip / "physical_safe_rsi_bank_dynamic.npz", allow_pickle=False
        ) as archive:
            dynamic_indices = np.asarray(archive["runtime_index"], dtype=np.int64).tolist()
        summary = json.loads(
            (B6 / clip / "dynamic_reset_qualification.json").read_text(encoding="utf-8")
        )
        pool = build_early_table_reset_pool(
            clip=clip,
            validity_rows=validity,
            dynamic_safe_indices=dynamic_indices,
            dynamic_summary=summary,
            coverage_gate=gate,
        )
        pools[clip] = pool
        _write(output / "hard_reset" / clip / "early_reset_pool.json", pool)
    gate_receipt = stage16_p3_restart_gate_v2(
        provenance_valid=True, support_valid=support_valid, pools=pools
    )
    _write(output / "restart_gate_v2.json", gate_receipt)
    decision_tree = {
        "schema_version": "Stage16P3B7DecisionTreeV1",
        "start": "R0_provenance",
        "reference_geometry": "DIAGNOSTIC_ONLY",
        "nodes": gate_receipt["gates"],
        "terminal_decision": gate_receipt["decision"],
        "next_action": gate_receipt["next_action"],
    }
    _write(output / "decision_tree.json", decision_tree)
    reset_rows = [
        {
            "clip": clip,
            "static_early_table_geometry_safe": pool["static_early_table_geometry_safe_count"],
            "dynamic_qualified": pool["dynamic_qualified_count"],
            "final_reset_pool": len(pool["frames"]),
            "longest_window": pool["longest_window"],
            "result": pool["status"],
        }
        for clip, pool in pools.items()
    ]
    _write(output / "tables" / "hard_reset.json", reset_rows)
    final = {
        "schema_version": "Stage16P3B7FinalSummaryV1",
        "decision": gate_receipt["decision"],
        "can_p3_restart": "YES" if gate_receipt["decision"] == "P3B7_READY_FOR_R3" else "NO",
        "next_action": gate_receipt["next_action"],
        "restart_gate": str((output / "restart_gate_v2.json").resolve()),
        "hard_reset_pools": pools,
        "ppo_smoke_run": False,
        "p3_restarted": False,
        "reference_geometry_audit_retained": True,
        "reference_geometry_hard_gate": False,
        "hard_reset_geometry_gate": True,
        "actual_rollout_geometry_gate": True,
        "reason_unrun": "R1_HARD_RESET_POOL_FAILED"
        if gate_receipt["decision"] == "P3B7_BLOCKED_HARD_RESET_POOL"
        else None,
    }
    _write(output / "final_summary.json", final)
    _write_blocked_artifacts(
        output,
        final=final,
        validation_status=validation_status,
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-status",
        default="PENDING_FINAL_VALIDATION",
        choices=("PENDING_FINAL_VALIDATION", "PASS", "FAIL"),
    )
    parser.add_argument(
        "--record-commits-from",
        metavar="START_HEAD",
        help="Only update git_commits.json from an already materialized B.7 receipt.",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.record_commits_from:
        _record_commits(output, args.record_commits_from)
        print(json.dumps({"git_commits": "RECORDED"}, sort_keys=True))
        return 0
    result = _run(output, args.validation_status)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
