#!/usr/bin/env python3
"""Finalize Phase 1/2 machine-readable handoff without mutating frozen inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    phase1 = root / "phase1"
    phase2 = root / "phase2"
    refs = {
        clip: _read(phase1 / f"reference_terminal_twist_{clip}.json")
        for clip in ("170105", "170650")
    }
    actual = {
        clip: _read(phase1 / f"actual_terminal_drift_{clip}.json") for clip in ("170105", "170650")
    }
    evaluations = {
        clip: _read(phase2 / f"hocap_{clip}_summary.json") for clip in ("170105", "170650")
    }
    for clip in ("170105", "170650"):
        report_path = phase1 / f"rsi_state_quality_{clip}.json"
        if report_path.exists():
            continue
        status = {
            "schema_version": "RSIStateQualityAuditV1",
            "clip": f"hocap_{clip}",
            "status": "NOT_RUN_ISAAC_PROCESS_ABORTED_AFTER_SCENE_SETUP",
            "attempt": (
                "fresh 248-environment zero-residual diagnostic exited after scene setup "
                "without emitting a report"
            ),
            "no_policy_training": True,
            "consequence": (
                "RSI state-quality fractions and gravity-risk fraction are unknown; this "
                "cannot authorize an RSI conclusion."
            ),
        }
        _write(report_path, status)
    decisions: dict[str, dict[str, Any]] = {}
    for clip in ("170105", "170650"):
        terminal = [row for row in actual[clip]["episodes"] if not row["terminal_stability"]]
        residual = [float(row["residual_v_norm_mps"]["terminal"]) for row in terminal]
        decisions[clip] = {
            "primary_cause": "UNKNOWN",
            "secondary_causes": [
                "SUPPORT_MISSING",
                "ZERO_GRAVITY_VELOCITY_PERSISTENCE_UNCONFIRMED",
            ],
            "confidence": (
                "high for reference-contract invalidity; low for a unique terminal-drift mechanism"
            ),
            "reference_twist_valid": refs[clip]["reference_twist_valid"],
            "terminal_stability_failures": len(terminal),
            "terminal_residual_v_mps": {
                "mean": sum(residual) / max(len(residual), 1),
                "count": len(residual),
            },
            "classification": ["TERMINAL_REFERENCE_TWIST", "SUPPORT_MISSING", "UNKNOWN"],
            "evidence": [
                refs[clip]["status"],
                refs[clip]["terminal_motion"],
                "R7 has no hidden force/object rollout write/wrist teleport.",
            ],
        }
        _write(phase1 / f"attribution_{clip}.json", decisions[clip])
    counterfactual_path = phase1 / "counterfactuals.json"
    if counterfactual_path.exists():
        counterfactual = _read(counterfactual_path)
        counterfactual["status"] = "NOT_RUN_ISAAC_DIAGNOSTIC_ABORT_AFTER_SCENE_SETUP"
        counterfactual["conclusion"] = (
            "No controlled-gravity or support-plane result is claimed. The Isaac diagnostic "
            "process initialized the scene then exited without its requested report."
        )
        _write(counterfactual_path, counterfactual)
    decision = {
        "decision": "PHASE3_OBJECT_TWIST_REWARD_NOT_RECOMMENDED",
        "reference_twist_valid": False,
        "terminal_twist_primary": "UNKNOWN_REFERENCE_TWIST_CONTRACT_INVALID",
        "support_limitation": "SUPPORT_UNKNOWN in source/reference; SUPPORT_ABSENT in simulator",
        "rsi_limitation": (
            "RSI state-quality physical diagnostic not completed after Isaac process abort"
        ),
        "contact_limitation": (
            "terminal contact exists in all frozen R7 episodes; contact is not absent"
        ),
        "geometry_limitation": (
            "absolute hand-object safety gates pass; legacy source-relative geometry remains "
            "a separate failed diagnostic"
        ),
        "evidence": [
            (
                "Both factor-8 references fail pose finite-difference consistency for stored "
                "linear and world angular twist."
            ),
            "Both stored references have nonzero terminal twist, including upward z component.",
            (
                "Object-twist reward target is invalid before any causal attribution or PPO "
                "reward change."
            ),
        ],
        "required_prerequisites": [
            "Repair/reference-version factor-8 pose/timestamp/twist interpolation together.",
            (
                "Re-run finite-difference and angular-frame sanity checks before treating "
                "twist as a target."
            ),
            "Complete fresh bounded RSI state-quality and gravity counterfactual diagnostics.",
        ],
    }
    _write(root / "phase3_entry_decision.json", decision)
    summary = {
        "schema_version": "Stage16DPhase1Phase2FinalSummaryV1",
        "phase1_status": "PARTIAL_REFERENCE_TWIST_INVALID__RSI_RUNTIME_DIAGNOSTIC_ABORTED",
        "phase2_status": "COMPLETE_FROM_FROZEN_R7_ALL_REPLICA_TRACES",
        "phase3": decision,
        "baseline": {clip: evaluations[clip]["aggregate"] for clip in evaluations},
    }
    _write(root / "final_summary.json", summary)
    phase1_summary = "# Stage 16-D Phase 1 summary\n\n"
    phase1_summary += (
        "Both factor-8 references fail the required pose-to-twist finite-difference sanity "
        "check. The stored reference terminal motion is nonzero, but cannot be used as a "
        "validated target. The formal R7 traces preserve causal-action, no-hidden-force, and "
        "no-rollout-state-write evidence. RSI physical state-quality diagnostics aborted after "
        "Isaac scene setup, so their state fractions remain unclaimed.\n"
    )
    (phase1 / "phase1_summary.md").write_text(phase1_summary, encoding="utf-8")
    final_md = "# Stage 16-D Phase 1 / Phase 2 Handoff\n\n"
    final_md += (
        "## Decision\n\n`PHASE3_OBJECT_TWIST_REWARD_NOT_RECOMMENDED`. Reference twist is not a "
        "valid reward target because factor-8 pose finite differences disagree with the "
        "stored linear and angular twist.\n\n"
    )
    final_md += "## Evaluation Suite V2\n\n"
    for clip in ("170105", "170650"):
        aggregate = evaluations[clip]["aggregate"]
        final_md += (
            f"- hocap_{clip}: SR_kinematic "
            f"{aggregate['kinematic_success']['pass_count']}/20; SR_physics "
            f"{aggregate['physics_success']['pass_count']}/20; SR_qualified "
            f"{aggregate['qualified_success']['pass_count']}/20.\n"
        )
    final_md += (
        "\n## Stop boundary\n\nNo reward modification, PPO training, contact reward, support "
        "curriculum, or external guidance was run.\n"
    )
    (root / "final_summary.md").write_text(final_md, encoding="utf-8")
    (root / "handoff.md").write_text(final_md, encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
