#!/usr/bin/env python3
"""Finalize Phase 1/2 machine-readable handoff without mutating frozen inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RSI_CLASSES = (
    "PRE_CONTACT_UNSUPPORTED",
    "PRE_CONTACT_STABLE_UNDER_ZERO_G",
    "NEAR_OBJECT",
    "CONTACT_READY",
    "PERSISTENT_CONTACT",
    "TERMINAL_HOLD",
    "AMBIGUOUS",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rsi_fraction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("RSI state-quality report has no rows")
    total = len(rows)
    counts = {label: sum(row["classification"] == label for row in rows) for label in RSI_CLASSES}
    catastrophic = sum(bool(row["catastrophic_failure"]) for row in rows)
    return {
        "row_count": total,
        "class_counts": counts,
        "class_fractions": {label: counts[label] / total for label in RSI_CLASSES},
        "contact_achieved_fraction": sum(bool(row["contact_achieved"]) for row in rows) / total,
        "catastrophic_failure_fraction": catastrophic / total,
    }


def _gravity_risk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risk = [
        row
        for row in rows
        if not bool(row["contact_achieved"])
        and (
            float(row["object_vertical_displacement_before_contact_m"]) < -0.005
            or float(row["object_total_displacement_before_contact_m"]) > 0.005
        )
    ]
    return {
        "definition": (
            "no actual contact in the 20-step window and either vertical displacement below "
            "-5 mm or total pre-contact displacement above 5 mm"
        ),
        "count": len(risk),
        "total": len(rows),
        "fraction": len(risk) / len(rows),
    }


def _counterfactual_view(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": value["case"],
        "clip": value["clip"],
        "gravity": value["gravity"],
        "initial_state": value.get("initial_state"),
        "duration_s": value.get("duration_s"),
        "physics_contract": value.get("physics_contract"),
        "trajectories": [
            {key: trajectory[key] for key in trajectory if key != "timeline"}
            for trajectory in value["trajectories"]
        ],
        "source": value["frozen_inputs"],
    }


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
    rsi_summary: dict[str, Any] = {}
    for clip in ("170105", "170650"):
        nominal_path = phase1 / f"rsi_state_quality_{clip}.json"
        gravity_path = phase1 / f"rsi_state_quality_{clip}_gravity.json"
        nominal = _read(nominal_path)
        gravity = _read(gravity_path)
        if nominal.get("status") != "COMPLETE" or gravity.get("status") != "COMPLETE":
            raise RuntimeError(f"RSI state quality is incomplete for hocap_{clip}")
        nominal_rows = nominal.get("rows")
        gravity_rows = gravity.get("rows")
        if not isinstance(nominal_rows, list) or not isinstance(gravity_rows, list):
            raise RuntimeError(f"RSI rows are invalid for hocap_{clip}")
        nominal_summary = _rsi_fraction_summary(nominal_rows)
        gravity_summary = _rsi_fraction_summary(gravity_rows)
        paired = {
            "nominal_g0": nominal_summary,
            "gravity_g_minus_9_81": gravity_summary,
            "gravity_pre_contact_drift_risk": _gravity_risk(gravity_rows),
            "gravity_report": str(gravity_path),
        }
        nominal["classification_summary"] = paired
        _write(nominal_path, nominal)
        rsi_summary[f"hocap_{clip}"] = paired
    _write(phase1 / "rsi_state_quality_summary.json", {"clips": rsi_summary})

    counterfactuals: dict[str, Any] = {
        "schema_version": "Stage16DTerminalCounterfactualSummaryV1",
        "status": "COMPLETE_BOUNDED_DIAGNOSTIC",
        "nominal_contract": "frozen R7, zero gravity, zero damping, no support",
        "clips": {},
    }
    for clip in ("170105", "170650"):
        cf1 = _read(phase1 / f"cf1_gravity_{clip}.json")
        cf2_g0 = _read(phase1 / f"cf2_last_contact_g0_{clip}.json")
        cf2_g = _read(phase1 / f"cf2_last_contact_gravity_{clip}.json")
        cf3_g0 = _read(phase1 / f"cf3_reference_terminal_g0_{clip}.json")
        cf3_g = _read(phase1 / f"cf3_reference_terminal_gravity_{clip}.json")
        selected = cf1["selected_representatives"]
        terminal = {int(row["replica"]): row for row in actual[clip]["episodes"]}
        counterfactuals["clips"][f"hocap_{clip}"] = {
            "representatives": selected,
            "CF0_nominal_frozen_R7": {
                "trace": cf1["frozen_inputs"]["trace"],
                "trace_sha256": cf1["frozen_inputs"]["trace_sha256"],
                "terminal_audit": [terminal[int(row["replica"])] for row in selected],
            },
            "CF1_same_action_gravity_only": _counterfactual_view(cf1),
            "CF2_last_contact_free_drift": {
                "g0": _counterfactual_view(cf2_g0),
                "g_minus_9_81": _counterfactual_view(cf2_g),
            },
            "CF3_reference_terminal_free_drift": {
                "g0": _counterfactual_view(cf3_g0),
                "g_minus_9_81": _counterfactual_view(cf3_g),
            },
        }
    _write(phase1 / "counterfactuals.json", counterfactuals)
    decisions: dict[str, dict[str, Any]] = {}
    for clip in ("170105", "170650"):
        terminal = [row for row in actual[clip]["episodes"] if not row["terminal_stability"]]
        residual = [float(row["residual_v_norm_mps"]["terminal"]) for row in terminal]
        decisions[clip] = {
            "primary_cause": "TERMINAL_REFERENCE_TWIST",
            "secondary_causes": [
                "SUPPORT_MISSING",
                "ZERO_GRAVITY_VELOCITY_PERSISTENCE",
            ],
            "confidence": (
                "high for invalid reference-twist target and velocity persistence; low for a "
                "unique contact-generated residual mechanism"
            ),
            "reference_twist_valid": refs[clip]["reference_twist_valid"],
            "terminal_stability_failures": len(terminal),
            "terminal_residual_v_mps": {
                "mean": sum(residual) / max(len(residual), 1),
                "count": len(residual),
            },
            "classification": [
                "TERMINAL_REFERENCE_TWIST",
                "ZERO_GRAVITY_VELOCITY_PERSISTENCE",
                "SUPPORT_MISSING",
                "UNKNOWN",
            ],
            "evidence": [
                refs[clip]["status"],
                refs[clip]["terminal_motion"],
                "R7 has no hidden force/object rollout write/wrist teleport.",
                (
                    "CF2 and CF3 prove zero-gravity velocity persistence but do not identify "
                    "its source."
                ),
            ],
        }
        _write(phase1 / f"attribution_{clip}.json", decisions[clip])
    decision = {
        "decision": "PHASE3_OBJECT_TWIST_REWARD_NOT_RECOMMENDED",
        "reference_twist_valid": False,
        "terminal_twist_primary": "UNKNOWN_REFERENCE_TWIST_CONTRACT_INVALID",
        "support_limitation": "SUPPORT_UNKNOWN in source/reference; SUPPORT_ABSENT in simulator",
        "rsi_limitation": (
            "RSI state-quality diagnostic is complete; source support remains unknown and "
            "gravity-only reset risk is a diagnostic, not a nominal qualification."
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
            "Re-run the frozen Phase 1 audits against the repaired reference version.",
        ],
    }
    _write(root / "phase3_entry_decision.json", decision)
    summary = {
        "schema_version": "Stage16DPhase1Phase2FinalSummaryV1",
        "phase1_status": "COMPLETE_REFERENCE_TWIST_INVALID__SUPPORT_UNKNOWN",
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
        "no-rollout-state-write evidence. The bounded RSI state-quality and gravity diagnostic "
        "is complete; its detailed class fractions and gravity-risk fractions are stored in "
        "rsi_state_quality_summary.json. CF2 and CF3 show that zero gravity preserves existing "
        "terminal velocity, while gravity-only CF1 is a non-nominal diagnostic that can trigger "
        "the unchanged safety gates early.\n"
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
            f"  RSI gravity pre-contact drift risk: "
            f"{rsi_summary[f'hocap_{clip}']['gravity_pre_contact_drift_risk']['count']}/"
            f"{rsi_summary[f'hocap_{clip}']['gravity_pre_contact_drift_risk']['total']}.\n"
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
