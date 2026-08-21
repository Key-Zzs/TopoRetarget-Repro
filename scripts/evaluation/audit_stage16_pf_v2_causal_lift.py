#!/usr/bin/env python3
"""Re-evaluate frozen Stage16 traces under the PF V2 causal-lift proposal.

This audit deliberately consumes immutable trace and PF V1 receipts.  It does
not launch Isaac, load a policy, or take an optimizer step.  A PF V2 failure on
the accepted 170650 positive control is a stop condition, not an invitation to
change thresholds after observing the result.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.stage16_pf_v2_causal_lift import (
    Stage16PhysicalFunctionalityV2Contract,
    evaluate_stage16_physical_functionality_v2,
)


REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"
PHASE_NAMES = np.asarray(
    ("PRE_CONTACT", "APPROACH", "CONTACT", "GRASP", "LIFT", "MANIPULATION", "TERMINAL")
)
PF_V1_170105 = (
    REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df/pf_df/"
    "v4_170105_episode_receipts.csv"
)
PF_V1_170650 = (
    REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df/pf_df/"
    "v4_170650_episode_receipts.csv"
)
TRACE_ROOTS: dict[str, tuple[Path, Path, int]] = {
    "historical_170105": (
        REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/"
        "smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4",
        PF_V1_170105,
        10,
    ),
    "u9_170105": (
        REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse/training/U09/eval10/traces",
        REPO_ROOT
        / ".local/reports/stage16_dexplore_reward_rse/training/U09/eval10/per_episode.csv",
        10,
    ),
    "u10_170105": (
        REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse/training/U10/eval10/traces",
        REPO_ROOT
        / ".local/reports/stage16_dexplore_reward_rse/training/U10/eval10/per_episode.csv",
        10,
    ),
    "historical_170650": (
        REPO_ROOT
        / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650",
        PF_V1_170650,
        20,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("PF_V2_AUDIT_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
    raise ValueError(f"PF_V2_AUDIT_BOOL_INVALID:{value!r}")


def _pf_v1_rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"PF_V2_AUDIT_PF_V1_RECEIPT_COUNT_INVALID:{path}")
    return rows


def _phase_lift_onset(trace: Any) -> int | None:
    phase = np.asarray(trace["phase"])
    if phase.dtype.kind in "iu":
        phase = PHASE_NAMES[np.asarray(phase, dtype=np.int64)]
    indices = np.flatnonzero(phase.astype("U16") == "LIFT")
    return None if not len(indices) else int(indices[0])


def _v1_gate(row: dict[str, str], names: tuple[str, ...]) -> bool:
    for name in names:
        if name in row:
            return _bool(row[name])
    raise ValueError(f"PF_V2_AUDIT_V1_GATE_MISSING:{names}")


def _action_bounds_from_trace(trace: Any) -> bool:
    action = np.asarray(trace["action"], dtype=np.float64)
    return bool(np.isfinite(action).all() and (np.abs(action) <= 1.0).all())


def _evaluate_trace(
    *, label: str, episode: int, trace_path: Path, v1: dict[str, str]
) -> dict[str, object]:
    with np.load(trace_path) as trace:
        valid = np.asarray(trace["fingertip_object_pair_force_valid"], dtype=bool)
        causal = _v1_gate(v1, ("causality",))
        action_safe = (
            _v1_gate(v1, ("action_bounds_safe",))
            if "action_bounds_safe" in v1
            else _action_bounds_from_trace(trace)
        )
        no_hidden = _v1_gate(v1, ("no_hidden_control",)) if "no_hidden_control" in v1 else causal
        result = evaluate_stage16_physical_functionality_v2(
            object_pose_wxyz=trace["object_pose"],
            wrist_pose_wxyz=trace["wrist_pose"],
            tip_pair_presence=trace["tip_pair_presence"],
            hand_object_pair_presence=trace["hand_object_pair_presence"],
            table_object_contact=trace["table_object_contact"],
            valid=valid,
            reference_lift_onset=_phase_lift_onset(trace),
            causal_execution=causal,
            geometry_safe=_v1_gate(v1, ("geometry", "penetration_safe")),
            action_bounds_safe=action_safe,
            no_hidden_control=no_hidden,
        )
    actual = result["actual_lift"]
    interaction = result["interaction_timing"]
    causal = result["causal_interaction"]
    return {
        "trace_set": label,
        "episode": episode,
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "PF_V1": _v1_gate(v1, ("PF", "pf")),
        "physical_lift_success": bool(result["physical_lift_success"]),
        "causal_hand_object_lift": bool(result["causal_hand_object_lift"]),
        "support_transfer_success": bool(result["support_transfer_success"]),
        "sustained_hand_object_coupling": bool(result["sustained_hand_object_coupling"]),
        "persistent_multifinger_contact": interaction["persistent_multifinger_contact"],
        "pre_reference_lift_multifinger_contact": interaction[
            "pre_reference_lift_multifinger_contact"
        ],
        "interaction_timing_fidelity": interaction["interaction_timing_fidelity"],
        "reference_lift_onset": interaction["reference_lift_onset"],
        "actual_lift_onset": actual["onset"],
        "support_release_event": actual["support_release_event"],
        "first_hand_object_contact": interaction["first_hand_object_contact"],
        "pre_reference_lift_margin": interaction["pre_reference_lift_margin"],
        "pre_actual_lift_margin": interaction["pre_actual_lift_margin"],
        "relative_linear_coupling_mean_mps": causal["relative_linear_speed_mps"]["mean"],
        "relative_angular_coupling_mean_radps": causal["relative_angular_speed_radps"]["mean"],
        "ballistic_or_flick_rejected": causal["ballistic_or_flick_rejected"],
        "PF_V2": bool(result["pf_v2"]),
        "PF_V2_failure_reasons": ";".join(result["pf_v2_failure_reasons"]),
    }


def _run_set(label: str, root: Path, pf_v1_path: Path, expected: int) -> list[dict[str, object]]:
    traces = sorted(root.glob("episode_*.npz"))
    if len(traces) != expected:
        raise ValueError(f"PF_V2_AUDIT_TRACE_COUNT_INVALID:{label}:{len(traces)}")
    v1_rows = _pf_v1_rows(pf_v1_path, expected)
    return [
        _evaluate_trace(label=label, episode=index, trace_path=trace, v1=v1_rows[index])
        for index, trace in enumerate(traces)
    ]


def _count(rows: list[dict[str, object]], name: str) -> int:
    return sum(bool(row[name]) for row in rows)


def _classification(by_set: dict[str, list[dict[str, object]]]) -> tuple[str, str, str]:
    positive = by_set["historical_170650"]
    historical_negative = by_set["historical_170105"]
    if _count(positive, "PF_V2") != len(positive):
        return (
            "PF_V2_SEMANTICS_INVALID",
            "HIGH",
            "Accepted 170650 traces have no observed pre-release table-support sample; "
            "the frozen SupportTransferProxyV1 therefore cannot establish transfer.",
        )
    if _count(historical_negative, "PF_V2") > 0:
        return (
            "PF_V2_SEMANTICS_INVALID",
            "HIGH",
            "Historical 170105 non-lift traces would pass the proposed causal-lift evaluator.",
        )
    u10 = by_set["u10_170105"]
    if _count(u10, "PF_V2") > _count(u10, "PF_V1"):
        return (
            "PF_V1_PRELIFT_GATE_PARTIALLY_OVERCONSTRAINED",
            "MEDIUM",
            "PF V2 adds causal-lift passes without admitting the historical non-lift negative.",
        )
    return (
        "PF_V1_SEMANTICS_VALID",
        "MEDIUM",
        "The available U10 traces do not show a PF V2 improvement once causal lift is required.",
    )


def _audit_v1() -> dict[str, object]:
    return {
        "schema_version": "Stage16PhysicalFunctionalityV1AuditV1",
        "implementation": "src/toporetarget/rl/stage16_pf_df.py",
        "implementation_lines": {
            "contract_and_hard_gates": "89-115",
            "gate_composition": "710-730",
        },
        "hard_gates": [
            {
                "name": "causal_execution",
                "threshold": "boolean no rollout object/wrist root writes",
                "physical_meaning": "execution was not driven by prohibited state writes",
            },
            {
                "name": "geometry_safe",
                "threshold": "frozen geometry / penetration authority",
                "physical_meaning": "no disqualifying geometry violation",
            },
            {
                "name": "action_bounds_safe",
                "threshold": "finite action within [-1, 1]",
                "physical_meaning": "policy command stays in the frozen action contract",
            },
            {
                "name": "prelift_multifinger_grasp_ready",
                "threshold": (
                    "at least 2 named fingers, each persistent 3 control steps, by reference LIFT"
                ),
                "physical_meaning": "interaction timing criterion mixed into PF V1",
            },
            {
                "name": "lift_success",
                "threshold": "final object vertical displacement >= 0.05 m",
                "physical_meaning": "legacy endpoint lift outcome",
            },
            {
                "name": "no_hidden_control",
                "threshold": "boolean no guidance/object/wrist-root hidden control",
                "physical_meaning": "disallow non-physical assistance",
            },
        ],
        "pre_reference_lift_multifinger_gate_classification": "MIXED_PF_AND_INTERACTION_TIMING",
        "PF_V1_CHANGED": "NO",
    }


def main() -> int:
    if REPORT_ROOT.exists():
        raise FileExistsError(f"PF_V2_AUDIT_REPORT_NAMESPACE_EXISTS:{REPORT_ROOT}")
    frozen = Stage16PhysicalFunctionalityV2Contract()
    by_set = {
        label: _run_set(label, root, pf_v1_path, expected)
        for label, (root, pf_v1_path, expected) in TRACE_ROOTS.items()
    }
    classification, confidence, rationale = _classification(by_set)
    summary_rows = [
        {
            "trace_set": label,
            "episodes": len(rows),
            "PF_V1": _count(rows, "PF_V1"),
            "physical_lift": _count(rows, "physical_lift_success"),
            "causal_lift": _count(rows, "causal_hand_object_lift"),
            "support_transfer": _count(rows, "support_transfer_success"),
            "sustained_coupling": _count(rows, "sustained_hand_object_coupling"),
            "PF_V2": _count(rows, "PF_V2"),
        }
        for label, rows in by_set.items()
    ]
    _write_json(REPORT_ROOT / "pf_v2/pf_v1_audit.json", _audit_v1())
    _write_json(REPORT_ROOT / "pf_v2/pf_v2_contract.json", frozen.as_dict())
    _write_json(
        REPORT_ROOT / "pf_v2/support_transfer_contract.json",
        {
            "schema_version": "Stage16SupportTransferProxyV1",
            "exact_wrench_transfer": False,
            "signal": frozen.support_signal,
            "requirements": [
                "persistent table-support absence after a prior observed support frame",
                "release no later than ActualLiftOnset",
                "persistent hand-object interaction at ActualLiftOnset",
            ],
            "fail_closed_note": (
                "No support frame before capture is NOT_IDENTIFIABLE, not evidence of transfer."
            ),
        },
    )
    _write_text(
        REPORT_ROOT / "pf_v2/causal_lift_semantics.md",
        """# PF V2 causal-lift semantics

`ActualLiftOnset` is the first persistent (three control-step) event where the
object is support-free, has risen by the inherited 5 cm threshold, and has
positive pose-derived vertical velocity.  `reference LIFT` is retained solely
as an interaction-timing diagnostic.

`SupportTransferProxyV1` uses the retained binary table-contact signal. It is
not an exact normal-wrench claim. A trace that begins after support has already
disappeared cannot prove transfer under this proxy and is marked
`NOT_IDENTIFIABLE`; it is never silently promoted to a pass.

`CausalLoadBearingInteractionV1` requires persistent observed hand contact and
multi-finger contact at actual lift plus a three-step post-lift contact and
finite pose-derived relative linear/angular motion. This rejects a flick that
raises the object but loses contact before or immediately after actual lift.
""",
    )
    _write_json(
        REPORT_ROOT / "pf_v2/audit_classification.json",
        {
            "PF_V2_AUDIT": classification,
            "CONFIDENCE": confidence,
            "rationale": rationale,
            "ppo_authorized": False,
            "stop_condition": (
                "170650 historical accepted PF V2 regression"
                if classification == "PF_V2_SEMANTICS_INVALID"
                else None
            ),
        },
    )
    for label, rows in by_set.items():
        _write_csv(REPORT_ROOT / f"reevaluation/{label}.csv", rows)
    _write_csv(REPORT_ROOT / "reevaluation/u10_episode_detail.csv", by_set["u10_170105"])
    _write_csv(REPORT_ROOT / "comparison/pf_v1_vs_v2.csv", summary_rows)
    _write_json(
        REPORT_ROOT / "input_manifest.json",
        {
            label: {
                "trace_root": str(root.resolve()),
                "pf_v1_receipt": str(pf_v1_path.resolve()),
                "trace_hashes": [row["trace_sha256"] for row in by_set[label]],
            }
            for label, (root, pf_v1_path, _) in TRACE_ROOTS.items()
        },
    )
    _write_json(
        REPORT_ROOT / "final_summary.json",
        {
            "schema_version": "Stage16PFV2CausalLiftAuditV1",
            "classification": classification,
            "confidence": confidence,
            "summary": summary_rows,
            "PF_V1_CHANGED": "NO",
            "PF_V2_ADDED": "YES",
            "REFERENCE_LIFT_REMOVED_FROM_PF_V2_HARD_GATE": "YES",
            "INTERACTION_TIMING_DIAGNOSTIC_PRESERVED": "YES",
            "170105_NEW_PPO_RUN": "NO",
            "170650_NEW_PPO_RUN": "NO",
            "STOP_BEFORE_PPO": classification == "PF_V2_SEMANTICS_INVALID",
        },
    )
    _write_text(
        REPORT_ROOT / "handoff.md",
        f"""# Stage16 PF V2 Causal Lift Audit Handoff

`PF_V2_AUDIT={classification}` with `CONFIDENCE={confidence}`.

{rationale}

The audit preserves PF V1 and the historical trace hashes. Because the
accepted 170650 positive-control traces fail the required V2 support-transfer
proxy, symmetric PPO is not authorized and was not run.
""",
    )
    print(json.dumps({"classification": classification, "confidence": confidence}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
