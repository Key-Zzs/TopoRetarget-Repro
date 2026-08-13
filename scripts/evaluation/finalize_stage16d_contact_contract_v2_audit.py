#!/usr/bin/env python3
"""Finalize the read-only Stage 16-D ReferenceContactContractV2 audit.

The script consumes the frozen V3 inputs, V2 evidence materialization, and
newly captured all-hand-body Formal20 telemetry.  It never changes the
historical V3 3 cm mask, a checkpoint, a policy, or an IsaacLab environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.reference_contact_contract import FINGER_ORDER  # noqa: E402

CLIPS = ("hocap_170105", "hocap_170650")
TIP_BODIES = {
    "thumb": "r_thumb_distal",
    "index": "r_index_finger_distal",
    "middle": "r_middle_finger_distal",
    "ring": "r_ring_finger_distal",
    "pinky": "r_pinky_distal",
}
CONTROL_DT_S = 0.05
PERSISTENCE = 3
EPS = 1.0e-8


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"CONTACT_AUDIT_R2_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        result.append((start, end))
        start = end
    return result


def _persistent(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for start, end in _runs(mask):
        if end - start >= PERSISTENCE:
            result[start:end] = True
    return result


def _longest(mask: np.ndarray) -> int:
    return max((end - start for start, end in _runs(mask)), default=0)


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.count_nonzero(denominator))
    return None if count == 0 else float(np.count_nonzero(numerator & denominator) / count)


def _stat(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("CONTACT_AUDIT_R2_NONFINITE_STATISTIC")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_v2(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        needed = {
            "strong_contact_expected",
            "proximity_only",
            "no_contact_expected",
            "source_contact_supported",
            "topology_contact_supported",
            "reference_distance_m",
            "reference_evidence_class",
            "reference_evidence_source",
            "historical_v3_primary_mask",
            "reference_contact_evidence_conflict",
            "finger_order",
        }
        missing = sorted(needed.difference(archive.files))
        if missing:
            raise ValueError(f"CONTACT_AUDIT_R2_V2_FIELDS_MISSING:{missing}")
        arrays = {name: np.asarray(archive[name]) for name in needed}
    if arrays["strong_contact_expected"].shape != (321, 5):
        raise ValueError("CONTACT_AUDIT_R2_V2_SHAPE_INVALID")
    if tuple(str(item) for item in arrays["finger_order"].tolist()) != FINGER_ORDER:
        raise ValueError("CONTACT_AUDIT_R2_FINGER_ORDER_INVALID")
    return arrays


def _load_trace(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        needed = {
            "replica_hand_object_pair_force_world",
            "replica_hand_object_pair_presence",
            "replica_hand_object_pair_force_valid",
            "replica_contact_reward",
            "replica_reward_total",
            "replica_object_twist",
            "replica_object_twist_reference",
            "replica_reference_contact_mask",
            "hand_body_names",
            "hand_body_indices",
            "hand_body_groups",
            "hand_palm_mapping",
            "requested_clip",
        }
        missing = sorted(needed.difference(archive.files))
        if missing:
            raise ValueError(f"CONTACT_AUDIT_R2_TRACE_FIELDS_MISSING:{missing}")
        arrays = {name: np.asarray(archive[name]) for name in needed}
    force = arrays["replica_hand_object_pair_force_world"]
    presence = arrays["replica_hand_object_pair_presence"].astype(bool)
    valid = arrays["replica_hand_object_pair_force_valid"].astype(bool)
    if force.shape != (321, 20, 21, 3) or presence.shape != (321, 20, 21):
        raise ValueError("CONTACT_AUDIT_R2_FULL_PAIR_SHAPE_INVALID")
    if valid.shape != (321, 20) or valid[0].any() or not valid[1:].all():
        raise ValueError("CONTACT_AUDIT_R2_FULL_PAIR_VALIDITY_INVALID")
    if not np.isfinite(force[valid]).all():
        raise ValueError("CONTACT_AUDIT_R2_FULL_PAIR_NONFINITE")
    names = tuple(str(item) for item in arrays["hand_body_names"].tolist())
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError("CONTACT_AUDIT_R2_FULL_PAIR_BODY_MAPPING_INVALID")
    arrays["names"] = names
    arrays["force_magnitude"] = np.linalg.norm(force.astype(np.float64), axis=-1)
    arrays["palm_mapping"] = json.loads(str(arrays["hand_palm_mapping"].item()))
    return arrays


def _manifest(trace_path: Path, qualification_path: Path) -> dict[str, object]:
    qualification = _read_json(qualification_path)
    if qualification.get("status") != "FULL_PAIR_TELEMETRY_QUALIFIED":
        raise ValueError("CONTACT_AUDIT_R2_FULL_PAIR_NOT_QUALIFIED")
    return {
        "schema_version": "FullHandObjectPairTelemetryManifestV1",
        "status": "FULL_PAIR_TELEMETRY_QUALIFIED",
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "qualification": str(qualification_path.resolve()),
        "qualification_sha256": _sha256(qualification_path),
        "diagnostic_only": True,
        "reward_or_policy_effect": "none",
        "frame_zero": "invalid; no post-physics force sample",
        "shape": [321, 20, 21, 3],
        "force_frame": "world",
        "force_units": "N",
    }


def _assert_frozen_replay(output: Path, clip: str, evaluation_path: Path) -> dict[str, object]:
    """Fail closed unless the diagnostic export reused every named Formal20 input."""

    inputs = _read_json(output / "frozen_inputs.json")["clips"][clip]
    evaluation = _read_json(evaluation_path)
    checkpoint = inputs["checkpoint"]["checkpoint"]
    if evaluation.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("CONTACT_AUDIT_R2_CHECKPOINT_REPLAY_DRIFT")
    if evaluation.get("physics_contract_sha256") != inputs["physics"]["physics_contract_sha256"]:
        raise ValueError("CONTACT_AUDIT_R2_PHYSICS_REPLAY_DRIFT")
    seed = inputs["seeds"]
    replay_seed = evaluation.get("seed_set", {})
    if (
        replay_seed.get("identifier") != seed["identifier"]
        or replay_seed.get("frame_zero") != seed["frame_zero_seeds"]
    ):
        raise ValueError("CONTACT_AUDIT_R2_SEED_REPLAY_DRIFT")
    return {
        "checkpoint_sha256": checkpoint["sha256"],
        "physics_contract_sha256": inputs["physics"]["physics_contract_sha256"],
        "formal_seed_identifier": seed["identifier"],
        "formal_frame_zero_seed_count": len(seed["frame_zero_seeds"]),
    }


def _per_finger_rows(v2: dict[str, np.ndarray], trace: dict[str, Any]) -> list[dict[str, Any]]:
    names = trace["names"]
    valid = trace["replica_hand_object_pair_force_valid"].astype(bool)
    presence = trace["replica_hand_object_pair_presence"].astype(bool) & valid[..., None]
    force = trace["force_magnitude"]
    rows: list[dict[str, Any]] = []
    for index, finger in enumerate(FINGER_ORDER):
        tip = names.index(TIP_BODIES[finger])
        actual = presence[:, :, tip]
        primary = v2["historical_v3_primary_mask"][:, index, None] & valid
        strong = v2["strong_contact_expected"][:, index, None] & valid
        missing = strong & ~actual
        persistent_missing = np.stack([_persistent(missing[:, r]) for r in range(20)], axis=1)
        other_tips = [names.index(TIP_BODIES[name]) for name in FINGER_ORDER if name != finger]
        wrist = presence[:, :, names.index("r_wrist")]
        other_tip = presence[:, :, other_tips].any(axis=-1)
        excluded = {names.index("r_wrist"), tip, *other_tips}
        other_body = presence[:, :, [i for i in range(21) if i not in excluded]].any(axis=-1)
        force_values = force[:, :, tip][actual]
        candidate_substitute = other_tip | wrist | other_body
        rows.append(
            {
                "finger": finger,
                "v3_expected_fraction": float(v2["historical_v3_primary_mask"][:, index].mean()),
                "v2_strong_fraction": float(v2["strong_contact_expected"][:, index].mean()),
                "v2_proximity_only_fraction": float(v2["proximity_only"][:, index].mean()),
                "v3_mask_removed_by_v2_fraction": float(
                    (
                        v2["historical_v3_primary_mask"][:, index]
                        & ~v2["strong_contact_expected"][:, index]
                    ).mean()
                ),
                "source_supported_fraction": float(v2["source_contact_supported"][:, index].mean()),
                "actual_tip_contact_fraction": float(actual.sum() / valid.sum()),
                "v3_expected_recall": _rate(actual, primary),
                "v2_strong_recall": _rate(actual, strong),
                "persistent_v2_strong_recall": _rate(
                    actual, _persistent(strong[:, 0])[:, None] & valid
                ),
                "strong_missing_samples": int(missing.sum()),
                "longest_strong_missing_steps": int(
                    max((_longest(missing[:, r]) for r in range(20)), default=0)
                ),
                "persistent_strong_missing_samples": int(persistent_missing.sum()),
                "wrist_base_substitute_fraction": _rate(wrist, missing),
                "other_tip_substitute_fraction": _rate(other_tip, missing),
                "other_non_tip_substitute_fraction": _rate(other_body, missing),
                "any_substitute_fraction": _rate(candidate_substitute, missing),
                "actual_force_mean_n": _stat(force_values)["mean"],
                "actual_force_p95_n": _stat(force_values)["p95"],
                "actual_force_max_n": _stat(force_values)["max"],
                "actual_force_impulse_ns": float(force_values.sum() * CONTROL_DT_S),
                "interpretation": (
                    "REFERENCE_ALLOWED_FLOAT"
                    if not v2["historical_v3_primary_mask"][:, index].any()
                    else "PROXIMITY_MASK_MATERIAL"
                    if not v2["strong_contact_expected"][:, index].any()
                    else "GEOMETRIC_STRONG_EXPECTED_BUT_LONG_FLOAT"
                    if persistent_missing.any()
                    else "GEOMETRIC_STRONG_CONTACT_OBSERVED"
                ),
            }
        )
    return rows


def _body_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    valid = trace["replica_hand_object_pair_force_valid"].astype(bool)
    presence = trace["replica_hand_object_pair_presence"].astype(bool) & valid[..., None]
    force = trace["force_magnitude"]
    rows: list[dict[str, Any]] = []
    for index, (name, group) in enumerate(
        zip(trace["names"], trace["hand_body_groups"], strict=True)
    ):
        active = presence[:, :, index]
        stats = _stat(force[:, :, index][active])
        rows.append(
            {
                "body_index": index,
                "body_name": name,
                "body_group": str(group),
                "actual_contact_fraction": float(active.sum() / valid.sum()),
                "force_mean_n": stats["mean"],
                "force_p95_n": stats["p95"],
                "force_max_n": stats["max"],
                "force_impulse_ns": float(force[:, :, index][active].sum() * CONTROL_DT_S),
            }
        )
    return rows


def _events(v2: dict[str, np.ndarray], trace: dict[str, Any], clip: str) -> dict[str, object]:
    names = trace["names"]
    valid = trace["replica_hand_object_pair_force_valid"].astype(bool)
    presence = trace["replica_hand_object_pair_presence"].astype(bool) & valid[..., None]
    tips = [names.index(TIP_BODIES[finger]) for finger in FINGER_ORDER]
    actual = presence[:, :, tips]
    strong = v2["strong_contact_expected"][:, None, :] & valid[..., None]
    all_hand_contact = presence.any(axis=-1)
    twist_delta = np.linalg.norm(
        trace["replica_object_twist"][:, :, :3] - trace["replica_object_twist_reference"][:, :, :3],
        axis=-1,
    )
    records: list[dict[str, object]] = []
    for replica in range(20):
        missing = strong[:, replica].any(axis=-1) & ~(strong[:, replica] & actual[:, replica]).any(
            axis=-1
        )
        freeflight = missing & ~all_hand_contact[:, replica]
        for start, end in _runs(freeflight):
            if end - start >= PERSISTENCE:
                records.append(
                    {
                        "clip": clip,
                        "replica": replica,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "duration_steps": end - start,
                        "delta_object_linear_velocity_mps": _stat(twist_delta[start:end, replica]),
                        "expected_fingers": [
                            FINGER_ORDER[index]
                            for index in np.flatnonzero(strong[start:end, replica].any(axis=0))
                        ],
                    }
                )
    return {
        "schema_version": "Stage16DReferenceStrongFreeFlightV2",
        "definition": (
            "V2 strong expected contact, no expected tip contact, and no active "
            "hand-body contact for >=3 steps"
        ),
        "events": records,
        "event_count": len(records),
    }


def _compensation(v2: dict[str, np.ndarray], trace: dict[str, Any]) -> dict[str, object]:
    names = trace["names"]
    valid = trace["replica_hand_object_pair_force_valid"].astype(bool)
    presence = trace["replica_hand_object_pair_presence"].astype(bool) & valid[..., None]
    tips = [names.index(TIP_BODIES[finger]) for finger in FINGER_ORDER]
    force = trace["force_magnitude"][:, :, tips]
    actual = presence[:, :, tips]
    strong = v2["strong_contact_expected"][:, None, :] & valid[..., None]
    expected_any = strong.any(axis=-1)
    full = expected_any & np.all(~strong | actual, axis=-1)
    missing_count = (strong & ~actual).sum(axis=-1)
    output: dict[str, object] = {
        "schema_version": "Stage16DRewardV3CompensationV2",
        "full_v2_coverage_sample_count": int(full.sum()),
        "missing_one_or_more_sample_count": int((missing_count >= 1).sum()),
        "contact_reward": {
            "full_v2_coverage": _stat(trace["replica_contact_reward"][full]),
            "missing_one_or_more": _stat(trace["replica_contact_reward"][missing_count >= 1]),
            "missing_two_or_more": _stat(trace["replica_contact_reward"][missing_count >= 2]),
        },
        "total_reward": {
            "full_v2_coverage": _stat(trace["replica_reward_total"][full]),
            "missing_one_or_more": _stat(trace["replica_reward_total"][missing_count >= 1]),
        },
        "per_finger": {},
    }
    for index, finger in enumerate(FINGER_ORDER):
        missing = strong[:, :, index] & ~actual[:, :, index]
        total = force.sum(axis=-1)
        other = total - force[:, :, index]
        output["per_finger"][finger] = {
            "missing_samples": int(missing.sum()),
            "other_tip_force_share_when_missing": _stat((other / (total + EPS))[missing]),
            "other_tip_force_n_when_missing": _stat(other[missing]),
            "contact_reward_when_missing": _stat(trace["replica_contact_reward"][missing]),
        }
    return output


def _decision(per_finger: dict[str, list[dict[str, Any]]]) -> dict[str, object]:
    source_support = sum(
        row["source_supported_fraction"] for rows in per_finger.values() for row in rows
    )
    long_floats = [
        f"{clip}:{row['finger']}"
        for clip, rows in per_finger.items()
        for row in rows
        if row["persistent_strong_missing_samples"] > 0
    ]
    if source_support == 0.0:
        primary = "REFERENCE_CONTACT_EVIDENCE_INSUFFICIENT"
        reason = (
            "No frozen per-finger source or topology contact evidence is available. "
            "The <=2 cm cohort is a geometric strong-contact candidate, not "
            "confirmed reference contact."
        )
        next_step = (
            "Acquire/map per-finger source contact or topology evidence, then rerun "
            "this audit before changing V3."
        )
    elif len(long_floats) >= 2:
        primary = "PER_FINGER_NORMALIZED_V4_RECOMMENDED"
        reason = "Multiple source-supported expected fingers persistently float in Formal20."
        next_step = "Design and ablate V4; keep V3 frozen as baseline."
    else:
        primary = "KEEP_V3_WITH_REFINED_CONTACT_MASK"
        reason = "No source-supported multi-finger persistent-loss condition was observed."
        next_step = "Refine only the diagnostic/source contact mask and rerun the audit."
    return {
        "schema_version": "Stage16DContactAuditR2DecisionV1",
        "primary_recommendation": primary,
        "secondary_signal": "MULTIPLE_GEOMETRIC_STRONG_CANDIDATE_FINGERS_FLOAT"
        if len(long_floats) >= 2
        else None,
        "source_evidence_available": source_support > 0.0,
        "persistent_geometric_strong_float_fingers": long_floats,
        "reason": reason,
        "next_step": next_step,
        "v3_preserved": True,
        "v4_authorized": primary == "PER_FINGER_NORMALIZED_V4_RECOMMENDED",
    }


def _markdown(summary: dict[str, object]) -> str:
    decision = summary["decision"]
    lines = [
        "# Stage 16-D Contact Contract V2 Audit",
        "",
        f"Primary recommendation: `{decision['primary_recommendation']}`",
        "",
        "V3 remains frozen.  V2 is diagnostic-only and no policy/reward was changed.",
        "",
        (
            "| Clip | Finger | V3 expected | V2 <=2 cm | 2–3 cm only | Actual tip | "
            "V2 strong recall | Longest strong miss | Any substitute |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for clip, report in summary["clips"].items():
        for row in report["per_finger_summary"]:
            lines.append(
                (
                    "| {clip} | {finger} | {v3:.1%} | {strong:.1%} | {prox:.1%} | "
                    "{actual:.1%} | {recall:.1%} | {longest} | {sub:.1%} |"
                ).format(
                    clip=clip,
                    finger=row["finger"],
                    v3=row["v3_expected_fraction"],
                    strong=row["v2_strong_fraction"],
                    prox=row["v2_proximity_only_fraction"],
                    actual=row["actual_tip_contact_fraction"],
                    recall=row["v2_strong_recall"] or 0.0,
                    longest=row["longest_strong_missing_steps"],
                    sub=row["any_substitute_fraction"] or 0.0,
                )
            )
    lines += [
        "",
        "## Interpretation",
        "",
        decision["reason"],
        "",
        f"Next: {decision['next_step']}",
        "",
        (
            "`r_wrist` is reported as `WRIST_BASE_CONTACT_BODY`; the asset provides "
            "no separately named palm collision body, so it is not relabeled as palm evidence."
        ),
        "",
    ]
    return "\n".join(lines)


def finalize(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    clips: dict[str, dict[str, object]] = {}
    all_per_finger: dict[str, list[dict[str, Any]]] = {}
    for clip in CLIPS:
        short = clip.removeprefix("hocap_")
        v2 = _load_v2(output / f"reference_contact_contract_v2_{short}.npz")
        telemetry = output / "full_pair_telemetry" / clip
        trace_path = telemetry / "trace_full_pair.npz"
        qualification_path = telemetry / "qualification.json"
        evaluation_path = telemetry / "full_pair_r2_evaluation.json"
        trace = _load_trace(trace_path)
        if str(trace["requested_clip"].item()) != clip:
            raise ValueError("CONTACT_AUDIT_R2_ACTIVE_OBJECT_MISMATCH")
        manifest = _manifest(trace_path, qualification_path)
        manifest["frozen_replay"] = _assert_frozen_replay(output, clip, evaluation_path)
        _write_json(telemetry / "telemetry_manifest.json", manifest)
        fingers = _per_finger_rows(v2, trace)
        bodies = _body_rows(trace)
        events = _events(v2, trace, clip)
        compensation = _compensation(v2, trace)
        _write_csv(output / f"per_finger_summary_{short}.csv", fingers)
        _write_csv(output / f"full_body_contact_summary_{short}.csv", bodies)
        _write_json(output / f"freeflight_events_{short}.json", events)
        _write_json(output / f"reward_compensation_{short}.json", compensation)
        _write_json(
            output / f"finger_contact_evidence_{short}.json",
            {
                "clip": clip,
                "source_evidence_status": "SOURCE_PER_FINGER_EVIDENCE_UNAVAILABLE",
                "evidence_class_counts": {
                    key: int((v2["reference_evidence_class"] == key).sum())
                    for key in np.unique(v2["reference_evidence_class"])
                },
                "reference_evidence_conflict_count": int(
                    v2["reference_contact_evidence_conflict"].sum()
                ),
                "per_finger": fingers,
            },
        )
        clips[clip] = {
            "full_pair_telemetry": manifest,
            "per_finger_summary": fingers,
            "full_body_summary": bodies,
            "freeflight": events,
            "reward_compensation": compensation,
            "palm_mapping": trace["palm_mapping"],
        }
        all_per_finger[clip] = fingers
    decision = _decision(all_per_finger)
    summary = {
        "schema_version": "Stage16DContactContractV2AuditSummaryV1",
        "status": "STAGE16D_CONTACT_CONTRACT_V2_AUDIT_COMPLETE",
        "decision": decision,
        "v3_primary_mask": {"formula": "distance_m < 0.03", "mutated": False},
        "v2": {"diagnostic_only": True, "strong_distance_m": 0.02, "persistence_steps": 3},
        "clips": clips,
    }
    _write_json(output / "final_summary.json", summary)
    (output / "final_summary.md").write_text(_markdown(summary), encoding="utf-8")
    _write_json(
        output / "candidate_v4_proposals.json",
        {
            "status": "NOT_AUTHORIZED" if not decision["v4_authorized"] else "PROPOSAL_REQUIRED",
            "reason": decision["reason"],
            "v3_baseline_mutated": False,
            "next_step": decision["next_step"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16d_contact_contract_v2_audit",
    )
    args = parser.parse_args()
    summary = finalize(args.output.resolve())
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
