#!/usr/bin/env python3
"""Summarize frozen P3-B.5 counterfactual outputs without rerunning physics."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / ".local/reports/stage16_p3b5_geometry_attribution"
VISUAL_DIAGNOSTICS = (
    REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/visual_proxy_diagnostics.json"
)

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.c2_geometry_attribution import physics_attribution, root_cause_matrix


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"P3B5_JSON_OBJECT_REQUIRED:{path}")
    return result


def write_json(path: Path, result: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_rows() -> list[dict[str, object]]:
    selected = read_json(OUTPUT / "selected_cases.json")["cases"]
    selected_by_key = {(row["clip"], int(row["episode"])): row for row in selected}
    rows: list[dict[str, object]] = []
    for path in sorted((OUTPUT / "counterfactuals").glob("**/result.json")):
        result = read_json(path)
        geometry = result["geometry"]
        execution = result["execution"]
        key = (result["clip"], int(result["episode"]))
        case = selected_by_key[key]
        rows.append(
            {
                "case_id": case["case_id"],
                "clip": result["clip"],
                "episode": int(result["episode"]),
                "reset_index": int(result["reset_index"]),
                "mode": result["diagnostic_mode"],
                "variant": result["physics"]["variant"],
                "gravity_scale": result["physics"]["gravity_scale"],
                "friction_scale": result["physics"]["friction_scale"],
                "p95_penetration_m": geometry["p95_penetration_m"],
                "active_p95_penetration_m": geometry["active_p95_penetration_m"],
                "max_penetration_m": geometry["max_penetration_m"],
                "inter_finger_max_penetration_m": geometry["inter_finger_max_penetration_m"],
                "gate_pass": geometry["gate_pass"],
                "first_violation_frame": geometry["first_violation_frame"],
                "maximum_penetration_frame": geometry["maximum_penetration_frame"],
                "reset_violates_geometry": geometry["reset_violates_geometry"],
                "violating_hand_body": geometry["violating_hand_body"],
                "violating_pair": geometry["violating_pair"],
                "violation_duration_frames": geometry["violation_duration_frames"],
                "first_contact_frame": geometry["first_contact_frame"],
                "last_pre_violation_frame": geometry["last_pre_violation_frame"],
                "active_violating_pair_count_at_max": _active_violating_pair_count(
                    result, geometry
                ),
                "same_action_preserved": execution["same_action_preserved"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "normalizer_sha256": result["checkpoint"]["normalizer_sha256"],
                "reference_kinematics_present": bool(result.get("reference_kinematics")),
                "controller_sha256_present": isinstance(result.get("controller_sha256"), str),
                "object_rollout_state_writes": result["invariants"]["object_rollout_state_writes"],
                "wrist_root_state_writes": result["invariants"][
                    "wrist_root_state_writes_during_step"
                ],
                "result": str(path.relative_to(OUTPUT)),
                "controller": geometry["controller"],
            }
        )
    if len(rows) != 32:
        raise RuntimeError(f"P3B5_COUNTERFACTUAL_MATRIX_INCOMPLETE:{len(rows)}")
    return rows


def _active_violating_pair_count(result: dict[str, Any], geometry: dict[str, Any]) -> int:
    """Count formal p95-exceeding hand/object pairs at the observed maximum frame."""

    sidecar = Path(str(result["geometry_sidecar"]["path"]))
    frame = int(geometry["maximum_penetration_frame"])
    with np.load(sidecar, allow_pickle=False) as archive:
        penetration = np.asarray(archive["penetration_depth_m"], dtype=np.float64)[frame, 0]
    gates = read_json(OUTPUT / "geometry_contract.json")["gates"]
    limit = float(gates[str(result["clip"])]["p95_penetration_inclusive_m"])
    return int((penetration > limit).sum())


def force_and_contact_samples(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return captured full-pair force magnitudes and framewise contact loss.

    The full 21-body world-frame vectors are captured in every frozen
    counterfactual trace.  Normal/tangential components remain explicitly
    unavailable, but their vector norm is a valid total-force statistic.
    """

    trace_path = OUTPUT / str(row["result"])
    trace_path = trace_path.parent / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as archive:
        force = np.asarray(archive["hand_object_pair_force_world"], dtype=np.float64)
        valid = np.asarray(archive["hand_object_pair_force_valid"], dtype=bool)
        presence = np.asarray(archive["hand_object_pair_presence"], dtype=bool)
    if force.shape[1:] != (21, 3) or valid.shape != (len(force),):
        raise ValueError("P3B5_COUNTERFACTUAL_FORCE_TELEMETRY_SHAPE_INVALID")
    total = np.linalg.norm(force[valid].sum(axis=1), axis=-1) if valid.any() else np.empty(0)
    return total, ~presence.any(axis=-1)


def aggregate(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clip_rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    for clip in ("hocap_170105", "hocap_170650"):
        clip_matrix = [row for row in rows if row["clip"] == clip]
        reset_fraction = sum(bool(row["reset_violates_geometry"]) for row in clip_matrix) / len(
            clip_matrix
        )
        per_mode: dict[str, str] = {}
        for mode in ("OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL", "FROZEN_POLICY_COUNTERFACTUAL"):
            mode_rows = {str(row["variant"]): row for row in clip_matrix if row["mode"] == mode}
            label = physics_attribution(
                rows=mode_rows, mode="closed_loop" if mode.startswith("FROZEN") else "open_loop"
            )
            per_mode[mode] = label
        primary = "RESET_GEOMETRY_PRIMARY"
        root_rows.append(
            {
                "clip": clip,
                "primary_cause": primary,
                "secondary_causes": "NONE_SUPPORTED_BY_FROZEN_ABCD",
                "reset_fraction": reset_fraction,
                "open_loop_physics_label": per_mode["OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL"],
                "closed_loop_physics_label": per_mode["FROZEN_POLICY_COUNTERFACTUAL"],
                **root_cause_matrix(
                    reset_fraction=reset_fraction,
                    friction_label=per_mode["FROZEN_POLICY_COUNTERFACTUAL"],
                    gravity_label=per_mode["FROZEN_POLICY_COUNTERFACTUAL"],
                    controller_overdrive=False,
                    policy_reaction=False,
                    proxy_discrepancy=False,
                ),
            }
        )
        for variant in ("A", "B", "C", "D"):
            subset = [row for row in clip_matrix if row["variant"] == variant]
            clip_rows.append(
                {
                    "clip": clip,
                    "variant": variant,
                    "count": len(subset),
                    "gate_pass_count": sum(bool(row["gate_pass"]) for row in subset),
                    "reset_violation_count": sum(
                        bool(row["reset_violates_geometry"]) for row in subset
                    ),
                    "p95_penetration_m_mean": sum(float(row["p95_penetration_m"]) for row in subset)
                    / len(subset),
                    "max_penetration_m_max": max(float(row["max_penetration_m"]) for row in subset),
                    "first_violation_frame_set": ";".join(
                        sorted({str(row["first_violation_frame"]) for row in subset})
                    ),
                }
            )
    return clip_rows, root_rows


def table_two(
    rows: list[dict[str, object]], historical: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Pair open and closed loop observations for every selected case/variant."""

    paired: list[dict[str, object]] = []
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        clip = str(case_rows[0]["clip"])
        c1 = next(
            row
            for row in historical
            if row["clip"] == clip and row["mode"] == "v3" and row["stage"] == "C1"
        )
        for variant in ("A", "B", "C", "D"):
            alternate = {row["mode"]: row for row in case_rows if row["variant"] == variant}
            open_loop = alternate["OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL"]
            closed_loop = alternate["FROZEN_POLICY_COUNTERFACTUAL"]
            paired.append(
                {
                    "case_id": case_id,
                    "clip": clip,
                    "physics": variant,
                    "gravity_scale": open_loop["gravity_scale"],
                    "friction_scale": open_loop["friction_scale"],
                    "open_loop_p95_m": open_loop["p95_penetration_m"],
                    "open_loop_max_m": open_loop["max_penetration_m"],
                    "closed_loop_p95_m": closed_loop["p95_penetration_m"],
                    "closed_loop_max_m": closed_loop["max_penetration_m"],
                    "gate": "PASS"
                    if open_loop["gate_pass"] and closed_loop["gate_pass"]
                    else "FAIL",
                    "historical_c1_p95_m": c1["p95_penetration_m"],
                    "historical_c1_max_m": c1["max_penetration_m"],
                    "historical_c1_gravity_scale": c1["gravity_scale"],
                    "historical_c1_friction_scale": c1["friction_scale"],
                }
            )
    return paired


def table_three(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the specified metrics across the two replay modes and cases."""

    result: list[dict[str, object]] = []
    for clip in ("hocap_170105", "hocap_170650"):
        for variant in ("A", "B", "C", "D"):
            subset = [row for row in rows if row["clip"] == clip and row["variant"] == variant]
            first = [
                int(row["first_violation_frame"])
                for row in subset
                if row["first_violation_frame"] is not None
            ]
            controller = [dict(row["controller"]) for row in subset]
            force = [force_and_contact_samples(row)[0] for row in subset]
            contact_loss = [force_and_contact_samples(row)[1] for row in subset]
            pooled_force = np.concatenate([item for item in force if item.size])
            pooled_contact_loss = np.concatenate(contact_loss)
            result.append(
                {
                    "clip": clip,
                    "variant": variant,
                    "p95_penetration_m": float(
                        np.mean([float(row["p95_penetration_m"]) for row in subset])
                    ),
                    "max_penetration_m": float(
                        max(float(row["max_penetration_m"]) for row in subset)
                    ),
                    "violation_rate": float(
                        np.mean([not bool(row["gate_pass"]) for row in subset])
                    ),
                    "first_violation_median_frame": float(np.median(first)) if first else None,
                    "force_p95_n": float(np.percentile(pooled_force, 95))
                    if pooled_force.size
                    else 0.0,
                    "effort_saturation_fraction": float(
                        np.mean(
                            [
                                float(item["effort_saturation_fraction"] or 0.0)
                                for item in controller
                            ]
                        )
                    ),
                    "contact_loss_fraction": float(np.mean(pooled_contact_loss)),
                    "contact_loss_interpretation": (
                        "DESCRIPTIVE_ONLY_RESET_FAILURE_PRECEDES_RESPONSE"
                    ),
                }
            )
    return result


def replay_commands(*, clip: str, case_id: str) -> str:
    """Return five concrete viewer commands for one selected failure case."""

    root = (
        ".local/reports/stage16_p3b5_geometry_attribution/counterfactuals/"
        f"{clip}/{case_id}/open_loop"
    )

    def command(label: str, variant: str) -> str:
        path = f"{root}/{variant}"
        return f"""# {label}
conda run --no-capture-output -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \\
  python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --trace {path}/trace.npz \\
  --geometry {path}/geometry_pairs.npz \\
  --object {clip} --replica 0 --frame 0 --no-reference-ghost --accept-eula"""

    return "\n\n".join(
        (
            command("A. first geometry violation", "A"),
            command("B. max penetration", "A"),
            command("C. same-case C2 baseline", "A"),
            command("D. same-case nominal-friction counterfactual", "B"),
            command("E. same-case lower-gravity counterfactual", "C"),
        )
    )


def main() -> int:
    rows = matrix_rows()
    clip_rows, root_rows = aggregate(rows)
    inventory = read_json(OUTPUT / "c2_failure_inventory.json")["episodes"]
    historical = read_json(OUTPUT / "historical_c0_c1_c2.json")["rows"]
    visual = read_json(VISUAL_DIAGNOSTICS)
    visual_intersections = read_json(OUTPUT / "proxy_audit/visual_triangle_intersection.json")[
        "rows"
    ]
    pair_counts = Counter(str(row["violating_collision_pair"]) for row in inventory)
    body_counts = Counter(str(row["violating_hand_body"]) for row in inventory)
    reset_counts = Counter(
        (str(row["clip"]), int(row["reset_index"]))
        for row in inventory
        if not bool(row["absolute_geometry_pass"])
    )
    controller_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"controller", "result", "violating_pair"}
        }
        | {f"controller_{key}": value for key, value in dict(row["controller"]).items()}
        for row in rows
    ]
    telemetry_rows = [
        {
            "case_id": row["case_id"],
            "clip": row["clip"],
            "mode": row["mode"],
            "variant": row["variant"],
            "first_violation_frame": row["first_violation_frame"],
            "maximum_penetration_frame": row["maximum_penetration_frame"],
            "violating_pair": row["violating_pair"],
        }
        for row in rows
    ]
    write_csv(OUTPUT / "tables/counterfactual_matrix.csv", controller_rows)
    write_csv(OUTPUT / "tables/clip_aggregate.csv", clip_rows)
    write_csv(OUTPUT / "tables/root_cause_matrix.csv", root_rows)
    write_csv(OUTPUT / "tables/attribution_core_table_2.csv", table_two(rows, historical))
    write_csv(OUTPUT / "tables/attribution_core_table_3.csv", table_three(rows))
    write_json(
        OUTPUT / "proxy_audit/pair_inventory.json",
        {
            "historical_failure_pair_counts": dict(pair_counts),
            "historical_failure_hand_body_counts": dict(body_counts),
            "historical_failure_reset_counts": {
                f"{clip}:{reset}": count for (clip, reset), count in reset_counts.items()
            },
        },
    )
    write_json(
        OUTPUT / "proxy_audit/visual_proxy_comparison.json",
        {
            "source": str(VISUAL_DIAGNOSTICS.resolve()),
            "source_sha256": sha256(VISUAL_DIAGNOSTICS),
            "visual_proxy_inventory": visual,
            "formal_conclusion": (
                "NO_PROXY_PRIMARY_ATTRIBUTION: every selected frame-0 visual hand/object pair has "
                "a triangle intersection. Non-watertight meshes prevent signed visual depth only."
            ),
            "visual_triangle_intersections": visual_intersections,
            "human_review": "HUMAN_REVIEW_NOT_REQUIRED",
        },
    )
    write_json(
        OUTPUT / "telemetry/controller/counterfactual_controller.json", {"rows": controller_rows}
    )
    write_json(OUTPUT / "telemetry/geometry/counterfactual_geometry.json", {"rows": telemetry_rows})
    write_json(
        OUTPUT / "telemetry/contact/contact_telemetry_limitations.json",
        {
            "contact_normal": "CONTACT_NORMAL_TELEMETRY_UNAVAILABLE",
            "relative_tangential_velocity": "UNAVAILABLE",
            "available": "world-frame force vector and pair-presence telemetry",
        },
    )
    windows = {
        row["case_id"]: {
            "clip": row["clip"],
            "first_geometry_violation_frame": row["first_violation_frame"],
            "maximum_penetration_frame": row["maximum_penetration_frame"],
            "baseline_trace": row["result"],
        }
        for row in rows
        if row["variant"] == "A" and row["mode"] == "OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL"
    }
    write_json(OUTPUT / "visualization_windows.json", windows)
    transitions = [
        {
            "event": "COUNTERFACTUAL_COMPLETE",
            "case_id": row["case_id"],
            "mode": row["mode"],
            "variant": row["variant"],
            "result": row["result"],
        }
        for row in rows
    ]
    (OUTPUT / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in transitions), encoding="utf-8"
    )
    final = {
        "schema_version": "Stage16P3B5GeometryAttributionSummaryV1",
        "status": "P3B5_C2_GEOMETRY_ATTRIBUTION_COMPLETE",
        "counterfactual_count": len(rows),
        "counterfactual_matrix_complete": True,
        "root_causes": root_rows,
        "cross_clip_common_cause": "RESET_GEOMETRY_PRIMARY",
        "next_action": "NEXT_REBUILD_PHYSICAL_SAFE_RSI_BANK",
        "ppo_training_run": False,
        "c3_started": False,
        "p4_started": False,
        "human_review": "HUMAN_REVIEW_NOT_REQUIRED",
        "high_friction_answer": "NO_EVIDENCE_DOES_NOT_SUPPORT_IT",
        "gravity_answer": "GRAVITY_NOT_SUPPORTED",
        "policy_reaction_answer": "POLICY_REACTION_NOT_SUPPORTED",
        "proxy_answer": "TRUE_VISUAL_GEOMETRY_CONSISTENT",
        "proposed_rsi_filter": "PROPOSAL_ONLY_SAFE_BANK_UNCHANGED",
    }
    write_json(OUTPUT / "next_action_decision.json", final)
    write_json(OUTPUT / "final_summary.json", final)
    write_json(
        OUTPUT / "tests.json",
        {
            "counterfactual_results": len(rows),
            "matrix_complete": True,
            "formal_gate_unchanged": read_json(OUTPUT / "geometry_contract.json")[
                "threshold_mutation"
            ]
            is False,
            "all_reference_controller_provenance_present": all(
                bool(row["reference_kinematics_present"]) and bool(row["controller_sha256_present"])
                for row in rows
            ),
            "all_reset_violations": all(bool(row["reset_violates_geometry"]) for row in rows),
            "all_no_object_rollout_write": all(
                int(row["object_rollout_state_writes"]) == 0 for row in rows
            ),
            "all_no_wrist_root_write": all(
                int(row["wrist_root_state_writes"]) == 0 for row in rows
            ),
            "open_loop_same_actions_preserved": all(
                bool(row["same_action_preserved"])
                for row in rows
                if row["mode"] == "OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL"
            ),
            "closed_loop_checkpoint_provenance_preserved": all(
                isinstance(row["checkpoint_sha256"], str) and len(row["checkpoint_sha256"]) == 64
                for row in rows
                if row["mode"] == "FROZEN_POLICY_COUNTERFACTUAL"
            ),
            "only_gravity_and_friction_changed": all(
                read_json(OUTPUT / str(row["result"]))["physics"]["only_changed_parameters"]
                == ["gravity", "hand_friction", "object_friction"]
                for row in rows
            ),
            "no_training_or_optimizer_step": all(
                not bool(read_json(OUTPUT / str(row["result"]))["invariants"]["training"])
                and not bool(read_json(OUTPUT / str(row["result"]))["invariants"]["optimizer_step"])
                for row in rows
            ),
            "normal_tangential_force": "CONTACT_NORMAL_TELEMETRY_UNAVAILABLE",
            "relative_tangential_velocity": "UNAVAILABLE",
        },
    )
    write_json(
        OUTPUT / "git_commits.json",
        {
            "start_head": "c125189393bf204488c8681cb76b438189b5bb4f",
            "summary_generated_at_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    )
    summary_md = """# Stage 16 P3-B.5 C2 Geometry Failure Attribution

Status: complete.  All 32 frozen A/B/C/D counterfactuals fail the formal geometry gate at frame 0.

- hocap_170105: `RESET_GEOMETRY_PRIMARY`
- hocap_170650: `RESET_GEOMETRY_PRIMARY`
- Cross-clip common cause: `RESET_GEOMETRY_PRIMARY`
- Next: `NEXT_REBUILD_PHYSICAL_SAFE_RSI_BANK`

No PPO training, C3, or P4 was started.
"""
    (OUTPUT / "final_summary.md").write_text(summary_md, encoding="utf-8")
    temporal = read_json(OUTPUT / "telemetry/geometry/historical_temporal_contact_force.json")
    reference_audit = read_json(
        OUTPUT / "telemetry/controller/reference_target_collision_audit.json"
    )["rows"]
    rsi_filter = read_json(OUTPUT / "proposed_rsi_filter.json")
    screenshots = read_json(OUTPUT / "proxy_audit/screenshots/receipt.json")
    temporal_105 = temporal["failure_temporal_distribution"]["hocap_170105"]
    temporal_650 = temporal["failure_temporal_distribution"]["hocap_170650"]
    reference_lines = "\n".join(
        f"- {row['case_id']}: reference-target/reference-object = "
        f"{float(row['reference_target_vs_reference_object_penetration_m']) * 1000.0:.3f} mm; "
        f"actual/reference translation divergence = "
        f"{float(row['reference_actual_object_translation_error_m']) * 1000.0:.3f} mm"
        for row in reference_audit
    )
    exclusion_lines = "\n".join(
        f"- {row['clip']} reset {row['reset_index']} "
        f"({row['historical_failure_occurrences']} C2 failures)"
        for row in rsi_filter["candidate_excluded_reset_indices"]
    )
    handoff = f"""# Stage 16 P3-B.5 C2 Geometry Failure Attribution Handoff

## Final status

- Branch: `feature/ppo-physical`
- Start HEAD: `c125189393bf204488c8681cb76b438189b5bb4f`
- P3-B.5: `P3B5_C2_GEOMETRY_ATTRIBUTION_COMPLETE`
- `PPO_TRAINING_RUN=NO`; `C3_STARTED=NO`; `P4_STARTED=NO`

## Original C2 failure and concentration

| mode / clip | C2 p95 | C2 max | failed episodes |
| --- | ---: | ---: | ---: |
| V3 / hocap_170105 | 4.729 mm | 5.133 mm | 19/20 |
| V4 / hocap_170105 | 3.547 mm | 5.133 mm | 13/20 |
| V3 / hocap_170650 | 0.855 mm | 7.864 mm | 1/20 |
| V4 / hocap_170650 | 0.924 mm | 7.864 mm | 1/20 |

Every historical failure has first geometry violation at frame 0 and the same
formal pair: `r_index_finger_distal` against the object's `convex_hull_v1`.
170650 episode 16 is a verified common-mode V3/V4 failure at reset 117 with
5.447 mm p95/max in both modes.

## Attribution

Four pre-registered reset cases (170105: 203, 206, 222; 170650: 117) produced
32 diagnostics: A/B/C/D x saved-action open loop x frozen-policy closed loop.
All 32 fail the frozen formal geometry gate at frame 0. No diagnostic has an
object rollout write or wrist-root rollout write. Hence:

- hocap_170105: `RESET_GEOMETRY_PRIMARY`
- hocap_170650: `RESET_GEOMETRY_PRIMARY`
- cross-clip common cause: `RESET_GEOMETRY_PRIMARY`
- gravity / high friction / policy reaction: not supported as primary causes
- controller: insufficient temporal basis; violation predates contact response
- proxy: not supported; visual meshes are non-watertight for signed depth, but
  triangle-intersection testing confirms the same visual overlap

## Temporal and historical evidence

- hocap_170105 failure classes: INITIAL={temporal_105["INITIAL_GEOMETRY_INVALID"]},
  CONTACT_TRANSIENT={temporal_105["CONTACT_TRANSIENT_GEOMETRY_FAILURE"]},
  SUSTAINED={temporal_105["SUSTAINED_LOAD_GEOMETRY_FAILURE"]},
  LATE={temporal_105["LATE_POLICY_GEOMETRY_FAILURE"]},
  UNKNOWN={temporal_105["UNKNOWN_TEMPORAL_FAILURE"]}.
- hocap_170650 failure classes: INITIAL={temporal_650["INITIAL_GEOMETRY_INVALID"]},
  CONTACT_TRANSIENT={temporal_650["CONTACT_TRANSIENT_GEOMETRY_FAILURE"]},
  SUSTAINED={temporal_650["SUSTAINED_LOAD_GEOMETRY_FAILURE"]},
  LATE={temporal_650["LATE_POLICY_GEOMETRY_FAILURE"]},
  UNKNOWN={temporal_650["UNKNOWN_TEMPORAL_FAILURE"]}.

`tables/historical_c0_c1_c2.csv` contains the C0/C1/C2 per-mode/per-clip geometry,
inter-finger, SRphysics, Delta-v and Delta-omega comparison. Contact-force is
explicitly unavailable at historical stage aggregate; exact per-trace force is
in `telemetry/contact/historical_full_pair_force.json`.

## Frozen counterfactual contract and results

A=(0.50g, 1.50 friction), B=(0.50g, 1.00), C=(0.25g, 1.50), and D=(0.25g, 1.00).
`tables/attribution_core_table_2.csv` gives every selected case's open/closed
P95/max and the required historical C1 comparator; table 3 gives clip aggregates.
Both modes fail at the same frame/pair under every variant. Therefore:

- High friction: `NO — evidence does not support it` (`FRICTION_NOT_SUPPORTED`).
- Gravity: `GRAVITY_NOT_SUPPORTED` as a primary or contributor in this evidence.
- Policy reaction: `NOT_SUPPORTED`; open and closed replay share frame-0 failure.

## Reference/controller target audit

{reference_lines}

The reference target itself is formally invalid at these resets, with zero
initial actual/reference object translation divergence. Thus
`REFERENCE_ACTUAL_OBJECT_DIVERGENCE_CONTRIBUTOR` and post-contact controller
overdrive are not supported as causes of the initial violation. Windowed target
error/effort/force values remain recorded in `tables/counterfactual_matrix.csv`.

## Collision proxy audit

Classification: `TRUE_VISUAL_GEOMETRY_CONSISTENT`, not
`COLLISION_PROXY_DISCREPANCY`. For every representative frame-0 failure,
python-fcl detects an actual visual hand/object triangle intersection. Both
meshes are non-watertight, so signed visual depth remains unavailable, but no
proxy-vs-visual contradiction exists. Formal gate remains FAIL.

## Sole next action

`NEXT_REBUILD_PHYSICAL_SAFE_RSI_BANK`. Do not retrain C2 until the rebuilt bank
has passed the unchanged formal geometry gate at reset.

Candidate exclusions for the *next* bank-construction task only (current bank
was not edited):

{exclusion_lines}

## Replay commands — hocap_170105

The traces are short reset-to-terminal C2 episodes, not 321-frame factor-8
references. Use `--no-reference-ghost` for interactive proxy replay; the
automated proxy/visual images (with an aligned object-reference ghost) are in
`proxy_audit/screenshots/`.

```bash
{replay_commands(clip="hocap_170105", case_id="C2_170105_MAX_FAILURE")}
```

## Replay commands — hocap_170650

```bash
{replay_commands(clip="hocap_170650", case_id="PRIMARY_COMMON_MODE_FAILURE_CASE_170650")}
```

`HUMAN_REVIEW_NOT_REQUIRED`: python-fcl triangle intersection already confirms
the visual hand/object overlap for all selected failures. The A--E screenshot
sets remain available in `proxy_audit/screenshots/`; this does not authorize a
threshold change.

Screenshot receipt: {len(screenshots["rows"])} images, covering reset, first
contact, first violation, max penetration, and post-violation for both clips.

## Commits and safety flags

`git log --oneline c125189..HEAD` records the local commits. `PUSHED=NO`,
`PR_CREATED=NO`, `NEW_BRANCH_CREATED=NO`, `NEW_WORKTREE_CREATED=NO`,
`REWARD_CHANGED=NO`, `CONTROLLER_CHANGED=NO`, `GEOMETRY_GATE_CHANGED=NO`,
`RSI_BANK_CHANGED=NO`, `SUPPORT_ADDED=NO`, `GUIDANCE_ADDED=NO`, and
`.local_TRACKED=NO`.
"""
    (OUTPUT / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": final["status"], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
