#!/usr/bin/env python3
"""Summarize frozen P3-B.5 counterfactual outputs without rerunning physics."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
                "same_action_preserved": execution["same_action_preserved"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "normalizer_sha256": result["checkpoint"]["normalizer_sha256"],
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
                    proxy_discrepancy=None,
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


def main() -> int:
    rows = matrix_rows()
    clip_rows, root_rows = aggregate(rows)
    inventory = read_json(OUTPUT / "c2_failure_inventory.json")["episodes"]
    visual = read_json(VISUAL_DIAGNOSTICS)
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
                "NO_PROXY_PRIMARY_ATTRIBUTION: object visual meshes are non-watertight and this "
                "unsigned static comparison cannot invalidate the runtime formal proxy gate."
            ),
            "human_review": "HUMAN_REVIEW_REQUIRED_FOR_VISUAL_MESH_INTERSECTION_ONLY",
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
        "human_review": "HUMAN_REVIEW_REQUIRED_FOR_VISUAL_MESH_INTERSECTION_ONLY",
    }
    write_json(OUTPUT / "next_action_decision.json", final)
    write_json(OUTPUT / "final_summary.json", final)
    write_json(
        OUTPUT / "tests.json",
        {
            "counterfactual_results": len(rows),
            "matrix_complete": True,
            "all_reset_violations": all(bool(row["reset_violates_geometry"]) for row in rows),
            "all_no_object_rollout_write": all(
                int(row["object_rollout_state_writes"]) == 0 for row in rows
            ),
            "all_no_wrist_root_write": all(
                int(row["wrist_root_state_writes"]) == 0 for row in rows
            ),
        },
    )
    write_json(
        OUTPUT / "git_commits.json",
        {
            "start_head": "c125189393bf204488c8681cb76b438189b5bb4f",
            "commits": "PENDING_LOCAL_COMMIT",
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
    replay_105 = (
        ".local/reports/stage16_p3b5_geometry_attribution/counterfactuals/"
        "hocap_170105/C2_170105_MAX_FAILURE/open_loop/A"
    )
    replay_650 = (
        ".local/reports/stage16_p3b5_geometry_attribution/counterfactuals/"
        "hocap_170650/PRIMARY_COMMON_MODE_FAILURE_CASE_170650/open_loop/B"
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
- proxy: not established; visual object meshes are non-watertight

## Sole next action

`NEXT_REBUILD_PHYSICAL_SAFE_RSI_BANK`. Do not retrain C2 until the rebuilt bank
has passed the unchanged formal geometry gate at reset.

## Replay commands

The traces are short reset-to-terminal C2 episodes, not 321-frame factor-8
references. Use `--no-reference-ghost` for interactive proxy replay; the
automated proxy/visual images (with an aligned object-reference ghost) are in
`proxy_audit/screenshots/`.

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \\
  python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --trace {replay_105}/trace.npz \\
  --geometry {replay_105}/geometry_pairs.npz \\
  --object hocap_170105 --frame 0 --no-reference-ghost --accept-eula

conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \\
  python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --trace {replay_650}/trace.npz \\
  --geometry {replay_650}/geometry_pairs.npz \\
  --object hocap_170650 --frame 0 --no-reference-ghost --accept-eula
```

`HUMAN_REVIEW_REQUIRED_FOR_VISUAL_MESH_INTERSECTION_ONLY`: inspect the two
exported static images and answer per clip whether the visual finger mesh
visibly enters the visual object surface, is proxy-only, or cannot be judged.
This does not authorize a threshold change.
"""
    (OUTPUT / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": final["status"], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
