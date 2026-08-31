#!/usr/bin/env python3
"""Write the final P8 handoff and guarded replay ledger."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P8 = (
    ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p8_two_canary_physicalization"
)
P4 = ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p4_hocap_semantic_preflight"
P1 = ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p1_target_object_audit"
P3 = ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p3_golden_suite"
P7 = ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p7_unseen_object_refreeze"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def guarded_command(
    clip: str,
    status: str,
    start: int | None = None,
    end: int | None = None,
    *,
    low_poly: bool = False,
    no_reference: bool = False,
) -> str:
    trace = f".local/reports/dataset_semantic_authority_two_clip_canary/p8_two_canary_physicalization/per_episode/{clip}.physical_policy_eval10.npz"
    args = [
        "/home/deepcybo/miniconda3/bin/conda run --no-capture-output -n toporetarget-rl python",
        "scripts/rl/isaaclab/replay_physical_hoi_trace.py",
        "--accept-eula",
        '--trace "$TRACE"',
        f"--object {clip}",
    ]
    if start is not None:
        args.extend([f"--start-frame {start}", f"--end-frame {end}"])
    args.extend(
        [
            "--reference-ghost" if not no_reference else "--no-reference-ghost",
            "--mocap-ghost",
            "--mocap-object",
            "--mocap-fingertips",
        ]
    )
    if low_poly:
        args.append("--mocap-object-low-poly")
    return (
        f'TRACE={trace}; test -s "$TRACE" || '
        f"{{ echo FAILED_BEST_DIAGNOSTIC:{status}:no_policy_trace; exit 2; }}; " + " ".join(args)
    )


def main() -> int:
    episodes = [read(path) for path in sorted((P8 / "per_episode").glob("*.json"))]
    p4 = read(P4 / "corpus_summary.json")
    p3 = read(P3 / "certification.json")
    wrong = read(P1 / "wrong_target_cases.json")
    rows = list(csv.DictReader((P8 / "main_metrics.csv").open(encoding="utf-8")))

    help_command = (
        "/home/deepcybo/miniconda3/bin/conda run --no-capture-output -n toporetarget-rl "
        "python scripts/rl/isaaclab/replay_physical_hoi_trace.py --help"
    )
    c1 = episodes[0]["episode"]
    c2 = episodes[1]["episode"]
    commands = [
        "# P8 replay status: FAILED_BEST_DIAGNOSTIC for both clips.",
        "# replay --help was actually executed and returned PASS before this file was written.",
        f"# HELP: {help_command}",
        "# No Stage16D physical-policy trace exists after the terminal support gates.",
        "# The guarded commands below are real copy-paste commands and stop explicitly; support JSON is not substituted.",
        "",
        f"## Canary 1: {c1} — PHYSICAL_SCENE_INVALID",
        guarded_command(c1, "PHYSICAL_SCENE_INVALID"),
        guarded_command(c1, "PHYSICAL_SCENE_INVALID_PICK_LIFT", 0, 360),
        guarded_command(
            c1, "PHYSICAL_SCENE_INVALID_PLACE_RELEASE", 360, 720, low_poly=True, no_reference=True
        ),
        "",
        f"## Canary 2: {c2} — SUPPORT_UNRESOLVED",
        guarded_command(c2, "SUPPORT_UNRESOLVED"),
        guarded_command(c2, "SUPPORT_UNRESOLVED_PICK_LIFT", 0, 360),
        guarded_command(
            c2, "SUPPORT_UNRESOLVED_PLACE_RELEASE", 360, 720, low_poly=True, no_reference=True
        ),
        "",
        "# Default layers requested by contract: actual PhysX hand/object/support, raw HOCap MANO/object, retarget reference ghost.",
        "# The no-reference-ghost and mocap-object-low-poly variants are included above.",
        "# All commands currently exit with FAILED_BEST_DIAGNOSTIC because no valid policy trace was authorized.",
    ]
    (P8 / "replay/visualization_commands.md").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    (P8 / "replay/manual_acceptance.md").write_text(
        "# P8 manual acceptance\n\n"
        "Replay status: FAILED_BEST_DIAGNOSTIC for both canaries. No physical-policy trace was produced, "
        "so neither visual replay nor manual physical acceptance can be PASS.\n\n"
        "Canary 1 must be inspected after hand-object geometry is repaired: correct target object, right hand, "
        "support, causal pick, transport, place, release, retreat, no flick, no teleport, and raw/retarget/actual fidelity.\n\n"
        "Canary 2 must be inspected only after source-support authority is resolved, with the same checks.\n",
        encoding="utf-8",
    )
    (P8 / "replay/headless_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "P8ReplayHeadlessValidationV1",
                "status": "BLOCKED_NO_PHYSICAL_POLICY_TRACE",
                "checks": {
                    name: "NOT_RUN_NO_TRACE"
                    for name in (
                        "trace_loads",
                        "all_frames_finite",
                        "reference_loads",
                        "object_id_matches",
                        "mesh_hash_matches",
                        "support_loads",
                        "raw_ghost_aligns",
                        "full_frame_range_completes",
                    )
                },
                "reason": "Both canaries terminated before a Stage16D physical-policy trace existed.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (P8 / "final_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "DatasetSemanticAuthorityP8FinalSummaryV1",
                "status": "COMPLETE_WITH_TERMINAL_GATES",
                "scientific_acceptance": "FAIL_NO_CANARY_ACCEPTED",
                "p6_status": read(
                    ROOT
                    / ".local/reports/dataset_semantic_authority_two_clip_canary/p6_semantic_certification/final_authority_decision.json"
                )["status"],
                "p7_status": read(P7 / "unseen_object_frozen5_manifest.json")["status"],
                "canaries": episodes,
                "p4_corpus": p4,
                "golden_suite": p3,
                "wrong_target_cases": wrong,
                "main_table_rows": rows,
                "policy_outcomes_observed": False,
                "created_utc": utc(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Dataset Semantic Authority + Two-Canary Physicalization Handoff",
        "",
        "## Git",
        "",
        "- branch: feature/dexplore-reward-rse",
        "- START_HEAD: 598d80f458f2b3cdaa5a72487c7ce3eeed5e63b4",
        "- P5_EXECUTION_HEAD: 3632ca001671635cd98ee248b77855d11c1ebbf6",
        "- P8 started only after CANARY_1=APPROVE and CANARY_2=APPROVE.",
        "- PUSHED=NO; PR_CREATED=NO; .local_TRACKED=NO.",
        "",
        "## Two wrong-target root causes",
        "",
        "| Episode | Old selected object | Correct object | Selector valid? | Binding valid? | Root cause |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in wrong:
        lines.append(
            f"| {item['old_clip_id']} | {item['old_selected_object']} | {item['correct_object']} | "
            f"{item['selection_valid']} | {item['binding_valid']} | {item['root_cause']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Semantic Authority",
            "",
            "DatasetSemanticAuthorityV1 -> CanonicalHOIRecordV1 -> HOCapPrimaryObjectAuthorityV2 -> ObjectAssetBindingV1 -> HOISemanticPreflightV1 -> RetargetSemanticValidityV1 -> ManualRetargetAcceptanceV1 -> physical support/PhysX/PF/DF gates.",
            "",
            "## HOCap corpus preflight",
            "",
            f"raw sequences={p4['raw_sequences']}; episode candidates={p4['episode_candidates']}; binding PASS={p4['binding_counts']['PASS']}.",
            f"semantic PASS/QUARANTINE/FAIL={p4['semantic_counts']['SEMANTIC_PREFLIGHT_PASS']}/{p4['semantic_counts']['SEMANTIC_PREFLIGHT_QUARANTINE']}/{p4['semantic_counts']['SEMANTIC_PREFLIGHT_FAIL']}.",
            f"target PASS/official-vs-geometry conflict/bimanual={p4['target_counts']['TARGET_OBJECT_PASS']}/{p4['target_counts']['OFFICIAL_VS_GEOMETRY_CONFLICT']}/{p4['target_counts']['BIMANUAL_SAME_OBJECT']}.",
            "",
            "## Golden suite",
            "",
            f"{p3['status']}; positive_pass={p3['positive_pass']}; real_negative_detected={p3['real_negative_detected']}; synthetic_negative_detected={p3['synthetic_negative_detected']}.",
            "",
            "## Two retarget canaries",
            "",
            "| Episode | Object | Machine semantic | Human decision | Final retarget status |",
            "|---|---|---|---|---|",
            "| hocap_subject_9_20231027_125315__right__G21_3__ep00 | G21_3 | RETARGET_SEMANTIC_PASS | APPROVE | PASS |",
            "| hocap_subject_6_20231025_110646__right__G05_1__ep00 | G05_1 | RETARGET_SEMANTIC_PASS | APPROVE | PASS |",
            "",
            "## P7 unseen-object set",
            "",
            "selection unit=CanonicalHOIRecordV1; instance/object-id overlap=0; mesh overlap=0; geometry/alias overlap=0; category overlap=UNKNOWN_NO_AUTHORITY; downstream outcome used=NO; downstream run=NO.",
            f"manifest={P7 / 'unseen_object_frozen5_manifest.json'}; manifest SHA256=cf57c80bb93fb013fd3ee79dbe5af47a848800e11c3409655667b1c4ec60873c.",
            "",
            "## Two physicalization results",
            "",
            "See main_metrics.csv. Canary 1=PHYSICAL_SCENE_INVALID; Canary 2=SUPPORT_UNRESOLVED. Frozen PF, PF phase metrics, DF metrics, Confirm20, PPO, qualification, and policy trace export are gate-blocked, not scores.",
            "",
            "## Replay status",
            "",
            "Both are FAILED_BEST_DIAGNOSTIC. The exact replay entrypoint was help-validated, but no valid physical-policy trace exists.",
            "See replay/visualization_commands.md for three guarded commands per canary, including no-reference-ghost and low-poly variants.",
            "",
            "## Next step",
            "",
            "NEXT_ANALYZE_PHYSICAL_FAILURES_ON_SEMANTICALLY_VALID_REFERENCES after resolving canary-1 hand-object geometry and canary-2 source-support authority. The frozen P7 set is not run automatically.",
            "",
            "## Safety flags",
            "",
            "P0_COMPLETE=YES P1_COMPLETE=YES P2_COMPLETE=YES P3_COMPLETE=YES P4_COMPLETE=YES P5_COMPLETE=YES P6_COMPLETE=YES P7_COMPLETE=YES P8_COMPLETE=YES",
            "P5_HUMAN_GATE_ENFORCED=YES P6_P8_STARTED_BEFORE_USER_APPROVAL=NO TWO_CANARY_MACHINE_SEMANTIC_PASS=YES TWO_CANARY_HUMAN_APPROVED=YES",
            "P8_INDEPENDENT_PPO_PER_EPISODE=YES NEW_PPO_TRAINING_BEFORE_P5_APPROVAL=NO REFROZEN_UNSEEN_SET_DOWNSTREAM_RUN=NO",
            "REWARD_CHANGED=NO RSE_CHANGED=NO PF_THRESHOLDS_CHANGED=NO DF_THRESHOLDS_CHANGED=NO FRICTION_CHANGED=NO MASS_CHANGED=NO MATERIAL_CHANGED=NO",
            "H3D_OLD_MANIFEST_CONSUMED=NO VISUALIZATION_SCRIPTS_DELETED=NO HISTORICAL_REPORTS_DELETED=NO PUSHED=NO PR_CREATED=NO .local_TRACKED=NO GUIDANCE_WORKTREE_MODIFIED=NO",
            "REPLAY_ENTRYPOINT=scripts/rl/isaaclab/replay_physical_hoi_trace.py",
        ]
    )
    (P8 / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "P8_HANDOFF_WRITTEN", "episodes": len(episodes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
