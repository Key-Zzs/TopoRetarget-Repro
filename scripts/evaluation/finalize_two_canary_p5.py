#!/usr/bin/env python3
# ruff: noqa: E501 - report generator contains long self-contained HTML and markdown lines.
"""Finalize the receipt-bound P5 HTML review package for two canaries.

This command only packages already completed exact-retarget outputs and their
semantic qualifications.  It never runs a solver, physicalization, support,
PhysX, PPO, reward, RSE, PF, or DF stage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".local/reports/dataset_semantic_authority_two_clip_canary"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            ".local/reports/dataset_semantic_authority_two_clip_canary/"
            "p5_two_canary_retarget/two_canary_manifest.json"
        ),
    )
    parser.add_argument(
        "--start-head",
        default="598d80f458f2b3cdaa5a72487c7ce3eeed5e63b4",
        help="Git HEAD recorded before the bounded P5 execution began.",
    )
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _wrapper_html(*, manifest_row: dict[str, Any], source_html: Path, source_sha256: str) -> str:
    episode_id = html.escape(str(manifest_row["episode_id"]))
    target = html.escape(str(manifest_row["target_object"]))
    hand = html.escape(str(manifest_row["active_hand"]))
    frame_range = html.escape(str(manifest_row["selected_frame_range"]))
    source = html.escape(os.path.relpath(source_html, source_html.parents[4]))
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Two-canary P5 review: {episode_id}</title>
<style>body {{ margin:0; font-family: sans-serif; background:#111; color:#eee; }}
header {{ padding:16px 22px; background:#202b36; }}
header h1 {{ margin:0 0 8px; font-size:20px; }}
header p {{ margin:4px 0; }}
.layers {{ display:flex; gap:16px; flex-wrap:wrap; color:#9fe1ff; font-weight:600; }}
iframe {{ display:block; width:100%; height:calc(100vh - 152px); border:0; background:white; }}</style>
</head>
<body>
<header>
<h1>P5 receipt-bound semantic retarget review</h1>
<p><strong>Episode:</strong> {episode_id} &nbsp; <strong>Active hand:</strong> {hand}
&nbsp; <strong>Target object:</strong> {target} &nbsp; <strong>Frames:</strong> {frame_range}</p>
<p class="layers">RAW HUMAN · CANONICAL HUMAN · WARM ROBOT · FINAL ROBOT · TARGET OBJECT</p>
<p>Embedded source viewer SHA-256: {source_sha256}</p>
</header>
<iframe title="Receipt-bound RAW/CANONICAL/WARM/FINAL viewer" src="{source}"></iframe>
</body></html>
"""


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    if manifest.get("schema_version") != "TwoNewSemanticCanariesV1" or not isinstance(
        episodes, list
    ):
        raise ValueError("TWO_CANARY_MANIFEST_INVALID")
    if len(episodes) != 2 or manifest.get("clips") != episodes:
        raise ValueError("TWO_CANARY_MANIFEST_MUST_CONTAIN_TWO_IDENTICAL_EPISODES_AND_CLIPS")
    if manifest.get("status") != "FROZEN_BEFORE_RETARGET" or manifest.get(
        "downstream_outcomes_used"
    ):
        raise ValueError("TWO_CANARY_MANIFEST_NOT_PRE_RETARGET_FROZEN")
    manifest_core = dict(manifest)
    manifest_hash = str(manifest_core.pop("manifest_sha256", ""))
    digest = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    if digest != manifest_hash:
        raise ValueError("TWO_CANARY_MANIFEST_HASH_MISMATCH")
    return manifest, episodes


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest, rows = _load_manifest(manifest_path)
    p5 = root / "p5_two_canary_retarget"
    canary_summaries: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        canary_root = p5 / f"canary_{rank}"
        episode_id = str(row["episode_id"])
        episode_report = canary_root / "report" / "episodes" / episode_id
        geometric_receipt = episode_report / "geometric_retarget_receipt.json"
        qualification = episode_report / "retarget" / "semantic_qualification.json"
        source_html = episode_report / "retarget" / "continuous_refinement_visualization.html"
        for artifact in (geometric_receipt, qualification, source_html):
            if not artifact.is_file():
                raise FileNotFoundError(f"P5_CANARY_ARTIFACT_MISSING:{artifact}")
        geometric = json.loads(geometric_receipt.read_text(encoding="utf-8"))
        semantic = json.loads(qualification.read_text(encoding="utf-8"))
        if (
            geometric.get("episode_id") != episode_id
            or geometric.get("selection_manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise ValueError(f"P5_CANARY_RECEIPT_BINDING_MISMATCH:{episode_id}")
        qualification_result = semantic.get("qualification") or semantic.get("final", {}).get(
            "qualification", {}
        )
        if (
            semantic.get("identifier") != episode_id
            or semantic.get("object_id") != row["target_object"]
        ):
            raise ValueError(f"P5_CANARY_SEMANTIC_BINDING_MISMATCH:{episode_id}")
        source_sha256 = _sha256(source_html)
        wrapper = canary_root / "visualization.html"
        wrapper.write_text(
            _wrapper_html(manifest_row=row, source_html=source_html, source_sha256=source_sha256),
            encoding="utf-8",
        )
        wrapper_sha256 = _sha256(wrapper)
        visualization_receipt = {
            "schema_version": "TwoCanaryReceiptBoundVisualizationV1",
            "status": "RECEIPT_BOUND_HTML_READY_FOR_MANUAL_REVIEW",
            "episode_id": episode_id,
            "clip_id": row["clip_id"],
            "active_hand": row["active_hand"],
            "target_object": row["target_object"],
            "selected_frame_range": row["selected_frame_range"],
            "manifest_sha256": manifest["manifest_sha256"],
            "geometric_receipt_sha256": _sha256(geometric_receipt),
            "semantic_qualification_sha256": _sha256(qualification),
            "source_viewer": {"path": str(source_html), "sha256": source_sha256},
            "visualization": {"path": str(wrapper), "sha256": wrapper_sha256},
            "required_layers": [
                "RAW HUMAN",
                "CANONICAL HUMAN",
                "WARM ROBOT",
                "FINAL ROBOT",
                "TARGET OBJECT",
            ],
            "physicalization_started": False,
        }
        _write_json(canary_root / "visualization_receipt.json", visualization_receipt)
        canary_summaries.append(
            {
                "rank": rank,
                "episode_id": episode_id,
                "active_hand": row["active_hand"],
                "target_object": row["target_object"],
                "selected_frame_range": row["selected_frame_range"],
                "geometric_status": geometric.get("status"),
                "semantic_status": qualification_result.get("status"),
                "semantic_gate_sha256": qualification_result.get("gate_contract_sha256"),
                "semantic_root_cause": semantic.get("root_cause"),
                "earliest_divergence": semantic.get("earliest_divergence"),
                "retarget_output": semantic.get("artifacts", {}).get("final", {}).get("path"),
                "retarget_output_sha256": semantic.get("artifacts", {})
                .get("final", {})
                .get("sha256"),
                "visualization": str(wrapper),
                "visualization_sha256": wrapper_sha256,
                "visualization_receipt": str(canary_root / "visualization_receipt.json"),
            }
        )
    execution_head = _git_head()
    final_summary = {
        "schema_version": "DatasetSemanticAuthorityP5FinalSummaryV1",
        "status": "WAITING_FOR_USER_RETARGET_HTML_ACCEPTANCE",
        "created_utc": _utc(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "start_head": args.start_head,
        "p5_execution_head": execution_head,
        "final_head": execution_head,
        "manifest": {"path": str(manifest_path), "sha256": manifest["manifest_sha256"]},
        "canaries": canary_summaries,
        "user_decisions_required": ["CANARY_1=APPROVE|REJECT", "CANARY_2=APPROVE|REJECT"],
        "P6_P8_STARTED": False,
        "physicalization_started": False,
    }
    _write_json(p5 / "final_summary.json", final_summary)
    _write_json(
        p5 / "pause_receipt.json",
        {
            "schema_version": "DatasetSemanticAuthorityP5PauseReceiptV1",
            "status": final_summary["status"],
            "P6_P8_STARTED": False,
            "physicalization_started": False,
            "p5_execution_head": execution_head,
            "manifest_sha256": manifest["manifest_sha256"],
            "canary_count": 2,
            "decisions_required": final_summary["user_decisions_required"],
            "html_sha256": {
                str(item["rank"]): item["visualization_sha256"] for item in canary_summaries
            },
            "retarget_output_sha256": {
                str(item["rank"]): item["retarget_output_sha256"] for item in canary_summaries
            },
            "semantic_gate_sha256": {
                str(item["rank"]): item["semantic_gate_sha256"] for item in canary_summaries
            },
        },
    )
    _write_json(
        p5 / "resume_contract.json",
        {
            "schema_version": "DatasetSemanticAuthorityP5ResumeContractV1",
            "pause_status": final_summary["status"],
            "required_user_reply": "CANARY_1=APPROVE and CANARY_2=APPROVE",
            "accepted_reply_forms": [
                "CANARY_1=APPROVE; CANARY_2=APPROVE",
                "CANARY_1=REJECT; CANARY_2=APPROVE",
                "CANARY_1=APPROVE; CANARY_2=REJECT",
                "CANARY_1=REJECT; CANARY_2=REJECT",
            ],
            "resume_integrity": {
                "p5_execution_head": execution_head,
                "manifest_sha256": manifest["manifest_sha256"],
                "outcomes_used_for_selection": False,
                "html_sha256": {
                    str(item["rank"]): item["visualization_sha256"] for item in canary_summaries
                },
                "retarget_output_sha256": {
                    str(item["rank"]): item["retarget_output_sha256"] for item in canary_summaries
                },
                "semantic_gate_sha256": {
                    str(item["rank"]): item["semantic_gate_sha256"] for item in canary_summaries
                },
            },
            "P6_P8_STARTED": False,
        },
    )
    certification = json.loads(
        (root / "p3_golden_suite" / "certification.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(
        (root / "p4_hocap_semantic_preflight" / "corpus_summary.json").read_text(encoding="utf-8")
    )
    wrong_cases = json.loads(
        (root / "p1_target_object_audit" / "wrong_target_cases.json").read_text(encoding="utf-8")
    )
    canary_lines = [
        "| Canary | Episode | Active hand | Target | Frames | Machine semantic | Human decision | Retarget | HTML |",
        "|---|---|---|---|---:|---|---|---|---|",
    ]
    for item in canary_summaries:
        canary_lines.append(
            f"| {item['rank']} | `{item['episode_id']}` | `{item['active_hand']}` | `{item['target_object']}` | "
            f"`{item['selected_frame_range']}` | `{item['semantic_status']}` | `PENDING` | `{item['geometric_status']}` | "
            f"`{item['visualization']}` |"
        )
    wrong_lines = [
        "| Episode | Old selected object | Correct object | Selector valid? | Binding valid? | Root cause |",
        "|---|---|---|---:|---:|---|",
    ]
    for case in wrong_cases:
        wrong_lines.append(
            f"| `{case['case_id']}` | `{case['old_selected_object']}` | `{case['correct_object']}` | "
            f"`{case['selection_valid']}` | `{case['binding_valid']}` | `{case['root_cause']}` |"
        )
    handoff = "\n".join(
        [
            "# Dataset Semantic Authority + Two-Canary Retarget Handoff",
            "",
            f"GOAL_STATUS={final_summary['status']}",
            "",
            "## Git",
            "",
            f"- branch: `{final_summary['branch']}`",
            f"- START_HEAD: `{args.start_head}`",
            f"- P5_EXECUTION_HEAD: `{execution_head}`",
            f"- FINAL_HEAD: `{execution_head}`",
            "- push: NO; PR: NO",
            "",
            "## Two wrong-target root causes",
            "",
            *wrong_lines,
            "",
            "## Dataset Semantic Authority",
            "",
            "`DatasetSemanticAuthorityV1` -> `CanonicalHOIRecordV1` -> `TargetObjectAuthorityV1` + `ObjectAssetBindingV1` -> `HOISemanticPreflightV1` -> `RetargetSemanticValidityV1`.",
            "Ambiguous target/object binding, incomplete lifecycle, bimanual same-object, and frame/time defects are fail-closed.",
            "",
            "## HOCap corpus preflight",
            "",
            f"- raw sequences: `{corpus['raw_sequences']}`; episode candidates: `{corpus['episode_candidates']}`",
            f"- semantic counts: `{corpus['semantic_counts']}`",
            f"- target counts: `{corpus['target_counts']}`",
            f"- binding counts: `{corpus['binding_counts']}`",
            "",
            "## Golden suite",
            "",
            f"`{certification['status']}`; positive_pass=`{certification['positive_pass']}`, real_negative_detected=`{certification['real_negative_detected']}`, synthetic_negative_detected=`{certification['synthetic_negative_detected']}`.",
            "",
            "## Two retarget canaries",
            "",
            *canary_lines,
            "",
            "## P5 gate",
            "",
            "Both canaries are receipt-bound and machine-semantic PASS. Human review is still required. This first execution stopped before P6-P8: no PPO, support, PhysX, reward, RSE, PF, DF, or physicalization was started.",
            "",
            "Reply independently with `CANARY_1=APPROVE / REJECT` and `CANARY_2=APPROVE / REJECT`. Only two explicit APPROVE decisions may resume the downstream route.",
            "",
        ]
    )
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    (root / "final_summary.md").write_text(
        "# P5 final summary\n\n"
        f"Status: `{final_summary['status']}`\n\n"
        f"Two canaries: `{canary_summaries[0]['semantic_status']}`, `{canary_summaries[1]['semantic_status']}`.\n\n"
        "P6-P8 started: `false`. Physicalization started: `false`.\n\n"
        "Required decisions: `CANARY_1=APPROVE|REJECT`, `CANARY_2=APPROVE|REJECT`.\n",
        encoding="utf-8",
    )
    _write_json(root / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
