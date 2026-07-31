#!/usr/bin/env python3
"""Record the current dynamic-reference and Pen-Spin availability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.contracts import Stage16ReferenceClip


def main() -> int:
    root = Path(".local/reports/stage16_reference_tracking_ppo")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-reference",
        action="append",
        type=Path,
        default=[],
        help="User-approved Stage16ReferenceClip input; may be repeated.",
    )
    parser.add_argument("--operator-authorized", action="store_true")
    args = parser.parse_args()
    root.mkdir(parents=True, exist_ok=True)
    paused = Path(".local/control/final_jobs/PAUSED")
    accepted = [Stage16ReferenceClip.from_npz(path) for path in args.accepted_reference]
    accepted_inventory = [
        {
            "path": str(path.resolve()),
            "content_hash": clip.content_hash(),
            "frames": clip.frame_count,
            "sampling_hz": clip.control_hz,
            "source_sequence": clip.provenance["dataset_provenance"]["source_sequence"],
        }
        for path, clip in zip(args.accepted_reference, accepted, strict=True)
    ]
    final_jobs = {
        "state": "USER_AUTHORIZED_FINAL_TASKS"
        if args.operator_authorized
        else (paused.read_text(encoding="utf-8").strip() if paused.is_file() else "UNKNOWN"),
        "new_final_tasks_allowed": args.operator_authorized,
        "reason": (
            "Current user explicitly authorized Stage-16 final tasks."
            if args.operator_authorized
            else "Current operator pause manifest forbids new final tasks."
        ),
    }
    inventory = {
        "hocap_raw_root": "/mnt/nas/storage/Ref2Dex_storage/HOCap",
        "hocap_raw_data_available": Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data").is_dir(),
        "accepted_dynamic_hocap_robot_reference_available": bool(accepted),
        "accepted_references": accepted_inventory,
        "historical_pre_source_contract_stage12_outputs_eligible": False,
        "blocker": (
            None
            if accepted and args.operator_authorized
            else "An accepted dynamic HOCap reference and user/operator authorization are required."
        ),
        "final_job_control": final_jobs,
    }
    penspin = {
        "status": "STAGE16_PENSPIN_OUT_OF_SCOPE_BY_USER",
        "searched_roots": [
            "/mnt/nas/storage/Ref2Dex_storage",
            ".local/experiments",
            ".local/reports",
        ],
        "provenance": "User explicitly excluded Pen-Spin from this functional HOCap run.",
    }
    selection = {
        "status": "HOCAP_USER_APPROVED_FUNCTIONAL_SET",
        "author_exact_clip_ids": "AUTHOR_EXACT_CLIP_IDS_UNAVAILABLE",
        "frozen_selection_created": bool(accepted),
        "actual_clip_count": len(accepted),
        "paper_clip_target": 32,
        "reason": (
            "User-approved functional set; it is intentionally not a paper HOCap-32 selection."
            if accepted
            else (
                "Selection cannot be frozen until dynamic retargeted references pass the "
                "eligibility gate."
            )
        ),
    }
    (root / "reference_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "penspin_availability.json").write_text(
        json.dumps(penspin, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "reference_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reference_inventory": inventory, "penspin": penspin}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
