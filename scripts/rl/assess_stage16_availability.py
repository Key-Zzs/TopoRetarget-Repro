#!/usr/bin/env python3
"""Record the current dynamic-reference and Pen-Spin availability gates."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(".local/reports/stage16_reference_tracking_ppo")
    root.mkdir(parents=True, exist_ok=True)
    paused = Path(".local/control/final_jobs/PAUSED")
    final_jobs = {
        "state": paused.read_text(encoding="utf-8").strip() if paused.is_file() else "UNKNOWN",
        "new_final_tasks_allowed": False,
        "reason": "Current operator pause manifest forbids new final tasks.",
    }
    inventory = {
        "hocap_raw_root": "/mnt/nas/storage/Ref2Dex_storage/HOCap",
        "hocap_raw_data_available": Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data").is_dir(),
        "accepted_dynamic_hocap_robot_reference_available": False,
        "historical_pre_source_contract_stage12_outputs_eligible": False,
        "blocker": (
            "No accepted post-source-contract-repair dynamic HO-Cap RobotReference is "
            "available; final-job gate is operator-paused."
        ),
        "final_job_control": final_jobs,
    }
    penspin = {
        "status": "STAGE16_PENSPIN_DATA_UNAVAILABLE",
        "searched_roots": [
            "/mnt/nas/storage/Ref2Dex_storage",
            ".local/experiments",
            ".local/reports",
        ],
        "provenance": "No real Pen-Spin dataset was found. No substitute data is used.",
    }
    selection = {
        "status": "HOCAP_32_PROTOCOL_SET_RECONSTRUCTED",
        "author_exact_clip_ids": "AUTHOR_EXACT_CLIP_IDS_UNAVAILABLE",
        "frozen_selection_created": False,
        "reason": (
            "Selection cannot be frozen until dynamic retargeted references pass "
            "the eligibility gate."
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
