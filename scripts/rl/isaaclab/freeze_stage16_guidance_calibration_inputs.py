#!/usr/bin/env python3
"""Freeze completed V3/V4 action traces and checkpoints for G3/G5.

Only immutable, completed artifacts in the physical worktree are read.  Every
copy and its source are SHA-256 checked before the manifest is written here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PHYSICAL_ROOT = Path("/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro")
SOURCE_REPORTS = PHYSICAL_ROOT / ".local/reports"
OUTPUT_ROOT = REPO_ROOT / ".local/frozen_baselines/guidance_calibration_v1"
REPORT = REPO_ROOT / ".local/reports/stage16_guidance_g0_g5/g3/frozen_baseline_inputs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path, expected_hash: str) -> dict[str, str]:
    if not source.is_file():
        raise FileNotFoundError(f"GUIDANCE_G3_FROZEN_SOURCE_MISSING:{source}")
    source_hash = sha256(source)
    if source_hash != expected_hash:
        raise RuntimeError(f"GUIDANCE_G3_FROZEN_SOURCE_HASH_MISMATCH:{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_hash = sha256(destination)
    if copied_hash != source_hash:
        raise RuntimeError(f"GUIDANCE_G3_FROZEN_COPY_HASH_MISMATCH:{destination}")
    return {
        "source_path": str(source),
        "source_sha256": source_hash,
        "copied_path": str(destination),
        "copied_sha256": copied_hash,
        "source_read_only": "true",
    }


def summary() -> tuple[dict[str, Any], dict[str, Any]]:
    v3_path = SOURCE_REPORTS / "stage16d_reward_v3_pairforce_unblock/final_summary.json"
    v4_path = SOURCE_REPORTS / "stage16d_strict_per_finger_v4/final_summary.json"
    return (
        json.loads(v3_path.read_text(encoding="utf-8")),
        json.loads(v4_path.read_text(encoding="utf-8")),
    )


def main() -> int:
    v3, v4 = summary()
    records: dict[str, dict[str, Any]] = {}
    trace_paths = {
        ("v3", "hocap_170105"): SOURCE_REPORTS
        / "stage16d_reward_v3_pairforce_unblock/hocap_170105/formal/hocap_170105"
        / "ppo_v3_formal_selected_2129920_trace.npz",
        ("v3", "hocap_170650"): SOURCE_REPORTS
        / "stage16d_reward_v3_pairforce_unblock/hocap_170650/formal/hocap_170650"
        / "v3_formal_selected_2129920_trace.npz",
        ("v4", "hocap_170105"): SOURCE_REPORTS
        / "stage16d_strict_per_finger_v4/hocap_170105/formal/hocap_170105"
        / "ppo_v4_formal_selected_1064960_trace_replica0.npz",
        ("v4", "hocap_170650"): SOURCE_REPORTS
        / "stage16d_strict_per_finger_v4/hocap_170650/formal/hocap_170650"
        / "ppo_v4_formal_selected_1064960_trace_replica0.npz",
    }
    for version, source_summary, completion in (
        ("v3", v3, "STAGE16D_REWARD_V3_FORMAL_COMPLETE"),
        ("v4", v4, "STAGE16D_STRICT_V4_FORMAL_COMPLETE"),
    ):
        for clip in ("hocap_170105", "hocap_170650"):
            formal_key = "v3_formal_qualification" if version == "v3" else "formal_qualification"
            formal = source_summary["clips"][clip][formal_key]
            if formal["status"] != completion:
                raise RuntimeError(f"GUIDANCE_G3_BASELINE_NOT_COMPLETE:{version}:{clip}")
            checkpoint = Path(formal["checkpoint"])
            checkpoint_hash = formal["checkpoint_sha256"]
            trace = trace_paths[(version, clip)]
            trace_hash = sha256(trace)
            base = OUTPUT_ROOT / version / clip
            records[f"{version}_{clip}"] = {
                "version": version,
                "clip": clip,
                "formal_completion_status": formal["status"],
                "checkpoint": copy_verified(checkpoint, base / "checkpoint.pt", checkpoint_hash),
                "action_trace": copy_verified(trace, base / "action_trace.npz", trace_hash),
            }
    reward_sources = {
        "v3_contract": SOURCE_REPORTS
        / "stage16d_reward_v3_pairforce_unblock/reward_v3_contract.json",
        "v4_contract": SOURCE_REPORTS / "stage16d_strict_per_finger_v4/strict_v4_contract.json",
        "v3_mask_170105": SOURCE_REPORTS
        / "stage16d_reward_v3_contact/reference_contact_mask_170105.npz",
        "v3_mask_170650": SOURCE_REPORTS
        / "stage16d_reward_v3_contact/reference_contact_mask_170650.npz",
        "v4_mask_170105": SOURCE_REPORTS
        / "stage16d_strict_per_finger_v4/strict_source_contact_mask_hocap_170105.npz",
        "v4_mask_170650": SOURCE_REPORTS
        / "stage16d_strict_per_finger_v4/strict_source_contact_mask_hocap_170650.npz",
    }
    reward_artifacts = {
        key: copy_verified(source, OUTPUT_ROOT / "reward_inputs" / source.name, sha256(source))
        for key, source in reward_sources.items()
    }
    payload = {
        "schema_version": "Stage16GuidanceG3FrozenBaselinesV1",
        "source_worktree": str(PHYSICAL_ROOT),
        "source_read_only": True,
        "records": records,
        "reward_artifacts": reward_artifacts,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GUIDANCE_G3_BASELINES_FROZEN", "report": str(REPORT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
