#!/usr/bin/env python3
"""Freeze the immutable inputs for the bounded Stage 16-C.3R2 closeout.

The archive is deliberately hash-only: it does not duplicate generated USDs,
reference arrays, or prior reports.  A later phase must call ``--verify`` and
will fail closed when any frozen input differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r2_c5"
ARCHIVE_ROOT = REPO_ROOT / ".local/archive"

_REQUIRED = (
    ".local/reports/stage16c2_c5_isaaclab/final_summary.json",
    ".local/reports/stage16c2_c5_isaaclab/c2_environment_contract.json",
    ".local/reports/stage16c3_repair_c5_oracle/final_summary.json",
    ".local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json",
    ".local/reports/stage16c3_repair_c5_oracle/c3_final_wrist_qualification.json",
    ".local/reports/stage16c_isaaclab_platform/final_summary.json",
    ".local/reports/stage16c1_asset_migration/final_summary.json",
    ".local/reports/stage16c1_asset_migration/wuji_asset_manifest.json",
    ".local/reports/stage16c1_asset_migration/hocap_170105_asset_manifest.json",
    ".local/reports/stage16c1_asset_migration/hocap_170650_asset_manifest.json",
    ".local/generated_assets/isaaclab/wuji_hand2_beta1/wujihand2.usd",
    ".local/generated_assets/isaaclab/wuji_hand2_beta1/configuration/wujihand2_physics.usd",
    ".local/generated_assets/isaaclab/hocap_170105/hocap_170105.usda",
    ".local/generated_assets/isaaclab/hocap_170650/hocap_170650.usda",
    ".local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170105.world_wrist.stage16.npz",
    ".local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170650.world_wrist.stage16.npz",
    "configs/rl/stage16/isaaclab_asset_validation.yaml",
    "configs/rl/stage16/isaaclab_hocap_objects.yaml",
    "configs/rl/stage16/isaaclab_platform.yaml",
    "configs/rl/stage16/isaaclab_world_wrist_control.yaml",
    "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
    "configs/rl/stage16/isaaclab_world_wrist_reward.yaml",
    "configs/rl/stage16/isaaclab_wuji_hand2_beta1.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _paths() -> dict[str, str]:
    missing = [relative for relative in _REQUIRED if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16C3R2_INPUT_MISSING: {missing}")
    return {relative: _sha256(REPO_ROOT / relative) for relative in _REQUIRED}


def _partition_hashes(hashes: dict[str, str]) -> dict[str, dict[str, str]]:
    return {
        "runtime_hashes": {
            path: value for path, value in hashes.items() if path.startswith(".local/reports/")
        },
        "asset_hashes": {
            path: value
            for path, value in hashes.items()
            if path.startswith(".local/generated_assets/")
        },
        "reference_hashes": {
            path: value
            for path, value in hashes.items()
            if path.startswith(".local/stage16_reference_tracking_ppo/")
        },
        "controller_config_hashes": {
            path: value for path, value in hashes.items() if path.startswith("configs/")
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_payload(hashes: dict[str, str], archive: Path) -> dict[str, Any]:
    old_wrist = _load(
        REPO_ROOT / ".local/reports/stage16c3_repair_c5_oracle/c3_final_wrist_qualification.json"
    )
    old_contact = _load(
        REPO_ROOT / ".local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json"
    )
    start_head = _git("rev-parse", "HEAD")
    return {
        "schema_version": "toporetarget.stage16c3r2.frozen_inputs.v1",
        "status": "STAGE16C3R2_INPUTS_FROZEN",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "branch": _git("branch", "--show-current"),
        "start_head": start_head,
        "archive": str(archive.resolve()),
        "hashes": hashes,
        "preexisting_wrist_status": old_wrist.get("status"),
        "preexisting_wrist_blocker": old_wrist.get("blocker"),
        "preexisting_contact_status": old_contact.get("contact_causality"),
        "preexisting_contact_attempts": old_contact.get("attempts"),
        "git_status_short": _git("status", "--short", "--untracked-files=all"),
        "hash_drift_status": "STAGE16C3R2_INPUT_HASH_DRIFT",
    }


def _verify(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("hashes")
    if not isinstance(expected, dict):
        raise ValueError("invalid frozen manifest: hashes must be a mapping")
    observed = _paths()
    changed = {
        path: {"expected": expected.get(path), "observed": observed.get(path)}
        for path in sorted(set(expected) | set(observed))
        if expected.get(path) != observed.get(path)
    }
    return {
        "status": "STAGE16C3R2_INPUT_HASH_DRIFT" if changed else "STAGE16C3R2_INPUT_HASHES_MATCH",
        "changed": changed,
        "manifest": manifest.get("archive"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verify-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive and args.verify:
        raise SystemExit("use exactly one of --archive or --verify")
    if args.verify_output and not args.verify:
        raise SystemExit("--verify-output requires --verify")
    if args.verify:
        result = _verify(_load(args.verify))
        if args.verify_output:
            _write_json(args.verify_output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "STAGE16C3R2_INPUT_HASHES_MATCH" else 2
    short_head = _git("rev-parse", "--short", "HEAD")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = args.archive or ARCHIVE_ROOT / f"stage16c3r2_baseline_{timestamp}_{short_head}"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite baseline archive: {archive}")
    hashes = _paths()
    payload = _baseline_payload(hashes, archive)
    archive.mkdir(parents=True)
    _write_json(archive / "frozen_manifest.json", payload)
    for name, value in _partition_hashes(hashes).items():
        _write_json(archive / f"{name}.json", value)
    _write_json(
        archive / "wrist_baseline.json",
        {
            "source": str(
                REPO_ROOT
                / ".local/reports/stage16c3_repair_c5_oracle/c3_final_wrist_qualification.json"
            ),
            "sha256": hashes[
                ".local/reports/stage16c3_repair_c5_oracle/c3_final_wrist_qualification.json"
            ],
            "status": payload["preexisting_wrist_status"],
            "blocker": payload["preexisting_wrist_blocker"],
        },
    )
    _write_json(
        archive / "contact_crash_baseline.json",
        {
            "source": str(
                REPO_ROOT / ".local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json"
            ),
            "sha256": hashes[
                ".local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json"
            ],
            "status": payload["preexisting_contact_status"],
            "attempts": payload["preexisting_contact_attempts"],
        },
    )
    _write_json(REPORT_ROOT / "frozen_baseline.json", payload)
    print(json.dumps({"status": payload["status"], "archive": str(archive)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
