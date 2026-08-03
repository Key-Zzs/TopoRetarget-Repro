#!/usr/bin/env python3
"""Freeze Stage 16-C.3R3 inputs and retain a non-overwriting preflight bundle."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r3_joint_dynamics_c5"
ARCHIVE_ROOT = REPO_ROOT / ".local/archive"
_INPUTS = (
    ".local/reports/stage16c3r2_c5/final_summary.json",
    ".local/reports/stage16c3r2_c5/handoff.md",
    ".local/reports/stage16c3r2_c5/c3/path_b_explicit_3p3r_noncontact.json",
    ".local/reports/stage16c2_c5_isaaclab/c2_environment_contract.json",
    ".local/reports/stage16c1_asset_migration/wuji_asset_manifest.json",
    ".local/reports/stage16c1_asset_migration/hocap_170105_asset_manifest.json",
    ".local/reports/stage16c1_asset_migration/hocap_170650_asset_manifest.json",
    ".local/generated_assets/isaaclab/wuji_hand2_beta1_explicit_virtual_wrist/wujihand2_explicit_virtual_wrist.usda",
    ".local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170105.world_wrist.stage16.npz",
    ".local/stage16_reference_tracking_ppo/world_wrist_references/hocap_170650.world_wrist.stage16.npz",
    "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
    "configs/rl/stage16/isaaclab_world_wrist_control.yaml",
    "configs/rl/stage16/isaaclab_world_wrist_reward.yaml",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*command: str) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip() + "\n"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verify-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_output is not None and args.verify is None:
        raise ValueError("--verify-output requires --verify")
    if args.verify is not None:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        observed = {path: _hash(REPO_ROOT / path) for path in manifest["hashes"]}
        changed = {
            path: {"expected": manifest["hashes"][path], "observed": observed[path]}
            for path in observed
            if manifest["hashes"][path] != observed[path]
        }
        result = {
            "status": "STAGE16C3R3_INPUT_HASH_DRIFT"
            if changed
            else "STAGE16C3R3_INPUT_HASHES_MATCH",
            "changed": changed,
            "verified_manifest": str(args.verify),
            "hash_count": len(observed),
        }
        if args.verify_output is not None:
            if args.verify_output.exists():
                raise FileExistsError(f"STAGE16C3R3_VERIFY_REFUSES_OVERWRITE: {args.verify_output}")
            _write(args.verify_output, result)
        print(json.dumps(result, sort_keys=True))
        return 2 if changed else 0
    missing = [path for path in _INPUTS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16C3R3_INPUT_MISSING: {missing}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_head = _run("git", "rev-parse", "--short", "HEAD").strip()
    archive = ARCHIVE_ROOT / f"stage16c3r3_explicit_wrist_baseline_{timestamp}_{short_head}"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    hashes = {path: _hash(REPO_ROOT / path) for path in _INPUTS}
    manifest = {
        "schema_version": "toporetarget.stage16c3r3.frozen_inputs.v1",
        "status": "STAGE16C3R3_INPUTS_FROZEN",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "branch": _run("git", "branch", "--show-current").strip(),
        "head": _run("git", "rev-parse", "HEAD").strip(),
        "hash_drift_status": "STAGE16C3R3_INPUT_HASHES_FROZEN_NOT_YET_REVERIFIED",
        "hashes": hashes,
        "archive": str(archive),
    }
    _write(archive / "frozen_manifest.json", manifest)
    _write(
        archive / "explicit_wrist_asset.json",
        {k: v for k, v in hashes.items() if "explicit_virtual_wrist" in k},
    )
    _write(archive / "c2_contracts.json", {k: v for k, v in hashes.items() if "c2_" in k})
    _write(
        archive / "wrist_pd_baseline.json",
        json.loads((REPO_ROOT / _INPUTS[0]).read_text(encoding="utf-8")),
    )
    _write(
        archive / "reference_hashes.json", {k: v for k, v in hashes.items() if k.endswith(".npz")}
    )
    _write(archive / "object_hashes.json", {k: v for k, v in hashes.items() if "hocap_" in k})
    _write(
        archive / "runtime_hashes.json",
        {k: v for k, v in hashes.items() if k.startswith("configs/")},
    )
    _write(
        archive / "README.md",
        "# Frozen Stage 16-C.3R3 baseline\n\nHash-only immutable input archive.\n",
    )
    preflight = REPORT_ROOT / "preflight"
    _write(preflight / "git_status.txt", _run("git", "status", "--short", "--untracked-files=all"))
    _write(preflight / "staged.patch", _run("git", "diff", "--cached"))
    _write(preflight / "unstaged.patch", _run("git", "diff"))
    _write(
        preflight / "ownership.json", {"managed_pids": [], "unrelated_processes_terminated": False}
    )
    _write(preflight / "processes.txt", _run("ps", "-eo", "pid,ppid,pgid,stat,pcpu,pmem,nlwp,args"))
    _write(preflight / "gpu_before.txt", _run("nvidia-smi"))
    _write(preflight / "resources.txt", _run("df", "-h", ".") + _run("free", "-h"))
    _write(REPORT_ROOT / "frozen_baseline.json", manifest)
    print(json.dumps({"status": manifest["status"], "archive": str(archive)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
