#!/usr/bin/env python3
"""Freeze the immutable Stage-16.1a failure baseline without copying large assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.STDOUT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    args = parser.parse_args()
    head = _run("git", "rev-parse", "HEAD").strip()
    short_head = _run("git", "rev-parse", "--short", "HEAD").strip()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_name = f"stage16_controllability_failure_baseline_{timestamp}_{short_head}"
    archive = args.archive_root / archive_name
    if archive.exists():
        raise FileExistsError(f"baseline archive already exists: {archive}")
    preflight = args.report_root / "preflight"
    archive.mkdir(parents=True)
    preflight.mkdir(parents=True, exist_ok=True)
    git_status = _run("git", "status", "--short", "--untracked-files=all")
    preflight.joinpath("git_status.txt").write_text(git_status, encoding="utf-8")
    preflight.joinpath("staged.patch").write_text(_run("git", "diff", "--cached"), encoding="utf-8")
    preflight.joinpath("unstaged.patch").write_text(_run("git", "diff"), encoding="utf-8")
    preflight.joinpath("process_inventory.txt").write_text(
        _run("ps", "-eo", "pid,ppid,stat,etime,cmd"), encoding="utf-8"
    )
    _write_json(
        preflight / "ownership.json",
        {
            "start_head": head,
            "branch": _run("git", "branch", "--show-current").strip(),
            "status_before_freeze": (
                "clean" if not git_status.strip() else "preexisting_wip_preserved"
            ),
            "unrelated_files_modified": [],
            "note": "all pre-existing diffs were retained; no reset/stash/restore/clean was used",
        },
    )
    references = [
        REPO / ".local/stage16_reference_tracking_ppo/references/hocap_170105.stage16.npz",
        REPO / ".local/stage16_reference_tracking_ppo/references/hocap_170650.stage16.npz",
    ]
    meshes = [
        REPO / ".local/stage16_reference_tracking_ppo/objects/hocap_170105.obj",
        REPO / ".local/stage16_reference_tracking_ppo/objects/hocap_170650.obj",
    ]
    scene = REPO / ".local/build/stage16_reference_tracking_ppo/wuji_hand2_stage16_free_object.xml"
    existing = REPO / ".local/reports/stage16_1_3/stage16_1_controllability.json"
    for source in [*references, *meshes, scene, existing]:
        if not source.is_file():
            raise FileNotFoundError(source)
    reference_hashes = {str(path): _sha256(path) for path in references}
    object_hashes = {str(path): _sha256(path) for path in meshes}
    simulator_config = {
        "start_head": head,
        "scene": str(scene),
        "scene_sha256": _sha256(scene),
        "backend_source_at_head": _run(
            "git", "rev-parse", f"{head}:src/toporetarget/rl/environments/mujoco_backend.py"
        ).strip(),
        "gravity_mps2": [0.0, 0.0, 0.0],
        "fixed_robot_base": True,
        "synthetic_ground": False,
        "formal_termination_unchanged": {
            "object_position_m": 0.05,
            "object_axis_m": 0.05,
            "object_orientation_deg": 45.0,
        },
    }
    existing_summary = json.loads(existing.read_text(encoding="utf-8"))
    _write_json(archive / "reference_hashes.json", reference_hashes)
    _write_json(archive / "object_hashes.json", object_hashes)
    _write_json(archive / "simulator_config.json", simulator_config)
    _write_json(archive / "existing_failure_summary.json", existing_summary)
    _write_json(
        archive / "frozen_manifest.json",
        {
            "status": "STAGE16_CONTROLLABILITY_FAILURE_BASELINE_FROZEN",
            "created_at_utc": timestamp,
            "start_head": head,
            "references": reference_hashes,
            "objects": object_hashes,
            "scene_sha256": simulator_config["scene_sha256"],
            "existing_failure_summary_sha256": _sha256(existing),
            "large_artifacts_copied": False,
            "old_runs_overwritten": False,
        },
    )
    archive.joinpath("README.md").write_text(
        "# Frozen Stage-16 controllability failure baseline\n\n"
        "This archive references immutable prior Stage-16 artifacts by absolute path and SHA-256. "
        "It intentionally does not copy meshes, references, checkpoints, or prior runs.\n",
        encoding="utf-8",
    )
    _write_json(
        args.report_root / "frozen_baseline.json",
        {
            "status": "STAGE16_CONTROLLABILITY_FAILURE_BASELINE_FROZEN",
            "archive": str(archive.resolve()),
            "frozen_manifest": str((archive / "frozen_manifest.json").resolve()),
        },
    )
    print(
        json.dumps(
            {"archive": str(archive), "status": "STAGE16_CONTROLLABILITY_FAILURE_BASELINE_FROZEN"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
