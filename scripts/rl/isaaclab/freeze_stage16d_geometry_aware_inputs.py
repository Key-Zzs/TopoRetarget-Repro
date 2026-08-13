#!/usr/bin/env python3
"""Freeze the Stage 16-D geometry-aware recovery baseline before calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.contracts import (  # noqa: E402
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
    geometry_contract_sha256,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"
BASELINE_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
TRAJECTORY_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
CLIPS = ("hocap_170105", "hocap_170650")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inventory(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"STAGE16D_GEOOPT_INPUT_HASH_DRIFT:MISSING:{path}")
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != "feature/reference-tracking-isaaclab":
        raise RuntimeError(f"STAGE16D_BRANCH_LINEAGE_FAILURE:{branch}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "6bc64c598d5a7f3bea4b6e703bffbde7b86abf83", head],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("STAGE16D_BRANCH_LINEAGE_FAILURE:6bc64c5_NOT_ANCESTOR")
    status = _git("status", "--short", "--untracked-files=all")
    if status:
        raise RuntimeError(f"STAGE16D_PREFLIGHT_WORKTREE_NOT_CLEAN:\n{status}")

    source_reports = tuple(
        BASELINE_ROOT / f"source_runtime_penetration_{clip.removeprefix('hocap_')}.json"
        for clip in CLIPS
    )
    corrected_reports = tuple(
        BASELINE_ROOT / f"corrected_runtime_penetration_{clip.removeprefix('hocap_')}.json"
        for clip in CLIPS
    )
    trajectory_reports = (
        BASELINE_ROOT / "trajectory_requalification_170105.json",
        BASELINE_ROOT / "trajectory_requalification_170650.json",
        BASELINE_ROOT / "terminal_failure_analysis_170105.json",
        TRAJECTORY_ROOT / "trajectory_qualification_170105_v3.json",
        TRAJECTORY_ROOT / "trajectory_qualification_170650_v3.json",
    )
    optimizer_reports = (
        TRAJECTORY_ROOT / "optimizer_config.json",
        TRAJECTORY_ROOT / "optimizer_170105_s3.json",
        TRAJECTORY_ROOT / "optimizer_170650_s3.json",
        BASELINE_ROOT / "terminal_refinement_170105.json",
        BASELINE_ROOT / "optimizer_upgrade_170105.json",
    )
    contract_inputs = (
        BASELINE_ROOT / "final_summary.json",
        BASELINE_ROOT / "handoff.md",
        BASELINE_ROOT / "geometry_metric_contract.json",
        BASELINE_ROOT / "geometry_query_backend_contract.json",
        BASELINE_ROOT / "geometry_query_numerical_tolerance.json",
        BASELINE_ROOT / "runtime_collision_geometry_manifest.json",
        BASELINE_ROOT / "runtime_physx_geometry_crosscheck.json",
        REPO_ROOT / "configs/rl/stage16/stage16d_trajectory_gate.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_trajectory_optimizer.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_reward.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_ppo.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_geometry_gate_attainability.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_online_geometry_signal.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_geometry_optimizer.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_geometry_recovery.yaml",
    )
    trace_inputs = tuple(
        TRAJECTORY_ROOT / f"trajectory_trace_{clip.removeprefix('hocap_')}_v3.npz" for clip in CLIPS
    )
    action_inputs = tuple(
        TRAJECTORY_ROOT / f"optimizer_{clip.removeprefix('hocap_')}_s3.actions.npy"
        for clip in CLIPS
    )
    state_inputs = tuple(
        BASELINE_ROOT / f"source_collision_state_{clip.removeprefix('hocap_')}.npz"
        for clip in CLIPS
    ) + tuple(
        BASELINE_ROOT / f"corrected_collision_state_{clip.removeprefix('hocap_')}.npz"
        for clip in CLIPS
    )
    all_inputs = (
        source_reports
        + corrected_reports
        + trajectory_reports
        + optimizer_reports
        + contract_inputs
        + trace_inputs
        + action_inputs
        + state_inputs
    )
    hashes = _inventory(all_inputs)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = REPO_ROOT / f".local/archive/stage16d_geometry_aware_baseline_{stamp}_{head[:7]}"
    if archive.exists():
        raise FileExistsError(archive)
    archive.mkdir(parents=True)

    v1_contract = GEOMETRY_METRIC_CONTRACT.as_dict(query_contract=GEOMETRY_QUERY_CONTRACT)
    source_baseline = {clip: _load(path) for clip, path in zip(CLIPS, source_reports, strict=True)}
    corrected_baseline = {
        clip: _load(path) for clip, path in zip(CLIPS, corrected_reports, strict=True)
    }
    trajectory_baseline = {
        "final": _load(BASELINE_ROOT / "final_summary.json")["trajectories"],
        "reports": [
            row
            for row in hashes
            if Path(row["path"]) in {path.relative_to(REPO_ROOT) for path in trajectory_reports}
        ],
    }
    optimizer_baseline = {
        "legacy_penetration_placeholder": "penetration_m = 0.0",
        "reports": [
            row
            for row in hashes
            if Path(row["path"]) in {path.relative_to(REPO_ROOT) for path in optimizer_reports}
        ],
    }
    frozen_manifest = {
        "schema_version": "Stage16DGeometryAwareFrozenInputManifestV1",
        "status": "STAGE16D_GEOOPT_INPUTS_FROZEN",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": branch,
        "head": head,
        "required_ancestor": "6bc64c598d5a7f3bea4b6e703bffbde7b86abf83",
        "worktree_clean": True,
        "archive": str(archive.relative_to(REPO_ROOT)),
        "input_count": len(hashes),
        "formal_results_computed": False,
        "source_modified": False,
        "stage12_modified": False,
        "absolute_geometry_gate_modified": False,
    }
    _write(archive / "frozen_manifest.json", frozen_manifest)
    _write(archive / "geometry_v1_contract.json", v1_contract)
    _write(archive / "source_penetration_baseline.json", source_baseline)
    _write(archive / "corrected_penetration_baseline.json", corrected_baseline)
    _write(archive / "trajectory_baseline.json", trajectory_baseline)
    _write(archive / "optimizer_baseline.json", optimizer_baseline)
    _write(
        archive / "hashes.json",
        {
            "schema_version": "Stage16DGeometryAwareBaselineHashesV1",
            "inputs": hashes,
            "v1_contract_sha256": geometry_contract_sha256(v1_contract),
        },
    )
    (archive / "README.md").write_text(
        "# Stage 16-D geometry-aware frozen baseline\n\n"
        "This archive contains small contracts, reports, and hashes only. It predates the "
        "attainability decision and all geometry-aware optimizer results. Source data, assets, "
        "trajectory arrays, checkpoints, and datasets are referenced by hash and are not copied.\n",
        encoding="utf-8",
    )
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "frozen_inputs.json", frozen_manifest)
    shutil.copy2(archive / "hashes.json", output / "frozen_input_hashes.json")
    print(json.dumps({"status": frozen_manifest["status"], "archive": frozen_manifest["archive"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
