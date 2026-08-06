#!/usr/bin/env python3
"""Freeze D.4R3 inputs and prior blocker evidence before calibration."""

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

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
PRIOR_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"
METRIC_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
TRAJECTORY_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
ARTIFACT_ROOT = REPO_ROOT / ".local/physics_consistent_retargeting"


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


def _required_inputs() -> tuple[Path, ...]:
    fixed = (
        PRIOR_ROOT / "final_summary.json",
        PRIOR_ROOT / "handoff.md",
        PRIOR_ROOT / "geometry_metric_decision.json",
        PRIOR_ROOT / "geometry_v1_attainability.json",
        PRIOR_ROOT / "no_contact_geometry_floor.json",
        PRIOR_ROOT / "stable_contact_calibration.json",
        PRIOR_ROOT / "source_kinematic_dynamic_comparison.json",
        PRIOR_ROOT / "local_contact_feasibility.json",
        METRIC_ROOT / "final_summary.json",
        METRIC_ROOT / "handoff.md",
        METRIC_ROOT / "geometry_metric_contract.json",
        METRIC_ROOT / "geometry_query_backend_contract.json",
        METRIC_ROOT / "geometry_query_numerical_tolerance.json",
        METRIC_ROOT / "runtime_collision_geometry_manifest.json",
        METRIC_ROOT / "source_runtime_penetration_170105.json",
        METRIC_ROOT / "source_runtime_penetration_170650.json",
        METRIC_ROOT / "corrected_runtime_penetration_170105.json",
        METRIC_ROOT / "corrected_runtime_penetration_170650.json",
        TRAJECTORY_ROOT / "final_summary.json",
        TRAJECTORY_ROOT / "handoff.md",
        TRAJECTORY_ROOT / "contact_topology.json",
        TRAJECTORY_ROOT / "task_semantics_170105.json",
        TRAJECTORY_ROOT / "task_semantics_170650.json",
        TRAJECTORY_ROOT / "trajectory_qualification_170105_v3.json",
        TRAJECTORY_ROOT / "trajectory_qualification_170650_v3.json",
        TRAJECTORY_ROOT / "optimizer_170105_s3.json",
        TRAJECTORY_ROOT / "optimizer_170650_s3.json",
        TRAJECTORY_ROOT / "optimizer_170105_s3.actions.npy",
        TRAJECTORY_ROOT / "optimizer_170650_s3.actions.npy",
        ARTIFACT_ROOT / "hocap_170105/manifest.json",
        ARTIFACT_ROOT / "hocap_170650/manifest.json",
        REPO_ROOT / "configs/rl/stage16/stage16d_stable_grasp_topologies.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_stable_grasp_calibration.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_dynamic_contact_reference.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_geometry_metric_v2.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_online_geometry_signal.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_geometry_optimizer.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_ppo.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_stable_grasp_recovery.yaml",
    )
    attempts = tuple(sorted(PRIOR_ROOT.glob("stable_contact_calibration_*.json")))
    return fixed + attempts


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != "feature/reference-tracking-isaaclab":
        raise RuntimeError(f"STAGE16D_BRANCH_LINEAGE_FAILURE:{branch}")
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "7488657", head],
        cwd=REPO_ROOT,
        check=False,
    )
    if lineage.returncode != 0:
        raise RuntimeError("STAGE16D_BRANCH_LINEAGE_FAILURE:7488657_NOT_ANCESTOR")
    status = _git("status", "--short", "--untracked-files=all")
    if status:
        raise RuntimeError(f"STAGE16D_PREFLIGHT_WORKTREE_NOT_CLEAN:\n{status}")
    inputs = _required_inputs()
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"STAGE16D_STABLE_GRASP_INPUT_HASH_DRIFT:MISSING:{missing}")
    hashes = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in inputs
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = REPO_ROOT / f".local/archive/stage16d_stable_grasp_baseline_{stamp}_{head[:7]}"
    archive.mkdir(parents=True)
    v1_contract = GEOMETRY_METRIC_CONTRACT.as_dict(query_contract=GEOMETRY_QUERY_CONTRACT)
    prior_decision = _load(PRIOR_ROOT / "geometry_metric_decision.json")
    prior_failures = {
        "schema_version": "Stage16DStableGraspPriorCalibrationFailuresV1",
        "decision": prior_decision,
        "attempts": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "status": _load(path).get("status"),
                "sha256": _sha256(path),
            }
            for path in sorted(PRIOR_ROOT.glob("stable_contact_calibration_*.json"))
        ],
    }
    trajectory_baseline = {
        "schema_version": "Stage16DStableGraspTrajectoryBaselineV1",
        "geometry_aware_closeout": _load(PRIOR_ROOT / "final_summary.json"),
        "physics_retargeting": _load(TRAJECTORY_ROOT / "final_summary.json")["trajectories"],
    }
    manifest = {
        "schema_version": "Stage16DStableGraspFrozenInputManifestV1",
        "status": "STAGE16D_STABLE_GRASP_INPUTS_FROZEN",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": branch,
        "head": head,
        "required_ancestor": "7488657",
        "worktree_clean": True,
        "archive": str(archive.relative_to(REPO_ROOT)),
        "input_count": len(hashes),
        "calibration_results_observed": False,
        "source_modified": False,
        "stage12_modified": False,
        "collision_assets_modified": False,
        "physics_modified": False,
        "absolute_gate_modified": False,
    }
    artifacts = {
        "frozen_manifest.json": manifest,
        "prior_attainability.json": prior_decision,
        "geometry_v1_contract.json": v1_contract,
        "prior_calibration_failures.json": prior_failures,
        "trajectory_baseline.json": trajectory_baseline,
        "hashes.json": {
            "schema_version": "Stage16DStableGraspInputHashesV1",
            "inputs": hashes,
            "geometry_v1_contract_sha256": geometry_contract_sha256(v1_contract),
        },
    }
    for name, payload in artifacts.items():
        _write(archive / name, payload)
        _write(output / ("frozen_inputs.json" if name == "frozen_manifest.json" else name), payload)
    for path in inputs:
        small_text = path.suffix.lower() in {".json", ".md", ".yaml", ".yml"}
        if small_text and path.stat().st_size <= 2_000_000:
            destination = archive / "evidence" / path.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    (archive / "README.md").write_text(
        "# Stage 16-D stable-grasp frozen baseline\n\n"
        "Small contracts and reports are copied for provenance. Arrays, datasets, "
        "assets, trajectories, and checkpoints are referenced only by SHA-256. "
        "The archive predates every D.4R3 calibration result.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "archive": manifest["archive"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
