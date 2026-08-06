#!/usr/bin/env python3
"""Freeze Stage 16-D inputs and immutable Stage 16-C failure evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input must be a JSON object: {path}")
    return payload


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
    ).strip()
    if branch != "feature/reference-tracking-isaaclab":
        raise RuntimeError("STAGE16D_BRANCH_LINEAGE_FAILURE")
    for commit in ("e613a4e", "363f2c8"):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT
        )
        if result.returncode != 0:
            raise RuntimeError("STAGE16D_BRANCH_LINEAGE_FAILURE")
    c5a = load(REPO_ROOT / ".local/reports/stage16c5a_state_replication/frozen_inputs.json")
    verified: dict[str, Any] = {}
    for name, row in c5a["inputs"].items():
        path = REPO_ROOT / row["path"]
        actual = sha256(path)
        matches = actual == row["sha256"]
        verified[name] = {
            "path": row["path"],
            "expected_sha256": row["sha256"],
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            raise RuntimeError(f"STAGE16D_INPUT_HASH_DRIFT:{name}")
    reference_paths = {
        clip: REPO_ROOT
        / ".local/stage16_reference_tracking_ppo/world_wrist_references"
        / f"{clip}.world_wrist.stage16.npz"
        for clip in ("hocap_170105", "hocap_170650")
    }
    reference_expected = load(
        REPO_ROOT / ".local/reports/stage16c5a_r2_physx_contract/reference_hashes.json"
    )["hashes"]
    for clip, path in reference_paths.items():
        actual = sha256(path)
        if actual != reference_expected[clip]:
            raise RuntimeError(f"STAGE16D_INPUT_HASH_DRIFT:{clip}")
        verified[f"reference_{clip}"] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "expected_sha256": reference_expected[clip],
            "actual_sha256": actual,
            "matches": True,
        }
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = REPO_ROOT / f".local/archive/stage16d_inputs_{stamp}_{head[:7]}"
    if archive.exists():
        raise FileExistsError(archive)
    archive.mkdir(parents=True)
    selected = (
        ".local/reports/stage16c3r5_reference_retiming_c4/final_summary.json",
        ".local/reports/stage16c3r5_reference_retiming_c4/contact_causality_scale8.json",
        ".local/reports/stage16c3r5_reference_retiming_c4/c4_gpu_vector_benchmark_scale8.json",
        ".local/reports/stage16c5a_state_replication/final_summary.json",
        ".local/reports/stage16c5a_state_replication/frozen_inputs.json",
        ".local/reports/stage16c5a_r2_physx_contract/stage16c5a_r2_summary.json",
        ".local/reports/stage16c5a_r2_physx_contract/reference_hashes.json",
        ".local/reports/stage16c5a_r2_physx_contract/asset_hashes.json",
        ".local/reports/stage16c5a_r2_physx_contract/controller_hashes.json",
        ".local/reports/stage16c5a_r2_physx_contract/physics_hashes.json",
        ".local/reports/stage16c5a_r3_contact_topology_final/contact_topology_diagnosis.json",
        ".local/reports/stage16c5a_r3_robust_oracle_final_retry1/robust_oracle_report.json",
        ".local/reports/stage16c5_r4_distributional_final/distributional_replication_gate.json",
        "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
        "configs/rl/stage16/isaaclab_c5_r4_robust_cem.yaml",
    )
    archive_files: list[dict[str, Any]] = []
    for relative in selected:
        source = REPO_ROOT / relative
        destination = archive / relative.replace("/", "__")
        shutil.copy2(source, destination)
        archive_files.append(
            {
                "source": relative,
                "archive": destination.name,
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    b2_rows: dict[str, Any] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        path = REPO_ROOT / f".local/reports/stage16c5_r4_cem_final/b2_{clip}.json"
        report = load(path)
        final = report["records"][-1]["selected_evaluation"]
        b2_rows[clip] = {
            "final_failure_probability": final["failure_probability"],
            "p95_position_m": final["p95_object_position_error_m"],
            "p95_rotation_deg": final["p95_object_rotation_error_deg"],
            "p95_axis_m": final["p95_axis_error_m"],
            "source": str(path.relative_to(REPO_ROOT)),
            "source_sha256": sha256(path),
        }
    manifest = {
        "schema_version": "Stage16DFrozenInputManifestV1",
        "status": "STAGE16D_INPUTS_FROZEN",
        "branch": branch,
        "head": head,
        "archive": str(archive.relative_to(REPO_ROOT)),
        "verified_inputs": verified,
        "archive_files": archive_files,
        "source_npz_modified": False,
        "stage12_final_modified": False,
    }
    failure_ledger = {
        "schema_version": "Stage16CFailureLedgerForStage16DV1",
        "stage16c5_status": "STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL",
        "deterministic_replication": [
            "SAME_SCENE_CONTACT_DIVERGENCE",
            "TRUE_CONTACT_SOLVER_NONDETERMINISM",
        ],
        "strict_object_tracking_b2": b2_rows,
        "c5c": "NOT_STARTED_GATE_BLOCKED",
        "c6_ppo": {"status": "NOT_AUTHORIZED", "samples": 0, "checkpoints": 0},
        "proves": (
            "current physics model and optimizer cannot strictly track the frozen "
            "object world trajectory"
        ),
        "does_not_prove": [
            "source task semantics are infeasible",
            "no contact-consistent corrected trajectory exists",
            "PPO cannot learn",
            "HO-Cap source data are invalid",
        ],
    }
    runtime = {
        "physics_hz": 120,
        "control_hz": 20,
        "decimation": 6,
        "reference_time_scale": 8,
        "gravity": [0, 0, 0],
        "ground": False,
        "support": False,
        "object": "free_rigid_body",
    }
    action_observation = {
        "action_dimension": 26,
        "action_semantics": "6D wrist residual + 20D finger residual",
        "observation_dimension": 764,
        "clip_id_in_observation": False,
    }
    write(archive / "frozen_input_manifest.json", manifest)
    write(archive / "stage16c_failure_ledger.json", failure_ledger)
    write(archive / "runtime_contract.json", runtime)
    write(
        archive / "asset_contract.json",
        load(REPO_ROOT / ".local/reports/stage16c5a_r2_physx_contract/asset_hashes.json"),
    )
    write(
        archive / "reference_contract.json",
        load(REPO_ROOT / ".local/reports/stage16c5a_r2_physx_contract/reference_hashes.json"),
    )
    write(
        archive / "physics_contract.json",
        load(REPO_ROOT / ".local/reports/stage16c5a_r2_physx_contract/physics_hashes.json"),
    )
    write(archive / "action_observation_contract.json", action_observation)
    (archive / "README.md").write_text(
        "# Stage 16-D Frozen Inputs\n\n"
        "Small immutable contracts and Stage 16-C failure evidence only. Source NPZ, "
        "Stage-12 artifacts, meshes, assets, datasets, and generated trajectories are "
        "not copied or modified.\n",
        encoding="utf-8",
    )
    write(output / "frozen_inputs.json", manifest)
    write(output / "stage16c_failure_ledger.json", failure_ledger)
    print(json.dumps({"status": manifest["status"], "archive": manifest["archive"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
