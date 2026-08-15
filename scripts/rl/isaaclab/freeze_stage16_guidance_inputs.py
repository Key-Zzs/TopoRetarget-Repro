#!/usr/bin/env python3
"""Freeze V2-reference provenance for the assisted-guidance G0 gate.

The source tree is read-only.  The only copies created are immutable SHA-256
checked reference inputs inside this worktree's ignored ``.local`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.guidance import ObjectGuidanceContractV1  # noqa: E402

PHYSICAL_REFERENCE_ROOT = Path(
    "/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro/.local/reports/"
    "stage16d_reference_kinematics_v2/references"
)
PHYSICAL_QUALIFICATION = PHYSICAL_REFERENCE_ROOT.parent / "reference_kinematics_qualification.json"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_guidance_g0_g5/g0"
DEFAULT_COPY_ROOT = REPO_ROOT / ".local/frozen_baselines/reference_kinematics_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference-root", type=Path, default=PHYSICAL_REFERENCE_ROOT)
    parser.add_argument("--source-qualification", type=Path, default=PHYSICAL_QUALIFICATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--copy-reference-root", type=Path, default=DEFAULT_COPY_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_reference_root.resolve()
    qualification_path = args.source_qualification.resolve()
    if not qualification_path.is_file():
        raise FileNotFoundError(f"GUIDANCE_G0_REFERENCE_QUALIFICATION_MISSING:{qualification_path}")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("status") != "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED":
        raise RuntimeError("GUIDANCE_G0_REFERENCE_KINEMATICS_V2_NOT_VALIDATED")
    config_path = REPO_ROOT / "configs/physics/object_guidance_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("guidance"), dict):
        raise ValueError("GUIDANCE_G0_CONFIG_INVALID")
    guidance_values = dict(config["guidance"])
    guidance_values.pop("mode", None)
    contract = ObjectGuidanceContractV1(mode="reference_wrench_v1", **guidance_values)
    output_root = args.output_root.resolve()
    copy_root = args.copy_reference_root.resolve()
    references: dict[str, dict[str, object]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        source = source_root / f"{clip}.reference_kinematics_v2.npz"
        if not source.is_file():
            raise FileNotFoundError(f"GUIDANCE_G0_REFERENCE_MISSING:{source}")
        destination = copy_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copy_hash = sha256(destination)
        if source_hash != copy_hash:
            raise RuntimeError(f"GUIDANCE_G0_REFERENCE_COPY_HASH_MISMATCH:{clip}")
        references[clip] = {
            "source_path": str(source),
            "source_sha256": source_hash,
            "copied_path": str(destination),
            "copied_sha256": copy_hash,
            "source_read_only": True,
        }
    guidance_contract = {
        **contract.as_dict(),
        "sha256": contract.sha256(),
        "mode": "reference_wrench_v1",
        "policy_observes_guidance": False,
        "guidance_in_reward": False,
        "object_teleport": False,
        "object_rollout_state_write": False,
        "wrist_root_rollout_state_write": False,
        "hidden_attachment": False,
        "application": "explicit_world_wrench_to_physx",
    }
    frozen_inputs = {
        "schema_version": "Stage16GuidanceG0FrozenInputsV1",
        "worktree": str(REPO_ROOT),
        "branch": git_value("branch", "--show-current"),
        "start_head": git_value("rev-parse", "HEAD"),
        "guidance_base_sha": git_value("rev-parse", "HEAD"),
        "guidance_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "reference_kinematics_qualification": {
            "path": str(qualification_path),
            "sha256": sha256(qualification_path),
            "status": qualification["status"],
        },
        "references": references,
        "physical_worktree_modified": False,
        "physical_artifacts_read_only": True,
    }
    write_json(output_root / "guidance_contract.json", guidance_contract)
    write_json(output_root / "frozen_inputs.json", frozen_inputs)
    print(json.dumps({"g0": "COMPLETE", "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
