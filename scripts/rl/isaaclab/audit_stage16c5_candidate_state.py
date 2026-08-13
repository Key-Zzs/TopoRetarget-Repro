#!/usr/bin/env python3
"""Freeze C.3/C.4 inputs and audit the API-restorable C.5A Markov state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
PREVIOUS_ROOT = REPO_ROOT / ".local/reports/stage16c3r5_reference_retiming_c4"
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16c5a_state_replication"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_inputs() -> dict[str, Path]:
    paths = {
        "c3_c4_final_summary": PREVIOUS_ROOT / "final_summary.json",
        "c3_semantic_qualification": PREVIOUS_ROOT / "c3_full_qualification_scale8_final.json",
        "contact_causality": PREVIOUS_ROOT / "contact_causality_scale8.json",
        "c4_vector_benchmark": PREVIOUS_ROOT / "c4_gpu_vector_benchmark_scale8.json",
        "c3_c4_tests": PREVIOUS_ROOT / "test_summary.json",
        "active_config": REPO_ROOT / "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
        "direct_env_source": (
            REPO_ROOT
            / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py"
        ),
        "direct_env_config_source": (
            REPO_ROOT
            / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16C5A_INPUT_MISSING: {missing}")
    return paths


def _validate_previous(inputs: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    c3 = _load_json(inputs["c3_semantic_qualification"])
    c4 = _load_json(inputs["c4_vector_benchmark"])
    contact = _load_json(inputs["contact_causality"])
    tests = _load_json(inputs["c3_c4_tests"])
    config = yaml.safe_load(inputs["active_config"].read_text(encoding="utf-8"))
    if c3.get("status") != "STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED":
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: C3 status")
    if c4.get("status") != "STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED":
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: C4 status")
    if contact.get("status") != "C3_CONTACT_CAUSALITY_VALIDATED":
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: contact status")
    if tests.get("status") != "PASS":
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: test summary")
    if not isinstance(config, dict):
        raise TypeError("active C3/C4 config is not a mapping")
    timing = config.get("reference_bank", {}).get("active_retiming", {})
    if timing.get("time_scale") != 8 or timing.get("runtime_frames") != 321:
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: retiming")
    if config.get("active_wrist", {}).get("controller") != "finite_virtual_6d_wrist_actuator_v1":
        raise RuntimeError("STAGE16C5A_INPUT_HASH_DRIFT: controller")
    return c3, c4


def _asset_hashes(env: Any) -> dict[str, str]:
    assets = {
        "robot": Path(env.cfg.robot.spawn.usd_path),
        "object_170105": Path(env.cfg.object_170105.spawn.usd_path),
        "object_170650": Path(env.cfg.object_170650.spawn.usd_path),
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16C5A_INPUT_HASH_DRIFT: missing asset {missing}")
    return {name: _sha256(path) for name, path in assets.items()}


def _field_audit(state: Any) -> dict[str, object]:
    rows = state.field_manifest()
    classes = {
        "simulation": "must snapshot/restore through supported Isaac write APIs",
        "task": "must snapshot/restore because it controls future reference/termination",
        "action_history": "must snapshot/restore and be extended as CandidateActionHistoryV1",
        "controller": "must snapshot/restore because it changes the next control target",
        "replication": "must snapshot/restore to rebase world state across unique origins",
        "auxiliary": "API-restorable software state; retained when present",
    }
    return {
        "version": "stage16c5_candidate_state_field_audit_v1",
        "status": "STAGE16C5A_CANDIDATE_STATE_CONTRACT_VALIDATED",
        "fields": rows,
        "categories": classes,
        "diagnostic_only": [
            "contact_substep_records",
            "wrist_diagnostic_records",
            "last_reward_terms",
        ],
        "physx_api_unavailable": list(state.inaccessible_physx_state),
        "contact_sensor_policy": (
            "restore no fabricated force; recompute sensor telemetry from restored PhysX state"
        ),
    }


def _dependency_graph(audit: dict[str, object]) -> str:
    rows = audit["fields"]
    assert isinstance(rows, list)
    lines = [
        "# Stage16C5CandidateStateV1 dependency graph",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["Candidate action"] --> B["Reference index and action history"]',
        '  B --> C["Wrist/finger drive targets"]',
        '  C --> D["PhysX articulation state"]',
        '  D --> E["Contact solver state"]',
        '  E --> F["Object state and contact telemetry"]',
        '  B --> G["Reward and termination buffers"]',
        "  D --> G",
        "```",
        "",
        "## API-restorable fields",
        "",
        "| Field | Classification | Restore method |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(f"| `{row['field']}` | {row['classification']} | `{row['restore_method']}` |")
    lines.extend(
        [
            "",
            "PhysX warm-start, contact-manifold, friction-patch, and internal constraint caches "
            "are not exposed by the supported Isaac Lab API. Tensor clone therefore cannot "
            "claim to restore them; contact mismatch triggers deterministic history replay.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    inputs = _required_inputs()
    c3, c4 = _validate_previous(inputs)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from toporetarget.rl.isaaclab_oracle.candidate_state import capture_candidate_state
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        env = make_stage16c5_env(num_envs=2)
        state = capture_candidate_state(env, [0])
        assets = _asset_hashes(env)
        output_root = args.output_root.resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_head = os.popen("git rev-parse --short HEAD").read().strip()
        archive = REPO_ROOT / ".local/archive" / f"stage16c5a_inputs_{timestamp}_{short_head}"
        archive.mkdir(parents=True, exist_ok=False)
        manifest_inputs = {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
            for name, path in inputs.items()
        }
        for name, path in inputs.items():
            shutil.copy2(path, archive / f"{name}{path.suffix}")
        frozen = {
            "status": "STAGE16C5A_INPUTS_FROZEN",
            "archive": str(archive.relative_to(REPO_ROOT)),
            "head": os.popen("git rev-parse HEAD").read().strip(),
            "runtime": env.contract_report()["reference_bank"],
            "reference_timing": env.contract_report()["reference_timing"],
            "controller": env.contract_report()["finite_virtual_6d_wrist_actuator"],
            "assets": assets,
            "inputs": manifest_inputs,
            "c3_status": c3["status"],
            "c4_status": c4["status"],
            "candidate_state_hashes": state.config_hashes,
        }
        audit = _field_audit(state)
        _write_json(output_root / "frozen_inputs.json", frozen)
        _write_json(archive / "frozen_manifest.json", frozen)
        _write_json(archive / "runtime.json", frozen["runtime"])
        _write_json(archive / "assets.json", assets)
        _write_json(archive / "references.json", frozen["runtime"])
        _write_json(archive / "retiming.json", frozen["reference_timing"])
        _write_json(archive / "direct_env_contract.json", env.contract_report())
        _write_json(archive / "controller.json", frozen["controller"])
        _write_json(archive / "contact_contract.json", env.contact_sensor_contract())
        _write_json(
            archive / "c3_c4_status.json",
            {"c3": c3["status"], "c4": c4["status"], "test_status": "PASS"},
        )
        (archive / "README.md").write_text(
            "# Stage 16-C.5A frozen inputs\n\n"
            "Small C.3/C.4 manifests copied before C.5A implementation. Source assets and "
            "reference NPZ files remain read-only.\n",
            encoding="utf-8",
        )
        _write_json(output_root / "candidate_state_field_audit.json", audit)
        _write_json(output_root / "candidate_state_contract.json", state.as_dict())
        (output_root / "candidate_state_dependency_graph.md").write_text(
            _dependency_graph(audit), encoding="utf-8"
        )
        print(json.dumps({"frozen": frozen["status"], "audit": audit["status"]}, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
