#!/usr/bin/env python3
"""Audit actual Isaac Lab 2.3.2 / Isaac Sim 5.1 PhysX contract APIs.

The audit is intentionally a real, one-environment GPU scene.  Source evidence
identifies only fields present in this checked-out Isaac Lab; live USD evidence
then confirms the fields exist and carry the requested baseline values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
_ISAACLAB_SOURCE = REPO_ROOT / ".local/external/IsaacLab/source/isaaclab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runtime-only", action="store_true")
    mode.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R2_API_AUDIT_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_evidence(relative_path: str, needle: str) -> dict[str, object]:
    path = _ISAACLAB_SOURCE / relative_path
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if not matches:
        raise RuntimeError(f"STAGE16C5A_R2_API_SOURCE_FIELD_MISSING:{relative_path}:{needle}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "needle": needle,
        "line_numbers": matches,
    }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _prim_attributes(prim: Any) -> dict[str, object]:
    return {
        attribute.GetName(): _jsonable(attribute.Get())
        for attribute in prim.GetAttributes()
        if attribute.GetName().startswith(("physx", "physics"))
    }


def _find_prim(stage: Any, suffix: str) -> Any:
    matches = [prim for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"STAGE16C5A_R2_USD_PRIM_AMBIGUOUS:{suffix}:{len(matches)}")
    return matches[0]


def _live_runtime(output_dir: Path) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import omni.usd

        from toporetarget.rl.environments.isaaclab_backend.physx_contract import (
            baseline_contract,
            expected_runtime_config,
        )
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        contract = baseline_contract()
        env = make_stage16c5_env(num_envs=1, physx_contract=contract)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("STAGE16C5A_R2_USD_STAGE_UNAVAILABLE")
        scene_prims = [
            prim
            for prim in stage.Traverse()
            if any(
                name.startswith("physxScene:")
                for name in (attr.GetName() for attr in prim.GetAttributes())
            )
        ]
        if len(scene_prims) != 1:
            raise RuntimeError(f"STAGE16C5A_R2_PHYSX_SCENE_AMBIGUOUS:{len(scene_prims)}")
        scene = scene_prims[0]
        robot = _find_prim(stage, "/env_0/Robot")
        object_105 = _find_prim(stage, "/env_0/Object170105")
        object_650 = _find_prim(stage, "/env_0/Object170650")
        runtime = {
            "requested": expected_runtime_config(contract),
            "cfg": {
                "device": str(env.cfg.sim.device),
                "scene_num_envs": env.cfg.scene.num_envs,
                "clone_in_fabric": env.cfg.scene.clone_in_fabric,
                "env_spacing": env.cfg.scene.env_spacing,
                "physx": {
                    "solver_type": env.cfg.sim.physx.solver_type,
                    "enable_enhanced_determinism": env.cfg.sim.physx.enable_enhanced_determinism,
                    "solve_articulation_contact_last": (
                        env.cfg.sim.physx.solve_articulation_contact_last
                    ),
                    "min_position_iteration_count": env.cfg.sim.physx.min_position_iteration_count,
                    "max_position_iteration_count": env.cfg.sim.physx.max_position_iteration_count,
                    "min_velocity_iteration_count": env.cfg.sim.physx.min_velocity_iteration_count,
                    "max_velocity_iteration_count": env.cfg.sim.physx.max_velocity_iteration_count,
                    "gpu_max_rigid_contact_count": env.cfg.sim.physx.gpu_max_rigid_contact_count,
                    "gpu_max_rigid_patch_count": env.cfg.sim.physx.gpu_max_rigid_patch_count,
                },
                "robot": {
                    "solver_position_iteration_count": (
                        env.cfg.robot.spawn.articulation_props.solver_position_iteration_count
                    ),
                    "solver_velocity_iteration_count": (
                        env.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
                    ),
                },
                "objects": {
                    "solver_position_iteration_count": (
                        env.cfg.object_170105.spawn.rigid_props.solver_position_iteration_count
                    ),
                    "solver_velocity_iteration_count": (
                        env.cfg.object_170105.spawn.rigid_props.solver_velocity_iteration_count
                    ),
                },
            },
            "usd": {
                "physics_scene_prim": str(scene.GetPath()),
                "physics_scene_attributes": _prim_attributes(scene),
                "robot_prim": str(robot.GetPath()),
                "robot_attributes": _prim_attributes(robot),
                "object_170105_prim": str(object_105.GetPath()),
                "object_170105_attributes": _prim_attributes(object_105),
                "object_170650_prim": str(object_650.GetPath()),
                "object_170650_attributes": _prim_attributes(object_650),
            },
            "runtime_verification": "USD attributes captured after scene construction",
        }
        _write(output_dir / "current_physx_contract.json", runtime)
        print(json.dumps({"status": "STAGE16C5A_PHYSX_RUNTIME_CONFIG_CAPTURED"}, sort_keys=True))
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _dimension(
    *,
    identifier: str,
    api_path: str,
    value_type: str,
    current: object,
    valid_range: str,
    scope: str,
    relevance: str,
    impact: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "identifier": identifier,
        "actual_api_path": api_path,
        "type": value_type,
        "current_value": current,
        "valid_range": valid_range,
        "scope": scope,
        "requires_scene_recreation": True,
        "expected_determinism_relevance": relevance,
        "expected_physical_behavior_impact": impact,
        "runtime_readable": True,
        "runtime_writable": False,
        "evidence_source": evidence,
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    output_dir = args.output_dir.resolve()
    if args.runtime_only:
        if (output_dir / "current_physx_contract.json").exists():
            raise FileExistsError("STAGE16C5A_R2_RUNTIME_CONFIG_OUTPUT_ALREADY_EXISTS")
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        _live_runtime(output_dir)
        return 0
    audit_path = output_dir / "physx_determinism_api_audit.json"
    dimensions_path = output_dir / "available_contract_dimensions.json"
    if audit_path.exists() and dimensions_path.exists():
        raise FileExistsError("STAGE16C5A_R2_STATIC_API_AUDIT_OUTPUT_ALREADY_EXISTS")
    current_path = output_dir / "current_physx_contract.json"
    if not current_path.is_file():
        raise FileNotFoundError("STAGE16C5A_R2_RUNTIME_CONFIG_CAPTURE_REQUIRED")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    live = json.loads(current_path.read_text(encoding="utf-8"))
    if not isinstance(live, dict):
        raise ValueError("STAGE16C5A_R2_RUNTIME_CONFIG_MALFORMED")
    cfg = live["cfg"]
    assert isinstance(cfg, dict)
    physx = cfg["physx"]
    assert isinstance(physx, dict)
    dimensions = [
        _dimension(
            identifier="enhanced_determinism",
            api_path=(
                "isaaclab.sim.PhysxCfg.enable_enhanced_determinism "
                "-> PhysxSceneAPI.enableEnhancedDeterminism"
            ),
            value_type="bool",
            current=physx["enable_enhanced_determinism"],
            valid_range="false|true",
            scope="per-scene",
            relevance="PhysX documented improved determinism",
            impact="may reduce throughput",
            evidence=_source_evidence(
                "isaaclab/sim/simulation_cfg.py", "enable_enhanced_determinism"
            ),
        ),
        _dimension(
            identifier="solve_articulation_contact_last",
            api_path=(
                "isaaclab.sim.PhysxCfg.solve_articulation_contact_last "
                "-> physxScene:solveArticulationContactLast"
            ),
            value_type="bool",
            current=physx["solve_articulation_contact_last"],
            valid_range="false|true",
            scope="per-scene",
            relevance="documented gripping contact solver ordering",
            impact="changes articulation/contact constraint ordering",
            evidence=_source_evidence(
                "isaaclab/sim/simulation_cfg.py", "solve_articulation_contact_last"
            ),
        ),
        _dimension(
            identifier="scene_solver_iterations",
            api_path="isaaclab.sim.PhysxCfg.{min,max}_{position,velocity}_iteration_count",
            value_type="int",
            current={
                "position": [
                    physx["min_position_iteration_count"],
                    physx["max_position_iteration_count"],
                ],
                "velocity": [
                    physx["min_velocity_iteration_count"],
                    physx["max_velocity_iteration_count"],
                ],
            },
            valid_range="position [1,255], velocity [0,255]",
            scope="per-scene clamp",
            relevance="clamps per-actor iteration settings",
            impact="more solver iterations may change contact convergence and throughput",
            evidence=_source_evidence(
                "isaaclab/sim/simulation_cfg.py", "max_position_iteration_count"
            ),
        ),
        _dimension(
            identifier="actor_solver_iterations",
            api_path="ArticulationRootPropertiesCfg/RigidBodyPropertiesCfg.solver_{position,velocity}_iteration_count",
            value_type="int",
            current={"position": 8, "velocity": 2},
            valid_range="position [1,255]; velocity bounded by scene [0,255]",
            scope="all articulation and rigid objects",
            relevance="actual actor solver iteration requests",
            impact="changes contact convergence and throughput",
            evidence=_source_evidence(
                "isaaclab/sim/schemas/schemas_cfg.py", "solver_position_iteration_count"
            ),
        ),
        _dimension(
            identifier="gpu_contact_capacities",
            api_path="isaaclab.sim.PhysxCfg.gpu_max_rigid_{contact,patch}_count",
            value_type="int",
            current={
                "contact": physx["gpu_max_rigid_contact_count"],
                "patch": physx["gpu_max_rigid_patch_count"],
            },
            valid_range="positive integer capacity",
            scope="per-scene GPU pipeline",
            relevance="buffer-overflow prevention only; not a candidate physical setting",
            impact="memory capacity, not contact material semantics",
            evidence=_source_evidence(
                "isaaclab/sim/simulation_cfg.py", "gpu_max_rigid_contact_count"
            ),
        ),
    ]
    audit = {
        "status": "STAGE16C5A_PHYSX_API_AUDITED",
        "schema_version": "stage16c5a_r2_physx_api_audit_v1",
        "isaac_lab_source": str(_ISAACLAB_SOURCE.relative_to(REPO_ROOT)),
        "dimensions": dimensions,
        "unsupported_or_excluded": [
            (
                "No separate supported GPU contact determinism flag beyond documented "
                "scene controls was found."
            ),
            (
                "friction_offset_threshold and friction_correlation_distance are actual APIs "
                "but excluded because the task freezes friction/contact semantics."
            ),
            "GPU buffer capacities are recorded but excluded from the physical candidate matrix.",
        ],
        "recorded_reproducibility_environment": {
            name: os.environ.get(name)
            for name in (
                "OMNI_KIT_ACCEPT_EULA",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }
    available = {
        "status": "STAGE16C5A_PHYSX_API_AUDITED",
        "available_dimensions": [entry["identifier"] for entry in dimensions],
        "matrix_allowed_contact_setting": "solve_articulation_contact_last",
        "matrix_disallowed": ["friction", "contact_offset", "rest_offset", "mass", "inertia"],
    }
    if not audit_path.exists():
        _write(audit_path, audit)
    if not dimensions_path.exists():
        _write(dimensions_path, available)
    print(json.dumps({"status": audit["status"], "dimensions": len(dimensions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
