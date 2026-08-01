#!/usr/bin/env python3
"""Audit Stage-16B mesh/mass/inertia/support and transient contact impulses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from toporetarget.rl.environments.world_wrist_backend import (
    WorldWristFingerBackend,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
)
from toporetarget.rl.object_dynamics_audit import (
    OBJECT_DYNAMICS_AUDIT_ID,
    impulse_sensitivity_candidates,
    inertial_wrench_demand,
    static_reference_contact_proxy,
    support_model_audit,
)
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mesh_audit(path: Path) -> dict[str, Any]:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"object mesh must resolve to one Trimesh: {path}")
    watertight = bool(mesh.is_watertight)
    return {
        "path": str(path.resolve()),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_m": np.asarray(mesh.extents).tolist(),
        "bounds_m": np.asarray(mesh.bounds).tolist(),
        "watertight": watertight,
        "volume_m3": float(mesh.volume),
        "center_mass_local_m": np.asarray(mesh.center_mass).tolist(),
        "volume_mass_inference_eligible": watertight,
        "volume_warning": None if watertight else "NON_WATERTIGHT_VOLUME_NOT_AUTHORITATIVE",
    }


def _zero_residual_trace(backend: WorldWristFingerBackend) -> dict[str, Any]:
    state = backend.reset(reference_index=0)
    first_transient: dict[str, Any] | None = None
    endpoint_contact_frames: list[int] = []
    contact_substeps = 0
    total_impulse = 0.0
    max_substep_impulse = 0.0
    max_penetration = 0.0
    max_linear_speed = float(np.linalg.norm(state["object_twist"][:3]))
    max_angular_speed = float(np.linalg.norm(state["object_twist"][3:]))
    reason: str | None = None
    for step in range(backend.reference.frame_count + 4):
        state, _, reason = backend.transition(np.zeros(26, dtype=np.float64))
        for row in backend.last_physics_trace:
            if row["hand_object_contact_count"]:
                contact_substeps += 1
                if first_transient is None:
                    first_transient = {
                        "control_step": step,
                        "substep": row["substep"],
                        "reference_index_after_step": backend.reference_index,
                        "normal_force_n": row["hand_object_normal_force_n"],
                        "normal_impulse_ns": row["hand_object_normal_impulse_ns"],
                        "penetration_m": row["hand_object_max_penetration_m"],
                        "contacts": row["hand_object_contacts"],
                    }
            impulse = float(row["hand_object_normal_impulse_ns"])
            total_impulse += impulse
            max_substep_impulse = max(max_substep_impulse, impulse)
            max_penetration = max(max_penetration, float(row["hand_object_max_penetration_m"]))
        if backend.contact_summary()["hand_object_contact_count"]:
            endpoint_contact_frames.append(step)
        max_linear_speed = max(max_linear_speed, float(np.linalg.norm(state["object_twist"][:3])))
        max_angular_speed = max(max_angular_speed, float(np.linalg.norm(state["object_twist"][3:])))
        if reason is not None:
            break
    return {
        "policy": "zero_residual_26d",
        "direct_object_control": False,
        "termination": reason or "FAILURE_EVALUATION_STEP_BOUND",
        "progress": backend.reference_index / (backend.reference.frame_count - 1),
        "first_transient_hand_object_contact": first_transient,
        "contact_substeps": contact_substeps,
        "control_endpoint_contact_frames": endpoint_contact_frames,
        "transient_contact_missed_by_endpoint_count": bool(
            contact_substeps and not endpoint_contact_frames
        ),
        "total_normal_impulse_ns": total_impulse,
        "max_substep_normal_impulse_ns": max_substep_impulse,
        "max_penetration_m": max_penetration,
        "max_object_linear_speed_mps": max_linear_speed,
        "max_object_angular_speed_radps": max_angular_speed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    args = parser.parse_args()
    if len(args.reference) != 2 or len(args.object_mesh) != 2:
        raise ValueError("audit requires the frozen two references and two meshes")
    hand_model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    clips = []
    for reference_path, mesh_path in zip(args.reference, args.object_mesh, strict=True):
        reference = WorldWristFingerReferenceV1.from_npz(reference_path)
        scene = materialize_world_wrist_free_object_scene(
            WUJI_MJCF,
            args.scene_root / reference_path.stem,
            object_mesh=mesh_path,
        )
        backend = WorldWristFingerBackend(
            scene_path=scene,
            reference=reference,
            joint_lower=hand_model.jnt_range[: hand_model.njnt, 0],
            joint_upper=hand_model.jnt_range[: hand_model.njnt, 1],
            impedance_profile=WristImpedanceProfileV1(250.0, 1.0, 2.0, 0.5, 25.0, 1.5),
            action_scale=WristFingerActionScaleV1(0.02, float(np.deg2rad(10.0)), 0.20),
            seed=20260801,
        )
        mesh = _mesh_audit(mesh_path)
        model = backend.model_report()
        static_contact = static_reference_contact_proxy(backend)
        inertial = inertial_wrench_demand(backend)
        zero_trace = _zero_residual_trace(backend)
        candidates = impulse_sensitivity_candidates(
            mass_kg=float(model["object_mass_kg"]),
            principal_inertia_kgm2=np.asarray(model["object_principal_inertia_kgm2"]),
            impulse_ns=float(zero_trace["max_substep_normal_impulse_ns"]),
        )
        clips.append(
            {
                "clip": reference_path.stem,
                "reference": str(reference_path.resolve()),
                "reference_provenance": reference.provenance,
                "mesh": mesh,
                "model": model,
                "support": support_model_audit(backend),
                "static_reference_contact_proxy": static_contact,
                "reference_inertial_wrench_demand": inertial,
                "zero_residual_contact_impulse_trace": zero_trace,
                "shared_mass_inertia_scale_counterfactuals": candidates,
            }
        )
    result = {
        "id": OBJECT_DYNAMICS_AUDIT_ID,
        "status": "OBJECT_DYNAMICS_PHYSICAL_PROVENANCE_UNRESOLVED",
        "clips": clips,
        "selection": {
            "selected_for_formal_controller_rerun": "baseline_0.05kg_mujoco_mesh_inertia",
            "selection_reason": (
                "retain pre-existing shared engineering assumption; no source mass, inertia, "
                "support, or force provenance permits score-based physical retuning"
            ),
            "physically_validated": False,
            "clip_specific_tuning": False,
        },
        "conclusion": (
            "The references are geometric trajectories. Under the current unsupported free-body "
            "model, brief contact impulses create persistent object twist; dynamic feasibility "
            "and physical mass/support remain unresolved."
        ),
    }
    _write_json(args.report_root / "object_dynamics_audit.json", result)
    _write_json(args.report_root / "object_dynamics_profile_selection.json", result["selection"])
    print(json.dumps({"status": result["status"], "report": str(args.report_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
