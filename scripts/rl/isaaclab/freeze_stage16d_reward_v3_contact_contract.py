#!/usr/bin/env python3
"""Freeze and gate Stage 16-D Reward V3 from reference-only geometry.

This is intentionally a pre-training program.  It refuses a legacy trace that
records only an aggregate object contact force: an aggregate cannot be safely
decomposed into five fingertip--active-object pair forces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (  # noqa: E402
    CONTACT_FINGER_ORDER,
    EVALUATION_FINGERTIP_LINKS,
    ReferenceGatedContactRewardContractV1,
    exact_pair_force_trace_status,
    fingertip_force_indices,
    reference_expected_contact_mask,
    reference_mask_summary,
)

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_OBJECT_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_contact"
DEFAULT_V1_FORMAL = {
    clip: REPO_ROOT
    / ".local/reports/stage16d_ppo26d_continuation"
    / clip
    / "ppo_r7_formal_trace_replica0.npz"
    for clip in CLIPS
}
DEFAULT_V2_FORMAL = (
    REPO_ROOT
    / ".local/reports/stage16d_reference_kinematics_v2/phase3/hocap_170650"
    / "runs/p1_post_capacity/dev_evaluations/hocap_170650/reward_v2_p1_trace_replica0.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _world_vertices(
    vertices: np.ndarray, translation: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    rotation = Rotation.from_quat(np.concatenate((quaternion_wxyz[1:], quaternion_wxyz[:1])))
    return rotation.apply(vertices) + translation


def reference_distances_to_visual_mesh(
    *, reference: Path, object_mesh: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute `[321,5]` unsigned distances in the frozen world reference frame."""

    with np.load(reference, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        links = tuple(str(name) for name in metadata["tracked_link_names"])
        indices = [links.index(name) for name in EVALUATION_FINGERTIP_LINKS]
        tip_positions = np.asarray(
            archive["tracked_link_positions_world_ref"][:, indices], np.float64
        )
        object_position = np.asarray(archive["object_pose_translation_world_ref"], np.float64)
        object_quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], np.float64)
    if tip_positions.shape != (321, 5, 3):
        raise ValueError(f"CONTACT_MASK_REFERENCE_TIP_SHAPE_INVALID:{tip_positions.shape}")
    mesh = trimesh.load(object_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError(f"CONTACT_MASK_VISUAL_MESH_INVALID:{object_mesh}")
    distances: list[np.ndarray] = []
    for tips, position, quaternion in zip(
        tip_positions, object_position, object_quaternion, strict=True
    ):
        world_mesh = trimesh.Trimesh(
            vertices=_world_vertices(mesh.vertices, position, quaternion),
            faces=mesh.faces,
            process=False,
        )
        try:
            nearest = trimesh.proximity.closest_point(world_mesh, tips)[1]
        except ModuleNotFoundError:
            nearest = trimesh.proximity.closest_point_naive(world_mesh, tips)[1]
        distances.append(np.asarray(nearest, dtype=np.float64))
    result = np.stack(distances)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("CONTACT_MASK_VISUAL_UNSIGNED_DISTANCE_INVALID")
    return result, {
        "distance_source": "official_or_source_visual_object_surface_mesh",
        "object_mesh": str(object_mesh.resolve()),
        "object_mesh_sha256": _sha256(object_mesh),
        "watertight_required": False,
        "collision_proxy_approximation": False,
        "reference_fingertip_source": "EvaluationFingertipSetV1.distal_link_root",
    }


def _trace_force_report(
    path: Path, *, clip: str, contract: ReferenceGatedContactRewardContractV1
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "clip": clip,
            "status": "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED",
            "reason": f"required frozen V1 formal trace is missing: {path}",
        }
    with np.load(path, allow_pickle=False) as archive:
        report = exact_pair_force_trace_status(archive.files)
        report.update(
            {
                "clip": clip,
                "trace": str(path.resolve()),
                "trace_sha256": _sha256(path),
                "required_formula": "S_contact=sum(mask_f*norm(F_f_active_object))",
                "aggregate_force_permitted_for_calibration": False,
            }
        )
        # Preserve evidence that historical pair presence does not repair the
        # missing vector decomposition.  It is diagnostic only, never a scale.
        if "replica_contact_pair_presence" in archive.files:
            presence = np.asarray(archive["replica_contact_pair_presence"], dtype=bool)
            report["historical_pair_presence_shape"] = list(presence.shape)
            report["historical_pair_presence_nonzero"] = int(presence.sum())
        elif "contact_pair_presence" in archive.files:
            presence = np.asarray(archive["contact_pair_presence"], dtype=bool)
            report["historical_pair_presence_shape"] = list(presence.shape)
            report["historical_pair_presence_nonzero"] = int(presence.sum())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v1-formal-170105", type=Path, default=DEFAULT_V1_FORMAL["hocap_170105"])
    parser.add_argument("--v1-formal-170650", type=Path, default=DEFAULT_V1_FORMAL["hocap_170650"])
    parser.add_argument("--v2-formal-diagnostic", type=Path, default=DEFAULT_V2_FORMAL)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return status 2 unless a complete exact-pair-force calibration input exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = ReferenceGatedContactRewardContractV1()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping = {
        "schema_version": "Stage16DContactRewardFingertipMappingV1",
        "finger_order": list(CONTACT_FINGER_ORDER),
        "reference_links": list(EVALUATION_FINGERTIP_LINKS),
        "force_sensor_body_order": list(HAND_COLLISION_BODY_NAMES),
        "force_sensor_indices": list(fingertip_force_indices(HAND_COLLISION_BODY_NAMES)),
        "clip_specific_mapping": False,
    }
    _write_json(output / "fingertip_mapping.json", mapping)
    _write_json(output / "contact_reward_fingertip_mapping.json", mapping)

    frozen_inputs: dict[str, Any] = {
        "schema_version": "Stage16DRewardV3FrozenInputsV1",
        "reference_kinematics_version_required": 2,
        "contract": contract.as_dict(),
        "references": {},
        "visual_object_meshes": {},
    }
    mask_reports: dict[str, Any] = {}
    mask_valid = True
    for clip in CLIPS:
        clip_short = clip.removeprefix("hocap_")
        reference = (args.reference_root / f"{clip}.reference_kinematics_v2.npz").resolve()
        object_mesh = (args.object_root / f"{clip}.obj").resolve()
        if not reference.is_file() or not object_mesh.is_file():
            raise FileNotFoundError(f"CONTACT_MASK_FROZEN_INPUT_MISSING:{reference}:{object_mesh}")
        with np.load(reference, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata.get("reference_kinematics_version", -1)) != 2:
                raise ValueError("CONTACT_MASK_REQUIRES_REFERENCE_KINEMATICS_V2")
        distances, geometry = reference_distances_to_visual_mesh(
            reference=reference, object_mesh=object_mesh
        )
        mask = np.asarray(reference_expected_contact_mask(distances, contract=contract), dtype=bool)
        summary = reference_mask_summary(mask, clip=clip)
        summary.update(
            {
                "schema_version": "Stage16DReferenceContactMaskV1",
                "strict_primary_mask": "distance_m < 0.03",
                "diagnostic_mask_2cm_frames": int((distances < 0.02).any(axis=1).sum()),
                "per_fingertip_diagnostic_mask_2cm_frames": {
                    finger: int((distances[:, index] < 0.02).sum())
                    for index, finger in enumerate(CONTACT_FINGER_ORDER)
                },
                "geometry": geometry,
                "reference_contact_topology_comparison": "DIAGNOSTIC_NOT_USED_AS_MASK_SOURCE",
                "phase1_persistent_window_comparison": "DIAGNOSTIC_NOT_USED_AS_MASK_SOURCE",
            }
        )
        invalid = summary["mask_fraction"] > 0.95 or summary["mask_fraction"] < 0.01
        summary["status"] = "REFERENCE_CONTACT_MASK_INVALID" if invalid else "PASS"
        mask_valid &= not invalid
        np.savez_compressed(
            output / f"reference_contact_mask_{clip_short}.npz",
            reference_expected_contact_mask=mask,
            reference_fingertip_to_object_distance_m=distances.astype(np.float32),
            finger_order=np.asarray(CONTACT_FINGER_ORDER),
            metadata=np.asarray(json.dumps(summary, sort_keys=True)),
        )
        _write_json(output / f"reference_contact_mask_{clip_short}.json", summary)
        mask_reports[clip] = summary
        frozen_inputs["references"][clip] = {"path": str(reference), "sha256": _sha256(reference)}
        frozen_inputs["visual_object_meshes"][clip] = geometry

    trace_reports = {
        "hocap_170105": _trace_force_report(
            args.v1_formal_170105, clip="hocap_170105", contract=contract
        ),
        "hocap_170650": _trace_force_report(
            args.v1_formal_170650, clip="hocap_170650", contract=contract
        ),
    }
    v2_diagnostic: dict[str, Any] = {"path": str(args.v2_formal_diagnostic.resolve())}
    if args.v2_formal_diagnostic.is_file():
        with np.load(args.v2_formal_diagnostic, allow_pickle=False) as archive:
            v2_diagnostic.update(exact_pair_force_trace_status(archive.files))
            v2_diagnostic["sha256"] = _sha256(args.v2_formal_diagnostic)
    else:
        v2_diagnostic["status"] = "V2_DIAGNOSTIC_NOT_AVAILABLE"
    pair_force_ready = all(
        row.get("status") == "PAIR_FORCE_AVAILABLE" for row in trace_reports.values()
    )
    if not mask_valid:
        status = "REFERENCE_CONTACT_MASK_INVALID"
    elif not pair_force_ready:
        status = "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED"
    else:
        status = "CONTACT_REWARD_FORCE_SCALE_READY_FOR_CALIBRATION"
    calibration = {
        "schema_version": "ContactRewardScaleCalibrationV1",
        "status": status,
        "primary_calibration_sources": trace_reports,
        "reward_v2_diagnostic_only": v2_diagnostic,
        "positive_contact_minimum_frames": 100,
        "lambda_rule": "combined_positive_contact_S_contact_p50",
        "lambda_c_n": None,
        "reason": (
            "No lambda is frozen until both V1 formal traces expose exact five-fingertip "
            "active-object pair force vectors."
            if status == "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED"
            else None
        ),
    }
    contract_receipt = {
        "schema_version": "Stage16DReferenceGatedContactRewardV3ContractV1",
        "status": status,
        "reward": "RewardV3 = RewardV2 + r_contact",
        "frozen_parameters": {**contract.as_dict(), "lambda_c_n": None},
        "reference_mask_source": "Stage16DReferenceKinematicsV2 and visual object mesh only",
        "actual_force_source": "filtered fingertip-to-active-object PhysX pair force only",
        "forbidden": [
            "net_fingertip_force",
            "contact_loss_termination",
            "terminal_reward",
            "penetration_reward",
            "guidance",
            "physics_mutation",
            "observation_change",
        ],
    }
    information_flow = {
        "schema_version": "ContactRewardInformationFlowAuditV1",
        "status": "PASS_STATIC_CONTRACT",
        "reward_inputs": ["reference_current_or_future_mask", "current_pair_contact_force"],
        "actor_observation_change": False,
        "actor_observation_dimension": 764,
        "forbidden_actor_inputs": [
            "future_actual_force",
            "future_actual_contact",
            "success_label",
            "future_object_state",
        ],
    }
    _write_json(output / "frozen_inputs.json", frozen_inputs)
    _write_json(output / "contact_reward_contract.json", contract_receipt)
    _write_json(output / "force_scale_calibration.json", calibration)
    _write_json(output / "contact_reward_scale_calibration.json", calibration)
    _write_json(output / "information_flow_audit.json", information_flow)
    _write_json(output / "contact_reward_information_flow_audit.json", information_flow)
    response = {
        "schema_version": "ContactRewardResponseAuditV1",
        "status": "BLOCKED_NO_FROZEN_LAMBDA" if calibration["lambda_c_n"] is None else "PASS",
        "reason": "Pair-force scale calibration did not freeze lambda_c."
        if calibration["lambda_c_n"] is None
        else None,
        "response_rows": [],
    }
    _write_json(output / "reward_response.json", response)
    _write_json(output / "contact_reward_response.json", response)
    (output / "reward_response.csv").write_text(
        f'status,reason\n{response["status"]},"{response["reason"] or ""}"\n',
        encoding="utf-8",
    )
    (output / "contact_reward_response.csv").write_text(
        (output / "reward_response.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "failure_transitions.jsonl").write_text(
        json.dumps({"from": "MASK_AUDIT", "to": status, "training_authorized": False}) + "\n",
        encoding="utf-8",
    )
    final_summary = {
        "schema_version": "Stage16DRewardV3ContactFinalSummaryV1",
        "status": "V3_IMPLEMENTATION_BLOCKED"
        if status == "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED"
        else status,
        "reward_contract": "TopoRetargetReferenceTrackingReward26DV3",
        "training": "NOT_STARTED_PAIR_FORCE_SCALE_UNRESOLVED"
        if status == "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED"
        else "NOT_STARTED_PRETRAINING_GATE_INCOMPLETE",
        "formal20": "NOT_RUN_PRETRAINING_GATE_BLOCKED",
        "simulation_data": "NOT_EXPORTED_PRETRAINING_GATE_BLOCKED",
        "replay": "NOT_RUN_NO_V3_TRACE",
        "next_required_input": "frozen V1 Formal20 trace with exact fingertip-object pair force",
    }
    _write_json(output / "final_summary.json", final_summary)
    (output / "final_summary.md").write_text(
        "# Stage 16-D Reward V3 Contact Pretraining Gate\n\n"
        f"Status: `{final_summary['status']}`.\n\n"
        "The reference masks passed, but the frozen V1 Formal20 traces retain only aggregate "
        "force and pair presence. They cannot freeze the required shared `lambda_c`; PPO, "
        "Formal20, simulation-data export, and replay are therefore not run.\n",
        encoding="utf-8",
    )
    _write_json(
        output / "replay_validation.json",
        {"status": "NOT_RUN_NO_V3_TRACE", "headless": None, "gui": None},
    )
    _write_json(
        output / "preflight_summary.json",
        {
            "schema_version": "Stage16DRewardV3ContactPreflightV1",
            "status": status,
            "mask_reports": mask_reports,
            "force_scale_calibration": str((output / "force_scale_calibration.json").resolve()),
            "training_authorized": status == "CONTACT_REWARD_FORCE_SCALE_READY_FOR_CALIBRATION",
        },
    )
    print(json.dumps({"status": status, "output_root": str(output)}))
    if args.require_ready and status != "CONTACT_REWARD_FORCE_SCALE_READY_FOR_CALIBRATION":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
