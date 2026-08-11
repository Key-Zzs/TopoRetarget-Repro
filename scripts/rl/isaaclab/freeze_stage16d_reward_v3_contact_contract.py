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
)

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_OBJECT_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
DEFAULT_MASK_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_contact"
DEFAULT_OLD_BLOCKER = DEFAULT_MASK_ROOT / "force_scale_calibration.json"
DEFAULT_V1_FORMAL = {
    clip: REPO_ROOT
    / ".local/reports/stage16d_reward_v3_pairforce_unblock/v1_pairforce"
    / clip
    / "trace.npz"
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


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("CONTACT_REWARD_CALIBRATION_STATISTICS_INVALID")
    return {
        "n": int(values.size),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _load_frozen_reference_mask(path: Path, *, clip: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CONTACT_REWARD_FROZEN_REFERENCE_MASK_MISSING:{path}")
    with np.load(path, allow_pickle=False) as archive:
        if "reference_expected_contact_mask" not in archive.files:
            raise ValueError("CONTACT_REWARD_FROZEN_REFERENCE_MASK_FIELD_MISSING")
        mask = np.asarray(archive["reference_expected_contact_mask"], dtype=bool)
        metadata = (
            json.loads(_scalar_text(archive["metadata"]))
            if "metadata" in archive.files
            else {"status": "LEGACY_MASK_METADATA_MISSING"}
        )
    if mask.shape != (321, 5):
        raise ValueError(f"CONTACT_REWARD_FROZEN_REFERENCE_MASK_SHAPE_INVALID:{clip}:{mask.shape}")
    if metadata.get("status") != "PASS":
        raise ValueError(f"CONTACT_REWARD_FROZEN_REFERENCE_MASK_NOT_VALIDATED:{clip}")
    return mask, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "metadata": metadata,
    }


def _qualify_exact_pair_force_trace(
    path: Path,
    *,
    clip: str,
    reference_mask: np.ndarray,
    expected_indices: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    report: dict[str, Any] = {
        "schema_version": "Stage16DExactPairForceArtifactQualificationV1",
        "clip": clip,
        "trace": str(path.resolve()),
        "required_shape": [321, 20, 5, 3],
        "required_fingertip_links": list(EVALUATION_FINGERTIP_LINKS),
        "required_force_sensor_indices": list(expected_indices),
        "aggregate_force_permitted_for_calibration": False,
        "required_formula": "S_contact=sum(mask_f*norm(F_f_active_object))",
    }
    if not path.is_file():
        report.update(
            {
                "status": "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED",
                "reason": f"required V1 pair-force trace is missing: {path}",
            }
        )
        return report, np.empty(0, dtype=np.float64)
    report["trace_sha256"] = _sha256(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "replica_fingertip_object_pair_force_world",
                "replica_fingertip_object_pair_force_valid",
                "fingertip_link_names",
                "fingertip_force_sensor_indices",
                "pair_force_frame",
                "pair_force_units",
                "pair_force_semantics",
                "reference_index",
                "replica_action",
                "replica_object_pose",
                "replica_contact_pair_presence",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"PAIR_FORCE_REQUIRED_FIELDS_MISSING:{missing}")
            pair_force = np.asarray(archive["replica_fingertip_object_pair_force_world"])
            valid = np.asarray(archive["replica_fingertip_object_pair_force_valid"], dtype=bool)
            names = tuple(str(value) for value in archive["fingertip_link_names"].tolist())
            indices = tuple(
                int(value)
                for value in np.asarray(archive["fingertip_force_sensor_indices"]).tolist()
            )
            frame = _scalar_text(archive["pair_force_frame"])
            units = _scalar_text(archive["pair_force_units"])
            semantics = _scalar_text(archive["pair_force_semantics"])
            report["fields"] = sorted(required)
            report["pair_force_dtype"] = str(pair_force.dtype)
            report["pair_force_shape"] = list(pair_force.shape)
            report["valid_shape"] = list(valid.shape)
            report["fingertip_link_names"] = list(names)
            report["fingertip_force_sensor_indices"] = list(indices)
            report["force_frame"] = frame
            report["force_units"] = units
            report["force_semantics"] = semantics
            if pair_force.shape != (321, 20, 5, 3):
                raise ValueError(f"PAIR_FORCE_SHAPE_INVALID:{pair_force.shape}")
            if valid.shape != (321, 20):
                raise ValueError(f"PAIR_FORCE_VALID_SHAPE_INVALID:{valid.shape}")
            if pair_force.dtype.kind != "f":
                raise ValueError(f"PAIR_FORCE_DTYPE_INVALID:{pair_force.dtype}")
            if names != EVALUATION_FINGERTIP_LINKS:
                raise ValueError(f"PAIR_FORCE_FINGERTIP_NAMES_INVALID:{names}")
            if indices != expected_indices:
                raise ValueError(f"PAIR_FORCE_MAPPING_INVALID:{indices}")
            if frame != "world" or units != "N" or not semantics:
                raise ValueError("PAIR_FORCE_SEMANTICS_INVALID")
            valid_vectors = pair_force[valid]
            if not np.isfinite(valid_vectors).all():
                raise ValueError("PAIR_FORCE_NONFINITE_VALID_SAMPLE")
            if bool(valid[0].any()):
                raise ValueError("PAIR_FORCE_INITIAL_SAMPLE_MUST_BE_INVALID")
            magnitudes = np.linalg.vector_norm(pair_force, axis=-1)
            contact_scale = (magnitudes * reference_mask[:, None, :]).sum(axis=-1)
            selected = valid & reference_mask.any(axis=-1)[:, None] & (contact_scale > 0.0)
            positive = np.asarray(contact_scale[selected], dtype=np.float64)
            report.update(
                {
                    "valid_sample_count": int(valid.sum()),
                    "valid_initial_sample_count": int(valid[0].sum()),
                    "reference_expected_valid_sample_count": int(
                        (valid & reference_mask.any(axis=-1)[:, None]).sum()
                    ),
                    "positive_calibration_sample_count": int(positive.size),
                    "positive_contact_statistics": _statistics(positive) if positive.size else None,
                    "status": "PASS",
                }
            )
            return report, positive
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(
            {
                "status": "PAIR_FORCE_ARTIFACT_INVALID",
                "reason": str(exc),
            }
        )
        return report, np.empty(0, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-contact-mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--v1-formal-170105", type=Path, default=DEFAULT_V1_FORMAL["hocap_170105"])
    parser.add_argument("--v1-formal-170650", type=Path, default=DEFAULT_V1_FORMAL["hocap_170650"])
    parser.add_argument("--v2-formal-diagnostic", type=Path, default=DEFAULT_V2_FORMAL)
    parser.add_argument("--old-blocker", type=Path, default=DEFAULT_OLD_BLOCKER)
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
    expected_indices = fingertip_force_indices(HAND_COLLISION_BODY_NAMES)
    mapping = {
        "schema_version": "Stage16DContactRewardFingertipMappingV1",
        "finger_order": list(CONTACT_FINGER_ORDER),
        "reference_links": list(EVALUATION_FINGERTIP_LINKS),
        "force_sensor_body_order": list(HAND_COLLISION_BODY_NAMES),
        "force_sensor_indices": list(expected_indices),
        "clip_specific_mapping": False,
    }
    _write_json(output / "fingertip_mapping.json", mapping)
    _write_json(output / "contact_reward_fingertip_mapping.json", mapping)

    frozen_inputs: dict[str, Any] = {
        "schema_version": "Stage16DRewardV3FrozenInputsV1",
        "reference_kinematics_version_required": 2,
        "contract": contract.as_dict(),
        "references": {},
        "reference_contact_masks": {},
    }
    masks: dict[str, np.ndarray] = {}
    mask_reports: dict[str, Any] = {}
    for clip in CLIPS:
        clip_short = clip.removeprefix("hocap_")
        reference = (args.reference_root / f"{clip}.reference_kinematics_v2.npz").resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"CONTACT_MASK_REFERENCE_INPUT_MISSING:{reference}")
        with np.load(reference, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata.get("reference_kinematics_version", -1)) != 2:
                raise ValueError("CONTACT_MASK_REQUIRES_REFERENCE_KINEMATICS_V2")
        mask, receipt = _load_frozen_reference_mask(
            args.reference_contact_mask_root / f"reference_contact_mask_{clip_short}.npz",
            clip=clip,
        )
        masks[clip] = mask
        mask_reports[clip] = receipt
        frozen_inputs["references"][clip] = {"path": str(reference), "sha256": _sha256(reference)}
        frozen_inputs["reference_contact_masks"][clip] = receipt

    trace_paths = {
        "hocap_170105": args.v1_formal_170105,
        "hocap_170650": args.v1_formal_170650,
    }
    trace_reports: dict[str, dict[str, Any]] = {}
    positive_by_clip: dict[str, np.ndarray] = {}
    for clip, trace_path in trace_paths.items():
        report, positive = _qualify_exact_pair_force_trace(
            trace_path, clip=clip, reference_mask=masks[clip], expected_indices=expected_indices
        )
        trace_reports[clip] = report
        positive_by_clip[clip] = positive
    v2_diagnostic: dict[str, Any] = {"path": str(args.v2_formal_diagnostic.resolve())}
    if args.v2_formal_diagnostic.is_file():
        with np.load(args.v2_formal_diagnostic, allow_pickle=False) as archive:
            v2_diagnostic.update(exact_pair_force_trace_status(archive.files))
            v2_diagnostic["sha256"] = _sha256(args.v2_formal_diagnostic)
    else:
        v2_diagnostic["status"] = "V2_DIAGNOSTIC_NOT_AVAILABLE"
    pair_force_ready = all(row.get("status") == "PASS" for row in trace_reports.values())
    clip_coverage = all(values.size > 0 for values in positive_by_clip.values())
    pooled = np.concatenate(list(positive_by_clip.values())) if pair_force_ready else np.empty(0)
    if not pair_force_ready:
        status = "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED"
    elif not clip_coverage:
        status = "CONTACT_REWARD_CALIBRATION_CLIP_COVERAGE_FAILURE"
    elif pooled.size < 100:
        status = "CONTACT_REWARD_CALIBRATION_INSUFFICIENT_POSITIVE_SAMPLES"
    elif not np.isfinite(pooled).all():
        status = "CONTACT_REWARD_CALIBRATION_NONFINITE"
    else:
        status = "CONTACT_REWARD_CONTRACT_FROZEN"
    lambda_c_n = (
        float(np.quantile(pooled, 0.5)) if status == "CONTACT_REWARD_CONTRACT_FROZEN" else None
    )
    if lambda_c_n is not None and lambda_c_n <= contract.epsilon_n:
        status = "CONTACT_REWARD_CALIBRATION_LAMBDA_INVALID"
        lambda_c_n = None
    calibration = {
        "schema_version": "ContactRewardScaleCalibrationV1",
        "status": status,
        "primary_calibration_sources": trace_reports,
        "reward_v2_diagnostic_only": v2_diagnostic,
        "positive_contact_minimum_frames": 100,
        "lambda_rule": "combined_positive_contact_S_contact_p50",
        "clip_statistics": {
            clip: _statistics(values) if values.size else {"n": 0}
            for clip, values in positive_by_clip.items()
        },
        "pooled_positive_contact_statistics": _statistics(pooled) if pooled.size else {"n": 0},
        "lambda_c_n": lambda_c_n,
        "reason": None if lambda_c_n is not None else status,
    }
    contract_receipt = {
        "schema_version": "Stage16DReferenceGatedContactRewardV3ContractV1",
        "status": status,
        "reward": "RewardV3 = RewardV2 + r_contact",
        "frozen_parameters": {**contract.as_dict(), "lambda_c_n": lambda_c_n},
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
    _write_json(
        output / "reference_contact_mask_contract.json",
        {
            "schema_version": "Stage16DReferenceContactMaskContractV1",
            "status": "PASS",
            "reference_kinematics_version": 2,
            "finger_order": list(CONTACT_FINGER_ORDER),
            "fingertip_links": list(EVALUATION_FINGERTIP_LINKS),
            "threshold_m": contract.xi_c_m,
            "mask_source": "frozen V2-reference visual-surface masks; no runtime force input",
            "masks": mask_reports,
        },
    )
    _write_json(output / "pair_force_runtime_mapping.json", mapping)
    _write_json(output / "pair_force_artifact_qualification.json", trace_reports)
    _write_json(output / "contact_reward_contract.json", contract_receipt)
    _write_json(output / "reward_v3_contract.json", contract_receipt)
    _write_json(output / "force_scale_calibration.json", calibration)
    _write_json(output / "contact_reward_scale_calibration.json", calibration)
    _write_json(output / "information_flow_audit.json", information_flow)
    _write_json(output / "contact_reward_information_flow_audit.json", information_flow)
    response_rows: list[dict[str, float | str]] = []
    if lambda_c_n is not None:
        statistics = calibration["pooled_positive_contact_statistics"]
        assert isinstance(statistics, dict)
        scales = (
            [("zero", 0.0)]
            + [
                (label, float(statistics[label]))
                for label in ("p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
            ]
            + [("twice_p99", 2.0 * float(statistics["p99"]))]
        )
        for label, scale in scales:
            reward = (
                0.0 if scale == 0.0 else float(np.exp(-lambda_c_n / (scale + contract.epsilon_n)))
            )
            response_rows.append({"scale_label": label, "S_contact_n": scale, "r_contact": reward})
    response = {
        "schema_version": "ContactRewardResponseAuditV1",
        "status": "PASS" if lambda_c_n is not None else "BLOCKED_NO_FROZEN_LAMBDA",
        "reason": None if lambda_c_n is not None else status,
        "response_rows": response_rows,
        "lambda_response_error": (
            abs(float(np.exp(-lambda_c_n / (lambda_c_n + contract.epsilon_n))) - np.exp(-1.0))
            if lambda_c_n is not None
            else None
        ),
    }
    _write_json(output / "reward_response.json", response)
    _write_json(output / "contact_reward_response.json", response)
    csv_lines = ["scale_label,S_contact_n,r_contact"]
    csv_lines.extend(
        f"{row['scale_label']},{row['S_contact_n']:.12g},{row['r_contact']:.12g}"
        for row in response_rows
    )
    (output / "reward_response.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output / "contact_reward_response.csv").write_text(
        (output / "reward_response.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "failure_transitions.jsonl").write_text(
        json.dumps(
            {
                "from": "PAIR_FORCE_QUALIFICATION",
                "to": status,
                "training_authorized": lambda_c_n is not None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final_summary = {
        "schema_version": "Stage16DRewardV3ContactFinalSummaryV1",
        "status": status,
        "reward_contract": "TopoRetargetReferenceTrackingReward26DV3",
        "training": "AUTHORIZED_NOT_STARTED"
        if lambda_c_n is not None
        else "NOT_STARTED_PAIR_FORCE_SCALE_UNRESOLVED",
        "formal20": "NOT_RUN_PRETRAINING_GATE_BLOCKED",
        "simulation_data": "NOT_EXPORTED_PRETRAINING_GATE_BLOCKED",
        "replay": "NOT_RUN_NO_V3_TRACE",
        "next_required_input": None
        if lambda_c_n is not None
        else "frozen V1 Formal20 trace with exact fingertip-object pair force",
    }
    _write_json(output / "final_summary.json", final_summary)
    (output / "final_summary.md").write_text(
        "# Stage 16-D Reward V3 Contact Pretraining Gate\n\n"
        f"Status: `{final_summary['status']}`.\n\n"
        + (
            "The frozen V1 Formal20 exact pair-force traces passed qualification and froze the "
            f"shared `lambda_c={lambda_c_n:.9g}` N.\n"
            if lambda_c_n is not None
            else "The V1 Formal20 pair-force contract is incomplete; PPO is not authorized.\n"
        ),
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
            "training_authorized": lambda_c_n is not None,
        },
    )
    _write_json(
        output / "reward_v3_preflight.json",
        {
            "schema_version": "Stage16DRewardV3ContactPreflightV2",
            "status": status,
            "training_authorized": lambda_c_n is not None,
            "pair_force_artifacts": trace_reports,
            "reference_contact_masks": mask_reports,
        },
    )
    _write_json(
        output / "reward_v3_contract_hash.json",
        {
            "path": str((output / "reward_v3_contract.json").resolve()),
            "sha256": _sha256(output / "reward_v3_contract.json"),
        },
    )
    _write_json(
        output / "old_v3_blocker.json",
        {
            "path": str(args.old_blocker.resolve()),
            "sha256": _sha256(args.old_blocker) if args.old_blocker.is_file() else None,
            "status": "HISTORICAL_BLOCKER_RECORDED",
        },
    )
    print(json.dumps({"status": status, "output_root": str(output)}))
    if args.require_ready and lambda_c_n is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
