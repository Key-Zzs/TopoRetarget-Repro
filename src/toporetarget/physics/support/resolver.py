"""Source-first support resolver with an explicit planar fallback."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from .planar_inference import (
    audit_candidate_support_intervals,
    detect_stable_pre_contact_interval,
    infer_planar_support,
)
from .source_evidence import (
    NormalizedSourceEvidence,
    call_support_evidence_adapter,
    normalize_source_evidence,
)
from .types import (
    GeometryValidation,
    PhysicsValidation,
    StableIntervalResult,
    StablePreContactDetectionContractV1,
    SupportExtentContractV1,
    SupportPlaneConsistencyGateV1,
    SupportResolutionMode,
    SupportResolutionResult,
    SupportResolutionStatus,
    SupportType,
)


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _source_geometry_result(
    *,
    source: NormalizedSourceEvidence,
    dataset: str,
    sequence: str,
    hashes: Mapping[str, str],
) -> SupportResolutionResult:
    support_type = source.support_type
    assert support_type is not None
    return SupportResolutionResult(
        status=SupportResolutionStatus.SOURCE_SUPPORT_VALIDATED.value,
        support_type=support_type,
        support_source=(
            "source_explicit_metadata" if source.explicit_validated else "source_scene_geometry"
        ),
        source_explicit=source.explicit_validated,
        source_recovered=source.recovered_validated,
        support_inferred=False,
        confidence=1.0,
        plane_normal=None,
        plane_offset=None,
        table_pose=None,
        table_extent=None,
        table_thickness=None,
        support_interval=None,
        visual_mesh_evidence={},
        collision_mesh_evidence={},
        geometry_validation={},
        physics_validation={},
        provenance={
            "dataset": dataset,
            "sequence": sequence,
            "source_evidence": source.as_dict(),
        },
        hashes=dict(hashes),
        diagnostics={"source_support_validation": source.validation},
    )


def resolve_support(
    *,
    dataset: str,
    sequence: str,
    object_visual_vertices_local: np.ndarray,
    object_pose_translation_world: np.ndarray,
    object_pose_quaternion_world_wxyz: np.ndarray,
    timestamps: Sequence[float],
    gravity_world_mps2: Sequence[float],
    object_collision_vertices_local: np.ndarray | None = None,
    source_support: object | None = None,
    source_adapter: object | None = None,
    source_reference_kind: str = "stage16_retargeted_runtime_object_pose",
    contact_mask: np.ndarray | None = None,
    object_twist_world: np.ndarray | None = None,
    mode: SupportResolutionMode | str = SupportResolutionMode.AUTO,
    detection_contract: StablePreContactDetectionContractV1 | None = None,
    extent_contract: SupportExtentContractV1 | None = None,
    geometry_gate: SupportPlaneConsistencyGateV1 | None = None,
) -> SupportResolutionResult:
    """Resolve explicit -> reconstructed -> inferred -> unresolved, in that order."""

    selected_mode = SupportResolutionMode(mode)
    visual = np.asarray(object_visual_vertices_local, dtype=np.float64)
    collision = (
        np.asarray(object_collision_vertices_local, dtype=np.float64)
        if object_collision_vertices_local is not None
        else None
    )
    translation = np.asarray(object_pose_translation_world, dtype=np.float64)
    quaternion = np.asarray(object_pose_quaternion_world_wxyz, dtype=np.float64)
    hashes = {
        "visual_mesh": _hash_array(visual),
        "collision_mesh": _hash_array(collision) if collision is not None else "NOT_PROVIDED",
        "object_pose_translation": _hash_array(translation),
        "object_pose_quaternion": _hash_array(quaternion),
        "contact_mask": (
            _hash_array(np.asarray(contact_mask, dtype=bool))
            if contact_mask is not None
            else "NOT_PROVIDED"
        ),
    }
    if source_adapter is not None:
        source_raw = call_support_evidence_adapter(source_adapter, sequence)
    else:
        source_raw = source_support
    source = normalize_source_evidence(source_raw)
    source_rejected = bool(source.explicit or source.recovered) and not source.has_source_support
    if selected_mode is not SupportResolutionMode.INFERRED_PLANAR and source.has_source_support:
        return _source_geometry_result(
            source=source,
            dataset=dataset,
            sequence=sequence,
            hashes=hashes,
        )
    if selected_mode is SupportResolutionMode.SOURCE_ONLY:
        return _unknown_result(
            dataset=dataset,
            sequence=sequence,
            hashes=hashes,
            source=source,
            reason="SOURCE_ONLY_REQUESTED_BUT_SOURCE_SUPPORT_NOT_VALIDATED",
        )
    stable = detect_stable_pre_contact_interval(
        timestamps=timestamps,
        object_translation_world=translation,
        object_quaternion_world_wxyz=quaternion,
        gravity=gravity_world_mps2,
        contact_mask=contact_mask,
        object_twist_world=object_twist_world,
        contract=detection_contract,
    )
    candidate_interval_audit = audit_candidate_support_intervals(
        visual_vertices_local=visual,
        collision_vertices_local=collision,
        object_translation_world=translation,
        object_quaternion_world_wxyz=quaternion,
        gravity=gravity_world_mps2,
        candidates=stable.candidate_intervals,
        extent_contract=extent_contract,
        detection_contract=detection_contract,
        geometry_gate=geometry_gate,
    )
    if stable.interval is None:
        return _unknown_result(
            dataset=dataset,
            sequence=sequence,
            hashes=hashes,
            source=source,
            reason=stable.reason or "PLANAR_SUPPORT_INFERENCE_NOT_AUTHORIZED",
            stable=stable,
            diagnostics={
                "candidate_interval_audit": candidate_interval_audit,
                "candidate_interval_policy": (
                    "stable_pre_manipulation_geometry_authorizes_inference;_"
                    "hand_contact_timing_is_separate_provenance"
                ),
            },
        )
    try:
        plane_fit, proxy, plane_evidence = infer_planar_support(
            visual_vertices_local=visual,
            collision_vertices_local=collision,
            object_translation_world=translation,
            object_quaternion_world_wxyz=quaternion,
            gravity=gravity_world_mps2,
            stable_interval=stable.interval,
            extent_contract=extent_contract,
            detection_contract=detection_contract,
            geometry_gate=geometry_gate,
        )
    except ValueError as error:
        return _unknown_result(
            dataset=dataset,
            sequence=sequence,
            hashes=hashes,
            source=source,
            reason=str(error),
            stable=stable,
        )
    return SupportResolutionResult(
        status=SupportResolutionStatus.SUPPORT_RECONSTRUCTION_BLOCKED.value,
        support_type=SupportType.INFERRED_PLANAR_SUPPORT,
        support_source="inferred_from_object_mesh_pose_gravity_stable_pre_contact",
        source_explicit=False,
        source_recovered=False,
        support_inferred=True,
        confidence=_confidence(plane_fit, source_rejected),
        plane_normal=plane_fit.plane_normal,
        plane_offset=plane_fit.plane_offset,
        table_pose=proxy.table_pose,
        table_extent=proxy.table_extent,
        table_thickness=proxy.table_thickness,
        support_interval=stable.interval,
        visual_mesh_evidence={
            "vertex_count": int(len(visual)),
            "mesh_hash": hashes["visual_mesh"],
            "source_reference_kind": source_reference_kind,
        },
        collision_mesh_evidence={
            "vertex_count": int(len(collision)) if collision is not None else 0,
            "mesh_hash": hashes["collision_mesh"],
            "runtime_collision_provided": collision is not None,
        },
        geometry_validation={},
        physics_validation={},
        provenance={
            "dataset": dataset,
            "sequence": sequence,
            "source_support": source.as_dict(),
            "source_support_rejected_before_inference": source_rejected,
            "source_explicit": False,
            "source_recovered": False,
            "support_inferred": True,
            "gravity_world_mps2": [float(v) for v in gravity_world_mps2],
            "source_reference_kind": source_reference_kind,
            "thresholds": (detection_contract or StablePreContactDetectionContractV1()).as_dict(),
            "extent_contract": (extent_contract or SupportExtentContractV1()).as_dict(),
            "geometry_gate": (geometry_gate or SupportPlaneConsistencyGateV1()).as_dict(),
        },
        hashes=hashes,
        stable_interval=stable,
        plane_fit=plane_fit,
        table_proxy=proxy,
        diagnostics={
            "plane_evidence": plane_evidence,
            "candidate_interval_audit": candidate_interval_audit,
            "candidate_interval_policy": (
                "stable_pre_manipulation_geometry_authorizes_inference;_"
                "hand_contact_timing_is_separate_provenance"
            ),
        },
    )


def validate_and_finalize_resolution(
    result: SupportResolutionResult,
    *,
    geometry: GeometryValidation | Mapping[str, object] | None = None,
    physics: PhysicsValidation | Mapping[str, object] | None = None,
    transfer_status: str = "NOT_RUN",
) -> SupportResolutionResult:
    """Attach validation evidence and derive a fail-closed final status."""

    geometry_dict = (
        geometry.as_dict() if isinstance(geometry, GeometryValidation) else dict(geometry or {})
    )
    physics_dict = (
        physics.as_dict() if isinstance(physics, PhysicsValidation) else dict(physics or {})
    )
    if result.support_type in {
        SupportType.SOURCE_EXPLICIT_SUPPORT,
        SupportType.SOURCE_RECONSTRUCTED_SUPPORT,
    }:
        status = SupportResolutionStatus.SOURCE_SUPPORT_VALIDATED.value
    elif result.support_type is SupportType.INFERRED_PLANAR_SUPPORT:
        geometry_pass = geometry_dict.get("status") == "PASS"
        physics_pass = physics_dict.get("status") == "PASS"
        if geometry_pass and physics_pass:
            status = (
                SupportResolutionStatus.INFERRED_SUPPORT_VALIDATED_TRANSFER_DEFERRED.value
                if transfer_status == "DEFERRED_BY_HAND_OBJECT_GEOMETRY"
                else SupportResolutionStatus.INFERRED_SUPPORT_VALIDATED.value
            )
        else:
            status = SupportResolutionStatus.SUPPORT_RECONSTRUCTION_BLOCKED.value
    else:
        status = result.status
    return SupportResolutionResult(
        **{
            **result.__dict__,
            "status": status,
            "geometry_validation": geometry_dict,
            "physics_validation": physics_dict,
            "transfer_status": transfer_status,
        }
    )


def _unknown_result(
    *,
    dataset: str,
    sequence: str,
    hashes: Mapping[str, str],
    source: NormalizedSourceEvidence,
    reason: str,
    stable: StableIntervalResult | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> SupportResolutionResult:
    return SupportResolutionResult(
        status=SupportResolutionStatus.SUPPORT_UNRESOLVED.value,
        support_type=SupportType.UNRESOLVED,
        support_source="none",
        source_explicit=source.explicit,
        source_recovered=source.recovered,
        support_inferred=False,
        confidence=0.0,
        plane_normal=None,
        plane_offset=None,
        table_pose=None,
        table_extent=None,
        table_thickness=None,
        support_interval=None,
        visual_mesh_evidence={},
        collision_mesh_evidence={},
        geometry_validation={},
        physics_validation={},
        provenance={
            "dataset": dataset,
            "sequence": sequence,
            "source_support": source.as_dict(),
        },
        hashes=dict(hashes),
        stable_interval=stable,
        diagnostics={"reason": reason, **dict(diagnostics or {})},
    )


def _confidence(plane_fit: object, source_rejected: bool) -> float:
    stats = plane_fit.h_visual_stats  # type: ignore[attr-defined]
    confidence = 1.0
    confidence *= max(0.0, 1.0 - float(stats["mad"]) / 0.0025)
    confidence *= max(0.0, 1.0 - float(stats["range"]) / 0.01)
    if source_rejected:
        confidence *= 0.9
    return float(min(1.0, max(0.0, confidence)))


__all__ = ["resolve_support", "validate_and_finalize_resolution"]
