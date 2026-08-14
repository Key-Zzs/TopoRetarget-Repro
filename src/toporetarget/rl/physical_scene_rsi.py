"""Physical reference validity and support-aware RSI qualification.

This module is deliberately offline and result-independent.  It reconstructs
the complete Wuji collision-body state from the V2 wrist/finger reference,
queries the frozen runtime collision proxies with python-fcl, and adds the
finite inferred table as an explicit collision actor for the support metrics.
It does not alter the Stage16 environment or any historical RSI bank.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.physics.support.types import SupportPlaneConsistencyGateV1

from .geometry_audit.exact_evaluator import evaluate_runtime_proxy_state
from .geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)
from .geometry_audit.runtime_geometry import (
    ConvexProxyGeometry,
    load_runtime_geometry_manifest,
)
from .geometry_audit.transforms import compose_poses
from .physics_retargeting.self_collision import InterFingerCapsulePenetrationV1
from .rsi.contact_ready_v2 import build_contact_ready_state_bank

CLIPS = ("hocap_170105", "hocap_170650")
FRAME_COUNT = 321
PHYSICAL_VALIDITY_SCHEMA = "PhysicalReferenceValidityMaskV1"
PHYSICAL_SAFE_BANK_SCHEMA = "PhysicalSafeRSIBankV1"

# These values are existing frozen gates, copied as named constants so the
# qualification report can prove that no result-dependent threshold was used.
HO_MAX_M = 0.010
HO_ACTIVE_P95_M = 0.003
INTER_FINGER_MAX_M = 0.003
TABLE_OBJECT_MAX_PENETRATION_M = 0.002
TABLE_HAND_MAX_PENETRATION_M = 0.002
TABLE_SUPPORT_GAP_M = 0.005

SEMANTIC_TO_BANK = {
    "PRE_CONTACT": "PRE_CONTACT_TABLE_SUPPORTED_SAFE",
    "NEAR_CONTACT": "NEAR_CONTACT_PHYSICAL_SAFE",
    "CONTACT_READY": "CONTACT_READY_PHYSICAL_SAFE",
    "PERSISTENT_CONTACT": "PERSISTENT_PHYSICAL_SAFE",
    "MANIPULATION": "MANIPULATION_PHYSICAL_SAFE",
    "TERMINAL_HOLD": "TERMINAL_PHYSICAL_SAFE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_table_proxy(path: Path) -> dict[str, Any]:
    payload = _json(path)
    required = {"table_pose", "table_extent", "table_thickness", "plane_normal", "plane_offset"}
    if not required.issubset(payload):
        raise ValueError(f"PHYSICAL_TABLE_PROXY_FIELDS_MISSING:{path}")
    pose = np.asarray(payload["table_pose"], dtype=np.float64)
    extent = np.asarray(payload["table_extent"], dtype=np.float64)
    normal = np.asarray(payload["plane_normal"], dtype=np.float64)
    if pose.shape != (7,) or extent.shape != (2,) or normal.shape != (3,):
        raise ValueError(f"PHYSICAL_TABLE_PROXY_SHAPE_INVALID:{path}")
    if not all(np.isfinite(value).all() for value in (pose, extent, normal)):
        raise ValueError(f"PHYSICAL_TABLE_PROXY_NONFINITE:{path}")
    if np.any(extent <= 0.0) or float(payload["table_thickness"]) <= 0.0:
        raise ValueError(f"PHYSICAL_TABLE_PROXY_DIMENSION_INVALID:{path}")
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError(f"PHYSICAL_TABLE_PROXY_NORMAL_INVALID:{path}")
    payload["table_pose"] = pose.tolist()
    payload["table_extent"] = extent.tolist()
    payload["plane_normal"] = (normal / norm).tolist()
    payload["table_thickness"] = float(payload["table_thickness"])
    payload["plane_offset"] = float(payload["plane_offset"])
    return payload


def _table_center_pose(proxy: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(proxy["table_pose"], dtype=np.float64).copy()
    normal = np.asarray(proxy["plane_normal"], dtype=np.float64)
    pose[:3] -= 0.5 * float(proxy["table_thickness"]) * normal
    return pose


def _table_query(
    *,
    backend: Any,
    shapes: list[Any],
    proxies: list[ConvexProxyGeometry],
    poses: np.ndarray,
    table_shape: Any,
    table_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame maximum penetration and minimum signed distance."""

    root_poses = np.asarray(poses, dtype=np.float64)
    if root_poses.ndim not in {2, 3} or root_poses.shape[-1] != 7:
        raise ValueError("PHYSICAL_TABLE_QUERY_POSES_INVALID")
    if root_poses.ndim == 3 and root_poses.shape[1] != len(proxies):
        raise ValueError("PHYSICAL_TABLE_QUERY_BODY_COUNT_INVALID")
    penetration = np.zeros(root_poses.shape[0], dtype=np.float64)
    signed = np.full(root_poses.shape[0], np.inf, dtype=np.float64)
    for frame in range(root_poses.shape[0]):
        for proxy_index, (proxy, shape) in enumerate(zip(proxies, shapes, strict=True)):
            root_pose = (
                root_poses[frame, proxy_index] if root_poses.ndim == 3 else root_poses[frame]
            )
            world_pose = compose_poses(root_pose, proxy.local_pose_xyz_wxyz)
            result = backend.query(shape, world_pose, table_shape, table_pose)
            if not result.converged:
                raise RuntimeError("PHYSICAL_TABLE_QUERY_NONCONVERGENCE")
            penetration[frame] = max(penetration[frame], float(result.penetration_depth_m))
            signed[frame] = min(signed[frame], float(result.signed_separation_m))
    return penetration, signed


def evaluate_physical_pose_geometry(
    *,
    clip: str,
    wrist_pose: np.ndarray,
    finger_q: np.ndarray,
    object_pose: np.ndarray,
    geometry_manifest_path: Path,
    table_proxy_path: Path,
    repo_root: Path,
    self_collision_manifest_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate exact runtime geometry for an arbitrary PhysX pose trace."""

    if clip not in CLIPS:
        raise ValueError(f"PHYSICAL_UNKNOWN_CLIP:{clip}")
    wrist = np.asarray(wrist_pose, dtype=np.float64)
    q = np.asarray(finger_q, dtype=np.float64)
    obj = np.asarray(object_pose, dtype=np.float64)
    if wrist.shape[:-1] != q.shape[:-1] or wrist.shape[-1] != 7 or q.shape[-1] != 20:
        raise ValueError("PHYSICAL_TRACE_HAND_POSE_SHAPE_INVALID")
    if obj.shape != wrist.shape:
        raise ValueError("PHYSICAL_TRACE_OBJECT_POSE_SHAPE_INVALID")
    trace_shape = wrist.shape[:-1]
    wrist_flat = wrist.reshape(-1, 7)
    q_flat = q.reshape(-1, 20)
    object_flat = obj.reshape(-1, 7)
    hand_pose = reconstruct_hand_collision_body_pose(wrist_flat, q_flat, repo_root=repo_root)
    _, raw = evaluate_runtime_proxy_state(
        manifest_path=geometry_manifest_path,
        clip=clip,
        object_pose=object_flat[:, None],
        hand_collision_body_pose=hand_pose[:, None],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    hand_object = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
    table_proxy = load_table_proxy(table_proxy_path)
    table_pose = _table_center_pose(table_proxy)
    from .geometry_audit.convex_query import PythonFCLConvexQueryBackend

    backend = PythonFCLConvexQueryBackend()
    hand_proxies, objects_by_clip = load_runtime_geometry_manifest(geometry_manifest_path)
    object_proxies = objects_by_clip[clip]
    table_shape = backend.box(
        (
            float(table_proxy["table_extent"][0]),
            float(table_proxy["table_extent"][1]),
            float(table_proxy["table_thickness"]),
        )
    )
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_shapes = [backend.proxy_shape(proxy) for proxy in object_proxies]
    hand_table, hand_table_signed = _table_query(
        backend=backend,
        shapes=hand_shapes,
        proxies=hand_proxies,
        poses=hand_pose,
        table_shape=table_shape,
        table_pose=table_pose,
    )
    object_table, object_table_signed = _table_query(
        backend=backend,
        shapes=object_shapes,
        proxies=object_proxies,
        poses=object_flat,
        table_shape=table_shape,
        table_pose=table_pose,
    )
    self_collision_path = self_collision_manifest_path or geometry_manifest_path
    import torch

    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        self_collision_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=0.65,
        device="cpu",
    )
    inter_finger = (
        metric.evaluate(torch.as_tensor(hand_pose, dtype=torch.float32))["maximum_penetration_m"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return {
        "hand_object_max_penetration_m": hand_object.reshape(trace_shape),
        "hand_table_max_penetration_m": hand_table.reshape(trace_shape),
        "hand_table_min_signed_distance_m": hand_table_signed.reshape(trace_shape),
        "object_table_max_penetration_m": object_table.reshape(trace_shape),
        "object_table_min_signed_distance_m": object_table_signed.reshape(trace_shape),
        "inter_finger_max_penetration_m": inter_finger.reshape(trace_shape),
    }


def _support_state(
    *,
    semantic: np.ndarray,
    object_table_penetration: np.ndarray,
    object_table_signed_distance: np.ndarray,
    hand_table_penetration: np.ndarray,
    hand_table_signed_distance: np.ndarray,
) -> np.ndarray:
    object_active = (object_table_penetration > 0.0) | (
        object_table_signed_distance <= TABLE_SUPPORT_GAP_M
    )
    hand_active = (hand_table_penetration > 0.0) | (
        hand_table_signed_distance <= TABLE_HAND_MAX_PENETRATION_M
    )
    result = np.full(len(semantic), "UNKNOWN", dtype="U24")
    result[object_active & ~np.isin(semantic, ["PRE_CONTACT", "AMBIGUOUS"])] = "SHARED_SUPPORT"
    result[object_active & np.isin(semantic, ["PRE_CONTACT", "AMBIGUOUS"])] = "TABLE_SUPPORTED"
    result[~object_active & hand_active] = "HAND_SUPPORTED"
    result[~object_active & ~hand_active & (semantic != "AMBIGUOUS")] = "AIRBORNE_OR_TRANSITION"
    return result


def _failure_reasons(
    *,
    hand_object: np.ndarray,
    hand_table: np.ndarray,
    object_table: np.ndarray,
    inter_finger: np.ndarray,
) -> np.ndarray:
    rows: list[str] = []
    for ho, ht, ot, inter in zip(hand_object, hand_table, object_table, inter_finger, strict=True):
        reasons: list[str] = []
        # For an individual reset frame, the formal active-p95 population has
        # one member, so its value is the frame's positive worst penetration.
        if ho > HO_ACTIVE_P95_M:
            reasons.append("H_O_ACTIVE_P95_GT_3MM")
        if ho >= HO_MAX_M:
            reasons.append("H_O_MAX_GE_10MM")
        if ht > TABLE_HAND_MAX_PENETRATION_M:
            reasons.append("H_T_GT_2MM")
        if ot > TABLE_OBJECT_MAX_PENETRATION_M:
            reasons.append("O_T_GT_2MM")
        if inter > INTER_FINGER_MAX_M:
            reasons.append("INTER_FINGER_GT_3MM")
        rows.append("|".join(reasons) if reasons else "")
    return np.asarray(rows, dtype="U128")


def _safe_bank(
    *,
    bank: dict[str, np.ndarray],
    validity: dict[str, np.ndarray],
    coverage_minimum_pre_contact: int = 8,
) -> dict[str, np.ndarray]:
    semantic = np.asarray(bank["semantic_class"]).astype("U24")
    support = np.asarray(validity["support_state"]).astype("U24")
    valid = np.asarray(validity["overall_reference_geometry_valid"], dtype=bool)
    admissible = valid & (support != "UNKNOWN") & (semantic != "AMBIGUOUS")
    # PRE_CONTACT is admitted only when an explicit finite table actor supports
    # the object.  Other semantic states preserve their source causality but
    # still require the same physical geometry mask.
    admissible &= (semantic != "PRE_CONTACT") | np.isin(
        support, ["TABLE_SUPPORTED", "SHARED_SUPPORT"]
    )
    safe_name = np.full(len(semantic), "", dtype="U40")
    for source_class, target_name in SEMANTIC_TO_BANK.items():
        safe_name[(semantic == source_class) & admissible] = target_name
    selected = np.flatnonzero(safe_name != "")
    pre_count = int(np.count_nonzero(safe_name == SEMANTIC_TO_BANK["PRE_CONTACT"]))
    coverage = {
        "schema_version": "PhysicalSafeRSICoverageGateV1",
        "gate_frozen_before_results": True,
        "minimum_pre_contact_table_supported": coverage_minimum_pre_contact,
        "pre_contact_table_supported_count": pre_count,
        "pre_contact_table_supported_pass": pre_count >= coverage_minimum_pre_contact,
        "semantic_counts": dict(Counter(semantic.tolist())),
        "admitted_bank_counts": dict(Counter(safe_name[selected].tolist())),
        "support_counts": dict(Counter(support.tolist())),
        "geometry_valid_count": int(valid.sum()),
        "selected_count": int(len(selected)),
    }
    return {
        "runtime_index": np.asarray(bank["runtime_index"], dtype=np.int64)[selected],
        "source_index_or_interval": np.asarray(bank["source_index_or_interval"], dtype=np.int64)[
            selected
        ],
        "semantic_class": semantic[selected],
        "source_expected_contact": np.asarray(bank["source_expected_contact"], dtype=bool)[
            selected
        ],
        "classification_confidence": np.asarray(bank["classification_confidence"])[selected],
        "classification_evidence": np.asarray(bank["classification_evidence"])[selected],
        "retargeted_geometry_gap_m": np.asarray(
            bank["retargeted_geometry_gap_m"], dtype=np.float64
        )[selected],
        "physical_safe_bank": safe_name[selected],
        "support_state": support[selected],
        "reference_geometry_failure_reason": np.asarray(
            validity["reference_geometry_failure_reason"]
        )[selected],
        "all_runtime_index": np.asarray(bank["runtime_index"], dtype=np.int64),
        "all_semantic_class": semantic,
        "all_physical_safe_bank": safe_name,
        "all_support_state": support,
        "all_geometry_valid": valid,
        "coverage_gate_json": np.asarray(json.dumps(coverage, sort_keys=True)),
        "contract_identifier": np.asarray(PHYSICAL_SAFE_BANK_SCHEMA),
    }


def build_physical_reference_validity_mask(
    *,
    clip: str,
    reference_path: Path,
    source_contact_evidence_path: Path,
    geometry_manifest_path: Path,
    table_proxy_path: Path,
    repo_root: Path,
    self_collision_manifest_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build all 321 rows of the physical reference mask and safe bank."""

    if clip not in CLIPS:
        raise ValueError(f"PHYSICAL_UNKNOWN_CLIP:{clip}")
    with np.load(reference_path, allow_pickle=False) as archive:
        wrist_pose = np.concatenate(
            [
                np.asarray(archive["wrist_pose_translation_world_ref"], dtype=np.float64),
                np.asarray(archive["wrist_pose_quaternion_world_ref_wxyz"], dtype=np.float64),
            ],
            axis=1,
        )
        finger_q = np.asarray(archive["q_finger_ref"], dtype=np.float64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    if wrist_pose.shape != (FRAME_COUNT, 7) or finger_q.shape != (FRAME_COUNT, 20):
        raise ValueError("PHYSICAL_REFERENCE_MUST_HAVE_321_V2_FRAMES")
    if timestamps.shape != (FRAME_COUNT,) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("PHYSICAL_REFERENCE_TIMESTAMPS_INVALID")

    hand_pose = reconstruct_hand_collision_body_pose(wrist_pose, finger_q, repo_root=repo_root)
    hand_proxies, objects_by_clip = load_runtime_geometry_manifest(geometry_manifest_path)
    object_proxies = objects_by_clip[clip]
    with np.load(reference_path, allow_pickle=False) as archive:
        object_pose = np.concatenate(
            [
                np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64),
                np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64),
            ],
            axis=1,
        )
        object_twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
    geometry, raw = evaluate_runtime_proxy_state(
        manifest_path=geometry_manifest_path,
        clip=clip,
        object_pose=object_pose[:, None],
        hand_collision_body_pose=hand_pose[:, None],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    hand_object = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
    worst_pair = np.asarray(raw["frame_worst_pair_index"], dtype=np.int64)[:, 0]
    pair_ids = [str(value) for value in raw["pair_ids"].tolist()]

    # The table is a frozen finite box.  It participates in the exact query but
    # is never written or moved during a rollout.
    table_proxy = load_table_proxy(table_proxy_path)
    table_pose = _table_center_pose(table_proxy)
    from .geometry_audit.convex_query import PythonFCLConvexQueryBackend

    backend = PythonFCLConvexQueryBackend()
    table_shape = backend.box(
        (
            float(table_proxy["table_extent"][0]),
            float(table_proxy["table_extent"][1]),
            float(table_proxy["table_thickness"]),
        )
    )
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_shapes = [backend.proxy_shape(proxy) for proxy in object_proxies]
    hand_table, hand_table_signed = _table_query(
        backend=backend,
        shapes=hand_shapes,
        proxies=hand_proxies,
        poses=hand_pose,
        table_shape=table_shape,
        table_pose=table_pose,
    )
    object_table, object_table_signed = _table_query(
        backend=backend,
        shapes=object_shapes,
        proxies=object_proxies,
        poses=object_pose,
        table_shape=table_shape,
        table_pose=table_pose,
    )

    self_collision_path = self_collision_manifest_path or geometry_manifest_path
    import torch

    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        self_collision_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=0.65,
        device="cpu",
    )
    inter_finger = metric.evaluate(torch.as_tensor(hand_pose, dtype=torch.float32))[
        "maximum_penetration_m"
    ]
    inter_finger_np = inter_finger.detach().cpu().numpy().astype(np.float64)

    state_bank = build_contact_ready_state_bank(
        reference_path=reference_path,
        source_contact_evidence_path=source_contact_evidence_path,
    )
    semantic = np.asarray(state_bank["semantic_class"]).astype("U24")
    support = _support_state(
        semantic=semantic,
        object_table_penetration=object_table,
        object_table_signed_distance=object_table_signed,
        hand_table_penetration=hand_table,
        hand_table_signed_distance=hand_table_signed,
    )
    reasons = _failure_reasons(
        hand_object=hand_object,
        hand_table=hand_table,
        object_table=object_table,
        inter_finger=inter_finger_np,
    )
    valid = reasons == ""
    # The map is reset-oriented: for one frame, the positive active p95
    # population is that frame.  The full-clip formal result below still uses
    rows = {
        "runtime_index": np.arange(FRAME_COUNT, dtype=np.int64),
        "source_index_or_interval": np.asarray(
            state_bank["source_index_or_interval"], dtype=np.int64
        ),
        "timestamp_s": timestamps,
        "semantic_class": semantic,
        "reference_object_pose": object_pose,
        "reference_object_twist": object_twist,
        "reference_wrist_pose": wrist_pose,
        "reference_q_finger": finger_q,
        "source_expected_contact": np.asarray(state_bank["source_expected_contact"], dtype=bool),
        "classification_confidence": np.asarray(state_bank["classification_confidence"]),
        "classification_evidence": np.asarray(state_bank["classification_evidence"]),
        "retargeted_geometry_gap_m": np.asarray(
            state_bank["retargeted_geometry_gap_m"], dtype=np.float64
        ),
        "reference_linear_speed_mps": np.asarray(
            state_bank["reference_linear_speed_mps"], dtype=np.float64
        ),
        "reference_angular_speed_radps": np.asarray(
            state_bank["reference_angular_speed_radps"], dtype=np.float64
        ),
        "support_state": support,
        "hand_object_max_penetration_m": hand_object,
        "hand_object_worst_pair": np.asarray(
            [pair_ids[index] for index in worst_pair], dtype="U512"
        ),
        "hand_table_max_penetration_m": hand_table,
        "hand_table_min_signed_distance_m": hand_table_signed,
        "object_table_max_penetration_m": object_table,
        "object_table_min_signed_distance_m": object_table_signed,
        "inter_finger_max_penetration_m": inter_finger_np,
        "overall_reference_geometry_valid": valid,
        "reference_geometry_failure_reason": reasons,
    }
    safe_bank = _safe_bank(bank=state_bank, validity=rows)
    active = hand_object[hand_object > 0.0]
    trajectory = {
        "schema_version": "PhysicalReferenceTrajectoryGeometryQualificationV1",
        "clip": clip,
        "frame_count": FRAME_COUNT,
        "active_hand_object_p95_m": float(np.quantile(active, 0.95)) if active.size else 0.0,
        "hand_object_max_m": float(hand_object.max(initial=0.0)),
        "hand_table_max_m": float(hand_table.max(initial=0.0)),
        "object_table_max_m": float(object_table.max(initial=0.0)),
        "inter_finger_max_m": float(inter_finger_np.max(initial=0.0)),
        "active_hand_object_p95_pass": bool(
            (float(np.quantile(active, 0.95)) if active.size else 0.0) <= HO_ACTIVE_P95_M
        ),
        "hand_object_max_pass": bool(hand_object.max(initial=0.0) < HO_MAX_M),
        "hand_table_max_pass": bool(hand_table.max(initial=0.0) <= TABLE_HAND_MAX_PENETRATION_M),
        "object_table_max_pass": bool(
            object_table.max(initial=0.0) <= TABLE_OBJECT_MAX_PENETRATION_M
        ),
        "inter_finger_max_pass": bool(inter_finger_np.max(initial=0.0) <= INTER_FINGER_MAX_M),
        "formal_trajectory_geometry_valid": bool(
            (float(np.quantile(active, 0.95)) if active.size else 0.0) <= HO_ACTIVE_P95_M
            and hand_object.max(initial=0.0) < HO_MAX_M
            and hand_table.max(initial=0.0) <= TABLE_HAND_MAX_PENETRATION_M
            and object_table.max(initial=0.0) <= TABLE_OBJECT_MAX_PENETRATION_M
            and inter_finger_np.max(initial=0.0) <= INTER_FINGER_MAX_M
        ),
        "frame_valid_count": int(valid.sum()),
        "frame_invalid_indices": np.flatnonzero(~valid).tolist(),
        "support_state_counts": dict(Counter(support.tolist())),
        "semantic_counts": dict(Counter(semantic.tolist())),
    }
    return (
        rows,
        safe_bank,
        {
            "schema_version": PHYSICAL_VALIDITY_SCHEMA,
            "clip": clip,
            "reference_geometry": geometry,
            "trajectory": trajectory,
            "thresholds": {
                "hand_object_active_p95_m": HO_ACTIVE_P95_M,
                "hand_object_max_exclusive_m": HO_MAX_M,
                "hand_table_max_inclusive_m": TABLE_HAND_MAX_PENETRATION_M,
                "object_table_max_inclusive_m": TABLE_OBJECT_MAX_PENETRATION_M,
                "inter_finger_max_inclusive_m": INTER_FINGER_MAX_M,
                "table_support_gap_m": TABLE_SUPPORT_GAP_M,
                "source_support_gate": SupportPlaneConsistencyGateV1().as_dict(),
            },
            "source_contact_evidence": str(source_contact_evidence_path.resolve()),
            "geometry_manifest": str(geometry_manifest_path.resolve()),
            "table_proxy": str(table_proxy_path.resolve()),
            "physical_safe_rsi": json.loads(str(safe_bank["coverage_gate_json"].item())),
        },
    )


__all__ = [
    "CLIPS",
    "FRAME_COUNT",
    "PHYSICAL_SAFE_BANK_SCHEMA",
    "PHYSICAL_VALIDITY_SCHEMA",
    "build_physical_reference_validity_mask",
    "evaluate_physical_pose_geometry",
    "load_table_proxy",
    "sha256_file",
]
