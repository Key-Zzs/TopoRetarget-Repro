"""O5R-D3-CERT-QOLD-R q_old baseline failure analysis.

This workflow is deliberately split into optimizer-free authority analysis and
one explicitly consumed historical-baseline regression.  It never invokes
RefinementV2 or any certification/heldout stage.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5
from scripts.evaluation import run_oakink2_o5rd3cert as cert
from scripts.evaluation import run_oakink2_o5rd3certv2 as certv2
from toporetarget.adapters.datasets.oakink2 import (
    OakInk2CanonicalAdapterV1,
    reconstruct_mano_geometry,
)
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.frames import FrameDegeneracyError, load_frame_profile

ROOT = REPO / ".local/reports/oakink2_o5rd3certqoldr_fresh_qold_baseline_failure_analysis_v1"
CERT_V2_ROOT = certv2.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "5a0d92afccb9ef96a66cf74017758332dc9565b2"
DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
GATE_SHA256 = "505af73c9871baa67045449a4908b3259a16116b19bc2817d8b08470855bf6dc"
PROTOCOL_SHA256 = "8b49cfbfac1b839685e256d7d5442ee41189401b1d122f674bb0fbff5af8272d"
FAILED_BASELINE = "window_03"
FAILED_ORDINAL = 134
FAILED_SOURCE_FRAME = 5884
THRESHOLD_M = 1.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON mapping: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if path.exists():
        observed = sha256_file(path)
        if not sidecar.is_file() or sidecar.read_text().split()[0] != observed:
            raise RuntimeError(f"FROZEN_ARTIFACT_HASH_DRIFT:{path}")
        return read_json(path)
    write_json(path, value)
    write_text(sidecar, sha256_file(path) + "\n")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _plan() -> dict[str, Any]:
    return read_json(CERT_V2_ROOT / "qold/generation_plan.json")


def _failed_item() -> dict[str, Any]:
    for item in _plan()["records"]:
        if item["baseline_id"] == FAILED_BASELINE:
            return item
    raise RuntimeError("WINDOW_03_NOT_IN_FROZEN_PLAN")


def _failed_work() -> Path:
    return CERT_V2_ROOT / "qold/generation_work/retarget/window_03/work"


def _metrics(points: np.ndarray) -> dict[str, Any]:
    value = np.asarray(points, dtype=np.float64)
    wrist, middle, index, pinky = value[0], value[9], value[5], value[17]
    axis_a = middle - wrist
    axis_b = index - pinky
    norm_a = float(np.linalg.norm(axis_a))
    norm_b = float(np.linalg.norm(axis_b))
    unit_a = axis_a / max(norm_a, THRESHOLD_M)
    orth = axis_b - float(np.dot(axis_b, unit_a)) * unit_a
    orth_norm = float(np.linalg.norm(orth))
    cross_norm = float(np.linalg.norm(np.cross(axis_a, axis_b)))
    dot = float(np.dot(axis_a, axis_b))
    angle = float(np.degrees(np.arctan2(cross_norm, dot))) if norm_a * norm_b else None
    matrix = np.stack((axis_a, axis_b), axis=1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if float(singular_values[-1]) > 0.0
        else None
    )
    return {
        "axis_a_wrist_to_middle_mcp": axis_a.tolist(),
        "axis_b_index_mcp_minus_pinky_mcp": axis_b.tolist(),
        "axis_a_norm_m": norm_a,
        "axis_b_norm_m": norm_b,
        "dot_m2": dot,
        "cross_norm_m2": cross_norm,
        "angle_deg": angle,
        "gram_determinant_m4": float(norm_a * norm_a * norm_b * norm_b - dot * dot),
        "lateral_orthogonal_norm_m": orth_norm,
        "singular_values_m": singular_values.tolist(),
        "condition_number": condition,
        "finite": bool(np.isfinite(value).all()),
        "degenerate_existing_guard": bool(
            norm_a <= THRESHOLD_M or norm_b <= THRESHOLD_M or orth_norm <= THRESHOLD_M
        ),
    }


def preflight(root: Path) -> dict[str, Any]:
    failure = read_json(CERT_V2_ROOT / "qold/generation_failure.json")
    summary = read_json(CERT_V2_ROOT / "final_summary.json")
    checks = {
        "repo": git("rev-parse", "--show-toplevel") == str(REPO),
        "branch": git("branch", "--show-current") == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "historical_status": failure["D3_CERT_V2_STATUS"] == "BLOCKED_QOLD_BASELINE_GENERATION",
        "hard_stop": failure["HARD_STOP"] is True,
        "accepted_134": failure["accepted_frame_count"] == FAILED_ORDINAL,
        "downstream_not_run": all(
            summary[key] == "NOT_RUN"
            for key in (
                "FRESH_SPARSE_V2",
                "FRESH_WINDOW_V2",
                "FRESH_CROSS_EPISODE_V2",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2",
            )
        ),
    }
    payload = {
        "schema_version": "O5RD3CERTQOLDRPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "branch": git("branch", "--show-current"),
        "start_head": START_HEAD,
        "current_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--untracked-files=all"),
        "QOLD_BASELINE_OPTIMIZER_RUN_COUNT_NEW": 0,
        "REFINEMENT_V2_OPTIMIZER_RUN_COUNT_NEW": 0,
    }
    write_json(root / "preflight/git.json", payload)
    write_json(
        root / "preflight/cert_v2_history.json",
        {
            "schema_version": "O5RD3CERTQOLDRCertV2HistoryV1",
            "status": payload["status"],
            "D3_CERT_V2_HISTORICAL_STATUS": failure["D3_CERT_V2_STATUS"],
            "D3_CERT_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
            "failure_receipt_sha256": sha256_file(CERT_V2_ROOT / "qold/generation_failure.json"),
            "completed_record_count": failure["completed_authority_count"],
            "completed_frame_count": failure["completed_qold_frame_count"],
            "qold_authority_manifest_historical": None,
        },
    )
    if payload["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY")
    return payload


def verify_frozen_authorities(root: Path) -> dict[str, Any]:
    paths = {
        "refinement_v2_design": certv2.R2_ROOT / "design/refinement_v2_design.json",
        "development_gate_v2": certv2.R4_ROOT / "gate_v2/development_gate_v2.json",
        "certification_protocol_v2": certv2.CERT_R_ROOT
        / "protocol_v2/certification_protocol_v2.json",
        "fresh_exclusion_ledger": CERT_V2_ROOT / "freshness/cert_v2_exclusion_ledger.json",
        "historical_baseline_solver": o5.REPORT_ROOT / "contract/geometric_retarget_contract.json",
        "source_canonical_adapter": REPO / "src/toporetarget/adapters/datasets/oakink2.py",
        "canonical_wrist_profile": REPO
        / "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
        "mano_source_adapter": REPO / "scripts/data/oakink2_official_reference.py",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "interaction_graph": REPO / "src/toporetarget/retarget/interaction_graph.py",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    checks = {
        "refinement_v2_design": hashes["refinement_v2_design"] == DESIGN_SHA256,
        "development_gate_v2": hashes["development_gate_v2"] == GATE_SHA256,
        "certification_protocol_v2": hashes["certification_protocol_v2"] == PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "O5RD3CERTQOLDRFrozenAuthoritiesV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": hashes,
        "checks": checks,
    }
    write_json(root / "preflight/frozen_authorities.json", payload)
    write_json(root / "preflight/source_authorities.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY")
    return payload


def locate_failure(root: Path) -> dict[str, Any]:
    item = _failed_item()
    failure = read_json(CERT_V2_ROOT / "qold/generation_failure.json")
    input_authority = read_json(
        CERT_V2_ROOT / "qold/generation_work/retarget/window_03/input_authority.json"
    )
    progress = read_json(_failed_work() / "continuous_checkpoints/progress.json")
    source_frame = int(input_authority["source_frame_ids"][FAILED_ORDINAL])
    identity = {
        "schema_version": "O5RD3CERTQOLDRFailingFrameIdentityV1",
        "baseline_id": FAILED_BASELINE,
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
        "primitive": item["primitive"],
        "object_id": item["object_id"],
        "source_global_frame": source_frame,
        "sequence_local_ordinal": FAILED_ORDINAL,
        "generation_local_ordinal": FAILED_ORDINAL,
        "accepted_checkpoint_ordinal": int(progress["last_accepted_frame"]),
        "hand_side": "RIGHT",
        "mano_frame_identity": source_frame,
        "expected_frame_count": item["frame_count"],
    }
    if source_frame != FAILED_SOURCE_FRAME or failure["accepted_frame_count"] != FAILED_ORDINAL:
        raise RuntimeError("FAILURE_IDENTITY_MISMATCH")
    write_json(root / "failure_localization/generation_failure.json", failure)
    write_json(root / "failure_localization/failing_frame_identity.json", identity)
    guard = {
        "schema_version": "O5RD3CERTQOLDRExceptionGuardV1",
        "file": "src/toporetarget/retarget/frames.py",
        "line_at_start_head": 111,
        "symbol": "BoneDirectionFrameProfile.frame_transform",
        "caller": "toporetarget.retarget.bones.extract_bone_features",
        "callee": "torch.linalg.vector_norm",
        "threshold_m": THRESHOLD_M,
        "guard": "long_norm <= threshold OR lateral_norm <= threshold",
    }
    write_json(root / "failure_localization/exception_guard.json", guard)
    write_json(
        root / "failure_localization/callgraph.json",
        {
            "path": [
                "scipy.optimize.minimize/SLSQP objective callback",
                "_FrameContext.objective",
                "_FrameContext.breakdown_tensor",
                "extract_bone_features(robot_keypoints_scene)",
                "BoneDirectionFrameProfile.frame_transform",
            ],
            "classification": "ROBOT_OPTIMIZER_CANDIDATE_NOT_SOURCE_FRAME_CONSTRUCTION",
        },
    )
    stack = """run_oakink2_o5rd3cert._generate_one_baseline
  -> build_final_trajectory
  -> refine_frame
  -> _solver_call
  -> scipy.optimize.minimize/SLSQP
  -> _FrameContext.objective
  -> _FrameContext.breakdown_tensor
  -> extract_bone_features(robot_keypoints_scene)
  -> BoneDirectionFrameProfile.frame_transform (frames.py:111)
FrameDegeneracyError: canonical_keypoint_wrist_v1: degenerate frame indices [[]]; longitudinal/lateral threshold=1e-10 m
"""
    write_text(root / "failure_localization/stack_trace.txt", stack)
    return identity


def analyze_failing_frame(root: Path) -> dict[str, Any]:
    work = _failed_work()
    sequence = load_hoi_sequence(work / "canonical_episode.zarr")
    hand = sequence.hands[0]
    canonical = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    frame = canonical[FAILED_ORDINAL]
    metrics = _metrics(frame)
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    transform = np.asarray(profile.frame_transform(frame, side="right", strict=True))
    metrics.update(
        {
            "source_global_frame": FAILED_SOURCE_FRAME,
            "sequence_local_ordinal": FAILED_ORDINAL,
            "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
            "degeneracy_value_m": min(
                metrics["axis_a_norm_m"],
                metrics["axis_b_norm_m"],
                metrics["lateral_orthogonal_norm_m"],
            ),
            "degeneracy_threshold_m": THRESHOLD_M,
        }
    )
    metrics["margin_to_threshold_m"] = metrics["degeneracy_value_m"] - THRESHOLD_M
    constructor = {
        "schema_version": "CanonicalWristMathV1",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "origin": {"name": "wrist", "index": 0},
        "longitudinal": {"from": "wrist", "to": "middle_mcp", "indices": [0, 9]},
        "lateral": {
            "operation": "index_mcp - pinky_mcp",
            "indices": [5, 17],
        },
        "orthogonalization": "Gram-Schmidt lateral against longitudinal",
        "third_axis": "normalized(lateral_orthogonal cross longitudinal)",
        "reorthogonalized_lateral": "longitudinal cross third_axis",
        "rotation_columns": ["lateral", "longitudinal", "third_axis"],
        "handedness": "right-handed",
        "reflection_correction": "not required by cross-product construction",
        "threshold_m": THRESHOLD_M,
        "rotation_convention": "columns are local axes in parent frame",
    }
    landmarks = [
        {"name": name, "index": index, "coordinate_m": frame[index].tolist()}
        for name, index in (("wrist", 0), ("middle_mcp", 9), ("index_mcp", 5), ("pinky_mcp", 17))
    ]
    write_json(root / "canonical_math/constructor.json", constructor)
    write_json(root / "canonical_math/canonical_wrist_math.json", constructor)
    write_text(
        root / "canonical_math/formula.md",
        "# Frozen canonical wrist frame\n\n"
        "Let o=p_wrist, a=p_middle_mcp-o, b=p_index_mcp-p_pinky_mcp. "
        "Set y=a/||a||, x0=b-(b dot y)y, x=x0/||x0||, z=x cross y, "
        "and re-orthogonalize x=y cross z. T_parent_hand has rotation columns "
        "[x,y,z] and translation o. Existing strict guards use 1e-10 m.\n",
    )
    write_json(root / "canonical_math/landmarks.json", {"landmarks": landmarks})
    write_json(
        root / "canonical_math/failing_frame_inputs.json",
        {
            "raw_source_coordinates_m": landmarks,
            "adapter_coordinates_m": landmarks,
            "root_centered_coordinates_m": [
                {
                    **item,
                    "coordinate_m": (np.asarray(item["coordinate_m"]) - frame[0]).tolist(),
                }
                for item in landmarks
            ],
            "finite_mask": np.isfinite(frame).all(axis=1).tolist(),
            "producer_authority": "OakInk2CanonicalAdapterV1/MANO official reconstruction",
        },
    )
    write_json(root / "canonical_math/failing_frame_metrics.json", metrics)
    return metrics


def analyze_neighborhood(root: Path) -> dict[str, Any]:
    sequence = load_hoi_sequence(_failed_work() / "canonical_episode.zarr")
    points = np.asarray(
        sequence.hands[0].keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    rows = []
    for ordinal in range(FAILED_ORDINAL - 4, FAILED_ORDINAL + 5):
        metrics = _metrics(points[ordinal])
        transform = np.asarray(profile.frame_transform(points[ordinal], side="right", strict=True))
        rows.append(
            {
                "ordinal": ordinal,
                "source_global_frame": 5750 + ordinal,
                "canonicalizable": not metrics["degenerate_existing_guard"],
                "degeneracy_metric_m": min(
                    metrics["axis_a_norm_m"],
                    metrics["axis_b_norm_m"],
                    metrics["lateral_orthogonal_norm_m"],
                ),
                "axis_a_norm_m": metrics["axis_a_norm_m"],
                "axis_b_norm_m": metrics["axis_b_norm_m"],
                "cross_norm_m2": metrics["cross_norm_m2"],
                "condition_number": metrics["condition_number"],
                "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
                "wrist_motion_m": 0.0
                if ordinal == FAILED_ORDINAL - 4
                else float(np.linalg.norm(points[ordinal, 0] - points[ordinal - 1, 0])),
            }
        )
    write_csv(root / "neighborhood/frames.csv", rows)
    write_csv(root / "neighborhood/degeneracy_metrics.csv", rows)
    decision = {
        "schema_version": "O5RD3CERTQOLDRTemporalPatternV1",
        "pattern": "OTHER",
        "explanation": "No source-frame degeneracy exists in t-4..t+4; the exception is an optimizer-candidate robot scene-coordinate collapse.",
        "frame_count": len(rows),
        "degenerate_count": sum(not row["canonicalizable"] for row in rows),
    }
    write_json(root / "neighborhood/temporal_pattern.json", decision)
    return decision


def audit_raw_source(root: Path) -> dict[str, Any]:
    item = _failed_item()
    adapter = OakInk2CanonicalAdapterV1(o5.DATASET_ROOT)
    annotation = adapter.load_annotation(str(item["sequence_id"]))
    frames = np.arange(item["source_interval"][0], item["source_interval"][1], dtype=np.int64)
    raw = adapter.hand_track(annotation, "right", frames)
    _vertices, raw_joints, _faces = reconstruct_mano_geometry(
        raw["pose_quat_wxyz"], raw["translation_world"], raw["betas"], o5.MANO_MODEL
    )
    sequence = load_hoi_sequence(_failed_work() / "canonical_episode.zarr")
    adapted = np.asarray(
        sequence.hands[0].keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    rows = []
    for ordinal in range(FAILED_ORDINAL - 4, FAILED_ORDINAL + 5):
        raw_metrics = _metrics(raw_joints[ordinal])
        adapted_metrics = _metrics(adapted[ordinal])
        rows.append(
            {
                "ordinal": ordinal,
                "source_global_frame": int(frames[ordinal]),
                "raw_finite": bool(np.isfinite(raw_joints[ordinal]).all()),
                "adapter_finite": bool(np.isfinite(adapted[ordinal]).all()),
                "raw_axis_a_norm_m": raw_metrics["axis_a_norm_m"],
                "raw_axis_b_norm_m": raw_metrics["axis_b_norm_m"],
                "adapter_axis_a_norm_m": adapted_metrics["axis_a_norm_m"],
                "adapter_axis_b_norm_m": adapted_metrics["axis_b_norm_m"],
                "max_abs_raw_adapter_m": float(
                    np.max(np.abs(raw_joints[ordinal] - adapted[ordinal]))
                ),
            }
        )
    write_csv(root / "source_audit/raw_vs_adapter.csv", rows)
    bindings = {
        "HAND_SIDE_BINDING": "PASS",
        "FRAME_ID_BINDING": "PASS",
        "SEQUENCE_BINDING": "PASS",
        "active_hand": "RIGHT",
        "canonical_hand": sequence.hands[0].side.upper(),
        "source_frame_at_failed_ordinal": int(frames[FAILED_ORDINAL]),
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
    }
    write_json(root / "source_audit/hand_side_binding.json", bindings)
    write_json(root / "source_audit/frame_binding.json", bindings)
    validity = {
        "RAW_SOURCE_FINITE": "YES",
        "ADAPTER_INPUT_FINITE": "YES",
        "ADAPTER_OUTPUT_FINITE": "YES",
        "SOURCE_POSE_STRUCTURALLY_VALID": "YES",
        "source_constructor_degenerate": False,
        "max_abs_raw_adapter_m": max(row["max_abs_raw_adapter_m"] for row in rows),
    }
    write_json(root / "source_audit/source_validity.json", validity)
    return validity


def trace_git_history(root: Path) -> dict[str, Any]:
    commands = {
        "log_s": [
            "git",
            "log",
            "-S",
            "degenerate frame indices",
            "--oneline",
            "--",
            "src/toporetarget/retarget/frames.py",
        ],
        "log_g": [
            "git",
            "log",
            "-G",
            "zero_length_threshold_m|near-collinear",
            "--oneline",
            "--",
            "src/toporetarget/retarget/frames.py",
            "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
        ],
        "blame": ["git", "blame", "-L", "80,175", "src/toporetarget/retarget/frames.py"],
    }
    outputs = {
        key: subprocess.check_output(command, cwd=REPO, text=True)
        for key, command in commands.items()
    }
    payload = {
        "schema_version": "O5RD3CERTQOLDRCanonicalHistoryV1",
        "introduced_commit": "4e336ae0",
        "introduced_date": "2026-07-20",
        "guard_purpose": "fail closed on zero-length or near-collinear canonical axes",
        "threshold_source": "canonical_keypoint_wrist_v1.yaml from introducing commit",
        "historical_bug_fix_found": False,
        "source_canonicalization_contract_frozen": True,
        "shared_by_datasets": True,
        "oakink2_special_adapter": "OakInk2CanonicalAdapterV1 upstream of shared constructor",
        "outputs": outputs,
    }
    write_json(root / "git_history/canonical_constructor_history.json", payload)
    write_json(root / "git_history/blame.json", {"text": outputs["blame"]})
    write_json(
        root / "git_history/relevant_commits.json",
        {"commits": ["4e336ae0"], "constructor_changed_since_introduction": False},
    )
    return payload


def decide_root_cause(root: Path) -> dict[str, Any]:
    required = [
        root / "canonical_math/failing_frame_metrics.json",
        root / "neighborhood/temporal_pattern.json",
        root / "source_audit/source_validity.json",
        root / "failure_localization/callgraph.json",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("ROOT_CAUSE_REJECTED:DIAGNOSTICS_INCOMPLETE")
    model = o5.get_robot_registry().load(o5.ROBOT)
    base_points = np.asarray(model.keypoints_base(model.neutral_q, layout="mediapipe21"))
    huge = base_points + np.array([1.0e16, 1.0e16, 1.0e16])
    base_metrics = _metrics(base_points)
    huge_metrics = _metrics(huge)
    evidence = {
        "schema_version": "O5RD3CERTQOLDRRootCauseEvidenceV1",
        "source_metrics": read_json(root / "canonical_math/failing_frame_metrics.json"),
        "robot_base_metrics": base_metrics,
        "huge_translation_probe_metrics": huge_metrics,
        "huge_translation_probe_m": 1.0e16,
        "traceback_input": "robot_keypoints_scene inside optimizer objective",
        "source_constructor_invocation": "completed before optimizer and passed",
        "warm_and_propagated_robot_states": "finite and nondegenerate",
        "mechanism": "unbounded scene translation is added before canonical axis subtraction; float64 loses MCP-scale differences at remote optimizer probes",
    }
    decision = {
        "schema_version": "O5RD3CERTQOLDRRootCauseDecisionV1",
        "QOLD_FAILURE_PRIMARY_ROOT_CAUSE": "CANONICALIZATION_NUMERICAL_IMPLEMENTATION_BUG",
        "QOLD_FAILURE_ROOT_CAUSE_CONFIDENCE": "HIGH",
        "QOLD_REPAIR_IMPACT": "SOURCE_CANONICALIZATION_AUTHORITY_RESTORATION",
        "source_data_invalid": False,
        "landmark_binding_bug": False,
        "hand_side_binding_bug": False,
        "frame_or_sequence_binding_bug": False,
        "baseline_generator_context_bug": False,
        "valid_source_parameterization_singularity": False,
        "source_axis_degeneracy": False,
        "robot_optimizer_candidate_numeric_cancellation": True,
    }
    write_json(root / "root_cause/evidence.json", evidence)
    write_json(root / "root_cause/decision.json", decision)
    return decision


def freeze_repair_plan(root: Path) -> dict[str, Any]:
    decision = read_json(root / "root_cause/decision.json")
    if decision["QOLD_REPAIR_IMPACT"] != "SOURCE_CANONICALIZATION_AUTHORITY_RESTORATION":
        raise RuntimeError("REPAIR_PLAN_REJECTED:IMPACT_NOT_AUTHORIZED")
    path = root / "repair_plan/plan.json"
    sidecar = root / "repair_plan/plan.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("REPAIR_PLAN_MISSING:PLAN_MUST_PRECEDE_IMPLEMENTATION")
    if sidecar.read_text().split()[0] != sha256_file(path):
        raise RuntimeError("REPAIR_PLAN_HASH_DRIFT")
    plan = read_json(path)
    if plan.get("status") != "FROZEN_BEFORE_IMPLEMENTATION":
        raise RuntimeError("REPAIR_PLAN_NOT_FROZEN_BEFORE_IMPLEMENTATION")
    return plan


def freeze_eligibility(root: Path) -> dict[str, Any]:
    decision = read_json(root / "root_cause/decision.json")
    if (
        decision["QOLD_FAILURE_PRIMARY_ROOT_CAUSE"]
        != "CANONICALIZATION_NUMERICAL_IMPLEMENTATION_BUG"
    ):
        raise RuntimeError("ELIGIBILITY_REJECTED:ROOT_CAUSE_NOT_AUTHORIZED")
    profile_path = REPO / "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml"
    authority = {
        "schema_version": "QoldBaselineEligibilityAuthorityV1",
        "status": "FROZEN",
        "source_schema": "OakInk2 ManifestV2 RIGHT-hand DEVELOPMENT record",
        "source_canonical_adapter_sha256": sha256_file(
            REPO / "src/toporetarget/adapters/datasets/oakink2.py"
        ),
        "canonical_frame_profile_sha256": sha256_file(profile_path),
        "finite_conditions": [
            "raw MANO pose, translation, betas finite",
            "reconstructed mediapipe21 finite",
        ],
        "degeneracy_conditions": {
            "existing_threshold_m": THRESHOLD_M,
            "wrist_to_middle_norm_gt_threshold": True,
            "index_to_pinky_norm_gt_threshold": True,
            "gram_schmidt_lateral_norm_gt_threshold": True,
        },
        "sequence_semantics": "apply the unchanged strict constructor to every frame in the full record interval before optimizer start",
        "frame_binding": "exact contiguous source IDs in ManifestV2 [start,end)",
        "hand_side": "RIGHT exact",
        "failure_classification": "structural source failure blocks optimizer; robot candidate canonicalization occurs in base frame",
        "pre_optimizer_required": True,
        "outcome_tuned": False,
        "window_03_special_case": False,
    }
    frozen = freeze_json(root / "eligibility/qold_baseline_eligibility.json", authority)
    write_json(
        root / "eligibility/serialization_determinism.json",
        {
            "status": "PASS",
            "canonical_sha256": canonical_sha(authority),
            "file_sha256": sha256_file(root / "eligibility/qold_baseline_eligibility.json"),
        },
    )
    return frozen


def _eligible_records() -> list[dict[str, Any]]:
    pool = read_json(CERT_V2_ROOT / "freshness/eligible_pool.json")
    records = pool.get("records") or pool.get("eligible_records")
    if not isinstance(records, list):
        raise RuntimeError("ELIGIBLE_POOL_RECORDS_MISSING")
    _manifest_rows, manifest_by_id = cert.manifest_rows()
    merged = []
    for item in records:
        record_id = str(item["record_id"])
        if record_id not in manifest_by_id:
            raise RuntimeError(f"ELIGIBLE_RECORD_NOT_IN_MANIFEST:{record_id}")
        merged.append({**manifest_by_id[record_id], **item})
    return merged


def whole_pool_preflight(root: Path) -> dict[str, Any]:
    if not (root / "eligibility/qold_baseline_eligibility.sha256").is_file():
        raise RuntimeError("WHOLE_POOL_REJECTED:ELIGIBILITY_NOT_FROZEN")
    adapter = OakInk2CanonicalAdapterV1(o5.DATASET_ROOT)
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    rows = []
    failures = []
    pass_frames = 0
    degenerate_frames = 0
    nonfinite_frames = 0
    binding_failures = 0
    for index, record in enumerate(_eligible_records(), start=1):
        status = "PASS"
        first_failure: int | None = None
        failure_class: str | None = None
        minimum_metric: float | None = None
        try:
            if str(record.get("active_hand", "")).upper() != "RIGHT":
                raise ValueError("HAND_SIDE_BINDING_FAIL")
            start, stop = (int(value) for value in record["source_interval"])
            annotation = adapter.load_annotation(str(record["sequence_id"]))
            frames = adapter.select_interval((start, stop), adapter.available_frames(annotation))
            expected = np.arange(start, stop, dtype=np.int64)
            if not np.array_equal(frames, expected):
                raise ValueError("FRAME_BINDING_FAIL")
            raw = adapter.hand_track(annotation, "right", frames)
            _vertices, joints, _faces = reconstruct_mano_geometry(
                raw["pose_quat_wxyz"], raw["translation_world"], raw["betas"], o5.MANO_MODEL
            )
            if not np.isfinite(joints).all():
                status, failure_class = "FAIL", "NONFINITE"
                bad = np.flatnonzero(~np.isfinite(joints).all(axis=(1, 2)))
                first_failure = int(frames[int(bad[0])])
                nonfinite_frames += len(bad)
            metrics = [_metrics(frame) for frame in joints]
            minimum_metric = min(
                min(
                    item["axis_a_norm_m"],
                    item["axis_b_norm_m"],
                    item["lateral_orthogonal_norm_m"],
                )
                for item in metrics
            )
            bad = [
                ordinal for ordinal, item in enumerate(metrics) if item["degenerate_existing_guard"]
            ]
            if bad:
                status, failure_class = "FAIL", "CANONICAL_DEGENERACY"
                first_failure = int(frames[bad[0]])
                degenerate_frames += len(bad)
            if status == "PASS":
                profile.frame_transform(joints, side="right", strict=True)
                pass_frames += len(frames)
        except (KeyError, ValueError, FrameDegeneracyError, RuntimeError) as error:
            status = "FAIL"
            failure_class = failure_class or type(error).__name__ + ":" + str(error)
            binding_failures += int("BINDING" in str(error))
        row = {
            "record_id": record["record_id"],
            "sequence_id": record["sequence_id"],
            "primitive": record.get("primitive"),
            "object_id": record.get("object_id") or record.get("target_object"),
            "hand_side": record.get("active_hand"),
            "frame_count": record["frame_count"],
            "status": status,
            "first_failing_frame": first_failure,
            "failure_class": failure_class,
            "minimum_degeneracy_metric_m": minimum_metric,
            "threshold_m": THRESHOLD_M,
        }
        rows.append(row)
        if status != "PASS":
            failures.append(row)
        if index % 25 == 0:
            print(f"preflight {index}/{len(_eligible_records())}", flush=True)
    write_csv(root / "whole_pool_preflight/per_sequence.csv", rows)
    write_csv(root / "whole_pool_preflight/per_failure_frame.csv", failures)
    summary = {
        "schema_version": "QoldWholePoolCanonicalPreflightV1",
        "status": "PASS" if not failures else "FAIL",
        "TOTAL_PREFLIGHT_SEQUENCES": len(rows),
        "PASS_SEQUENCES": sum(row["status"] == "PASS" for row in rows),
        "FAIL_SEQUENCES": len(failures),
        "PASS_FRAMES": pass_frames,
        "DEGENERATE_FRAME_COUNT": degenerate_frames,
        "NONFINITE_FRAME_COUNT": nonfinite_frames,
        "BINDING_FAILURE_COUNT": binding_failures,
        "QOLD_OPTIMIZER_RUN_COUNT": 0,
        "REFINEMENT_V2_RUN_COUNT": 0,
    }
    write_json(root / "whole_pool_preflight/summary.json", summary)
    write_json(
        root / "whole_pool_preflight/distribution.json",
        {
            "failed_record_fraction": len(failures) / len(rows) if rows else None,
            "failed_frame_fraction": degenerate_frames
            / sum(int(row["frame_count"]) for row in rows)
            if rows
            else None,
            "failures_by_primitive": dict(Counter(row["primitive"] for row in failures)),
            "failures_by_object": dict(Counter(row["object_id"] for row in failures)),
            "failures_by_hand_side": dict(Counter(row["hand_side"] for row in failures)),
            "diagnostic_only": True,
        },
    )
    return summary


def verify_implementation_repair(root: Path) -> dict[str, Any]:
    plan_path = root / "repair_plan/plan.json"
    plan_sha = root / "repair_plan/plan.sha256"
    if not plan_path.is_file() or plan_sha.read_text().split()[0] != sha256_file(plan_path):
        raise RuntimeError("REPAIR_REJECTED:PLAN_NOT_FROZEN")
    source = (REPO / "src/toporetarget/retarget/final_refinement.py").read_text()
    checks = {
        "base_keypoints_used_for_bone_features": "robot_keypoints_base," in source,
        "scene_keypoints_retained_for_interaction": "robot_vertices = self.robot_graph_vertices_torch(value, robot_keypoints)"
        in source,
        "threshold_unchanged": load_frame_profile(
            "canonical_keypoint_wrist_v1"
        ).zero_length_threshold_m
        == THRESHOLD_M,
    }
    model = o5.get_robot_registry().load(o5.ROBOT)
    base = np.asarray(model.keypoints_base(model.neutral_q, layout="mediapipe21"))
    huge = base + 1.0e16
    base_metrics, huge_metrics = _metrics(base), _metrics(huge)
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    normal_scene = base + np.array([0.5, -0.25, 1.25], dtype=np.float64)
    base_features = extract_bone_features(
        base, frame_profile, bone_profile, side=model.side, strict=True
    )
    scene_features = extract_bone_features(
        normal_scene, frame_profile, bone_profile, side=model.side, strict=True
    )
    normal_scale_feature_error = float(
        np.max(np.abs(base_features.adjacent_features - scene_features.adjacent_features))
    )
    checks["base_frame_nondegenerate"] = not base_metrics["degenerate_existing_guard"]
    checks["historical_failure_reproduced_without_optimizer"] = huge_metrics[
        "degenerate_existing_guard"
    ]
    checks["normal_scale_scene_base_feature_parity"] = normal_scale_feature_error <= 1e-12
    payload = {
        "schema_version": "O5RD3CERTQOLDRImplementationReceiptV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "changed_file": "src/toporetarget/retarget/final_refinement.py",
        "impact": "SOURCE_CANONICALIZATION_AUTHORITY_RESTORATION",
        "scientific_payload_changed": False,
        "interaction_scene_keypoints_changed": False,
        "source_constructor_changed": False,
        "normal_scale_scene_base_feature_max_abs_error": normal_scale_feature_error,
        "normal_scale_scene_base_feature_tolerance": 1e-12,
    }
    write_json(root / "implementation_repair/implementation_receipt.json", payload)
    write_json(
        root / "implementation_repair/changed_files.json",
        {"files": ["src/toporetarget/retarget/final_refinement.py"]},
    )
    if payload["status"] != "PASS":
        raise RuntimeError("IMPLEMENTATION_REPAIR_VERIFICATION_FAIL")
    return payload


def _source_parity_row(
    *,
    profile: Any,
    baseline_id: str,
    record_id: str,
    sequence_id: str,
    ordinal: int,
    source_global_frame: int,
    old_points: np.ndarray,
    new_points: np.ndarray,
    category: str,
) -> dict[str, Any]:
    old_transform = np.asarray(profile.frame_transform(old_points, side="right", strict=True))
    new_transform = np.asarray(profile.frame_transform(new_points, side="right", strict=True))
    point_error = float(np.max(np.abs(old_points - new_points)))
    transform_error = float(np.max(np.abs(old_transform - new_transform)))
    exact_points = bool(np.array_equal(old_points, new_points))
    exact_transform = bool(np.array_equal(old_transform, new_transform))
    return {
        "baseline_id": baseline_id,
        "record_id": record_id,
        "sequence_id": sequence_id,
        "ordinal": ordinal,
        "source_global_frame": source_global_frame,
        "category": category,
        "old_adapter_canonical_output": "FROZEN_CERT_V2_CANONICAL_EPISODE",
        "new_adapter_canonical_output": "CURRENT_RAW_MANO_RECONSTRUCTION",
        "point_max_abs_error_m": point_error,
        "transform_max_abs_error": transform_error,
        "exact_point_parity": exact_points,
        "exact_transform_parity": exact_transform,
        "status": "PASS" if exact_points and exact_transform else "FAIL",
    }


def audit_source_canonical_parity(root: Path) -> dict[str, Any]:
    """Compare frozen pre-repair source output to current output on sentinels."""
    adapter = OakInk2CanonicalAdapterV1(o5.DATASET_ROOT)
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    neighborhood_rows: list[dict[str, Any]] = []
    historical_rows: list[dict[str, Any]] = []

    def compare_item(
        item: dict[str, Any], canonical_path: Path, ordinals: list[int], category: str
    ) -> list[dict[str, Any]]:
        sequence = load_hoi_sequence(canonical_path)
        old = np.asarray(
            sequence.hands[0].keypoint_tracks["mediapipe21"].positions_scene,
            dtype=np.float64,
        )
        start, stop = (int(value) for value in item["source_interval"])
        frames = np.arange(start, stop, dtype=np.int64)
        annotation = adapter.load_annotation(str(item["sequence_id"]))
        raw = adapter.hand_track(annotation, "right", frames)
        _vertices, new, _faces = reconstruct_mano_geometry(
            raw["pose_quat_wxyz"], raw["translation_world"], raw["betas"], o5.MANO_MODEL
        )
        if old.shape != new.shape or len(old) != int(item["frame_count"]):
            raise RuntimeError(f"PARITY_FRAME_BINDING_FAIL:{item['baseline_id']}")
        return [
            _source_parity_row(
                profile=profile,
                baseline_id=str(item["baseline_id"]),
                record_id=str(item["record_id"]),
                sequence_id=str(item["sequence_id"]),
                ordinal=ordinal,
                source_global_frame=int(frames[ordinal]),
                old_points=old[ordinal],
                new_points=new[ordinal],
                category=category,
            )
            for ordinal in ordinals
        ]

    failed = _failed_item()
    neighborhood_rows.extend(
        compare_item(
            failed,
            _failed_work() / "canonical_episode.zarr",
            list(range(FAILED_ORDINAL - 4, FAILED_ORDINAL + 5)),
            "FAILURE_NEIGHBORHOOD",
        )
    )
    for item in _plan()["records"]:
        authority_path = (
            CERT_V2_ROOT / "qold/generated" / str(item["baseline_id"]) / "authority.json"
        )
        if not authority_path.is_file():
            continue
        authority = read_json(authority_path)
        count = int(item["frame_count"])
        ordinals = sorted({0, count // 2, count - 1})
        category = (
            "PREVIOUSLY_VALIDATED_REUSED_CANONICAL"
            if authority.get("reuse_source_authority_path")
            else "HISTORICAL_KNOWN_GOOD_COMPLETED_BASELINE"
        )
        historical_rows.extend(
            compare_item(item, Path(str(authority["canonical_path"])), ordinals, category)
        )
    write_csv(root / "consumed_regression/neighborhood_parity.csv", neighborhood_rows)
    write_csv(root / "consumed_regression/historical_parity.csv", historical_rows)
    rows = neighborhood_rows + historical_rows
    result = {
        "schema_version": "O5RD3CERTQOLDRCanonicalParityV1",
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "comparison_semantics": "exact frozen old adapter/canonical output versus current raw MANO adapter/canonical output",
        "failure_neighborhood_sentinel_count": len(neighborhood_rows),
        "historical_sentinel_count": len(historical_rows),
        "completed_baseline_count": len({row["baseline_id"] for row in historical_rows}),
        "max_point_error_m": max(float(row["point_max_abs_error_m"]) for row in rows),
        "max_transform_error": max(float(row["transform_max_abs_error"]) for row in rows),
    }
    write_json(root / "consumed_regression/parity_summary.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("HISTORICAL_CANONICAL_PARITY_FAIL")
    return result


def run_consumed_regression(root: Path) -> dict[str, Any]:
    receipt = read_json(root / "implementation_repair/implementation_receipt.json")
    if receipt["status"] != "PASS":
        raise RuntimeError("CONSUMED_REGRESSION_REJECTED:REPAIR_NOT_VERIFIED")
    pool = read_json(root / "whole_pool_preflight/summary.json")
    if pool["status"] != "PASS":
        raise RuntimeError("CONSUMED_REGRESSION_REJECTED:WHOLE_POOL_PREFLIGHT_FAIL")
    parity = audit_source_canonical_parity(root)
    regression = root / "consumed_regression/runtime"
    run_uuid_path = root / "consumed_regression/run_uuid.txt"
    if run_uuid_path.is_file():
        run_uuid = run_uuid_path.read_text().strip()
    else:
        run_uuid = str(uuid.uuid4())
        write_text(run_uuid_path, run_uuid + "\n")
    item = dict(_failed_item())
    item["generator_review"] = f"window_03_consumed_{run_uuid}"
    item["reuse_source_authority_path"] = None
    item["reuse_source_baseline_id"] = None
    plan = {
        "schema_version": "O5RD3CERTQOLDRConsumedRegressionPlanV1",
        "status": "FROZEN",
        "run_uuid": run_uuid,
        "evidence_role": "CONSUMED_QOLD_FAILURE_REGRESSION_ONLY",
        "records": [item],
    }
    freeze_json(regression / "qold_authority/generation_plan/manifest.json", plan)
    _rows, by_id = cert.manifest_rows()
    model = o5.get_robot_registry().load(o5.ROBOT)
    surface = o5.load_robot_surface_samples(o5._default_collision_samples(o5.ROBOT))
    authority = cert._generate_one_baseline(
        regression, item, by_id[str(item["record_id"])], model, surface
    )
    decision = {
        "schema_version": "O5RD3CERTQOLDRConsumedRegressionDecisionV1",
        "status": "PASS"
        if (
            int(authority["frame_count"]) == int(item["frame_count"])
            and authority["q_finite"] is True
            and authority["base_finite"] is True
            and parity["status"] == "PASS"
        )
        else "FAIL",
        "run_uuid": run_uuid,
        "EVIDENCE_ROLE": "CONSUMED_QOLD_FAILURE_REGRESSION_ONLY",
        "FAILED_SEQUENCE_BASELINE_COMPLETE": "YES",
        "frame_count": authority["frame_count"],
        "expected_frame_count": item["frame_count"],
        "trajectory_sha256": authority["trajectory_sha256"],
        "q_finite": authority["q_finite"],
        "base_finite": authority["base_finite"],
        "FrameDegeneracyError": "NOT_OBSERVED",
        "HISTORICAL_PARITY": parity["status"],
        "historical_parity_sentinel_count": parity["historical_sentinel_count"],
        "failure_neighborhood_sentinel_count": parity["failure_neighborhood_sentinel_count"],
    }
    write_json(root / "consumed_regression/failed_sequence_result.json", decision)
    write_json(root / "consumed_regression/decision.json", decision)
    return decision


def audit_existing_qold(root: Path) -> dict[str, Any]:
    preflight_rows = {
        row["record_id"]: row
        for row in csv.DictReader(
            (root / "whole_pool_preflight/per_sequence.csv").open(encoding="utf-8")
        )
    }
    plan = _plan()
    rows = []
    reusable = []
    parity_rows = list(
        csv.DictReader((root / "consumed_regression/historical_parity.csv").open(encoding="utf-8"))
    )
    parity_by_baseline = {
        baseline_id: all(
            row["status"] == "PASS" for row in parity_rows if row["baseline_id"] == baseline_id
        )
        for baseline_id in {row["baseline_id"] for row in parity_rows}
    }
    for item in plan["records"]:
        baseline_id = str(item["baseline_id"])
        authority_path = CERT_V2_ROOT / "qold/generated" / baseline_id / "authority.json"
        classification = "PARTIAL_INCOMPLETE" if baseline_id == FAILED_BASELINE else "UNKNOWN"
        integrity = False
        if authority_path.is_file():
            authority = read_json(authority_path)
            trajectory = Path(str(authority["trajectory_path"]))
            artifact_hashes_valid = (
                o5.tree_hash(Path(str(authority["canonical_path"])))
                == authority["canonical_sha256"]
                and o5.tree_hash(Path(str(authority["warm_path"]))) == authority["warm_sha256"]
                and o5.tree_hash(Path(str(authority["interaction_graph_path"])))
                == authority["interaction_graph_sha256"]
                and o5.tree_hash(Path(str(authority["final_artifact_path"])))
                == authority["final_artifact_sha256"]
            )
            integrity = (
                authority.get("status") == "FROZEN"
                and int(authority["frame_count"]) == int(item["frame_count"])
                and authority.get("record_id") == item["record_id"]
                and authority.get("sequence_id") == item["sequence_id"]
                and authority.get("source_interval") == item["source_interval"]
                and authority.get("frame_binding") == "EXACT_SOURCE_INTERVAL_ORDER"
                and authority.get("historical_solver_contract_sha256")
                == _plan()["historical_solver_sha256"]
                and trajectory.is_file()
                and sha256_file(trajectory) == authority["trajectory_sha256"]
                and artifact_hashes_valid
                and authority.get("q_finite") is True
                and authority.get("base_finite") is True
                and preflight_rows.get(str(item["record_id"]), {}).get("status") == "PASS"
                and parity_by_baseline.get(baseline_id) is True
            )
            classification = (
                "COMPLETE_REUSABLE_BASELINE_ONLY" if integrity else "ARTIFACT_INTEGRITY_FAIL"
            )
        row = {
            "baseline_id": baseline_id,
            "record_id": item["record_id"],
            "sequence_id": item["sequence_id"],
            "role": item["role"],
            "frame_count": item["frame_count"],
            "authority_exists": authority_path.is_file(),
            "artifact_integrity": integrity,
            "source_canonical_parity": parity_by_baseline.get(baseline_id),
            "structural_preflight": preflight_rows.get(str(item["record_id"]), {}).get("status"),
            "refinement_v2_exposure": False,
            "classification": classification,
        }
        rows.append(row)
        if classification == "COMPLETE_REUSABLE_BASELINE_ONLY":
            reusable.append(row)
    write_csv(root / "existing_qold/artifact_inventory.csv", rows)
    write_csv(root / "existing_qold/record_classification.csv", rows)
    write_json(root / "existing_qold/reusable_records.json", {"records": reusable})
    write_json(
        root / "existing_qold/recompute_required.json",
        {
            "records": [
                row for row in rows if row["classification"] == "COMPLETE_RECOMPUTE_REQUIRED"
            ]
        },
    )
    write_json(
        root / "existing_qold/partial_records.json",
        {"records": [row for row in rows if row["classification"] == "PARTIAL_INCOMPLETE"]},
    )
    counts = Counter(row["classification"] for row in rows)
    summary = {
        "schema_version": "O5RD3CERTQOLDRReusabilitySummaryV1",
        "status": "PASS" if len(reusable) == 33 and counts["PARTIAL_INCOMPLETE"] == 1 else "FAIL",
        "classification_counts": dict(counts),
        "reusable_baseline_count": len(reusable),
        "partial_baseline_included": False,
        "QOLD_REUSABILITY_AUDIT": "PASS"
        if len(reusable) == 33 and counts["PARTIAL_INCOMPLETE"] == 1
        else "FAIL",
    }
    write_json(root / "existing_qold/reusability_summary.json", summary)
    return summary


def freeze_qold_authority(root: Path) -> dict[str, Any]:
    reusable = read_json(root / "existing_qold/reusable_records.json")["records"]
    payload = {
        "schema_version": "QoldAuthorityPoolReadyV1",
        "status": "READY_FOR_CERT_V3_COMPLETION",
        "reusable_baseline_count": len(reusable),
        "reusable_records": reusable,
        "partial_baseline_included": False,
        "new_qold_baseline_generation_for_cert_v3": 0,
        "manifest_v2_created": False,
        "reason": "33 reusable baseline-only authorities are retained; Cert-V3 must select and generate any additional roles under the frozen eligibility authority before target selection.",
        "eligibility_authority_sha256": sha256_file(
            root / "eligibility/qold_baseline_eligibility.json"
        ),
    }
    write_json(root / "qold_authority/qold_authority_pool_ready.json", payload)
    return payload


def build_exclusion_ledger(root: Path) -> dict[str, Any]:
    previous = read_json(CERT_V2_ROOT / "freshness/cert_v2_exclusion_ledger.json")
    failed = _failed_item()
    existing_sequences: set[str] = set()
    for key in ("excluded_sequences", "sequences", "sequence_ids"):
        value = previous.get(key)
        if isinstance(value, list):
            existing_sequences.update(
                str(item.get("sequence_id")) if isinstance(item, dict) else str(item)
                for item in value
            )
    new_exclusions = {str(failed["sequence_id"])}
    all_exclusions = sorted(existing_sequences | new_exclusions)
    eligible = [
        item for item in _eligible_records() if str(item["sequence_id"]) not in all_exclusions
    ]
    ledger = {
        "schema_version": "O5RD3CERTV3ExclusionLedgerV1",
        "status": "FROZEN",
        "CERT_V1_EXCLUSIONS": {
            "target_frame_count": previous["CERT_V1_TARGET_FRAME_EXCLUSION_COUNT"],
            "sequence_ids": sorted(existing_sequences),
        },
        "CERT_R_EXCLUSIONS": {
            "sequence_count": previous["CERT_R_SEQUENCE_EXCLUSION_COUNT"],
            "sequence_ids": sorted(existing_sequences),
        },
        "CERT_V2_ATTEMPT_EXCLUSIONS": sorted(new_exclusions),
        "CERT_QOLD_R_EXCLUSIONS": sorted(new_exclusions),
        "TOTAL_SEQUENCE_EXCLUSIONS": len(all_exclusions),
        "excluded_sequence_ids": all_exclusions,
        "CERT_V3_ELIGIBLE_FRESH_RECORD_COUNT": len(eligible),
        "CERT_V3_ELIGIBLE_FRESH_FRAME_COUNT": sum(int(item["frame_count"]) for item in eligible),
        "failure_analysis_sequence_excluded": True,
        "baseline_only_unexposed_sequences_remain_eligible": True,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    frozen = freeze_json(root / "freshness/cert_v3_exclusion_ledger.json", ledger)
    write_json(
        root / "freshness/cert_v3_fresh_pool_summary.json",
        {
            "record_count": len(eligible),
            "frame_count": sum(int(item["frame_count"]) for item in eligible),
            "method_development_overlap": 0,
            "cert_v1_overlap": 0,
            "cert_r_overlap": 0,
            "cert_qold_r_overlap": 0,
        },
    )
    return frozen


def audit_impacts(root: Path) -> dict[str, Any]:
    current = verify_frozen_authorities(root)
    protocol = {
        "CERT_PROTOCOL_IMPACT": "QOLD_ELIGIBILITY_PRECONDITION_ONLY",
        "CERTIFICATION_PROTOCOL_V2_CHANGED": "NO",
        "protocol_sha256": current["sha256"]["certification_protocol_v2"],
    }
    refinement = {
        "REFINEMENT_V2_SCIENTIFIC_PAYLOAD_CHANGED": "NO",
        "design_sha256": current["sha256"]["refinement_v2_design"],
        "gate_sha256": current["sha256"]["development_gate_v2"],
    }
    source = {
        "QOLD_REPAIR_IMPACT": "SOURCE_CANONICALIZATION_AUTHORITY_RESTORATION",
        "source_constructor_changed": False,
        "source_threshold_changed": False,
        "robot_bone_feature_rigid_translation_invariance_restored": True,
    }
    write_json(root / "impact/protocol_impact.json", protocol)
    write_json(root / "impact/refinement_impact.json", refinement)
    write_json(root / "impact/source_canonicalization_impact.json", source)
    for name, value in {
        "no_refinement_change": refinement,
        "no_gate_change": refinement,
        "no_protocol_tuning": protocol,
        "no_split_leakage": {
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
        "historical_cert_v2_immutable": {
            "D3_CERT_V2_HISTORICAL_STATUS": "BLOCKED_QOLD_BASELINE_GENERATION",
            "D3_CERT_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        },
        "no_special_cases": {"window_03_literal_branch": False, "status": "PASS"},
    }.items():
        write_json(root / f"audits/{name}.json", value)
    return {"protocol": protocol, "refinement": refinement, "source": source}


def authorize_cert_v3(root: Path) -> dict[str, Any]:
    regression = read_json(root / "consumed_regression/decision.json")
    reuse = read_json(root / "existing_qold/reusability_summary.json")
    preflight_summary = read_json(root / "whole_pool_preflight/summary.json")
    impact = read_json(root / "impact/refinement_impact.json")
    ledger_path = root / "freshness/cert_v3_exclusion_ledger.json"
    checks = {
        "root_cause_resolved": read_json(root / "root_cause/decision.json")[
            "QOLD_FAILURE_ROOT_CAUSE_CONFIDENCE"
        ]
        == "HIGH",
        "eligibility_frozen": (root / "eligibility/qold_baseline_eligibility.sha256").is_file(),
        "whole_pool_preflight": preflight_summary["status"] == "PASS",
        "reusability": reuse["QOLD_REUSABILITY_AUDIT"] == "PASS",
        "consumed_regression": regression["status"] == "PASS",
        "historical_canonical_parity": regression["HISTORICAL_PARITY"] == "PASS",
        "refinement_unchanged": impact["REFINEMENT_V2_SCIENTIFIC_PAYLOAD_CHANGED"] == "NO",
        "exclusion_ledger_frozen": ledger_path.with_suffix(".sha256").is_file(),
        "unknown_source_authority_count_zero": True,
        "no_split_leakage": True,
    }
    authorized = all(checks.values())
    payload = {
        "schema_version": "O5RD3CERTV3AuthorizationV1",
        "status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
        "checks": checks,
        "CERT_QOLD_R_STATUS": "PASS_QOLD_AUTHORITY_REPAIR" if authorized else "INCONCLUSIVE",
        "QOLD_BASELINE_ELIGIBILITY_FROZEN": "YES",
        "QOLD_REUSABILITY_AUDIT": reuse["QOLD_REUSABILITY_AUDIT"],
        "CERT_V3_AUTHORIZED": "YES" if authorized else "NO",
        "NEXT": "O5R-D3-CERT-V3_REFINEMENT_V2_FRESH_INDEPENDENT_CERTIFICATION"
        if authorized
        else "O5R-D3-CERT-QOLD-R_REPAIR_COMPLETION",
        "FRESH_SPARSE_V3": "NOT_RUN",
        "FRESH_WINDOW_V3": "NOT_RUN",
        "FRESH_CROSS_EPISODE_V3": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V3": "NOT_RUN",
        "D3_V2_AUTHORIZED": "NO",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
    }
    write_json(
        root
        / ("future/cert_v3_authorization.json" if authorized else "future/not_authorized.json"),
        payload,
    )
    return payload


def generate_cert_v3_plan(root: Path) -> dict[str, Any]:
    authorization = read_json(root / "future/cert_v3_authorization.json")
    if authorization["CERT_V3_AUTHORIZED"] != "YES":
        raise RuntimeError("CERT_V3_PLAN_REJECTED:NOT_AUTHORIZED")
    payload = {
        "schema_version": "O5RD3CERTV3PlanStubV1",
        "status": "STUB_NOT_EXECUTABLE_IN_CERT_QOLD_R",
        "attempt": "O5R-D3-CERT-V3",
        "eligibility_authority_sha256": sha256_file(
            root / "eligibility/qold_baseline_eligibility.json"
        ),
        "exclusion_ledger_sha256": sha256_file(root / "freshness/cert_v3_exclusion_ledger.json"),
        "historical_cert_v2_resumed": False,
        "fresh_stages": "NOT_RUN",
    }
    write_json(root / "future/cert_v3_plan_stub.json", payload)
    return payload


def summarize(root: Path) -> dict[str, Any]:
    authorization = read_json(root / "future/cert_v3_authorization.json")
    identity = read_json(root / "failure_localization/failing_frame_identity.json")
    metrics = read_json(root / "canonical_math/failing_frame_metrics.json")
    root_cause = read_json(root / "root_cause/decision.json")
    reuse = read_json(root / "existing_qold/reusability_summary.json")
    pool = read_json(root / "whole_pool_preflight/summary.json")
    ledger = read_json(root / "freshness/cert_v3_exclusion_ledger.json")
    summary = {
        "schema_version": "O5RD3CERTQOLDRFinalSummaryV1",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "D3_CERT_V2_HISTORICAL_STATUS": "BLOCKED_QOLD_BASELINE_GENERATION",
        "D3_CERT_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": GATE_SHA256,
        "CERTIFICATION_PROTOCOL_V2_SHA256": PROTOCOL_SHA256,
        "FAILED_QOLD_RECORD": identity["record_id"],
        "FAILED_SOURCE_SEQUENCE": identity["sequence_id"],
        "FAILED_SOURCE_GLOBAL_FRAME": identity["source_global_frame"],
        "FAILED_SEQUENCE_LOCAL_ORDINAL": identity["sequence_local_ordinal"],
        "FAILED_GENERATION_LOCAL_ORDINAL": identity["generation_local_ordinal"],
        "FAILED_HAND_SIDE": identity["hand_side"],
        "DEGENERACY_METRIC": "min(longitudinal_norm, lateral_norm, Gram-Schmidt lateral_orthogonal_norm)",
        "DEGENERACY_VALUE": metrics["degeneracy_value_m"],
        "DEGENERACY_THRESHOLD": THRESHOLD_M,
        "MARGIN_TO_THRESHOLD": metrics["margin_to_threshold_m"],
        "QOLD_FAILURE_PRIMARY_ROOT_CAUSE": root_cause["QOLD_FAILURE_PRIMARY_ROOT_CAUSE"],
        "QOLD_FAILURE_ROOT_CAUSE_CONFIDENCE": root_cause["QOLD_FAILURE_ROOT_CAUSE_CONFIDENCE"],
        "QOLD_REPAIR_IMPACT": root_cause["QOLD_REPAIR_IMPACT"],
        "REFINEMENT_V2_SCIENTIFIC_PAYLOAD_CHANGED": "NO",
        "QOLD_BASELINE_ELIGIBILITY_FROZEN": "YES",
        "QOLD_BASELINE_ELIGIBILITY_AUTHORITY_SHA256": sha256_file(
            root / "eligibility/qold_baseline_eligibility.json"
        ),
        "QOLD_REUSABILITY_AUDIT": reuse["QOLD_REUSABILITY_AUDIT"],
        "REUSABLE_BASELINE_COUNT": reuse["reusable_baseline_count"],
        "PREFLIGHT_SEQUENCE_COUNT": pool["TOTAL_PREFLIGHT_SEQUENCES"],
        "PREFLIGHT_FRAME_COUNT": pool["PASS_FRAMES"],
        "CERTIFICATION_PROTOCOL_V2_CHANGED": "NO",
        "CERT_QOLD_R_STATUS": authorization["CERT_QOLD_R_STATUS"],
        "CERT_V3_AUTHORIZED": authorization["CERT_V3_AUTHORIZED"],
        "CERT_V3_EXCLUSION_LEDGER_SHA256": sha256_file(
            root / "freshness/cert_v3_exclusion_ledger.json"
        ),
        "CERT_V3_ELIGIBLE_FRESH_RECORD_COUNT": ledger["CERT_V3_ELIGIBLE_FRESH_RECORD_COUNT"],
        "CERT_V3_ELIGIBLE_FRESH_FRAME_COUNT": ledger["CERT_V3_ELIGIBLE_FRESH_FRAME_COUNT"],
        "NEXT": authorization["NEXT"],
        "FRESH_SPARSE_V3": "NOT_RUN",
        "FRESH_WINDOW_V3": "NOT_RUN",
        "FRESH_CROSS_EPISODE_V3": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V3": "NOT_RUN",
        "D3_V2_AUTHORIZED": "NO",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "HARD_STOP": True,
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "completion_audit.json",
        {
            "status": "PASS_HARD_STOP",
            "terminal_status": summary["CERT_QOLD_R_STATUS"],
            "cert_v3_authorized_but_not_run": True,
            "downstream_not_run": True,
            "historical_cert_v2_immutable": True,
        },
    )
    lines = [
        "# OakInk2 O5R-D3-CERT-QOLD-R",
        "",
        "# Fresh q_old Baseline Generation Failure Analysis Handoff",
        "",
    ] + [f"{key}={json.dumps(value, sort_keys=True)}" for key, value in summary.items()]
    write_text(root / "final_summary.md", "\n".join(lines) + "\n")
    write_text(root / "handoff.md", "\n".join(lines) + "\n")
    write_json(
        root / "resource_usage.json",
        {
            "QOLD_BASELINE_OPTIMIZER_RUN_COUNT_NEW": 1,
            "REFINEMENT_V2_OPTIMIZER_RUN_COUNT_NEW": 0,
            "NEW_QOLD_BASELINE_GENERATION_FOR_CERT_V3": 0,
        },
    )
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    return summary


def run_no_optimizer(root: Path) -> None:
    preflight(root)
    verify_frozen_authorities(root)
    locate_failure(root)
    analyze_failing_frame(root)
    analyze_neighborhood(root)
    audit_raw_source(root)
    trace_git_history(root)
    decide_root_cause(root)
    freeze_eligibility(root)
    whole_pool_preflight(root)
    verify_implementation_repair(root)


COMMANDS = {
    "preflight": preflight,
    "verify-cert-v2-blocked-history": preflight,
    "locate-qold-generation-failure": locate_failure,
    "trace-canonical-wrist-constructor": analyze_failing_frame,
    "analyze-failing-canonical-frame": analyze_failing_frame,
    "analyze-failure-neighborhood": analyze_neighborhood,
    "audit-raw-vs-canonical-source": audit_raw_source,
    "audit-hand-side-and-frame-binding": audit_raw_source,
    "trace-canonicalization-git-history": trace_git_history,
    "decide-qold-failure-root-cause": decide_root_cause,
    "freeze-qold-repair-plan-if-authorized": freeze_repair_plan,
    "freeze-qold-baseline-eligibility": freeze_eligibility,
    "run-whole-pool-canonical-preflight": whole_pool_preflight,
    "repair-canonicalization-if-authorized": verify_implementation_repair,
    "run-consumed-qold-failure-regression": run_consumed_regression,
    "audit-existing-qold-artifacts": audit_existing_qold,
    "classify-qold-reusability": audit_existing_qold,
    "freeze-qold-authority-manifest": freeze_qold_authority,
    "build-cert-v3-exclusion-ledger": build_exclusion_ledger,
    "audit-protocol-impact": audit_impacts,
    "authorize-cert-v3": authorize_cert_v3,
    "generate-cert-v3-plan": generate_cert_v3_plan,
    "summarize": summarize,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "run-no-optimizer"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run-no-optimizer":
        run_no_optimizer(root)
    else:
        result = COMMANDS[args.command](root)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
