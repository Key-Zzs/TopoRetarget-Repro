#!/usr/bin/env python3
"""Materialize immutable runtime targets from HumanObjectCouplingContactProfileV1.

This is intentionally a reader of the existing Stage16 closeout evidence.  It
never reruns MANO fitting, rewrites an historical trace, or derives a target
from PPO outcome.  The resulting compact NPZ is the only source-profile input
the runtime needs to cache on GPU at episode construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.source_profile_tracking import (  # noqa: E402
    SOURCE_PROFILE_CLIPS,
    SOURCE_PROFILE_TRACKING_V1,
    Stage16SourceProfileTrackingV1,
)

DEFAULT_PROFILE_ROOT = (
    REPO_ROOT / ".local/reports/stage16_170650_closure_and_human_object_profile/profile"
)
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 321 or [int(row["runtime_frame"]) for row in rows] != list(range(321)):
        raise ValueError(f"SOURCE_PROFILE_TARGET_FRAME_CONTRACT_INVALID:{path}")
    return rows


def _number(value: str) -> float:
    if value == "NOT_IDENTIFIABLE":
        return float("nan")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("SOURCE_PROFILE_TARGET_NONFINITE_TEXT_FIELD")
    return result


def _object_span(path: Path) -> float:
    with np.load(path, allow_pickle=False) as archive:
        points = np.asarray(archive["object_axis_points_world_ref"], dtype=np.float64)
    if points.shape != (321, 6, 3) or not np.isfinite(points).all():
        raise ValueError(f"SOURCE_PROFILE_TARGET_OBJECT_AXIS_INVALID:{path}")
    spans = np.linalg.norm(points[:, :, None] - points[:, None, :], axis=-1).max(axis=(1, 2))
    # The object axis points are rigid by construction; V2 interpolation leaves
    # only small floating-point variation.  A median gives one fixed clip
    # characteristic length rather than a runtime-varying scale.
    span = float(np.median(spans))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("SOURCE_PROFILE_TARGET_OBJECT_SPAN_INVALID")
    if not np.allclose(spans, span, rtol=1.0e-3, atol=1.0e-6):
        raise ValueError("SOURCE_PROFILE_TARGET_OBJECT_SPAN_NOT_RIGID")
    return span


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    contract_output = args.contract_output.resolve()
    if output.exists() or contract_output.exists():
        raise FileExistsError("SOURCE_PROFILE_TARGET_OUTPUT_EXISTS")
    contract = Stage16SourceProfileTrackingV1()
    source_root = args.source_profile_root.resolve()
    reference_root = args.reference_root.resolve()
    contact_values: list[np.ndarray] = []
    geometry_values: list[np.ndarray] = []
    geometry_valid: list[np.ndarray] = []
    linear_values: list[np.ndarray] = []
    angular_values: list[np.ndarray] = []
    spans: list[float] = []
    inputs: dict[str, dict[str, object]] = {}
    for clip in SOURCE_PROFILE_CLIPS:
        short = clip.removeprefix("hocap_")
        contact_path = source_root / short / "contact_regions.csv"
        topology_path = source_root / short / "topology.csv"
        coupling_path = source_root / short / "coupling.csv"
        reference_path = reference_root / f"{clip}.reference_kinematics_v2.npz"
        contact_rows = _rows(contact_path)
        topology_rows = _rows(topology_path)
        coupling_rows = _rows(coupling_path)
        activity = np.asarray(
            [[row[f"{finger}_contact"] == "True" for finger in FINGERS] for row in contact_rows],
            dtype=np.float32,
        )
        raw_geometry = np.asarray(
            [
                [
                    _number(row["contact_centroid_object_x_m"]),
                    _number(row["contact_centroid_object_y_m"]),
                    _number(row["contact_centroid_object_z_m"]),
                ]
                for row in topology_rows
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(raw_geometry).all(axis=1)
        linear = np.asarray(
            [_number(row["linear_coupling_ratio"]) for row in coupling_rows], dtype=np.float64
        )
        angular = np.asarray(
            [_number(row["angular_coupling_ratio"]) for row in coupling_rows], dtype=np.float64
        )
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            raise ValueError(f"SOURCE_PROFILE_TARGET_COUPLING_INVALID:{clip}")
        span = _object_span(reference_path)
        contact_values.append(activity)
        geometry_values.append(
            np.where(valid[:, None], raw_geometry / span, 0.0).astype(np.float32)
        )
        geometry_valid.append(valid)
        linear_values.append(linear)
        angular_values.append(angular)
        spans.append(span)
        inputs[clip] = {
            "contact_regions": {"path": str(contact_path), "sha256": _sha256(contact_path)},
            "topology": {"path": str(topology_path), "sha256": _sha256(topology_path)},
            "coupling": {"path": str(coupling_path), "sha256": _sha256(coupling_path)},
            "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
            "object_characteristic_length_m": span,
            "geometry_valid_frames": int(valid.sum()),
        }
    # These two normalizers are computed once over every source frame from both
    # clips.  They are global source-distribution scales, not outcome-derived
    # weights and not a per-object correction.
    linear_scale = float(np.quantile(np.concatenate(linear_values), 0.95))
    angular_scale = float(np.quantile(np.concatenate(angular_values), 0.95))
    if linear_scale <= 0.0 or angular_scale <= 0.0:
        raise ValueError("SOURCE_PROFILE_TARGET_GLOBAL_COUPLING_SCALE_INVALID")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(SOURCE_PROFILE_TRACKING_V1),
        clip_ids=np.asarray(SOURCE_PROFILE_CLIPS),
        contact_activity=np.stack(contact_values),
        geometry_object_normalized=np.stack(geometry_values),
        geometry_valid=np.stack(geometry_valid),
        linear_coupling_normalized=np.stack(linear_values).astype(np.float32) / linear_scale,
        angular_coupling_normalized=np.stack(angular_values).astype(np.float32) / angular_scale,
        object_characteristic_length_m=np.asarray(spans, dtype=np.float32),
        linear_coupling_scale=np.asarray(linear_scale, dtype=np.float64),
        angular_coupling_scale=np.asarray(angular_scale, dtype=np.float64),
    )
    contract_output.parent.mkdir(parents=True, exist_ok=True)
    contract_output.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_PROFILE_TRACKING_V1,
                "contract": contract.as_dict(),
                "target": {"path": str(output), "sha256": _sha256(output)},
                "inputs": inputs,
                "global_normalization": {
                    "linear_coupling_scale": linear_scale,
                    "angular_coupling_scale": angular_scale,
                    "authority": (
                        "global_p95_over_all_raw_source_profile_frames_before_actual_evaluation"
                    ),
                },
                "source_profile_redefined": False,
                "dtw": "NO",
                "learned_time_warp": "NO",
                "outcome_dependent_phase_shift": "NO",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "SOURCE_PROFILE_TARGETS_MATERIALIZED", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
