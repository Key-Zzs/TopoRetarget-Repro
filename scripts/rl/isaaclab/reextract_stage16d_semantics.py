#!/usr/bin/env python3
"""Re-extract complete Stage 16-D source contact telemetry without inventing forces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
OLD_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
CLIPS = ("170105", "170650")
CONTACT_OFFSET_M = 0.002


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _group(pair_id: str) -> str:
    body = pair_id.split("/", 3)[2]
    for group in ("thumb", "index", "middle", "ring", "pinky"):
        if group in body:
            return group
    return "palm" if body == "r_wrist" else "unknown"


def _windows(active: np.ndarray) -> list[dict[str, int]]:
    indices = np.flatnonzero(active)
    if not indices.size:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [
        {"start": int(chunk[0]), "end": int(chunk[-1]), "steps": int(chunk.size)}
        for chunk in np.split(indices, split)
    ]


def _clip_payload(clip: str) -> dict[str, Any]:
    raw_path = REPORT_ROOT / f"source_runtime_penetration_pairs_{clip}.npz"
    trace_path = OLD_ROOT / f"source_trace_{clip}.npz"
    semantic_path = OLD_ROOT / f"task_semantics_{clip}.json"
    with np.load(raw_path, allow_pickle=False) as geometry:
        signed = np.asarray(geometry["signed_separation_m"], dtype=np.float64)[:, 0]
        pair_ids = np.asarray(geometry["pair_ids"], dtype=str)
    with np.load(trace_path, allow_pickle=False) as trace:
        object_twist = np.asarray(trace["object_twist"], dtype=np.float64)
    if signed.shape != (321, len(pair_ids)) or object_twist.shape != (321, 6):
        raise RuntimeError(f"STAGE16D_SEMANTIC_REEXTRACTION_SHAPE_FAILURE:{clip}")
    active = signed <= CONTACT_OFFSET_M
    delta_twist = np.diff(object_twist, axis=0, prepend=object_twist[:1])
    rows: list[dict[str, Any]] = []
    for frame in range(321):
        pair_indices = np.flatnonzero(active[frame])
        rows.append(
            {
                "frame": frame,
                "body_pairs": [str(pair_ids[index]) for index in pair_indices],
                "contact_groups": sorted({_group(str(pair_ids[index])) for index in pair_indices}),
                "minimum_signed_proxy_separation_m": float(signed[frame].min()),
                "active_pair_count": int(pair_indices.size),
                "aggregate_force_n": None,
                "aggregate_impulse_ns": None,
                "object_delta_v_mps": delta_twist[frame, :3].tolist(),
                "object_delta_omega_radps": delta_twist[frame, 3:].tolist(),
            }
        )
    pair_summaries = []
    for index, pair_id in enumerate(pair_ids):
        pair_active = active[:, index]
        windows = _windows(pair_active)
        pair_summaries.append(
            {
                "pair_id": str(pair_id),
                "contact_group": _group(str(pair_id)),
                "onset": windows[0]["start"] if windows else None,
                "duration_steps": int(pair_active.sum()),
                "persistence": float(pair_active.mean()),
                "windows": windows,
                "final_hold_steps": int(pair_active[-20:].sum()),
                "minimum_signed_proxy_separation_m": float(signed[:, index].min()),
            }
        )
    original = _load_json(semantic_path)
    confidence = float(original["classification_confidence"])
    return {
        "clip": f"hocap_{clip}",
        "frame_count": 321,
        "contact_definition": "runtime convex signed separation <= frozen 2mm contact offset",
        "pair_summaries": pair_summaries,
        "frames": rows,
        "aggregate_force_availability": (
            "NOT_AVAILABLE_IN_FROZEN_SOURCE_REFERENCE_TRACE; null values are preserved"
        ),
        "aggregate_impulse_availability": (
            "NOT_AVAILABLE_IN_FROZEN_SOURCE_REFERENCE_TRACE; null values are preserved"
        ),
        "classifier": {
            "reused_algorithm": "Stage16D shared semantic classifier",
            "confidence": confidence,
            "threshold": 0.60,
            "explicit_class_authorized": confidence >= 0.60,
            "task_class": original["task_class"],
        },
        "status": "STAGE16D_TASK_SEMANTICS_VALIDATED_WITH_GENERIC_FALLBACK",
        "source_trace": str(trace_path.relative_to(REPO_ROOT)),
        "source_geometry": str(raw_path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    clips = {f"hocap_{clip}": _clip_payload(clip) for clip in CLIPS}
    payload = {
        "schema_version": "Stage16DSourceContactTelemetryReextractionV1",
        "shared_classifier": True,
        "clip_specific_control_logic": False,
        "status": "STAGE16D_TASK_SEMANTICS_VALIDATED_WITH_GENERIC_FALLBACK",
        "clips": clips,
        "limitations": [
            "source reference traces do not contain aggregate force or impulse telemetry",
            "no classifier confidence was increased",
            "generic fallback is a soft semantic qualification, not a geometry or trajectory pass",
        ],
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "semantic_reextraction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "clips": list(clips)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
