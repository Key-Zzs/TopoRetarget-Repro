#!/usr/bin/env python3
"""Derive missing P3-B.5 telemetry and reference-target evidence offline.

This reads frozen C2 traces and counterfactual artifacts only. It does not
create environments, alter an RSI bank, train PPO, or change a formal gate.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / ".local/reports/stage16_p3b5_geometry_attribution"
PILOT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity/physical_pilot"
MANIFEST = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.exact_evaluator import evaluate_runtime_proxy_state
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.physics_retargeting.self_collision import (
    InterFingerCapsulePenetrationV1,
    load_self_collision_contract,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P3B5_JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def force_statistics(trace: dict[str, np.ndarray]) -> dict[str, object]:
    pair_force = np.asarray(trace["hand_object_pair_force_world"], dtype=np.float64)
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    if pair_force.ndim != 3 or pair_force.shape[1:] != (21, 3) or valid.shape != (len(pair_force),):
        raise ValueError("P3B5_HAND_PAIR_TELEMETRY_SHAPE_INVALID")
    valid_force = pair_force[valid]
    magnitudes = np.linalg.norm(valid_force, axis=-1)
    body_max = magnitudes.max(axis=0, initial=0.0)
    return {
        "valid_post_physics_frames": int(valid.sum()),
        "total_hand_object_force_peak_n": float(
            np.linalg.norm(valid_force.sum(axis=1), axis=-1).max(initial=0)
        ),
        "total_hand_object_force_p95_n": float(
            np.percentile(np.linalg.norm(valid_force.sum(axis=1), axis=-1), 95)
            if valid.any()
            else 0.0
        ),
        "per_body_force_peak_n": {
            name: float(body_max[index]) for index, name in enumerate(HAND_COLLISION_BODY_NAMES)
        },
        "contact_impulse": "UNAVAILABLE_NOT_CAPTURED_AS_PER_STEP_IMPULSE",
        "normal_tangential": "CONTACT_NORMAL_TELEMETRY_UNAVAILABLE",
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_per_frame_evidence(
    *,
    name: str,
    rows: list[dict[str, object]],
    include_controller_windows: bool,
) -> None:
    """Write padded, lossless per-frame evidence from already frozen NPZ traces.

    JSON summaries are convenient for review but cannot faithfully carry all
    pair-force vectors.  This compact archive keeps every captured full-pair
    force and exact geometry value, plus a JSON index that maps axis zero to
    a concrete trace and provenance receipt.
    """

    loaded: list[tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for row in rows:
        trace_path = Path(str(row["trace"]))
        geometry_path = Path(str(row["geometry"]))
        loaded.append((row, _load_npz(trace_path), _load_npz(geometry_path)))
    if not loaded:
        raise ValueError("P3B5_PER_FRAME_EVIDENCE_EMPTY")
    max_frames = max(len(trace["object_pose"]) for _, trace, _ in loaded)
    count = len(loaded)
    frame_valid = np.zeros((count, max_frames), dtype=bool)
    pair_force = np.full((count, max_frames, 21, 3), np.nan, dtype=np.float32)
    tip_force = np.full((count, max_frames, 5, 3), np.nan, dtype=np.float32)
    pair_presence = np.zeros((count, max_frames, 21), dtype=bool)
    tip_presence = np.zeros((count, max_frames, 5), dtype=bool)
    force_valid = np.zeros((count, max_frames), dtype=bool)
    contact_force = np.full((count, max_frames, 3), np.nan, dtype=np.float32)
    penetration = np.full((count, max_frames, 21), np.nan, dtype=np.float64)
    frame_worst = np.full((count, max_frames), np.nan, dtype=np.float64)
    frame_pair = np.full((count, max_frames), -1, dtype=np.int64)
    inter_finger = np.full((count, max_frames), np.nan, dtype=np.float64)
    index_rows: list[dict[str, object]] = []
    controller_arrays: dict[str, np.ndarray] = {}
    if include_controller_windows:
        window = 21
        controller_arrays = {
            "finger_target": np.full((count, window, 20), np.nan, dtype=np.float32),
            "finger_actual": np.full((count, window, 20), np.nan, dtype=np.float32),
            "finger_error": np.full((count, window, 20), np.nan, dtype=np.float32),
            "finger_qdot": np.full((count, window, 20), np.nan, dtype=np.float32),
            "wrist_target": np.full((count, window, 7), np.nan, dtype=np.float32),
            "wrist_actual": np.full((count, window, 7), np.nan, dtype=np.float32),
            "wrist_translation_error_m": np.full((count, window), np.nan, dtype=np.float32),
            "action26": np.full((count, window, 26), np.nan, dtype=np.float32),
            "actuator_effort": np.full((count, window, 26), np.nan, dtype=np.float32),
            "controller_window_frame": np.full((count, window), -1, dtype=np.int64),
        }
    self_collision = load_self_collision_contract(
        REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml", repo_root=REPO_ROOT
    )
    inter_finger_metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        REPO_ROOT / self_collision.runtime_collision_manifest_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=self_collision.capsule_radius_scale,
        device="cpu",
    )
    for index, (row, trace, geometry) in enumerate(loaded):
        frames = len(trace["object_pose"])
        frame_valid[index, :frames] = True
        pair_force[index, :frames] = trace["hand_object_pair_force_world"]
        tip_force[index, :frames] = trace.get(
            "tip_pair_force_world", trace["fingertip_object_pair_force_world"]
        )
        pair_presence[index, :frames] = trace["hand_object_pair_presence"]
        tip_presence[index, :frames] = trace.get("tip_pair_presence", trace["actual_contact_mask"])
        force_valid[index, :frames] = trace["hand_object_pair_force_valid"]
        contact_force[index, :frames] = trace["contact_force_world"]
        penetration[index, :frames] = geometry["penetration_depth_m"][:, 0]
        frame_worst[index, :frames] = geometry["frame_worst_penetration_m"][:, 0]
        frame_pair[index, :frames] = geometry["frame_worst_pair_index"][:, 0]
        import torch

        with torch.no_grad():
            inter_finger[index, :frames] = (
                inter_finger_metric.evaluate(
                    torch.as_tensor(trace["hand_collision_body_pose"], dtype=torch.float32)
                )["maximum_penetration_m"]
                .detach()
                .cpu()
                .numpy()
            )
        first = row.get("first_geometry_violation_frame")
        center = int(first) if first is not None else int(np.nanargmax(frame_worst[index, :frames]))
        if include_controller_windows:
            start, stop = max(0, center - 10), min(frames, center + 11)
            size = stop - start
            controller_arrays["finger_target"][index, :size] = trace["finger_target"][start:stop]
            controller_arrays["finger_actual"][index, :size] = trace["finger_q"][start:stop]
            controller_arrays["finger_error"][index, :size] = (
                trace["finger_target"][start:stop] - trace["finger_q"][start:stop]
            )
            controller_arrays["finger_qdot"][index, :size] = trace["finger_qdot"][start:stop]
            controller_arrays["wrist_target"][index, :size] = trace["wrist_target"][start:stop]
            controller_arrays["wrist_actual"][index, :size] = trace["wrist_pose"][start:stop]
            controller_arrays["wrist_translation_error_m"][index, :size] = np.linalg.norm(
                trace["wrist_target"][start:stop, :3] - trace["wrist_pose"][start:stop, :3], axis=-1
            )
            controller_arrays["action26"][index, :size] = trace["action"][start:stop]
            controller_arrays["actuator_effort"][index, :size] = trace["actuator_effort"][
                start:stop
            ]
            controller_arrays["controller_window_frame"][index, :size] = np.arange(start, stop)
        index_rows.append(
            {
                "axis0": index,
                "label": row.get("label"),
                "case_id": row.get("case_id"),
                "clip": row["clip"],
                "mode": row.get("mode"),
                "variant": row.get("variant"),
                "episode": row.get("episode"),
                "frames": frames,
                "trace": str(Path(str(row["trace"])).resolve()),
                "geometry": str(Path(str(row["geometry"])).resolve()),
                "controller_window_center_frame": center,
                "finger_effort_limit": 0.6 if include_controller_windows else None,
                "contact_normal": "CONTACT_NORMAL_TELEMETRY_UNAVAILABLE",
                "contact_impulse": "UNAVAILABLE_NOT_CAPTURED_AS_PER_STEP_IMPULSE",
            }
        )
    destination = OUTPUT / "telemetry"
    np.savez_compressed(
        destination / "contact" / f"{name}_full_pair_per_frame.npz",
        frame_valid=frame_valid,
        hand_object_pair_force_world=pair_force,
        tip_pair_force_world=tip_force,
        hand_object_pair_presence=pair_presence,
        tip_pair_presence=tip_presence,
        hand_object_pair_force_valid=force_valid,
        contact_force_world=contact_force,
    )
    np.savez_compressed(
        destination / "geometry" / f"{name}_per_frame.npz",
        frame_valid=frame_valid,
        penetration_depth_m=penetration,
        frame_worst_penetration_m=frame_worst,
        frame_worst_pair_index=frame_pair,
        inter_finger_max_penetration_m=inter_finger,
    )
    if include_controller_windows:
        np.savez_compressed(destination / "controller" / f"{name}_windows.npz", **controller_arrays)
    write_json(destination / f"{name}_per_frame_index.json", {"rows": index_rows})


def inspect_trace(
    *,
    clip: str,
    trace_path: Path,
    geometry_path: Path,
    p95_limit: float,
    label: str,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(geometry_path, allow_pickle=False) as archive:
        worst = np.asarray(archive["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
        pair_index = np.asarray(archive["frame_worst_pair_index"], dtype=np.int64)[:, 0]
        pair_ids = [str(item) for item in archive["pair_ids"].tolist()]
    contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    active = np.flatnonzero(contact)
    violated = np.flatnonzero(worst > p95_limit)
    maximum = int(np.argmax(worst))
    first = None if not violated.size else int(violated[0])
    body = HAND_COLLISION_BODY_NAMES[int(pair_index[maximum])]
    return {
        "label": label,
        "clip": clip,
        "trace": str(trace_path.resolve()),
        "geometry": str(geometry_path.resolve()),
        "frame_count": len(worst),
        "reset_frame": 0,
        "first_actual_hand_object_contact_frame": None if not active.size else int(active[0]),
        "first_geometry_violation_frame": first,
        "maximum_penetration_frame": maximum,
        "last_frame_before_violation": None if first in {None, 0} else first - 1,
        "violating_hand_body": body,
        "violating_object_proxy": pair_ids[int(pair_index[maximum])].split("<->", 1)[1],
        "maximum_penetration_m": float(worst[maximum]),
        "initial_penetration_m": float(worst[0]),
        "violation_duration_frames": int((worst > p95_limit).sum()),
        "active_violating_pair_count_at_max": int(
            (
                np.asarray(
                    trace.get("hand_object_pair_presence", np.zeros((len(worst), 21), dtype=bool))
                )[maximum]
            ).sum()
        ),
        "force": force_statistics(trace),
    }


def reference_target_audit(case: dict[str, object], p95_limit: float) -> dict[str, object]:
    trace_path = Path(str(case["trace"]["path"]))
    with np.load(trace_path, allow_pickle=False) as archive:
        wrist_reference = np.asarray(archive["wrist_reference"], dtype=np.float64)
        finger_reference = np.asarray(archive["finger_reference"], dtype=np.float64)
        object_reference = np.asarray(archive["object_reference"], dtype=np.float64)
        object_actual = np.asarray(archive["object_pose"], dtype=np.float64)
        wrist_actual = np.asarray(archive["wrist_pose"], dtype=np.float64)
        finger_actual = np.asarray(archive["finger_q"], dtype=np.float64)
    reference_hand = reconstruct_hand_collision_body_pose(
        wrist_reference, finger_reference, repo_root=REPO_ROOT
    )
    actual_hand = reconstruct_hand_collision_body_pose(
        wrist_actual, finger_actual, repo_root=REPO_ROOT
    )
    reference_geometry, reference_raw = evaluate_runtime_proxy_state(
        manifest_path=MANIFEST,
        clip=str(case["clip"]),
        object_pose=object_reference[:, None],
        hand_collision_body_pose=reference_hand[:, None],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    target_actual_geometry, target_actual_raw = evaluate_runtime_proxy_state(
        manifest_path=MANIFEST,
        clip=str(case["clip"]),
        object_pose=object_actual[:, None],
        hand_collision_body_pose=reference_hand[:, None],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    actual_geometry, actual_raw = evaluate_runtime_proxy_state(
        manifest_path=MANIFEST,
        clip=str(case["clip"]),
        object_pose=object_actual[:, None],
        hand_collision_body_pose=actual_hand[:, None],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    frame = 0
    quaternion_dot = float(
        np.clip(abs(np.dot(object_actual[frame, 3:7], object_reference[frame, 3:7])), 0.0, 1.0)
    )
    return {
        "case_id": case["case_id"],
        "clip": case["clip"],
        "failure_frame": frame,
        "reference_target_vs_reference_object_penetration_m": float(
            reference_raw["frame_worst_penetration_m"][frame, 0]
        ),
        "reference_target_vs_actual_object_penetration_m": float(
            target_actual_raw["frame_worst_penetration_m"][frame, 0]
        ),
        "actual_hand_vs_actual_object_penetration_m": float(
            actual_raw["frame_worst_penetration_m"][frame, 0]
        ),
        "reference_target_geometry_pass": bool(
            float(reference_raw["frame_worst_penetration_m"][frame, 0]) <= p95_limit
        ),
        "reference_actual_object_translation_error_m": float(
            np.linalg.norm(object_actual[frame, :3] - object_reference[frame, :3])
        ),
        "reference_actual_object_orientation_error_deg": float(
            np.degrees(2.0 * np.arccos(quaternion_dot))
        ),
        "reference_geometry_aggregate": reference_geometry,
        "target_actual_geometry_aggregate": target_actual_geometry,
        "actual_geometry_aggregate": actual_geometry,
        "conclusion": "RESET_STATE_GEOMETRY_INVALID_BEFORE_CONTACT_RESPONSE"
        if float(actual_raw["frame_worst_penetration_m"][frame, 0]) > p95_limit
        else "NOT_RESET_INVALID",
    }


def _rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fcl_mesh(vertices: np.ndarray, faces: np.ndarray) -> Any:
    import fcl

    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    model.endModel()
    return model


def visual_triangle_intersection(case: dict[str, object]) -> dict[str, object]:
    """Check visual triangle intersections at reset; signed depth remains unavailable."""

    import fcl
    import trimesh

    trace_path = Path(str(case["trace"]["path"]))
    manifest = read_json(MANIFEST)
    hand = manifest["hand_shapes"][4]
    object_path = (
        REPO_ROOT
        / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
        / f"{case['clip']}.obj"
    )
    hand_path = REPO_ROOT / str(hand["source_asset_path"])
    with np.load(trace_path, allow_pickle=False) as archive:
        hand_pose = np.asarray(archive["hand_collision_body_pose"], dtype=np.float64)[0, 4]
        object_pose = np.asarray(archive["object_pose"], dtype=np.float64)[0]
    object_mesh = trimesh.load_mesh(object_path, process=False)
    hand_mesh = trimesh.load_mesh(hand_path, process=False)
    if not isinstance(object_mesh, trimesh.Trimesh) or not isinstance(hand_mesh, trimesh.Trimesh):
        raise RuntimeError("P3B5_VISUAL_TRIANGLE_MESH_LOAD_FAILURE")
    hand_vertices = (
        np.asarray(hand_mesh.vertices) @ _rotation_matrix(hand_pose[3:7]).T + hand_pose[:3]
    )
    object_vertices = (
        np.asarray(object_mesh.vertices) @ _rotation_matrix(object_pose[3:7]).T + object_pose[:3]
    )
    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
    result = fcl.CollisionResult()
    contacts = fcl.collide(
        fcl.CollisionObject(_fcl_mesh(object_vertices, np.asarray(object_mesh.faces))),
        fcl.CollisionObject(_fcl_mesh(hand_vertices, np.asarray(hand_mesh.faces))),
        request,
        result,
    )
    return {
        "case_id": case["case_id"],
        "clip": case["clip"],
        "frame": 0,
        "triangle_triangle_intersection": bool(contacts > 0),
        "contact_count_lower_bound": int(contacts),
        "object_visual_watertight": bool(object_mesh.is_watertight),
        "hand_visual_watertight": bool(hand_mesh.is_watertight),
        "signed_visual_penetration": "UNAVAILABLE_NONWATERTIGHT_OBJECT",
        "classification": (
            "TRUE_VISUAL_GEOMETRY_CONSISTENT"
            if contacts > 0
            else "COLLISION_PROXY_GEOMETRY_DISCREPANCY"
        ),
    }


def main() -> int:
    gates = read_json(OUTPUT / "geometry_contract.json")["gates"]
    inventory = read_json(OUTPUT / "c2_failure_inventory.json")["episodes"]
    selected = read_json(OUTPUT / "selected_cases.json")["cases"]
    temporal: list[dict[str, object]] = []
    for row in inventory:
        trace = Path(str(row["trace"]["path"]))
        geometry = Path(str(row["geometry_sidecar"]["path"]))
        temporal.append(
            inspect_trace(
                clip=str(row["clip"]),
                trace_path=trace,
                geometry_path=geometry,
                p95_limit=float(gates[str(row["clip"])]["p95_penetration_inclusive_m"]),
                label=f"historical:{row['mode']}:episode_{int(row['episode']):03d}",
            )
            | {
                "mode": row["mode"],
                "episode": row["episode"],
                "reset_index": row["reset_index"],
                "absolute_geometry_pass": row["absolute_geometry_pass"],
            }
        )
    reference_rows = [
        reference_target_audit(case, float(gates[str(case["clip"])]["p95_penetration_inclusive_m"]))
        for case in selected
    ]
    visual_intersections = [visual_triangle_intersection(case) for case in selected]
    failing = [row for row in temporal if not bool(row["absolute_geometry_pass"])]
    excluded = Counter((str(row["clip"]), int(row["reset_index"])) for row in failing)
    candidate_exclusions = [
        {
            "clip": clip,
            "reset_index": reset,
            "historical_failure_occurrences": count,
            "reason": (
                "C2 formal geometry failure at reset frame 0; proposed only, safe bank unchanged"
            ),
        }
        for (clip, reset), count in sorted(excluded.items())
    ]
    time_distribution: dict[str, dict[str, int]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        subset = [row for row in failing if row["clip"] == clip]
        classes = Counter(
            "INITIAL_GEOMETRY_INVALID"
            if row["first_geometry_violation_frame"] == 0
            else "UNKNOWN_TEMPORAL_FAILURE"
            if row["first_geometry_violation_frame"] is None
            else "CONTACT_TRANSIENT_GEOMETRY_FAILURE"
            for row in subset
        )
        time_distribution[clip] = {
            name: int(classes.get(name, 0))
            for name in (
                "INITIAL_GEOMETRY_INVALID",
                "CONTACT_TRANSIENT_GEOMETRY_FAILURE",
                "SUSTAINED_LOAD_GEOMETRY_FAILURE",
                "LATE_POLICY_GEOMETRY_FAILURE",
                "UNKNOWN_TEMPORAL_FAILURE",
            )
        }
    write_json(
        OUTPUT / "telemetry/geometry/historical_temporal_contact_force.json",
        {"rows": temporal, "failure_temporal_distribution": time_distribution},
    )
    write_json(OUTPUT / "telemetry/contact/historical_full_pair_force.json", {"rows": temporal})
    _write_per_frame_evidence(
        name="historical_c2",
        rows=[
            {
                **row,
                "trace": row["trace"],
                "geometry": row["geometry"],
                "first_geometry_violation_frame": row["first_geometry_violation_frame"],
            }
            for row in temporal
        ],
        include_controller_windows=False,
    )
    counterfactual_rows: list[dict[str, object]] = []
    for result_path in sorted((OUTPUT / "counterfactuals").glob("**/result.json")):
        result = read_json(result_path)
        counterfactual_rows.append(
            {
                "label": (
                    f"{result['clip']}:{result['diagnostic_mode']}:{result['physics']['variant']}"
                ),
                "case_id": result_path.parents[2].name,
                "clip": result["clip"],
                "mode": result["diagnostic_mode"],
                "variant": result["physics"]["variant"],
                "episode": result["episode"],
                "trace": result["trace"]["path"],
                "geometry": result["geometry_sidecar"]["path"],
                "first_geometry_violation_frame": result["geometry"]["first_violation_frame"],
            }
        )
    if len(counterfactual_rows) != 32:
        raise RuntimeError(
            f"P3B5_COUNTERFACTUAL_PER_FRAME_EVIDENCE_INCOMPLETE:{len(counterfactual_rows)}"
        )
    _write_per_frame_evidence(
        name="counterfactual",
        rows=counterfactual_rows,
        include_controller_windows=True,
    )
    write_json(
        OUTPUT / "telemetry/controller/reference_target_collision_audit.json",
        {"rows": reference_rows},
    )
    write_json(
        OUTPUT / "proxy_audit/visual_triangle_intersection.json",
        {"rows": visual_intersections},
    )
    write_json(
        OUTPUT / "proposed_rsi_filter.json",
        {
            "status": "PROPOSAL_ONLY_SAFE_BANK_UNCHANGED",
            "candidate_excluded_reset_indices": candidate_exclusions,
        },
    )
    write_csv(
        OUTPUT / "tables/attribution_core_table_1.csv",
        [
            {
                "clip": row["clip"],
                "mode": row["mode"],
                "episode": row["episode"],
                "reset_index": row["reset_index"],
                "first_violation": row["first_geometry_violation_frame"],
                "max_frame": row["maximum_penetration_frame"],
                "hand_body": row["violating_hand_body"],
                "object_proxy": row["violating_object_proxy"],
                "initial_penetration_mm": float(row["initial_penetration_m"]) * 1000.0,
                "p95_penetration_mm": float(
                    next(
                        item["hand_object_p95_penetration_m"]
                        for item in inventory
                        if item["mode"] == row["mode"]
                        and item["clip"] == row["clip"]
                        and item["episode"] == row["episode"]
                    )
                )
                * 1000.0,
                "max_penetration_mm": float(row["maximum_penetration_m"]) * 1000.0,
                "temporal_class": (
                    "INITIAL_GEOMETRY_INVALID"
                    if row["first_geometry_violation_frame"] == 0
                    else "NO_FAILURE"
                    if bool(row["absolute_geometry_pass"])
                    else "UNKNOWN_TEMPORAL_FAILURE"
                ),
            }
            for row in temporal
        ],
    )
    print(json.dumps({"status": "P3B5_OFFLINE_EVIDENCE_COMPLETE", "rows": len(temporal)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
