"""Lazy, deterministic GRAB and ContactPose benchmark selection."""

# ruff: noqa: E501

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.grab import GrabDatasetAdapter as _GrabDatasetAdapter  # noqa: F401
from toporetarget.data.contacts.grab import load_grab_contact_mapping
from toporetarget.data.indexes.grab import build_grab_index, load_grab_index
from toporetarget.data.readers.grab import (
    load_grab_auxiliary,
    load_ply_mesh,
    read_grab_npz,
    resolve_grab_resource,
)

from .contactpose import ContactPoseDatasetAdapter
from .schema import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkUnit,
    file_hash,
    git_commit,
    stable_hash,
    utc_now,
    write_json,
    write_rows_csv,
    write_units,
)

WINDOW = 60
EXISTING_SEQUENCE = "s1/airplane_lift"
EXISTING_RANGE = [240, 300]


@lru_cache(maxsize=256)
def _mesh_audit_cached(path_string: str, size: int, mtime_ns: int) -> tuple[bool, str]:
    path = Path(path_string)
    try:
        vertices, faces = load_ply_mesh(path)
    except Exception as exc:
        return False, f"mesh_load:{type(exc).__name__}:{exc}"
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        return False, "mesh_nonfinite_or_shape"
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        return False, "mesh_faces_missing"
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not bool(mesh.is_watertight):
            return False, "mesh_not_watertight_strict_sdf_unavailable"
    except ImportError:
        return False, "trimesh_unavailable"
    except Exception as exc:
        return False, f"mesh_audit:{type(exc).__name__}:{exc}"
    return True, "strict_reference_winding_available"


def _mesh_audit(path: Path) -> tuple[bool, str]:
    stat = path.stat()
    return _mesh_audit_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns))


def _label_table() -> dict[int, dict[str, Any]]:
    return load_grab_contact_mapping().table()


def _hand_label_ids(table: dict[int, dict[str, Any]], side: str) -> dict[int, dict[str, Any]]:
    category = "right_hand" if side == "right" else "left_hand"
    return {key: value for key, value in table.items() if value.get("category") == category}


def _region_name(item: dict[str, Any]) -> str:
    return str(item.get("name", "unknown")).lower().replace("_", "-")


def _window_stats(
    labels: np.ndarray, table: dict[int, dict[str, Any]], side: str
) -> list[dict[str, Any]]:
    hand_table = _hand_label_ids(table, side)
    windows: list[dict[str, Any]] = []
    if labels.ndim != 2 or labels.shape[0] < WINDOW:
        return windows
    for start in range(labels.shape[0] - WINDOW + 1):
        clip = labels[start : start + WINDOW]
        observed = set(int(item) for item in np.unique(clip))
        regions = [hand_table[item] for item in sorted(observed) if item in hand_table]
        hand_mask = np.isin(clip, list(hand_table))
        frame_ratio = float(np.mean(np.any(hand_mask, axis=1)))
        names = [_region_name(item) for item in regions]
        thumb = any("thumb" in name for name in names)
        long_finger = any(
            any(finger in name for finger in ("index", "middle", "ring", "pinky")) for name in names
        )
        non_tip = any(name.endswith(("1", "2")) or "hand" in name for name in names)
        region_diversity = len(set(re.sub(r"[0-9]+$", "", name) for name in names))
        windows.append(
            {
                "start": start,
                "end": start + WINDOW,
                "contact_frame_ratio": frame_ratio,
                "contact_regions": sorted(names),
                "contact_region_count": len(set(names)),
                "contact_region_diversity": region_diversity,
                "thumb_contact": thumb,
                "long_finger_contact": long_finger,
                "non_tip_or_palm_contact": non_tip,
                "valid_contact": bool(
                    frame_ratio >= 0.70 and thumb and long_finger and region_diversity >= 2
                ),
            }
        )
    return windows


def _best_window(path: Path, side: str, table: dict[int, dict[str, Any]]) -> dict[str, Any]:
    try:
        auxiliary = load_grab_auxiliary(path, include_table=False, contact_mode="source")
        labels = np.asarray(auxiliary.get("contact", {}).get("object"))
    except Exception as exc:
        return {"rejection_reasons": [f"contact_load:{type(exc).__name__}:{exc}"]}
    windows = _window_stats(labels, table, side)
    if not windows:
        return {"rejection_reasons": ["no_60_frame_contiguous_window"]}
    windows.sort(
        key=lambda item: (
            -float(item["contact_frame_ratio"]),
            -int(item["contact_region_diversity"]),
            -int(item["non_tip_or_palm_contact"]),
            int(item["start"]),
        )
    )
    return {
        "best": windows[0],
        "window_count": len(windows),
        "valid_windows": sum(bool(item["valid_contact"]) for item in windows),
    }


def _grab_candidate(
    entry: dict[str, Any], root: Path, table: dict[int, dict[str, Any]], target_hand: str
) -> dict[str, Any]:
    path = root / str(entry["relative_path"])
    result: dict[str, Any] = {
        "native_sample_id": entry["sequence_id"],
        "source_path": str(path),
        "source_hash": entry.get("source_hash")
        or f"stat:{entry.get('file_size')}:{entry.get('mtime_ns')}",
        "subject": entry.get("subject_id", path.parent.name),
        "object_name": entry.get("object_token", "unknown"),
        "action": entry.get("action_token", "unknown"),
        "candidate_hand": target_hand,
        "rejection_reasons": [],
    }
    try:
        # Contact labels are the cheap, decisive first gate.  Do not unpack
        # every MANO/object parameter array for candidates that cannot provide
        # the required 60-frame semantic-contact window.
        window = _best_window(path, target_hand, table)
        if "best" not in window:
            result["rejection_reasons"].extend(window.get("rejection_reasons", []))
            return result
        result.update(window["best"])
        if not result["valid_contact"]:
            result["rejection_reasons"].extend(
                [
                    "contact_frame_ratio_below_0.70"
                    if result["contact_frame_ratio"] < 0.70
                    else "",
                    "thumb_contact_missing" if not result["thumb_contact"] else "",
                    "long_finger_contact_missing" if not result["long_finger_contact"] else "",
                    "fewer_than_two_contact_regions"
                    if result["contact_region_diversity"] < 2
                    else "",
                ]
            )
            result["rejection_reasons"] = [item for item in result["rejection_reasons"] if item]
            return result
        record = read_grab_npz(path, compute_source_hash=False)
        result.update(
            {
                "subject": record.subject_id,
                "object_name": record.object_name,
                "action": record.motion_intent,
                "native_fps": record.native_fps,
                "num_frames": record.num_frames,
            }
        )
        sides = [target_hand] if target_hand in record.hands else []
        if not sides:
            sides = [side for side in ("right", "left") if side in record.hands]
        if not sides:
            result["rejection_reasons"].append("target_hand_missing")
            return result
        side = sides[0]
        result["candidate_hand"] = side
        vtemp = resolve_grab_resource(root, record.hands[side].vtemp_relative, "personalized vtemp")
        object_mesh = resolve_grab_resource(root, record.object.mesh_relative, "object mesh")
        result["personalized_vtemp_path"] = str(vtemp)
        result["personalized_vtemp_hash"] = (
            f"stat:{vtemp.stat().st_size}:{vtemp.stat().st_mtime_ns}"
        )
        result["object_mesh_path"] = str(object_mesh)
        result["object_mesh_hash"] = (
            f"stat:{object_mesh.stat().st_size}:{object_mesh.stat().st_mtime_ns}"
        )
        if not vtemp.is_file():
            result["rejection_reasons"].append("personalized_mano_vtemp_missing")
        good_mesh, mesh_reason = _mesh_audit(object_mesh)
        result["sdf_validity"] = "valid" if good_mesh else "invalid"
        if not good_mesh:
            result["rejection_reasons"].append(mesh_reason)
        result["source_contact_geometry_proxy"] = "object_contact_labels_plus_mesh_audit"
        result["selection_score"] = {
            "contact_frame_ratio": float(result.get("contact_frame_ratio", 0.0)),
            "contact_region_diversity": float(result.get("contact_region_diversity", 0)),
            "non_tip_palm_coverage": float(bool(result.get("non_tip_or_palm_contact", False))),
        }
    except Exception as exc:
        result["rejection_reasons"].append(f"candidate_parse:{type(exc).__name__}:{exc}")
    return result


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(item.get("contact_frame_ratio", 0.0)),
        -int(item.get("contact_region_diversity", 0)),
        -int(bool(item.get("non_tip_or_palm_contact", False))),
        float(item.get("source_contact_geometry_consistency", 0.0)),
        -int(bool(item.get("object_diversity_bonus", False))),
        str(item.get("native_sample_id", "")),
        int(item.get("start", 0)),
    )


def select_grab(
    *,
    grab_root: str | Path,
    output_root: str | Path,
    additional_target: int = 3,
    additional_minimum: int = 2,
    index_path: str | Path | None = None,
    scan_limit: int = 16,
) -> dict[str, Any]:
    """Inspect candidate metadata/contact labels and select fixed 60-frame clips."""

    root = Path(grab_root).expanduser().resolve()
    destination = Path(output_root)
    index = Path(index_path) if index_path else destination / "grab_index"
    if (index / "index.jsonl").is_file():
        index_result = {"index": str(index), "reused_existing_index": True}
    else:
        index_result = build_grab_index(grab_root=root, output=index, hash_files=False)
    table = _label_table()
    entries = load_grab_index(index)
    candidates: list[dict[str, Any]] = []
    if scan_limit < additional_target:
        raise ValueError("scan_limit must be at least additional_target")
    scanned_non_fixed = 0
    for entry in entries:
        if entry.get("sequence_id") == EXISTING_SEQUENCE:
            continue
        candidates.append(_grab_candidate(entry, root, table, "right"))
        scanned_non_fixed += 1
        if scanned_non_fixed >= scan_limit:
            break
    valid = [item for item in candidates if not item.get("rejection_reasons")]
    objects = {EXISTING_SEQUENCE.split("/")[-1]}
    subjects = {"s1"}
    selected: list[dict[str, Any]] = []
    for item in sorted(valid, key=_candidate_sort_key):
        if len(selected) >= additional_target:
            break
        bonus = int(item.get("object_name") not in objects) + int(
            item.get("subject") not in subjects
        )
        item["object_diversity_bonus"] = bool(bonus)
        if (
            len(selected) < 2
            and item.get("object_name") in objects
            and any(other.get("object_name") not in objects for other in valid)
        ):
            continue
        item["selection_reasons"] = [
            "60 contiguous native frames",
            f"contact_frame_ratio={float(item['contact_frame_ratio']):.3f}",
            "thumb plus long-finger semantic contact",
            f"{int(item['contact_region_diversity'])} contact-region families",
            "strict reference winding mesh audit",
        ]
        selected.append(item)
        objects.add(str(item.get("object_name")))
        subjects.add(str(item.get("subject")))
    if len(selected) < additional_target:
        for item in sorted(valid, key=_candidate_sort_key):
            if item in selected:
                continue
            if len(selected) >= additional_target:
                break
            item["selection_reasons"] = [
                "best remaining valid candidate under deterministic ranking"
            ]
            selected.append(item)
    existing_path = root / "grab" / "s1" / "airplane_lift.npz"
    if not existing_path.is_file():
        existing_matches = [
            item for item in entries if item.get("sequence_id") == EXISTING_SEQUENCE
        ]
        existing_path = (
            root / existing_matches[0]["relative_path"] if existing_matches else existing_path
        )
    fixed = _grab_candidate(
        {"sequence_id": EXISTING_SEQUENCE, "relative_path": str(existing_path.relative_to(root))},
        root,
        table,
        "right",
    )
    fixed.update(
        {
            "start": EXISTING_RANGE[0],
            "end": EXISTING_RANGE[1],
            "selection_reasons": ["pre-existing accepted Stage 10 reference clip"],
        }
    )
    if fixed.get("rejection_reasons"):
        fixed["rejection_reasons"] = [item for item in fixed["rejection_reasons"] if item]
    all_candidates = [fixed, *candidates]
    selected_payload = [fixed, *selected]
    for rank, item in enumerate(selected_payload, 1):
        item["frozen_selection_rank"] = rank
        if item.get("source_path"):
            item["source_hash"] = file_hash(item["source_path"])
        if item.get("object_mesh_path"):
            item["object_mesh_hash"] = file_hash(item["object_mesh_path"])
        if item.get("personalized_vtemp_path"):
            item["personalized_vtemp_hash"] = file_hash(item["personalized_vtemp_path"])
        item["benchmark_id"] = (
            f"grab_{item['subject']}_{Path(str(item['native_sample_id'])).name}_{item['candidate_hand']}_f{int(item['start']):06d}_f{int(item['end']):06d}"
        )
    write_json(index_result, destination / "grab_index_result.json")
    write_json(all_candidates, destination / "grab_candidates.json")
    write_rows_csv(
        all_candidates,
        destination / "grab_candidates.csv",
        [
            "native_sample_id",
            "subject",
            "object_name",
            "action",
            "candidate_hand",
            "start",
            "end",
            "contact_frame_ratio",
            "contact_region_diversity",
            "contact_regions",
            "sdf_validity",
            "rejection_reasons",
            "selection_reasons",
        ],
    )
    write_json(selected_payload, destination / "grab_selected.json")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pass" if len(selected) >= additional_minimum else "blocked",
        "grab_root": str(root),
        "index": str(index),
        "target_additional_clips": additional_target,
        "minimum_additional_clips": additional_minimum,
        "selected": selected_payload,
        "candidate_count": len(all_candidates),
        "candidate_pool_count": max(0, len(entries) - 1),
        "evaluated_candidate_count": len(candidates),
        "scan_truncated": len(candidates) < max(0, len(entries) - 1),
        "valid_additional_count": len(valid),
        "selection_code": "toporetarget.benchmark.selection.grab.v1",
    }


def select_contactpose(
    *, root: str | Path, output_root: str | Path, target: int = 4, minimum: int = 3
) -> dict[str, Any]:
    adapter = ContactPoseDatasetAdapter(root)
    audit = adapter.inspect()
    candidates = adapter.index()
    selected = adapter.select(candidates, target=target)
    for item in selected:
        item["source_hash"] = file_hash(item["source_path"])
        if item.get("object_mesh_path"):
            item["object_mesh_hash"] = file_hash(item["object_mesh_path"])
    write_json(audit, Path(output_root) / "contactpose_index.json")
    write_json(candidates, Path(output_root) / "contactpose_candidates.json")
    write_rows_csv(
        candidates,
        Path(output_root) / "contactpose_candidates.csv",
        [
            "native_sample_id",
            "subject",
            "object_name",
            "grasp_name",
            "hand",
            "dynamic",
            "object_mesh_path",
            "contact_annotation_type",
            "contact_regions",
            "rejection_reasons",
        ],
    )
    write_json(selected, Path(output_root) / "contactpose_selected.json")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pass" if len(selected) >= minimum else "blocked",
        "target": target,
        "minimum": minimum,
        "selected": selected,
        "candidate_count": len(candidates),
        "excluded_deep_concave_diagnostic_set": sorted(adapter.excluded_objects),
        "audit": audit,
    }


def freeze_selection(
    *,
    grab_result: dict[str, Any],
    contactpose_result: dict[str, Any],
    config: dict[str, Any],
    output_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build an immutable selection manifest before any baseline execution."""

    destination = Path(output_root)
    all_records = list(grab_result.get("selected", [])) + list(
        contactpose_result.get("selected", [])
    )
    units: list[BenchmarkUnit] = []
    for item in all_records:
        dataset = "grab" if str(item.get("native_sample_id", "")).startswith("s") else "contactpose"
        dynamic = bool(item.get("dynamic", dataset == "grab"))
        frame_range = [int(item.get("start", 0)), int(item.get("end", 1))]
        if dataset == "contactpose":
            frame_range = [0, 1]
        units.append(
            BenchmarkUnit(
                benchmark_id=str(
                    item.get("benchmark_id") or f"{dataset}_{item.get('native_sample_id')}"
                ),
                dataset=dataset,
                native_sample_id=str(item.get("native_sample_id", "")),
                subject=str(item.get("subject", "unknown")),
                object_name=str(item.get("object_name", "unknown")),
                action=str(item.get("action", item.get("grasp_name", "unknown"))),
                hand=str(item.get("candidate_hand", item.get("hand", "right"))),
                side=str(item.get("candidate_hand", item.get("hand", "right"))),
                frame_range=frame_range,
                dynamic=dynamic,
                native_static_grasp=not dynamic,
                temporal_metrics_applicable=dynamic,
                native_fps=float(item["native_fps"])
                if item.get("native_fps") is not None
                else None,
                source_path=str(item.get("source_path", "")),
                source_hash=item.get("source_hash"),
                object_mesh_path=item.get("object_mesh_path"),
                object_mesh_hash=item.get("object_mesh_hash"),
                contact_annotation_type=str(
                    item.get(
                        "contact_annotation_type",
                        "grab_semantic_object_vertex" if dataset == "grab" else "unavailable",
                    )
                ),
                contact_annotation_hash=item.get("contact_annotation_hash"),
                contact_regions=list(item.get("contact_regions", [])),
                contact_mode=str(
                    item.get("contact_mode", "semantic" if dataset == "grab" else "native")
                ),
                canonical_validity=str(item.get("canonical_validity", "pending")),
                sdf_validity=str(
                    item.get(
                        "sdf_validity", "valid" if not item.get("rejection_reasons") else "invalid"
                    )
                ),
                source_identity=str(item.get("source_identity", item.get("source_path", ""))),
                selection_score=dict(item.get("selection_score", {})),
                selection_reasons=list(item.get("selection_reasons", [])),
                rejection_reasons=list(item.get("rejection_reasons", [])),
                frozen_selection_rank=item.get("frozen_selection_rank"),
                provenance={"adapter": item.get("adapter", "lazy_selection"), "raw": item},
            )
        )
    write_units(units, destination)
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_name": "Multi-Dataset Interaction Benchmark v1",
        "git_commit": git_commit(repo_root),
        "selection_code": "toporetarget.benchmark.selection.v1",
        "dataset_roots": {
            "grab": grab_result.get("grab_root"),
            "contactpose": contactpose_result.get("audit", {}).get("root"),
        },
        "selected_units": [unit.as_dict() for unit in units],
        "all_rejected_candidates": {
            "grab": [
                item for item in grab_result.get("selected", []) if item.get("rejection_reasons")
            ],
            "contactpose": [
                item
                for item in contactpose_result.get("selected", [])
                if item.get("rejection_reasons")
            ],
        },
        "selection_results": {"grab": grab_result, "contactpose": contactpose_result},
        "config": config,
        "random_seed": 20260724,
        "frozen_at": utc_now(),
    }
    payload["manifest_hash"] = stable_hash(payload)
    write_json(payload, destination / "benchmark_selection_manifest.json")
    (destination / "benchmark_selection.lock").write_text(
        f"schema_version={BENCHMARK_SCHEMA_VERSION}\nmanifest={payload['manifest_hash']}\n"
        "selection_is_frozen=true\nresults_must_not_change_selection=true\n",
        encoding="utf-8",
    )
    return payload
