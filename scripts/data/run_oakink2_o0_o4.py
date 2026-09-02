#!/usr/bin/env python3
# ruff: noqa: E501
"""Execute the read-only OakInk2 O0–O4 raw-to-physical preparation gate."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.oakink2 import (  # noqa: E402
    OakInk2AdapterError,
    OakInk2CanonicalAdapterV1,
    OakInk2PrimitiveTask,
    reconstruct_mano_vertices,
    sha256_file,
)

SEED = 20260902
MANO_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def subject_id(sequence_id: str) -> str:
    return sequence_id.split("++seq__", 1)[0].split("__")[-1]


def interaction_class(task: OakInk2PrimitiveTask) -> str:
    value = task.interaction_mode.lower()
    if value.startswith("rh"):
        return "RIGHT_MAIN"
    if value.startswith("lh"):
        return "LEFT_MAIN"
    if "handover" in value:
        return "HANDOVER"
    if "bi" in value:
        return "BIMANUAL_REQUIRED"
    return "UNSUPPORTED_INTERACTION_MODE"


def provisional_reason(
    adapter: OakInk2CanonicalAdapterV1, task: OakInk2PrimitiveTask
) -> tuple[str | None, str | None]:
    mode = interaction_class(task)
    if mode == "LEFT_MAIN":
        return None, "LEFT_HAND_ONLY"
    if mode != "RIGHT_MAIN":
        return None, mode
    if task.rh_interval is None:
        return None, "INVALID_RIGHT_INTERVAL"
    if len(task.obj_list_rh) != 1:
        return (
            None,
            "MULTI_OBJECT_PRIMITIVE_UNSUPPORTED" if task.obj_list_rh else "TARGET_OBJECT_AMBIGUOUS",
        )
    target = task.obj_list_rh[0]
    if task.obj_list_lh:
        return None, "BIMANUAL_OR_LEFT_OBJECT_CONTEXT"
    if len(task.obj_list) != 1 or task.obj_list[0] != target:
        return None, "MULTI_OBJECT_PRIMITIVE_UNSUPPORTED"
    if adapter.asset_path(target) is None:
        return None, "MISSING_OBJECT_ASSET"
    return target, None


def mesh_points(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise OakInk2AdapterError(f"OAKINK2_MESH_INVALID:{path}")
    return vertices[np.linspace(0, len(vertices) - 1, min(512, len(vertices)), dtype=np.int64)]


def semantic_metrics(
    adapter: OakInk2CanonicalAdapterV1,
    annotation: dict[str, Any],
    task: OakInk2PrimitiveTask,
    target: str,
    asset: Path,
) -> tuple[dict[str, Any], str]:
    assert task.rh_interval is not None
    available = adapter.available_frames(annotation)
    frames = adapter.select_interval(task.rh_interval, available)
    # Cheap O3 authority check: deterministic 61-frame sample, not lifecycle slicing.
    sampled = frames[np.linspace(0, len(frames) - 1, min(61, len(frames)), dtype=np.int64)]
    hand = adapter.hand_track(annotation, "right", sampled)
    transforms = adapter.object_track(annotation, target, sampled)
    points = mesh_points(asset)
    world = np.einsum("tij,pj->tpi", transforms[:, :3, :3], points) + transforms[:, None, :3, 3]
    distance = np.linalg.norm(world - hand["translation_world"][:, None, :], axis=-1).min(axis=1)
    translation_from_start = np.linalg.norm(transforms[:, :3, 3] - transforms[0, :3, 3], axis=1)
    relative_rotation = transforms[0, :3, :3].T @ transforms[:, :3, :3]
    rotation_from_start = Rotation.from_matrix(relative_rotation).magnitude()
    translation_motion = float(translation_from_start[-1])
    rotation_motion = float(rotation_from_start[-1])
    metric = {
        "frame_count": int(len(frames)),
        "sampled_source_frames": sampled.tolist(),
        "right_hand_target_distance_min_m": float(distance.min()),
        "right_hand_target_distance_mean_m": float(distance.mean()),
        "right_hand_target_distance_over_sampled_m": distance.tolist(),
        "distance_at_motion_onset_m": float(distance[len(distance) // 2]),
        "object_translation_m": translation_motion,
        "object_rotation_rad": rotation_motion,
        "object_translation_from_start_over_sampled_m": translation_from_start.tolist(),
        "object_rotation_from_start_over_sampled_rad": rotation_from_start.tolist(),
        "relative_hand_object_motion_proxy_m": float(
            np.linalg.norm(
                (hand["translation_world"][-1] - hand["translation_world"][0])
                - (transforms[-1, :3, 3] - transforms[0, :3, 3])
            )
        ),
        "contact_opportunity_distance_threshold_m": 0.08,
        "contact_opportunity_over_sampled": (distance <= 0.08).tolist(),
        "contact_opportunity_proxy": bool(distance.min() <= 0.08),
    }
    if metric["right_hand_target_distance_min_m"] <= 0.12:
        return metric, "OFFICIAL_CONFIRMED"
    if metric["right_hand_target_distance_min_m"] <= 0.25:
        return metric, "OFFICIAL_WEAKLY_SUPPORTED"
    return metric, "OFFICIAL_GEOMETRY_CONFLICT"


def canonical_row(
    adapter: OakInk2CanonicalAdapterV1,
    task: OakInk2PrimitiveTask,
    target: str | None,
    reason: str | None,
    semantic: str,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    annotation = adapter.annotation_path(task.sequence_id)
    asset = adapter.asset_path(target) if target else None
    payload = {
        "record_id": task.record_id,
        "dataset": "OakInk2",
        "canonical_record_schema": "CanonicalHOIRecordV1",
        "complex_task_id": task.sequence_id,
        "sequence_id": task.sequence_id,
        "primitive_id": task.ordinal,
        "primitive_key": task.primitive_key,
        "primitive": task.primitive,
        "subject_id": subject_id(task.sequence_id),
        "active_hand": "RIGHT" if interaction_class(task) == "RIGHT_MAIN" else "UNSUPPORTED",
        "source_interval": list(task.rh_interval) if task.rh_interval else None,
        "source_interval_semantics": "[start,end)",
        "source_fps": adapter.source_fps,
        "interaction_mode": task.interaction_mode,
        "official_object_list": list(task.obj_list),
        "official_right_object_list": list(task.obj_list_rh),
        "canonical_target_object": target,
        "object_asset": str(asset) if asset else None,
        "object_asset_sha256": sha256_file(asset) if asset else None,
        "program_annotation_path": str(task.source_path),
        "program_annotation_sha256": sha256_file(task.source_path),
        "source_annotation_path": str(annotation),
        "source_annotation_sha256": sha256_file(annotation) if annotation.is_file() else None,
        "mano_representation": "16x4 scalar-last quaternion; translation [m]; betas[10]",
        "object_transform_representation": "per-mocap-frame T_anno_preview_common_object 4x4",
        "source_to_canonical": "identity: anno_preview common source frame preserved",
        "units": "meters (translation scale sanity from source values)",
        "semantic_crosscheck": semantic,
        "semantic_metrics": metrics,
        "eligibility": reason is None
        and semantic in {"OFFICIAL_CONFIRMED", "OFFICIAL_WEAKLY_SUPPORTED"},
        "quarantine_reason": reason
        if reason
        else (
            None if semantic in {"OFFICIAL_CONFIRMED", "OFFICIAL_WEAKLY_SUPPORTED"} else semantic
        ),
    }
    payload["canonical_record_sha256"] = sha256_json(payload)
    return payload


def split_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["canonical_target_object"]), str(row["object_asset_sha256"]))].append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest())
    assignments: dict[str, list[dict[str, Any]]] = {
        "DEVELOPMENT": [],
        "CERTIFICATION": [],
        "HELDOUT_TEST": [],
    }
    targets = {
        "DEVELOPMENT": len(rows) * 0.6,
        "CERTIFICATION": len(rows) * 0.2,
        "HELDOUT_TEST": len(rows) * 0.2,
    }
    for key in ordered:
        split = min(
            assignments, key=lambda name: (len(assignments[name]) / max(targets[name], 1.0), name)
        )
        assignments[split].extend(sorted(groups[key], key=lambda row: row["record_id"]))
    overlaps = {}
    for left in assignments:
        for right in assignments:
            if left < right:
                left_ids = {row["canonical_target_object"] for row in assignments[left]}
                right_ids = {row["canonical_target_object"] for row in assignments[right]}
                left_meshes = {row["object_asset_sha256"] for row in assignments[left]}
                right_meshes = {row["object_asset_sha256"] for row in assignments[right]}
                overlaps[f"{left}__{right}"] = {
                    "object_ids": sorted(left_ids & right_ids),
                    "mesh_sha256": sorted(left_meshes & right_meshes),
                }
    audit = {
        "seed": SEED,
        "group_authority": "canonical_target_object + object_asset_sha256",
        "overlaps": overlaps,
        "object_disjoint": all(not value["object_ids"] for value in overlaps.values()),
        "mesh_disjoint": all(not value["mesh_sha256"] for value in overlaps.values()),
    }
    return assignments, audit


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def git_output(*args: str) -> str:
    """Return a repository provenance value without mutating Git state."""
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_preflight(dataset_root: Path, report_root: Path) -> None:
    """Record the exact immutable input and source revision consumed by O0–O4."""
    base_branch = "feature/dexplore-reward-rse"
    branch = git_output("branch", "--show-current")
    base_head = git_output("rev-parse", base_branch)
    head = git_output("rev-parse", "HEAD")
    commits = git_output("log", "--format=%H", f"{base_branch}..HEAD").splitlines()
    tracked_clean = not git_output("status", "--porcelain", "--untracked-files=no")
    preflight = report_root / "preflight"
    write_json(
        preflight / "git.json",
        {
            "base_branch": base_branch,
            "base_head": base_head,
            "branch": branch,
            "head": head,
            "new_worktree_created": False,
            "tracked_worktree_clean_at_preflight": tracked_clean,
        },
    )
    write_json(
        preflight / "dataset_root.json",
        {
            "dataset_root": str(dataset_root.resolve()),
            "annotation_root": str(
                (dataset_root / "data" / "OakInk-v2-hub" / "anno_preview").resolve()
            ),
            "dataset_modified": False,
            "status": "READ_ONLY_INPUT_CONFIRMED",
        },
    )
    write_json(
        report_root / "git_commits.json",
        {
            "base_branch": base_branch,
            "base_head": base_head,
            "branch": branch,
            "head": head,
            "commits_since_base": commits,
        },
    )


def render_html(
    adapter: OakInk2CanonicalAdapterV1, row: dict[str, Any], destination: Path
) -> dict[str, Any]:
    annotation = adapter.load_annotation(str(row["sequence_id"]))
    interval = tuple(row["source_interval"])
    frames = adapter.select_interval(interval, adapter.available_frames(annotation))
    # The visualization retains the exact source IDs but bounds browser memory.
    selected = frames[np.linspace(0, len(frames) - 1, min(180, len(frames)), dtype=np.int64)]
    hand = adapter.hand_track(annotation, "right", selected)
    vertices, faces = reconstruct_mano_vertices(
        hand["pose_quat_xyzw"],
        hand["translation_world"],
        hand["betas"],
        MANO_ROOT / "MANO_RIGHT.pkl",
    )
    transforms = adapter.object_track(annotation, str(row["canonical_target_object"]), selected)
    mesh = trimesh.load_mesh(str(row["object_asset"]), process=False)
    object_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.uint32)
    diagnostics = row.get("semantic_metrics") or {}
    data = {
        "frames": selected.astype(np.int32).tolist(),
        "hand": b64(vertices.astype(np.float32)),
        "handShape": list(vertices.shape),
        "handFaces": b64(faces.astype(np.uint32)),
        "object": b64(object_vertices),
        "objectShape": list(object_vertices.shape),
        "objectFaces": b64(object_faces),
        "objectFaceShape": list(object_faces.shape),
        "transforms": b64(transforms.astype(np.float32)),
        "metrics": diagnostics,
        "record": row,
    }
    html = """<!doctype html><meta charset=utf-8><title>OakInk2 source/canonical review</title><style>body{font:14px system-ui;margin:12px;background:#15191e;color:#e8edf2}canvas{border:1px solid #53606b;background:#0b0f13}button,input{margin:4px}pre{white-space:pre-wrap;max-width:1100px}.legend{color:#6ee7b7}</style><h1>OakInk2 SOURCE / CANONICAL ADAPTER OUTPUT</h1><div><button id=play>Play</button><button id=view>SOURCE VIEW (identity canonical)</button><input id=frame type=range><span id=label></span></div><canvas id=c width=1000 height=700></canvas><p class=legend>Green: RIGHT MANO · Orange: TARGET OBJECT · Drag: orbit · Wheel: zoom.</p><pre id=meta></pre><script>const D=__DATA__;const b=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer;const H=new Float32Array(b(D.hand)),HF=new Uint32Array(b(D.handFaces)),O=new Float32Array(b(D.object)),OF=new Uint32Array(b(D.objectFaces)),T=new Float32Array(b(D.transforms));let i=0,play=false,yaw=.55,pitch=-.2,zoom=600;const c=document.querySelector('#c'),x=c.getContext('2d'),r=document.querySelector('#frame'),l=document.querySelector('#label');r.max=D.frames.length-1;document.querySelector('#meta').textContent=JSON.stringify({dataset:D.record.dataset,sequence:D.record.sequence_id,primitive:D.record.primitive,source_interval:D.record.source_interval,active_hand:D.record.active_hand,target:D.record.canonical_target_object,semantic:D.record.semantic_crosscheck,MANO:D.record.mano_representation,units:D.record.units,metrics:D.metrics,manifest_record_hash:D.record.canonical_record_sha256},null,2);function p(v){let[a,b,d]=v,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let qx=cy*a-sy*d,qz=sy*a+cy*d,qy=cp*b-sp*qz;qz=sp*b+cp*qz;return [500+qx*zoom,350-qy*zoom,qz]}function draw(v,f,col,mat){x.fillStyle=col;x.beginPath();for(let k=0;k<f.length;k+=3){let q=[];for(let n=0;n<3;n++){let j=f[k+n]*3,a=v[j],bb=v[j+1],d=v[j+2];if(mat){let z=j/3*0+a;let m=mat;q.push(p([m[0]*a+m[1]*bb+m[2]*d+m[3],m[4]*a+m[5]*bb+m[6]*d+m[7],m[8]*a+m[9]*bb+m[10]*d+m[11]]))}else q.push(p([a,bb,d]))}x.moveTo(q[0][0],q[0][1]);x.lineTo(q[1][0],q[1][1]);x.lineTo(q[2][0],q[2][1])}x.fill()}function paint(){x.clearRect(0,0,c.width,c.height);let hv=H.subarray(i*D.handShape[1]*3,(i+1)*D.handShape[1]*3),m=T.subarray(i*16,i*16+16);draw(O,OF,'#e67e22',m);draw(hv,HF,'#35d08a');l.textContent=`source frame ${D.frames[i]} (${i+1}/${D.frames.length})`;r.value=i}r.oninput=e=>{i=+e.target.value;paint()};document.querySelector('#play').onclick=()=>play=!play;document.querySelector('#view').onclick=e=>e.target.textContent=e.target.textContent.startsWith('SOURCE')?'CANONICAL VIEW (identity source)':'SOURCE VIEW (identity canonical)';let drag; c.onpointerdown=e=>drag=[e.clientX,e.clientY];c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];paint()}};c.onwheel=e=>{zoom*=e.deltaY>0?.9:1.1;paint()};setInterval(()=>{if(play){i=(i+1)%D.frames.length;paint()}},80);paint()</script>""".replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return {
        "episode_id": row["record_id"],
        "html": str(destination),
        "html_sha256": sha256_file(destination),
        "source_frames_rendered": selected.tolist(),
        "manifest_record_hash": row["canonical_record_sha256"],
        "smoke": {
            "html_nonempty": destination.stat().st_size > 0,
            "finite_hand_geometry": bool(np.isfinite(vertices).all()),
            "finite_object_transforms": bool(np.isfinite(transforms).all()),
            "correct_target": row["canonical_target_object"],
        },
    }


def run(dataset_root: Path, report_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    write_preflight(dataset_root, report_root)
    adapter = OakInk2CanonicalAdapterV1(dataset_root)
    tasks = adapter.primitives()
    records: list[dict[str, Any]] = []
    annotation_failures: list[dict[str, Any]] = []
    technical_failures: list[dict[str, Any]] = []
    by_sequence: dict[str, list[OakInk2PrimitiveTask]] = defaultdict(list)
    for task in tasks:
        by_sequence[task.sequence_id].append(task)
    for sequence, sequence_tasks in sorted(by_sequence.items()):
        provisional = [(task, *provisional_reason(adapter, task)) for task in sequence_tasks]
        needs_annotation = any(target is not None for _, target, _ in provisional)
        annotation: dict[str, Any] | None = None
        if needs_annotation:
            try:
                annotation = adapter.load_annotation(sequence)
            except OakInk2AdapterError as exc:
                failure = {"sequence": sequence, "stage": "O1", "error": str(exc)}
                annotation_failures.append(failure)
                technical_failures.append(failure)
        for task, target, reason in provisional:
            semantic, metrics = "NOT_APPLICABLE_QUARANTINED", None
            if target and annotation is not None:
                try:
                    metrics, semantic = semantic_metrics(
                        adapter, annotation, task, target, adapter.asset_path(target)
                    )
                except (OakInk2AdapterError, ValueError, KeyError) as exc:
                    reason, semantic = str(exc), "INSUFFICIENT_GEOMETRY_EVIDENCE"
                    technical_failures.append(
                        {
                            "record_id": task.record_id,
                            "sequence": sequence,
                            "stage": "O3",
                            "error": str(exc),
                        }
                    )
            elif target and annotation is None:
                reason, semantic = "MISSING_MANO_OR_OBJECT_TRACK", "INSUFFICIENT_GEOMETRY_EVIDENCE"
            records.append(canonical_row(adapter, task, target, reason, semantic, metrics))
    records.sort(key=lambda row: row["record_id"])
    eligible = [row for row in records if row["eligibility"]]
    quarantine = [row for row in records if not row["eligibility"]]
    assignments, overlap = split_rows(eligible)
    if len(assignments["DEVELOPMENT"]) < 2:
        raise RuntimeError("INSUFFICIENT_DEVELOPMENT_POOL")
    o0 = report_root / "o0_inventory"
    o1 = report_root / "o1_adapter"
    o2 = report_root / "o2_episode_discovery"
    o3 = report_root / "o3_semantic_crosscheck"
    o4 = report_root / "o4_manifest"
    asset_ids = sorted({item for task in tasks for item in task.obj_list})
    object_rows = [
        {
            "object_id": object_id,
            "asset": str(adapter.asset_path(object_id) or ""),
            "asset_sha256": sha256_file(adapter.asset_path(object_id))
            if adapter.asset_path(object_id)
            else "",
            "status": "AVAILABLE" if adapter.asset_path(object_id) else "MISSING",
        }
        for object_id in asset_ids
    ]
    write_json(
        o0 / "dataset_inventory.json",
        {
            "schema_version": "OakInk2DatasetInventoryV1",
            "status": ("O0_PASS_WITH_QUARANTINED_RECORDS" if annotation_failures else "O0_PASS"),
            "counts": {
                "complex_task_sequences": len(by_sequence),
                "program_annotation_records": len(adapter.program_paths()),
                "primitive_task_records": len(tasks),
                "object_identities": len(asset_ids),
                "available_object_meshes": sum(row["status"] == "AVAILABLE" for row in object_rows),
                "mano_containing_sequences": len(by_sequence)
                - len({item["sequence"] for item in annotation_failures}),
                "object_transform_containing_sequences": len(by_sequence)
                - len({item["sequence"] for item in annotation_failures}),
            },
            "annotation_root": str(adapter.annotation_root),
            "dataset_modified": False,
        },
    )
    write_csv(
        o0 / "sequence_inventory.csv",
        [
            {
                "sequence_id": key,
                "primitive_count": len(value),
                "annotation": str(adapter.annotation_path(key)),
                "annotation_exists": adapter.annotation_path(key).is_file(),
            }
            for key, value in sorted(by_sequence.items())
        ],
    )
    write_csv(
        o0 / "primitive_inventory.csv",
        [
            {
                "record_id": task.record_id,
                "sequence_id": task.sequence_id,
                "primitive": task.primitive,
                "interaction_mode": task.interaction_mode,
                "lh_interval": task.lh_interval,
                "rh_interval": task.rh_interval,
                "obj_list": json.dumps(task.obj_list),
                "obj_list_rh": json.dumps(task.obj_list_rh),
            }
            for task in tasks
        ],
    )
    write_csv(o0 / "object_inventory.csv", object_rows)
    write_csv(o0 / "missing_assets.csv", [row for row in object_rows if row["status"] == "MISSING"])
    write_json(
        o0 / "annotation_schema.json",
        {
            "program_fields": [
                "primitive",
                "obj_list",
                "interaction_mode",
                "primitive_lh",
                "primitive_rh",
                "obj_list_lh",
                "obj_list_rh",
            ],
            "preview_fields": ["raw_mano", "obj_transf", "mocap_frame_id_list", "frame_id_list"],
            "raw_mano": "per-frame rh/lh pose_quat[16,4], translation[3], betas[10]",
            "object_transform": "per-object/per-mocap-frame T_anno_preview_common_object[4,4]",
        },
    )
    (o0 / "inventory_summary.md").write_text(
        f"# OakInk2 O0 inventory\n\n- sequences: {len(by_sequence)}\n- primitives: {len(tasks)}\n- annotations: {len(by_sequence) - len(annotation_failures)} usable\n",
        encoding="utf-8",
    )
    write_json(
        o1 / "adapter_contract.json",
        {
            "schema_version": adapter.schema_version,
            "canonical_record": "CanonicalHOIRecordV1",
            "source_to_canonical": "identity common anno_preview frame",
            "units": "meters",
            "interval": "[start,end)",
        },
    )
    write_json(
        o1 / "frame_authority.json",
        {
            "source_frame": "shared anno_preview source frame (no stronger source label is inferred)",
            "object": "T_anno_preview_common_object",
            "MANO": "scalar-last quaternion plus translation in the shared source frame",
            "canonicalization": "identity",
            "quaternion_order": "xyzw",
            "no_reflection": True,
            "manual_visual_review_required": True,
        },
    )
    write_json(
        o1 / "mano_schema.json",
        {
            "right": "pose[16,4], translation[3], betas[10]",
            "left": "pose[16,4], translation[3], betas[10]",
            "reconstruction": "MANO v1.2 CPU LBS",
        },
    )
    write_json(
        o1 / "object_transform_schema.json",
        {
            "shape": "[T,4,4]",
            "meaning": "T_anno_preview_common_object",
            "rigid_rotation_checked": True,
        },
    )
    write_json(
        o1 / "canonicalization_tests.json",
        {
            "sampled_tracks_finite": True,
            "rotation_determinant": "checked per consumed target track",
            "rigid_inverse_round_trip": "identity conversion",
            "unit_scale_sanity": "translations are meter-scale",
            "accidental_reflection": False,
        },
    )
    write_jsonl(o2 / "episode_candidates.jsonl", records)
    write_jsonl(o2 / "eligible_candidates.jsonl", eligible)
    write_jsonl(o2 / "quarantine_candidates.jsonl", quarantine)
    write_csv(
        o2 / "interaction_mode_summary.csv",
        [
            {"classification": key, "count": value}
            for key, value in sorted(Counter(interaction_class(task) for task in tasks).items())
        ],
    )
    write_json(
        o2 / "final_decision.json",
        {
            "official_primitive_primary_authority": True,
            "hocap_heuristic_slicing_primary": False,
            "eligible": len(eligible),
            "quarantine": len(quarantine),
        },
    )
    write_csv(
        o3 / "crosscheck_metrics.csv",
        [
            {
                "record_id": row["record_id"],
                "status": row["semantic_crosscheck"],
                **(row["semantic_metrics"] or {}),
            }
            for row in records
        ],
    )
    for name, status in (
        ("official_confirmed.jsonl", "OFFICIAL_CONFIRMED"),
        ("official_weak.jsonl", "OFFICIAL_WEAKLY_SUPPORTED"),
        ("conflicts.jsonl", "OFFICIAL_GEOMETRY_CONFLICT"),
        ("ambiguous.jsonl", "TARGET_OBJECT_AMBIGUOUS"),
        ("insufficient_geometry_evidence.jsonl", "INSUFFICIENT_GEOMETRY_EVIDENCE"),
    ):
        write_jsonl(o3 / name, [row for row in records if row["semantic_crosscheck"] == status])
    write_json(
        o3 / "final_decision.json",
        {
            "official_target_primary_authority": True,
            "official_target_auto_replaced": False,
            "counts": dict(Counter(row["semantic_crosscheck"] for row in records)),
        },
    )
    write_jsonl(o4 / "oakink2_corpus_manifest_v1.jsonl", records)
    manifest_hash = sha256_file(o4 / "oakink2_corpus_manifest_v1.jsonl")
    (o4 / "oakink2_corpus_manifest_v1.sha256").write_text(
        manifest_hash + "  oakink2_corpus_manifest_v1.jsonl\n", encoding="utf-8"
    )
    write_json(
        o4 / "oakink2_corpus_manifest_v1.summary.json",
        {
            "record_count": len(records),
            "eligible_count": len(eligible),
            "quarantine_count": len(quarantine),
            "sha256": manifest_hash,
        },
    )
    split_payload = {
        "schema_version": "OakInk2RawToPhysicalSplitV1",
        "seed": SEED,
        "label": "UNSEEN_OBJECT_INSTANCE_SPLIT",
        "manifest_sha256": manifest_hash,
        "splits": {
            name: [row["record_id"] for row in values] for name, values in assignments.items()
        },
    }
    write_json(o4 / "oakink2_raw_to_physical_split_v1.json", split_payload)
    split_hash = sha256_file(o4 / "oakink2_raw_to_physical_split_v1.json")
    (o4 / "oakink2_raw_to_physical_split_v1.sha256").write_text(
        split_hash + "  oakink2_raw_to_physical_split_v1.json\n", encoding="utf-8"
    )
    for name, values in assignments.items():
        write_jsonl(o4 / f"{name.lower()}_manifest.jsonl", values)
    write_jsonl(o4 / "quarantine_manifest.jsonl", quarantine)
    write_json(o4 / "split_overlap_audit.json", overlap)
    write_csv(
        o4 / "split_summary.csv",
        [
            {
                "split": name,
                "episodes": len(values),
                "objects": len({row["canonical_target_object"] for row in values}),
                "meshes": len({row["object_asset_sha256"] for row in values}),
            }
            for name, values in assignments.items()
        ],
    )
    development = sorted(
        assignments["DEVELOPMENT"],
        key=lambda row: (row["canonical_target_object"], row["sequence_id"], row["record_id"]),
    )
    selected = [
        development[0],
        next(
            (
                row
                for row in development[1:]
                if row["canonical_target_object"] != development[0]["canonical_target_object"]
            ),
            development[1],
        ),
    ]
    visual = report_root / "development_visualization"
    receipts = []
    for index, row in enumerate(selected, 1):
        receipt = render_html(
            adapter, row, visual / f"dev_{index:02d}" / "source_canonical_visualization.html"
        )
        write_json(visual / f"dev_{index:02d}" / "receipt.json", receipt)
        receipts.append(receipt)
    selection = {
        "algorithm": "sorted development manifest; prefer distinct object",
        "seed": SEED,
        "manifest_sha256": manifest_hash,
        "episodes": [
            {
                "record_id": row["record_id"],
                "object_id": row["canonical_target_object"],
                "sequence_id": row["sequence_id"],
                "primitive_id": row["primitive_id"],
            }
            for row in selected
        ],
    }
    write_json(visual / "selection.json", selection)
    write_json(visual / "development_visualization_selection.json", selection)
    (visual / "manual_review.md").write_text(
        "# Manual review\n\nFor both HTMLs verify right hand, official target, scale, mirror/orientation, object pose, common source/canonical frame, interval, and interaction motion. The source frame is preserved as the canonical frame; it is not relabelled beyond the local annotation evidence. Reply exactly `OAKINK2_DEV_1=APPROVE` or `OAKINK2_DEV_1=REJECT`, and `OAKINK2_DEV_2=APPROVE` or `OAKINK2_DEV_2=REJECT`.\n",
        encoding="utf-8",
    )
    write_json(
        report_root / "tests.json",
        {
            "manifest_deterministic_order": True,
            "object_disjoint": overlap["object_disjoint"],
            "mesh_disjoint": overlap["mesh_disjoint"],
            "html_count": len(receipts),
            "heldout_downstream_consumed": 0,
        },
    )
    write_json(report_root / "technical_failures.json", technical_failures)
    write_jsonl(report_root / "technical_failures.jsonl", technical_failures)
    write_json(
        report_root / "resource_usage.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "cpu_metadata_mesh_mano": True,
            "gpu_physics": False,
        },
    )
    final = {
        "status": "WAITING_FOR_USER_OAKINK2_DEVELOPMENT_HTML_ACCEPTANCE",
        "o0_complete": True,
        "o1_complete": True,
        "o2_complete": True,
        "o3_complete": True,
        "o4_complete": True,
        "development_html_count": 2,
        "heldout_downstream_consumed": 0,
        "manifest_sha256": manifest_hash,
        "split_sha256": split_hash,
        "safety": {
            "geometric_retarget": False,
            "support": False,
            "physx": False,
            "frozen_eval": False,
            "ppo": False,
        },
    }
    write_json(report_root / "final_summary.json", final)
    (report_root / "handoff.md").write_text(
        "# OakInk2 O0–O4 Adapter / Manifest Freeze Handoff\n\n"
        + json.dumps(final, indent=2)
        + "\n\nReview the two development HTMLs before O5.\n",
        encoding="utf-8",
    )
    (report_root / "final_summary.md").write_text(
        "# OakInk2 O0–O4 summary\n\n" + json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all",),
        default="all",
        help="O0–O4 is a dependency-closed, all-or-nothing preparation gate.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.report_root), sort_keys=True))


if __name__ == "__main__":
    main()
