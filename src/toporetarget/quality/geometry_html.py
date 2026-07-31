"""Self-contained G3 geometry audit HTML and frame snapshots."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.readers.grab import load_grab_auxiliary, load_ply_mesh
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import invert_transform, transform_points
from toporetarget.geometry.signed_distance.closest_point import (
    TriangleAABBTree,
    closest_points_on_triangles,
)
from toporetarget.geometry.signed_distance.derived_proxy import (
    _surface_samples,
    build_hybrid_signed_distance_backend,
)

from .geometry import FINGER_TIPS, _finger_name, _mapping_names
from .schema import QUALITY_SCHEMA_VERSION, ClipSpec, write_json


def _sample_points(points: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if len(value) <= count:
        return value
    ids = np.linspace(0, len(value) - 1, count, dtype=np.int64)
    return value[ids]


def _rounded(value: Any, digits: int = 7) -> Any:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return np.round(array, digits).tolist()
    return array.tolist()


def _contact_context(
    canonical_path: str | Path, source_path: str | Path, clip: ClipSpec, object_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sequence = load_hoi_sequence(canonical_path)
    hand = next(item for item in sequence.hands if item.side == clip.hand)
    keypoints_scene = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)
    keypoints_local = transform_points(
        np.asarray([invert_transform(pose) for pose in object_pose]), keypoints_scene
    )
    raw = load_grab_auxiliary(
        source_path,
        frame_range=FrameRange(clip.start_frame, clip.end_frame),
        include_table=False,
        contact_mode="semantic",
    )
    labels = np.asarray(raw["contact"]["object"], dtype=np.int64)
    names = _mapping_names()
    active_points: list[np.ndarray] = []
    for frame in range(min(len(labels), len(keypoints_local))):
        active: dict[str, bool] = {finger: False for finger in FINGER_TIPS}
        for label in np.unique(labels[frame]):
            finger = _finger_name(names.get(int(label), ""))
            if finger is not None and int(label) != 0:
                active[finger] = True
        active_points.append(
            keypoints_local[frame, [FINGER_TIPS[f] for f, value in active.items() if value]]
        )
    return keypoints_local, (
        np.concatenate(active_points, axis=0) if active_points else np.empty((0, 3))
    )


def render_geometry_audit_html(
    *,
    geometry_manifest: dict[str, Any],
    canonical_path: str | Path,
    source_path: str | Path,
    clip: ClipSpec,
    output: str | Path,
) -> dict[str, Any]:
    row = next(item for item in geometry_manifest["rows"] if item["unit_id"] == clip.unit_id)
    artifact_root = Path(str(row["artifact_root"]))
    source_vertices, source_faces = load_ply_mesh(row["object_mesh_path"])
    proxy_data = np.load(artifact_root / "proxy_mesh.npz")
    proxy_vertices = np.asarray(proxy_data["vertices"], dtype=np.float64)
    proxy_faces = np.asarray(proxy_data["faces"], dtype=np.int64)
    backend, geometry = build_hybrid_signed_distance_backend(
        source_vertices,
        source_faces,
        source_path=row["object_mesh_path"],
    )
    source_sample = _surface_samples(
        source_vertices,
        source_faces[
            np.all(source_faces >= 0, axis=1) & np.all(source_faces < len(source_vertices), axis=1)
        ],
        2000,
        geometry.policy.surface_sample_seed,
    )
    proxy_sample = _sample_points(proxy_vertices, 2000)
    source_tree = TriangleAABBTree(proxy_vertices[proxy_faces])
    source_to_proxy = closest_points_on_triangles(
        source_sample,
        proxy_vertices[proxy_faces],
        tree=source_tree,
        query_chunk_size=256,
    )[3]
    object_track = load_hoi_sequence(canonical_path).primary_rigid_object()
    poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    keypoints_local, active_contact_points = _contact_context(
        canonical_path, source_path, clip, poses
    )
    synthetic_ids = np.asarray(geometry.synthetic_face_ids, dtype=np.int64)
    patch_triangles = (
        proxy_vertices[proxy_faces[synthetic_ids]]
        if len(synthetic_ids)
        else np.empty((0, 3, 3), dtype=np.float64)
    )
    removed_ids = np.asarray(geometry.near_zero_face_ids, dtype=np.int64)
    removed_centers = (
        source_vertices[source_faces[removed_ids]].mean(axis=1)
        if len(removed_ids)
        else np.empty((0, 3), dtype=np.float64)
    )
    boundary_segments = np.asarray(
        [
            [geometry.source_distance_vertices[a], geometry.source_distance_vertices[b]]
            for a, b in geometry.boundary_edges
        ],
        dtype=np.float64,
    )
    bbox_min = np.min(source_vertices, axis=0)
    bbox_max = np.max(source_vertices, axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    probes = np.asarray(
        [
            (bbox_min + bbox_max) * 0.5,
            bbox_max + np.asarray([diagonal, 0.0, 0.0]),
            bbox_min - np.asarray([diagonal, 0.0, 0.0]),
        ],
        dtype=np.float64,
    )
    probe_result = backend.query_local(probes)
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "geometry_schema_version": geometry.policy.schema_version,
        "profile_id": geometry.policy.profile_id,
        "unit_id": clip.unit_id,
        "object_name": clip.object_name,
        "semantics": {
            "original_mesh": "visualization, object samples, closest point, unsigned magnitude, provenance",
            "proxy_mesh": "sign classification only",
            "signed_distance": "sign(proxy) * unsigned_distance(original_mesh)",
            "convex_hull_accepted": False,
            "paper_unspecified_geometry_engineering": True,
        },
        "source_mesh": {
            "vertices": _rounded(_sample_points(source_vertices, 2000)),
            "vertex_count": int(len(source_vertices)),
            "face_count": int(len(source_faces)),
            "mesh_hash": geometry.source_mesh_hash,
        },
        "proxy_mesh": {
            "vertices": _rounded(proxy_sample),
            "vertex_count": int(len(proxy_vertices)),
            "face_count": int(len(proxy_faces)),
            "mesh_hash": geometry.proxy_mesh_hash,
            "candidate_id": geometry.candidate_id,
        },
        "boundary_edges": _rounded(boundary_segments),
        "boundary_loops": geometry.boundary_loops,
        "boundary_loop_positions": _rounded(
            [
                geometry.source_distance_vertices[np.asarray(loop, dtype=np.int64)]
                for loop in geometry.boundary_loops
            ]
        ),
        "synthetic_patch_faces": _rounded(patch_triangles),
        "synthetic_patch_face_ids": synthetic_ids.tolist(),
        "near_zero_removed_face_ids": removed_ids.tolist(),
        "near_zero_removed_face_centers": _rounded(removed_centers),
        "original_to_proxy_deviation": {
            "points": _rounded(source_sample),
            "distance_m": _rounded(source_to_proxy),
        },
        "source_semantic_contact_regions": {
            "active_points_local": _rounded(active_contact_points),
            "selected60_hand_keypoints_local": _rounded(keypoints_local),
        },
        "selected60_frame_context": {
            "object_pose_scene": _rounded(poses, 8),
            "frame_range": [clip.start_frame, clip.end_frame],
        },
        "boundary_exclusion_zone": {
            "radius_m": backend.boundary_exclusion_radius_m,
            "centerline": _rounded(boundary_segments),
        },
        "sign_probe_points": _rounded(probes),
        "sign_probe_results": {
            "signed_distance": _rounded(probe_result.signed_distance),
            "unsigned_distance_original": _rounded(probe_result.unsigned_distance),
            "inside_proxy": _rounded(probe_result.inside),
            "sign_valid": _rounded(probe_result.sign_valid),
        },
        "surface_deviation_summary": geometry.surface_deviation,
        "cache_signature": geometry.cache_signature,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = _HTML_TEMPLATE.replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    destination.write_text(document, encoding="utf-8")
    for name, frame in (("first", 0), ("middle", 30), ("last", 59)):
        _write_snapshot(
            payload, destination.with_name(f"banana_original_vs_proxy_{name}.png"), frame
        )
    write_json(
        {
            "path": str(destination.resolve()),
            "png": [
                str(destination.with_name(f"banana_original_vs_proxy_{name}.png").resolve())
                for name in ("first", "middle", "last")
            ],
            "status": "pass",
        },
        destination.with_name("banana_original_vs_proxy_visualization.json"),
    )
    return {
        "path": str(destination.resolve()),
        "png": [
            str(destination.with_name(f"banana_original_vs_proxy_{name}.png").resolve())
            for name in ("first", "middle", "last")
        ],
        "status": "pass",
    }


def _write_snapshot(payload: dict[str, Any], path: Path, frame: int) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    source = np.asarray(payload["source_mesh"]["vertices"], dtype=np.float64)
    proxy = np.asarray(payload["proxy_mesh"]["vertices"], dtype=np.float64)
    axes[0].scatter(source[:, 0], source[:, 2], s=1, alpha=0.2, label="original mesh")
    axes[0].scatter(proxy[:, 0], proxy[:, 2], s=1, alpha=0.2, label="proxy sign mesh")
    boundary = np.asarray(payload["boundary_edges"], dtype=np.float64)
    for segment in boundary:
        axes[0].plot(segment[:, 0], segment[:, 2], color="red", linewidth=1.0)
    patch = np.asarray(payload["synthetic_patch_faces"], dtype=np.float64)
    for triangle in patch:
        closed = np.vstack((triangle, triangle[0]))
        axes[0].plot(closed[:, 0], closed[:, 2], color="orange")
    axes[0].set_title("original / proxy / boundary / patch")
    axes[0].legend(loc="best", fontsize=8)
    heat_points = np.asarray(payload["original_to_proxy_deviation"]["points"], dtype=np.float64)
    heat_distance = np.asarray(
        payload["original_to_proxy_deviation"]["distance_m"], dtype=np.float64
    )
    scatter = axes[1].scatter(
        heat_points[:, 0], heat_points[:, 2], c=heat_distance, s=2, cmap="turbo"
    )
    figure.colorbar(scatter, ax=axes[1], label="original to proxy deviation (m)")
    context = np.asarray(
        payload["source_semantic_contact_regions"]["selected60_hand_keypoints_local"][frame]
    )
    if len(context):
        axes[1].scatter(context[:, 0], context[:, 2], color="black", s=10, label="hand context")
    axes[1].set_title(f"surface deviation / frame {frame}")
    axes[1].legend(loc="best", fontsize=8)
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def smoke_geometry_html(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    required = (
        "original_mesh",
        "proxy_mesh",
        "boundary_edges",
        "boundary_loops",
        "synthetic_patch_faces",
        "original_to_proxy_deviation",
        "source_semantic_contact_regions",
        "selected60_frame_context",
        "boundary_exclusion_zone",
        "sign_probe_points",
    )
    checks = {token: token in source for token in required}
    return {
        "path": str(Path(path).resolve()),
        "exists": Path(path).is_file(),
        "checks": checks,
        "status": "pass" if all(checks.values()) and Path(path).stat().st_size > 0 else "fail",
    }


_HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>banana original vs proxy geometry audit</title>
<style>body{font:14px system-ui;background:#111827;color:#e5e7eb;margin:0}main{display:grid;grid-template-columns:1fr 320px;gap:12px;padding:12px}canvas{width:100%;height:700px;background:#1f2937;border-radius:8px}.panel{background:#1f2937;padding:12px;border-radius:8px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px}button{padding:6px;margin:2px}</style></head>
<body><main><section><canvas id="scene" width="1100" height="700"></canvas><div class="panel"><button id="first">first</button><button id="middle">middle</button><button id="last">last</button><button id="reset">reset</button></div></section><aside><div class="panel" id="summary"></div><div class="panel">Audit layers: original mesh, proxy sign mesh, boundary edges/loops, synthetic patch faces, near-zero removed faces, original-to-proxy deviation heatmap, source semantic contact regions, selected 60-frame hand/object context, boundary exclusion zone, sign probes. The proxy is sign-only; distance magnitude is from the original mesh.</div></aside></main>
<script>const DATA=__DATA__;let frame=0,zoom=1;const c=document.getElementById('scene'),x=c.getContext('2d');function proj(p){return[c.width/2+p[0]*250*zoom,c.height/2-p[2]*250*zoom]}function cloud(a,col,size){x.fillStyle=col;(a||[]).forEach(p=>{let q=proj(p);x.beginPath();x.arc(q[0],q[1],size,0,Math.PI*2);x.fill()})}function line(a,col,w){x.strokeStyle=col;x.lineWidth=w;(a||[]).forEach(s=>{if(s.length<2)return;x.beginPath();let q=proj(s[0]);x.moveTo(q[0],q[1]);for(let i=1;i<s.length;i++){q=proj(s[i]);x.lineTo(q[0],q[1])}x.stroke()})}function draw(){x.clearRect(0,0,c.width,c.height);cloud(DATA.source_mesh.vertices,'#60a5fa',1);cloud(DATA.proxy_mesh.vertices,'#f59e0b',1);line(DATA.boundary_edges,'#ef4444',2);line(DATA.synthetic_patch_faces,'#fb923c',2);cloud(DATA.source_semantic_contact_regions.active_points_local,'#22c55e',3);let kp=DATA.source_semantic_contact_regions.selected60_hand_keypoints_local[frame]||[];cloud(kp,'#f8fafc',2);let d=DATA.original_to_proxy_deviation;let max=Math.max(...d.distance_m,1e-9);(d.points||[]).forEach((p,i)=>{let q=proj(p);x.fillStyle=`hsl(${240-240*d.distance_m[i]/max},90%,55%)`;x.fillRect(q[0],q[1],2,2)});let pr=DATA.sign_probe_points.map((p,i)=>[p,i]);pr.forEach(([p,i])=>{let q=proj(p);x.fillStyle='#fff';x.fillText((DATA.sign_probe_results.inside_proxy[i]?'inside ':'outside ')+i,q[0]+4,q[1])});document.getElementById('summary').textContent=JSON.stringify({unit:DATA.unit_id,semantics:DATA.semantics,source_mesh:DATA.source_mesh,proxy_mesh:DATA.proxy_mesh,surface_deviation_summary:DATA.surface_deviation_summary,boundary_exclusion_zone:DATA.boundary_exclusion_zone,sign_probe_results:DATA.sign_probe_results,frame},null,2)}document.getElementById('first').onclick=()=>{frame=0;draw()};document.getElementById('middle').onclick=()=>{frame=30;draw()};document.getElementById('last').onclick=()=>{frame=59;draw()};document.getElementById('reset').onclick=()=>{zoom=1;draw()};draw();</script></body></html>"""


__all__ = ["render_geometry_audit_html", "smoke_geometry_html"]
