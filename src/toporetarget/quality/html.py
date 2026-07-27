"""Self-contained quality experiment mesh/proxy viewers and static smoke checks."""

# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.visualization import _primitive_mesh
from toporetarget.workflows.mesh_visualization import _interaction_payload

from .schema import QUALITY_SCHEMA_VERSION, ClipSpec


def _hand(sequence: Any, side: str) -> Any:
    return next(item for item in sequence.hands if item.side == side or item.hand_id == side)


def _rounded(value: Any, digits: int = 6) -> Any:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return np.round(array, decimals=digits).tolist()
    return array.tolist()


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    return value


_MAX_OBJECT_FACES = 6000

_PROFILE_SEMANTICS: dict[str, dict[str, str]] = {
    "source_mano": {
        "role": "reference source",
        "description": "原始 GRAB/MANO source mesh；只作为观测参考，不是机器人求解结果。",
    },
    "paper_warm": {
        "role": "paper-core warm-start",
        "description": "Eq. (1)-(2) relative bone-direction warm-start；没有经过 final refinement。",
    },
    "morphology_seed_only_v1": {
        "role": "faithful initialization extension",
        "description": "Morphology-aware seed-only；只改变初始化候选，最终目标仍保持 paper objective。",
    },
    "scipy_slsqp_active_set_contact_rich_v2": {
        "role": "paper-core final · v2",
        "description": "Full-state temporal final refinement；保留的 paper-consistent interpretation。",
    },
    "scipy_slsqp_active_set_contact_rich_v3_fixed": {
        "role": "paper-core final · v3 fixed",
        "description": "Finger-only temporal regularization plus independent base priors；validated engineering profile。",
    },
    "E0_paper_warm_plus_development_base_final": {
        "role": "2x2 control E0",
        "description": "Paper warm-start + development base final；不使用 C_STAR。",
    },
    "E1_m_star_plus_development_base_final": {
        "role": "2x2 control E1",
        "description": "M_STAR morphology warm-start + development base final；不使用 C_STAR。",
    },
    "E2_paper_warm_plus_C_star": {
        "role": "2x2 contact E2",
        "description": "Paper warm-start + selected contact-preserving C_STAR。",
    },
    "E3_m_star_plus_C_star": {
        "role": "2x2 contact E3",
        "description": "M_STAR morphology warm-start + selected contact-preserving C_STAR。",
    },
}


def _mesh_subset(
    vertices: np.ndarray, faces: np.ndarray, *, max_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic face-preserving preview mesh.

    The viewer must render triangles, but embedding every object triangle for all
    60 frames makes the standalone document unnecessarily large.  Selecting
    faces and remapping only the referenced vertices keeps topology valid while
    retaining a real surface rather than an unrelated point cloud.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0 or len(faces) <= max_faces:
        return vertices, faces
    selected = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
    selected_faces = faces[selected]
    used = np.unique(selected_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[selected_faces]


def _transform_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    return points @ transforms[:, :3, :3].transpose(0, 2, 1) + transforms[:, None, :3, 3]


def _robot_visual_payload(model: Any, qpos: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    """Serialize actual visual primitive topology plus per-frame transforms."""

    qpos = np.asarray(qpos, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if qpos.ndim != 2 or base.shape != (len(qpos), 4, 4):
        raise ValueError(f"robot qpos/base shapes are incompatible: {qpos.shape}, {base.shape}")
    first_instances = model.visual_geometry_instances(qpos[0], base[0])
    parts: list[dict[str, Any]] = []
    for instance_index, first in enumerate(first_instances):
        vertices, faces = _primitive_mesh(first)
        transforms: list[np.ndarray] = []
        for frame in range(len(qpos)):
            instances = model.visual_geometry_instances(qpos[frame], base[frame])
            if len(instances) != len(first_instances):
                raise ValueError("robot visual geometry topology changed across frames")
            transforms.append(
                np.asarray(instances[instance_index].world_transform, dtype=np.float64)
            )
        parts.append(
            {
                "name": f"{first.link_name}:{instance_index}",
                "vertices": _rounded(vertices),
                "faces": np.asarray(faces, dtype=np.int64).tolist(),
                "transforms": _rounded(np.asarray(transforms), digits=8),
            }
        )
    return {"parts": parts}


def _closest_object_points(
    keypoints: np.ndarray, object_vertices: np.ndarray, object_poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find diagnostic nearest object vertices for every robot keypoint/frame."""

    keypoints = np.asarray(keypoints, dtype=np.float64)
    object_world = _transform_points(object_vertices, object_poses)
    closest = np.empty_like(keypoints)
    distances = np.empty(keypoints.shape[:2], dtype=np.float64)
    for frame in range(len(keypoints)):
        delta = object_world[frame][None, :, :] - keypoints[frame, :, None, :]
        distance2 = np.einsum("fvc,fvc->fv", delta, delta)
        nearest = np.argmin(distance2, axis=1)
        closest[frame] = object_world[frame, nearest]
        distances[frame] = np.sqrt(np.maximum(distance2[np.arange(len(nearest)), nearest], 0.0))
    return closest, distances


def _profile_payload(
    path: str | Path,
    *,
    warm: bool,
    model: Any,
    object_vertices: np.ndarray,
    object_poses: np.ndarray,
) -> dict[str, Any]:
    artifact = load_warm_start(path) if warm else load_final_trajectory(path)
    arrays = artifact.arrays
    keypoints = np.asarray(arrays["robot_keypoints_scene"], dtype=np.float64)
    base = np.asarray(arrays["base_pose_scene"], dtype=np.float64)
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    visual_mesh = _robot_visual_payload(model, qpos, base)
    contact_points, contact_distances = _closest_object_points(
        keypoints, object_vertices, object_poses
    )
    metrics: list[dict[str, Any]] = []
    for index in range(len(keypoints)):
        item: dict[str, Any] = {"local_frame": index}
        for name in (
            "e_im",
            "e_bone",
            "max_penetration",
            "solve_time_s",
            "accepted",
            "solver_success",
        ):
            if name in arrays:
                value = np.asarray(arrays[name])[index]
                item[name] = bool(value) if np.asarray(value).dtype == bool else float(value)
        metrics.append(item)
    return {
        "keypoints": _rounded(keypoints),
        "base": _rounded(base, 8),
        "robot_mesh": visual_mesh,
        "contact_points": _rounded(contact_points),
        "contact_distances": _rounded(contact_distances),
        "metrics": metrics,
        "artifact_path": str(Path(path).resolve()),
    }


def _bounds(
    source_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_poses: np.ndarray,
    profiles: dict[str, Any],
) -> list[list[float]]:
    chunks = [np.asarray(source_vertices, dtype=np.float64).reshape(-1, 3)]
    if len(object_vertices):
        chunks.append(_transform_points(object_vertices, object_poses).reshape(-1, 3))
    for profile in profiles.values():
        for part in profile.get("robot_mesh", {}).get("parts", []):
            chunks.append(
                _transform_points(
                    np.asarray(part["vertices"], dtype=np.float64),
                    np.asarray(part["transforms"], dtype=np.float64),
                ).reshape(-1, 3)
            )
    points = np.concatenate(chunks, axis=0)
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    extent = max(float(np.max(high - low)), 1e-3)
    margin = extent * 0.08
    return np.round(np.stack((low - margin, high + margin)), decimals=8).tolist()


def render_clip_html(
    *,
    clip: ClipSpec,
    canonical_path: str | Path,
    source_path: str | Path,
    profile_paths: dict[str, tuple[str | Path, bool, str]],
    output: str | Path,
    asset_root: str | Path,
    recommended_profile: str,
    graph_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
) -> Path:
    del source_path
    sequence = load_hoi_sequence(canonical_path)
    hand = _hand(sequence, clip.hand)
    source_keypoints = np.asarray(
        hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    model = load_artimano_model("rh", asset_root=asset_root)
    object_track = sequence.rigid_objects[0]
    object_vertices, object_faces = _mesh_subset(
        np.asarray(object_track.mesh.vertices_local, dtype=np.float64),
        np.asarray(object_track.mesh.faces, dtype=np.int64),
        max_faces=_MAX_OBJECT_FACES,
    )
    object_poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    if len(source_keypoints) != clip.length or len(object_poses) != clip.length:
        raise ValueError("quality HTML expects a 60-frame canonical clip")
    source_vertices = np.asarray(hand.vertices_scene, dtype=np.float64)
    if source_vertices.shape[0] != clip.length:
        raise ValueError(f"source MANO vertices must have {clip.length} frames")
    if hand.mesh is None:
        raise ValueError("canonical hand is missing MANO mesh faces")
    source_faces = np.asarray(hand.mesh.faces, dtype=np.int64)
    profiles: dict[str, Any] = {
        "source_mano": {
            "label": "source MANO",
            "paper_method": True,
            "keypoints": _rounded(source_keypoints),
            "robot_mesh": {"parts": []},
            "metrics": [],
        }
    }
    for profile_id, (path, warm, label) in profile_paths.items():
        payload = _profile_payload(
            path,
            warm=warm,
            model=model,
            object_vertices=object_vertices,
            object_poses=object_poses,
        )
        payload["label"] = label
        payload.update(
            _PROFILE_SEMANTICS.get(
                profile_id,
                {
                    "role": "quality profile",
                    "description": "质量实验 profile；具体 artifact 见 provenance。",
                },
            )
        )
        payload["paper_method"] = profile_id.startswith(("paper", "scipy"))
        payload["paper_external_extension"] = profile_id.startswith(
            ("morphology", "contact", "combined")
        )
        profiles[profile_id] = payload
    if graph_path is not None and evaluation_path is not None:
        graph = load_interaction_graph(graph_path)
        evaluation = load_interaction_evaluation(evaluation_path)
        if graph.frame_count != clip.length or evaluation.frame_count != clip.length:
            raise ValueError("quality HTML graph/evaluation artifacts must have 60 frames")
        display_source_indices = np.arange(
            clip.start_frame, clip.start_frame + clip.length, dtype=np.int64
        )
        interaction_profiles: dict[str, Any] = {}
        for profile_id, profile in profiles.items():
            if profile_id == "source_mano":
                continue
            interaction_profiles[profile_id] = _interaction_payload(
                graph,
                evaluation,
                np.asarray(profile["keypoints"], dtype=np.float64),
                display_source_indices,
                np.arange(clip.length, dtype=np.int64),
            )
        interaction_provenance = {
            "source_graph_artifact": str(Path(graph_path).resolve()),
            "evaluation_artifact": str(Path(evaluation_path).resolve()),
            "source_graph_artifact_hash": interaction_artifact_hash(graph_path),
            "evaluation_artifact_hash": interaction_artifact_hash(evaluation_path),
            "directed_weights_unchanged": True,
            "display_weight_rule": "mean(w_ij,w_ji)",
            "shared_connectivity": True,
        }
    else:
        interaction_profiles = {}
        interaction_provenance = {
            "available": False,
            "reason": "graph_path/evaluation_path were not supplied",
        }
    paper_warm = profiles.get("paper_warm")
    if paper_warm is None:
        paper_warm = next(
            profile for profile_id, profile in profiles.items() if profile_id != "source_mano"
        )
    default_final = profiles.get(recommended_profile, paper_warm)
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "title": f"GRAB Arti-MANO quality · {clip.unit_id} · {clip.sequence}",
        "clip": clip.as_dict(),
        "frame_count": clip.length,
        "recommended_profile": recommended_profile,
        "profiles": profiles,
        "warm": paper_warm["robot_mesh"],
        "final": default_final["robot_mesh"],
        "interaction_profiles": interaction_profiles,
        "interaction_provenance": interaction_provenance,
        "source_mesh": {"vertices": _rounded(source_vertices), "faces": source_faces.tolist()},
        "object_mesh": {
            "vertices": _rounded(object_vertices),
            "faces": object_faces.tolist(),
            "poses": _rounded(object_poses, digits=8),
            "sampling": {
                "method": "deterministic_face_subsample",
                "max_faces": _MAX_OBJECT_FACES,
                "face_count": int(len(object_faces)),
            },
        },
        "bounds": _bounds(source_vertices, object_vertices, object_poses, profiles),
        "layers": [
            "source MANO triangular mesh",
            "selected Arti-MANO visual triangular mesh",
            "object triangular mesh preview",
            "MediaPipe-21 / robot skeleton anchors",
            "source-to-robot contact vectors",
            "robot-to-object closest-point vectors",
            "per-frame quality metrics",
        ],
    }
    document = _HTML_TEMPLATE.replace(
        "__DATA__", json.dumps(_finite(payload), separators=(",", ":"))
    )
    document = document.replace("__LAYERS__", ", ".join(payload["layers"]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def smoke_html(path: str | Path, *, expected_frames: int = 60, profiles: int = 1) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    required = {
        'id="frame"': "frame slider",
        'id="play"': "play/pause",
        "profileSelect": "profile selector",
        "meshSource": "mesh layer toggles",
        "recommended_profile": "recommended profile",
        'id="metrics"': "metrics panel",
        "source_mesh": "source mesh payload",
        "object_mesh": "object mesh payload",
        "drawMesh(": "triangle renderer",
        "DATA.bounds": "adaptive bounds",
        "pointerdown": "orbit interaction",
        'id="mode"': "visualization modes",
        "graphSource": "graph layers",
        "edgeThreshold": "edge filters",
        "residualTarget": "residual diagnostic",
        "interaction_profiles": "profile interaction payloads",
        "interaction_provenance": "viewer provenance",
    }
    checks = {name: token in source for token, name in required.items()}
    start = source.find("const DATA = ")
    data_ok = False
    frame_count = 0
    profile_count = 0
    if start >= 0:
        try:
            raw_start = start + len("const DATA = ")
            data, raw_end = json.JSONDecoder().raw_decode(source[raw_start:])
            raw = source[raw_start : raw_start + raw_end]
            frame_count = int(data.get("frame_count", 0))
            profile_count = len(data.get("profiles", {}))
            data_ok = frame_count == expected_frames and profile_count >= profiles
            data_ok = data_ok and "NaN" not in raw and "Infinity" not in raw
            source_mesh = data.get("source_mesh", {})
            object_mesh = data.get("object_mesh", {})
            data_ok = data_ok and bool(source_mesh.get("vertices"))
            data_ok = data_ok and bool(source_mesh.get("faces"))
            data_ok = data_ok and bool(object_mesh.get("vertices"))
            data_ok = data_ok and bool(object_mesh.get("faces"))
            data_ok = data_ok and len(data.get("bounds", [])) == 2
            data_ok = data_ok and all(
                "robot_mesh" in profile for profile in data.get("profiles", {}).values()
            )
            data_ok = data_ok and len(data.get("interaction_profiles", {})) >= max(profiles - 1, 1)
        except (ValueError, json.JSONDecodeError):
            data_ok = False
    result = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "path": str(Path(path).resolve()),
        "exists": Path(path).is_file(),
        "nonempty": Path(path).stat().st_size > 0 if Path(path).is_file() else False,
        "checks": checks,
        "data_ok": data_ok,
        "frame_count": frame_count,
        "profile_count": profile_count,
        "browser_environment_blocked": False,
        "status": "pass" if all(checks.values()) and data_ok else "fail",
    }
    return result


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRAB Arti-MANO quality viewer</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#111827;color:#e5e7eb}main{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:100vh}#view{min-width:0;min-height:0;position:relative}canvas{display:block;width:100%;height:100%;background:#f8fafc;cursor:grab}canvas.dragging{cursor:grabbing}aside{box-sizing:border-box;overflow:auto;padding:16px;background:#1f2937}h1{font-size:18px;margin:0 0 8px}h2{font-size:13px;margin:16px 0 7px;color:#93c5fd}label{display:block;margin:6px 0;font-size:12px}select,input[type=number]{width:100%;box-sizing:border-box;background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:4px;padding:4px}input[type=range]{width:100%}button{border:0;border-radius:5px;padding:6px 12px;background:#2563eb;color:#fff;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:5px 12px}.row{display:flex;gap:5px}.row button{flex:1}.hint{color:#9ca3af;font-size:11px;line-height:1.4}pre{white-space:pre-wrap;font:11px/1.4 ui-monospace,monospace;color:#d1d5db}.legend{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}.badge{display:inline-block;padding:2px 5px;border-radius:3px;background:#334155;color:#dbeafe;font-size:11px;margin:2px 2px 0 0}
</style>
</head>
<body>
<main>
<section id="view"><canvas id="scene"></canvas></section>
<aside>
<h1 id="title"></h1>
<div class="hint">Drag to orbit · wheel to zoom · graph states reuse the frozen Stage 8 connectivity.</div>
<h2>Profile</h2><select id="profileSelect"></select><div class="hint" id="profileTag"></div>
<h2>Visualization mode</h2><select id="mode"><option value="mesh">mesh</option><option value="full-graph">full-graph</option><option value="figure4-style">figure4-style</option><option value="laplacian-diagnostic">laplacian-diagnostic</option><option value="combined">combined</option></select>
<h2>Frame</h2><input id="frame" type="range" min="0" max="59" value="0" step="1"><div><span id="frameLabel"></span> <button id="play">Play</button> <button id="reset">Reset view</button></div>
<h2>Mesh layers</h2><div class="grid"><label><input id="meshSource" type="checkbox" checked> <span class="legend" style="background:#3b82f6"></span>source</label><label><input id="meshWarm" type="checkbox" checked> <span class="legend" style="background:#f59e0b"></span>warm</label><label><input id="meshFinal" type="checkbox" checked> <span class="legend" style="background:#22c55e"></span>selected final</label><label><input id="objectContext" type="checkbox" checked> object context</label><label><input id="objectSurface" type="checkbox" checked> object surface</label><label><input id="contactVectors" type="checkbox" checked> contact vectors</label></div>
<h2>Graph states</h2><div class="grid"><label><input id="graphSource" type="checkbox" checked> source graph</label><label><input id="graphWarm" type="checkbox" checked> warm graph</label><label><input id="graphFinal" type="checkbox" checked> final graph</label><label><input id="showLabels" type="checkbox"> labels</label></div>
<h2>Edge filters</h2><div class="grid"><label><input id="edgeHH" type="checkbox" checked> hand-hand</label><label><input id="edgeHO" type="checkbox" checked> hand-object</label><label><input id="edgeOO" type="checkbox" checked> object-object</label><label><input id="handObjectOnly" type="checkbox"> hand-object only</label></div>
<label>weight mode<select id="weightMode"><option>none</option><option>opacity</option><option>width</option><option>color</option></select></label><label>edge threshold <span id="thresholdValue"></span><input id="edgeThreshold" type="range" min="0" max="0.2" step="0.001" value="0"></label><label>top-k edges (0 = all)<input id="topKEdges" type="number" min="0" step="1" value="0"></label>
<h2>Residual diagnostic</h2><label>residual target<select id="residualTarget"><option value="warm">warm-start</option><option value="final">selected final</option></select></label><label>residual display<select id="residualDisplay"><option>scalar</option><option>vector</option><option>both</option></select></label><label>residual scope<select id="residualScope"><option value="all">all</option><option value="hand">hand only</option><option value="object">object only</option></select></label><label>residual threshold <span id="residualThresholdValue"></span><input id="residualThreshold" type="range" min="0" max="0.05" step="0.0001" value="0"></label><label>top-k residual vertices (0 = all)<input id="topKResidual" type="number" min="0" step="1" value="0"></label>
<h2>Frame metrics</h2><pre id="metrics"></pre><h2>Provenance</h2><div class="hint" id="provenance"></div>
</aside></main>
<script>
const DATA = __DATA__;
const canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),frameInput=document.getElementById('frame'),frameLabel=document.getElementById('frameLabel'),metricsBox=document.getElementById('metrics'),modeInput=document.getElementById('mode'),profileSelect=document.getElementById('profileSelect');
let frame=0,yaw=-0.75,pitch=0.3,zoom=1,playing=false,timer=null,dragging=false,lastX=0,lastY=0;
const colors={source:'#3b82f6',warm:'#f59e0b',final:'#22c55e'},source=DATA.source_mesh,object=DATA.object_mesh;
const meshLayers={source:document.getElementById('meshSource'),warm:document.getElementById('meshWarm'),final:document.getElementById('meshFinal')},graphLayers={source:document.getElementById('graphSource'),warm:document.getElementById('graphWarm'),final:document.getElementById('graphFinal')},edgeLayers={0:document.getElementById('edgeHH'),1:document.getElementById('edgeHO'),2:document.getElementById('edgeOO')};
const low=DATA.bounds[0],high=DATA.bounds[1],center=[(low[0]+high[0])/2,(low[1]+high[1])/2,(low[2]+high[2])/2],extent=Math.max(high[0]-low[0],high[1]-low[1],high[2]-low[2],1e-3);
document.getElementById('title').textContent=DATA.title;frameInput.max=String(DATA.frame_count-1);const profileIds=Object.keys(DATA.profiles);profileIds.forEach(id=>{const option=document.createElement('option');option.value=id;option.textContent=DATA.profiles[id].label||id;profileSelect.appendChild(option)});profileSelect.value=DATA.recommended_profile in DATA.profiles&&DATA.recommended_profile!=='source_mano'?DATA.recommended_profile:(profileIds.find(id=>id!=='source_mano')||profileIds[0]);
function activeProfile(){return DATA.profiles[profileSelect.value]||DATA.profiles[profileIds.find(id=>id!=='source_mano')||profileIds[0]]}function activeInteraction(){return DATA.interaction_profiles[profileSelect.value]||DATA.interaction_profiles[profileIds.find(id=>id!=='source_mano')||profileIds[0]]||null}
function transformPoint(m,p){return[m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]]}function rotate(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return[x1,cp*y-sp*z1,sp*y+cp*z1]}function project(p,w,h){const q=rotate(p),camera=extent*3,scale=Math.min(w,h)*0.72*zoom/Math.max(0.15,camera-q[2]);return[w/2+q[0]*scale,h/2-q[1]*scale,q[2]]}
function transformRobot(payload,index){return payload.parts.map(part=>{const m=part.transforms[index];return part.vertices.map(p=>transformPoint(m,p))})}function drawMesh(vertices,faces,color,alpha,w,h){if(!vertices?.length||!faces?.length)return;const projected=vertices.map(p=>project(p,w,h)),ordered=faces.map(face=>[face,(projected[face[0]][2]+projected[face[1]][2]+projected[face[2]][2])/3]);ordered.sort((a,b)=>a[1]-b[1]);ctx.fillStyle=color;ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=0.25;for(const [face] of ordered){const a=projected[face[0]],b=projected[face[1]],c=projected[face[2]];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill();ctx.stroke()}ctx.globalAlpha=1}function drawRobot(payload,index,color,alpha,w,h){const vertices=transformRobot(payload,index);payload.parts.forEach((part,i)=>drawMesh(vertices[i],part.faces,color,alpha,w,h))}
function drawObjectContext(index,w,h){const pose=object.poses[index];ctx.fillStyle='#64748b';ctx.globalAlpha=0.42;for(const p of object.vertices){const s=project(transformPoint(p,pose),w,h);ctx.fillRect(s[0]-1,s[1]-1,2,2)}ctx.globalAlpha=1}function drawObjectSurface(index,w,h){const pose=object.poses[index];drawMesh(object.vertices.map(p=>transformPoint(p,pose)),object.faces,'#64748b',0.24,w,h)}
const handEdges=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
function drawPoints(points,color,size,w,h){ctx.fillStyle=color;ctx.globalAlpha=0.9;(points||[]).forEach(p=>{const s=project(p,w,h);ctx.beginPath();ctx.arc(s[0],s[1],size,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1}function drawSkeleton(points,color,w,h){if(!points?.length)return;ctx.strokeStyle=color;ctx.lineWidth=1.5;for(const e of handEdges){const a=points[e[0]],b=points[e[1]];if(!a||!b)continue;const pa=project(a,w,h),pb=project(b,w,h);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke()}drawPoints(points,color,2.8,w,h)}function drawArrow(a,b,color,dashed,w,h){if(!a||!b)return;const pa=project(a,w,h),pb=project(b,w,h);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.2;ctx.setLineDash(dashed?[5,4]:[]);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.setLineDash([])}function drawContacts(profile,w,h){const s=DATA.profiles.source_mano.keypoints[frame]||[],r=profile.keypoints?.[frame]||[],c=profile.contact_points?.[frame]||[];for(let i=0;i<Math.min(s.length,r.length);i++){drawArrow(s[i],r[i],'#14b8a6',true,w,h);if(c[i])drawArrow(r[i],c[i],'#f97316',false,w,h)}drawPoints(c,'#f97316',3,w,h)}
function categoryStyle(category){return category===0?[0.25,0.8]:category===1?[0.85,1.7]:[0.18,0.55]}function weightColor(weight){const t=Math.max(0,Math.min(1,weight/0.15)),hue=235-235*t;return`hsl(${hue},80%,50%)`}function drawEdge(points,edge,color,alpha,width,w,h){const a=project(points[edge[0]],w,h),b=project(points[edge[1]],w,h);ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.globalAlpha=1}function selectedEdges(graphFrame){const threshold=Number(document.getElementById('edgeThreshold').value),topK=Number(document.getElementById('topKEdges').value)||0,only=document.getElementById('handObjectOnly').checked,ids=[];for(let i=0;i<graphFrame.edges.length;i++){const category=graphFrame.categories[i];if(!edgeLayers[category].checked||only&&category!==1||graphFrame.weights[i]<threshold)continue;ids.push(i)}if(topK>0)ids.sort((a,b)=>graphFrame.weights[b]-graphFrame.weights[a]);return topK>0?ids.slice(0,topK):ids}
function drawGraphState(state,index,w,h,interaction){const graphFrame=interaction.frames[index],points=interaction.vertices[state][index];for(const i of selectedEdges(graphFrame)){const cat=graphFrame.categories[i],style=categoryStyle(cat),weight=graphFrame.weights[i],wm=document.getElementById('weightMode').value;let alpha=style[0],width=style[1],color=colors[state];if(wm==='opacity')alpha*=Math.max(0.12,Math.min(1,weight/0.12));if(wm==='width')width*=0.5+Math.min(2.5,weight/0.05);if(wm==='color')color=weightColor(weight);drawEdge(points,graphFrame.edges[i],color,alpha,width,w,h)}drawPoints(points,colors[state],2.5,w,h);if(document.getElementById('showLabels').checked){ctx.fillStyle=colors[state];ctx.font='9px monospace';interaction.vertex_metadata.forEach((meta,i)=>{const s=project(points[i],w,h);ctx.fillText(String(meta.semantic_name||meta.sample_id||i),s[0]+3,s[1]-3)})}}
function residualMask(index,target,interaction){const residual=interaction.residuals[target][index],scope=document.getElementById('residualScope').value,threshold=Number(document.getElementById('residualThreshold').value),topK=Number(document.getElementById('topKResidual').value)||0,norms=residual.map(v=>Math.hypot(v[0],v[1],v[2])),ids=[];for(let i=0;i<norms.length;i++){if(scope==='hand'&&i>=21||scope==='object'&&i<21||norms[i]<threshold)continue;ids.push(i)}ids.sort((a,b)=>norms[b]-norms[a]);return{residual,norms,ids:topK>0?ids.slice(0,topK):ids}}function residualColor(t){return`hsl(${235-235*Math.max(0,Math.min(1,t))},85%,50%)`}function drawResidual(index,w,h,interaction){const target=document.getElementById('residualTarget').value,display=document.getElementById('residualDisplay').value,selected=residualMask(index,target,interaction),points=interaction.vertices[target][index],max=Math.max(...selected.norms,1e-9),scale=4;for(const i of selected.ids){const p=points[i],n=selected.norms[i],s=project(p,w,h);if(display==='scalar'||display==='both'){ctx.fillStyle=residualColor(n/max);ctx.beginPath();ctx.arc(s[0],s[1],4,0,Math.PI*2);ctx.fill()}if(display==='vector'||display==='both'){const e=project([p[0]+selected.residual[i][0]*scale,p[1]+selected.residual[i][1]*scale,p[2]+selected.residual[i][2]*scale],w,h);ctx.strokeStyle='#dc2626';ctx.beginPath();ctx.moveTo(s[0],s[1]);ctx.lineTo(e[0],e[1]);ctx.stroke()}}return interaction.residual_summaries[target][index]}
function resize(){const ratio=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.floor(rect.width*ratio));canvas.height=Math.max(1,Math.floor(rect.height*ratio));ctx.setTransform(ratio,0,0,ratio,0,0);draw()}
function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,mode=modeInput.value,profile=activeProfile(),interaction=activeInteraction();ctx.clearRect(0,0,w,h);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,w,h);if(document.getElementById('objectContext').checked)drawObjectContext(frame,w,h);if(document.getElementById('objectSurface').checked)drawObjectSurface(frame,w,h);if(mode==='mesh'||mode==='combined'){if(meshLayers.source.checked)drawMesh(source.vertices[frame],source.faces,'#3b82f6',0.30,w,h);if(meshLayers.warm.checked)drawRobot(DATA.warm,frame,'#f59e0b',0.25,w,h);if(meshLayers.final.checked&&profile.robot_mesh?.parts?.length)drawRobot(profile.robot_mesh,frame,profile.paper_external_extension?'#f97316':'#22c55e',0.48,w,h);if(document.getElementById('contactVectors').checked)drawContacts(profile,w,h)}let residualSummary=null;if(interaction&&mode!=='mesh'){for(const state of Object.keys(graphLayers))if(graphLayers[state].checked)drawGraphState(state,frame,w,h,interaction);if(mode==='laplacian-diagnostic'||mode==='combined')residualSummary=drawResidual(frame,w,h,interaction)}const graphFrame=interaction?.frames?.[frame],metric=profile.metrics?.[frame]||{};frameLabel.textContent=`local ${frame} · source ${DATA.clip.start_frame+frame}`;const lines=[`mode: ${mode}`,`profile: ${profile.role||profileSelect.value}`,`id: ${profileSelect.value}`];if(graphFrame)lines.push(`graph: ${String(graphFrame.graph_hash).slice(0,12)}`,`edges: ${graphFrame.edges.length} (HH ${graphFrame.stats.hand_hand_edge_count}, HO ${graphFrame.stats.hand_object_edge_count}, OO ${graphFrame.stats.object_object_edge_count})`);for(const [key,value] of Object.entries(metric))lines.push(`${key}: ${typeof value==='number'?value.toPrecision(5):value}`);if(residualSummary)lines.push('',`residual target: ${document.getElementById('residualTarget').value}`,`residual max: ${residualSummary.max.toPrecision(5)}`,`residual mean: ${residualSummary.mean.toPrecision(5)}`,`hand mean: ${residualSummary.hand_mean.toPrecision(5)}`,`object mean: ${residualSummary.object_mean.toPrecision(5)}`,`top vertices: ${residualSummary.top.map(x=>x.vertex_id).join(', ')}`);metricsBox.textContent=lines.join('\n');document.getElementById('thresholdValue').textContent=Number(document.getElementById('edgeThreshold').value).toFixed(3);document.getElementById('residualThresholdValue').textContent=Number(document.getElementById('residualThreshold').value).toFixed(4);document.getElementById('profileTag').innerHTML=`<span class="badge">${profile.role||'profile'}</span><br>${profile.description||''}<br><span class="hint">${profile.artifact_path||''}</span>`;document.getElementById('provenance').innerHTML=`graph hash: ${String(DATA.interaction_provenance.source_graph_artifact_hash||'unavailable').slice(0,16)}<br>evaluation hash: ${String(DATA.interaction_provenance.evaluation_artifact_hash||'unavailable').slice(0,16)}<br>directed weights unchanged: ${DATA.interaction_provenance.directed_weights_unchanged??'n/a'}<br>display weights: ${DATA.interaction_provenance.display_weight_rule||'n/a'}`;frameInput.value=String(frame)}
function setFrame(value){frame=Math.max(0,Math.min(DATA.frame_count-1,Number(value)));draw()}function applyModeDefaults(value){if(value==='figure4-style'){document.getElementById('handObjectOnly').checked=true;document.getElementById('edgeHH').checked=false;document.getElementById('edgeOO').checked=false}else if(value==='full-graph'){document.getElementById('handObjectOnly').checked=false;document.getElementById('edgeHH').checked=true;document.getElementById('edgeHO').checked=true;document.getElementById('edgeOO').checked=true}draw()}
modeInput.addEventListener('change',e=>applyModeDefaults(e.target.value));frameInput.addEventListener('input',e=>setFrame(e.target.value));profileSelect.addEventListener('change',draw);document.querySelectorAll('input,select').forEach(item=>item.addEventListener('change',draw));document.getElementById('play').addEventListener('click',()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';if(playing)timer=setInterval(()=>setFrame((frame+1)%DATA.frame_count),100);else clearInterval(timer)});document.getElementById('reset').addEventListener('click',()=>{yaw=-0.75;pitch=0.3;zoom=1;draw()});canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*0.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*0.01));lastX=e.clientX;lastY=e.clientY;draw()});canvas.addEventListener('pointerup',e=>{dragging=false;canvas.classList.remove('dragging');canvas.releasePointerCapture(e.pointerId)});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.35,Math.min(4,zoom*Math.exp(-e.deltaY*0.001)));draw()},{passive:false});window.addEventListener('resize',resize);resize();
</script>
</body></html>"""


__all__ = ["render_clip_html", "smoke_html"]
