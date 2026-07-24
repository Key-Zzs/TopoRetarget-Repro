"""Self-contained HTML visualization for source and retargeted hand meshes."""

# The embedded HTML/JavaScript is intentionally readable and is not Python code.
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import transform_points
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.delaunay import edge_category
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.laplacian import laplacian_numpy
from toporetarget.robots.registry import get_robot_registry
from toporetarget.robots.visualization import _primitive_mesh

from .schema import read_json

HTML_SCHEMA_VERSION = "toporetarget.mesh_viewer.v1"
INTERACTION_HTML_SCHEMA_VERSION = "toporetarget.interaction_mesh_viewer.v1"
_DEFAULT_MAX_OBJECT_POINTS = 1200
_GRAPH_STATE_COLORS = {"source": "#3b82f6", "warm": "#f59e0b", "final": "#22c55e"}
_EDGE_CATEGORY_CODES = {"hand-hand": 0, "hand-object": 1, "object-object": 2}


def _rounded(value: Any, digits: int = 6) -> Any:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return np.round(array, decimals=digits).tolist()
    return array.tolist()


def _finite_metric(array: np.ndarray, index: int) -> float | int | bool | None:
    value = np.asarray(array)[index]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _robot_name(manifest: dict[str, Any]) -> str:
    robot = str(manifest.get("robot", ""))
    if robot not in {spec.name for spec in get_robot_registry().specs()}:
        raise ValueError(f"manifest robot is not registered: {robot!r}")
    return robot


def _source_frame_indices(sequence: Any, final: Any, manifest: dict[str, Any]) -> np.ndarray:
    frame_count = final.frame_count
    if sequence.num_frames == frame_count:
        return np.arange(frame_count, dtype=np.int64)
    source_indices = np.asarray(final.arrays.get("source_frame_indices", []), dtype=np.int64)
    if source_indices.shape == (frame_count,):
        return source_indices
    frame_indices = np.asarray(final.arrays.get("frame_indices", []), dtype=np.int64)
    offset = int(final.metadata.get("source_frame_offset", manifest["selected_frame_range"][0]))
    if frame_indices.shape == (frame_count,):
        return frame_indices + offset
    return np.arange(frame_count, dtype=np.int64) + offset


def _source_mesh(
    sequence: Any, final: Any, manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    hand_id = str(final.metadata.get("source_hand_id", ""))
    hand = sequence.hand(hand_id)
    source_indices = _source_frame_indices(sequence, final, manifest)
    if np.any(source_indices < 0) or np.any(source_indices >= sequence.num_frames):
        raise ValueError(
            "source frame indices do not fit canonical sequence: "
            f"range=[{int(source_indices.min())},{int(source_indices.max())}], "
            f"num_frames={sequence.num_frames}"
        )
    if hand.vertices_scene is not None:
        vertices = np.asarray(hand.vertices_scene[source_indices], dtype=np.float64)
    elif hand.mesh is not None:
        local = np.broadcast_to(
            np.asarray(hand.mesh.vertices_local, dtype=np.float64),
            (sequence.num_frames,) + hand.mesh.vertices_local.shape,
        )
        vertices = transform_points(hand.wrist_pose_scene.pose_scene, local)[source_indices]
    else:
        raise ValueError(f"canonical hand {hand_id!r} has no mesh vertices")
    if hand.mesh is None:
        raise ValueError(f"canonical hand {hand_id!r} has no mesh faces")
    faces = np.asarray(hand.mesh.faces, dtype=np.int64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"source hand mesh must have shape [T,V,3], got {vertices.shape}")
    return vertices, faces


def _robot_payload(model: Any, qpos: np.ndarray, base_pose: np.ndarray) -> dict[str, Any]:
    if qpos.shape[0] != base_pose.shape[0]:
        raise ValueError("robot qpos and base_pose_scene frame counts differ")
    first_instances = model.visual_geometry_instances(qpos[0], base_pose[0])
    parts: list[dict[str, Any]] = []
    for instance_index, first in enumerate(first_instances):
        vertices, faces = _primitive_mesh(first)
        transforms = [np.asarray(first.world_transform, dtype=np.float64)]
        for frame in range(1, qpos.shape[0]):
            instances = model.visual_geometry_instances(qpos[frame], base_pose[frame])
            if len(instances) != len(first_instances):
                raise ValueError("robot visual geometry topology changed across frames")
            current = instances[instance_index]
            transforms.append(np.asarray(current.world_transform, dtype=np.float64))
        parts.append(
            {
                "name": f"{first.link_name}:{instance_index}",
                "vertices": _rounded(vertices),
                "faces": faces.tolist(),
                "transforms": _rounded(np.asarray(transforms), digits=8),
            }
        )
    return {"parts": parts}


def _object_payload(
    sequence: Any, final: Any, source_indices: np.ndarray, max_points: int
) -> dict[str, Any]:
    object_id = str(final.metadata.get("object_id", ""))
    if not object_id:
        return {"vertices": [], "poses": [], "object_id": None}
    track = sequence.rigid_object(object_id)
    vertices = np.asarray(track.mesh.vertices_local, dtype=np.float64)
    if len(vertices) > max_points:
        selected = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
        vertices = vertices[selected]
    poses = np.asarray(track.pose_scene.pose_scene[source_indices], dtype=np.float64)
    return {
        "object_id": object_id,
        "vertices": _rounded(vertices),
        "poses": _rounded(poses, digits=8),
    }


def _metrics(final: Any, warm: Any, source_indices: np.ndarray) -> dict[str, Any]:
    metric_sources = {
        "warm_total_objective": (warm, "total_objective"),
        "final_objective": (final, "final_objective"),
        "warm_e_im": (final, "warm_e_im"),
        "e_im": (final, "e_im"),
        "warm_e_bone": (final, "warm_e_bone"),
        "e_bone": (final, "e_bone"),
        "e_base_pos": (final, "e_base_pos"),
        "e_base_rot": (final, "e_base_rot"),
        "max_penetration": (final, "max_penetration"),
        "min_full_signed_distance": (final, "min_full_signed_distance"),
        "iterations": (final, "iterations"),
        "solve_time_s": (final, "solve_time_s"),
        "solver_success": (final, "solver_success"),
        "accepted": (final, "accepted"),
    }
    result: dict[str, Any] = {"source_frame_indices": source_indices.tolist()}
    frames: list[dict[str, Any]] = []
    for index in range(final.frame_count):
        item: dict[str, Any] = {"local_frame": index, "source_frame": int(source_indices[index])}
        for name, (owner, array_name) in metric_sources.items():
            if array_name in owner.arrays:
                item[name] = _finite_metric(owner.arrays[array_name], index)
        frames.append(item)
    result["frames"] = frames
    return result


def _edge_category_codes(edges: np.ndarray) -> np.ndarray:
    """Classify the frozen Stage 8 undirected edges for HTML filtering."""

    value = np.asarray(edges, dtype=np.int64)
    if value.ndim != 2 or value.shape[1:] != (2,):
        raise ValueError(f"edges must have shape [E,2], got {value.shape}")
    return np.asarray([_EDGE_CATEGORY_CODES[edge_category(edge)] for edge in value], dtype=np.int8)


def _visual_undirected_weights(directed: Any, edges: np.ndarray) -> np.ndarray:
    """Return mean(w_ij,w_ji) for display only; optimization weights stay directed."""

    source = np.asarray(directed.source_index, dtype=np.int64)
    destination = np.asarray(directed.destination_index, dtype=np.int64)
    weights = np.asarray(directed.weights, dtype=np.float64)
    pair_values: dict[tuple[int, int], list[float]] = {}
    for first, second, weight in zip(source, destination, weights, strict=True):
        first_index, second_index = int(first), int(second)
        key = (min(first_index, second_index), max(first_index, second_index))
        pair_values.setdefault(key, []).append(float(weight))
    result = []
    for first, second in np.asarray(edges, dtype=np.int64):
        values = pair_values.get((int(first), int(second)), [])
        if len(values) != 2:
            raise ValueError(
                f"Stage 8 directed graph does not contain two weights for {(first, second)}"
            )
        result.append(float(np.mean(values)))
    return np.asarray(result, dtype=np.float64)


def _filter_edge_indices(
    edges: np.ndarray,
    categories: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float = 0.0,
    top_k: int = 0,
    hand_object_only: bool = False,
) -> np.ndarray:
    """Select graph edges for drawing without changing the stored graph."""

    if threshold < 0:
        raise ValueError("edge weight threshold must be non-negative")
    if top_k < 0:
        raise ValueError("top-k edges must be non-negative")
    mask = np.asarray(weights) >= float(threshold)
    if hand_object_only:
        mask &= np.asarray(categories) == _EDGE_CATEGORY_CODES["hand-object"]
    selected = np.flatnonzero(mask)
    if top_k > 0 and len(selected) > top_k:
        order = np.argsort(-np.asarray(weights)[selected], kind="stable")
        selected = selected[order[:top_k]]
    return selected.astype(np.int64, copy=False)


def _residual_summary(
    residual: np.ndarray, vertex_metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    value = np.asarray(residual, dtype=np.float64)
    if value.shape != (71, 3):
        raise ValueError(f"residual must have shape [71,3], got {value.shape}")
    norms = np.linalg.norm(value, axis=1)
    hand = norms[:21]
    obj = norms[21:]
    order = np.argsort(-norms, kind="stable")
    top = []
    for index in order[:5]:
        metadata = vertex_metadata[int(index)] if int(index) < len(vertex_metadata) else {}
        top.append(
            {
                "vertex_id": int(index),
                "name": metadata.get("semantic_name", metadata.get("sample_id", index)),
                "norm": float(norms[index]),
            }
        )
    return {
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "hand_mean": float(np.mean(hand)),
        "object_mean": float(np.mean(obj)),
        "top": top,
    }


def _display_source_indices(final: Any, manifest: dict[str, Any], frame_count: int) -> np.ndarray:
    values = np.asarray(final.arrays.get("source_frame_indices", []), dtype=np.int64)
    if values.shape == (frame_count,):
        return values
    start = int(manifest.get("selected_frame_range", [0, frame_count])[0])
    return np.arange(frame_count, dtype=np.int64) + start


def _interaction_payload(
    graph: Any,
    evaluation: Any,
    final_keypoints: np.ndarray,
    display_source_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    """Serialize the fixed Stage 8 graph and Stage 8/9 states for the browser."""

    graph.validate()
    evaluation.validate()
    if graph.frame_count != evaluation.frame_count or graph.frame_count != len(final_keypoints):
        raise ValueError("graph, evaluation, and final trajectory frame counts differ")
    if not np.array_equal(graph.source_vertices[:, 21:], evaluation.robot_vertices[:, 21:]):
        raise ValueError("Stage 8 object sample identity changed between graph and evaluation")
    final_vertices = np.concatenate(
        [np.asarray(final_keypoints, dtype=np.float64), graph.source_vertices[:, 21:]], axis=1
    )
    final_residuals: list[np.ndarray] = []
    graph_frames: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    warm_summaries: list[dict[str, Any]] = []
    final_summaries: list[dict[str, Any]] = []
    metadata = graph.source_vertex_metadata
    for index in frame_indices.tolist():
        edges = np.asarray(graph.edge_frames[index], dtype=np.int64)
        categories = _edge_category_codes(edges)
        weights = _visual_undirected_weights(graph.directed_frames[index], edges)
        final_residual = (
            laplacian_numpy(
                final_vertices[index],
                graph.directed_frames[index].source_index,
                graph.directed_frames[index].destination_index,
                graph.directed_frames[index].weights,
            )
            - graph.source_laplacian[index]
        )
        final_residuals.append(final_residual)
        graph_frames.append(
            {
                "local_frame": int(index),
                "source_frame": int(display_source_indices[index]),
                "edges": edges.tolist(),
                "categories": categories.tolist(),
                "weights": _rounded(weights),
                "graph_hash": graph.graph_hashes[index],
                "stats": graph.frame_statistics[index],
            }
        )
        source_zero = np.zeros((71, 3), dtype=np.float64)
        source_summaries.append(_residual_summary(source_zero, metadata))
        warm_summaries.append(_residual_summary(evaluation.residual[index], metadata))
        final_summaries.append(_residual_summary(final_residual, metadata))
    selected = np.asarray(frame_indices, dtype=np.int64)
    return {
        "vertex_count": 71,
        "hand_vertex_count": 21,
        "object_vertex_count": 50,
        "vertex_metadata": metadata,
        "vertices": {
            "source": _rounded(graph.source_vertices[selected]),
            "warm": _rounded(evaluation.robot_vertices[selected]),
            "final": _rounded(final_vertices[selected]),
        },
        "frames": graph_frames,
        "residuals": {
            "source": _rounded(np.zeros((len(selected), 71, 3), dtype=np.float64)),
            "warm": _rounded(evaluation.residual[selected]),
            "final": _rounded(np.asarray(final_residuals)),
        },
        "residual_summaries": {
            "source": source_summaries,
            "warm": warm_summaries,
            "final": final_summaries,
        },
        "source_graph_artifact_hash": graph.artifact_hash,
        "evaluation_artifact_hash": evaluation.artifact_hash,
        "shared_connectivity": True,
        "shared_weights": True,
        "object_sample_identity": "graph.source_vertices[:,21:] reused for source/warm/final",
        "visual_weight_assumption": "w_vis(i,j)=0.5*(w_ij+w_ji); directed Stage 8 weights are unchanged",
    }


def _apply_transform(vertices: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return vertices @ transforms[:, :3, :3].transpose(0, 2, 1) + transforms[:, None, :3, 3]


def _bounds(
    source_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_poses: np.ndarray,
    robot_payloads: Sequence[dict[str, Any]],
) -> list[list[float]]:
    chunks = [source_vertices.reshape(-1, 3)]
    if len(object_vertices):
        chunks.append(_apply_transform(object_vertices, object_poses).reshape(-1, 3))
    for payload in robot_payloads:
        for part in payload["parts"]:
            vertices = np.asarray(part["vertices"], dtype=np.float64)
            transforms = np.asarray(part["transforms"], dtype=np.float64)
            chunks.append(_apply_transform(vertices, transforms).reshape(-1, 3))
    points = np.concatenate(chunks, axis=0)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    extent = max(float(np.max(high - low)), 1e-3)
    margin = extent * 0.08
    return np.round(np.stack((low - margin, high + margin)), decimals=8).tolist()


def _html_document(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{payload["title"]}</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
  body {{ margin: 0; background: #111827; color: #e5e7eb; }}
  main {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; height: 100vh; }}
  #view {{ min-width: 0; min-height: 0; position: relative; }}
  canvas {{ display: block; width: 100%; height: 100%; background: #f8fafc; cursor: grab; }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{ box-sizing: border-box; overflow: auto; padding: 18px; background: #1f2937; }}
  h1 {{ font-size: 18px; margin: 0 0 10px; }}
  h2 {{ font-size: 13px; margin: 20px 0 8px; color: #93c5fd; }}
  label {{ display: block; margin: 8px 0; font-size: 13px; }}
  input[type=range] {{ width: 100%; }}
  button {{ border: 0; border-radius: 5px; padding: 6px 12px; background: #2563eb; color: white; cursor: pointer; }}
  pre {{ white-space: pre-wrap; font: 12px ui-monospace, monospace; line-height: 1.45; color: #d1d5db; }}
  .legend {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }}
  .hint {{ color: #9ca3af; font-size: 12px; line-height: 1.45; }}
</style>
</head>
<body>
<main>
  <section id="view"><canvas id="scene"></canvas></section>
  <aside>
    <h1>{payload["title"]}</h1>
    <div class="hint">Drag to orbit · wheel to zoom · source and robot meshes are in scene coordinates.</div>
    <h2>Frame</h2>
    <input id="frame" type="range" min="0" max="{payload["frame_count"] - 1}" value="0" step="1">
    <div><span id="frameLabel"></span> <button id="play">Play</button></div>
    <h2>Layers</h2>
    <label><input id="source" type="checkbox" checked> <span class="legend" style="background:#3b82f6"></span>Source MANO mesh</label>
    <label><input id="warm" type="checkbox" checked> <span class="legend" style="background:#f59e0b"></span>Warm-start robot mesh</label>
    <label><input id="final" type="checkbox" checked> <span class="legend" style="background:#22c55e"></span>Final robot mesh</label>
    <label><input id="object" type="checkbox" checked> <span class="legend" style="background:#64748b"></span>Object points</label>
    <h2>Frame metrics</h2>
    <pre id="metrics"></pre>
    <h2>About</h2>
    <div class="hint">This is a visual inspection aid. A good retarget still needs the numeric interaction, collision, continuity, and acceptance checks reported by Stage 9/10.</div>
  </aside>
</main>
<script>
const DATA = {data};
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const frameInput = document.getElementById('frame');
const frameLabel = document.getElementById('frameLabel');
const metricsBox = document.getElementById('metrics');
let frame = 0, yaw = -0.75, pitch = 0.3, zoom = 1.0, playing = false, timer = null;
let dragging = false, lastX = 0, lastY = 0;
const layers = {{ source: document.getElementById('source'), warm: document.getElementById('warm'), final: document.getElementById('final'), object: document.getElementById('object') }};
const low = DATA.bounds[0], high = DATA.bounds[1];
const center = [(low[0]+high[0])/2, (low[1]+high[1])/2, (low[2]+high[2])/2];
const extent = Math.max(high[0]-low[0], high[1]-low[1], high[2]-low[2]);

function resize() {{
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}}
function rotate(p) {{
  let x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*x + sy*z, z1 = -sy*x + cy*z;
  const y2 = cp*y - sp*z1, z2 = sp*y + cp*z1;
  return [x1, y2, z2];
}}
function project(p, width, height) {{
  const q = rotate(p), camera = extent * 3.0;
  const scale = Math.min(width, height) * 0.72 * zoom / Math.max(0.15, camera-q[2]);
  return [width/2 + q[0]*scale, height/2 - q[1]*scale, q[2]];
}}
function drawMesh(vertices, faces, color, alpha, width, height) {{
  const projected = vertices.map(p => project(p, width, height));
  const ordered = faces.map(face => [face, (projected[face[0]][2]+projected[face[1]][2]+projected[face[2]][2])/3]);
  ordered.sort((a,b) => a[1]-b[1]);
  ctx.fillStyle = color; ctx.strokeStyle = color; ctx.globalAlpha = alpha;
  for (const [face] of ordered) {{
    const a=projected[face[0]], b=projected[face[1]], c=projected[face[2]];
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.lineTo(c[0],c[1]); ctx.closePath(); ctx.fill();
  }}
  ctx.globalAlpha = 1;
}}
function robotVertices(payload, index) {{
  const output = [];
  for (const part of payload.parts) {{
    const m = part.transforms[index], vertices = part.vertices;
    output.push(vertices.map(p => [
      m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],
      m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],
      m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]
    ]));
  }}
  return output;
}}
function drawRobot(payload, index, color, alpha, width, height) {{
  const vertices = robotVertices(payload, index);
  payload.parts.forEach((part, partIndex) => drawMesh(vertices[partIndex], part.faces, color, alpha, width, height));
}}
function drawObject(index, width, height) {{
  const pose = DATA.object.poses[index], points = DATA.object.vertices;
  ctx.fillStyle = '#64748b'; ctx.globalAlpha = 0.45;
  for (const p of points) {{
    const q = [pose[0][0]*p[0]+pose[0][1]*p[1]+pose[0][2]*p[2]+pose[0][3], pose[1][0]*p[0]+pose[1][1]*p[1]+pose[1][2]*p[2]+pose[1][3], pose[2][0]*p[0]+pose[2][1]*p[1]+pose[2][2]*p[2]+pose[2][3]];
    const s = project(q,width,height); ctx.fillRect(s[0]-1,s[1]-1,2,2);
  }}
  ctx.globalAlpha = 1;
}}
function draw() {{
  const rect = canvas.getBoundingClientRect(), width = rect.width, height = rect.height;
  ctx.clearRect(0,0,width,height); ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,width,height);
  if (layers.object.checked && DATA.object.vertices.length) drawObject(frame,width,height);
  if (layers.source.checked) drawMesh(DATA.source.vertices[frame], DATA.source.faces, '#3b82f6', 0.30, width, height);
  if (layers.warm.checked) drawRobot(DATA.warm,frame,'#f59e0b',0.25,width,height);
  if (layers.final.checked) drawRobot(DATA.final,frame,'#22c55e',0.48,width,height);
  const item = DATA.metrics.frames[frame];
  frameLabel.textContent = `local ${{item.local_frame}} · source ${{item.source_frame}}`;
  const lines = [`sequence: ${{DATA.source_sequence}}`, `robot: ${{DATA.robot}}`, `object: ${{DATA.object.object_id || 'none'}}`];
  for (const [key,value] of Object.entries(item)) if (!['local_frame','source_frame'].includes(key)) lines.push(`${{key}}: ${{typeof value === 'number' ? value.toPrecision(5) : value}}`);
  metricsBox.textContent = lines.join('\\n');
}}
function setFrame(value) {{ frame = Math.max(0, Math.min(DATA.frame_count-1, Number(value))); frameInput.value = frame; draw(); }}
frameInput.addEventListener('input', event => setFrame(event.target.value));
Object.values(layers).forEach(item => item.addEventListener('change', draw));
document.getElementById('play').addEventListener('click', () => {{
  playing = !playing; document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) timer = setInterval(() => setFrame((frame+1) % DATA.frame_count), 100);
  else clearInterval(timer);
}});
canvas.addEventListener('pointerdown', event => {{ dragging=true; lastX=event.clientX; lastY=event.clientY; canvas.classList.add('dragging'); canvas.setPointerCapture(event.pointerId); }});
canvas.addEventListener('pointermove', event => {{ if (!dragging) return; yaw += (event.clientX-lastX)*0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch+(event.clientY-lastY)*0.01)); lastX=event.clientX; lastY=event.clientY; draw(); }});
canvas.addEventListener('pointerup', event => {{ dragging=false; canvas.classList.remove('dragging'); canvas.releasePointerCapture(event.pointerId); }});
canvas.addEventListener('wheel', event => {{ event.preventDefault(); zoom = Math.max(0.35, Math.min(4.0, zoom*Math.exp(-event.deltaY*0.001))); draw(); }}, {{passive:false}});
window.addEventListener('resize', resize); resize();
</script>
</body>
</html>
'''


def render_mesh_html(
    manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    mode: str = "mesh",
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_object_points: int = _DEFAULT_MAX_OBJECT_POINTS,
    asset_root: str | Path | None = None,
    open_browser: bool = False,
) -> dict[str, Any]:
    """Build one self-contained HTML viewer from a Stage 10 manifest."""

    # Keep the historical public entry point while using the unified viewer.
    from .interaction_html import render_interaction_mesh_html

    return render_interaction_mesh_html(
        manifest_path,
        output=output,
        mode=mode,
        start_frame=start_frame,
        end_frame=end_frame,
        max_object_points=max_object_points,
        asset_root=asset_root,
        open_browser=open_browser,
    )

    if max_object_points <= 0:
        raise ValueError("max_object_points must be positive")
    manifest = read_json(manifest_path)
    artifacts = manifest["artifacts"]
    sequence = load_hoi_sequence(artifacts["canonical"]["path"])
    warm = load_warm_start(artifacts["warm_start"]["path"])
    final = load_final_trajectory(artifacts["final"]["path"])
    if warm.frame_count != final.frame_count:
        raise ValueError("warm-start and final artifacts have different frame counts")
    frame_count = final.frame_count
    selected_start = 0 if start_frame is None else int(start_frame)
    selected_end = frame_count if end_frame is None else int(end_frame)
    if selected_start < 0 or selected_end <= selected_start or selected_end > frame_count:
        raise ValueError(f"HTML frame range must be within [0,{frame_count})")
    if selected_start != 0 or selected_end != frame_count:
        indices = np.arange(selected_start, selected_end, dtype=np.int64)
        source_indices_all = _source_frame_indices(sequence, final, manifest)
        source_vertices_all, source_faces = _source_mesh(sequence, final, manifest)
        source_vertices = source_vertices_all[indices]
        source_indices = source_indices_all[indices]
        qpos_warm = warm.arrays["qpos"][indices]
        base_warm = warm.arrays["base_pose_scene"][indices]
        qpos_final = final.arrays["qpos"][indices]
        base_final = final.arrays["base_pose_scene"][indices]
        warm_for_metrics = type(
            "WarmSlice", (), {"arrays": {k: v[indices] for k, v in warm.arrays.items()}}
        )()
        final_for_metrics = type(
            "FinalSlice",
            (),
            {
                "arrays": {k: v[indices] for k, v in final.arrays.items()},
                "frame_count": len(indices),
            },
        )()
    else:
        source_vertices, source_faces = _source_mesh(sequence, final, manifest)
        source_indices = _source_frame_indices(sequence, final, manifest)
        qpos_warm = np.asarray(warm.arrays["qpos"])
        base_warm = np.asarray(warm.arrays["base_pose_scene"])
        qpos_final = np.asarray(final.arrays["qpos"])
        base_final = np.asarray(final.arrays["base_pose_scene"])
        warm_for_metrics, final_for_metrics = warm, final
    model = get_robot_registry().load(
        _robot_name(manifest), asset_root=asset_root or manifest.get("asset_root")
    )
    warm_payload = _robot_payload(model, qpos_warm, base_warm)
    final_payload = _robot_payload(model, qpos_final, base_final)
    object_payload = _object_payload(sequence, final, source_indices, max_object_points)
    object_vertices = np.asarray(object_payload["vertices"], dtype=np.float64)
    object_poses = np.asarray(object_payload["poses"], dtype=np.float64)
    payload = {
        "schema_version": HTML_SCHEMA_VERSION,
        "title": f"TopoRetarget mesh viewer · {manifest.get('source_sequence', manifest.get('run_id', 'run'))}",
        "source_sequence": str(manifest.get("source_sequence", "")),
        "robot": _robot_name(manifest),
        "frame_count": int(selected_end - selected_start),
        "source": {"vertices": _rounded(source_vertices), "faces": source_faces.tolist()},
        "warm": warm_payload,
        "final": final_payload,
        "object": object_payload,
        "metrics": _metrics(final_for_metrics, warm_for_metrics, source_indices),
        "bounds": _bounds(
            source_vertices, object_vertices, object_poses, (warm_payload, final_payload)
        ),
    }
    destination = (
        Path(output)
        if output is not None
        else Path(manifest["run_root"]) / "review" / "trajectory_mesh.html"
    )
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_html_document(payload), encoding="utf-8")
    if open_browser:
        webbrowser.open(destination.resolve().as_uri())
    return {
        "status": "pass",
        "schema_version": HTML_SCHEMA_VERSION,
        "output": str(destination.resolve()),
        "frame_count": int(selected_end - selected_start),
        "source_vertices": int(source_vertices.shape[1]),
        "source_faces": int(source_faces.shape[0]),
        "robot_visual_parts": len(warm_payload["parts"]),
        "object_points": int(len(object_vertices)),
        "opened_browser": bool(open_browser),
    }


__all__ = ["HTML_SCHEMA_VERSION", "render_mesh_html"]
