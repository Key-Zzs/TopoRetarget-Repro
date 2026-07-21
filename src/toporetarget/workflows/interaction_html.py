"""Single-file HTML interaction-graph and Laplacian diagnostics viewer."""

# The embedded HTML/JavaScript is intentionally readable and is not Python code.
# ruff: noqa: E501

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.robots.registry import get_robot_registry

from .mesh_visualization import (
    INTERACTION_HTML_SCHEMA_VERSION,
    _bounds,
    _display_source_indices,
    _interaction_payload,
    _metrics,
    _object_payload,
    _robot_name,
    _robot_payload,
    _rounded,
    _source_frame_indices,
    _source_mesh,
)
from .schema import read_json


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
  main {{ display: grid; grid-template-columns: minmax(0, 1fr) 340px; height: 100vh; }}
  #view {{ min-width: 0; min-height: 0; position: relative; }}
  canvas {{ display: block; width: 100%; height: 100%; background: #f8fafc; cursor: grab; }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{ box-sizing: border-box; overflow: auto; padding: 16px; background: #1f2937; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  h2 {{ font-size: 13px; margin: 16px 0 7px; color: #93c5fd; }}
  label {{ display: block; margin: 6px 0; font-size: 12px; }}
  select, input[type=number] {{ width: 100%; box-sizing: border-box; background: #111827; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 4px; padding: 4px; }}
  input[type=range] {{ width: 100%; }}
  button {{ border: 0; border-radius: 5px; padding: 6px 12px; background: #2563eb; color: white; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 12px; }}
  .hint {{ color: #9ca3af; font-size: 11px; line-height: 1.4; }}
  pre {{ white-space: pre-wrap; font: 11px ui-monospace, monospace; line-height: 1.4; color: #d1d5db; }}
  .legend {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<main>
  <section id="view"><canvas id="scene"></canvas></section>
  <aside>
    <h1>{payload["title"]}</h1>
    <div class="hint">Drag to orbit · wheel to zoom · all graph states use the same frozen Stage 8 connectivity.</div>
    <h2>Visualization mode</h2>
    <select id="mode">
      <option value="mesh">mesh</option>
      <option value="full-graph">full-graph</option>
      <option value="figure4-style">figure4-style</option>
      <option value="laplacian-diagnostic">laplacian-diagnostic</option>
      <option value="combined">combined</option>
    </select>
    <h2>Frame</h2>
    <input id="frame" type="range" min="0" max="{payload["frame_count"] - 1}" value="0" step="1">
    <div><span id="frameLabel"></span> <button id="play">Play</button></div>
    <h2>Mesh layers</h2>
    <div class="grid">
      <label><input id="meshSource" type="checkbox" checked> <span class="legend" style="background:#3b82f6"></span>source</label>
      <label><input id="meshWarm" type="checkbox" checked> <span class="legend" style="background:#f59e0b"></span>warm</label>
      <label><input id="meshFinal" type="checkbox" checked> <span class="legend" style="background:#22c55e"></span>final</label>
      <label><input id="objectContext" type="checkbox" checked> object context</label>
    </div>
    <h2>Graph states</h2>
    <div class="grid">
      <label><input id="graphSource" type="checkbox" checked> source graph</label>
      <label><input id="graphWarm" type="checkbox" checked> warm graph</label>
      <label><input id="graphFinal" type="checkbox" checked> final graph</label>
      <label><input id="showLabels" type="checkbox"> labels</label>
    </div>
    <h2>Edge filters</h2>
    <div class="grid">
      <label><input id="edgeHH" type="checkbox" checked> hand-hand</label>
      <label><input id="edgeHO" type="checkbox" checked> hand-object</label>
      <label><input id="edgeOO" type="checkbox" checked> object-object</label>
      <label><input id="handObjectOnly" type="checkbox"> hand-object only</label>
    </div>
    <label>weight mode
      <select id="weightMode"><option>none</option><option>opacity</option><option>width</option><option>color</option></select>
    </label>
    <label>edge threshold <span id="thresholdValue"></span>
      <input id="edgeThreshold" type="range" min="0" max="0.2" step="0.001" value="0">
    </label>
    <label>top-k edges (0 = all)<input id="topKEdges" type="number" min="0" step="1" value="0"></label>
    <h2>Residual diagnostic</h2>
    <label>residual target
      <select id="residualTarget"><option value="warm">warm-start</option><option value="final">final</option></select>
    </label>
    <label>residual display
      <select id="residualDisplay"><option>scalar</option><option>vector</option><option>both</option></select>
    </label>
    <label>residual scope
      <select id="residualScope"><option value="all">all</option><option value="hand">hand only</option><option value="object">object only</option></select>
    </label>
    <label>residual threshold <span id="residualThresholdValue"></span>
      <input id="residualThreshold" type="range" min="0" max="0.05" step="0.0001" value="0">
    </label>
    <label>top-k residual vertices (0 = all)<input id="topKResidual" type="number" min="0" step="1" value="0"></label>
    <h2>Frame metrics</h2>
    <pre id="metrics"></pre>
    <h2>Provenance</h2>
    <div class="hint">Graph hash: {str(payload["interaction"]["source_graph_artifact_hash"] or "")[:16]}<br>Directed weights remain unchanged; only display uses mean(w_ij,w_ji).</div>
  </aside>
</main>
<script>
const DATA = {data};
const canvas = document.getElementById('scene'), ctx = canvas.getContext('2d');
const frameInput = document.getElementById('frame'), frameLabel = document.getElementById('frameLabel'), metricsBox = document.getElementById('metrics');
const modeInput = document.getElementById('mode');
let frame = 0, yaw = -0.75, pitch = 0.3, zoom = 1.0, playing = false, timer = null;
let dragging = false, lastX = 0, lastY = 0;
const colors = {{ source: '#3b82f6', warm: '#f59e0b', final: '#22c55e' }};
const meshLayers = {{ source: document.getElementById('meshSource'), warm: document.getElementById('meshWarm'), final: document.getElementById('meshFinal') }};
const graphLayers = {{ source: document.getElementById('graphSource'), warm: document.getElementById('graphWarm'), final: document.getElementById('graphFinal') }};
const edgeLayers = {{ 0: document.getElementById('edgeHH'), 1: document.getElementById('edgeHO'), 2: document.getElementById('edgeOO') }};
const low = DATA.bounds[0], high = DATA.bounds[1], center = [(low[0]+high[0])/2,(low[1]+high[1])/2,(low[2]+high[2])/2];
const extent = Math.max(high[0]-low[0], high[1]-low[1], high[2]-low[2]);
function resize() {{ const ratio=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect(); canvas.width=Math.max(1,Math.floor(rect.width*ratio)); canvas.height=Math.max(1,Math.floor(rect.height*ratio)); ctx.setTransform(ratio,0,0,ratio,0,0); draw(); }}
function rotate(p) {{ let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2]; const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch); const x1=cy*x+sy*z,z1=-sy*x+cy*z; return [x1,cp*y-sp*z1,sp*y+cp*z1]; }}
function project(p,w,h) {{ const q=rotate(p), camera=extent*3.0, scale=Math.min(w,h)*0.72*zoom/Math.max(0.15,camera-q[2]); return [w/2+q[0]*scale,h/2-q[1]*scale,q[2]]; }}
function transformRobot(payload,index) {{ return payload.parts.map(part => {{ const m=part.transforms[index]; return part.vertices.map(p=>[m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]]); }}); }}
function drawMesh(vertices,faces,color,alpha,w,h) {{ const projected=vertices.map(p=>project(p,w,h)), ordered=faces.map(face=>[face,(projected[face[0]][2]+projected[face[1]][2]+projected[face[2]][2])/3]); ordered.sort((a,b)=>a[1]-b[1]); ctx.fillStyle=color;ctx.globalAlpha=alpha; for(const [face] of ordered) {{ const a=projected[face[0]],b=projected[face[1]],c=projected[face[2]];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill(); }} ctx.globalAlpha=1; }}
function drawRobot(payload,index,color,alpha,w,h) {{ const vertices=transformRobot(payload,index); payload.parts.forEach((part,i)=>drawMesh(vertices[i],part.faces,color,alpha,w,h)); }}
function drawObjectContext(index,w,h) {{ const pose=DATA.object.poses[index];ctx.fillStyle='#64748b';ctx.globalAlpha=0.42;for(const p of DATA.object.vertices) {{ const q=[pose[0][0]*p[0]+pose[0][1]*p[1]+pose[0][2]*p[2]+pose[0][3],pose[1][0]*p[0]+pose[1][1]*p[1]+pose[1][2]*p[2]+pose[1][3],pose[2][0]*p[0]+pose[2][1]*p[1]+pose[2][2]*p[2]+pose[2][3]];const s=project(q,w,h);ctx.fillRect(s[0]-1,s[1]-1,2,2);}}ctx.globalAlpha=1; }}
function categoryStyle(category) {{ return category===0 ? [0.25,0.8] : category===1 ? [0.85,1.7] : [0.18,0.55]; }}
function weightColor(weight) {{ const t=Math.max(0,Math.min(1,weight/0.15)), hue=235-235*t; return `hsl(${{hue}},80%,50%)`; }}
function drawEdge(points,edge,color,alpha,width,w,h) {{ const a=project(points[edge[0]],w,h),b=project(points[edge[1]],w,h);ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.globalAlpha=1; }}
function selectedEdges(graphFrame) {{ const threshold=Number(document.getElementById('edgeThreshold').value), topK=Number(document.getElementById('topKEdges').value)||0, only=document.getElementById('handObjectOnly').checked; const ids=[];for(let i=0;i<graphFrame.edges.length;i++) {{ const category=graphFrame.categories[i];if(!edgeLayers[category].checked||only&&category!==1||graphFrame.weights[i]<threshold)continue;ids.push(i); }}if(topK>0&&ids.length>topK)ids.sort((a,b)=>graphFrame.weights[b]-graphFrame.weights[a]);return topK>0?ids.slice(0,topK):ids; }}
function drawGraphState(state,index,w,h) {{ const graphFrame=DATA.interaction.frames[index],points=DATA.interaction.vertices[state][index],base=categoryStyle(0);for(const i of selectedEdges(graphFrame)) {{ const cat=graphFrame.categories[i],style=categoryStyle(cat),weight=graphFrame.weights[i],wm=document.getElementById('weightMode').value;let alpha=style[0],width=style[1],color=colors[state];if(wm==='opacity')alpha*=Math.max(0.12,Math.min(1,weight/0.12));if(wm==='width')width*=0.5+Math.min(2.5,weight/0.05);if(wm==='color')color=weightColor(weight);drawEdge(points,graphFrame.edges[i],color,alpha,width,w,h); }}ctx.globalAlpha=1;ctx.fillStyle=colors[state];for(const p of points){{const s=project(p,w,h);ctx.globalAlpha=0.9;ctx.beginPath();ctx.arc(s[0],s[1],2.5,0,Math.PI*2);ctx.fill();}}ctx.globalAlpha=1;if(document.getElementById('showLabels').checked){{ctx.fillStyle=colors[state];ctx.font='9px monospace';DATA.interaction.vertex_metadata.forEach((meta,i)=>{{const s=project(points[i],w,h);ctx.fillText(String(meta.semantic_name||meta.sample_id||i),s[0]+3,s[1]-3);}});}} }}
function residualMask(index,target) {{ const residual=DATA.interaction.residuals[target][index],scope=document.getElementById('residualScope').value,threshold=Number(document.getElementById('residualThreshold').value),topK=Number(document.getElementById('topKResidual').value)||0,norms=residual.map(v=>Math.hypot(v[0],v[1],v[2])),ids=[];for(let i=0;i<norms.length;i++){{if(scope==='hand'&&i>=21||scope==='object'&&i<21||norms[i]<threshold)continue;ids.push(i);}}ids.sort((a,b)=>norms[b]-norms[a]);return {{residual,norms,ids:topK>0?ids.slice(0,topK):ids}}; }}
function residualColor(t) {{ const hue=235-235*Math.max(0,Math.min(1,t));return `hsl(${{hue}},85%,50%)`; }}
function drawResidual(index,w,h) {{ const target=document.getElementById('residualTarget').value,display=document.getElementById('residualDisplay').value,selected=residualMask(index,target),points=DATA.interaction.vertices[target][index],max=Math.max(...selected.norms,1e-9),scale=4.0;for(const i of selected.ids){{const p=points[i],n=selected.norms[i],s=project(p,w,h);if(display==='scalar'||display==='both'){{ctx.fillStyle=residualColor(n/max);ctx.globalAlpha=0.9;ctx.beginPath();ctx.arc(s[0],s[1],4,0,Math.PI*2);ctx.fill();}}if(display==='vector'||display==='both'){{const q=[p[0]+selected.residual[i][0]*scale,p[1]+selected.residual[i][1]*scale,p[2]+selected.residual[i][2]*scale],e=project(q,w,h);ctx.strokeStyle='#dc2626';ctx.globalAlpha=0.85;ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(s[0],s[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();}}}}ctx.globalAlpha=1;return DATA.interaction.residual_summaries[target][index]; }}
function applyModeDefaults(value) {{ if(value==='figure4-style'){{document.getElementById('handObjectOnly').checked=true;document.getElementById('edgeHH').checked=false;document.getElementById('edgeOO').checked=false;}}else if(value==='full-graph'){{document.getElementById('handObjectOnly').checked=false;document.getElementById('edgeHH').checked=true;document.getElementById('edgeHO').checked=true;document.getElementById('edgeOO').checked=true;}}draw(); }}
function draw() {{ const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,mode=modeInput.value;ctx.clearRect(0,0,w,h);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,w,h);if(document.getElementById('objectContext').checked&&DATA.object.vertices.length)drawObjectContext(frame,w,h);if(mode==='mesh'||mode==='combined'){{if(meshLayers.source.checked)drawMesh(DATA.source.vertices[frame],DATA.source.faces,'#3b82f6',0.30,w,h);if(meshLayers.warm.checked)drawRobot(DATA.warm,frame,'#f59e0b',0.25,w,h);if(meshLayers.final.checked)drawRobot(DATA.final,frame,'#22c55e',0.48,w,h);}}let residualSummary=null;if(mode!=='mesh'){{for(const state of Object.keys(graphLayers))if(graphLayers[state].checked)drawGraphState(state,frame,w,h);if(mode==='laplacian-diagnostic'||mode==='combined')residualSummary=drawResidual(frame,w,h);}}const graphFrame=DATA.interaction.frames[frame],metric=DATA.metrics.frames[frame];frameLabel.textContent=`local ${{metric.local_frame}} · source ${{metric.source_frame}}`;const lines=[`mode: ${{mode}}`,`graph: ${{graphFrame.graph_hash.slice(0,12)}}`,`edges: ${{graphFrame.edges.length}} (HH ${{graphFrame.stats.hand_hand_edge_count}}, HO ${{graphFrame.stats.hand_object_edge_count}}, OO ${{graphFrame.stats.object_object_edge_count}})`];for(const [key,value] of Object.entries(metric))if(!['local_frame','source_frame'].includes(key))lines.push(`${{key}}: ${{typeof value==='number'?value.toPrecision(5):value}}`);if(residualSummary){{lines.push('',`residual target: ${{document.getElementById('residualTarget').value}}`,`residual max: ${{residualSummary.max.toPrecision(5)}}`,`residual mean: ${{residualSummary.mean.toPrecision(5)}}`,`hand mean: ${{residualSummary.hand_mean.toPrecision(5)}}`,`object mean: ${{residualSummary.object_mean.toPrecision(5)}}`,`top vertices: ${{residualSummary.top.map(x=>x.vertex_id).join(', ')}}`);}}metricsBox.textContent=lines.join('\\n');document.getElementById('thresholdValue').textContent=Number(document.getElementById('edgeThreshold').value).toFixed(3);document.getElementById('residualThresholdValue').textContent=Number(document.getElementById('residualThreshold').value).toFixed(4); }}
function drawMeshLayers() {{ const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(meshLayers.source.checked)drawMesh(DATA.source.vertices[frame],DATA.source.faces,'#3b82f6',0.30,w,h);if(meshLayers.warm.checked)drawRobot(DATA.warm,frame,'#f59e0b',0.25,w,h);if(meshLayers.final.checked)drawRobot(DATA.final,frame,'#22c55e',0.48,w,h); }}
const drawBase=draw;
draw=function() {{ drawBase(); if(modeInput.value!=='mesh'&&modeInput.value!=='combined')drawMeshLayers(); }};
function setFrame(value){{frame=Math.max(0,Math.min(DATA.frame_count-1,Number(value)));frameInput.value=frame;draw();}}
modeInput.addEventListener('change',e=>applyModeDefaults(e.target.value));frameInput.addEventListener('input',e=>setFrame(e.target.value));document.querySelectorAll('input,select').forEach(item=>item.addEventListener('change',draw));
document.getElementById('play').addEventListener('click',()=>{{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';if(playing)timer=setInterval(()=>setFrame((frame+1)%DATA.frame_count),100);else clearInterval(timer);}});
canvas.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId);}});canvas.addEventListener('pointermove',e=>{{if(!dragging)return;yaw+=(e.clientX-lastX)*0.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*0.01));lastX=e.clientX;lastY=e.clientY;draw();}});canvas.addEventListener('pointerup',e=>{{dragging=false;canvas.classList.remove('dragging');canvas.releasePointerCapture(e.pointerId);}});canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(0.35,Math.min(4.0,zoom*Math.exp(-e.deltaY*0.001)));draw();}},{{passive:false}});
modeInput.value=DATA.initial_mode||'mesh';
if(modeInput.value==='figure4-style'){{
  document.getElementById('handObjectOnly').checked=true;
  document.getElementById('edgeHH').checked=false;
  document.getElementById('edgeOO').checked=false;
}}else if(modeInput.value==='full-graph'){{
  document.getElementById('handObjectOnly').checked=false;
  document.getElementById('edgeHH').checked=true;
  document.getElementById('edgeHO').checked=true;
  document.getElementById('edgeOO').checked=true;
}}
window.addEventListener('resize',resize);resize();
</script>
</body>
</html>
'''


def render_interaction_mesh_html(
    manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    mode: str = "mesh",
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_object_points: int = 1200,
    asset_root: str | Path | None = None,
    open_browser: bool = False,
) -> dict[str, Any]:
    """Build the unified mesh/graph/residual viewer from immutable artifacts."""

    valid_modes = {"mesh", "full-graph", "figure4-style", "laplacian-diagnostic", "combined"}
    if mode not in valid_modes:
        raise ValueError(f"unsupported HTML visualization mode: {mode}")
    manifest = read_json(manifest_path)
    artifacts = manifest["artifacts"]
    sequence = load_hoi_sequence(artifacts["canonical"]["path"])
    warm = load_warm_start(artifacts["warm_start"]["path"])
    graph = load_interaction_graph(artifacts["graph"]["path"])
    evaluation = load_interaction_evaluation(artifacts["evaluation"]["path"])
    final = load_final_trajectory(artifacts["final"]["path"])
    if not (warm.frame_count == graph.frame_count == evaluation.frame_count == final.frame_count):
        raise ValueError("source/warm/graph/evaluation/final frame counts differ")
    frame_count = final.frame_count
    selected_start = 0 if start_frame is None else int(start_frame)
    selected_end = frame_count if end_frame is None else int(end_frame)
    if selected_start < 0 or selected_end <= selected_start or selected_end > frame_count:
        raise ValueError(f"HTML frame range must be within [0,{frame_count})")
    selected = np.arange(selected_start, selected_end, dtype=np.int64)
    source_local_indices = _source_frame_indices(sequence, final, manifest)
    display_indices = _display_source_indices(final, manifest, frame_count)
    source_vertices_all, source_faces = _source_mesh(sequence, final, manifest)
    source_vertices = source_vertices_all[selected]
    source_local_indices = source_local_indices[selected]
    display_indices = display_indices[selected]
    qpos_warm = warm.arrays["qpos"][selected]
    base_warm = warm.arrays["base_pose_scene"][selected]
    qpos_final = final.arrays["qpos"][selected]
    base_final = final.arrays["base_pose_scene"][selected]
    model = get_robot_registry().load(_robot_name(manifest), asset_root=asset_root or manifest.get("asset_root"))
    warm_payload = _robot_payload(model, qpos_warm, base_warm)
    final_payload = _robot_payload(model, qpos_final, base_final)
    object_payload = _object_payload(sequence, final, source_local_indices, max_object_points)
    interaction = _interaction_payload(
        graph,
        evaluation,
        final.arrays["robot_keypoints_scene"],
        _display_source_indices(final, manifest, frame_count),
        selected,
    )
    object_vertices = np.asarray(object_payload["vertices"], dtype=np.float64)
    object_poses = np.asarray(object_payload["poses"], dtype=np.float64)
    hashes = {
        "canonical": artifact_hash(artifacts["canonical"]["path"]),
        "warm_start": artifact_hash(artifacts["warm_start"]["path"]),
        "graph": interaction_artifact_hash(artifacts["graph"]["path"]),
        "evaluation": interaction_artifact_hash(artifacts["evaluation"]["path"]),
        "final": artifact_hash(artifacts["final"]["path"]),
    }
    payload = {
        "schema_version": INTERACTION_HTML_SCHEMA_VERSION,
        "title": f"TopoRetarget interaction viewer · {manifest.get('source_sequence', manifest.get('run_id', 'run'))}",
        "source_sequence": str(manifest.get("source_sequence", "")),
        "robot": _robot_name(manifest),
        "frame_count": int(len(selected)),
        "initial_mode": mode,
        "source": {"vertices": _rounded(source_vertices), "faces": source_faces.tolist()},
        "warm": warm_payload,
        "final": final_payload,
        "object": object_payload,
        "interaction": interaction,
        "metrics": _metrics(
            type("FinalSlice", (), {"arrays": {k: v[selected] for k, v in final.arrays.items()}, "frame_count": len(selected)})(),
            type("WarmSlice", (), {"arrays": {k: v[selected] for k, v in warm.arrays.items()}})(),
            display_indices,
        ),
        "artifact_hashes": hashes,
        "viewer_provenance": {
            "solver_invocation_count": 0,
            "solver_started": False,
            "stage8_graph_reused": True,
            "stage8_weights_reused": True,
            "stage6_object_sample_identity_reused": True,
            "stage9_artifact_modified": False,
            "frame_range": [int(display_indices[0]), int(display_indices[-1]) + 1],
        },
        "bounds": _bounds(
            source_vertices,
            object_vertices,
            object_poses,
            (warm_payload, final_payload),
        ),
    }
    destination = Path(output) if output is not None else Path(manifest["run_root"]) / "review" / "trajectory_mesh.html"
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_html_document(payload), encoding="utf-8")
    if open_browser:
        webbrowser.open(destination.resolve().as_uri())
    return {
        "status": "pass",
        "schema_version": INTERACTION_HTML_SCHEMA_VERSION,
        "mode": mode,
        "output": str(destination.resolve()),
        "frame_count": int(len(selected)),
        "source_vertices": int(source_vertices.shape[1]),
        "source_faces": int(source_faces.shape[0]),
        "robot_visual_parts": len(warm_payload["parts"]),
        "graph_vertices": int(interaction["vertex_count"]),
        "graph_edges_frame0": int(len(interaction["frames"][0]["edges"])),
        "object_points": int(len(object_vertices)),
        "artifact_hashes": hashes,
        "opened_browser": bool(open_browser),
    }


__all__ = ["INTERACTION_HTML_SCHEMA_VERSION", "render_interaction_mesh_html"]
