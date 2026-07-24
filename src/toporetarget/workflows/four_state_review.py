"""Self-contained four-state Stage 9 human-review HTML."""

# The embedded HTML/JavaScript is intentionally readable and is not Python code.
# ruff: noqa: E501

from __future__ import annotations

import json
import math
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
    _bounds,
    _display_source_indices,
    _interaction_payload,
    _object_payload,
    _robot_name,
    _robot_payload,
    _rounded,
    _source_frame_indices,
    _source_mesh,
)
from .schema import read_json

FOUR_STATE_REVIEW_SCHEMA_VERSION = "toporetarget.stage9_four_state_review.v1"
_REQUIRED_REVIEW_FRAMES = (0, 10, 30, 36, 39)


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(second[:3, :3]) @ np.asarray(first[:3, :3]).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def _motion_metrics(trajectory: Any) -> dict[str, np.ndarray]:
    base = np.asarray(trajectory.arrays["base_pose_scene"], dtype=np.float64)
    qpos = np.asarray(trajectory.arrays["qpos"], dtype=np.float64)
    count = trajectory.frame_count
    translation = np.zeros(count, dtype=np.float64)
    rotation = np.zeros(count, dtype=np.float64)
    q_step = np.zeros(count, dtype=np.float64)
    if count > 1:
        translation[1:] = np.linalg.norm(np.diff(base[:, :3, 3], axis=0), axis=1)
        q_step[1:] = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
        rotation[1:] = [_rotation_angle(base[index - 1], base[index]) for index in range(1, count)]
    return {
        "base_translation_step_mm": translation * 1000.0,
        "base_rotation_step_rad": rotation,
        "q_step_rad": q_step,
    }


def build_review_keyframes(
    frames: list[dict[str, Any]],
    *,
    required_frames: tuple[int, ...] = _REQUIRED_REVIEW_FRAMES,
) -> list[dict[str, Any]]:
    """Return required and metric-worst frames with explicit reasons."""

    if not frames:
        return []
    by_local = {int(row["local_frame"]): row for row in frames}
    reasons: dict[int, list[str]] = {}

    def add(frame: int, reason: str) -> None:
        if frame in by_local:
            reasons.setdefault(frame, [])
            if reason not in reasons[frame]:
                reasons[frame].append(reason)

    for frame in required_frames:
        add(frame, "required selected frame")
    add(min(by_local), "first frame")
    add(max(by_local), "last frame")

    metrics = (
        ("old_long_finger_rmse_mm", "old long-finger RMSE max"),
        ("fixed_long_finger_rmse_mm", "fixed long-finger RMSE max"),
        ("old_weighted_e_im", "old weighted E_IM max"),
        ("fixed_weighted_e_im", "fixed weighted E_IM max"),
        ("old_weighted_e_bone", "old weighted E_bone max"),
        ("fixed_weighted_e_bone", "fixed weighted E_bone max"),
        ("old_base_translation_step_mm", "old base translation step max"),
        ("fixed_base_translation_step_mm", "fixed base translation step max"),
        ("old_base_rotation_step_rad", "old base rotation step max"),
        ("fixed_base_rotation_step_rad", "fixed base rotation step max"),
        ("old_q_step_rad", "old q step max"),
        ("fixed_q_step_rad", "fixed q step max"),
    )
    for key, label in metrics:
        candidates = [(int(row["local_frame"]), _safe_number(row.get(key))) for row in frames]
        finite: list[tuple[int, float]] = []
        for frame, value in candidates:
            if value is not None:
                finite.append((frame, value))
        if finite:
            add(max(finite, key=lambda item: item[1])[0], label)

    for key, label in (
        ("old_contact_proxy", "old contact proxy max"),
        ("fixed_contact_proxy", "fixed contact proxy max"),
    ):
        candidates = [(int(row["local_frame"]), _safe_number(row.get(key))) for row in frames]
        finite = []
        for frame, value in candidates:
            if value is not None:
                finite.append((frame, value))
        if not finite:
            continue
        values = np.asarray([value for _, value in finite], dtype=np.float64)
        if float(np.ptp(values)) <= 1e-15:
            add(min(by_local), f"{label}; all frames tied at {values[0]:.6g}")
        else:
            add(max(finite, key=lambda item: item[1])[0], label)

    return [
        {
            "viewer_frame": int(by_local[frame]["viewer_frame"]),
            "local_frame": frame,
            "global_frame": int(by_local[frame]["global_frame"]),
            "reasons": reasons[frame],
        }
        for frame in sorted(reasons)
    ]


def _comparison_frames(
    old: Any,
    fixed: Any,
    comparison: dict[str, Any],
    display_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows = comparison.get("rows", [])
    if len(rows) != old.frame_count or fixed.frame_count != old.frame_count:
        raise ValueError("comparison rows and old/fixed frame counts differ")
    old_motion = _motion_metrics(old)
    fixed_motion = _motion_metrics(fixed)
    old_accepted = np.asarray(
        old.arrays.get("accepted", np.ones(old.frame_count, dtype=bool)), dtype=bool
    )
    fixed_accepted = np.asarray(
        fixed.arrays.get("accepted", np.ones(fixed.frame_count, dtype=bool)), dtype=bool
    )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        old_rmse = _safe_number(row.get("baseline_long_finger_rmse_m"))
        fixed_rmse = _safe_number(row.get("repaired_long_finger_rmse_m"))
        old_im = _safe_number(row.get("baseline_e_im"))
        fixed_im = _safe_number(row.get("repaired_e_im"))
        old_bone = _safe_number(row.get("baseline_e_bone"))
        fixed_bone = _safe_number(row.get("repaired_e_bone"))
        old_contact = _safe_number(row.get("baseline_contact_proxy"))
        fixed_contact = _safe_number(row.get("repaired_contact_proxy"))
        old_penetration = _safe_number(row.get("baseline_raw_penetration_m"))
        fixed_penetration = _safe_number(row.get("repaired_raw_penetration_m"))
        result.append(
            {
                "viewer_frame": index,
                "local_frame": index,
                "global_frame": int(display_indices[index]),
                "old_long_finger_rmse_mm": None if old_rmse is None else old_rmse * 1000.0,
                "fixed_long_finger_rmse_mm": None if fixed_rmse is None else fixed_rmse * 1000.0,
                "delta_long_finger_rmse_mm": None
                if old_rmse is None or fixed_rmse is None
                else (fixed_rmse - old_rmse) * 1000.0,
                "old_weighted_e_im": old_im,
                "fixed_weighted_e_im": fixed_im,
                "delta_weighted_e_im": None
                if old_im is None or fixed_im is None
                else fixed_im - old_im,
                "old_weighted_e_bone": old_bone,
                "fixed_weighted_e_bone": fixed_bone,
                "delta_weighted_e_bone": None
                if old_bone is None or fixed_bone is None
                else fixed_bone - old_bone,
                "old_contact_proxy": old_contact,
                "fixed_contact_proxy": fixed_contact,
                "old_raw_penetration_mm": None
                if old_penetration is None
                else old_penetration * 1000.0,
                "fixed_raw_penetration_mm": None
                if fixed_penetration is None
                else fixed_penetration * 1000.0,
                "old_base_translation_step_mm": float(
                    old_motion["base_translation_step_mm"][index]
                ),
                "fixed_base_translation_step_mm": float(
                    fixed_motion["base_translation_step_mm"][index]
                ),
                "old_base_rotation_step_rad": float(old_motion["base_rotation_step_rad"][index]),
                "fixed_base_rotation_step_rad": float(
                    fixed_motion["base_rotation_step_rad"][index]
                ),
                "old_q_step_rad": float(old_motion["q_step_rad"][index]),
                "fixed_q_step_rad": float(fixed_motion["q_step_rad"][index]),
                "old_accepted": bool(old_accepted[index]),
                "fixed_accepted": bool(fixed_accepted[index]),
            }
        )
    return result


def _read_required_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"review report must be a JSON object: {path}")
    return value


def _html_document(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#0f172a;color:#e2e8f0}
main{display:grid;grid-template-columns:minmax(0,1fr) 450px;height:100vh}
#view{min-width:0;min-height:0;position:relative}
#scene{display:block;width:100%;height:100%;background:#f8fafc;cursor:grab}
#scene.dragging{cursor:grabbing}
aside{overflow:auto;padding:14px;background:#172033;border-left:1px solid #334155}
h1{font-size:18px;margin:0 0 8px}h2{font-size:13px;color:#93c5fd;margin:16px 0 7px}
.banner{padding:9px;border-radius:6px;background:#7f1d1d;color:#fecaca;font-weight:700;font-size:12px}
.hint,.small{font-size:11px;color:#94a3b8;line-height:1.45}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:5px 10px}
label{font-size:12px;display:block;margin:4px 0}
select,input[type=number],textarea{width:100%;background:#0f172a;color:#e2e8f0;border:1px solid #475569;border-radius:4px;padding:5px}
input[type=range]{width:100%}
button{border:0;border-radius:5px;padding:5px 9px;background:#2563eb;color:#fff;cursor:pointer;margin:2px}
button.key{background:#334155;font:11px ui-monospace,monospace}
button.key.active{background:#7c3aed}
.legend{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
table{width:100%;border-collapse:collapse;font-size:10px}
th,td{padding:4px;border:1px solid #334155;text-align:right}
th:first-child,td:first-child{text-align:left}
#timeline{width:100%;height:110px;background:#0f172a;border:1px solid #334155}
pre{white-space:pre-wrap;font:10px ui-monospace,monospace;color:#cbd5e1}
.check{display:flex;gap:6px;align-items:flex-start;margin:7px 0;font-size:11px;line-height:1.35}
.check input{margin-top:2px}
.status-pass{color:#86efac}.status-fail{color:#fca5a5}
</style>
</head>
<body>
<main>
<section id="view"><canvas id="scene"></canvas></section>
<aside>
<h1>__TITLE__</h1>
<div class="banner">Stage 9 improvement gate: <span id="decision"></span> · faithful finalization requires human signoff</div>
<div class="hint">Four immutable states in one page. Drag to orbit · wheel to zoom · Space play/pause · Left/Right step.</div>

<h2>Frame and worst-frame navigation</h2>
<input id="frame" type="range" min="0" max="__MAX_FRAME__" value="0" step="1">
<div><span id="frameLabel"></span><button id="play">Play</button></div>
<div id="keyframes"></div>
<div id="keyReason" class="small"></div>

<h2>View</h2>
<div class="grid">
<label>mode<select id="mode"><option value="mesh">mesh</option><option value="interaction">interaction</option><option value="difference">old vs fixed difference</option><option value="combined">combined</option></select></label>
<label>finger<select id="finger"><option value="all">all</option><option value="thumb">thumb</option><option value="index">index</option><option value="middle">middle</option><option value="ring">ring</option><option value="pinky">pinky</option></select></label>
</div>

<h2>Mesh layers</h2>
<div class="grid">
<label><input id="meshSource" type="checkbox" checked><span class="legend" style="background:#3b82f6"></span>source MANO</label>
<label><input id="meshWarm" type="checkbox" checked><span class="legend" style="background:#f59e0b"></span>Stage 7 warm</label>
<label><input id="meshOld" type="checkbox" checked><span class="legend" style="background:#22c55e"></span>old current final</label>
<label><input id="meshFixed" type="checkbox" checked><span class="legend" style="background:#d946ef"></span>faithful fixed final</label>
<label><input id="objectContext" type="checkbox" checked><span class="legend" style="background:#64748b"></span>object</label>
<label><input id="differenceVectors" type="checkbox" checked>old→fixed vectors</label>
</div>

<h2>Interaction and base</h2>
<div class="grid">
<label><input id="graphSource" type="checkbox">source graph</label>
<label><input id="graphWarm" type="checkbox">warm graph</label>
<label><input id="graphOld" type="checkbox" checked>old graph</label>
<label><input id="graphFixed" type="checkbox" checked>fixed graph</label>
<label><input id="handObjectOnly" type="checkbox" checked>hand-object edges only</label>
<label><input id="showLabels" type="checkbox">vertex labels</label>
<label><input id="baseOld" type="checkbox" checked>old base axes</label>
<label><input id="baseFixed" type="checkbox" checked>fixed base axes</label>
</div>

<h2>60-frame timeline</h2>
<select id="timelineMetric">
<option value="old_long_finger_rmse_mm">old long-finger RMSE mm</option>
<option value="fixed_long_finger_rmse_mm">fixed long-finger RMSE mm</option>
<option value="delta_long_finger_rmse_mm">fixed-old long-finger delta mm</option>
<option value="old_weighted_e_im">old weighted E_IM</option>
<option value="fixed_weighted_e_im">fixed weighted E_IM</option>
<option value="old_base_translation_step_mm">old base translation step mm</option>
<option value="fixed_base_translation_step_mm">fixed base translation step mm</option>
<option value="old_base_rotation_step_rad">old base rotation step rad</option>
<option value="fixed_base_rotation_step_rad">fixed base rotation step rad</option>
<option value="old_q_step_rad">old q step rad</option>
<option value="fixed_q_step_rad">fixed q step rad</option>
<option value="old_contact_proxy">old contact proxy</option>
<option value="fixed_contact_proxy">fixed contact proxy</option>
</select>
<canvas id="timeline" width="420" height="110"></canvas>

<h2>Current-frame metrics</h2>
<table><thead><tr><th>metric</th><th>old</th><th>fixed</th><th>delta</th></tr></thead><tbody id="metricRows"></tbody></table>
<pre id="frameDetails"></pre>

<h2>Per-finger aggregate comparison</h2>
<table><thead><tr><th>finger</th><th>old mm</th><th>fixed mm</th><th>delta mm</th></tr></thead><tbody id="fingerRows"></tbody></table>

<h2>Human acceptance checklist</h2>
<div id="checklist"></div>
<label>review notes<textarea id="notes" rows="5" placeholder="Record local/global frame, finger/link, and observed defect."></textarea></label>
<button id="copyReview">Copy review record</button><span id="copyStatus" class="small"></span>

<h2>Boundaries</h2>
<div class="hint">Source MANO and Arti-MANO have different morphology: judge semantic contact, fingertip/pad direction, object-relative motion, penetration, and continuity—not vertex overlap. Contact proxy is diagnostic, not ground truth. This viewer invokes no solver and modifies no input artifact.</div>
<h2>Provenance</h2><pre id="provenance"></pre>
</aside>
</main>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const canvas=$('scene'),ctx=canvas.getContext('2d'),timeline=$('timeline'),tctx=timeline.getContext('2d');
const colors={source:'#3b82f6',warm:'#f59e0b',old:'#22c55e',fixed:'#d946ef'};
const fingerIndices={all:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],thumb:[1,2,3,4],index:[5,6,7,8],middle:[9,10,11,12],ring:[13,14,15,16],pinky:[17,18,19,20]};
let frame=0,yaw=-0.75,pitch=0.3,zoom=1,playing=false,timer=null,dragging=false,lastX=0,lastY=0;
const low=DATA.bounds[0],high=DATA.bounds[1],center=[(low[0]+high[0])/2,(low[1]+high[1])/2,(low[2]+high[2])/2],extent=Math.max(high[0]-low[0],high[1]-low[1],high[2]-low[2]);
function resize(){const ratio=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.floor(rect.width*ratio));canvas.height=Math.max(1,Math.floor(rect.height*ratio));ctx.setTransform(ratio,0,0,ratio,0,0);draw()}
function rotate(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return[x1,cp*y-sp*z1,sp*y+cp*z1]}
function project(p,w,h){const q=rotate(p),camera=extent*3,scale=Math.min(w,h)*0.72*zoom/Math.max(0.15,camera-q[2]);return[w/2+q[0]*scale,h/2-q[1]*scale,q[2]]}
function transformRobot(payload,index){return payload.parts.map(part=>{const m=part.transforms[index];return part.vertices.map(p=>[m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]])})}
function drawMesh(vertices,faces,color,alpha,w,h){const projected=vertices.map(p=>project(p,w,h)),ordered=faces.map(face=>[face,(projected[face[0]][2]+projected[face[1]][2]+projected[face[2]][2])/3]);ordered.sort((a,b)=>a[1]-b[1]);ctx.fillStyle=color;ctx.globalAlpha=alpha;for(const[face]of ordered){const a=projected[face[0]],b=projected[face[1]],c=projected[face[2]];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill()}ctx.globalAlpha=1}
function drawRobot(payload,index,color,alpha,w,h){const vertices=transformRobot(payload,index);payload.parts.forEach((part,i)=>drawMesh(vertices[i],part.faces,color,alpha,w,h))}
function drawObject(index,w,h){const pose=DATA.object.poses[index];ctx.fillStyle='#64748b';ctx.globalAlpha=.45;for(const p of DATA.object.vertices){const q=[pose[0][0]*p[0]+pose[0][1]*p[1]+pose[0][2]*p[2]+pose[0][3],pose[1][0]*p[0]+pose[1][1]*p[1]+pose[1][2]*p[2]+pose[1][3],pose[2][0]*p[0]+pose[2][1]*p[1]+pose[2][2]*p[2]+pose[2][3]],s=project(q,w,h);ctx.fillRect(s[0]-1,s[1]-1,2,2)}ctx.globalAlpha=1}
function drawEdge(points,edge,color,alpha,width,w,h){const a=project(points[edge[0]],w,h),b=project(points[edge[1]],w,h);ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.globalAlpha=1}
function drawGraph(state,w,h){const gf=DATA.interaction.frames[frame],points=DATA.interaction.vertices[state][frame],finger=new Set(fingerIndices[$('finger').value]);for(let i=0;i<gf.edges.length;i++){const edge=gf.edges[i],category=gf.categories[i];if($('handObjectOnly').checked&&category!==1)continue;if(category===0&&(!finger.has(edge[0])||!finger.has(edge[1])))continue;if(category===1&&!finger.has(edge[0]))continue;drawEdge(points,edge,colors[state],category===1?.65:.25,category===1?1.5:.7,w,h)}ctx.fillStyle=colors[state];for(const i of fingerIndices[$('finger').value]){const s=project(points[i],w,h);ctx.beginPath();ctx.arc(s[0],s[1],2.6,0,Math.PI*2);ctx.fill();if($('showLabels').checked){const meta=DATA.interaction.vertex_metadata[i]||{};ctx.font='9px monospace';ctx.fillText(String(meta.semantic_name||i),s[0]+3,s[1]-3)}}}
function drawDifference(w,h){if(!$('differenceVectors').checked)return;const old=DATA.interaction.vertices.old[frame],fixed=DATA.interaction.vertices.fixed[frame];ctx.setLineDash([4,3]);for(const i of fingerIndices[$('finger').value]){const a=project(old[i],w,h),b=project(fixed[i],w,h);ctx.strokeStyle='#be123c';ctx.lineWidth=1.5;ctx.globalAlpha=.9;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}ctx.setLineDash([]);ctx.globalAlpha=1}
function transformPoint(m,p){return[m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]]}
function drawBase(state,w,h){const m=DATA.base_poses[state][frame],origin=transformPoint(m,[0,0,0]),axes=[[.025,0,0],[0,.025,0],[0,0,.025]],axisColors=['#ef4444','#22c55e','#3b82f6'];ctx.setLineDash(state==='fixed'?[4,2]:[]);axes.forEach((p,i)=>{const a=project(origin,w,h),b=project(transformPoint(m,p),w,h);ctx.strokeStyle=axisColors[i];ctx.globalAlpha=state==='fixed'?1:.65;ctx.lineWidth=state==='fixed'?2:1;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()});ctx.setLineDash([]);ctx.globalAlpha=1}
function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,mode=$('mode').value;ctx.clearRect(0,0,w,h);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,w,h);if($('objectContext').checked)drawObject(frame,w,h);if(mode!=='interaction'){if($('meshSource').checked)drawMesh(DATA.source.vertices[frame],DATA.source.faces,colors.source,.25,w,h);if($('meshWarm').checked)drawRobot(DATA.warm,frame,colors.warm,.18,w,h);if($('meshOld').checked)drawRobot(DATA.old,frame,colors.old,.42,w,h);if($('meshFixed').checked)drawRobot(DATA.fixed,frame,colors.fixed,.34,w,h)}if(mode!=='mesh'){for(const state of['source','warm','old','fixed'])if($('graph'+state[0].toUpperCase()+state.slice(1)).checked)drawGraph(state,w,h);drawDifference(w,h)}if($('baseOld').checked)drawBase('old',w,h);if($('baseFixed').checked)drawBase('fixed',w,h);updatePanels()}
function format(v){return v===null||v===undefined?'N/A':typeof v==='number'?(Math.abs(v)>=100?v.toFixed(2):Math.abs(v)>=1?v.toFixed(4):v.toExponential(4)):String(v)}
function updatePanels(){const r=DATA.metrics.frames[frame];$('frameLabel').textContent=`local ${r.local_frame} · global ${r.global_frame}`;const rows=[['long-finger RMSE mm','old_long_finger_rmse_mm','fixed_long_finger_rmse_mm'],['weighted E_IM','old_weighted_e_im','fixed_weighted_e_im'],['weighted E_bone','old_weighted_e_bone','fixed_weighted_e_bone'],['base translation step mm','old_base_translation_step_mm','fixed_base_translation_step_mm'],['base rotation step rad','old_base_rotation_step_rad','fixed_base_rotation_step_rad'],['q step rad','old_q_step_rad','fixed_q_step_rad'],['contact proxy','old_contact_proxy','fixed_contact_proxy'],['raw penetration mm','old_raw_penetration_mm','fixed_raw_penetration_mm']];$('metricRows').innerHTML=rows.map(([name,a,b])=>{const av=r[a],bv=r[b],d=typeof av==='number'&&typeof bv==='number'?bv-av:null;return`<tr><td>${name}</td><td>${format(av)}</td><td>${format(bv)}</td><td>${format(d)}</td></tr>`}).join('');$('frameDetails').textContent=`old accepted: ${r.old_accepted}\nfixed accepted: ${r.fixed_accepted}`;const k=DATA.keyframes.find(x=>x.local_frame===r.local_frame);$('keyReason').textContent=k?k.reasons.join(' · '):'';document.querySelectorAll('button.key').forEach(x=>x.classList.toggle('active',Number(x.dataset.frame)===frame));drawTimeline()}
function setFrame(v){frame=Math.max(0,Math.min(DATA.frame_count-1,Number(v)));$('frame').value=frame;draw()}
function drawTimeline(){const metric=$('timelineMetric').value,values=DATA.metrics.frames.map(x=>Number(x[metric]??0)),mn=Math.min(...values),mx=Math.max(...values),w=timeline.width,h=timeline.height;tctx.clearRect(0,0,w,h);tctx.fillStyle='#0f172a';tctx.fillRect(0,0,w,h);for(const k of DATA.keyframes){const x=k.viewer_frame/Math.max(values.length-1,1)*w;tctx.strokeStyle='#475569';tctx.beginPath();tctx.moveTo(x,0);tctx.lineTo(x,h);tctx.stroke()}tctx.strokeStyle=metric.startsWith('fixed')?'#d946ef':'#22c55e';tctx.lineWidth=2;tctx.beginPath();values.forEach((v,i)=>{const x=i/Math.max(values.length-1,1)*w,y=h-8-(v-mn)/Math.max(mx-mn,1e-12)*(h-22);i?tctx.lineTo(x,y):tctx.moveTo(x,y)});tctx.stroke();const x=frame/Math.max(values.length-1,1)*w;tctx.fillStyle='#fbbf24';tctx.fillRect(x-1,0,3,h);tctx.fillStyle='#cbd5e1';tctx.font='10px monospace';tctx.fillText(`${metric} [${format(mn)}, ${format(mx)}]`,6,11)}
function buildKeyframes(){$('keyframes').innerHTML=DATA.keyframes.map(k=>`<button class="key" data-frame="${k.viewer_frame}" title="${k.reasons.join(' | ')}">${k.local_frame}</button>`).join('');document.querySelectorAll('button.key').forEach(b=>b.onclick=()=>setFrame(b.dataset.frame))}
function buildFingerRows(){const order=['thumb','index','middle','ring','pinky'];$('fingerRows').innerHTML=order.map(name=>{const r=DATA.per_finger[name];return`<tr><td>${name}</td><td>${format(r.baseline_rmse_mm)}</td><td>${format(r.repaired_rmse_mm)}</td><td>${format(r.delta_mm)}</td></tr>`}).join('')}
function buildChecklist(){const items=['Thumb opposition/contact is preserved','Index and middle maintain the intended object-surface relationship','Contact links and hand-object edges are semantically correct','No visible floating, sliding, or mesh penetration','No unexplained base drift or source-hand lag','No base rotation jitter or sudden translation','Thumb/index/middle/ring/pinky remain anatomically plausible','No frame-to-frame joint jump or contact switch','Required and metric-worst frames were all inspected','Manual acceptance is not marked pass until a human reviewer signs the record'];$('checklist').innerHTML=items.map((x,i)=>`<label class="check"><input type="checkbox" data-check="${i}"><span>${x}</span></label>`).join('')}
function reviewRecord(){const checked=[...document.querySelectorAll('[data-check]')].map((x,i)=>({item:i+1,pass:x.checked}));return JSON.stringify({schema_version:'toporetarget.stage9_human_visual_review.v1',machine_decision:DATA.decision.final_status,recommended_profile:DATA.decision.recommended_profile,reviewed_keyframes:DATA.keyframes.map(x=>x.local_frame),checks:checked,notes:$('notes').value,human_manual_acceptance:'pending_until_saved_by_reviewer'},null,2)}
function applyUrlReviewPreset(){const p=new URLSearchParams(window.location.search),setSelect=(id,value)=>{if(value!==null&&[...$(id).options].some(x=>x.value===value))$(id).value=value};setSelect('mode',p.get('mode'));setSelect('finger',p.get('finger'));setSelect('timelineMetric',p.get('metric'));for(const [name,setter]of[['frame',v=>frame=Math.max(0,Math.min(DATA.frame_count-1,Number(v)))],['yaw',v=>yaw=Number(v)],['pitch',v=>pitch=Number(v)],['zoom',v=>zoom=Math.max(.35,Math.min(4,Number(v)))]]){const value=p.get(name);if(value!==null&&Number.isFinite(Number(value)))setter(value)}$('frame').value=frame}
$('copyReview').onclick=async()=>{await navigator.clipboard.writeText(reviewRecord());$('copyStatus').textContent=' copied'};
$('frame').oninput=e=>setFrame(e.target.value);$('timelineMetric').onchange=drawTimeline;timeline.onclick=e=>setFrame(Math.round(e.offsetX/timeline.clientWidth*(DATA.frame_count-1)));document.querySelectorAll('input,select').forEach(x=>x.addEventListener('change',draw));
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';if(playing)timer=setInterval(()=>setFrame((frame+1)%DATA.frame_count),100);else clearInterval(timer)};
window.onkeydown=e=>{if(e.code==='Space'){e.preventDefault();$('play').click()}else if(e.code==='ArrowLeft')setFrame(frame-1);else if(e.code==='ArrowRight')setFrame(frame+1)};
canvas.onpointerdown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*.01));lastX=e.clientX;lastY=e.clientY;draw()};canvas.onpointerup=e=>{dragging=false;canvas.classList.remove('dragging');canvas.releasePointerCapture(e.pointerId)};canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
$('decision').textContent=DATA.decision.final_status;$('provenance').textContent=JSON.stringify(DATA.viewer_provenance,null,2);buildKeyframes();buildFingerRows();buildChecklist();applyUrlReviewPreset();window.onresize=resize;resize();
</script>
</body>
</html>"""
    return (
        template.replace("__TITLE__", str(payload["title"]))
        .replace("__MAX_FRAME__", str(payload["frame_count"] - 1))
        .replace("__DATA__", data)
    )


def render_four_state_review_html(
    manifest_path: str | Path,
    *,
    old_final: str | Path,
    comparison_final: str | Path,
    review_report_root: str | Path,
    output: str | Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_object_points: int = 1200,
    asset_root: str | Path | None = None,
    open_browser: bool = False,
) -> dict[str, Any]:
    """Build one source/warm/old/fixed acceptance viewer without running a solver."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    old_final = Path(old_final).expanduser().resolve()
    comparison_final = Path(comparison_final).expanduser().resolve()
    report_root = Path(review_report_root).expanduser().resolve()
    manifest = read_json(manifest_path)
    artifacts = manifest["artifacts"]
    sequence = load_hoi_sequence(artifacts["canonical"]["path"])
    warm = load_warm_start(artifacts["warm_start"]["path"])
    graph = load_interaction_graph(artifacts["graph"]["path"])
    evaluation = load_interaction_evaluation(artifacts["evaluation"]["path"])
    old = load_final_trajectory(old_final)
    fixed = load_final_trajectory(comparison_final)
    if not (
        warm.frame_count
        == graph.frame_count
        == evaluation.frame_count
        == old.frame_count
        == fixed.frame_count
    ):
        raise ValueError("source/warm/graph/evaluation/old/fixed frame counts differ")
    old_source = _source_frame_indices(sequence, old, manifest)
    fixed_source = _source_frame_indices(sequence, fixed, manifest)
    if not np.array_equal(old_source, fixed_source):
        raise ValueError("old and fixed source-frame mappings differ")

    frame_count = old.frame_count
    selected_start = 0 if start_frame is None else int(start_frame)
    selected_end = frame_count if end_frame is None else int(end_frame)
    if selected_start < 0 or selected_end <= selected_start or selected_end > frame_count:
        raise ValueError(f"HTML frame range must be within [0,{frame_count})")
    selected = np.arange(selected_start, selected_end, dtype=np.int64)
    display_indices_all = _display_source_indices(old, manifest, frame_count)
    display_indices = display_indices_all[selected]
    source_vertices_all, source_faces = _source_mesh(sequence, old, manifest)
    source_vertices = source_vertices_all[selected]
    model = get_robot_registry().load(
        _robot_name(manifest), asset_root=asset_root or manifest.get("asset_root")
    )
    warm_payload = _robot_payload(
        model,
        np.asarray(warm.arrays["qpos"])[selected],
        np.asarray(warm.arrays["base_pose_scene"])[selected],
    )
    old_payload = _robot_payload(
        model,
        np.asarray(old.arrays["qpos"])[selected],
        np.asarray(old.arrays["base_pose_scene"])[selected],
    )
    fixed_payload = _robot_payload(
        model,
        np.asarray(fixed.arrays["qpos"])[selected],
        np.asarray(fixed.arrays["base_pose_scene"])[selected],
    )
    object_payload = _object_payload(sequence, old, old_source[selected], max_object_points)
    old_interaction = _interaction_payload(
        graph,
        evaluation,
        np.asarray(old.arrays["robot_keypoints_scene"]),
        display_indices_all,
        selected,
    )
    fixed_interaction = _interaction_payload(
        graph,
        evaluation,
        np.asarray(fixed.arrays["robot_keypoints_scene"]),
        display_indices_all,
        selected,
    )
    for name in ("vertices", "residuals", "residual_summaries"):
        old_interaction[name]["old"] = old_interaction[name].pop("final")
        old_interaction[name]["fixed"] = fixed_interaction[name]["final"]

    comparison = _read_required_json(report_root, "repaired_vs_baselines.json")
    decision = _read_required_json(report_root, "stage9_final_decision.json")
    validation = _read_required_json(report_root, "repaired_60f_validation.json")
    regression = _read_required_json(report_root, "bounded_regression.json")
    all_frames = _comparison_frames(old, fixed, comparison, display_indices_all)
    selected_frames: list[dict[str, Any]] = []
    for viewer_frame, local_frame in enumerate(selected.tolist()):
        item = dict(all_frames[local_frame])
        item["viewer_frame"] = viewer_frame
        selected_frames.append(item)
    keyframes = build_review_keyframes(selected_frames)
    object_vertices = np.asarray(object_payload["vertices"], dtype=np.float64)
    object_poses = np.asarray(object_payload["poses"], dtype=np.float64)
    output_path = (
        Path(output) if output is not None else report_root / "stage9_four_state_visual_review.html"
    ).expanduser()
    payload = {
        "schema_version": FOUR_STATE_REVIEW_SCHEMA_VERSION,
        "title": "Stage 9 four-state causal closure human review",
        "frame_count": int(len(selected)),
        "source_sequence": str(manifest.get("source_sequence", "")),
        "robot": _robot_name(manifest),
        "source": {"vertices": _rounded(source_vertices), "faces": source_faces.tolist()},
        "warm": warm_payload,
        "old": old_payload,
        "fixed": fixed_payload,
        "object": object_payload,
        "interaction": old_interaction,
        "base_poses": {
            "old": _rounded(np.asarray(old.arrays["base_pose_scene"])[selected], digits=8),
            "fixed": _rounded(np.asarray(fixed.arrays["base_pose_scene"])[selected], digits=8),
        },
        "metrics": {"frames": selected_frames},
        "keyframes": keyframes,
        "per_finger": comparison["per_finger"],
        "quality_gate": comparison["gate"],
        "decision": decision,
        "validation": validation,
        "bounded_regression": regression,
        "artifact_hashes": {
            "canonical": artifact_hash(artifacts["canonical"]["path"]),
            "warm": artifact_hash(artifacts["warm_start"]["path"]),
            "graph": interaction_artifact_hash(artifacts["graph"]["path"]),
            "evaluation": interaction_artifact_hash(artifacts["evaluation"]["path"]),
            "old_final": artifact_hash(old_final),
            "fixed_final": artifact_hash(comparison_final),
        },
        "viewer_provenance": {
            "manifest": str(manifest_path),
            "old_final": str(old_final),
            "fixed_final": str(comparison_final),
            "review_report_root": str(report_root),
            "solver_invocation_count": 0,
            "solver_started": False,
            "inputs_modified": False,
            "frame_range": [int(display_indices[0]), int(display_indices[-1]) + 1],
            "projection_is_paper_method": False,
            "contact_proxy_is_ground_truth": False,
        },
        "bounds": _bounds(
            source_vertices,
            object_vertices,
            object_poses,
            (warm_payload, old_payload, fixed_payload),
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(payload), encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return {
        "status": "pass",
        "schema_version": FOUR_STATE_REVIEW_SCHEMA_VERSION,
        "output": str(output_path.resolve()),
        "frame_count": int(len(selected)),
        "keyframes": keyframes,
        "artifact_hashes": payload["artifact_hashes"],
        "machine_decision": decision["final_status"],
        "recommended_profile": decision["recommended_profile"],
        "opened_browser": bool(open_browser),
    }


__all__ = [
    "FOUR_STATE_REVIEW_SCHEMA_VERSION",
    "build_review_keyframes",
    "render_four_state_review_html",
]
