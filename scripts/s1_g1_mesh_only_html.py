#!/usr/bin/env python
"""Build a mesh/surface-only G1 E0-vs-S1 review page.

This intentionally omits metric tables and numerical diagnostics.  The input
trajectories are final optimizer trajectories; the page is only a visual
review aid for the completed G1 prescreen candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.workflows.s1_penetration import _visual_data


def build_page(visual: dict, output: Path) -> None:
    encoded = json.dumps(visual, separators=(",", ":"), allow_nan=False)
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>G1 mesh-only E0 vs S1</title>
<style>
html,body{margin:0;background:#111827;color:#e5e7eb;font:14px system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:16px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:10px}
button,select,input{background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;padding:6px}
canvas{display:block;width:100%;height:auto;background:#0b1220;border:1px solid #374151}
.legend{color:#cbd5e1;margin:8px 0}
</style></head><body><main>
<h1>G1 mesh/surface view — E0 vs S1</h1>
<p>Final optimizer trajectories; G1 lambda=.1 prescreen candidate. Visual-only review page.</p>
<div class="controls">
<label>trajectory <select id="mode">
<option value="e0">E0</option><option value="s1">S1</option>
</select></label>
<label>time <input id="frame" type="range" min="0" max="59" value="0"></label>
<button id="play">Play</button><button id="reset">Reset view</button>
<label><input id="overlay" type="checkbox" checked> overlay other trajectory</label>
<label><input id="collision" type="checkbox" checked> collision surface</label>
<label><input id="query" type="checkbox" checked> query set</label>
<label><input id="closest" type="checkbox" checked> closest surface points</label>
<label><input id="penetrating" type="checkbox" checked> penetrating samples</label>
</div>
<div class="legend">object surface purple · source hand cyan · E0 blue · S1 green ·
collision amber/red · query yellow · closest gray</div>
<canvas id="scene" width="1400" height="760"></canvas>
<script>
const V=__VISUAL__;
const $=id=>document.getElementById(id);
let camera={scale:1,ox:0,oy:0},timer=null;
function frame(){return Math.max(0,Math.min(+$('frame').value,V.object.length-1))}
function project(p,all){
  const xs=all.map(q=>q[0]),ys=all.map(q=>q[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const sx=1260/Math.max(maxx-minx,1e-9),sy=620/Math.max(maxy-miny,1e-9);
  const s=.82*Math.min(sx,sy)*camera.scale;
  return [70+(p[0]-minx)*s+camera.ox,700-(p[1]-miny)*s+camera.oy];
}
function pointsFor(i,mode){
  const sets=[V.object[i],V.source[i],V[mode][i]];
  if($('overlay').checked)sets.push(V[mode==='e0'?'s1':'e0'][i]);
  if($('collision').checked)sets.push(V.collision[i]);
  if($('query').checked)sets.push(V.query[i]);
  if($('closest').checked)sets.push(V.closest[i]);
  if($('penetrating').checked)sets.push(V.penetrating[i]);
  return sets;
}
function draw(){
  const i=frame(),mode=$('mode').value,c=$('scene'),ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  const sets=pointsFor(i,mode),all=[].concat(...sets);
  if(!all.length)return;
  const colors=[
    '#a78bfa','#38bdf8',mode==='e0'?'#60a5fa':'#34d399',
    mode==='e0'?'#34d399':'#60a5fa','#f59e0b','#fde047','#9ca3af','#f87171'
  ];
  sets.forEach((pts,k)=>pts.forEach((p,j)=>{
    ctx.fillStyle=k===4?(V.phi[i][j]<0?'#f87171':'#f59e0b'):colors[k];
    const q=project(p,all),r=k===0?2.2:3.2;ctx.fillRect(q[0],q[1],r,r);
  }));
  ctx.fillStyle='#e5e7eb';
  ctx.fillText('G1 · '+mode.toUpperCase()+' · visual-only mesh/surface review',24,28);
}
$('frame').oninput=draw;$('mode').onchange=draw;
['overlay','collision','query','closest','penetrating'].forEach(id=>$(id).onchange=draw);
$('reset').onclick=()=>{camera={scale:1,ox:0,oy:0};draw()};
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return}
  timer=setInterval(()=>{$('frame').value=(frame()+1)%V.object.length;draw()},120);$('play').textContent='Pause';};
draw();
</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page.replace("__VISUAL__", encoded), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    e0 = experiment / "e0/G1/final.zarr"
    s1 = experiment / "artifacts/G1/S1_L01_prescreen/final.zarr"
    if not e0.exists() or not s1.exists():
        raise SystemExit("G1 E0 and S1 prescreen final artifacts are required")
    build_page(_visual_data(experiment, "G1", e0, s1), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
