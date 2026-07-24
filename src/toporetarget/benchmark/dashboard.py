"""Self-contained HTML dashboard for benchmark artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from .schema import read_json


def build_dashboard(benchmark_root: str | Path) -> Path:
    root = Path(benchmark_root)
    rows = (
        read_json(root / "results_per_unit.json")
        if (root / "results_per_unit.json").is_file()
        else []
    )
    paired = (
        read_json(root / "eq9_paired_comparison.json")
        if (root / "eq9_paired_comparison.json").is_file()
        else []
    )
    selection = (
        read_json(root / "benchmark_selection_manifest.json")
        if (root / "benchmark_selection_manifest.json").is_file()
        else (
            read_json(root / "selection_result.json")
            if (root / "selection_result.json").is_file()
            else {}
        )
    )
    status = (
        read_json(root / "benchmark_status.json")
        if (root / "benchmark_status.json").is_file()
        else {}
    )
    data = {"rows": rows, "paired": paired, "selection": selection, "status": status}
    serialized = json.dumps(data, sort_keys=True, default=str).replace("</", "<\\/")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TopoRetarget Q1-Q3 Benchmark</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;background:#f7f7f8;color:#222}} table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ddd;padding:.4rem;text-align:left;font-size:.85rem}} th{{background:#eee}} select{{margin-right:1rem;padding:.3rem}} .warn{{color:#9a4b00}} .pass{{color:#176b32}} pre{{white-space:pre-wrap}}</style>
</head><body><h1>TopoRetarget Q1-Q3 Frozen Benchmark</h1>
<p id="provenance"></p><p id="status"></p><label>Dataset <select id="dataset"><option value="">all</option></select></label>
<label>Profile <select id="profile"><option value="">all</option></select></label>
<label>Contact mode <select id="mode"><option value="">all</option></select></label>
<table><thead><tr><th>Unit</th><th>Dataset</th><th>Object</th><th>Subject</th><th>Hand</th><th>Profile</th><th>Status</th><th>Accepted</th><th>E_IM</th><th>E_bone</th><th>Penetration mm</th><th>Error</th></tr></thead><tbody id="rows"></tbody></table>
<h2>Eq. 9 paired comparison</h2><pre id="paired"></pre>
<h2>Manifest / integrity</h2><pre id="manifest"></pre>
<script>const DATA={serialized};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;", "'":"&#39;"}}[c]));
const fill=(id,values)=>{{const e=document.getElementById(id);[...new Set(values.filter(Boolean))].sort().forEach(v=>e.insertAdjacentHTML('beforeend',`<option>${{esc(v)}}</option>`));}};
fill('dataset',DATA.rows.map(r=>r.dataset));fill('profile',DATA.rows.map(r=>r.profile));fill('mode',DATA.rows.map(r=>r.contact_mode));
function render(){{const d=document.getElementById('dataset').value,p=document.getElementById('profile').value,m=document.getElementById('mode').value;const rows=DATA.rows.filter(r=>(!d||r.dataset===d)&&(!p||r.profile===p)&&(!m||r.contact_mode===m));document.getElementById('rows').innerHTML=rows.map(r=>`<tr><td>${{esc(r.benchmark_id)}}</td><td>${{esc(r.dataset)}}</td><td>${{esc(r.object_name)}}</td><td>${{esc(r.subject)}}</td><td>${{esc(r.hand)}}</td><td>${{esc(r.profile)}}</td><td class="${{r.status==='complete'?'pass':'warn'}}">${{esc(r.status)}}</td><td>${{esc(r.strict_accepted)}}</td><td>${{esc(r.e_im)}}</td><td>${{esc(r.e_bone)}}</td><td>${{esc(r.raw_max_penetration)}}</td><td>${{esc(r.error||r.missing_reason)}}</td></tr>`).join('');}}
['dataset','profile','mode'].forEach(id=>document.getElementById(id).onchange=render);render();document.getElementById('paired').textContent=JSON.stringify(DATA.paired,null,2);document.getElementById('manifest').textContent=JSON.stringify({{manifest_hash:DATA.selection.manifest_hash,selected_units:DATA.selection.selected_units?.length,source_roots:DATA.selection.dataset_roots}},null,2);document.getElementById('provenance').textContent=`Selection manifest: ${{DATA.selection.manifest_hash||'not frozen'}}; results are not used to change selection.`;document.getElementById('status').textContent=`Status: ${{DATA.status.status||'unknown'}}`;</script>
</body></html>"""
    destination = root / "benchmark_dashboard.html"
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["build_dashboard"]
