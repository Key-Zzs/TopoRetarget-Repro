#!/usr/bin/env python3
"""Render the fail-closed Stage 16-C.5 R4 Oracle dashboard as standalone HTML."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2", type=Path, action="append", required=True)
    parser.add_argument("--distribution-gate", type=Path, required=True)
    parser.add_argument("--pool-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dashboard input is not an object: {path}")
    return payload


def _polyline(values: Sequence[float], *, width: int = 720, height: int = 180) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = max(high - low, 1.0e-9)
    points = []
    for index, value in enumerate(values):
        x = 10 + (width - 20) * index / max(1, len(values) - 1)
        y = 10 + (height - 20) * (high - value) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _chart(title: str, values: Sequence[float], unit: str) -> str:
    return f"""
    <section class="chart">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 720 180" role="img" aria-label="{html.escape(title)}">
        <line x1="10" y1="170" x2="710" y2="170" class="axis"/>
        <polyline points="{_polyline(values)}" class="trace"/>
      </svg>
      <p>min {min(values):.5g} {unit} · max {max(values):.5g} {unit}</p>
    </section>
    """


def _clip_section(payload: Mapping[str, Any]) -> str:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("B2 dashboard input has no records")
    evaluations = [row["selected_evaluation"] for row in records]
    horizons = [float(row["selected_horizon"]) for row in records]
    margins = [float(row["worst_normalized_gate_margin"]) for row in evaluations]
    positions = [float(row["p95_object_position_error_m"]) for row in evaluations]
    rotations = [float(row["p95_object_rotation_error_deg"]) for row in evaluations]
    axes = [float(row["p95_axis_error_m"]) for row in evaluations]
    contacts = [float(row["mean_contact_stability"]) for row in evaluations]
    clip = str(payload["clip"])
    return f"""
    <article>
      <h2>{html.escape(clip)} · B2 30-step Oracle</h2>
      <div class="cards">
        <div><strong>{len(records)}</strong><span>control steps</span></div>
        <div><strong>{int(sum(value == 1 for value in horizons))}</strong>
          <span>H=1 selections</span></div>
        <div><strong>{int(sum(value == 5 for value in horizons))}</strong>
          <span>H=5 selections</span></div>
        <div><strong>{int(sum(value == 10 for value in horizons))}</strong>
          <span>H=10 selections</span></div>
      </div>
      {_chart("Selected horizon", horizons, "steps")}
      {_chart("Worst normalized gate margin", margins, "x gate")}
      {_chart("p95 object position", positions, "m")}
      {_chart("p95 object rotation", rotations, "deg")}
      {_chart("p95 object axis", axes, "m")}
      {_chart("Contact stability penalty", contacts, "score")}
    </article>
    """


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"R4 dashboard refuses overwrite: {args.output}")
    b2 = [_load(path) for path in args.b2]
    if tuple(sorted(str(row.get("clip")) for row in b2)) != (
        "hocap_170105",
        "hocap_170650",
    ):
        raise ValueError("dashboard requires exactly the two B2 clip reports")
    distribution = _load(args.distribution_gate)
    pool = _load(args.pool_benchmark)
    selected = pool.get("selected_layout")
    if not isinstance(selected, Mapping):
        raise ValueError("dashboard pool benchmark has no selected layout")
    body = "\n".join(_clip_section(row) for row in sorted(b2, key=lambda row: row["clip"]))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 16-C.5 R4 Robust Oracle Dashboard</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1040px; padding: 32px; background: #10141b; color: #ecf2fa; }}
h1 {{ margin-bottom: 6px; }} .status {{ color: #ffbf69; font-weight: 700; }}
.notice {{ border-left: 4px solid #ff6b6b; background: #29191d; padding: 16px;
  margin: 20px 0 32px; }}
article {{ margin: 38px 0; }}
.cards {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }}
.cards div {{ background: #18202c; padding: 14px; border-radius: 8px; }}
.cards strong,.cards span {{ display:block; }} .cards strong {{ font-size: 1.5rem; }}
.chart {{ background: #151c26; margin: 14px 0; padding: 14px; border-radius: 8px; }}
.chart h3,.chart p {{ margin: 0 0 8px; }} svg {{ width:100%; height:auto; background:#0d1118; }}
.axis {{ stroke:#64748b; stroke-width:1; }} .trace {{ fill:none; stroke:#65d6ad; stroke-width:3; }}
code {{ color:#8bd5ff; }}
</style>
</head>
<body>
<h1>Stage 16-C.5 R4 Robust Oracle</h1>
<p class="status">STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL · PPO NOT AUTHORIZED</p>
<div class="notice">
  <strong>Fail-closed evidence.</strong> Contact-phase distributional replication failed and both
  B2 rollouts reached formal failure probability 1.0. B3/C5C were therefore not started. This
  dashboard reports measured task/contact/horizon traces; it does not fabricate unavailable
  hand/object geometry or claim visual success.
</div>
<p>Distributional gate: <code>{html.escape(str(distribution.get("passes")))}</code>. Selected pool:
<code>{selected.get("population")} × 3 horizons × {selected.get("replicas")} replicas</code>,
{float(selected.get("gpu_memory_mib", 0.0)):.0f} MiB measured VRAM.</p>
{body}
<footer><p>Factor-8 runtime · abstract virtual wrist · no PPO · no checkpoint ·
no sim-to-real.</p></footer>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "clips": len(b2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
