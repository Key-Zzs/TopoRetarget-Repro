#!/usr/bin/env python3
"""Create the Stage 16-D numerical review dashboard with truthful layer availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        choices=("source", "stage12", "optimized", "ppo"),
        required=True,
    )
    parser.add_argument("--mode", choices=("headless", "interactive"), default="headless")
    parser.add_argument("--clip-root", type=Path, action="append", required=True)
    for flag in (
        "show-reference-hand",
        "show-corrected-hand",
        "show-source-object",
        "show-corrected-object",
        "show-contacts",
        "show-contact-topology",
        "show-penetration",
        "show-task-progress",
    ):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-frames", type=Path)
    parser.add_argument("--output-dashboard", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    return parser.parse_args()


def load_clip(root: Path) -> dict[str, object]:
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    quality = json.loads((root / "quality.json").read_text())
    object_motion = np.linalg.norm(
        arrays["object_pose"][:, :3] - arrays["object_pose"][0, :3], axis=1
    )
    source_object_deviation = np.linalg.norm(
        arrays["object_pose"][:, :3] - arrays["source_object_pose"][:, :3], axis=1
    )
    source_wrist_deviation = np.linalg.norm(
        arrays["wrist_pose"][:, :3] - arrays["source_wrist_pose"][:, :3], axis=1
    )
    return {
        "clip": root.name,
        "status": quality["status"],
        "semantic_progress": arrays["semantic_progress"].astype(float).tolist(),
        "contact_recall": arrays["contact_recall"].astype(float).tolist(),
        "penetration_lower_bound_mm": (
            arrays["penetration_lower_bound"].astype(float) * 1000.0
        ).tolist(),
        "object_motion_m": object_motion.tolist(),
        "source_object_deviation_m": source_object_deviation.tolist(),
        "source_wrist_deviation_m": source_wrist_deviation.tolist(),
        "action_norm": np.linalg.norm(arrays["actions"], axis=1).tolist(),
        "termination_reason_code": arrays["termination_reason_code"].astype(int).tolist(),
    }


def dashboard(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return (  # noqa: E501 - compact self-contained dashboard, not Python logic
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Stage 16-D numerical dashboard</title><style>body{{font:14px system-ui;background:#111827;color:#e5e7eb;margin:24px}}canvas{{background:#fff;border-radius:8px;width:100%;height:220px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}code{{color:#93c5fd}}</style></head><body><h1>Stage 16-D Physics-Consistent Retargeting</h1><p>Numerical fallback dashboard. Collision proxy penetration is a lower bound; no PPO trajectory exists.</p><div id="summary"></div><div class="grid" id="plots"></div><script>const D={data};const metrics=['semantic_progress','contact_recall','penetration_lower_bound_mm','object_motion_m','source_object_deviation_m','source_wrist_deviation_m','action_norm'];document.getElementById('summary').innerHTML=D.clips.map(c=>`<p><code>${{c.clip}}</code> · ${{c.status}}</p>`).join('');for(const m of metrics){{const box=document.createElement('div');box.innerHTML=`<h3>${{m}}</h3><canvas width="700" height="220"></canvas>`;document.getElementById('plots').appendChild(box);const c=box.querySelector('canvas'),x=c.getContext('2d');const all=D.clips.flatMap(r=>r[m]);const lo=Math.min(...all),hi=Math.max(...all),span=Math.max(hi-lo,1e-12);D.clips.forEach((r,k)=>{{x.strokeStyle=k?'#ef4444':'#2563eb';x.beginPath();r[m].forEach((v,i)=>{{const px=i/(r[m].length-1)*c.width,py=c.height-(v-lo)/span*c.height;i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke()}});x.fillStyle='#111827';x.fillText(`${{lo.toPrecision(3)}} .. ${{hi.toPrecision(3)}}`,8,16)}}</script></body></html>"""  # noqa: E501
    )


def main() -> int:
    args = parse_args()
    if len(args.clip_root) != 2:
        raise ValueError("Stage16D dashboard requires both trajectory roots")
    clips = [load_clip(root) for root in args.clip_root]
    payload = {
        "schema_version": "Stage16DVisualReviewV1",
        "status": "STAGE16D_NUMERICAL_DASHBOARD_GENERATED",
        "mode": args.mode,
        "requested_trajectories": args.trajectory,
        "clips": clips,
        "rendering": "NOT_RUN_HEADLESS_NUMERICAL_FALLBACK",
        "ppo_success_view": "UNAVAILABLE_PPO_NOT_RUN",
        "ppo_failure_view": "UNAVAILABLE_PPO_NOT_RUN",
        "source_optimized_contact_sheets": "represented by dashboard contact timelines",
        "formal_visual_acceptance": False,
        "layers": {
            name.removeprefix("show_"): bool(value)
            for name, value in vars(args).items()
            if name.startswith("show_")
        },
    }
    args.output_dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.output_dashboard.write_text(dashboard(payload), encoding="utf-8")
    payload["dashboard"] = str(args.output_dashboard.resolve())
    if args.output_video is not None:
        payload["output_video"] = "NOT_WRITTEN_HEADLESS_NUMERICAL_FALLBACK"
    if args.output_frames is not None:
        args.output_frames.mkdir(parents=True, exist_ok=True)
        note = args.output_frames / "README.json"
        note.write_text(
            json.dumps({"status": "NUMERICAL_DASHBOARD_USED_NO_RASTER_FRAMES"}, indent=2) + "\n"
        )
        payload["output_frames"] = str(note.resolve())
    args.output_review.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "dashboard": str(args.output_dashboard)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
