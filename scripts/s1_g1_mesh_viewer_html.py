#!/usr/bin/env python
"""Generate a reference-style self-contained G1 hand-mesh viewer.

The page is artifact-only: it loads canonical MANO plus completed E0 and S1
trajectories and never invokes refinement.  Metric data is intentionally
omitted from the visible page.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.robots.registry import get_robot_registry
from toporetarget.workflows.mesh_visualization import (
    HTML_SCHEMA_VERSION,
    _bounds,
    _html_document,
    _robot_payload,
    _source_frame_indices,
    _source_mesh,
)
from toporetarget.quality.html import _mesh_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    selection = experiment / "selection/G1"
    canonical = selection / "canonical.zarr"
    e0_path = experiment / "e0/G1/final.zarr"
    final_s1_path = experiment / "artifacts/G1/S1_L01/final.zarr"
    prescreen_s1_path = experiment / "artifacts/G1/S1_L01_prescreen/final.zarr"
    s1_path = final_s1_path if final_s1_path.exists() else prescreen_s1_path
    repo = Path(__file__).resolve().parents[1]
    asset_root = repo / "third_party/robot_hands/artimano"
    required = (canonical, e0_path, s1_path, asset_root)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing viewer input: " + ", ".join(missing))

    sequence = load_hoi_sequence(canonical)
    e0 = load_final_trajectory(e0_path)
    s1 = load_final_trajectory(s1_path)
    if e0.frame_count != s1.frame_count:
        raise SystemExit("E0 and S1 frame counts differ")
    manifest = {
        "robot": "artimano_rh",
        "asset_root": str(asset_root),
        "selected_frame_range": [0, s1.frame_count],
        "source_sequence": "G1 airplane lift",
        "s1_artifact": "final" if s1_path == final_s1_path else "prescreen",
    }
    source_indices = _source_frame_indices(sequence, s1, manifest)
    source_vertices, source_faces = _source_mesh(sequence, s1, manifest)
    model = get_robot_registry().load("artimano_rh", asset_root=asset_root)
    e0_payload = _robot_payload(
        model,
        e0.arrays["qpos"],
        e0.arrays["base_pose_scene"],
    )
    s1_payload = _robot_payload(
        model,
        s1.arrays["qpos"],
        s1.arrays["base_pose_scene"],
    )
    object_track = sequence.rigid_object(str(sequence.rigid_objects[0].object_id))
    object_vertices, object_faces = _mesh_subset(
        np.asarray(object_track.mesh.vertices_local, dtype=np.float64),
        np.asarray(object_track.mesh.faces, dtype=np.int64),
        max_faces=6000,
    )
    object_payload = {
        "vertices": object_vertices.round(6).tolist(),
        "faces": object_faces.tolist(),
        "poses": np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
        .round(8)
        .tolist(),
        "object_id": str(object_track.object_id),
    }
    payload = {
        "schema_version": HTML_SCHEMA_VERSION,
        "title": "G1 airplane lift · source/E0/S1 hand mesh viewer",
        "source_sequence": "G1 airplane lift",
        "robot": "artimano_rh",
        "frame_count": int(s1.frame_count),
        "source": {"vertices": source_vertices.round(6).tolist(), "faces": source_faces.tolist()},
        "warm": e0_payload,
        "final": s1_payload,
        "object": object_payload,
        "metrics": {"frames": [{} for _ in range(s1.frame_count)]},
        "bounds": _bounds(
            source_vertices,
            np.asarray(object_payload["vertices"], dtype=float),
            np.asarray(object_payload["poses"], dtype=float),
            (e0_payload, s1_payload),
        ),
    }
    document = _html_document(payload)
    document = document.replace(
        '<h2>Frame metrics</h2>\n    <pre id="metrics"></pre>',
        '<h2>Mesh-only review</h2>\n'
        '    <div class="hint">Only source MANO, E0, and S1 hand meshes are shown.</div>\n'
        '    <pre id="metrics" style="display:none"></pre>',
    )
    document = document.replace("Warm-start robot mesh", "E0 robot mesh")
    document = document.replace("Final robot mesh", "S1 robot mesh")
    document = document.replace(
        '    <label><input id="object" type="checkbox" checked> '
        '<span class="legend" style="background:#64748b"></span>Object points</label>\n',
        '    <label><input id="object" type="checkbox" checked> '
        '<span class="legend" style="background:#7c3aed"></span>Object mesh</label>\n',
    )
    document = document.replace(
        "source and robot meshes are in scene coordinates.",
        "source, E0, S1, and object meshes are in scene coordinates.",
    )
    document = re.sub(
        r"function drawObject\(index, width, height\) \{.*?\n\}\nfunction draw\(\)",
        """function drawObject(index, width, height) {
  const pose = DATA.object.poses[index], points = DATA.object.vertices;
  const world = points.map(p => [pose[0][0]*p[0]+pose[0][1]*p[1]+pose[0][2]*p[2]+pose[0][3], pose[1][0]*p[0]+pose[1][1]*p[1]+pose[1][2]*p[2]+pose[1][3], pose[2][0]*p[0]+pose[2][1]*p[1]+pose[2][2]*p[2]+pose[2][3]]);
  if (DATA.object.faces?.length) { drawMesh(world, DATA.object.faces, '#7c3aed', 0.28, width, height); return; }
  ctx.fillStyle = '#7c3aed'; ctx.globalAlpha = 0.45;
  for (const q of world) { const s = project(q,width,height); ctx.fillRect(s[0]-1,s[1]-1,2,2); }
  ctx.globalAlpha = 1;
}
function draw()""",
        document,
        count=1,
        flags=re.DOTALL,
    )
    document = document.replace(
        "frameLabel.textContent = `local ${item.local_frame} · source ${item.source_frame}`;",
        "frameLabel.textContent = 'current frame';",
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
