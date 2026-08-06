#!/usr/bin/env python3
"""Run the independent Stage 16-D collision-proxy and visual geometry audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from toporetarget.data.storage import direct_zarr3_arrays
from toporetarget.rl.physics_retargeting.geometry_audit import audit_trace_geometry

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-stage12", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"mesh did not resolve to Trimesh: {path}")
    return np.asarray(mesh.vertices, dtype=np.float64)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage16D refuses overwrite: {args.output}")
    manifest_path = REPO_ROOT / ".local/reports/stage16c1_asset_migration/wuji_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["collision_proxy_inventory"]
    with np.load(args.trace, allow_pickle=False) as trace:
        body_names = [str(value) for value in trace["hand_collision_body_names"]]
        object_poses = np.asarray(trace["replica_object_pose"])
        hand_poses = np.asarray(trace["replica_hand_collision_body_pose"])
    hand_vertices = {
        name: mesh_vertices(REPO_ROOT / inventory[name]["source"]) for name in body_names
    }
    object_path = (
        REPO_ROOT / f".local/stage16_reference_tracking_ppo/world_wrist_objects/{args.clip}.obj"
    )
    object_vertices = mesh_vertices(object_path)
    source_penetration = direct_zarr3_arrays(
        args.source_stage12, ("max_penetration",), array_prefix=""
    )["max_penetration"]
    result = audit_trace_geometry(
        object_vertices_local=object_vertices,
        visual_vertices_local=object_vertices,
        hand_vertices_by_body=hand_vertices,
        body_names=body_names,
        object_poses=object_poses,
        hand_body_poses=hand_poses,
        source_penetration_m=source_penetration,
    )
    result.update(
        {
            "clip": args.clip,
            "trace": str(args.trace.resolve()),
            "object_visual_mesh": str(object_path.resolve()),
            "hand_asset_manifest": str(manifest_path.resolve()),
            "source_stage12": {
                **result["source_stage12"],
                "path": str(args.source_stage12.resolve()),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["formal_geometry_gate"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
