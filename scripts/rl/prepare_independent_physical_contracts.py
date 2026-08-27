#!/usr/bin/env python3
"""Freeze outcome-independent physical gates, geometry, topology, and seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.import_hocap_objects import _bounded_convex_proxy  # noqa: E402
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    ConvexProxyGeometry,
    load_runtime_geometry_manifest,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_episode_manifest,
    atomic_write_json,
    stable_hash,
)
from toporetarget.rl.physics_retargeting.contact_topology import (  # noqa: E402
    extract_persistent_contact_topology,
)
from toporetarget.rl.physics_retargeting.contracts import derive_task_gate  # noqa: E402
from toporetarget.rl.physics_retargeting.task_semantics import (  # noqa: E402
    extract_task_semantics,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

FINGER_BODIES = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--world-reference", type=Path, required=True)
    parser.add_argument("--reference-v2", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--object-usd", type=Path, required=True)
    parser.add_argument("--strict-source-mask", type=Path, required=True)
    parser.add_argument("--base-runtime-geometry-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_PHYSICAL_CONTRACT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_PHYSICAL_CONTRACT_INPUT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                vertices.append([float(value) for value in values[1:4]])
            elif values[0] == "f":
                polygon = [int(value.split("/", 1)[0]) - 1 for value in values[1:]]
                faces.extend(
                    [polygon[0], polygon[index], polygon[index + 1]]
                    for index in range(1, len(polygon) - 1)
                )
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 4
        or triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or len(triangles) < 4
        or not np.isfinite(points).all()
        or int(triangles.min(initial=0)) < 0
        or int(triangles.max(initial=0)) >= len(points)
    ):
        raise ValueError("INDEPENDENT_PHYSICAL_CONTRACT_OBJECT_MESH_INVALID")
    return points, triangles


def _reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files if name != "metadata"}


def _contact_records(path: Path) -> tuple[list[dict[str, object]], int]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"strict_source_contact_mask", "finger_names", "control_index"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"INDEPENDENT_PHYSICAL_CONTACT_MASK_FIELDS_MISSING:{missing}")
        mask = np.asarray(archive["strict_source_contact_mask"], dtype=bool)
        names = tuple(str(value) for value in archive["finger_names"].tolist())
        control = np.asarray(archive["control_index"], dtype=np.int64)
    if (
        names != ("thumb", "index", "middle", "ring", "pinky")
        or mask.ndim != 2
        or mask.shape[1] != 5
        or control.shape != (mask.shape[0],)
        or not np.array_equal(control, np.arange(mask.shape[0]))
    ):
        raise ValueError("INDEPENDENT_PHYSICAL_CONTACT_MASK_INVALID")
    records = [
        {
            "control_step": int(step),
            "net_contact_force_world_on_object_n": [0.0, 0.0, 1.0],
            "present_hand_body_names": [
                FINGER_BODIES[index] for index in np.flatnonzero(mask[step])
            ],
        }
        for step in np.flatnonzero(mask.any(axis=1))
    ]
    if not records:
        raise ValueError("INDEPENDENT_PHYSICAL_SOURCE_CONTACT_EMPTY")
    return records, int(mask.shape[0])


def _seeds(clip: str) -> list[int]:
    seeds: list[int] = []
    for index in range(20):
        digest = hashlib.sha256(f"independent_physical_eval_v1:{clip}:{index}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed in seeds:
            raise RuntimeError("INDEPENDENT_PHYSICAL_EVALUATION_SEED_COLLISION")
        seeds.append(seed)
    return seeds


def main() -> int:
    args = _parser().parse_args()
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_PHYSICAL_CONTRACT_CLIP_ID_INVALID")
    manifest_path = args.selection_manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_episode_manifest(manifest)
    matches = [row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]
    if len(matches) != 1:
        raise ValueError("INDEPENDENT_PHYSICAL_CONTRACT_CLIP_NOT_FROZEN")

    world_path = args.world_reference.resolve()
    v2_path = args.reference_v2.resolve()
    mesh_path = args.object_mesh.resolve()
    usd_path = args.object_usd.resolve()
    mask_path = args.strict_source_mask.resolve()
    base_geometry_path = args.base_runtime_geometry_manifest.resolve()
    inputs = {
        "selection_manifest": _artifact(manifest_path),
        "world_reference": _artifact(world_path),
        "reference_v2": _artifact(v2_path),
        "object_mesh": _artifact(mesh_path),
        "object_usd": _artifact(usd_path),
        "strict_source_mask": _artifact(mask_path),
        "base_runtime_geometry_manifest": _artifact(base_geometry_path),
    }
    output = args.output_root.resolve()
    paths = {
        "task_semantics": output / "task_semantics.json",
        "contact_topology": output / "contact_topology.json",
        "evaluation_gates": output / "frozen_evaluation_gates.json",
        "runtime_geometry": output / "runtime_collision_geometry_manifest.json",
        "seed_manifest": output / "evaluation_seed_manifest.json",
        "receipt": output / "physical_contract_receipt.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"INDEPENDENT_PHYSICAL_CONTRACT_REFUSES_OVERWRITE:{existing}")

    reference = _reference(world_path)
    records, runtime_frames = _contact_records(mask_path)
    semantic = extract_task_semantics(
        clip=args.clip_id,
        reference=reference,
        contact_records=records,
        reference_time_scale=8,
    )
    if semantic.retimed_frame_count != runtime_frames:
        raise ValueError("INDEPENDENT_PHYSICAL_CONTRACT_RUNTIME_DOMAIN_MISMATCH")
    topology = extract_persistent_contact_topology(
        clip=args.clip_id,
        contact_records=records,
        retimed_frame_count=runtime_frames,
    )
    vertices, _faces = _obj(mesh_path)
    bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    gate = derive_task_gate(semantic, object_bbox_diagonal_m=bbox_diagonal)

    hand_proxies, _development_objects = load_runtime_geometry_manifest(base_geometry_path)
    proxy_vertices, proxy_faces, support_gap = _bounded_convex_proxy(vertices.tolist())
    object_proxy = ConvexProxyGeometry(
        shape_id=f"{args.clip_id}:convex_hull_v1",
        body_name=args.clip_id,
        geometry_type="convex_hull",
        vertices=np.asarray(proxy_vertices, dtype=np.float64),
        faces=np.asarray(proxy_faces, dtype=np.int64),
        local_pose_xyz_wxyz=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        scale_xyz=np.ones(3),
        source_asset_path=str(mesh_path),
        source_asset_sha256=inputs["object_mesh"]["sha256"],
        generated_asset_path=str(usd_path),
        generated_asset_sha256=inputs["object_usd"]["sha256"],
    )

    seeds = _seeds(args.clip_id)
    atomic_write_json(paths["task_semantics"], semantic.as_dict())
    atomic_write_json(
        paths["contact_topology"],
        {
            "schema_version": "PersistentContactTopologyV1Set",
            "clips": {args.clip_id: topology.as_dict()},
        },
    )
    atomic_write_json(
        paths["evaluation_gates"],
        {
            "schema_version": "IndependentPhysicalEvaluationGateFreezeV1",
            "status": "STRICT_V4_EVALUATION_GATES_FROZEN",
            "source": "raw_source_contact_and_reference_only_before_policy_evaluation",
            "task_gates": {"clips": {args.clip_id: gate.as_dict()}},
            "contact_topology": {"clips": {args.clip_id: topology.as_dict()}},
            "inputs": inputs,
        },
    )
    atomic_write_json(
        paths["runtime_geometry"],
        {
            "schema_version": "RuntimeCollisionGeometryManifestV1",
            "hand_shapes": [proxy.as_dict() for proxy in hand_proxies],
            "object_shapes": {args.clip_id: [object_proxy.as_dict()]},
            "validation": {
                "runtime_hand_shape_count": len(hand_proxies),
                "runtime_object_shape_count": {args.clip_id: 1},
            },
            "independent_object_proxy": {
                "method": "bounded_convex_proxy_v1_shared_with_isaac_import",
                "max_support_gap_m": support_gap,
            },
            "inputs": inputs,
        },
    )
    atomic_write_json(
        paths["seed_manifest"],
        {
            "schema_version": "IndependentPhysicalEvaluationSeedManifestV1",
            "clip_id": args.clip_id,
            "seed_derivation": "sha256(independent_physical_eval_v1:clip:index)_uint31",
            "eval10": seeds[:10],
            "confirm20": seeds,
            "outcomes_observed": False,
            "selection_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    artifacts = {name: _artifact(path) for name, path in paths.items() if name != "receipt"}
    receipt = {
        "schema_version": "IndependentPhysicalContractsReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "runtime_frames": runtime_frames,
        "source_contact_steps": len(records),
        "outcomes_observed": False,
        "per_clip_threshold_tuning": False,
        "gate_derivation": "shared_PhysicsConsistentTaskGateV1",
        "contract_hash": stable_hash(
            {
                "semantic": semantic.as_dict(),
                "topology": topology.as_dict(),
                "gate": gate.as_dict(),
                "seeds": seeds,
            }
        ),
        "inputs": inputs,
        "artifacts": artifacts,
    }
    atomic_write_json(paths["receipt"], receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
