#!/usr/bin/env python3
"""Build data-derived stable-grasp topology and object-canonical contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.calibration_initialization import (  # noqa: E402
    object_canonical_frame,
)
from toporetarget.rl.geometry_audit.stable_grasp_calibration import (  # noqa: E402
    extract_grasp_topology_families,
)
from toporetarget.rl.physics_retargeting.contact_topology import (  # noqa: E402
    body_contact_group,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
TOPOLOGY_PATH = (
    REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting/contact_topology.json"
)
MANIFEST_PATH = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    topology = _load(TOPOLOGY_PATH)
    manifest = _load(MANIFEST_PATH)
    available_groups = sorted(
        {
            group
            for row in manifest["hand_shapes"]
            if (group := body_contact_group(str(row["body_name"]))) is not None
        }
    )
    families = extract_grasp_topology_families(topology, available_groups=available_groups)
    family_payload = {
        "schema_version": "GraspTopologyFamilyContractV1",
        "status": "STAGE16D_GRASP_TOPOLOGY_FAMILIES_FROZEN",
        "derivation_inputs": [
            str(TOPOLOGY_PATH.relative_to(REPO_ROOT)),
            str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        ],
        "available_collision_body_groups": available_groups,
        "same_algorithm_for_all_clips": True,
        "clip_specific_code": False,
        "families": [family.as_dict() for family in families],
    }
    _write(output / "grasp_topology_family_contract.json", family_payload)
    for clip in sorted(topology["clips"]):
        _write(
            output / f"topology_applicability_{clip.removeprefix('hocap_')}.json",
            {
                "schema_version": "GraspTopologyApplicabilityV1",
                "clip": clip,
                "source_required_groups": topology["clips"][clip]["required_body_groups"],
                "applicable_families": [
                    family.as_dict() for family in families if clip in family.applicable_clips
                ],
            },
        )
    object_frames: dict[str, Any] = {}
    for object_id, shapes in sorted(manifest["object_shapes"].items()):
        if len(shapes) != 1:
            raise RuntimeError(f"STAGE16D_OBJECT_PROXY_PIECE_COUNT_UNSUPPORTED:{object_id}")
        row = shapes[0]
        vertices = np.asarray(row["convex_vertices_m"], dtype=np.float64) * np.asarray(
            row["scale_xyz"], dtype=np.float64
        )
        object_frames[object_id] = {
            "shape_id": row["shape_id"],
            "geometry_sha256": row["geometry_sha256"],
            "canonical_frame": object_canonical_frame(vertices).as_dict(),
        }
    _write(
        output / "calibration_initialization_contract.json",
        {
            "schema_version": "ObjectCanonicalGraspInitializerV1",
            "status": "STAGE16D_CALIBRATION_INITIALIZATION_FROZEN",
            "inputs": "runtime object convex proxy plus live reset-time calibration hand poses",
            "object_frames": object_frames,
            "initialization_state_writes": "reset only",
            "rollout_object_state_writes": 0,
            "rollout_wrist_state_writes": 0,
            "corrected_trajectory_used": False,
            "source_object_pose_used": False,
            "hidden_support": False,
        },
    )
    print(json.dumps({"status": family_payload["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
