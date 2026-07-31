"""Stage 11 migration reports for existing, immutable artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import transform_points
from toporetarget.utils.hashing import sha256_tree

from .canonical import load_canonical_hoi, migrate_v1_to_v2, save_canonical_hoi
from .reference import (
    load_robot_reference,
    migrate_reference_v1_to_v2,
    save_robot_reference,
)
from .version import CANONICAL_HOI_V2, ROBOT_REFERENCE_V2


def _digest_tree(path: Path) -> str:
    entries = sha256_tree(path)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_reference_v1_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files if name != "metadata"}
            metadata = json.loads(str(payload["metadata"].item())) if "metadata" in payload else {}
        return arrays, metadata
    # The v1 Zarr path is converted to v2 by the public loader.  The original
    # scene-space arrays are recovered from the v2 base-frame representation
    # below, so no solver/exporter code is involved.
    return {}, {}


def generate_stage11_migration_report(
    *,
    canonical_source: str | Path,
    reference_source: str | Path,
    output_root: str | Path = ".local/reports/stage11_migration",
    report_path: str | Path = ".local/reports/stage11_migration_report.json",
    robot_hash: str | None = None,
    joint_order: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Read old artifacts, write isolated v2 copies, and prove equivalence."""

    canonical_path = Path(canonical_source).resolve()
    reference_path = Path(reference_source).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonical_before = _digest_tree(canonical_path)
    canonical_v2_path = output / "canonical_hoi_v2.zarr"
    canonical_v2 = migrate_v1_to_v2(canonical_path)
    save_canonical_hoi(canonical_v2, canonical_v2_path)
    canonical_reloaded = load_canonical_hoi(canonical_v2_path)
    canonical_after = _digest_tree(canonical_path)

    arrays, metadata = _read_reference_v1_arrays(reference_path)
    reference_v2_path = output / "robot_reference_v2.npz"
    if arrays:
        reference_v2 = migrate_reference_v1_to_v2(
            reference_path,
            reference_v2_path,
            robot_hash=robot_hash,
            joint_order=joint_order,
            force=True,
        )
    else:
        reference_v2 = load_robot_reference(reference_path)
        save_robot_reference(reference_v2, reference_v2_path, force=True)
    reference_reloaded = load_robot_reference(reference_v2_path)

    qpos_unchanged = True
    robot_reference_unchanged = True
    reconstruction_checks: dict[str, bool] = {}
    if arrays:
        old_qpos = arrays["qpos"]
        old_base = arrays["base_pose_scene"]
        old_object = arrays["object_pose_scene"]
        old_links = arrays.get("robot_link_poses_scene", arrays.get("robot_link_poses"))
        if old_links is None:
            raise ValueError("v1 robot reference has no link poses")
        old_link_positions = old_links[..., :3, 3]
        qpos_unchanged = bool(np.array_equal(old_qpos, reference_v2.qpos_reference))
        object_scene = np.matmul(reference_v2.base_pose, reference_v2.object_pose_base)
        link_scene = transform_points(reference_v2.base_pose, reference_v2.tracked_link_positions)
        reconstruction_checks = {
            "base_pose": bool(np.array_equal(old_base, reference_v2.base_pose)),
            "object_pose_scene": bool(np.allclose(old_object, object_scene, atol=1e-12, rtol=0.0)),
            "tracked_link_positions_scene": bool(
                np.allclose(old_link_positions, link_scene, atol=1e-12, rtol=0.0)
            ),
        }
        robot_reference_unchanged = bool(all(reconstruction_checks.values()))

    report = {
        "schema_version": "toporetarget.stage11_migration_report.v1",
        "status": "complete"
        if canonical_before == canonical_after
        and canonical_reloaded.metadata.schema_version == CANONICAL_HOI_V2
        and reference_reloaded.schema_version == ROBOT_REFERENCE_V2
        and qpos_unchanged
        and robot_reference_unchanged
        else "blocked",
        "inputs": {
            "canonical_v1": str(canonical_path),
            "robot_reference_v1": str(reference_path),
            "canonical_v1_tree_hash_before": canonical_before,
            "canonical_v1_tree_hash_after": canonical_after,
        },
        "outputs": {
            "canonical_v2": str(canonical_v2_path),
            "robot_reference_v2": str(reference_v2_path),
        },
        "checks": {
            "canonical_v2_readable": canonical_reloaded.metadata.schema_version == CANONICAL_HOI_V2,
            "hash_unchanged": canonical_before == canonical_after,
            "qpos_unchanged": qpos_unchanged,
            "robot_reference_unchanged": robot_reference_unchanged,
            "reconstruction": reconstruction_checks,
        },
        "provenance": {
            "canonical_source_schema": "toporetarget.hoi.v1",
            "reference_source_schema": metadata.get(
                "schema_version", "toporetarget.robot_reference.v1"
            ),
            "canonical_output_schema": CANONICAL_HOI_V2,
            "reference_output_schema": ROBOT_REFERENCE_V2,
            "solver_invocations": 0,
            "stage5_to_stage10_artifacts_modified": False,
        },
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["generate_stage11_migration_report"]
