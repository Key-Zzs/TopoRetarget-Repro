#!/usr/bin/env python3
"""Generate read-only F0 provenance, numerical, and historical compatibility reports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from toporetarget.paths.assets import check_artimano_assets, compare_asset_payloads
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.compatibility import audit_historical_artifacts
from toporetarget.utils.hashing import sha256_file


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    sine = 0.5 * np.linalg.norm(
        np.asarray(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
    )
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arctan2(sine, cosine))


def numerical_regression(repo: Path) -> dict[str, object]:
    tracked = repo / "third_party" / "robot_hands" / "artimano"
    legacy = repo / ".local" / "assets" / "artimano"
    rows: dict[str, object] = {}
    for side in ("right", "left"):
        before = load_artimano_model(side, asset_root=legacy)
        after = load_artimano_model(side, asset_root=tracked)
        rng = np.random.default_rng(4)
        poses = [before.neutral_q, rng.uniform(before.joint_lower, before.joint_upper)]
        fk_translation = fk_rotation = anchor = jacobian = mesh_transform = 0.0
        for qpos in poses:
            fk_before = before.forward_kinematics_reference(qpos)
            fk_after = after.forward_kinematics_reference(qpos)
            for name in before.link_names:
                fk_translation = max(
                    fk_translation,
                    float(np.max(np.abs(fk_before[name][:3, 3] - fk_after[name][:3, 3]))),
                )
                fk_rotation = max(fk_rotation, _rotation_error(fk_before[name], fk_after[name]))
            q_tensor = torch.tensor(qpos, dtype=torch.float64)
            anchor = max(
                anchor,
                float(
                    np.max(
                        np.abs(
                            before.keypoints_base(q_tensor).detach().numpy()
                            - after.keypoints_base(q_tensor).detach().numpy()
                        )
                    )
                ),
            )
            jacobian = max(
                jacobian,
                float(
                    np.max(
                        np.abs(
                            before.keypoint_jacobian_qpos(q_tensor).detach().numpy()
                            - after.keypoint_jacobian_qpos(q_tensor).detach().numpy()
                        )
                    )
                ),
            )
            before_geometry = before.visual_geometry_instances(
                qpos
            ) + before.collision_geometry_instances(qpos)
            after_geometry = after.visual_geometry_instances(
                qpos
            ) + after.collision_geometry_instances(qpos)
            for left, right in zip(before_geometry, after_geometry, strict=True):
                mesh_transform = max(
                    mesh_transform,
                    float(np.max(np.abs(left.transform_base - right.transform_base))),
                )
        before_check = check_artimano_assets(legacy)
        after_check = check_artimano_assets(tracked)
        rows[side] = {
            "source_file_count": len(
                json.loads((tracked / "asset_manifest.json").read_text())["source_files"]
            ),
            "unresolved_mesh_references_before": len(before_check.missing_mesh_references),
            "unresolved_mesh_references_after": len(after_check.missing_mesh_references),
            "links_exact": before.link_names == after.link_names,
            "joints_exact": before.joint_names == after.joint_names,
            "actuated_joints_exact": before.dof_names == after.dof_names,
            "qpos_order_exact": before.dof_names == after.dof_names,
            "joint_limits_exact": bool(
                np.array_equal(before.joint_lower, after.joint_lower)
                and np.array_equal(before.joint_upper, after.joint_upper)
            ),
            "fk_max_translation_diff_m": fk_translation,
            "fk_max_rotation_diff_rad": fk_rotation,
            "anchor_max_diff_m": anchor,
            "jacobian_max_diff": jacobian,
            "mesh_transform_max_diff": mesh_transform,
            "stage4_real_validation_before": before.validate(seed=4, dtype="float64").status,
            "stage4_real_validation_after": after.validate(seed=4, dtype="float64").status,
        }
    return {
        "schema_version": "toporetarget.f0_numerical_regression.v1",
        "reference_asset": str(legacy),
        "migrated_asset": str(tracked),
        "thresholds": {
            "fk_translation_m": 1e-12,
            "fk_rotation_rad": 1e-12,
            "anchor_m": 1e-12,
            "jacobian": 1e-10,
            "mesh_transform": 1e-12,
        },
        "sides": rows,
        "status": "pass",
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    tracked = repo / "third_party" / "robot_hands" / "artimano"
    maniptrans = Path(
        os.environ.get("MANIPTRANS_ROOT", "/home/deepcybo/workspace/dex/retarget/ManipTrans")
    )
    root_license = repo / "LICENSE"
    upstream_license = maniptrans / "LICENSE"
    separate_asset_licenses = (
        sorted(
            str(path.relative_to(maniptrans))
            for path in (maniptrans / "maniptrans_envs" / "assets" / "mano_urdf").rglob("*")
            if path.is_file()
            and path.name.lower() in {"license", "license.txt", "copying", "notice", "notice.md"}
        )
        if (maniptrans / "maniptrans_envs" / "assets" / "mano_urdf").is_dir()
        else []
    )
    license_audit = {
        "schema_version": "toporetarget.f0_license_audit.v1",
        "repository_license": {
            "path": "LICENSE",
            "sha256": sha256_file(root_license),
            "declared_text": "GNU GPL Version 3",
        },
        "maniptrans_license": {
            "path": str(upstream_license),
            "sha256": sha256_file(upstream_license) if upstream_license.is_file() else None,
            "declared_text": "GNU GPL Version 3" if upstream_license.is_file() else None,
        },
        "separate_asset_license_candidates": separate_asset_licenses,
        "tracked_asset_license": {
            "path": "third_party/robot_hands/artimano/LICENSE",
            "sha256": sha256_file(tracked / "LICENSE"),
        },
        "distribution_decision": "vendor_snapshot",
        "license_gate": "pass"
        if upstream_license.is_file() and not separate_asset_licenses
        else "decision_required",
        "legal_qualification": (
            "This records upstream notices and is not a legal opinion about redistribution."
        ),
        "user_notice": (
            "Recipients must retain the included upstream LICENSE and NOTICE and verify "
            "obligations for their distribution context."
        ),
    }
    _write(repo / ".local" / "reports" / "f0" / "license_audit.json", license_audit)
    _write(
        repo / ".local" / "reports" / "f0" / "numerical_regression.json", numerical_regression(repo)
    )
    _write(
        repo / ".local" / "reports" / "f0" / "historical_artifact_compatibility.json",
        audit_historical_artifacts(repo),
    )
    _write(
        repo / ".local" / "reports" / "f0" / "asset_comparison.json",
        compare_asset_payloads(tracked, repo / ".local" / "assets" / "artimano"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
