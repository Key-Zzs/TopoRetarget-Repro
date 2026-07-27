"""Validation and machine-readable reports for robot-hand models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.keypoints.registry import get_layout
from toporetarget.utils.hashing import sha256_file

from .simulation import validate_urdf_mjcf
from .urdf.geometry import geometry_summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class RobotValidationReport:
    robot_name: str
    status: str
    checks: list[dict[str, Any]]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot_name,
            "status": self.status,
            "checks": _jsonable(self.checks),
            "metrics": _jsonable(self.metrics),
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
            writer.writeheader()
            for check in self.checks:
                writer.writerow(
                    {
                        "check": check["name"],
                        "passed": check["passed"],
                        "detail": json.dumps(_jsonable(check.get("detail", {})), sort_keys=True),
                    }
                )


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _random_q(model: Any, *, seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower, upper = model.joint_lower, model.joint_upper
    safe_lower = np.where(np.isfinite(lower), lower, -np.pi)
    safe_upper = np.where(np.isfinite(upper), upper, np.pi)
    return (
        safe_lower[None, :]
        + rng.uniform(0.1, 0.9, size=(count, model.num_dofs)) * (safe_upper - safe_lower)[None, :]
    )


def _check_generic_asset_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "asset_manifest.json"
    if not manifest_path.is_file():
        return {"status": "invalid", "message": "asset_manifest.json is missing"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tracked = manifest.get("tracked_files", [])
        missing: list[str] = []
        changed: list[str] = []
        for item in tracked:
            relative = str(item["path"])
            path = root / relative
            if not path.is_file():
                missing.append(relative)
            elif sha256_file(path) != str(item["sha256"]):
                changed.append(relative)
        valid = not missing and not changed and bool(tracked)
        return {
            "status": "ok" if valid else "invalid",
            "destination": str(root),
            "manifest_present": True,
            "tracked_file_count": len(tracked),
            "missing_files": missing,
            "changed_files": changed,
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "message": "Asset validation passed" if valid else "Asset validation failed",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "invalid", "message": f"invalid generic asset manifest: {exc}"}


def validate_robot_model(
    model: Any, *, seed: int = 4, dtype: str = "float64"
) -> RobotValidationReport:
    import torch

    checks: list[dict[str, Any]] = []
    actual = {
        "link_count": len(model.link_names),
        "joint_count": len(model.joint_names),
        "actuated_joint_count": len(model.urdf.actuated_joints),
        "fixed_joint_count": len(model.urdf.fixed_joints),
    }
    expected = {
        "link_count": model.spec.expected_link_count,
        "joint_count": model.spec.expected_total_joint_count,
        "actuated_joint_count": model.spec.expected_actuated_joint_count,
        "fixed_joint_count": model.spec.expected_fixed_joint_count,
    }
    _check(checks, "topology", actual == expected, {"actual": actual, "expected": expected})
    asset_integrity = None
    if model.asset_root is not None and (model.asset_root / "asset_manifest.json").is_file():
        if model.spec.asset_id == "artimano":
            from toporetarget.paths.assets import check_artimano_assets

            asset_check = check_artimano_assets(model.asset_root)
            asset_integrity = asset_check.as_dict()
            asset_ok = asset_check.status == "ok"
        else:
            asset_integrity = _check_generic_asset_manifest(model.asset_root)
            asset_ok = asset_integrity["status"] == "ok"
        _check(checks, "asset_manifest", asset_ok, asset_integrity)
    _check(
        checks,
        "base_link",
        model.base_link == model.urdf.root_link,
        {"base": model.base_link, "root": model.urdf.root_link},
    )
    _check(
        checks,
        "dof_order",
        tuple(model.dof_names) == tuple(model.spec.dof_order),
        {"dof_order": list(model.dof_names)},
    )
    lower, upper, neutral = model.joint_lower, model.joint_upper, model.neutral_q
    neutral_valid = bool(np.all(neutral >= lower) and np.all(neutral <= upper))
    _check(
        checks,
        "neutral_q_limits",
        neutral_valid,
        {"min_margin": float(np.min(np.minimum(neutral - lower, upper - neutral)))},
    )
    midpoint = np.where(np.isfinite(lower) & np.isfinite(upper), (lower + upper) / 2.0, neutral)
    q_samples = np.vstack([neutral, midpoint, _random_q(model, seed=seed, count=10)])
    _check(
        checks,
        "random_q_limits",
        bool(np.all(q_samples >= lower) and np.all(q_samples <= upper)),
        {"seed": seed, "count": 10, "includes": ["neutral", "midpoint", "random"]},
    )
    _check(
        checks,
        "anchor_profile",
        model.anchor_profile.layout_name == model.spec.semantic_keypoint_layout,
        {"profile": model.anchor_profile.profile_id, "hash": model.anchor_profile.sha256},
    )

    try:
        geometry = geometry_summary(model.urdf)
        geometry_ok = not geometry["unresolved_mesh_references"]
        _check(checks, "geometry_references", geometry_ok, geometry)
        for q in q_samples:
            model.visual_geometry_instances(q)
            model.collision_geometry_instances(q)
        _check(
            checks,
            "geometry_loading",
            True,
            {
                "visual": geometry["visual_geometry_count"],
                "collision": geometry["collision_geometry_count"],
            },
        )
    except (OSError, ValueError, RuntimeError) as exc:
        geometry = geometry_summary(model.urdf)
        _check(checks, "geometry_loading", False, {"error": str(exc), "summary": geometry})

    layout = get_layout(model.spec.semantic_keypoint_layout)
    q_torch = torch.tensor(neutral, dtype=getattr(torch, dtype))
    points = model.keypoints_base(q_torch).detach().cpu().numpy()
    expected_anchor_shape = (len(model.anchor_profile.anchors), 3)
    _check(
        checks,
        "anchor_shape",
        points.shape == expected_anchor_shape,
        {"shape": list(points.shape), "expected": list(expected_anchor_shape)},
    )
    _check(checks, "anchor_finite", bool(np.isfinite(points).all()), {})
    edges = np.asarray(layout.edges, dtype=np.int64)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=-1)
    _check(
        checks,
        "anchor_bone_lengths",
        bool(np.all(lengths > 1e-9)),
        {"zero_length_bone_count": int(np.sum(lengths <= 1e-9)), "minimum": float(np.min(lengths))},
    )

    fk_translation_max = 0.0
    fk_rotation_max = 0.0
    for q in q_samples:
        q_torch = torch.tensor(q, dtype=getattr(torch, dtype))
        torch_fk = model.forward_kinematics_base(q_torch)
        np_fk = model.forward_kinematics_reference(q)
        for name in model.link_names:
            torch_value = torch_fk[name].detach().cpu().numpy()
            reference = np_fk[name]
            fk_translation_max = max(
                fk_translation_max, float(np.max(np.abs(torch_value[:3, 3] - reference[:3, 3])))
            )
            relative = torch_value[:3, :3].T @ reference[:3, :3]
            sine = 0.5 * np.linalg.norm(
                np.array(
                    [
                        relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1],
                    ]
                )
            )
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.arctan2(sine, cosine)
            fk_rotation_max = max(fk_rotation_max, float(angle))
    _check(
        checks,
        "fk_cross_check",
        fk_translation_max <= 1e-10 and fk_rotation_max <= 1e-10,
        {
            "translation_max_error": fk_translation_max,
            "rotation_geodesic_max_error": fk_rotation_max,
            "tolerance": {"translation": 1e-10, "rotation": 1e-10},
        },
    )

    base = torch.eye(4, dtype=getattr(torch, dtype))
    base[:3, 3] = torch.tensor([0.1, -0.02, 0.03], dtype=getattr(torch, dtype))
    scene = model.forward_kinematics_scene(torch.tensor(neutral, dtype=getattr(torch, dtype)), base)
    base_fk = model.forward_kinematics_base(torch.tensor(neutral, dtype=getattr(torch, dtype)))
    equivariance = max(
        float(torch.max(torch.abs(scene[name] - base @ base_fk[name])).item())
        for name in model.link_names
    )
    _check(
        checks,
        "base_equivariance",
        equivariance <= 1e-8,
        {"max_point_error": equivariance, "tolerance": 1e-8},
    )
    simulation = validate_urdf_mjcf(model, seed=seed, random_count=10)
    _check(
        checks,
        "urdf_mjcf_consistency",
        simulation["status"] in {"pass", "not_applicable"},
        simulation,
    )
    metrics = {
        "topology": actual,
        "geometry": geometry,
        "fk": {
            "translation_max_error": fk_translation_max,
            "rotation_geodesic_max_error": fk_rotation_max,
        },
        "base_equivariance_max_error": equivariance,
        "asset_manifest_hash": model.asset_manifest_hash,
        "urdf_hash": model.urdf_hash,
        "spec_hash": model.spec_hash,
        "anchor_profile_hash": model.anchor_profile.sha256,
        "asset_integrity": asset_integrity,
        "urdf_mjcf": simulation,
        "seed": seed,
        "dtype": dtype,
    }
    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return RobotValidationReport(model.name, status, checks, metrics)


def jacobian_check(
    model: Any, qpos: Any, *, epsilon: float = 1e-6, dtype: str = "float64"
) -> dict[str, Any]:
    import torch

    torch_dtype = getattr(torch, dtype)
    q = torch.tensor(np.asarray(qpos, dtype=np.float64), dtype=torch_dtype)
    autograd = model.keypoint_jacobian_qpos(q).detach().cpu().numpy()
    # Build the finite-difference reference one DoF at a time to keep its ordering explicit.
    columns = []
    for index in range(model.num_dofs):
        delta = torch.zeros(model.num_dofs, dtype=torch_dtype)
        delta[index] = epsilon
        columns.append(
            ((model.keypoints_base(q + delta) - model.keypoints_base(q - delta)) / (2.0 * epsilon))
            .detach()
            .cpu()
            .numpy()
        )
    finite_difference = np.stack(columns, axis=-1)
    error = np.abs(autograd - finite_difference)
    absolute_max = float(np.max(error))
    rmse = float(np.sqrt(np.mean((autograd - finite_difference) ** 2)))
    relative = float(absolute_max / max(float(np.max(np.abs(finite_difference))), 1e-12))
    worst = np.unravel_index(np.argmax(error), error.shape)
    layout = get_layout(model.spec.semantic_keypoint_layout)
    result = {
        "robot": model.name,
        "autograd_shape": list(autograd.shape),
        "finite_difference_shape": list(finite_difference.shape),
        "epsilon": epsilon,
        "dtype": dtype,
        "maximum_absolute_error": absolute_max,
        "rmse": rmse,
        "relative_error": relative,
        "worst_keypoint": {"index": int(worst[0]), "semantic": layout.semantic_names[worst[0]]},
        "worst_dof": {"index": int(worst[2]), "name": model.dof_names[worst[2]]},
        "tolerance": {"absolute_max": 1e-5, "rmse": 1e-6, "relative": 1e-4},
        "passed": absolute_max <= 1e-5 and rmse <= 1e-6 and relative <= 1e-4,
        "per_dof_maximum": {
            name: float(error[..., index].max()) for index, name in enumerate(model.dof_names)
        },
        "per_keypoint_maximum": {
            name: float(error[index].max()) for index, name in enumerate(layout.semantic_names)
        },
    }
    return result


def write_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = ["RobotValidationReport", "jacobian_check", "validate_robot_model", "write_json"]
