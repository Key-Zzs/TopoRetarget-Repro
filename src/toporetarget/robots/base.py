"""Generic robot-hand model interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.keypoints.registry import get_layout
from toporetarget.paths.assets import AssetResolution
from toporetarget.utils.hashing import sha256_file

from .anchors import AnchorProfile, RobotKeypointSet, load_anchor_profile
from .profiles import load_qpos_order_profile
from .spec import RobotHandSpec
from .urdf.geometry import (
    RobotGeometryInstance,
    collision_geometry_instances,
    visual_geometry_instances,
)
from .urdf.kinematics import (
    forward_kinematics_numpy,
    forward_kinematics_torch,
    joint_origins_torch,
)
from .urdf.model import UrdfModel


class RobotHandModel:
    """A differentiable finger-DoF robot hand loaded from a tracked spec and URDF."""

    def __init__(
        self,
        spec: RobotHandSpec,
        urdf_model: UrdfModel,
        *,
        asset_root: Path | None = None,
        asset_manifest: dict[str, Any] | None = None,
        config_root: Path | None = None,
        asset_resolution: AssetResolution | None = None,
    ) -> None:
        self.spec = spec
        self.urdf = urdf_model
        self.asset_root = None if asset_root is None else asset_root.resolve()
        self._asset_manifest = asset_manifest
        self.asset_resolution = asset_resolution
        self.config_root = config_root
        self._anchor_profile: AnchorProfile | None = None
        self._qpos_order_profile: dict[str, Any] | None = None
        if spec.base_link != urdf_model.root_link:
            configured = spec.base_link
            actual = urdf_model.root_link
            raise ValueError(f"{spec.name}: base link mismatch: {configured!r} != {actual!r}")
        actual_dofs = tuple(joint.name for joint in urdf_model.actuated_joints)
        if set(actual_dofs) != set(spec.dof_order):
            missing = sorted(set(actual_dofs) - set(spec.dof_order))
            extra = sorted(set(spec.dof_order) - set(actual_dofs))
            raise ValueError(f"{spec.name}: DoF config mismatch; missing={missing}, extra={extra}")
        if len(spec.neutral_q) != len(spec.dof_order):
            raise ValueError(f"{spec.name}: neutral_q length must equal configured DoF count")
        self._dof_index = {name: index for index, name in enumerate(spec.dof_order)}
        self._q_index_by_joint = {name: self._dof_index[name] for name in actual_dofs}
        if spec.qpos_order_profile is not None:
            self._qpos_order_profile = load_qpos_order_profile(
                spec.qpos_order_profile,
                config_root=self.config_root
                or Path(__file__).resolve().parents[3] / "configs" / "robots",
                expected_dof_order=spec.dof_order,
            )

    def _validate_anchor_references(self, profile: AnchorProfile | None = None) -> None:
        joint_names = set(self.urdf.joint_names)
        link_names = set(self.urdf.link_names)
        selected = self._anchor_profile if profile is None else profile
        if selected is None:
            return
        for anchor in selected.anchors:
            if anchor.link_name is not None and anchor.link_name not in link_names:
                raise ValueError(
                    f"{self.name}: anchor {anchor.semantic_name} references unknown "
                    f"link {anchor.link_name}"
                )
            if anchor.joint_name is not None and anchor.joint_name not in joint_names:
                raise ValueError(
                    f"{self.name}: anchor {anchor.semantic_name} references unknown "
                    f"joint {anchor.joint_name}"
                )

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def side(self) -> str:
        return self.spec.side

    @property
    def base_link(self) -> str:
        return self.spec.base_link

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self.spec.dof_order

    @property
    def num_dofs(self) -> int:
        return len(self.dof_names)

    @property
    def joint_lower(self) -> np.ndarray:
        return np.asarray(
            [self.urdf.joint_by_name[name].limit.lower for name in self.dof_names], dtype=np.float64
        )

    @property
    def joint_upper(self) -> np.ndarray:
        return np.asarray(
            [self.urdf.joint_by_name[name].limit.upper for name in self.dof_names], dtype=np.float64
        )

    @property
    def neutral_q(self) -> np.ndarray:
        return np.asarray(self.spec.neutral_q, dtype=np.float64)

    @property
    def link_names(self) -> tuple[str, ...]:
        return self.urdf.link_names

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.urdf.joint_names

    @property
    def spec_hash(self) -> str:
        return self.spec.sha256

    @property
    def urdf_hash(self) -> str:
        return sha256_file(self.urdf.urdf_path)

    @property
    def asset_manifest(self) -> dict[str, Any] | None:
        return self._asset_manifest

    @property
    def asset_manifest_hash(self) -> str | None:
        if self.asset_root is None:
            return None
        manifest_path = self.asset_root / "asset_manifest.json"
        return sha256_file(manifest_path) if manifest_path.is_file() else None

    @property
    def asset_source(self) -> str | None:
        return None if self.asset_resolution is None else self.asset_resolution.source

    @property
    def asset_warnings(self) -> tuple[str, ...]:
        return () if self.asset_resolution is None else self.asset_resolution.warnings

    @property
    def anchor_profile(self) -> AnchorProfile:
        if self._anchor_profile is None:
            self._anchor_profile = load_anchor_profile(
                self.spec.keypoint_anchor_profile, config_root=self.config_root
            )
            self._validate_anchor_references(self._anchor_profile)
        return self._anchor_profile

    @property
    def qpos_order_profile(self) -> dict[str, Any] | None:
        return self._qpos_order_profile

    def _qpos_tensor(
        self, qpos: Any, *, dtype: Any | None = None, device: Any | None = None
    ) -> Any:
        import torch

        if isinstance(qpos, torch.Tensor):
            result = (
                qpos
                if dtype is None and device is None
                else qpos.to(dtype=dtype or qpos.dtype, device=device or qpos.device)
            )
        else:
            result = torch.as_tensor(qpos, dtype=dtype, device=device)
        if not result.is_floating_point():
            result = result.to(dtype=dtype or torch.get_default_dtype())
        if result.ndim < 1 or result.shape[-1] != self.num_dofs:
            raise ValueError(
                f"qpos must have shape [...,{self.num_dofs}], got {tuple(result.shape)}"
            )
        return result

    def forward_kinematics_base(self, qpos: Any) -> dict[str, Any]:
        return forward_kinematics_torch(self.urdf, self._reorder_to_urdf(qpos))

    def forward_kinematics_reference(self, qpos: Any) -> dict[str, np.ndarray]:
        return forward_kinematics_numpy(self.urdf, self._reorder_to_urdf_numpy(qpos))

    def _reorder_to_urdf(self, qpos: Any) -> Any:
        value = self._qpos_tensor(qpos)
        indices = [self._dof_index[joint.name] for joint in self.urdf.actuated_joints]
        return value[..., indices]

    def _reorder_to_urdf_numpy(self, qpos: Any) -> np.ndarray:
        value = np.asarray(qpos, dtype=np.float64)
        if value.ndim < 1 or value.shape[-1] != self.num_dofs:
            raise ValueError(f"qpos must have shape [...,{self.num_dofs}], got {value.shape}")
        indices = [self._dof_index[joint.name] for joint in self.urdf.actuated_joints]
        return value[..., indices]

    def forward_kinematics_scene(self, qpos: Any, base_pose_scene: Any) -> dict[str, Any]:
        import torch

        q = self._qpos_tensor(qpos)
        base = torch.as_tensor(base_pose_scene, dtype=q.dtype, device=q.device)
        if base.shape[-2:] != (4, 4):
            raise ValueError(f"base_pose_scene must end in [4,4], got {tuple(base.shape)}")
        return {
            name: base @ transform for name, transform in self.forward_kinematics_base(q).items()
        }

    def link_transform_base(self, qpos: Any, link_name: str) -> Any:
        if link_name not in self.link_names:
            raise KeyError(link_name)
        return self.forward_kinematics_base(qpos)[link_name]

    def link_transform_scene(self, qpos: Any, base_pose_scene: Any, link_name: str) -> Any:
        return self.forward_kinematics_scene(qpos, base_pose_scene)[link_name]

    def keypoints_base(self, qpos: Any, layout: str | None = None) -> Any:
        layout = self.spec.semantic_keypoint_layout if layout is None else layout
        if layout != self.spec.semantic_keypoint_layout:
            raise ValueError(
                f"{self.name} only declares semantic layout {self.spec.semantic_keypoint_layout!r}"
            )
        definition = get_layout(layout)
        if definition.point_count != len(self.anchor_profile.anchors):
            raise ValueError("anchor profile and layout point counts differ")
        transforms = self.forward_kinematics_base(qpos)
        joint_transforms = joint_origins_torch(self.urdf, self._reorder_to_urdf(qpos))
        points = []
        for anchor in self.anchor_profile.anchors:
            if anchor.anchor_type == "link_origin":
                assert anchor.link_name is not None
                points.append(transforms[anchor.link_name][..., :3, 3])
            elif anchor.anchor_type == "joint_origin":
                assert anchor.joint_name is not None
                points.append(joint_transforms[anchor.joint_name][..., :3, 3])
            else:
                assert anchor.link_name is not None
                assert anchor.local_xyz is not None
                local = self._qpos_tensor(qpos)
                local = local.new_tensor(anchor.local_xyz)
                link_transform = transforms[anchor.link_name]
                points.append(link_transform[..., :3, :3] @ local + link_transform[..., :3, 3])
        return torch_stack(points, dim=-2)

    def keypoints_scene(self, qpos: Any, base_pose_scene: Any, layout: str | None = None) -> Any:
        import torch

        q = self._qpos_tensor(qpos)
        base = torch.as_tensor(base_pose_scene, dtype=q.dtype, device=q.device)
        points = self.keypoints_base(q, layout=layout)
        return points @ base[..., :3, :3].transpose(-1, -2) + base[..., None, :3, 3]

    def keypoint_metadata(self, layout: str | None = None) -> dict[str, Any]:
        layout = self.spec.semantic_keypoint_layout if layout is None else layout
        if layout != self.spec.semantic_keypoint_layout:
            raise ValueError(f"unsupported semantic layout for {self.name}: {layout}")
        return {
            "layout_name": layout,
            "coordinate_source": "robot_urdf_anchors",
            "robot_name": self.name,
            "robot_side": self.side,
            "anchor_profile_id": self.anchor_profile.profile_id,
            "anchor_profile_version": self.anchor_profile.version,
            "anchor_profile_hash": self.anchor_profile.sha256,
            "urdf_hash": self.urdf_hash,
            "asset_manifest_hash": self.asset_manifest_hash,
            "resolved_asset_source": self.asset_source,
            "resolved_asset_root": None if self.asset_root is None else str(self.asset_root),
            "legacy_fallback_used": bool(
                self.asset_resolution is not None and self.asset_resolution.legacy_fallback_used
            ),
            "asset_warnings": list(self.asset_warnings),
        }

    def keypoint_set_base(self, qpos: Any, layout: str | None = None) -> RobotKeypointSet:
        return RobotKeypointSet(
            self.keypoints_base(qpos, layout=layout), self.keypoint_metadata(layout)
        )

    def keypoint_set_scene(
        self, qpos: Any, base_pose_scene: Any, layout: str | None = None
    ) -> RobotKeypointSet:
        return RobotKeypointSet(
            self.keypoints_scene(qpos, base_pose_scene, layout=layout),
            self.keypoint_metadata(layout),
        )

    def keypoint_jacobian_qpos(self, qpos: Any, layout: str | None = None) -> Any:
        import torch

        value = self._qpos_tensor(qpos)
        if value.ndim == 1:
            return torch.autograd.functional.jacobian(
                lambda item: self.keypoints_base(item, layout=layout), value, create_graph=True
            )
        flat = value.reshape(-1, self.num_dofs)
        jacobians = [
            torch.autograd.functional.jacobian(
                lambda item: self.keypoints_base(item, layout=layout), item, create_graph=True
            )
            for item in flat
        ]
        anchor_count = len(self.anchor_profile.anchors)
        return torch.stack(jacobians).reshape(*value.shape[:-1], anchor_count, 3, self.num_dofs)

    def visual_geometry_instances(
        self, qpos: Any, base_pose_scene: Any | None = None
    ) -> list[RobotGeometryInstance]:
        return visual_geometry_instances(
            self.urdf,
            self._reorder_to_urdf_numpy(qpos),
            None if base_pose_scene is None else np.asarray(base_pose_scene, dtype=np.float64),
        )

    def collision_geometry_instances(
        self, qpos: Any, base_pose_scene: Any | None = None
    ) -> list[RobotGeometryInstance]:
        return collision_geometry_instances(
            self.urdf,
            self._reorder_to_urdf_numpy(qpos),
            None if base_pose_scene is None else np.asarray(base_pose_scene, dtype=np.float64),
        )

    def qpos_from_named_dict(
        self, values: dict[str, float], *, dtype: Any | None = None, device: Any | None = None
    ) -> Any:
        unknown = sorted(set(values) - set(self.dof_names))
        missing = sorted(set(self.dof_names) - set(values))
        if unknown or missing:
            raise ValueError(f"named qpos mismatch; missing={missing}, unknown={unknown}")
        import torch

        return torch.tensor(
            [float(values[name]) for name in self.dof_names], dtype=dtype, device=device
        )

    def qpos_to_named_dict(self, qpos: Any) -> dict[str, Any]:
        value = self._qpos_tensor(qpos)
        if value.ndim != 1:
            raise ValueError("qpos_to_named_dict requires one qpos")
        return {name: value[index].item() for index, name in enumerate(self.dof_names)}

    def validate(self, **kwargs: Any) -> Any:
        from .reports import validate_robot_model

        return validate_robot_model(self, **kwargs)

    def describe(self) -> dict[str, Any]:
        from .urdf.geometry import geometry_summary

        return {
            "name": self.name,
            "side": self.side,
            "asset_id": self.spec.asset_id,
            "asset_root": None if self.asset_root is None else str(self.asset_root),
            "resolved_asset_source": self.asset_source,
            "legacy_fallback_used": bool(
                self.asset_resolution is not None and self.asset_resolution.legacy_fallback_used
            ),
            "asset_warnings": list(self.asset_warnings),
            "asset_manifest_hash": self.asset_manifest_hash,
            "urdf": self.spec.urdf_relative_path,
            "urdf_path": str(self.urdf.urdf_path),
            "urdf_hash": self.urdf_hash,
            "spec_hash": self.spec_hash,
            "base_link": self.base_link,
            "link_count": len(self.link_names),
            "joint_count": len(self.joint_names),
            "actuated_joint_count": len(self.urdf.actuated_joints),
            "fixed_joint_count": len(self.urdf.fixed_joints),
            "dof_order": list(self.dof_names),
            "joint_limits": {
                name: [self.joint_by_name(name).limit.lower, self.joint_by_name(name).limit.upper]
                for name in self.dof_names
            },
            "neutral_q": self.neutral_q.tolist(),
            "anchor_profile": {
                "id": self.anchor_profile.profile_id,
                "version": self.anchor_profile.version,
                "hash": self.anchor_profile.sha256,
            },
            "contract": {
                "asset_bundle": self.spec.asset_bundle.as_dict(),
                "kinematics": self.spec.kinematics.as_dict(),
                "semantic_anchors": self.spec.semantic_anchors.as_dict(),
                "surface": self.spec.surface_profile.as_dict(),
                "collision": self.spec.collision_profile.as_dict(),
                "simulation": self.spec.simulation.as_dict(),
                "profile_paths": {
                    "qpos_order": self.spec.qpos_order_profile,
                    "surface": self.spec.surface_profile_path,
                    "urdf_collision": self.spec.urdf_collision_profile,
                    "mjcf_collision": self.spec.mjcf_collision_profile,
                },
                "qpos_order_profile": self.qpos_order_profile,
            },
            "geometry": geometry_summary(self.urdf),
            "assumptions": list(self.spec.assumptions),
            "notes": self.spec.notes,
        }

    def joint_by_name(self, name: str):
        return self.urdf.joint_by_name[name]


def torch_stack(values: list[Any], *, dim: int) -> Any:
    import torch

    return torch.stack(values, dim=dim)


__all__ = ["RobotHandModel"]
