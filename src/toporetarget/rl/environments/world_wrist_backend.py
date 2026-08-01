"""Finite-wrench world-wrist MuJoCo backend for the Stage-16B extension.

The wrist is an abstract actuated free body with a finite Cartesian impedance
controller.  It is intentionally *not* presented as a real arm model.  The
free object remains a separate MuJoCo free joint and this module never writes
its pose or velocity after reset, except in explicitly named kinematic-object
diagnostics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import rotation_geodesic_error

from ..axis_points import object_axis_points_from_poses
from ..termination import BASE_RELATIVE_HOCAP_TERMINATION, TerminationInput, classify_termination
from ..world_wrist import (
    WorldWristFingerReferenceV1,
    matrix_from_quaternion_wxyz,
    quaternion_wxyz_from_matrix,
    se3_exp_local,
    so3_log,
)
from .mujoco_backend import _require_mujoco

WORLD_WRIST_BACKEND_ID = "world_wrist_finger_backend_v1"
WRIST_IMPEDANCE_PROFILE_ID = "wrist_impedance_profile_v1"
WRIST_FINGER_ACTION_SCALE_ID = "wrist_finger_action_scale_v1"


def _clip_norm(values: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value if norm <= limit else value * (limit / max(norm, 1e-12))


def _pose_from_qpos(qpos: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = matrix_from_quaternion_wxyz(qpos[3:7])
    result[:3, 3] = qpos[:3]
    return result


def _qpos_from_pose(pose: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(pose, dtype=np.float64)[:3, 3], quaternion_wxyz_from_matrix(pose[:3, :3])]
    )


def materialize_world_wrist_free_object_scene(
    hand_mjcf: str | Path,
    output_directory: str | Path,
    *,
    object_mesh: str | Path,
    object_mass_kg: float = 0.05,
) -> Path:
    """Create an ignored zero-gravity scene with dynamic wrist and object free joints."""

    source = Path(hand_mjcf).resolve()
    mesh_path = Path(object_mesh).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"object mesh does not exist: {mesh_path}")
    if object_mass_kg <= 0.0:
        raise ValueError("object_mass_kg must be positive")
    destination_root = Path(output_directory)
    destination_root.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.01")
    option.set("gravity", "0 0 0")
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        compiler.set("meshdir", str((source.parent / compiler.get("meshdir", "")).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Wuji MJCF has no worldbody")
    wrist = next(
        (body for body in worldbody.findall("body") if body.get("name") == "r_wrist"), None
    )
    if wrist is None:
        raise ValueError("Wuji MJCF has no direct r_wrist root body")
    if wrist.find("freejoint") is not None:
        raise ValueError("source Wuji MJCF already contains a root freejoint")
    wrist.insert(
        1 if wrist.find("inertial") is not None else 0,
        ET.Element("freejoint", name="stage16_wrist_free"),
    )
    collision_index = 0
    for body in root.findall(".//body"):
        body_name = body.get("name", "unnamed")
        for geom in body.findall("geom"):
            contype = int(geom.get("contype", "1"))
            conaffinity = int(geom.get("conaffinity", "1"))
            if geom.get("name") is None and (contype != 0 or conaffinity != 0):
                geom.set("name", f"stage16b_hand_collision_{body_name}_{collision_index:02d}")
                collision_index += 1
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="stage16b_object_mesh", file=str(mesh_path))
    object_body = ET.SubElement(worldbody, "body", name="stage16b_object")
    ET.SubElement(object_body, "freejoint", name="stage16b_object_free")
    ET.SubElement(
        object_body,
        "geom",
        name="stage16b_object_geom",
        type="mesh",
        mesh="stage16b_object_mesh",
        mass=f"{object_mass_kg:.17g}",
        friction="1 0.005 0.0001",
        rgba="0.8 0.3 0.2 1",
    )
    destination = destination_root / "wuji_hand2_stage16b_world_wrist_free_object.xml"
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


@dataclass(frozen=True)
class WristImpedanceProfileV1:
    """One globally shared finite Cartesian controller profile."""

    translation_stiffness_npm: float = 250.0
    translation_damping_ratio: float = 1.0
    rotation_stiffness_nmprad: float = 15.0
    rotation_damping_ratio: float = 1.0
    force_limit_n: float = 25.0
    torque_limit_nm: float = 1.5
    feedforward_twist_gain: float = 1.0

    def validate(self) -> None:
        if (
            self.translation_stiffness_npm <= 0.0
            or self.rotation_stiffness_nmprad <= 0.0
            or self.translation_damping_ratio <= 0.0
            or self.rotation_damping_ratio <= 0.0
            or self.force_limit_n <= 0.0
            or self.torque_limit_nm <= 0.0
        ):
            raise ValueError("impedance profile requires finite, positive gains and wrench limits")

    def as_dict(self) -> dict[str, Any]:
        return {"id": WRIST_IMPEDANCE_PROFILE_ID, **asdict(self)}


@dataclass(frozen=True)
class WristFingerActionScaleV1:
    """Globally frozen normalized residual action scales for all clips."""

    translation_m: float = 0.01
    rotation_rad: float = float(np.deg2rad(5.0))
    finger_joint_range_fraction: float = 0.10

    def validate(self) -> None:
        if (
            self.translation_m <= 0.0
            or self.rotation_rad <= 0.0
            or not 0.0 < self.finger_joint_range_fraction <= 0.2
        ):
            raise ValueError("action scales must be positive predefined global values")

    def as_dict(self) -> dict[str, Any]:
        return {"id": WRIST_FINGER_ACTION_SCALE_ID, **asdict(self)}


@dataclass(frozen=True)
class WristControlResult:
    target_pose_world: np.ndarray
    applied_wrench_world: np.ndarray
    position_error_world_m: np.ndarray
    rotation_error_local_rad: np.ndarray
    linear_velocity_error_world_mps: np.ndarray
    angular_velocity_error_local_radps: np.ndarray
    force_saturated: bool
    torque_saturated: bool


class CartesianWristImpedanceController:
    """Deterministic finite-wrench impedance controller for the abstract wrist."""

    def __init__(
        self,
        profile: WristImpedanceProfileV1,
        *,
        wrist_mass_kg: float,
        wrist_inertia_kgm2: np.ndarray,
    ) -> None:
        profile.validate()
        if wrist_mass_kg <= 0.0 or np.any(np.asarray(wrist_inertia_kgm2) <= 0.0):
            raise ValueError("wrist mass and inertia must be positive")
        self.profile = profile
        self.translation_damping_nspm = (
            2.0
            * np.sqrt(profile.translation_stiffness_npm * wrist_mass_kg)
            * profile.translation_damping_ratio
        )
        self.rotation_damping_nmsprad = (
            2.0
            * np.sqrt(profile.rotation_stiffness_nmprad * float(np.mean(wrist_inertia_kgm2)))
            * profile.rotation_damping_ratio
        )

    def target_pose(
        self,
        reference_pose_world: np.ndarray,
        normalized_residual: np.ndarray,
        scale: WristFingerActionScaleV1,
    ) -> np.ndarray:
        residual = np.asarray(normalized_residual, dtype=np.float64)
        if residual.shape != (6,) or np.any(np.abs(residual) > 1.0 + 1e-9):
            raise ValueError("wrist residual must be a bounded 6-D normalized action")
        return np.asarray(reference_pose_world, dtype=np.float64) @ se3_exp_local(
            scale.translation_m * residual[:3], scale.rotation_rad * residual[3:]
        )

    def compute(
        self,
        *,
        target_pose_world: np.ndarray,
        target_twist_world: np.ndarray,
        current_pose_world: np.ndarray,
        current_twist_world: np.ndarray,
    ) -> WristControlResult:
        target_twist = np.asarray(target_twist_world, dtype=np.float64)
        current_twist = np.asarray(current_twist_world, dtype=np.float64)
        if target_twist.shape != (6,) or current_twist.shape != (6,):
            raise ValueError("wrist twists must have shape [6]")
        current_rotation = np.asarray(current_pose_world, dtype=np.float64)[:3, :3]
        position_error = (
            np.asarray(target_pose_world)[:3, 3] - np.asarray(current_pose_world)[:3, 3]
        )
        rotation_error_local = so3_log(current_rotation.T @ np.asarray(target_pose_world)[:3, :3])
        linear_velocity_error = target_twist[:3] - current_twist[:3]
        angular_velocity_error_local = current_rotation.T @ (target_twist[3:] - current_twist[3:])
        raw_force = (
            self.profile.translation_stiffness_npm * position_error
            + self.translation_damping_nspm
            * self.profile.feedforward_twist_gain
            * linear_velocity_error
        )
        raw_torque_local = (
            self.profile.rotation_stiffness_nmprad * rotation_error_local
            + self.rotation_damping_nmsprad
            * self.profile.feedforward_twist_gain
            * angular_velocity_error_local
        )
        force = _clip_norm(raw_force, self.profile.force_limit_n)
        torque_local = _clip_norm(raw_torque_local, self.profile.torque_limit_nm)
        return WristControlResult(
            target_pose_world=np.asarray(target_pose_world, dtype=np.float64).copy(),
            applied_wrench_world=np.concatenate([force, current_rotation @ torque_local]),
            position_error_world_m=position_error,
            rotation_error_local_rad=rotation_error_local,
            linear_velocity_error_world_mps=linear_velocity_error,
            angular_velocity_error_local_radps=angular_velocity_error_local,
            force_saturated=bool(np.linalg.norm(raw_force) > self.profile.force_limit_n),
            torque_saturated=bool(np.linalg.norm(raw_torque_local) > self.profile.torque_limit_nm),
        )


@dataclass
class WorldWristBackendSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    xfrc_applied: np.ndarray
    time: float
    reference_index: int
    step_index: int
    previous_action: np.ndarray
    second_previous_action: np.ndarray


class WorldWristFingerBackend:
    """26-D Stage-16B physical backend with no live object pose writes."""

    def __init__(
        self,
        *,
        scene_path: str | Path,
        reference: WorldWristFingerReferenceV1,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        impedance_profile: WristImpedanceProfileV1 = WristImpedanceProfileV1(),
        action_scale: WristFingerActionScaleV1 = WristFingerActionScaleV1(),
        control_dt_s: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.mujoco = _require_mujoco()
        self.reference = reference
        self.reference.validate(expected_hz=20.0, joint_lower=joint_lower, joint_upper=joint_upper)
        impedance_profile.validate()
        action_scale.validate()
        self.model = self.mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = self.mujoco.MjData(self.model)
        self.joint_lower = np.asarray(joint_lower, dtype=np.float64)
        self.joint_upper = np.asarray(joint_upper, dtype=np.float64)
        self.impedance_profile = impedance_profile
        self.action_scale = action_scale
        self.control_dt_s = float(control_dt_s)
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        self.decimation = int(round(self.control_dt_s / self.model.opt.timestep))
        if self.decimation < 1 or not np.isclose(
            self.decimation * self.model.opt.timestep, self.control_dt_s
        ):
            raise ValueError("control period must be an integer MuJoCo timestep multiple")
        self.rng = np.random.default_rng(seed)
        self.reference_index = 0
        self.step_index = 0
        self.previous_action = np.zeros(26, dtype=np.float64)
        self.second_previous_action = np.zeros(26, dtype=np.float64)
        self.last_control: WristControlResult | None = None
        self.last_physics_trace: list[dict[str, Any]] = []

        def ids(kind: Any, names: tuple[str, ...]) -> np.ndarray:
            result = np.asarray(
                [self.mujoco.mj_name2id(self.model, kind, name) for name in names], dtype=np.int64
            )
            if np.any(result < 0):
                raise ValueError(f"required MuJoCo names are missing: {list(names)}")
            return result

        finger_joint_ids = ids(self.mujoco.mjtObj.mjOBJ_JOINT, reference.joint_order)
        self.finger_qpos_addresses = self.model.jnt_qposadr[finger_joint_ids]
        self.finger_dof_addresses = self.model.jnt_dofadr[finger_joint_ids]
        self.finger_actuator_indices = np.asarray(
            [
                np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)[0]
                for joint in finger_joint_ids
            ],
            dtype=np.int64,
        )
        wrist_joint = int(ids(self.mujoco.mjtObj.mjOBJ_JOINT, ("stage16_wrist_free",))[0])
        object_joint = int(ids(self.mujoco.mjtObj.mjOBJ_JOINT, ("stage16b_object_free",))[0])
        self.wrist_qpos_address = int(self.model.jnt_qposadr[wrist_joint])
        self.wrist_dof_address = int(self.model.jnt_dofadr[wrist_joint])
        self.object_qpos_address = int(self.model.jnt_qposadr[object_joint])
        self.object_dof_address = int(self.model.jnt_dofadr[object_joint])
        self.wrist_body_id = int(ids(self.mujoco.mjtObj.mjOBJ_BODY, ("r_wrist",))[0])
        self.object_body_id = int(ids(self.mujoco.mjtObj.mjOBJ_BODY, ("stage16b_object",))[0])
        self.link_body_ids = ids(self.mujoco.mjtObj.mjOBJ_BODY, reference.tracked_link_names)
        self.mujoco.mj_forward(self.model, self.data)
        submass = float(self.model.body_subtreemass[self.wrist_body_id])
        # The freejoint drives the entire hand subtree.  Its effective angular
        # inertia is therefore the MuJoCo generalized mass matrix block, not
        # the wrist body's local inertial tensor alone.
        full_mass = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        self.mujoco.mj_fullM(self.model, full_mass, self.data.qM)
        inertia = np.diag(
            full_mass[
                self.wrist_dof_address + 3 : self.wrist_dof_address + 6,
                self.wrist_dof_address + 3 : self.wrist_dof_address + 6,
            ]
        )
        self.wrist_effective_inertia_kgm2 = inertia.copy()
        self.controller = CartesianWristImpedanceController(
            impedance_profile, wrist_mass_kg=submass, wrist_inertia_kgm2=inertia
        )

    @property
    def action_dim(self) -> int:
        return 26

    def _wrist_pose(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = self.data.xmat[self.wrist_body_id].reshape(3, 3)
        result[:3, 3] = self.data.xpos[self.wrist_body_id]
        return result

    def _object_pose(self) -> np.ndarray:
        return _pose_from_qpos(
            self.data.qpos[self.object_qpos_address : self.object_qpos_address + 7]
        )

    def _state(self) -> dict[str, np.ndarray]:
        object_pose = self._object_pose()
        return {
            "wrist_pose": self._wrist_pose(),
            "wrist_twist": self.data.qvel[
                self.wrist_dof_address : self.wrist_dof_address + 6
            ].copy(),
            "q": self.data.qpos[self.finger_qpos_addresses].copy(),
            "qdot": self.data.qvel[self.finger_dof_addresses].copy(),
            "object_pose": object_pose,
            "object_twist": self.data.qvel[
                self.object_dof_address : self.object_dof_address + 6
            ].copy(),
            "object_axis_points": object_axis_points_from_poses(object_pose[None])[0],
            "links": self.data.xpos[self.link_body_ids].copy(),
        }

    def reset(self, *, reference_index: int | None = None) -> dict[str, np.ndarray]:
        self.reference_index = (
            int(self.rng.integers(0, self.reference.frame_count))
            if reference_index is None
            else int(reference_index)
        )
        if not 0 <= self.reference_index < self.reference.frame_count:
            raise ValueError("reference_index is outside the clip")
        self.step_index = 0
        self.previous_action.fill(0.0)
        self.second_previous_action.fill(0.0)
        self.last_physics_trace.clear()
        self.mujoco.mj_resetData(self.model, self.data)
        index = self.reference_index
        self.data.qpos[self.wrist_qpos_address : self.wrist_qpos_address + 7] = _qpos_from_pose(
            self.reference.wrist_pose_world_ref[index]
        )
        self.data.qvel[self.wrist_dof_address : self.wrist_dof_address + 6] = (
            self.reference.wrist_twist_world_ref[index]
        )
        self.data.qpos[self.finger_qpos_addresses] = self.reference.q_finger_ref[index]
        self.data.qvel[self.finger_dof_addresses] = self.reference.qdot_finger_ref[index]
        self.data.qpos[self.object_qpos_address : self.object_qpos_address + 7] = _qpos_from_pose(
            self.reference.object_pose_world_ref[index]
        )
        self.data.qvel[self.object_dof_address : self.object_dof_address + 6] = (
            self.reference.object_twist_world_ref[index]
        )
        self.data.xfrc_applied.fill(0.0)
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def snapshot(self) -> WorldWristBackendSnapshot:
        return WorldWristBackendSnapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            xfrc_applied=self.data.xfrc_applied.copy(),
            time=float(self.data.time),
            reference_index=self.reference_index,
            step_index=self.step_index,
            previous_action=self.previous_action.copy(),
            second_previous_action=self.second_previous_action.copy(),
        )

    def restore(self, snapshot: WorldWristBackendSnapshot) -> None:
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.ctrl[:] = snapshot.ctrl
        self.data.xfrc_applied[:] = snapshot.xfrc_applied
        self.data.time = snapshot.time
        self.reference_index = snapshot.reference_index
        self.step_index = snapshot.step_index
        self.previous_action[:] = snapshot.previous_action
        self.second_previous_action[:] = snapshot.second_previous_action
        self.mujoco.mj_forward(self.model, self.data)

    def _set_kinematic_object_for_diagnostic(self, index: int) -> None:
        """Diagnostic-only object playback, isolated from formal world dynamics."""

        self.data.qpos[self.object_qpos_address : self.object_qpos_address + 7] = _qpos_from_pose(
            self.reference.object_pose_world_ref[index]
        )
        self.data.qvel[self.object_dof_address : self.object_dof_address + 6] = (
            self.reference.object_twist_world_ref[index]
        )

    def exogenous_wrist_playback_step(self) -> dict[str, np.ndarray]:
        """Play wrist reference pose only for W1 diagnosis, never PPO/formal rollout.

        This method is intentionally separate from :meth:`step` and does not
        expose an action argument.  It exists solely to demonstrate that
        restoring world wrist motion changes approach reachability; W1 is not
        a dynamic wrist-control result.
        """

        index = self.reference_index
        self.data.qpos[self.wrist_qpos_address : self.wrist_qpos_address + 7] = _qpos_from_pose(
            self.reference.wrist_pose_world_ref[index]
        )
        self.data.qvel[self.wrist_dof_address : self.wrist_dof_address + 6] = (
            self.reference.wrist_twist_world_ref[index]
        )
        self.data.ctrl[self.finger_actuator_indices] = self.reference.q_finger_ref[index]
        self.mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        self.reference_index = min(index + 1, self.reference.frame_count - 1)
        return self._state()

    def step(
        self, action: np.ndarray, *, kinematic_object_diagnostic: bool = False
    ) -> dict[str, np.ndarray]:
        normalized_action = np.asarray(action, dtype=np.float64)
        if normalized_action.shape != (26,) or not np.isfinite(normalized_action).all():
            raise ValueError("Stage-16B action must be finite with shape [26]")
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        index = self.reference_index
        target_index = min(index + 1, self.reference.frame_count - 1)
        target_pose = self.controller.target_pose(
            self.reference.wrist_pose_world_ref[target_index],
            normalized_action[:6],
            self.action_scale,
        )
        finger_target = np.clip(
            self.reference.q_finger_ref[target_index]
            + normalized_action[6:]
            * (self.joint_upper - self.joint_lower)
            * self.action_scale.finger_joint_range_fraction,
            self.joint_lower,
            self.joint_upper,
        )
        self.data.ctrl[self.finger_actuator_indices] = finger_target
        self.last_physics_trace = []
        for substep in range(self.decimation):
            current_pose = self._wrist_pose()
            current_twist = self.data.qvel[
                self.wrist_dof_address : self.wrist_dof_address + 6
            ].copy()
            control = self.controller.compute(
                target_pose_world=target_pose,
                target_twist_world=self.reference.wrist_twist_world_ref[target_index],
                current_pose_world=current_pose,
                current_twist_world=current_twist,
            )
            self.data.xfrc_applied.fill(0.0)
            self.data.xfrc_applied[self.wrist_body_id] = control.applied_wrench_world
            self.mujoco.mj_step(self.model, self.data)
            self.last_control = control
            self.last_physics_trace.append(
                {
                    "substep": substep,
                    "reference_index": index,
                    "wrist_wrench_world": control.applied_wrench_world.tolist(),
                    "force_saturated": control.force_saturated,
                    "torque_saturated": control.torque_saturated,
                    "object_qpos": self.data.qpos[
                        self.object_qpos_address : self.object_qpos_address + 7
                    ].tolist(),
                    "object_qvel": self.data.qvel[
                        self.object_dof_address : self.object_dof_address + 6
                    ].tolist(),
                    "contact_count": int(self.data.ncon),
                }
            )
        self.step_index += 1
        self.reference_index = target_index
        if kinematic_object_diagnostic:
            self._set_kinematic_object_for_diagnostic(self.reference_index)
            self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def predict_step(
        self, action: np.ndarray, *, kinematic_object_diagnostic: bool = False
    ) -> dict[str, np.ndarray]:
        snapshot = self.snapshot()
        try:
            return self.step(action, kinematic_object_diagnostic=kinematic_object_diagnostic)
        finally:
            self.restore(snapshot)

    def transition(
        self, action: np.ndarray, *, kinematic_object_diagnostic: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, float], str | None]:
        state = self.step(action, kinematic_object_diagnostic=kinematic_object_diagnostic)
        index = self.reference_index
        reward = world_wrist_reward(
            state=state,
            reference=self.reference,
            reference_index=index,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            action=action,
            previous_action=self.previous_action,
            second_previous_action=self.second_previous_action,
        )
        self.second_previous_action = self.previous_action.copy()
        self.previous_action = np.asarray(action, dtype=np.float64).copy()
        object_error = float(
            np.linalg.norm(
                state["object_pose"][:3, 3] - self.reference.object_pose_world_ref[index, :3, 3]
            )
        )
        orientation_error = float(
            rotation_geodesic_error(
                state["object_pose"], self.reference.object_pose_world_ref[index]
            )
        )
        axis_error = float(
            np.max(
                np.linalg.norm(
                    state["object_axis_points"]
                    - self.reference.object_axis_points_world_ref[index],
                    axis=1,
                )
            )
        )
        wrist_position_error = float(
            np.linalg.norm(
                state["wrist_pose"][:3, 3] - self.reference.wrist_pose_world_ref[index, :3, 3]
            )
        )
        wrist_rotation_error = float(
            rotation_geodesic_error(state["wrist_pose"], self.reference.wrist_pose_world_ref[index])
        )
        base_reason = classify_termination(
            TerminationInput(
                step=self.step_index,
                reference_index=index,
                reference_frame_count=self.reference.frame_count,
                object_height_m=float(state["object_pose"][2, 3]),
                object_linear_velocity_mps=float(np.linalg.norm(state["object_twist"][:3])),
                object_angular_velocity_radps=float(np.linalg.norm(state["object_twist"][3:])),
                object_position_error_m=object_error,
                object_orientation_error_rad=orientation_error,
                max_axis_point_error_m=axis_error,
            ),
            profile=BASE_RELATIVE_HOCAP_TERMINATION,
        )
        safety_reason = (
            "FAILURE_WRIST_POSITION_SAFETY"
            if wrist_position_error > 0.20
            else "FAILURE_WRIST_ORIENTATION_SAFETY"
            if wrist_rotation_error > np.deg2rad(90.0)
            else None
        )
        return state, reward, safety_reason or (None if base_reason is None else base_reason.value)

    def observation(self, state: dict[str, np.ndarray] | None = None) -> np.ndarray:
        return build_world_wrist_observation(
            state=self._state() if state is None else state,
            reference=self.reference,
            reference_index=self.reference_index,
            previous_action=self.previous_action,
        )

    def contact_summary(self) -> dict[str, Any]:
        object_geom = int(
            self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "stage16b_object_geom"
            )
        )
        contacts: list[dict[str, Any]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if object_geom not in {int(contact.geom1), int(contact.geom2)}:
                continue
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, self.data, index, force)
            contacts.append(
                {
                    "geom1": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    "geom2": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                    "position": np.asarray(contact.pos).tolist(),
                    "normal": np.asarray(contact.frame).reshape(3, 3)[:, 0].tolist(),
                    "distance_m": float(contact.dist),
                    "force_local": force.tolist(),
                }
            )
        return {"hand_object_contact_count": len(contacts), "contacts": contacts}

    def model_report(self) -> dict[str, Any]:
        return {
            "backend": WORLD_WRIST_BACKEND_ID,
            "wrist_body": "r_wrist",
            "wrist_freejoint": "stage16_wrist_free",
            "wrist_qpos_address": self.wrist_qpos_address,
            "wrist_dof_address": self.wrist_dof_address,
            "wrist_subtree_mass_kg": float(self.model.body_subtreemass[self.wrist_body_id]),
            "wrist_body_inertia_kgm2": self.model.body_inertia[self.wrist_body_id].tolist(),
            "wrist_effective_freejoint_inertia_kgm2": self.wrist_effective_inertia_kgm2.tolist(),
            "object_body": "stage16b_object",
            "object_freejoint": "stage16b_object_free",
            "object_qpos_address": self.object_qpos_address,
            "object_dof_address": self.object_dof_address,
            "gravity_mps2": self.model.opt.gravity.tolist(),
            "synthetic_ground": False,
            "formal_rollout_object_pose_write": False,
            "abstract_actuated_wrist_engineering_model": True,
            "impedance": self.impedance_profile.as_dict()
            | {
                "translation_damping_nspm": self.controller.translation_damping_nspm,
                "rotation_damping_nmsprad": self.controller.rotation_damping_nmsprad,
            },
            "action_scale": self.action_scale.as_dict(),
        }


@dataclass(frozen=True)
class WorldWristObservationContractV1:
    dof_count: int
    link_count: int
    lookahead_offsets: tuple[int, ...] = (0, 1, 3, 5)

    @property
    def dimension(self) -> int:
        current = 6 + 6 + self.dof_count + self.dof_count + 26 + 18 + 6 + 6
        per_offset = 6 + 6 + self.dof_count + 18 + 18 + self.link_count * 3 * 2
        return current + len(self.lookahead_offsets) * per_offset

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": "world_wrist_finger_observation_v1",
            "dimension": self.dimension,
            "lookahead_offsets": list(self.lookahead_offsets),
            "normalization": "shared_running_observation_normalizer_no_clip_id",
            "order": [
                "wrist_translation_error_world[3]",
                "wrist_rotation_log_local[3]",
                "wrist_twist_world[6]",
                "finger_q[20]",
                "finger_qdot[20]",
                "previous_action[26]",
                "object_axis_points_world[18]",
                "object_pose_relative_to_wrist[translation3_rotation_log3]",
                "object_twist_world[6]",
                "reference[offset]:wrist_pose[translation3_rotation_log3]",
                "reference[offset]:wrist_twist_world[6]",
                "reference[offset]:finger_q[20]",
                "reference[offset]:object_axis_points_world[18]",
                "reference[offset]:object_axis_points_wrist[18]",
                "reference[offset]:tracked_links_world[L*3]",
                "reference[offset]:tracked_links_wrist[L*3]",
            ],
        }


def _pose_feature(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    return np.concatenate([value[:3, 3], so3_log(value[:3, :3])])


def build_world_wrist_observation(
    *,
    state: dict[str, np.ndarray],
    reference: WorldWristFingerReferenceV1,
    reference_index: int,
    previous_action: np.ndarray,
    contract: WorldWristObservationContractV1 | None = None,
) -> np.ndarray:
    spec = contract or WorldWristObservationContractV1(
        reference.dof_count, len(reference.tracked_link_names)
    )
    if spec.dof_count != reference.dof_count or spec.link_count != len(
        reference.tracked_link_names
    ):
        raise ValueError("world observation contract does not match reference")
    current_wrist = state["wrist_pose"]
    reference_wrist = reference.wrist_pose_world_ref[reference_index]
    wrist_error = np.concatenate(
        [
            reference_wrist[:3, 3] - current_wrist[:3, 3],
            so3_log(current_wrist[:3, :3].T @ reference_wrist[:3, :3]),
        ]
    )
    object_relative = np.eye(4)
    object_relative[:3, :3] = current_wrist[:3, :3].T @ state["object_pose"][:3, :3]
    object_relative[:3, 3] = current_wrist[:3, :3].T @ (
        state["object_pose"][:3, 3] - current_wrist[:3, 3]
    )
    chunks = [
        wrist_error,
        state["wrist_twist"],
        state["q"],
        state["qdot"],
        np.asarray(previous_action, dtype=np.float64),
        state["object_axis_points"].reshape(-1),
        _pose_feature(object_relative),
        state["object_twist"],
    ]
    for offset in spec.lookahead_offsets:
        index = min(max(reference_index + offset, 0), reference.frame_count - 1)
        chunks.extend(
            [
                _pose_feature(reference.wrist_pose_world_ref[index]),
                reference.wrist_twist_world_ref[index],
                reference.q_finger_ref[index],
                reference.object_axis_points_world_ref[index].reshape(-1),
                reference.object_axis_points_wrist_ref[index].reshape(-1),
                reference.tracked_link_positions_world_ref[index].reshape(-1),
                reference.tracked_link_positions_wrist_ref[index].reshape(-1),
            ]
        )
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if result.shape != (spec.dimension,) or not np.isfinite(result).all():
        raise ValueError("world observation violates finite/dimension contract")
    return result


def world_wrist_reward(
    *,
    state: dict[str, np.ndarray],
    reference: WorldWristFingerReferenceV1,
    reference_index: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    second_previous_action: np.ndarray,
) -> dict[str, float]:
    """Engineering reward: preserved tracking terms plus explicit wrist terms."""

    index = int(reference_index)
    axis_error = np.linalg.norm(
        state["object_axis_points"] - reference.object_axis_points_world_ref[index], axis=1
    )
    link_error = np.linalg.norm(
        state["links"] - reference.tracked_link_positions_world_ref[index], axis=1
    )
    normalized_q_error = (state["q"] - reference.q_finger_ref[index]) / (joint_upper - joint_lower)
    wrist_position_error = np.linalg.norm(
        state["wrist_pose"][:3, 3] - reference.wrist_pose_world_ref[index, :3, 3]
    )
    wrist_rotation_error = float(
        rotation_geodesic_error(state["wrist_pose"], reference.wrist_pose_world_ref[index])
    )
    object_term = float(np.exp(-np.square(np.mean(axis_error) / 0.04)))
    link_term = float(np.mean(np.exp(-np.square(link_error / 0.025))))
    finger_term = float(np.mean(np.exp(-np.square(normalized_q_error / 0.1))))
    wrist_position_term = float(np.exp(-np.square(wrist_position_error / 0.02)))
    wrist_rotation_term = float(np.exp(-np.square(wrist_rotation_error / np.deg2rad(10.0))))
    first = np.asarray(action) - np.asarray(previous_action)
    second = (
        np.asarray(action) - 2.0 * np.asarray(previous_action) + np.asarray(second_previous_action)
    )
    smoothness = float(np.sum(first**2) + np.sum(second**2))
    total = (
        8.0 * object_term
        + link_term
        + finger_term
        + 2.0 * wrist_position_term
        + wrist_rotation_term
        - 0.01 * smoothness
    )
    result = {
        "object": object_term,
        "tracked_links": link_term,
        "finger_joints": finger_term,
        "wrist_position": wrist_position_term,
        "wrist_rotation": wrist_rotation_term,
        "smoothness": smoothness,
        "total": float(total),
    }
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError("world wrist reward is non-finite")
    return result


__all__ = [
    "CartesianWristImpedanceController",
    "WORLD_WRIST_BACKEND_ID",
    "WRIST_FINGER_ACTION_SCALE_ID",
    "WRIST_IMPEDANCE_PROFILE_ID",
    "WristControlResult",
    "WristFingerActionScaleV1",
    "WristImpedanceProfileV1",
    "WorldWristFingerBackend",
    "WorldWristObservationContractV1",
    "build_world_wrist_observation",
    "materialize_world_wrist_free_object_scene",
    "world_wrist_reward",
]
