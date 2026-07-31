"""Optional MuJoCo CPU correctness backend with a free object.

This is deliberately a correctness backend, not an author-exact simulator:
the paper does not disclose a simulator, physics solver, PD gains, object
collision asset process, or the target free-body scene.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actuators import residual_target
from ..axis_points import object_axis_points_from_poses
from ..contracts import Stage16ReferenceClip
from ..observations import ObservationContract, ObservationDelayBuffer, build_observation
from ..randomization import (
    DomainRandomizationConfig,
    apply_observation_noise,
    sample_randomization,
)
from ..rewards import paper_literal_reward
from ..termination import (
    PAPER_TERMINATION,
    TerminationInput,
    TerminationProfile,
    classify_termination,
)
from .base import (
    BackendCapabilities,
    PhysicsRandomizationBackend,
    RendererBackend,
    SimulationBackend,
)


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - only exercised without optional extra
        raise RuntimeError("MuJoCo backend requires `pip install -e .[rl]`") from exc
    return mujoco


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0.0:
        raise ValueError("zero object quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if norm == 0.0:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = unit / norm
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def materialize_free_object_scene(
    hand_mjcf: str | Path,
    output_directory: str | Path,
    *,
    object_half_extent_m: float = 0.025,
    object_mesh: str | Path | None = None,
    object_mass_kg: float = 0.05,
    include_ground: bool = True,
    gravity_mps2: tuple[float, float, float] | None = None,
) -> Path:
    """Create an ignored free-object scene without copying source assets.

    With ``object_mesh=None`` this is the synthetic-box qualification scene.
    A supplied mesh is referenced by absolute path and is used unchanged for a
    HOCap clip; physical mass/friction remain documented engineering choices.
    """

    source = Path(hand_mjcf).resolve()
    destination_root = Path(output_directory)
    destination_root.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.01")
    if gravity_mps2 is not None:
        option.set("gravity", " ".join(f"{value:.17g}" for value in gravity_mps2))
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        # The tracked MJCF uses a path relative to its original directory.
        # A generated Stage-16 scene lives under .local, so keep asset loading
        # read-only by resolving that mesh directory rather than copying meshes.
        compiler.set("meshdir", str((source.parent / compiler.get("meshdir", "")).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if include_ground:
        ET.SubElement(
            worldbody,
            "geom",
            name="stage16_ground",
            type="plane",
            size="2 2 0.1",
            rgba="0.2 0.2 0.2 1",
        )
    object_body = ET.SubElement(worldbody, "body", name="stage16_object", pos="0 0 0.15")
    ET.SubElement(object_body, "freejoint", name="stage16_object_free")
    geom_attributes = {
        "name": "stage16_object_geom",
        "mass": str(object_mass_kg),
        "friction": "1 0.005 0.0001",
        "rgba": "0.8 0.3 0.2 1",
    }
    if object_mesh is None:
        extent = f"{object_half_extent_m} {object_half_extent_m} {object_half_extent_m}"
        geom_attributes |= {"type": "box", "size": extent}
    else:
        mesh_path = Path(object_mesh).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Stage-16 object mesh is unavailable: {mesh_path}")
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="stage16_object_mesh", file=str(mesh_path))
        geom_attributes |= {"type": "mesh", "mesh": "stage16_object_mesh"}
    ET.SubElement(object_body, "geom", attrib=geom_attributes)
    destination = destination_root / "wuji_hand2_stage16_free_object.xml"
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


@dataclass(frozen=True)
class MujocoBackendConfig:
    control_dt_s: float = 0.05
    decimation: int = 5
    action_scale_fraction: float = 0.10
    nominal_pd_kp: float = 3.0
    nominal_pd_kd: float = 0.05
    termination_profile: TerminationProfile = PAPER_TERMINATION


class MujocoReferenceTrackingBackend(
    SimulationBackend, PhysicsRandomizationBackend, RendererBackend
):
    """One actual free-object MuJoCo environment with residual PD targets."""

    def __init__(
        self,
        *,
        scene_path: str | Path,
        reference: Stage16ReferenceClip,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        config: MujocoBackendConfig = MujocoBackendConfig(),
        randomization: DomainRandomizationConfig = DomainRandomizationConfig(),
        seed: int = 0,
    ) -> None:
        self.mujoco = _require_mujoco()
        self.reference = reference
        self.reference.validate(expected_hz=20.0)
        self.config = config
        self.randomization_config = randomization
        self.rng = np.random.default_rng(seed)
        self.model = self.mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = self.mujoco.MjData(self.model)
        self.joint_lower = np.asarray(joint_lower, dtype=np.float64)
        self.joint_upper = np.asarray(joint_upper, dtype=np.float64)
        if self.joint_lower.shape != (reference.dof_count,) or self.joint_upper.shape != (
            reference.dof_count,
        ):
            raise ValueError("joint bounds must match reference dimensions")

        def named_ids(kind: Any, names: tuple[str, ...]) -> np.ndarray:
            ids = np.asarray(
                [self.mujoco.mj_name2id(self.model, kind, name) for name in names], dtype=np.int64
            )
            if np.any(ids < 0):
                missing = [name for name, value in zip(names, ids, strict=True) if value < 0]
                raise ValueError(f"reference mapping is absent from MuJoCo asset: {missing}")
            return ids

        joint_ids = named_ids(self.mujoco.mjtObj.mjOBJ_JOINT, reference.joint_order)
        self.joint_qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.joint_dof_addresses = self.model.jnt_dofadr[joint_ids]
        object_joint_id = int(
            named_ids(self.mujoco.mjtObj.mjOBJ_JOINT, ("stage16_object_free",))[0]
        )
        self.object_qpos_address = int(self.model.jnt_qposadr[object_joint_id])
        self.object_dof_address = int(self.model.jnt_dofadr[object_joint_id])
        self.object_body_id = int(named_ids(self.mujoco.mjtObj.mjOBJ_BODY, ("stage16_object",))[0])
        self.link_body_ids = named_ids(self.mujoco.mjtObj.mjOBJ_BODY, reference.tracked_link_names)
        self.actuator_indices = np.asarray(
            [
                np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)[0]
                for joint_id in joint_ids
            ],
            dtype=np.int64,
        )
        if self.model.nu < reference.dof_count or len(np.unique(self.actuator_indices)) != len(
            self.actuator_indices
        ):
            raise ValueError(
                "MuJoCo actuators do not provide one position target per reference joint"
            )
        self._hand_body_ids = np.asarray(
            [body_id for body_id in range(1, self.model.nbody) if body_id != self.object_body_id],
            dtype=np.int64,
        )
        self._hand_geom_ids = np.flatnonzero(
            np.isin(self.model.geom_bodyid, self._hand_body_ids)
        ).astype(np.int64)
        self._nominal_model = {
            name: getattr(self.model, name).copy()
            for name in (
                "actuator_biasprm",
                "actuator_gainprm",
                "body_inertia",
                "body_ipos",
                "body_mass",
                "dof_armature",
                "dof_damping",
                "dof_frictionloss",
                "geom_friction",
                "geom_size",
            )
        }
        self.reference_index = 0
        self.step_index = 0
        self.previous_action = np.zeros(reference.dof_count)
        self.second_previous_action = np.zeros(reference.dof_count)
        self._sample = sample_randomization(self.rng, randomization)
        self._observation_delay = ObservationDelayBuffer(self._sample["observation_delay_steps"])
        self._next_disturbance_time_s = float(self._sample["next_disturbance_s"])

    def _restore_nominal_model(self) -> None:
        for name, value in self._nominal_model.items():
            getattr(self.model, name)[:] = value

    def _apply_randomization(self) -> None:
        """Apply each selected Table-5 property from immutable nominal values."""

        self._restore_nominal_model()
        sample = self._sample
        self.model.body_ipos[self.object_body_id] += np.asarray(sample["object_com_offset_m"])
        self.model.body_mass[self.object_body_id] *= sample["object_mass_inertia_scale"]
        self.model.body_inertia[self.object_body_id] *= sample["object_mass_inertia_scale"]
        self.model.geom_friction[self._hand_geom_ids] *= sample["robot_friction_scale"]
        self.model.geom_size[self._hand_geom_ids] *= sample["robot_collision_geometry_scale"]
        self.model.actuator_gainprm[self.actuator_indices, 0] *= sample["pd_stiffness_scale"]
        self.model.actuator_biasprm[self.actuator_indices, 1] *= sample["pd_stiffness_scale"]
        self.model.actuator_biasprm[self.actuator_indices, 2] *= sample["pd_damping_scale"]
        self.model.dof_damping[self.joint_dof_addresses] *= sample["joint_damping_scale"]
        self.model.dof_armature[self.joint_dof_addresses] *= sample["joint_armature_scale"]
        self.model.dof_frictionloss[self.joint_dof_addresses] *= sample["joint_friction_loss_scale"]
        self.model.body_mass[self._hand_body_ids] *= sample["robot_link_mass_scale"]
        self.model.body_inertia[self._hand_body_ids] *= sample["robot_link_inertia_scale"]
        self.mujoco.mj_setConst(self.model, self.data)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            vectorized=False,
            free_object_dynamics=True,
            contact=True,
            renderer=True,
            deterministic_reset=True,
            supported_randomizations=(
                "observation_noise",
                "observation_delay",
                "reset_noise",
                "object_com",
                "robot_friction_and_geometry",
                "object_mass_and_inertia",
                "pd",
                "joint_dynamics",
                "encoder_bias",
                "robot_link_mass_and_inertia",
                "external_disturbance",
            ),
        )

    def apply_randomization(self, config: DomainRandomizationConfig) -> None:
        """Install an explicit Table-5 configuration for the next reset."""

        self.randomization_config = config

    def render_rgb(self, *, width: int = 640, height: int = 480) -> np.ndarray:
        """Render one RGB frame when the local MuJoCo rendering backend is available."""

        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data)
            return renderer.render().copy()
        finally:
            renderer.close()

    def _current_object_pose(self) -> np.ndarray:
        position = self.data.qpos[self.object_qpos_address : self.object_qpos_address + 3]
        rotation = _quat_wxyz_to_matrix(
            self.data.qpos[self.object_qpos_address + 3 : self.object_qpos_address + 7]
        )
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose

    def _state(self) -> dict[str, np.ndarray]:
        q = self.data.qpos[self.joint_qpos_addresses].copy()
        qdot = self.data.qvel[self.joint_dof_addresses].copy()
        pose = self._current_object_pose()
        axes = object_axis_points_from_poses(pose[None])[0]
        links = self.data.xpos[self.link_body_ids].copy()
        object_velocity = self.data.qvel[
            self.object_dof_address : self.object_dof_address + 6
        ].copy()
        return {
            "q": q,
            "qdot": qdot,
            "object_pose": pose,
            "object_axis_points": axes,
            "links": links,
            "object_velocity": object_velocity,
        }

    def reset(self, **kwargs: Any) -> dict[str, np.ndarray]:
        requested_reference_index = kwargs.pop("reference_index", None)
        if kwargs:
            raise TypeError(f"unsupported reset keyword arguments: {sorted(kwargs)}")
        if requested_reference_index is None:
            self.reference_index = int(self.rng.integers(0, self.reference.frame_count))
        else:
            self.reference_index = int(requested_reference_index)
            if not 0 <= self.reference_index < self.reference.frame_count:
                raise ValueError("reference_index is outside the reference clip")
        self.step_index = 0
        self.previous_action.fill(0.0)
        self.second_previous_action.fill(0.0)
        self._sample = sample_randomization(self.rng, self.randomization_config)
        self._observation_delay = ObservationDelayBuffer(self._sample["observation_delay_steps"])
        self._next_disturbance_time_s = float(self._sample["next_disturbance_s"])
        self._apply_randomization()
        self.mujoco.mj_resetData(self.model, self.data)
        q_noise = self.rng.uniform(
            *self.randomization_config.reset_joint_range_rad,
            size=self.reference.dof_count,
        )
        if not self._sample["active_switches"]["reference_reset"]:
            q_noise.fill(0.0)
        self.data.qpos[self.joint_qpos_addresses] = np.clip(
            self.reference.q_finger_ref[self.reference_index] + q_noise,
            self.joint_lower,
            self.joint_upper,
        )
        pose = self.reference.object_pose_base_ref[self.reference_index]
        object_position_noise = np.asarray(
            self._sample.get("reset_object_position_noise_m", [0.0, 0.0, 0.0])
        )
        self.data.qpos[self.object_qpos_address : self.object_qpos_address + 3] = (
            pose[:3, 3] + object_position_noise
        )
        pose_rotation = pose[:3, :3] @ _rotation_from_axis_angle(
            np.asarray(self._sample["reset_object_orientation_axis"]),
            float(self._sample["reset_object_orientation_angle_rad"]),
        )
        # MuJoCo free joint uses wxyz; a reference matrix is converted through mj_mat2Quat.
        quaternion = np.empty(4)
        self.mujoco.mju_mat2Quat(quaternion, pose_rotation.reshape(-1))
        self.data.qpos[self.object_qpos_address + 3 : self.object_qpos_address + 7] = quaternion
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        target = residual_target(
            self.reference.q_finger_ref[self.reference_index],
            action,
            self.joint_lower,
            self.joint_upper,
            action_scale_fraction=self.config.action_scale_fraction,
        )
        self.data.ctrl[self.actuator_indices] = target
        for _ in range(self.config.decimation):
            self.data.xfrc_applied[self.object_body_id].fill(0.0)
            if self.data.time >= self._next_disturbance_time_s:
                self.data.xfrc_applied[self.object_body_id, :3] = self._sample[
                    "external_object_force_n"
                ]
                self.data.xfrc_applied[self.object_body_id, 3:] = self._sample[
                    "external_object_torque_nm"
                ]
                self._next_disturbance_time_s += float(
                    self.rng.uniform(*self.randomization_config.disturbance_interval_s)
                )
            self.mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        self.reference_index = min(self.reference_index + 1, self.reference.frame_count - 1)
        return self._state()

    def observation(self, state: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Build the policy observation with only Table-5 observed quantities perturbed."""

        current = self._state() if state is None else state
        q, qdot, axes = apply_observation_noise(
            q=current["q"],
            qdot=current["qdot"],
            axis_points=current["object_axis_points"],
            rng=self.rng,
            config=self.randomization_config,
        )
        q = q + float(self._sample["encoder_bias_rad"])
        q, qdot, axes = self._observation_delay.push(q, qdot, axes)
        return build_observation(
            q=q,
            qdot=qdot,
            previous_action=self.previous_action,
            current_object_axis_points=axes,
            reference=self.reference,
            reference_index=self.reference_index,
            contract=ObservationContract(
                dof_count=self.reference.dof_count,
                link_count=len(self.reference.tracked_link_names),
            ),
        )

    def transition(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, float], str | None]:
        previous_action = self.previous_action.copy()
        second_previous_action = self.second_previous_action.copy()
        state = self.step(action)
        target = self.reference_index
        reward = paper_literal_reward(
            object_axis_points=state["object_axis_points"],
            object_axis_points_ref=self.reference.object_axis_points_base_ref[target],
            link_positions=state["links"],
            link_positions_ref=self.reference.tracked_link_positions_base_ref[target],
            q=state["q"],
            q_ref=self.reference.q_finger_ref[target],
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            action=action,
            previous_action=previous_action,
            second_previous_action=second_previous_action,
        )
        self.second_previous_action = previous_action
        self.previous_action = np.asarray(action, dtype=np.float64).copy()
        reference_pose = self.reference.object_pose_base_ref[target]
        position_error = float(np.linalg.norm(state["object_pose"][:3, 3] - reference_pose[:3, 3]))
        relative = state["object_pose"][:3, :3].T @ reference_pose[:3, :3]
        orientation_error = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        termination = classify_termination(
            TerminationInput(
                step=self.step_index,
                reference_index=target,
                reference_frame_count=self.reference.frame_count,
                object_height_m=float(state["object_pose"][2, 3]),
                object_linear_velocity_mps=float(np.linalg.norm(state["object_velocity"][:3])),
                object_angular_velocity_radps=float(np.linalg.norm(state["object_velocity"][3:])),
                object_position_error_m=position_error,
                object_orientation_error_rad=orientation_error,
                max_axis_point_error_m=float(
                    np.max(
                        np.linalg.norm(
                            state["object_axis_points"]
                            - self.reference.object_axis_points_base_ref[target],
                            axis=-1,
                        )
                    )
                ),
            ),
            profile=self.config.termination_profile,
        )
        return state, reward, None if termination is None else termination.value


__all__ = ["MujocoBackendConfig", "MujocoReferenceTrackingBackend", "materialize_free_object_scene"]
