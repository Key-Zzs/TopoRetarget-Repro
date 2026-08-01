"""Optional MuJoCo CPU correctness backend with a free object.

This is deliberately a correctness backend, not an author-exact simulator:
the paper does not disclose a simulator, physics solver, PD gains, object
collision asset process, or the target free-body scene.
"""

from __future__ import annotations

import copy
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
    # The upstream Wuji asset intentionally leaves its collision geoms unnamed.
    # Give the *generated* Stage-16 scene stable audit-only labels.  This makes
    # every contact trace attributable without changing source assets or any
    # collision parameter.  Visual geoms (contype=conaffinity=0) remain unnamed.
    collision_index = 0
    for body in root.findall(".//body"):
        body_name = body.get("name", "unnamed_body")
        for geom in body.findall("geom"):
            contype = int(geom.get("contype", "1"))
            conaffinity = int(geom.get("conaffinity", "1"))
            if geom.get("name") is None and (contype != 0 or conaffinity != 0):
                geom.set("name", f"stage16_hand_collision_{body_name}_{collision_index:02d}")
                collision_index += 1
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


@dataclass
class MujocoBackendSnapshot:
    """Complete mutable state needed for deterministic diagnostic lookahead.

    This is deliberately a backend snapshot rather than object-state access.
    Object-aware diagnostics may clone a state and predict a bounded finger
    action, but cannot write object state in the live rollout.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    reference_index: int
    step_index: int
    previous_action: np.ndarray
    second_previous_action: np.ndarray
    next_disturbance_time_s: float
    rng_state: Any


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
        self._reference_qdot: np.ndarray | None = (
            None if reference.qdot_ref is None else reference.qdot_ref.copy()
        )
        self._reference_object_velocity: np.ndarray | None = (
            None if reference.object_velocity_ref is None else reference.object_velocity_ref.copy()
        )
        self.last_physics_trace: list[dict[str, Any]] = []

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

    def set_reference_velocities(self, *, qdot: np.ndarray, object_velocity: np.ndarray) -> None:
        """Install externally audited finite-difference reference velocities.

        The source clips may omit velocity arrays.  Stage-16.1 computes them
        from immutable 20 Hz poses, validates them, and supplies them here for
        the four reset profiles.  This method never alters reference poses.
        """

        qdot_array = np.asarray(qdot, dtype=np.float64)
        object_velocity_array = np.asarray(object_velocity, dtype=np.float64)
        expected_qdot = (self.reference.frame_count, self.reference.dof_count)
        expected_velocity = (self.reference.frame_count, 6)
        if qdot_array.shape != expected_qdot:
            raise ValueError(f"qdot shape must be {expected_qdot}")
        if object_velocity_array.shape != expected_velocity:
            raise ValueError(f"object_velocity shape must be {expected_velocity}")
        if not np.isfinite(qdot_array).all() or not np.isfinite(object_velocity_array).all():
            raise ValueError("reference velocities must be finite")
        self._reference_qdot = qdot_array.copy()
        self._reference_object_velocity = object_velocity_array.copy()

    def _apply_velocity_reset(self, profile: str) -> None:
        valid_profiles = {"zero", "full_reference", "object_reference", "hand_reference"}
        if profile not in valid_profiles:
            raise ValueError(f"unknown reset velocity profile {profile!r}")
        if profile == "zero":
            return
        if self._reference_qdot is None or self._reference_object_velocity is None:
            raise RuntimeError(
                "reference velocity profile requested before velocities were installed"
            )
        if profile in {"full_reference", "hand_reference"}:
            self.data.qvel[self.joint_dof_addresses] = self._reference_qdot[self.reference_index]
        if profile in {"full_reference", "object_reference"}:
            start = self.object_dof_address
            self.data.qvel[start : start + 6] = self._reference_object_velocity[
                self.reference_index
            ]

    def _set_object_to_reference(self, reference_index: int) -> None:
        """Drive only the diagnostic kinematic object; never use in formal control."""

        pose = self.reference.object_pose_base_ref[reference_index]
        self.data.qpos[self.object_qpos_address : self.object_qpos_address + 3] = pose[:3, 3]
        quaternion = np.empty(4)
        self.mujoco.mju_mat2Quat(quaternion, pose[:3, :3].reshape(-1))
        self.data.qpos[self.object_qpos_address + 3 : self.object_qpos_address + 7] = quaternion
        if self._reference_object_velocity is not None:
            start = self.object_dof_address
            self.data.qvel[start : start + 6] = self._reference_object_velocity[reference_index]

    def snapshot(self) -> MujocoBackendSnapshot:
        """Capture all mutable simulator/controller state for local simulation."""

        return MujocoBackendSnapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            act=self.data.act.copy(),
            ctrl=self.data.ctrl.copy(),
            time=float(self.data.time),
            reference_index=self.reference_index,
            step_index=self.step_index,
            previous_action=self.previous_action.copy(),
            second_previous_action=self.second_previous_action.copy(),
            next_disturbance_time_s=float(self._next_disturbance_time_s),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, snapshot: MujocoBackendSnapshot) -> None:
        """Restore a snapshot created by :meth:`snapshot` exactly."""

        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.act[:] = snapshot.act
        self.data.ctrl[:] = snapshot.ctrl
        self.data.time = snapshot.time
        self.reference_index = snapshot.reference_index
        self.step_index = snapshot.step_index
        self.previous_action[:] = snapshot.previous_action
        self.second_previous_action[:] = snapshot.second_previous_action
        self._next_disturbance_time_s = snapshot.next_disturbance_time_s
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        self.mujoco.mj_forward(self.model, self.data)

    def predict_step(
        self, action: np.ndarray, *, kinematic_object: bool = False
    ) -> dict[str, np.ndarray]:
        """One bounded cloned rollout for an engineering diagnostic oracle."""

        snapshot = self.snapshot()
        try:
            return self.step(action, kinematic_object=kinematic_object)
        finally:
            self.restore(snapshot)

    def contact_report(self, *, proximity_m: float = 0.01) -> dict[str, Any]:
        """Return contact and collision evidence in a JSON-friendly structure.

        `object_wrench_body` is MuJoCo's accumulated external wrench in the
        object body frame. Per-contact forces are expressed in world axes with
        an explicit geometry ordering and are intended for diagnosis, not for
        force-control commands.
        """

        object_geom = int(
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "stage16_object_geom")
        )
        object_position = self.data.xpos[self.object_body_id].copy()
        contacts: list[dict[str, Any]] = []
        object_wrench_world = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            force_local = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, force_local)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_world = frame @ force_local[:3]
            torque_world = frame @ force_local[3:]
            is_object_pair = object_geom in {geom1, geom2}
            if is_object_pair:
                # MuJoCo's contact frame force points from geom1 toward geom2.
                sign = 1.0 if geom2 == object_geom else -1.0
                object_force = sign * force_world
                object_torque = sign * (
                    torque_world + np.cross(np.asarray(contact.pos) - object_position, force_world)
                )
                object_wrench_world[:3] += object_force
                object_wrench_world[3:] += object_torque
            contacts.append(
                {
                    "geom1_id": geom1,
                    "geom2_id": geom2,
                    "geom1": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom1
                    ),
                    "geom2": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom2
                    ),
                    "body1": self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom1]),
                    ),
                    "body2": self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom2]),
                    ),
                    "position": np.asarray(contact.pos, dtype=np.float64).tolist(),
                    "normal": frame[:, 0].tolist(),
                    "distance_m": float(contact.dist),
                    "friction": np.asarray(contact.friction, dtype=np.float64).tolist(),
                    "force_local": force_local.tolist(),
                    "force_world": force_world.tolist(),
                    "is_hand_object": is_object_pair,
                }
            )
        proximity: list[dict[str, Any]] = []
        for hand_geom in self._hand_geom_ids:
            if (
                self.model.geom_contype[hand_geom] == 0
                or self.model.geom_conaffinity[hand_geom] == 0
            ):
                continue
            fromto = np.empty(6, dtype=np.float64)
            distance = float(
                self.mujoco.mj_geomDistance(
                    self.model, self.data, int(hand_geom), object_geom, proximity_m, fromto
                )
            )
            if distance < proximity_m:
                proximity.append(
                    {
                        "hand_geom": self.mujoco.mj_id2name(
                            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(hand_geom)
                        ),
                        "hand_body": self.mujoco.mj_id2name(
                            self.model,
                            self.mujoco.mjtObj.mjOBJ_BODY,
                            int(self.model.geom_bodyid[hand_geom]),
                        ),
                        "distance_m": distance,
                        "fromto": fromto.tolist(),
                    }
                )
        return {
            "ncon": int(self.data.ncon),
            "hand_object_contact_count": int(sum(row["is_hand_object"] for row in contacts)),
            "contacts": contacts,
            "expected_proximity_contact_set": proximity,
            "object_wrench_world": object_wrench_world.tolist(),
            "object_wrench_body": self.data.cfrc_ext[self.object_body_id].copy().tolist(),
            "object_geom": {
                "name": "stage16_object_geom",
                "body": "stage16_object",
                "contype": int(self.model.geom_contype[object_geom]),
                "conaffinity": int(self.model.geom_conaffinity[object_geom]),
                "friction": self.model.geom_friction[object_geom].copy().tolist(),
            },
        }

    def collision_configuration(self) -> list[dict[str, Any]]:
        """Expose generated hand/object collision settings for the static audit."""

        rows: list[dict[str, Any]] = []
        for geom_id in [*self._hand_geom_ids.tolist()]:
            if self.model.geom_contype[geom_id] == 0 and self.model.geom_conaffinity[geom_id] == 0:
                continue
            rows.append(
                {
                    "geom": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                    ),
                    "body": self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom_id]),
                    ),
                    "contype": int(self.model.geom_contype[geom_id]),
                    "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                    "friction": self.model.geom_friction[geom_id].copy().tolist(),
                    "size": self.model.geom_size[geom_id].copy().tolist(),
                }
            )
        object_geom = int(
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "stage16_object_geom")
        )
        rows.append(
            {
                "geom": "stage16_object_geom",
                "body": "stage16_object",
                "contype": int(self.model.geom_contype[object_geom]),
                "conaffinity": int(self.model.geom_conaffinity[object_geom]),
                "friction": self.model.geom_friction[object_geom].copy().tolist(),
                "size": self.model.geom_size[object_geom].copy().tolist(),
            }
        )
        return rows

    def _physics_trace_row(self, *, substep: int) -> dict[str, Any]:
        """Capture one post-step state for diagnostic-only full rollouts."""

        state = self._state()
        ctrl_low = self.model.actuator_ctrlrange[self.actuator_indices, 0]
        ctrl_high = self.model.actuator_ctrlrange[self.actuator_indices, 1]
        ctrl = self.data.ctrl[self.actuator_indices].copy()
        margin = np.minimum(state["q"] - self.joint_lower, self.joint_upper - state["q"])
        return {
            "physics_substep": substep,
            "time_s": float(self.data.time),
            "reference_index": self.reference_index,
            "q": state["q"].tolist(),
            "qdot": state["qdot"].tolist(),
            "object_qpos": self.data.qpos[
                self.object_qpos_address : self.object_qpos_address + 7
            ].tolist(),
            "object_qvel": self.data.qvel[
                self.object_dof_address : self.object_dof_address + 6
            ].tolist(),
            "ctrl": ctrl.tolist(),
            "ctrl_saturated": (np.isclose(ctrl, ctrl_low) | np.isclose(ctrl, ctrl_high)).tolist(),
            "actuator_force": self.data.actuator_force[self.actuator_indices].copy().tolist(),
            "joint_limit_margin": margin.tolist(),
            "object_kinetic_energy": float(self.data.energy[1]),
            "contact": self.contact_report(),
        }

    def reset(self, **kwargs: Any) -> dict[str, np.ndarray]:
        requested_reference_index = kwargs.pop("reference_index", None)
        velocity_profile = str(kwargs.pop("velocity_profile", "zero"))
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
        self._apply_velocity_reset(velocity_profile)
        self.mujoco.mj_forward(self.model, self.data)
        return self._state()

    def step(self, action: np.ndarray, *, kinematic_object: bool = False) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        target = residual_target(
            self.reference.q_finger_ref[self.reference_index],
            action,
            self.joint_lower,
            self.joint_upper,
            action_scale_fraction=self.config.action_scale_fraction,
        )
        self.data.ctrl[self.actuator_indices] = target
        self.last_physics_trace = []
        for substep in range(self.config.decimation):
            if kinematic_object:
                self._set_object_to_reference(self.reference_index)
                self.mujoco.mj_forward(self.model, self.data)
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
            self.last_physics_trace.append(self._physics_trace_row(substep=substep))
        self.step_index += 1
        self.reference_index = min(self.reference_index + 1, self.reference.frame_count - 1)
        if kinematic_object:
            # The physics substeps use the current reference frame, then the
            # returned state is aligned with the incremented control target.
            self._set_object_to_reference(self.reference_index)
            self.mujoco.mj_forward(self.model, self.data)
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
        self, action: np.ndarray, *, kinematic_object: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, float], str | None]:
        previous_action = self.previous_action.copy()
        second_previous_action = self.second_previous_action.copy()
        state = self.step(action, kinematic_object=kinematic_object)
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


__all__ = [
    "MujocoBackendConfig",
    "MujocoBackendSnapshot",
    "MujocoReferenceTrackingBackend",
    "materialize_free_object_scene",
]
