#!/usr/bin/env python3
"""Execute the bounded Stage-16.1a hand--object dynamic-coupling protocol.

This is an engineering qualification runner.  It preserves the formal
reference/action/termination contracts and records first threshold crossings
without truncating diagnostic physics rollouts.  It never starts PPO.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.dynamic_coupling import (
    ObjectAwareResidualOracle,
    ObjectAwareShootingOracle,
    ResetVelocityProfile,
    reference_acceleration,
    reference_velocities,
)
from toporetarget.rl.environments.mujoco_backend import (
    MujocoBackendConfig,
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.oracle import OracleResidualController, oracle_action
from toporetarget.rl.state_machine import (
    DynamicCouplingPhase,
    Stage161DynamicCouplingStateMachine,
)
from toporetarget.rl.termination import (
    BASE_RELATIVE_HOCAP_TERMINATION,
    TerminationInput,
    classify_termination,
)

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
AUDIT_FRAMES = (0, 5, 10, 20, 30, 40)
PRELOAD_CANDIDATES = (0.0, 0.01, 0.02, 0.05)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def _rotation_error_deg(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = actual[:3, :3].T @ reference[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _termination(
    backend: MujocoReferenceTrackingBackend, state: dict[str, np.ndarray]
) -> str | None:
    index = backend.reference_index
    reference = backend.reference
    reference_pose = reference.object_pose_base_ref[index]
    position_error = float(np.linalg.norm(state["object_pose"][:3, 3] - reference_pose[:3, 3]))
    relative = state["object_pose"][:3, :3].T @ reference_pose[:3, :3]
    orientation_error = float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    classification = classify_termination(
        TerminationInput(
            step=backend.step_index,
            reference_index=index,
            reference_frame_count=reference.frame_count,
            object_height_m=float(state["object_pose"][2, 3]),
            object_linear_velocity_mps=float(np.linalg.norm(state["object_velocity"][:3])),
            object_angular_velocity_radps=float(np.linalg.norm(state["object_velocity"][3:])),
            object_position_error_m=position_error,
            object_orientation_error_rad=orientation_error,
            max_axis_point_error_m=float(
                np.max(
                    np.linalg.norm(
                        state["object_axis_points"] - reference.object_axis_points_base_ref[index],
                        axis=1,
                    )
                )
            ),
        ),
        profile=BASE_RELATIVE_HOCAP_TERMINATION,
    )
    return None if classification is None else classification.value


def _metrics(
    backend: MujocoReferenceTrackingBackend, state: dict[str, np.ndarray]
) -> dict[str, float]:
    index = backend.reference_index
    reference = backend.reference
    pose = reference.object_pose_base_ref[index]
    return {
        "object_position_error_m": float(np.linalg.norm(state["object_pose"][:3, 3] - pose[:3, 3])),
        "object_rotation_error_deg": _rotation_error_deg(state["object_pose"], pose),
        "max_axis_error_m": float(
            np.max(
                np.linalg.norm(
                    state["object_axis_points"] - reference.object_axis_points_base_ref[index],
                    axis=1,
                )
            )
        ),
        "joint_rmse_rad": float(
            np.sqrt(np.mean(np.square(state["q"] - reference.q_finger_ref[index])))
        ),
        "link_rmse_m": float(
            np.sqrt(
                np.mean(
                    np.square(state["links"] - reference.tracked_link_positions_base_ref[index])
                )
            )
        ),
    }


def _make_backend(
    reference_path: Path,
    mesh_path: Path,
    scene_root: Path,
    *,
    seed: int,
) -> MujocoReferenceTrackingBackend:
    hand_model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = hand_model.jnt_range[: hand_model.njnt].copy()
    reference = Stage16ReferenceClip.from_npz(reference_path)
    reference.validate(expected_hz=20.0)
    joint_order = tuple(
        mujoco.mj_id2name(hand_model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(hand_model.njnt)
    )
    if reference.joint_order != joint_order:
        raise ValueError(f"reference joint order differs from Wuji MJCF: {reference_path}")
    scene = materialize_free_object_scene(
        WUJI_MJCF,
        scene_root / reference_path.stem,
        object_mesh=mesh_path,
        object_mass_kg=0.05,
        include_ground=False,
        gravity_mps2=(0.0, 0.0, 0.0),
    )
    backend = MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=bounds[:, 0],
        joint_upper=bounds[:, 1],
        config=MujocoBackendConfig(
            action_scale_fraction=0.05,
            termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
        ),
        seed=seed,
    )
    qdot, object_velocity = reference_velocities(reference)
    backend.set_reference_velocities(qdot=qdot, object_velocity=object_velocity)
    return backend


def _apply_preload(
    backend: MujocoReferenceTrackingBackend, fraction: float
) -> dict[str, np.ndarray]:
    """Apply a global normalized flexion preload from actuator joint semantics.

    In the Wuji position-actuator model, increasing the normalized range is the
    declared finger-flexion direction.  This uses no object or clip identifier,
    affects every formal action joint equally, and is only an engineering probe.
    """

    if fraction < 0.0:
        raise ValueError("preload fraction must be non-negative")
    target = np.clip(
        backend.data.qpos[backend.joint_qpos_addresses]
        + fraction * (backend.joint_upper - backend.joint_lower),
        backend.joint_lower,
        backend.joint_upper,
    )
    backend.data.qpos[backend.joint_qpos_addresses] = target
    backend.mujoco.mj_forward(backend.model, backend.data)
    return backend._state()


Policy = Callable[[MujocoReferenceTrackingBackend, dict[str, np.ndarray]], np.ndarray]


def _rollout(
    backend: MujocoReferenceTrackingBackend,
    policy: Policy,
    *,
    velocity_profile: ResetVelocityProfile,
    kinematic_object: bool,
    preload_fraction: float = 0.0,
    label: str,
) -> dict[str, object]:
    """Run every one of the 41 frames while retaining post-failure evidence."""

    state = backend.reset(reference_index=0, velocity_profile=velocity_profile.value)
    if preload_fraction:
        state = _apply_preload(backend, preload_fraction)
    frames: list[dict[str, object]] = []
    physics: list[dict[str, object]] = []
    first_termination: dict[str, object] | None = None
    action_rows: list[list[float]] = []
    for control_step in range(backend.reference.frame_count - 1):
        reference_before = backend.reference_index
        action = np.asarray(policy(backend, state), dtype=np.float64)
        if action.shape != (backend.reference.dof_count,) or np.any(np.abs(action) > 1.0 + 1e-12):
            raise ValueError("diagnostic policy must return a bounded formal 20D residual action")
        state = backend.step(action, kinematic_object=kinematic_object)
        if isinstance(policy, ObjectAwareResidualOracle):
            policy.record_actual_error(backend, state)
        reason = _termination(backend, state)
        measurement = _metrics(backend, state)
        contact = backend.contact_report()
        if reason is not None and first_termination is None:
            first_termination = {
                "control_step": control_step,
                "reference_index": backend.reference_index,
                "reason": reason,
                "metrics": measurement,
            }
        frames.append(
            {
                "label": label,
                "control_step": control_step,
                "reference_index_before": reference_before,
                "reference_index": backend.reference_index,
                "action": action.tolist(),
                "action_abs_max": float(np.max(np.abs(action))),
                "termination": reason,
                "kinematic_object": kinematic_object,
                "contact_count": contact["hand_object_contact_count"],
                "object_wrench_norm": float(np.linalg.norm(contact["object_wrench_world"])),
                "oracle": (
                    None
                    if not isinstance(policy, ObjectAwareResidualOracle)
                    or policy.last_diagnostics is None
                    else policy.last_diagnostics.json()
                ),
                **measurement,
            }
        )
        for physical in backend.last_physics_trace:
            physics.append({"label": label, "control_step": control_step, **physical})
        action_rows.append(action.tolist())
    final = _metrics(backend, state)
    return {
        "label": label,
        "diagnostic_continue_after_termination": True,
        "kinematic_object": kinematic_object,
        "velocity_profile": velocity_profile.value,
        "preload_fraction": preload_fraction,
        "frames_completed": len(frames),
        "first_termination": first_termination,
        "formal_success": first_termination is None,
        "final_metrics": final,
        "frames": frames,
        "physics": physics,
        "actions": action_rows,
        "final_contact": backend.contact_report(),
    }


def _zero_policy(
    backend: MujocoReferenceTrackingBackend, _state: dict[str, np.ndarray]
) -> np.ndarray:
    return np.zeros(backend.reference.dof_count, dtype=np.float64)


def _joint_oracle_policy(
    controller: OracleResidualController,
) -> Policy:
    def policy(backend: MujocoReferenceTrackingBackend, state: dict[str, np.ndarray]) -> np.ndarray:
        next_index = min(backend.reference_index + 1, backend.reference.frame_count - 1)
        return oracle_action(
            controller,
            state=state,
            reference_q=backend.reference.q_finger_ref[backend.reference_index],
            next_reference_q=backend.reference.q_finger_ref[next_index],
            joint_lower=backend.joint_lower,
            joint_upper=backend.joint_upper,
        )

    return policy


def _contact_audit(
    backend: MujocoReferenceTrackingBackend, *, preload_fraction: float
) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for frame in AUDIT_FRAMES:
        state = backend.reset(
            reference_index=frame, velocity_profile=ResetVelocityProfile.ZERO.value
        )
        if preload_fraction:
            state = _apply_preload(backend, preload_fraction)
        before = backend.contact_report()
        initial_position = state["object_pose"][:3, 3].copy()
        # A bounded 20D generic flexion pulse.  This is a contact diagnostic,
        # never an object control command or a formal policy episode.
        pulse = np.full(backend.reference.dof_count, 0.10, dtype=np.float64)
        state = backend.step(pulse)
        after = backend.contact_report()
        frames.append(
            {
                "frame": frame,
                "preload_fraction": preload_fraction,
                "initial": before,
                "after_generic_finger_pulse": after,
                "push_displacement_m": float(
                    np.linalg.norm(state["object_pose"][:3, 3] - initial_position)
                ),
                "push_speed_mps": float(np.linalg.norm(state["object_velocity"][:3])),
            }
        )
    counts = [int(row["initial"]["hand_object_contact_count"]) for row in frames]
    wrench = [float(np.linalg.norm(row["initial"]["object_wrench_world"])) for row in frames]
    displacement = [float(row["push_displacement_m"]) for row in frames]
    return {
        "preload_fraction": preload_fraction,
        "collision_configuration": backend.collision_configuration(),
        "frames": frames,
        "summary": {
            "contacts_total": int(sum(counts)),
            "max_contact_wrench": float(max(wrench)),
            "max_push_displacement_m": float(max(displacement)),
        },
    }


def _plot(root: Path, rollouts: list[dict[str, object]]) -> list[str]:
    """Generate numerical plots that remain useful if offscreen GL is unavailable."""

    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for rollout in rollouts:
        frames = rollout["frames"]
        if not isinstance(frames, list) or not frames:
            continue
        label = str(rollout["label"])
        step = [int(row["control_step"]) for row in frames]
        figure, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
        axes[0].plot(
            step, [float(row["object_position_error_m"]) for row in frames], label="position m"
        )
        axes[0].plot(step, [float(row["max_axis_error_m"]) for row in frames], label="axis m")
        axes[0].axhline(0.05, color="tab:red", linestyle="--", label="formal 5 cm gate")
        axes[0].legend()
        axes[1].plot(step, [float(row["joint_rmse_rad"]) for row in frames], label="joint RMSE rad")
        axes[1].plot(step, [float(row["link_rmse_m"]) for row in frames], label="link RMSE m")
        axes[1].legend()
        axes[2].plot(
            step, [float(row["contact_count"]) for row in frames], label="hand-object contacts"
        )
        axes[2].plot(
            step, [float(row["object_wrench_norm"]) for row in frames], label="object wrench"
        )
        axes[2].set_xlabel("control step")
        axes[2].legend()
        figure.suptitle(f"Stage16.1a diagnostic — {label}")
        path = root / f"{label}_curves.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def _contact_sheet(paths: list[str], destination: Path) -> str | None:
    """Build a compact contact sheet from numerical diagnostic plot frames."""

    if not paths:
        return None
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in paths]
    width = min(image.width for image in images)
    height = min(image.height for image in images)
    columns = 2
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        image.thumbnail((width, height))
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(destination)
    return str(destination.resolve())


def _write_dashboard(path: Path, summary: dict[str, object], visual_paths: list[str]) -> str:
    links = "\n".join(f'<li><a href="{item}">{Path(item).name}</a></li>' for item in visual_paths)
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Stage16.1a dashboard</title>'
        "<h1>Stage16.1a dynamic-coupling diagnostic</h1>"
        f"<pre>{json.dumps(summary, indent=2, default=_json_default)}</pre><ul>{links}</ul>",
        encoding="utf-8",
    )
    return str(path.resolve())


def _summary_row(rollout: dict[str, object]) -> dict[str, object]:
    first = rollout["first_termination"]
    final = rollout["final_metrics"]
    if not isinstance(final, dict):
        raise TypeError("rollout final metrics are invalid")
    return {
        "label": rollout["label"],
        "frames_completed": rollout["frames_completed"],
        "first_failure_frame": None if first is None else first["reference_index"],
        "first_failure_reason": None if first is None else first["reason"],
        "progress": (1.0 if first is None else float(first["reference_index"]) / 40.0),
        "final_position_cm": float(final["object_position_error_m"]) * 100.0,
        "final_axis_cm": float(final["max_axis_error_m"]) * 100.0,
        "final_rotation_deg": final["object_rotation_error_deg"],
        "final_joint_rmse_rad": final["joint_rmse_rad"],
        "final_link_rmse_mm": float(final["link_rmse_m"]) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", type=Path)
    parser.add_argument("--object-mesh", action="append", type=Path)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(".local/experiments/stage16_dynamic_coupling_v1"),
    )
    parser.add_argument(
        "--report-root", type=Path, default=Path(".local/reports/stage16_dynamic_coupling_v1")
    )
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    references = args.reference or [
        Path(".local/stage16_reference_tracking_ppo/references/hocap_170105.stage16.npz"),
        Path(".local/stage16_reference_tracking_ppo/references/hocap_170650.stage16.npz"),
    ]
    meshes = args.object_mesh or [
        Path(".local/stage16_reference_tracking_ppo/objects/hocap_170105.obj"),
        Path(".local/stage16_reference_tracking_ppo/objects/hocap_170650.obj"),
    ]
    if len(references) != len(meshes) or len(references) != 2:
        raise ValueError("Stage16.1a requires exactly two reference/mesh pairs")
    references = [path.resolve() for path in references]
    meshes = [path.resolve() for path in meshes]
    if any(not path.is_file() for path in [*references, *meshes]):
        raise FileNotFoundError("missing reference or mesh")
    report_root = args.report_root.resolve()
    experiment_root = args.experiment_root.resolve()
    if (report_root / "final_summary.json").exists():
        raise FileExistsError("refusing to overwrite an existing dynamic-coupling result")
    report_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    machine = Stage161DynamicCouplingStateMachine()
    backends = [
        _make_backend(reference, mesh, experiment_root / "scenes", seed=args.seed + index)
        for index, (reference, mesh) in enumerate(zip(references, meshes, strict=True))
    ]
    clips = [backend.reference for backend in backends]
    _write_json(
        report_root / "reference_dynamics.json",
        {
            path.stem: {key: value.tolist() for key, value in reference_acceleration(clip).items()}
            for path, clip in zip(references, clips, strict=True)
        },
    )
    reference_rows: list[dict[str, object]] = []
    for path, clip in zip(references, clips, strict=True):
        dynamics = reference_acceleration(clip)
        for frame in range(clip.frame_count):
            reference_rows.append(
                {
                    "clip": path.stem,
                    "frame": frame,
                    "timestamp_s": float(clip.timestamps[frame]),
                    "qdot": dynamics["qdot"][frame],
                    "qddot": dynamics["qddot"][frame],
                    "object_velocity": dynamics["object_velocity"][frame],
                    "object_acceleration": dynamics["object_acceleration"][frame],
                }
            )
    _write_csv(report_root / "reference_dynamics.csv", reference_rows)
    _write_json(
        report_root / "reference_acceleration.json",
        {
            path.stem: {
                "max_object_linear_acceleration_mps2": float(
                    np.linalg.norm(
                        reference_acceleration(clip)["object_acceleration"][:, :3], axis=1
                    ).max()
                ),
                "max_object_angular_acceleration_radps2": float(
                    np.linalg.norm(
                        reference_acceleration(clip)["object_acceleration"][:, 3:], axis=1
                    ).max()
                ),
            }
            for path, clip in zip(references, clips, strict=True)
        },
    )

    # Step A: isolate PD/actuators with a reference-driven object.
    step_a_rollouts = [
        _rollout(
            backend,
            _zero_policy,
            velocity_profile=ResetVelocityProfile.ZERO,
            kinematic_object=True,
            label=f"{reference.stem}_step_a_dynamic_hand_kinematic_object",
        )
        for backend, reference in zip(backends, references, strict=True)
    ]
    step_a_rows = [_summary_row(rollout) for rollout in step_a_rollouts]
    for row in step_a_rows:
        row["result"] = (
            "STEP_A_PD_PASS"
            if float(row["final_joint_rmse_rad"]) <= 0.02
            and float(row["final_link_rmse_mm"]) <= 5.0
            and int(row["frames_completed"]) == 40
            else "STEP_A_PD_GAIN_FAILURE"
        )
    step_a_pass = all(row["result"] == "STEP_A_PD_PASS" for row in step_a_rows)
    machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_A_PD,
        failure_class=(
            FailureClass.OBJECT_DYNAMICS_FAILURE
            if step_a_pass
            else FailureClass.ACTUATOR_OR_PD_FAILURE
        ),
        evidence={"rows": step_a_rows, "kinematic_object": True},
        repair="none_pd_excluded" if step_a_pass else "preserve_global_pd_profile_for_followup",
        rerun_scope="both_clips_41_frames",
        result="STEP_A_PD_PASS" if step_a_pass else "STEP_A_PD_FAILURE",
    )
    _write_json(
        report_root / "step_a_pd_qualification.json",
        {
            "status": "PASS" if step_a_pass else "FAIL",
            "rows": step_a_rows,
            "rollouts": step_a_rollouts,
        },
    )

    # Step B: static contact, expected proximity, and generic preload probes.
    contact_profiles: dict[str, list[dict[str, object]]] = {}
    for backend, reference in zip(backends, references, strict=True):
        contact_profiles[reference.stem] = [
            _contact_audit(backend, preload_fraction=fraction) for fraction in PRELOAD_CANDIDATES
        ]
    selected_preload = 0.0
    baseline_scores: list[float] = []
    candidate_scores: dict[float, list[float]] = {fraction: [] for fraction in PRELOAD_CANDIDATES}
    for reference in references:
        profiles = contact_profiles[reference.stem]
        for profile in profiles:
            score = float(profile["summary"]["contacts_total"]) + 10.0 * float(
                profile["summary"]["max_push_displacement_m"]
            )
            candidate_scores[float(profile["preload_fraction"])].append(score)
        baseline_scores.append(candidate_scores[0.0][-1])
    jointly_improving = [
        fraction
        for fraction in PRELOAD_CANDIDATES[1:]
        if all(
            new > old for new, old in zip(candidate_scores[fraction], baseline_scores, strict=True)
        )
    ]
    if jointly_improving:
        selected_preload = max(jointly_improving, key=lambda value: min(candidate_scores[value]))
    selected_contact = {
        reference.stem: next(
            profile
            for profile in contact_profiles[reference.stem]
            if float(profile["preload_fraction"]) == selected_preload
        )
        for reference in references
    }
    early_contact_pass = all(
        any(
            int(row["initial"]["hand_object_contact_count"]) > 0
            for row in profile["frames"]
            if int(row["frame"]) <= 10
        )
        for profile in selected_contact.values()
    )
    contact_pass = early_contact_pass and all(
        int(profile["summary"]["contacts_total"]) > 0
        and float(profile["summary"]["max_push_displacement_m"]) > 1e-5
        for profile in selected_contact.values()
    )
    machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_B_CONTACT,
        failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
        evidence={
            "selected_preload_fraction": selected_preload,
            "contact_pass": contact_pass,
            "contact_profiles": contact_profiles,
        },
        repair=(
            "freeze_grasp_preload_profile_v1"
            if selected_preload
            else "no_generic_preload_improvement"
        ),
        rerun_scope="both_clips_static_audit_and_push_probe",
        result="CONTACT_COUPLING_PASS" if contact_pass else "CONTACT_FORCE_CLOSURE_INSUFFICIENT",
    )
    _write_json(
        report_root / "step_b_contact_audit.json",
        {
            "status": (
                "CONTACT_COUPLING_PASS" if contact_pass else "CONTACT_FORCE_CLOSURE_INSUFFICIENT"
            ),
            "preload_direction": "positive normalized Wuji actuator-range flexion",
            "profiles": contact_profiles,
            "selected_profile": selected_preload,
            "pre_gate_contact_required": True,
            "pre_gate_contact_pass": early_contact_pass,
        },
    )
    _write_json(
        report_root / "step_b_contact_repairs.json",
        {
            "grasp_preload_profile_v1": {
                "fraction": selected_preload,
                "engineering_assumption": True,
                "shared_between_clips": True,
                "frozen_only_if_jointly_improving": bool(selected_preload),
            },
            "physical_collision_parameters_changed": False,
        },
    )

    # Step C: execute all four profiles even if Step B is not gate-passing;
    # they remain diagnosis-only evidence until contact coupling passes.
    velocity_rollouts: list[dict[str, object]] = []
    for profile in ResetVelocityProfile:
        for backend, reference in zip(backends, references, strict=True):
            velocity_rollouts.append(
                _rollout(
                    backend,
                    _zero_policy,
                    velocity_profile=profile,
                    kinematic_object=False,
                    preload_fraction=selected_preload,
                    label=f"{reference.stem}_step_c_{profile.value}",
                )
            )
    velocity_rows = [_summary_row(rollout) for rollout in velocity_rollouts]
    profile_min_progress = {
        profile.value: min(
            float(row["progress"])
            for row in velocity_rows
            if str(row["label"]).endswith(profile.value)
        )
        for profile in ResetVelocityProfile
    }

    def velocity_key(profile: str) -> tuple[float, float, float, float]:
        rows = [row for row in velocity_rows if str(row["label"]).endswith(profile)]
        return (
            min(float(row["progress"]) for row in rows),
            min(float(row["first_failure_frame"] or 40) for row in rows),
            -max(float(row["final_position_cm"]) for row in rows),
            -max(float(row["final_axis_cm"]) for row in rows),
        )

    selected_velocity = max(profile_min_progress, key=velocity_key)
    machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_C_VELOCITY_RESET,
        failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
        evidence={"rows": velocity_rows, "profile_min_progress": profile_min_progress},
        repair="freeze_reference_velocity_reset_v1",
        rerun_scope="both_clips_C0_C3_full_diagnostic",
        result=(
            f"{selected_velocity.upper()}_SELECTED_DIAGNOSTIC_ONLY"
            if contact_pass
            else "VELOCITY_MATRIX_COMPLETE_CONTACT_GATE_STILL_BLOCKED"
        ),
    )
    _write_json(
        report_root / "step_c_velocity_matrix.json",
        {
            "rows": velocity_rows,
            "profiles": profile_min_progress,
            "contact_gate_pass": contact_pass,
        },
    )
    _write_json(
        report_root / "step_c_selected_profile.json",
        {
            "reference_velocity_reset_v1": selected_velocity,
            "status": (
                "FULL_REFERENCE_VELOCITY_SELECTED"
                if selected_velocity == ResetVelocityProfile.FULL_REFERENCE.value
                else "ZERO_VELOCITY_SELECTED"
                if selected_velocity == ResetVelocityProfile.ZERO.value
                else "OBJECT_REFERENCE_VELOCITY_SELECTED"
                if selected_velocity == ResetVelocityProfile.OBJECT_REFERENCE.value
                else "HAND_REFERENCE_VELOCITY_SELECTED"
            ),
            "selection_rule": "maximize worst-clip progress, then first failure, then error",
            "engineering_assumption": True,
        },
    )

    # Step D: audit old joint-only oracle, then run object-aware finite differences.
    old_oracle_audit = {
        "existing_oracle": "OracleResidualController",
        "uses_joint_error": True,
        "uses_link_error": False,
        "uses_object_position_error": False,
        "uses_object_orientation_or_axis_error": False,
        "uses_object_velocity_error": False,
        "classification": "PREVIOUS_ORACLE_NOT_OBJECT_AWARE",
    }
    _write_json(report_root / "existing_oracle_audit.json", old_oracle_audit)
    oracle_rollouts: list[dict[str, object]] = []
    for backend, reference in zip(backends, references, strict=True):
        oracle = ObjectAwareResidualOracle()
        oracle_rollouts.append(
            _rollout(
                backend,
                oracle,
                velocity_profile=ResetVelocityProfile(selected_velocity),
                kinematic_object=False,
                preload_fraction=selected_preload,
                label=f"{reference.stem}_step_d_object_oracle",
            )
        )
    oracle_rows = [_summary_row(rollout) for rollout in oracle_rollouts]
    oracle_pass = all(row["first_failure_frame"] is None for row in oracle_rows)
    machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_D_OBJECT_ORACLE,
        failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
        evidence={"rows": oracle_rows, "contact_gate_pass": contact_pass},
        repair="object_aware_central_difference_ridge_oracle_v1",
        rerun_scope="D0_D1_both_clips_diagnostic",
        result="OBJECT_AWARE_ORACLE_PASS" if oracle_pass else "CONTACT_BREAKS_UNDER_CONTROL",
    )
    _write_json(
        report_root / "object_oracle_evaluation.json",
        {
            "stage": "D1 only; D2/D3 require prior stage pass",
            "same_20d_action_bounds": True,
            "direct_object_control": False,
            "rows": oracle_rows,
            "rollouts": oracle_rollouts,
        },
    )

    # Step E: local rank and fixed-budget shooting on the actual state.  It is
    # run as a diagnosis after failed D, never as a substitute PPO result.
    sensitivity: dict[str, list[dict[str, object]]] = {}
    shooting: dict[str, list[dict[str, object]]] = {}
    for backend, reference in zip(backends, references, strict=True):
        rows: list[dict[str, object]] = []
        shots: list[dict[str, object]] = []
        failure_frames = [
            int(row["first_failure_frame"])
            for row in velocity_rows
            if str(row["label"]).endswith(ResetVelocityProfile.ZERO.value)
            and row["first_failure_frame"] is not None
        ]
        feasibility_frames = sorted(
            {0, 5, 10, 20, 30, *failure_frames, *(value - 1 for value in failure_frames)}
        )
        for frame in feasibility_frames:
            backend.reset(
                reference_index=frame,
                velocity_profile=ResetVelocityProfile(selected_velocity).value,
            )
            oracle = ObjectAwareResidualOracle()
            action = oracle.action(backend)
            if oracle.last_diagnostics is None:
                raise RuntimeError("object-aware oracle did not emit diagnostics")
            rows.append(
                {"frame": frame, "action": action.tolist(), **oracle.last_diagnostics.json()}
            )
            shooter = ObjectAwareShootingOracle()
            for horizon in shooter.horizons:
                shots.append({"frame": frame, **shooter.diagnose(backend, action, horizon).json()})
        sensitivity[reference.stem] = rows
        shooting[reference.stem] = shots
    rank_positive = all(any(int(row["rank"]) > 0 for row in rows) for rows in sensitivity.values())
    local_descent = any(
        row["classification"] == "LOCAL_FEASIBILITY_PASS"
        for rows in shooting.values()
        for row in rows
    )
    root_cause = (
        "REFERENCE_DYNAMICAL_INFEASIBILITY"
        if not early_contact_pass
        else "ACTION_TO_OBJECT_RANK_DEFICIENT"
        if not rank_positive
        else "REFERENCE_DYNAMICAL_INFEASIBILITY"
        if not local_descent
        else "ROOT_CAUSE_UNRESOLVED"
    )
    machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_E_DYNAMIC_FEASIBILITY,
        failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
        evidence={
            "rank_positive": rank_positive,
            "local_descent": local_descent,
            "root_cause": root_cause,
        },
        repair="no further unbounded tuning",
        rerun_scope="fixed_frames_H5_H10",
        result=root_cause,
    )
    _write_json(report_root / "object_oracle_sensitivity.json", sensitivity)
    _write_json(report_root / "shooting_oracle_evaluation.json", shooting)
    _write_json(
        report_root / "controllability_analysis.json",
        {
            "rank_positive": rank_positive,
            "local_descent": local_descent,
            "sensitivity": sensitivity,
        },
    )
    _write_json(
        report_root / "local_feasibility.json",
        {"shooting": shooting, "fixed_horizons": [5, 10], "action_dimension": 20},
    )

    all_rollouts = [*step_a_rollouts, *velocity_rollouts, *oracle_rollouts]
    frame_rows = [row for rollout in all_rollouts for row in rollout["frames"]]
    physics_rows = [row for rollout in all_rollouts for row in rollout["physics"]]
    with (report_root / "diagnostic_rollout.jsonl").open("w", encoding="utf-8") as handle:
        for row in physics_rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    _write_json(report_root / "diagnostic_rollout.json", {"rollouts": all_rollouts})
    _write_csv(report_root / "frame_summary.csv", frame_rows)
    _write_csv(
        report_root / "contact_trace.csv",
        [
            {
                "label": row["label"],
                "control_step": row["control_step"],
                "reference_index": row["reference_index"],
                "contact": row["contact"],
            }
            for row in physics_rows
        ],
    )
    _write_csv(
        report_root / "object_motion.csv",
        [
            {
                "label": row["label"],
                "control_step": row["control_step"],
                "reference_index": row["reference_index"],
                "object_qpos": row["object_qpos"],
                "object_qvel": row["object_qvel"],
                "object_kinetic_energy": row["object_kinetic_energy"],
            }
            for row in physics_rows
        ],
    )
    _write_csv(
        report_root / "hand_tracking.csv",
        [
            {
                "label": row["label"],
                "control_step": row["control_step"],
                "reference_index": row["reference_index"],
                "q": row["q"],
                "qdot": row["qdot"],
                "ctrl": row["ctrl"],
                "actuator_force": row["actuator_force"],
                "joint_limit_margin": row["joint_limit_margin"],
            }
            for row in physics_rows
        ],
    )
    termination_trace = {rollout["label"]: rollout["first_termination"] for rollout in all_rollouts}
    _write_json(report_root / "termination_trace.json", termination_trace)
    visual_paths = _plot(report_root / "visual", all_rollouts)
    contact_sheet = _contact_sheet(visual_paths, report_root / "visual" / "contact_sheet.png")
    dashboard = _write_dashboard(
        report_root / "dashboard.html",
        {
            "stage": "Stage16.1a dynamic-coupling diagnostic",
            "rollout_count": len(all_rollouts),
            "formal_termination_preserved": True,
            "diagnostic_continue_after_termination": True,
        },
        visual_paths,
    )
    visual_review = {
        "status": "NUMERICAL_CONTACT_SHEETS_GENERATED",
        "renderer": "numerical plot fallback",
        "paths": visual_paths,
        "contact_sheet": contact_sheet,
        "dashboard": dashboard,
        "checked": False,
        "non_claim": "numerical diagnostic plots are not an interactive geometry acceptance",
    }
    _write_json(report_root / "visual_review.json", visual_review)

    final_status = (
        "STAGE16_1_CONTROLLABILITY_COMPLETE"
        if step_a_pass and contact_pass and oracle_pass
        else "STAGE16_1_CONTROLLABILITY_BLOCKED"
    )
    stage16_2_entry = (
        "STAGE16_2_ENTRY_AUTHORIZED"
        if final_status.endswith("COMPLETE")
        else "STAGE16_2_ENTRY_NOT_AUTHORIZED"
    )
    machine.record_dynamic(
        phase=DynamicCouplingPhase.FINAL_REQUALIFICATION,
        failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
        evidence={
            "step_a_pass": step_a_pass,
            "contact_pass": contact_pass,
            "oracle_pass": oracle_pass,
        },
        repair="none",
        rerun_scope="formal_gate_decision_only",
        result=final_status,
    )
    transitions = report_root / "failure_transition_log.jsonl"
    machine.write_jsonl(transitions)
    _write_json(report_root / "recovery_summary.json", machine.dynamic_summary())
    _write_json(
        report_root / "root_cause.json",
        {
            "primary": root_cause,
            "evidence": {
                "step_a_pass": step_a_pass,
                "contact_pass": contact_pass,
                "selected_preload_fraction": selected_preload,
                "velocity_profile": selected_velocity,
                "object_oracle_pass": oracle_pass,
                "rank_positive": rank_positive,
                "local_descent": local_descent,
            },
        },
    )
    final_summary = {
        "status": final_status,
        "stage16_2_entry": stage16_2_entry,
        "root_cause": root_cause,
        "step_a": step_a_rows,
        "step_b_contact_pass": contact_pass,
        "step_c_velocity_profile": selected_velocity,
        "step_d": oracle_rows,
        "visual_review": visual_review,
        "reports": str(report_root),
        "ppo_started": False,
    }
    _write_json(report_root / "final_summary.json", final_summary)
    _write_json(report_root / "tests.json", {"status": "NOT_RUN_BY_DIAGNOSTIC_SCRIPT"})
    print(json.dumps(final_summary, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
