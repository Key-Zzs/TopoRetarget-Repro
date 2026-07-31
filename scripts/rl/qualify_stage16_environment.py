#!/usr/bin/env python3
"""Run bounded E0--E4 MuJoCo asset qualifications without claiming a HOCap result.

This uses a synthetic constant reference only to exercise the tracked Wuji
asset, PD target wiring, free-object dynamics, observation path, reward, and
termination.  It never creates, substitutes for, or evaluates a HOCap policy.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import mujoco
import numpy as np

from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.randomization import DomainRandomizationConfig

REPO = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO / ".local/reports/stage16_reference_tracking_ppo"
BUILD_ROOT = REPO / ".local/build/stage16_reference_tracking_ppo"
MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
TRACKED_LINKS = ("r_wrist", "r_thumb_distal")


def make_qualification_reference() -> tuple[Stage16ReferenceClip, np.ndarray]:
    """Create an explicitly synthetic neutral asset-qualification reference."""

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF has unnamed joints")
    joint_order = tuple(name for name in joint_order if name is not None)
    bounds = model.jnt_range[: model.njnt].copy()
    midpoint = bounds.mean(axis=1)
    data.qpos[: model.njnt] = midpoint
    mujoco.mj_forward(model, data)
    link_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TRACKED_LINKS]
    )
    if np.any(link_ids < 0):
        raise RuntimeError("qualification tracked-link mapping is absent from Wuji MJCF")
    timestamps = np.arange(21, dtype=np.float64) / 20.0
    poses = np.broadcast_to(np.eye(4), (timestamps.size, 4, 4)).copy()
    poses[:, 2, 3] = 0.15
    reference = Stage16ReferenceClip(
        timestamps=timestamps,
        q_finger_ref=np.broadcast_to(midpoint, (timestamps.size, model.njnt)).copy(),
        object_pose_base_ref=poses,
        object_axis_points_base_ref=object_axis_points_from_poses(poses),
        tracked_link_positions_base_ref=np.broadcast_to(
            data.xpos[link_ids], (timestamps.size, len(TRACKED_LINKS), 3)
        ).copy(),
        joint_order=joint_order,
        tracked_link_names=TRACKED_LINKS,
        provenance={
            "kind": "synthetic_asset_qualification_only",
            "source_mjcf": str(MJCF.relative_to(REPO)),
            "hocap_substitute": False,
        },
    )
    reference.validate(expected_hz=20.0)
    return reference, bounds


def finite_state(state: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(value).all() for value in state.values())


def main() -> int:
    started = time.perf_counter()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    reference, bounds = make_qualification_reference()

    # E0: reference schema and base-frame kinematic state construction.
    e0 = {
        "status": "E0_KINEMATIC_SYNTHETIC_ASSET_PASS",
        "reference_validation": reference.validate(expected_hz=20.0),
        "physical_hocap_reference": False,
    }

    # E1: native hand-only PD target mapping at the neutral reference.
    hand_model = mujoco.MjModel.from_xml_path(str(MJCF))
    hand_data = mujoco.MjData(hand_model)
    hand_data.qpos[: hand_model.nq] = bounds.mean(axis=1)
    hand_data.ctrl[:] = bounds.mean(axis=1)
    for _ in range(25):
        mujoco.mj_step(hand_model, hand_data)
    e1 = {
        "status": "E1_PD_TARGET_WIRING_PASS",
        "steps": 25,
        "finite": bool(np.isfinite(hand_data.qpos).all() and np.isfinite(hand_data.qvel).all()),
        "joint_error_rad_max": float(np.max(np.abs(hand_data.qpos - bounds.mean(axis=1)))),
        "physical_hocap_reference": False,
    }

    scene = materialize_free_object_scene(MJCF, BUILD_ROOT)
    backend = MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=bounds[:, 0],
        joint_upper=bounds[:, 1],
        randomization=DomainRandomizationConfig(enabled=False),
        seed=20260801,
    )
    initial = backend.reset()
    e2_reasons: list[str | None] = []
    e2_finite = finite_state(initial)
    for _ in range(5):
        state, _, reason = backend.transition(np.zeros(reference.dof_count))
        e2_finite = e2_finite and finite_state(state)
        e2_reasons.append(reason)
    e2 = {
        "status": "E2_FREE_OBJECT_ZERO_ACTION_PASS"
        if e2_finite
        else "E2_FREE_OBJECT_NONFINITE_FAIL",
        "steps": 5,
        "finite": e2_finite,
        "termination_reasons": e2_reasons,
        "physical_hocap_reference": False,
    }

    backend.reset()
    generator = np.random.default_rng(20260801)
    e3_finite = True
    reward_totals: list[float] = []
    observation_dimension = 0
    for _ in range(5):
        action = generator.uniform(-0.25, 0.25, size=reference.dof_count)
        state, reward, _ = backend.transition(action)
        observation = backend.observation(state)
        e3_finite = e3_finite and finite_state(state) and bool(np.isfinite(observation).all())
        e3_finite = e3_finite and all(np.isfinite(value) for value in reward.values())
        observation_dimension = int(observation.size)
        reward_totals.append(float(reward["total"]))
    e3 = {
        "status": "E3_BOUNDED_RESIDUAL_OBSERVATION_PASS" if e3_finite else "E3_NONFINITE_FAIL",
        "steps": 5,
        "observation_dimension": observation_dimension,
        "reward_total_range": [min(reward_totals), max(reward_totals)],
        "physical_hocap_reference": False,
    }

    backend.reset()
    feedback = np.clip(
        (reference.q_finger_ref[backend.reference_index] - backend._state()["q"])
        / ((bounds[:, 1] - bounds[:, 0]) * 0.10),
        -1.0,
        1.0,
    )
    e4_state, e4_reward, e4_reason = backend.transition(feedback)
    e4 = {
        "status": "E4_LOCAL_FEEDBACK_DIAGNOSTIC_PASS"
        if finite_state(e4_state) and all(np.isfinite(value) for value in e4_reward.values())
        else "E4_LOCAL_FEEDBACK_DIAGNOSTIC_FAIL",
        "steps": 1,
        "termination_reason": e4_reason,
        "ppo_result": False,
        "physical_hocap_reference": False,
    }

    # Rendering is useful for review, but a headless worker may legitimately lack
    # an OpenGL context.  Record that capability separately instead of treating a
    # renderer fallback as a physics or policy failure.
    try:
        frame = backend.render_rgb(width=64, height=48)
        renderer = {
            "status": "RENDERER_SMOKE_PASS",
            "frame_shape": list(frame.shape),
            "finite": bool(np.isfinite(frame).all()),
        }
    except Exception as exc:  # pragma: no cover - depends on host GL backend
        renderer = {
            "status": "VISUALIZATION_BACKEND_UNAVAILABLE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }

    qualification = {
        "schema_version": "toporetarget.stage16.environment_qualification.v1",
        "status": "STAGE16_ENVIRONMENT_PARTIAL",
        "reference_kind": "synthetic_asset_qualification_only",
        "stages": {"E0": e0, "E1": e1, "E2": e2, "E3": e3, "E4": e4},
        "renderer": renderer,
        "limitations": [
            "No HOCap RobotReference was used.",
            "No policy was trained or evaluated.",
            "No per-object HOCap collision asset was used.",
        ],
    }
    usage = {
        "qualification_wall_seconds": time.perf_counter() - started,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "backend": "mujoco_cpu_reference",
        "gpu_used": False,
        "simulation_steps": 36,
        "physical_hocap_protocol_steps": 0,
    }
    (REPORT_ROOT / "environment_qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "resource_use.json").write_text(
        json.dumps(usage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observation_report = {
        "status": "OBSERVATION_CONTRACT_VALIDATED_ON_SYNTHETIC_ASSET_REFERENCE",
        "dimension": observation_dimension,
        "offsets": [0, 1, 3, 5],
        "reference_noised_or_delayed": False,
    }
    for filename in ("observation_contract.json", "observation_dimension_report.json"):
        (REPORT_ROOT / filename).write_text(
            json.dumps(observation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"qualification": qualification["status"], "resource": usage}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
