from __future__ import annotations

import numpy as np

from toporetarget.rl.grasp_lift_skill_collapse import (
    grasp_lift_episode_metrics,
    lift_milestones,
)


def _trace() -> dict[str, np.ndarray]:
    count = 12
    actual = np.zeros((count, 5), dtype=bool)
    actual[4:9, :2] = True
    presence = np.zeros((count, 21), dtype=bool)
    presence[4:9, :2] = True
    force = np.zeros((count, 5, 3))
    force[4:9, :2, 0] = 1.0
    phase = np.asarray(["PRE_CONTACT"] * 3 + ["GRASP"] * 3 + ["LIFT"] * 6)
    pose = np.zeros((count, 7))
    pose[9:, 2] = 0.06
    return {
        "hand_object_pair_force_valid": np.asarray([False] + [True] * 11),
        "hand_object_pair_presence": presence,
        "actual_contact_mask": actual,
        "fingertip_object_pair_force_world": force,
        "reference_contact_mask": actual,
        "contact_reward": np.zeros(count),
        "phase": phase,
        "object_pose": pose,
    }


def test_grasp_metric_requires_persistent_multifinger_contact() -> None:
    metric = grasp_lift_episode_metrics(_trace())
    assert metric["persistent_grasp"]
    assert metric["grasp_and_lift"]
    assert metric["persistent_grasp_at_semantic_lift"]
    assert metric["category"] == "GRASP_AND_LIFT"


def test_lift_milestones_keep_persistent_zero_unidentified_when_unobserved() -> None:
    rows = [
        {"update": 1, "samples": 1, "checkpoint_sha256": "one", "lift_episode_rate": 1.0},
        {"update": 2, "samples": 2, "checkpoint_sha256": "two", "lift_episode_rate": 0.0},
    ]
    milestones = lift_milestones(rows, 1.0)
    assert milestones["U_LAST_LIFT_STABLE"]["update"] == 1
    assert milestones["U_ZERO_LIFT"]["update"] == 2
    assert milestones["U_PERSISTENT_ZERO_LIFT"] is None
