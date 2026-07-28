from __future__ import annotations

import numpy as np

from toporetarget.retarget.final_refinement import FinalRetargetTrajectory
from toporetarget.workflows.grab_suite import load_suite


def test_final_artifact_accepts_generic_20d_robot_and_672_surface() -> None:
    frame_count = 1
    dof_count = 20
    surface_count = 672
    arrays = {
        "timestamps": np.zeros(frame_count),
        "qpos": np.zeros((frame_count, dof_count)),
        "base_pose_scene": np.tile(np.eye(4), (frame_count, 1, 1)),
        "base_corrections": np.zeros((frame_count, 6)),
        "robot_keypoints_base": np.zeros((frame_count, 21, 3)),
        "robot_keypoints_scene": np.zeros((frame_count, 21, 3)),
        "collision_points_scene": np.zeros((frame_count, surface_count, 3)),
        "slack_concat": np.zeros(surface_count),
        "query_offsets": np.asarray([0, surface_count]),
        "full_signed_distance": np.ones((frame_count, surface_count)),
        "full_closest_points": np.zeros((frame_count, surface_count, 3)),
        "full_surface_normals": np.zeros((frame_count, surface_count, 3)),
        "full_hard_residual": np.ones((frame_count, surface_count)),
        "full_soft_violation_count": np.zeros(frame_count, dtype=np.int64),
        "unqueried_soft_violation_count": np.zeros(frame_count, dtype=np.int64),
        "active_set_converged": np.ones(frame_count, dtype=bool),
        "robot_link_poses": np.zeros((frame_count, 26, 4, 4)),
        "valid_mask": np.ones(frame_count, dtype=bool),
    }
    artifact = FinalRetargetTrajectory(
        {
            "schema_version": "toporetarget.final_retarget.v2",
            "robot_dof_count": dof_count,
            "collision_surface_sample_count": surface_count,
        },
        arrays,
    )
    assert artifact.validate() is artifact
    assert artifact.arrays["qpos"].shape == (1, 20)
    assert artifact.arrays["collision_points_scene"].shape == (1, 672, 3)


def test_wuji_suite_freezes_three_native_windows() -> None:
    config, clips = load_suite("configs/experiments/wuji_hand2_grab3_v1.yaml")
    assert config["robot"] == "wuji_hand2_beta1_rh"
    assert [(clip.short_id, clip.start_frame, clip.end_frame) for clip in clips] == [
        ("W1_airplane_lift", 240, 300),
        ("W2_apple_eat_1", 212, 272),
        ("W3_alarmclock_lift", 407, 467),
    ]
    assert all(clip.length == 60 for clip in clips)
