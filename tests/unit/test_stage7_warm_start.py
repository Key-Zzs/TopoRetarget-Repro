from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from toporetarget.retarget.alignment import finite_difference_jacobian_check
from toporetarget.retarget.artifacts import WarmStartTrajectory, load_warm_start, save_warm_start
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.objectives import BoneDirectionObjective, BoneDirectionResidual
from toporetarget.retarget.solver import load_paper_weights


def _points() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    for start, y in zip((1, 5, 9, 13, 17), (0.01, 0.03, 0.05, 0.07, 0.09), strict=True):
        points[start : start + 4] = np.array(
            [[0.03, y, 0.0], [0.06, y, 0.0], [0.09, y, 0.0], [0.12, y, 0.0]]
        )
    return points


class _ToyModel:
    neutral_q = np.zeros(3)
    joint_lower = np.full(3, -1.0)
    joint_upper = np.full(3, 1.0)

    def keypoints_base(self, qpos: torch.Tensor, layout: str = "mediapipe21") -> torch.Tensor:
        points = qpos.new_tensor(_points())
        basis = qpos.new_zeros((21, 3))
        basis[5, 0] = 1.0
        points = points + qpos[0] * basis
        basis = qpos.new_zeros((21, 3))
        basis[9, 0] = 1.0
        points = points + qpos[1] * basis
        basis = qpos.new_zeros((21, 3))
        basis[13, 0] = 1.0
        return points + qpos[2] * basis

    def keypoints_scene(
        self, qpos: torch.Tensor, base_pose: torch.Tensor, layout: str = "mediapipe21"
    ) -> torch.Tensor:
        points = self.keypoints_base(qpos, layout=layout)
        return points @ base_pose[:3, :3].T + base_pose[:3, 3]


def test_paper_weights_are_loaded_from_single_paper_config() -> None:
    warm, smooth, path = load_paper_weights()
    assert (warm, smooth) == (1.0, 2.5)
    assert path.endswith("configs/paper/retarget.yaml")


def test_eq2_residual_scaling_and_objective_factor_of_two() -> None:
    model = _ToyModel()
    frame = load_frame_profile("canonical_keypoint_wrist_v1")
    bones = load_bone_profile("mediapipe21_full_finger_chain_v1")
    source = extract_bone_features(_points(), frame, bones).adjacent_features
    residual_model = BoneDirectionResidual(source, frame, bones, model, "right")
    previous = np.array([0.1, 0.2, 0.0])
    objective = BoneDirectionObjective(residual_model, 1.0, 2.5, previous)
    q = torch.tensor([0.2, 0.1, 0.0], dtype=torch.float64)
    values = objective.paper_objective(q)
    expected = float(values["ebone"] + values["temporal"])
    assert float(values["total"]) == pytest.approx(expected)
    assert float(values["library_cost_half"]) == pytest.approx(expected / 2.0)
    assert objective.residual_tensor(q).shape[0] == 15 * 3 + 3


def test_base_is_not_in_local_direction_objective() -> None:
    model = _ToyModel()
    frame = load_frame_profile("canonical_keypoint_wrist_v1")
    bones = load_bone_profile("mediapipe21_full_finger_chain_v1")
    q = torch.tensor([0.1, -0.1, 0.02], dtype=torch.float64)
    points = model.keypoints_base(q).detach().cpu().numpy()
    source = extract_bone_features(points, frame, bones).adjacent_features
    objective = BoneDirectionResidual(source, frame, bones, model, "right")
    base = torch.eye(4, dtype=torch.float64)
    base[0, 3] = 3.0
    assert torch.allclose(
        objective.residual_tensor(q), objective.residual_tensor(q, base_pose=base)
    )


def test_torch_jacobian_matches_central_finite_difference() -> None:
    model = _ToyModel()
    frame = load_frame_profile("canonical_keypoint_wrist_v1")
    bones = load_bone_profile("mediapipe21_full_finger_chain_v1")
    source = extract_bone_features(_points(), frame, bones).adjacent_features
    residual_model = BoneDirectionResidual(source, frame, bones, model, "right")
    report = finite_difference_jacobian_check(
        residual_model, np.array([0.1, -0.1, 0.02]), epsilon=1e-6
    )
    assert report["max_abs_difference"] < 1e-7
    assert report["relative_frobenius_error"] < 1e-7


def test_warm_start_artifact_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    arrays = {
        "qpos": np.zeros((1, 22)),
        "base_pose_scene": np.eye(4)[None],
        "robot_keypoints_base": np.zeros((1, 21, 3)),
        "robot_keypoints_scene": np.zeros((1, 21, 3)),
        "source_hand_frame_scene": np.eye(4)[None],
        "robot_hand_frame_base": np.eye(4)[None],
        "source_bone_directions": np.zeros((1, 20, 3)),
        "robot_bone_directions": np.zeros((1, 20, 3)),
        "source_adjacent_features": np.zeros((1, 15, 3)),
        "robot_adjacent_features": np.zeros((1, 15, 3)),
        "pair_residuals": np.zeros((1, 15, 3)),
        "ebone": np.zeros(1),
        "temporal_term": np.zeros(1),
        "total_objective": np.zeros(1),
        "valid_mask": np.ones(1, dtype=bool),
    }
    artifact = WarmStartTrajectory({"schema_version": "toporetarget.warm_start.v1"}, arrays)
    path = tmp_path / "artifact.zarr"
    save_warm_start(artifact, path)
    loaded = load_warm_start(path)
    assert loaded.frame_count == 1
    with pytest.raises(Exception, match="exists"):
        save_warm_start(artifact, path)
