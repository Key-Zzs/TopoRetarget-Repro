"""Stage 8 evaluation of Eq. (7) on a Stage 7 warm-start trajectory."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.artifacts import WarmStartTrajectory

from .graph_weights import DirectedGraphWeights
from .interaction_graph import (
    GRAPH_VERTEX_COUNT,
    INTERACTION_GRAPH_SCHEMA_VERSION,
    InteractionGraphTrajectory,
)
from .interaction_objective import InteractionMeshObjective, InteractionMeshResidual

INTERACTION_EVALUATION_SCHEMA_VERSION = "toporetarget.interaction_evaluation.v1"


class InteractionEvaluationError(RuntimeError):
    """Raised when a graph and warm-start cannot be evaluated together."""


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


@dataclass
class InteractionEvaluationTrajectory:
    metadata: dict[str, Any]
    timestamps: np.ndarray
    qpos: np.ndarray
    base_pose_scene: np.ndarray
    robot_keypoints_scene: np.ndarray
    robot_vertices: np.ndarray
    robot_laplacian: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    e_im: np.ndarray
    per_vertex_contribution: np.ndarray
    per_hand_point_contribution: np.ndarray
    per_object_point_contribution: np.ndarray
    max_residual_vertex: np.ndarray
    qpos_jacobian: np.ndarray
    base_translation_sensitivity: np.ndarray
    base_rotation_sensitivity: np.ndarray
    frame_valid: np.ndarray
    frame_status: list[str]
    source_path: Path | None = None
    artifact_hash: str | None = None

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def schema_version(self) -> str:
        return str(self.metadata.get("schema_version", ""))

    def validate(self) -> InteractionEvaluationTrajectory:
        t = self.frame_count
        if self.schema_version != INTERACTION_EVALUATION_SCHEMA_VERSION:
            raise InteractionEvaluationError(
                f"unsupported evaluation schema: {self.schema_version!r}"
            )
        shapes = {
            "timestamps": (t,),
            "qpos": (t, 22),
            "base_pose_scene": (t, 4, 4),
            "robot_keypoints_scene": (t, 21, 3),
            "robot_vertices": (t, GRAPH_VERTEX_COUNT, 3),
            "robot_laplacian": (t, GRAPH_VERTEX_COUNT, 3),
            "residual": (t, GRAPH_VERTEX_COUNT, 3),
            "scaled_residual": (t, GRAPH_VERTEX_COUNT * 3),
            "e_im": (t,),
            "per_vertex_contribution": (t, GRAPH_VERTEX_COUNT),
            "per_hand_point_contribution": (t, 21),
            "per_object_point_contribution": (t, 50),
            "max_residual_vertex": (t,),
            "qpos_jacobian": (t, GRAPH_VERTEX_COUNT * 3, 22),
            "base_translation_sensitivity": (t,),
            "base_rotation_sensitivity": (t,),
            "frame_valid": (t,),
        }
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise InteractionEvaluationError(
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}"
                )
        if len(self.frame_status) != t or not np.all(self.frame_valid):
            raise InteractionEvaluationError("evaluation contains invalid frames")
        if not np.all(np.isfinite(self.e_im)):
            raise InteractionEvaluationError("E_IM contains NaN or Inf")
        return self

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "base_pose_scene": np.asarray(self.base_pose_scene, dtype=np.float64),
            "robot_keypoints_scene": np.asarray(self.robot_keypoints_scene, dtype=np.float64),
            "robot_vertices": np.asarray(self.robot_vertices, dtype=np.float64),
            "robot_laplacian": np.asarray(self.robot_laplacian, dtype=np.float64),
            "residual": np.asarray(self.residual, dtype=np.float64),
            "scaled_residual": np.asarray(self.scaled_residual, dtype=np.float64),
            "e_im": np.asarray(self.e_im, dtype=np.float64),
            "per_vertex_contribution": np.asarray(self.per_vertex_contribution, dtype=np.float64),
            "per_hand_point_contribution": np.asarray(
                self.per_hand_point_contribution, dtype=np.float64
            ),
            "per_object_point_contribution": np.asarray(
                self.per_object_point_contribution, dtype=np.float64
            ),
            "max_residual_vertex": np.asarray(self.max_residual_vertex, dtype=np.int64),
            "qpos_jacobian": np.asarray(self.qpos_jacobian, dtype=np.float64),
            "base_translation_sensitivity": np.asarray(
                self.base_translation_sensitivity, dtype=np.float64
            ),
            "base_rotation_sensitivity": np.asarray(
                self.base_rotation_sensitivity, dtype=np.float64
            ),
            "frame_valid": np.asarray(self.frame_valid, dtype=bool),
        }

    @classmethod
    def from_arrays(
        cls,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
        *,
        source_path: Path | None = None,
        artifact_hash: str | None = None,
    ) -> InteractionEvaluationTrajectory:
        status = list(metadata.get("frame_status", ["valid"] * int(arrays["qpos"].shape[0])))
        result = cls(
            metadata=metadata,
            timestamps=arrays["timestamps"],
            qpos=arrays["qpos"],
            base_pose_scene=arrays["base_pose_scene"],
            robot_keypoints_scene=arrays["robot_keypoints_scene"],
            robot_vertices=arrays["robot_vertices"],
            robot_laplacian=arrays["robot_laplacian"],
            residual=arrays["residual"],
            scaled_residual=arrays["scaled_residual"],
            e_im=arrays["e_im"],
            per_vertex_contribution=arrays["per_vertex_contribution"],
            per_hand_point_contribution=arrays["per_hand_point_contribution"],
            per_object_point_contribution=arrays["per_object_point_contribution"],
            max_residual_vertex=arrays["max_residual_vertex"],
            qpos_jacobian=arrays["qpos_jacobian"],
            base_translation_sensitivity=arrays["base_translation_sensitivity"],
            base_rotation_sensitivity=arrays["base_rotation_sensitivity"],
            frame_valid=arrays["frame_valid"].astype(bool),
            frame_status=status,
            source_path=source_path,
            artifact_hash=artifact_hash,
        )
        return result.validate()


def _rot_z_torch(angle: float, *, dtype: Any, device: Any) -> Any:
    import torch

    c = torch.cos(torch.as_tensor(angle, dtype=dtype, device=device))
    s = torch.sin(torch.as_tensor(angle, dtype=dtype, device=device))
    result = torch.eye(4, dtype=dtype, device=device)
    result[0, 0] = c
    result[0, 1] = -s
    result[1, 0] = s
    result[1, 1] = c
    return result


def _frame_evaluation(
    graph_frame: int,
    graph_vertices: np.ndarray,
    directed: DirectedGraphWeights,
    qpos: np.ndarray,
    base_pose: np.ndarray,
    robot_model: Any,
) -> dict[str, Any]:
    import torch

    q = torch.as_tensor(qpos, dtype=torch.float64)
    base = torch.as_tensor(base_pose, dtype=torch.float64)
    object_points = torch.as_tensor(graph_vertices[21:], dtype=torch.float64)
    residual_model = InteractionMeshResidual(
        graph_vertices, directed.source_index, directed.destination_index, directed.weights
    )
    objective = InteractionMeshObjective(residual_model)

    def robot_vertices_for_q(item: Any, selected_base: Any = base) -> Any:
        hand = robot_model.keypoints_scene(item, selected_base, layout="mediapipe21")
        return torch.cat([hand, object_points], dim=-2)

    robot_vertices_tensor = robot_vertices_for_q(q)
    residual_tensor = objective.residual_tensor(robot_vertices_tensor)
    scaled_tensor = objective.scaled_residual_tensor(robot_vertices_tensor)
    loss_tensor = objective.loss_tensor(robot_vertices_tensor)
    robot_laplacian_tensor = residual_tensor + residual_model.source_laplacian_tensor(
        dtype=torch.float64, device=q.device
    )

    def residual_flat(item: Any) -> Any:
        return objective.residual_tensor(robot_vertices_for_q(item)).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(residual_flat, q, create_graph=False)
    translation_base = base.clone()
    translation_base[:3, 3] = translation_base[:3, 3] + torch.as_tensor(
        [1.0e-3, 0.0, 0.0], dtype=torch.float64
    )
    rotation_base = _rot_z_torch(1.0e-3, dtype=torch.float64, device=q.device) @ base
    translation_loss = objective.loss_tensor(robot_vertices_for_q(q, translation_base))
    rotation_loss = objective.loss_tensor(robot_vertices_for_q(q, rotation_base))
    diagnostics = objective.diagnostics(robot_vertices_tensor)
    return {
        "robot_keypoints_scene": _as_numpy(robot_vertices_tensor[:21]),
        "robot_vertices": _as_numpy(robot_vertices_tensor),
        "robot_laplacian": _as_numpy(robot_laplacian_tensor),
        "residual": _as_numpy(residual_tensor),
        "scaled_residual": _as_numpy(scaled_tensor),
        "e_im": float(loss_tensor.detach().cpu()),
        "per_vertex_contribution": _as_numpy(diagnostics["per_vertex_contribution"]),
        "per_hand_point_contribution": _as_numpy(diagnostics["per_hand_point_contribution"]),
        "per_object_point_contribution": _as_numpy(diagnostics["per_object_point_contribution"]),
        "max_residual_vertex": int(_as_numpy(diagnostics["max_residual_vertex"]).reshape(-1)[0]),
        "qpos_jacobian": _as_numpy(jacobian),
        "base_translation_sensitivity": float((translation_loss - loss_tensor).detach().cpu()),
        "base_rotation_sensitivity": float((rotation_loss - loss_tensor).detach().cpu()),
    }


def evaluate_interaction_graph(
    graph: InteractionGraphTrajectory,
    warm_start: WarmStartTrajectory,
    robot_model: Any,
    *,
    graph_artifact_hash: str | None = None,
    warm_start_artifact_hash: str | None = None,
) -> InteractionEvaluationTrajectory:
    """Evaluate a fixed shared graph; no topology build or optimization occurs."""

    graph.validate()
    warm_start.validate()
    if warm_start.frame_count != graph.frame_count:
        raise InteractionEvaluationError("graph and warm-start frame counts differ")
    if graph.metadata.get("source_cache_hash") != warm_start.metadata.get("source_cache_hash"):
        raise InteractionEvaluationError("graph and warm-start source cache hashes differ")
    if graph.metadata.get("source_hand_id") != warm_start.metadata.get("source_hand_id"):
        raise InteractionEvaluationError("graph and warm-start source hand IDs differ")
    if not np.array_equal(graph.timestamps, warm_start.arrays["timestamps"]):
        raise InteractionEvaluationError("graph and warm-start timestamps differ")
    if warm_start.metadata.get("robot_name") not in {None, robot_model.name}:
        raise InteractionEvaluationError("warm-start robot does not match loaded robot")
    if np.asarray(warm_start.arrays["qpos"]).shape[1] != robot_model.num_dofs:
        raise InteractionEvaluationError("warm-start qpos width does not match robot model")
    timings: list[float] = []
    values: list[dict[str, Any]] = []
    for index in range(graph.frame_count):
        started = time.perf_counter()
        values.append(
            _frame_evaluation(
                int(graph.frame_indices[index]),
                graph.source_vertices[index],
                graph.directed_frames[index],
                np.asarray(warm_start.arrays["qpos"][index], dtype=np.float64),
                np.asarray(warm_start.arrays["base_pose_scene"][index], dtype=np.float64),
                robot_model,
            )
        )
        timings.append(time.perf_counter() - started)
    metadata = {
        "schema_version": INTERACTION_EVALUATION_SCHEMA_VERSION,
        "graph_schema_version": INTERACTION_GRAPH_SCHEMA_VERSION,
        "graph_artifact_hash": graph_artifact_hash,
        "graph_hashes": graph.graph_hashes,
        "warm_start_artifact_hash": warm_start_artifact_hash,
        "source_cache_hash": graph.metadata.get("source_cache_hash"),
        "source_hand_id": graph.metadata.get("source_hand_id"),
        "source_hand_side": graph.metadata.get("source_hand_side"),
        "robot_name": robot_model.name,
        "robot_side": robot_model.side,
        "robot_spec_hash": robot_model.spec_hash,
        "robot_urdf_hash": robot_model.urdf_hash,
        "robot_asset_manifest_hash": robot_model.asset_manifest_hash,
        "robot_anchor_profile_id": robot_model.anchor_profile.profile_id,
        "robot_anchor_profile_hash": robot_model.anchor_profile.sha256,
        "frame_range": graph.metadata.get("frame_range"),
        "timestamps": graph.timestamps.tolist(),
        "vertex_count": GRAPH_VERTEX_COUNT,
        "shared_connectivity": True,
        "shared_weights": True,
        "object_points_exact_reuse": True,
        "robot_delaunay_invocation_count": 0,
        "optimization_performed": False,
        "qpos_updated": False,
        "base_updated": False,
        "qpos_modified": False,
        "base_pose_modified": False,
        "sdf_accessed": False,
        "collision_surface_accessed": False,
        "eq8_implemented": False,
        "jacobian_shape": [GRAPH_VERTEX_COUNT * 3, robot_model.num_dofs],
        "base_perturbation_diagnostic": {
            "translation_epsilon_m": 1.0e-3,
            "rotation_epsilon_rad": 1.0e-3,
            "does_not_define_stage9_parameterization": True,
        },
        "evaluation_time_s": float(sum(timings)),
        "per_frame_evaluation_time_s": timings,
        "frame_status": ["valid" for _ in values],
        "provenance": {
            "source_edges_reused": True,
            "source_weights_reused": True,
            "object_points_exactly_reused": True,
            "warm_start_evaluation_only": True,
            "stage9_started": False,
        },
        "assumptions": ["A_INTERACTION_BASE_DIFFERENTIABILITY_001"],
    }
    result = InteractionEvaluationTrajectory(
        metadata=metadata,
        timestamps=graph.timestamps.copy(),
        qpos=np.asarray(warm_start.arrays["qpos"], dtype=np.float64).copy(),
        base_pose_scene=np.asarray(warm_start.arrays["base_pose_scene"], dtype=np.float64).copy(),
        robot_keypoints_scene=np.stack([item["robot_keypoints_scene"] for item in values]),
        robot_vertices=np.stack([item["robot_vertices"] for item in values]),
        robot_laplacian=np.stack([item["robot_laplacian"] for item in values]),
        residual=np.stack([item["residual"] for item in values]),
        scaled_residual=np.stack([item["scaled_residual"] for item in values]),
        e_im=np.asarray([item["e_im"] for item in values], dtype=np.float64),
        per_vertex_contribution=np.stack([item["per_vertex_contribution"] for item in values]),
        per_hand_point_contribution=np.stack(
            [item["per_hand_point_contribution"] for item in values]
        ),
        per_object_point_contribution=np.stack(
            [item["per_object_point_contribution"] for item in values]
        ),
        max_residual_vertex=np.asarray(
            [item["max_residual_vertex"] for item in values], dtype=np.int64
        ),
        qpos_jacobian=np.stack([item["qpos_jacobian"] for item in values]),
        base_translation_sensitivity=np.asarray(
            [item["base_translation_sensitivity"] for item in values], dtype=np.float64
        ),
        base_rotation_sensitivity=np.asarray(
            [item["base_rotation_sensitivity"] for item in values], dtype=np.float64
        ),
        frame_valid=np.ones(len(values), dtype=bool),
        frame_status=["valid" for _ in values],
    )
    return result.validate()


__all__ = [
    "INTERACTION_EVALUATION_SCHEMA_VERSION",
    "InteractionEvaluationError",
    "InteractionEvaluationTrajectory",
    "evaluate_interaction_graph",
]
