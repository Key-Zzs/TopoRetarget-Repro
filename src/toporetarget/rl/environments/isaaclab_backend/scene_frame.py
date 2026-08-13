"""The frozen Stage 16-C scene-local/world-frame boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage16CSceneFrameContractV1:
    """Keep per-environment origins out of tracking and reference tensors."""

    identifier: str = "Stage16CSceneFrameContractV1"
    source_world_frame: str = "WorldWristFingerReferenceV1 world_scene"
    scene_local_world_frame: str = "per_environment_reference_world"
    isaac_stage_frame: str = "Isaac_global_stage"
    equation: str = "p_global_env = p_reference_world + env_origin"
    tracking_frame: str = "scene_local_world"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def scene_to_global(points, env_origins):
    """Broadcast scene-local positions into Isaac global-stage coordinates."""

    return points + env_origins


def global_to_scene(points, env_origins):
    """Broadcast Isaac global-stage positions into scene-local coordinates."""

    return points - env_origins


__all__ = [
    "Stage16CSceneFrameContractV1",
    "global_to_scene",
    "scene_to_global",
]
