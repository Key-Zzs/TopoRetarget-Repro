"""Isaac Lab configuration for the independent Stage 16-D environment."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from toporetarget.rl.physics_retargeting.self_collision import load_self_collision_contract

from .world_wrist_direct_env_cfg import (
    REPO_ROOT,
    IsaacWorldWristFingerDirectRLEnvCfg,
    configure_explicit_virtual_wrist,
    configure_uniform_reference_retiming,
)

_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
_SELF_COLLISION_CONTRACT = REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml"


@configclass
class IsaacPhysicsConsistentRetargetingEnvCfg(IsaacWorldWristFingerDirectRLEnvCfg):
    """Stage 16-D preserves C.2 physics/action/observation and changes task gates only."""

    semantic_contract_paths = {
        "hocap_170105": str(_REPORT_ROOT / "task_semantics_170105.json"),
        "hocap_170650": str(_REPORT_ROOT / "task_semantics_170650.json"),
    }
    contact_topology_path = str(_REPORT_ROOT / "contact_topology.json")
    task_gate_path = str(_REPORT_ROOT / "anti_degenerate_contract.json")
    reward_contract_path = str(_REPORT_ROOT / "reward_contract.json")
    self_collision_contract_path = str(_SELF_COLLISION_CONTRACT)
    physics_consistent_protocol = "physics_consistent_retargeting_v1"
    contact_telemetry = "aggregate"
    reference_time_scale = 8
    episode_length_s = 321 / 20.0
    reset_reference_index = "frame0"
    diagnostic_kinematic_object = False
    stage16d_fixed_clip: str | None = None


def configure_stage16d_nominal(
    cfg: IsaacPhysicsConsistentRetargetingEnvCfg,
    *,
    num_envs: int,
    clip: str | None = None,
) -> None:
    if num_envs < 1:
        raise ValueError("Stage16D needs at least one environment")
    if clip is not None and (not clip or any(token in clip for token in ("/", "\\", ".."))):
        raise ValueError("invalid Stage16D clip")
    cfg.scene.num_envs = num_envs
    cfg.balanced_clip_assignment = clip is None
    cfg.alternate_clip_on_reset = False
    configure_uniform_reference_retiming(cfg, time_scale=8)
    configure_explicit_virtual_wrist(
        cfg, profile_identifier="high_authority_bounded", authority_enabled=True
    )
    self_collision = load_self_collision_contract(
        Path(cfg.self_collision_contract_path), repo_root=REPO_ROOT
    )
    cfg.robot.spawn.articulation_props.enabled_self_collisions = (
        self_collision.enabled_self_collisions
    )
    if clip is not None:
        cfg.stage16d_fixed_clip = clip


def configure_independent_physics_contracts(
    cfg: IsaacPhysicsConsistentRetargetingEnvCfg,
    *,
    clip_id: str,
    contract_root: Path,
) -> None:
    """Bind per-lineage semantic/gate receipts without changing shared method values."""

    if not clip_id or any(token in clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_PHYSICS_CLIP_ID_INVALID")
    root = contract_root.resolve()
    paths = {
        "semantic": root / f"{clip_id}.task_semantics.json",
        "topology": root / "contact_topology.json",
        "task_gate": root / "anti_degenerate_contract.json",
        "reward": root / "reward_contract.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"INDEPENDENT_PHYSICS_CONTRACT_MISSING:{missing}")
    cfg.semantic_contract_paths = {clip_id: str(paths["semantic"])}
    cfg.contact_topology_path = str(paths["topology"])
    cfg.task_gate_path = str(paths["task_gate"])
    cfg.reward_contract_path = str(paths["reward"])


def stage16d_contract_paths(cfg: IsaacPhysicsConsistentRetargetingEnvCfg) -> tuple[Path, ...]:
    return tuple(Path(value) for value in cfg.semantic_contract_paths.values()) + (
        Path(cfg.contact_topology_path),
        Path(cfg.task_gate_path),
        Path(cfg.reward_contract_path),
        Path(cfg.self_collision_contract_path),
    )


__all__ = [
    "IsaacPhysicsConsistentRetargetingEnvCfg",
    "configure_independent_physics_contracts",
    "configure_stage16d_nominal",
    "stage16d_contract_paths",
]
