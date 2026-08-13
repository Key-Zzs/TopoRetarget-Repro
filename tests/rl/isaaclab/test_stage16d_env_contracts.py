from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stage16d_env_keeps_isaac_imports_optional() -> None:
    import toporetarget.rl.physics_retargeting as package

    assert package.PHYSICS_CONSISTENT_RETARGETING_PROTOCOL == "physics_consistent_retargeting_v1"


def test_stage16d_env_has_no_rollout_state_write_or_hidden_force() -> None:
    path = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend"
        / "physics_consistent_retargeting_env.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write_root_state_to_sim" not in calls
    assert "set_external_force_and_torque" not in calls
    assert "write_joint_state_to_sim" not in calls


def test_stage16d_contract_does_not_mutate_c2_source() -> None:
    source = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend"
        / "physics_consistent_retargeting_env.py"
    ).read_text(encoding="utf-8")
    assert 'strict_source_object_world_tracking_hard_gate": False' in source
    assert "free_physx_rollout_output" in source
    assert 'object_rollout_state_writes": 0' in source


def test_stage16d_owns_versioned_self_collision_and_terminal_windows() -> None:
    cfg = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend"
        / "physics_consistent_retargeting_env_cfg.py"
    ).read_text(encoding="utf-8")
    env = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend"
        / "physics_consistent_retargeting_env.py"
    ).read_text(encoding="utf-8")
    base = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend"
        / "world_wrist_direct_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "stage16d_self_collision.yaml" in cfg
    assert "enabled_self_collisions =" in cfg
    assert "InterFingerCapsulePenetrationV1" in env
    assert "_terminal_observed_steps" in env
    assert "enabled_self_collisions=False" in base
