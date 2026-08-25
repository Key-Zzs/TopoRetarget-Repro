"""CPU-only static contracts for the C.3 object-centric contact repair."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_SOURCE = (
    REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py"
)
ENV_CONFIG_SOURCE = (
    REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py"
)
PPO_ENV_SOURCE = (
    REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
)
CONTACT_CONFIG = REPO_ROOT / "configs/rl/stage16/isaaclab_contact_telemetry.yaml"
PHYSICAL_EVALUATOR_SOURCE = REPO_ROOT / "scripts/rl/isaaclab/evaluate_physical_hoi.py"


def test_contact_repair_uses_two_object_side_views_not_21_hand_side_views() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    assert "object_170105_hand_contact" in source
    assert "object_170650_hand_contact" in source
    assert 'prim_path=f"{{ENV_REGEX_NS}}/{object_name}"' in source
    assert "filter_prim_paths_expr=filter_prim_paths" in source
    assert 'sensor_count": len(self._object_contact_sensors)' in source
    assert "for body_name in HAND_COLLISION_BODY_NAMES" in source
    # The one factory is invoked only by the two-object tuple above; a
    # hand-body loop must only populate its filter list, not construct views.
    assert source.count("ContactSensorCfg(") == 1


def test_contact_telemetry_is_observational_and_bounded() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    config = CONTACT_CONFIG.read_text(encoding="utf-8")
    assert 'if self.cfg.contact_telemetry == "off"' in source
    assert "bounded_latest_only" in source
    assert "contact_record_capacity" in source
    assert "future_ppo_default: aggregate" in config
    assert "reward_or_control_effect: none" in config
    assert "fake_point_force: forbidden" in config


def test_contact_profile_uses_usd_clone_for_128_env_contact_views() -> None:
    config_source = ENV_CONFIG_SOURCE.read_text(encoding="utf-8")
    config = CONTACT_CONFIG.read_text(encoding="utf-8")
    assert "clone_in_fabric=False" in config_source
    assert "scene_clone: usd_clone" in config
    assert "raw_force_matrix_shape: [num_envs, 1, 21, 3]" in config


def test_ppo_contact_reward_selects_configured_objects_without_development_ids() -> None:
    source = PPO_ENV_SOURCE.read_text(encoding="utf-8")
    method = source[
        source.index("def _active_object_pair_force_matrix") : source.index(
            "def _reference_expected_contact_mask"
        )
    ]

    assert "self._object_specs" in method
    assert "self._clip_index" in method
    assert '"Object170105"' not in method
    assert '"Object170650"' not in method


def test_table_contact_telemetry_uses_scene_key_not_usd_prim_name() -> None:
    source = PHYSICAL_EVALUATOR_SOURCE.read_text(encoding="utf-8")
    snapshot = source[
        source.index("def _initial_trace_snapshot") : source.index("def _prepend_initial_trace")
    ]

    assert 'sensor_name = f"{env._object_specs[clip_index][2]}_support_contact"' in snapshot
    assert 'sensor_name = f"{env._object_specs[clip_index][1]}_support_contact"' not in snapshot
