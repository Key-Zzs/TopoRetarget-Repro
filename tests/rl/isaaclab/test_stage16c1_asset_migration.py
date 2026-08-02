from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (
    BoundedAssetRecovery,
    load_asset_migration_config,
    sha256_file,
)
from toporetarget.rl.environments.isaaclab_backend.asset_validation import (
    classify_c1_status,
    classify_c2_entry,
    validate_manifest_schema,
)
from toporetarget.rl.environments.isaaclab_backend.hocap_object_import import (
    build_hocap_import_command,
)
from toporetarget.rl.environments.isaaclab_backend.wuji_import import (
    build_wuji_import_command,
)
from toporetarget.rl.tracked_links import TRACKED_LINKS_WUJI_RH

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml"


def test_asset_config_freezes_exact_scope_mapping_and_dynamics() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = load_asset_migration_config(CONFIG)
    assert raw["stage"] == "16-C.1"
    assert raw["scope"] == {
        "allow_c2": False,
        "allow_direct_rl_env": False,
        "allow_physx_oracle": False,
        "allow_ppo": False,
        "authorize_c2_entry_when_c1_validated": True,
    }
    assert cfg.wuji.source_commit == "2b57d2621caed4e65207bb767ba25fc8eaec0881"
    assert cfg.wuji.fixed_base is False
    assert len(cfg.wuji.joint_order) == 20
    assert len(set(cfg.wuji.joint_order)) == 20
    assert set(cfg.wuji.semantic_mapping) == set(cfg.wuji.joint_order)
    assert cfg.wuji.tracked_links == TRACKED_LINKS_WUJI_RH
    assert cfg.wuji.collision_strategy == (
        "deterministic_support_hull_proxies_with_floating_root_overlay_v1"
    )
    assert [item.object_id for item in cfg.objects] == ["hocap_170105", "hocap_170650"]
    assert {item.collision_strategy for item in cfg.objects} == {"convex_hull_v1"}
    assert all(item.mass_kg == 0.05 for item in cfg.objects)
    assert all(not item.gravity_enabled and not item.ground_enabled for item in cfg.objects)
    assert all(item.support == "none" for item in cfg.objects)


def test_frozen_sources_validate_when_local_evidence_is_present() -> None:
    cfg = load_asset_migration_config(CONFIG)
    missing = [
        item.source_file for item in cfg.objects if not (REPO_ROOT / item.source_file).is_file()
    ]
    if missing:
        pytest.skip(f"ignored Stage16 evidence is absent: {missing}")
    joints = cfg.validate(REPO_ROOT)
    assert tuple(joint.name for joint in joints) == cfg.wuji.joint_order
    assert all(joint.axis in {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)} for joint in joints)
    assert all(joint.limits[0] < joint.limits[1] for joint in joints)
    for item in cfg.objects:
        assert sha256_file(REPO_ROOT / item.source_file) == item.source_sha256


def test_import_commands_are_explicit_portable_and_eula_scoped(tmp_path: Path) -> None:
    upstream = tmp_path / "wuji-description"
    wuji = build_wuji_import_command(
        REPO_ROOT, upstream_root=upstream, python="python3", accept_eula=True
    )
    objects = build_hocap_import_command(REPO_ROOT, python="python3", accept_eula=True)
    assert wuji[-3:] == ["--upstream-root", str(upstream), "--accept-eula"]
    assert objects[-1] == "--accept-eula"
    assert "OMNI_KIT_ACCEPT_EULA" not in " ".join(wuji + objects)
    for source in (
        REPO_ROOT / "scripts/rl/isaaclab/import_wuji_hand2.py",
        REPO_ROOT / "scripts/rl/isaaclab/import_hocap_objects.py",
        REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/wuji_import.py",
    ):
        assert "/home/" not in source.read_text(encoding="utf-8")


def test_base_import_has_no_isaac_dependency() -> None:
    code = (
        "import sys; import toporetarget.rl.environments.isaaclab_backend; "
        "assert not any(n.startswith(('isaaclab','isaacsim','omni','pxr')) for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manifest_schema_and_fail_closed_status() -> None:
    required = {
        "source_repo": "repo",
        "source_commit": "0" * 40,
        "source_file": "input.usd",
        "license": "MIT",
        "import_tool": {},
        "generated_usd": "output.usd",
        "generated_sha256": "1" * 64,
        "root_prim": "/root",
        "articulation_root": ["/root"],
        "fixed_base": False,
        "body_names": [],
        "joint_names": [],
        "joint_order": [],
        "joint_types": {},
        "joint_axes": {},
        "limits": {},
        "default_pose": {},
        "drive_configuration": {},
        "collision_geoms": [],
        "visual_geoms": [],
        "mass_inertia": {},
        "warnings": [],
    }
    validate_manifest_schema(required)
    with pytest.raises(ValueError, match="missing fields"):
        validate_manifest_schema({})
    assert classify_c1_status({"source": True, "runtime": True}).endswith("VALIDATED")
    assert classify_c1_status({"source": True, "runtime": False}).endswith("PARTIAL")
    assert classify_c1_status({"source": False, "runtime": False}).endswith("BLOCKED")


def test_c2_entry_is_authorized_only_after_validated_c1() -> None:
    validated = "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED"
    assert classify_c2_entry(validated, entry_authorized=True) == (
        "STAGE16C2_DIRECT_RL_ENV_AUTHORIZED"
    )
    assert classify_c2_entry(validated, entry_authorized=False).endswith("BLOCKED")
    assert classify_c2_entry(
        "STAGE16C1_ISAACLAB_ASSET_MIGRATION_PARTIAL", entry_authorized=True
    ).endswith("BLOCKED")


def test_recovery_budgets_are_bounded() -> None:
    recovery = BoundedAssetRecovery()
    for _ in range(3):
        recovery.record("URDF_IMPORT_FAILURE", phase="wuji_import")
    with pytest.raises(RuntimeError, match="repairs_per_class"):
        recovery.record("URDF_IMPORT_FAILURE", phase="wuji_import")
    assert recovery.as_dict()["major_transitions"] == 4


def test_generated_runtime_reports_when_present() -> None:
    report_root = REPO_ROOT / ".local/reports/stage16c1_asset_migration"
    reports = [
        report_root / f"hocap_{object_id}_vector_{count}.json"
        for object_id in ("170105", "170650")
        for count in (1, 128)
    ]
    if not all(path.is_file() for path in reports):
        pytest.skip("ignored Isaac Lab runtime evidence is absent")
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["all_finite"] is True
        assert payload["cuda_tensors"] is True
        assert payload["joints_with_response"] == 20
        assert payload["tracked_links_resolved"] == 16
        assert payload["joint_limit_max_abs_error_rad"] < 1e-5
        assert payload["tensor_shapes"]["finger_action"] == [payload["num_envs"], 20]
        if payload["num_envs"] == 128:
            assert payload["unique_env_origins"] == 128
            assert payload["subset_reset_max_position_error_m"] == 0.0
