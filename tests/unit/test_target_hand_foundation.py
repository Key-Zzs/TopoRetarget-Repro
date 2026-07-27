from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from toporetarget.paths.assets import (
    compare_asset_payloads,
    resolve_artimano_asset,
    vendor_artimano,
)
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.registry import RobotHandRegistry

REPO_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = Path(
    os.environ.get("MANIPTRANS_ROOT", "/home/deepcybo/workspace/dex/retarget/ManipTrans")
)


def test_tracked_default_and_contract_are_generic() -> None:
    model = load_artimano_model("right")
    assert model.asset_source == "tracked"
    assert model.asset_root == REPO_ROOT / "third_party" / "robot_hands" / "artimano"
    assert model.spec.asset_bundle.root_relative_path == "third_party/robot_hands/artimano"
    assert model.spec.kinematics.actuated_joint_order == model.dof_names
    assert model.spec.surface_profile.visual_geometry["load_meshes"] is True
    assert model.spec.collision_profile.geometry["load_meshes"] is True
    assert model.spec.simulation.mjcf_relative_path is None


def test_registry_lists_resolution_provenance() -> None:
    rows = RobotHandRegistry(repo_root=REPO_ROOT).list()
    assert {row["name"] for row in rows} == {"artimano_rh", "artimano_lh"}
    assert all(row["asset"]["resolved_asset_source"] == "tracked" for row in rows)
    assert all(row["asset"]["source_commit"] for row in rows)


def test_override_and_legacy_fallback_are_explicit(tmp_path: Path) -> None:
    override = tmp_path / "override"
    resolved = resolve_artimano_asset(
        tmp_path, asset_root=override, environ={"TOPORETARGET_ARTIMANO_ASSET_ROOT": "ignored"}
    )
    assert resolved.source == "override"
    assert resolved.root == override
    legacy = tmp_path / ".local" / "assets" / "artimano"
    legacy.mkdir(parents=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_artimano_asset(tmp_path, environ={})
    assert resolved.source == "legacy"
    assert resolved.legacy_fallback_used is True
    assert caught
    assert "deprecated" in str(caught[0].message)


@pytest.mark.skipif(
    not (SOURCE_ROOT / ".git").exists(), reason="requires a local ManipTrans checkout"
)
def test_vendor_dry_run_and_payload_manifest_are_deterministic(tmp_path: Path) -> None:
    result_a = vendor_artimano(
        SOURCE_ROOT,
        tmp_path / "a",
        dry_run=True,
        imported_at="2026-07-27T19:00:00+08:00",
    )
    result_b = vendor_artimano(
        SOURCE_ROOT,
        tmp_path / "b",
        dry_run=True,
        imported_at="2026-07-27T19:00:00+08:00",
    )
    payload_a, payload_b = result_a.as_dict(), result_b.as_dict()
    payload_a["destination"] = payload_b["destination"]
    assert payload_a == payload_b
    assert result_a.imported_file_count == 98
    assert (
        compare_asset_payloads(
            REPO_ROOT / "third_party" / "robot_hands" / "artimano",
            REPO_ROOT / ".local" / "assets" / "artimano",
        )["status"]
        == "match"
    )


def test_tracked_asset_has_no_local_git_entries() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", ".local"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout == ""
