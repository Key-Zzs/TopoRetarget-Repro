import json
import subprocess
from pathlib import Path

import toporetarget.paths.assets as assets


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "ManipTrans"
    asset = source / "maniptrans_envs" / "assets" / "mano_urdf"
    asset.mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "LICENSE").write_text("license", encoding="utf-8")
    (asset / "rh_urdf_meshes").mkdir()
    (asset / "lh_urdf_meshes").mkdir()
    for side in ("rh", "lh"):
        (asset / f"{side}_urdf_meshes" / "palm_vis.obj").write_text("mesh", encoding="utf-8")
        (asset / f"{side}_urdf_meshes" / "palm_collision.obj").write_text("mesh", encoding="utf-8")
        (asset / f"{side}_mano.urdf").write_text(
            f'<robot name="{side}"><link name="palm"><visual><geometry>'
            f'<mesh filename="{side}_urdf_meshes/palm_vis.obj"/></geometry></visual>'
            f'<collision><geometry><mesh filename="{side}_urdf_meshes/palm_collision.obj"/>'
            "</geometry></collision></link></robot>",
            encoding="utf-8",
        )
    return source


def test_import_manifest_and_hash_check(tmp_path: Path, monkeypatch) -> None:
    source = make_source(tmp_path)
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="abc123\n", stderr=""
        ),
    )
    destination = tmp_path / "imported"
    result = assets.import_artimano(source, destination)
    assert result.status == "imported"
    manifest = json.loads((destination / "asset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == "abc123"
    assert manifest["modified"] is False
    assert manifest["mesh_reference_validation"]["valid"] is True
    assert len(manifest["imported_files"]) == 6
    assert assets.check_artimano_assets(destination).status == "ok"
    (destination / "rh_mano.urdf").write_text("changed", encoding="utf-8")
    assert assets.check_artimano_assets(destination).status == "invalid"


def test_import_dry_run_does_not_create_destination(tmp_path: Path, monkeypatch) -> None:
    source = make_source(tmp_path)
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="abc123\n", stderr=""
        ),
    )
    destination = tmp_path / "dry-run"
    result = assets.import_artimano(source, destination, dry_run=True)
    assert result.status == "dry_run_ok"
    assert not destination.exists()
