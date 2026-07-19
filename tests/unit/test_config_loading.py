from pathlib import Path

from toporetarget.config.loader import load_path_config


def test_config_precedence(tmp_path: Path) -> None:
    local = tmp_path / ".local" / "config.yaml"
    local.parent.mkdir()
    local.write_text("storage_root: /local\n", encoding="utf-8")
    config = load_path_config(
        tmp_path,
        local_path=local,
        environ={"REF2DEX_STORAGE_ROOT": "/env"},
        overrides={"storage_root": "/cli"},
    )
    assert config.storage_root == Path("/cli")


def test_defaults_are_repository_relative_without_scanning(tmp_path: Path) -> None:
    config = load_path_config(tmp_path, environ={})
    assert config.artimano_asset_root == tmp_path / ".local" / "assets" / "artimano"
    assert config.paper_path == tmp_path / "docs" / "TopoRetarget.pdf"
