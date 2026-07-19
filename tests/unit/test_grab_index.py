from pathlib import Path

from tests.unit.test_grab_parser import _write_fixture
from toporetarget.data.indexes.grab import build_grab_index, load_grab_index


def test_grab_index_is_filename_first_and_deterministic(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "GRAB").parents[2]
    first = build_grab_index(grab_root=root, output=tmp_path / "index")
    entries = load_grab_index(tmp_path / "index")
    assert len(entries) == 1
    assert entries[0]["sequence_id"] == "s1/demo"
    assert entries[0]["metadata_quality"] == "filename_derived"
    assert "source_hash" not in entries[0]
    assert first["manifest"]["hashing_mode"] == "none"


def test_grab_index_marks_deleted_entries(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    root = source.parents[2]
    build_grab_index(grab_root=root, output=tmp_path / "index")
    source.unlink()
    refreshed = build_grab_index(grab_root=root, output=tmp_path / "index")
    assert refreshed["manifest"]["deleted_count"] == 1
    assert load_grab_index(tmp_path / "index") == []
