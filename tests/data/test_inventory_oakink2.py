from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "data" / "inventory_oakink2.py"
    spec = importlib.util.spec_from_file_location("inventory_oakink2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_fail_closed_without_trajectory_annotations(tmp_path: Path) -> None:
    hub = tmp_path / "data" / "OakInk-v2-hub"
    program = hub / "program" / "program_info"
    program.mkdir(parents=True)
    (program / "scene_01__A001++seq__one.json").write_text(
        json.dumps(
            {
                "((0, 8), (1, 7))": {
                    "primitive": "take",
                    "interaction_mode": "rh_main",
                    "obj_list": ["obj"],
                    "obj_list_lh": [],
                    "obj_list_rh": ["obj"],
                }
            }
        ),
        encoding="utf-8",
    )
    mesh = hub / "object_repair" / "align_ds" / "obj"
    mesh.mkdir(parents=True)
    (mesh / "model.obj").write_text("v 0 0 0\n", encoding="utf-8")
    result = _module().inventory(tmp_path, tmp_path / "report")
    assert result["status"] == "O0_BLOCKED_MISSING_REQUIRED_DATA"
    assert result["counts"]["primitive_task_records"] == 1
    assert (tmp_path / "report" / "primitive_inventory.csv").is_file()


def test_inventory_discovers_preview_pickles_without_filename_heuristics(tmp_path: Path) -> None:
    hub = tmp_path / "data" / "OakInk-v2-hub"
    program = hub / "program" / "program_info"
    program.mkdir(parents=True)
    (program / "scene_01__A001++seq__one.json").write_text("{}", encoding="utf-8")
    annotation = hub / "anno_preview"
    annotation.mkdir()
    (annotation / "scene_01__A001++seq__one.pkl").write_bytes(b"preview")

    result = _module().inventory(tmp_path, tmp_path / "report")

    assert result["status"] == "O0_PASS"
    assert result["counts"]["mano_containing_sequences"] == 1
    assert result["layout"]["trajectory_annotation_file_count"] == 1
