"""Tests for derived, non-solver static ContactPose closeout evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _module() -> object:
    path = Path(__file__).resolve().parents[2] / "scripts/stage12_contactpose_static_closeout.py"
    spec = importlib.util.spec_from_file_location("stage12_contactpose_static_closeout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_static_not_applicable_arrays_are_explicitly_excluded_from_finite_gate() -> None:
    module = _module()
    arrays = {
        "accepted": np.asarray([1], dtype=np.int8),
        "continuity_base_rotation_rad": np.asarray([np.nan]),
    }
    assert not module._finite_arrays(arrays)
    assert module._finite_arrays(
        arrays, allowed_not_applicable=frozenset({"continuity_base_rotation_rad"})
    )


def test_static_html_annotation_is_idempotent_and_marks_temporal_na(tmp_path: Path) -> None:
    module = _module()
    html = tmp_path / "result.html"
    html.write_text("<html><body>content</body></html>", encoding="utf-8")
    audit = {"object_watertight": True}
    module._annotate_static_html(html, audit, "compiled_sign")
    module._annotate_static_html(html, audit, "compiled_sign")
    document = html.read_text(encoding="utf-8")
    assert document.count('id="stage12-static-context"') == 1
    assert "NOT_APPLICABLE" in document
    assert "generalized_winding_on_original_mesh" in document
