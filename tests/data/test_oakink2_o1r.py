from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[2] / "scripts/data/run_oakink2_o1r.py"
    spec = importlib.util.spec_from_file_location("run_oakink2_o1r", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_review_frames_preserve_primary_and_half_open_end() -> None:
    frames, supplementary = _module().deterministic_review_frames((100, 200), 151)

    assert frames == [100, 124, 149, 151, 174, 199]
    assert supplementary == [100, 124, 149, 174, 199]
    assert 200 not in frames


def test_equivalence_threshold_and_negative_controls() -> None:
    module = _module()
    reference = np.zeros((4, 3), dtype=np.float64)
    identical = module.comparison_metrics(reference, reference.copy())
    one_mm = module.comparison_metrics(reference, reference + np.asarray([0.001, 0.0, 0.0]))
    unit_mismatch = module.comparison_metrics(reference, reference + 1.0)

    assert (
        module.equivalence_status(identical, identical, True) == "OFFICIAL_ADAPTER_EXACT_EQUIVALENT"
    )
    assert module.equivalence_status(one_mm, identical, True) == "OFFICIAL_ADAPTER_NUMERIC_MISMATCH"
    assert module.equivalence_status(unit_mismatch, identical, True) == "UNIT_MISMATCH"
    assert module.equivalence_status(identical, identical, False) == "VERTEX_ORDER_MISMATCH"


def test_tensor_fingerprint_detects_internal_model_change() -> None:
    module = _module()
    original = np.arange(12, dtype=np.float64).reshape(4, 3)
    changed = original.copy()
    changed[2, 1] += 1e-6

    assert (
        module.tensor_receipt(original)["canonical_tensor_sha256"]
        != module.tensor_receipt(changed)["canonical_tensor_sha256"]
    )


def test_mano_asset_authority_exact_mismatch_and_unresolved(
    tmp_path: Path,
) -> None:
    module = _module()
    official = tmp_path / "official.pkl"
    adapter = tmp_path / "adapter.pkl"
    official.write_bytes(b"same")
    adapter.write_bytes(b"same")

    assert module.mano_asset_authority_status(official, adapter, True) == "MANO_ASSET_EXACT_MATCH"
    adapter.write_bytes(b"different")
    assert module.mano_asset_authority_status(official, adapter, False) == "MANO_ASSET_MISMATCH"
    assert (
        module.mano_asset_authority_status(tmp_path / "missing.pkl", adapter, False)
        == "OFFICIAL_MANO_ASSET_UNRESOLVED"
    )


def test_surface_distance_uses_object_transform_without_frame_substitution() -> None:
    module = _module()
    hand = np.asarray([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[1, 0, 3] = 1.0
    surface = np.asarray([[0.0, 0.0, 0.0]])

    distance = module.hand_to_object_surface_distance(hand, transforms, surface)

    assert np.allclose(distance, [1.0, 1.0])


def test_official_reference_is_independent_and_viewer_has_required_controls() -> None:
    root = Path(__file__).parents[2]
    official = (root / "scripts/data/oakink2_official_reference.py").read_text(encoding="utf-8")
    runner = (root / "scripts/data/run_oakink2_o1r.py").read_text(encoding="utf-8")

    assert "import toporetarget" not in official
    assert "OakInk2CanonicalAdapterV1" not in official
    for marker in (
        "OFFICIAL ONLY",
        "ADAPTER ONLY",
        "OVERLAY",
        "Official 21-joint skeleton",
        "closed surface",
        "wireframe overlay",
        "Python-precomputed official and adapter vertices",
        "FRAME_BINDING_EXACT",
    ):
        assert marker in runner


def test_o3_keeps_official_target_primary_and_manifest_v1_immutable() -> None:
    source = (Path(__file__).parents[2] / "scripts/data/run_oakink2_o1r.py").read_text(
        encoding="utf-8"
    )

    assert '"official_target_primary_authority": True' in source
    assert '"official_target_auto_replaced": False' in source
    assert '"v1_bytes_changed": False' in source
    assert '"heldout_downstream_consumed": 0' in source
