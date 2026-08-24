from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.independent_physical_refinement import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = REPO_ROOT / "scripts/physics/run_independent_physical_support.py"
    spec = importlib.util.spec_from_file_location("independent_physical_support_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare():
    path = REPO_ROOT / "scripts/physics/prepare_physical_support.py"
    spec = importlib.util.spec_from_file_location("independent_physical_support_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_summarizer():
    path = REPO_ROOT / "scripts/physics/summarize_independent_support_preflight.py"
    spec = importlib.util.spec_from_file_location("independent_support_preflight_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_path_requires_matching_artifact_hash(tmp_path: Path) -> None:
    runner = _load_runner()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    row = {"path": str(artifact), "sha256": runner.sha256_file(artifact)}

    assert runner._receipt_path(row) == artifact.resolve()
    artifact.write_text('{"drift": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="INDEPENDENT_SUPPORT_RECEIPT_ARTIFACT_DRIFT"):
        runner._receipt_path(row)


def test_support_resolves_strict_mask_from_source_contact_receipt() -> None:
    source = (REPO_ROOT / "scripts/physics/run_independent_physical_support.py").read_text(
        encoding="utf-8"
    )

    assert 'source_contact["artifacts"]["strict_mask"]' in source
    assert 'source_contact["artifacts"]["support_native"]' in source
    assert 'contact_root / f"strict_source_contact_mask_' not in source
    assert '"IndependentSourcePolicyReceiptV3"' in source
    assert '"IndependentSourcePolicyPrerequisitesReceiptV2"' in source
    assert '"IndependentSourcePolicyReceiptV1"' not in source
    assert '"l0_then_physical_grouped_rse_v1"' in source
    assert '"gpu_physx_authorized": not preflight_reasons' in source
    assert '"l0_training_authorized": not preflight_reasons' in source
    assert '"CPU_SUPPORT_PREFLIGHT_ONLY"' in source


def test_native_contact_mask_excludes_transition_boundary(tmp_path: Path) -> None:
    prepare = _load_prepare()
    path = tmp_path / "native.npz"
    confirmed = np.zeros((4, 2), dtype=bool)
    probable = np.zeros((4, 2), dtype=bool)
    transition = np.zeros((4, 2), dtype=bool)
    transition[1, 0] = True
    probable[2, 1] = True
    confirmed[3, 0] = True
    np.savez_compressed(
        path,
        confirmed_contact=confirmed,
        probable_contact=probable,
        transition=transition,
    )

    assert prepare._native_contact_mask(path, frame_count=4).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_native_contact_mask_prefers_all_hands_support_exclusion(tmp_path: Path) -> None:
    prepare = _load_prepare()
    path = tmp_path / "all_hands_native.npz"
    np.savez_compressed(
        path,
        combined_support_exclusion_mask=np.asarray([True, True, False, False]),
        # These legacy/right-hand fields must not be substituted for the
        # manifest-bound combined bimanual authority.
        confirmed_contact=np.zeros((4, 5), dtype=bool),
        probable_contact=np.zeros((4, 5), dtype=bool),
        transition=np.zeros((4, 5), dtype=bool),
    )

    assert prepare._native_contact_mask(path, frame_count=4).tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_preflight_aggregate_authorizes_only_pass_rows(tmp_path: Path) -> None:
    summarizer = _load_summarizer()
    clips = [f"hocap_fake_{index}" for index in range(5)]
    manifest = {
        "HELD_OUT_SET_FROZEN": "YES",
        "held_out_count": 5,
        "clips": [
            {"clip_id": clip, "exclusion_audit": {"outcome_observed": False}} for clip in clips
        ],
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    for index, clip in enumerate(clips):
        root = tmp_path / "clips" / clip / "support"
        root.mkdir(parents=True)
        artifacts = {}
        for name in (
            "support_resolution",
            "geometry_validation",
            "plane_fit",
            "native_contact_authority",
        ):
            path = root / f"{name}.json"
            path.write_text("{}\n", encoding="utf-8")
            artifacts[name] = {
                "path": str(path),
                "sha256": summarizer.sha256_file(path),
            }
        passed = index < 2
        preflight = root / "preflight.json"
        preflight.write_text(
            json.dumps(
                {
                    "status": "PASS" if passed else "BLOCKED",
                    "support_type": "INFERRED_PLANAR_SUPPORT",
                    "support_patch_type": "AREA_SUPPORT" if passed else "POINT_SUPPORT",
                    "geometry_status": "PASS" if passed else "FAIL",
                    "contact_mask_used": True,
                }
            ),
            encoding="utf-8",
        )
        artifacts["preflight"] = {
            "path": str(preflight),
            "sha256": summarizer.sha256_file(preflight),
        }
        receipt = {
            "schema_version": "IndependentPhysicalSupportPreflightReceiptV1",
            "status": "PASS" if passed else "BLOCKED",
            "clip_id": clip,
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "terminal_scope": "CPU_SUPPORT_PREFLIGHT_ONLY",
            "gpu_physx_authorized": passed,
            "l0_training_authorized": passed,
            "ppo_optimizer_steps": 0,
            "artifacts": artifacts,
            "reasons": [] if passed else ["SUPPORT_PATCH_NOT_AREA:POINT_SUPPORT"],
        }
        (root / "support_preflight_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    aggregate = summarizer.build_aggregate(manifest=manifest, report_root=tmp_path)

    assert aggregate["status"] == "PARTIAL_PASS_FAIL_CLOSED"
    assert aggregate["l0_training_authorized_clips"] == clips[:2]
    assert aggregate["blocked_clips"] == clips[2:]
    assert aggregate["downstream"]["grouped_multiplicative_rse_ppo"] == "NOT_RUN"
