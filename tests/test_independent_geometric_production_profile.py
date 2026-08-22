from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.rl.independent_physical_refinement import stable_hash

ROOT = Path(__file__).resolve().parents[1]
FAST_EXACT_V2 = "wuji_continuous_sequential_fast_exact_v2"


def _load_script(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, object]:
    value: dict[str, object] = {
        "HELD_OUT_SET_FROZEN": "YES",
        "held_out_count": 5,
        "primary_object_authority_sha256": "primary-authority",
        "clips": [
            {
                "clip_id": f"hocap_22{index:04d}",
                "exclusion_audit": {"outcome_observed": False},
            }
            for index in range(5)
        ],
    }
    value["manifest_sha256"] = stable_hash(value)
    return value


def test_geometric_runner_accepts_only_frozen_fast_exact_v2() -> None:
    module = _load_script(
        "run_hocap_geometric_retarget_v2_test",
        "scripts/run_hocap_geometric_retarget_v2.py",
    )
    parser = module._parser()
    required = [
        "--manifest",
        "manifest.json",
        "--primary-object-authority",
        "authority.json",
        "--clip-id",
        "hocap_220000",
        "--data-root",
        "raw",
        "--run-root",
        "runs",
        "--report-root",
        "reports",
    ]

    assert parser.parse_args(required).execution_profile == FAST_EXACT_V2
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--execution-profile", "cached_checkpoint_cpu_float64_v1"])

    profile = RefinementExecutionProfile.load(FAST_EXACT_V2, ROOT)
    assert profile.math_equivalent
    assert profile.paper_objective_unchanged
    assert profile.paper_constraints_unchanged
    assert profile.continuity_contract_unchanged
    assert profile.final_full_surface_audit


def test_versioned_production_authority_binds_fast_exact_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script(
        "freeze_independent_physical_authority_test",
        "scripts/freeze_independent_physical_authority.py",
    )
    manifest_path = tmp_path / "manifest.json"
    authority_path = tmp_path / "primary.json"
    output = tmp_path / "production_authority.v2.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    authority_path.write_text(
        json.dumps({"authority_sha256": "primary-authority"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_independent_physical_authority.py",
            "--manifest",
            str(manifest_path),
            "--primary-object-authority",
            str(authority_path),
            "--output",
            str(output),
            "--contract-root",
            str(tmp_path),
        ],
    )

    assert module.main() == 0
    frozen = json.loads(output.read_text(encoding="utf-8"))
    retarget = frozen["authorities"]["retarget"]
    assert frozen["schema_version"] == "IndependentPhysicalRefinementProductionAuthorityV2"
    assert retarget["method"]["execution_profile_id"] == FAST_EXACT_V2
    assert retarget["method"]["math_equivalent"] is True
    assert retarget["command"][-2:] == ["--execution-profile", FAST_EXACT_V2]
    assert len(retarget["method"]["execution_profile_sha256"]) == 64
