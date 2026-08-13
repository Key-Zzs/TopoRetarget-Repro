"""CPU-only contracts for the Stage 16-C.5A R1 diagnostic drivers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(raw: str, derived: str) -> list[dict[str, object]]:
    return [
        {
            "clip": f"clip-{row // 4}",
            "phase": f"phase-{row % 4}",
            "frame": row,
            "measurement_fingerprints": [{"raw": raw, "derived": derived}],
        }
        for row in range(8)
    ]


def _worker(raw: str, derived: str) -> dict[str, object]:
    return {"rows": _rows(raw, derived)}


def test_cross_process_summary_requires_exactly_twenty_workers() -> None:
    driver = _module(
        "stage16c5_cross_process_diagnostic",
        "scripts/rl/isaaclab/run_stage16c5_cross_process_diagnostic.py",
    )
    reports = [_worker("raw", "derived") for _ in range(20)]
    result = driver.summarize_workers(reports)
    assert len(result) == 8
    assert all(row["raw_fingerprint_identical"] for row in result)
    assert all(row["derived_fingerprint_identical"] for row in result)
    try:
        driver.summarize_workers(reports[:-1])
    except ValueError as error:
        assert "exactly 20" in str(error)
    else:
        raise AssertionError("cross-process driver accepted fewer than 20 workers")


def test_measurement_fingerprint_is_stable_and_content_sensitive() -> None:
    diagnostic = _module(
        "stage16c5_natural_nondeterminism",
        "scripts/rl/isaaclab/diagnose_stage16c5_natural_nondeterminism.py",
    )
    first = {"raw": {"state": torch.tensor([[1.0, 2.0]])}}
    same = {"raw": {"state": torch.tensor([[1.0, 2.0]])}}
    changed = {"raw": {"state": torch.tensor([[1.0, 3.0]])}}
    assert diagnostic._fingerprint(first) == diagnostic._fingerprint(same)
    assert diagnostic._fingerprint(first) != diagnostic._fingerprint(changed)


def test_contact_telemetry_snapshot_is_read_only_and_per_environment() -> None:
    diagnostic = _module(
        "stage16c5_natural_nondeterminism_telemetry",
        "scripts/rl/isaaclab/diagnose_stage16c5_natural_nondeterminism.py",
    )
    records = [
        {"env_id": 0, "net_contact_force_world_on_object_n": [1.0, 0.0, 0.0]},
        {"env_id": 1, "net_contact_force_world_on_object_n": [2.0, 0.0, 0.0]},
    ]
    env = SimpleNamespace(
        cfg=SimpleNamespace(contact_telemetry="aggregate"),
        contact_substep_records=records,
        _contact_substep_record_total=2,
    )
    snapshot = diagnostic._contact_telemetry(env, torch.tensor([0, 1]))
    assert snapshot["latest_record_count"] == 2
    assert snapshot["per_environment"]["0"] != snapshot["per_environment"]["1"]
    assert records[0]["net_contact_force_world_on_object_n"] == [1.0, 0.0, 0.0]


def test_origin_and_telemetry_summaries_keep_physics_and_origin_checks_separate() -> None:
    summary = _module(
        "stage16c5_origin_and_telemetry",
        "scripts/rl/isaaclab/summarize_stage16c5_origin_and_telemetry.py",
    )
    origin_check = {
        "unique_origin_count": 33,
        **{
            name: {
                "world_position_max_abs": 3.0,
                "scene_local_max_abs": 0.01,
                "world_minus_origin_delta_max_abs": 0.01,
            }
            for name in (
                "robot_root_state",
                "object_170105_root_state",
                "object_170650_root_state",
            )
        },
    }
    rows = []
    for index in range(8):
        rows.append(
            {
                "clip": f"clip-{index // 4}",
                "phase": f"phase-{index % 4}",
                "frame": index,
                "measurement_fingerprints": [{"raw": "same", "derived": "same"}] * 20,
                "origin_invariance": [origin_check] * 20,
            }
        )
    report = {"rows": rows}
    assert summary.summarize_origin(report)["result"] == "ENV_ORIGIN_NORMALIZATION_VALID"
    telemetry = summary.summarize_telemetry([report, report, report])
    assert telemetry["result"] == "CONTACT_TELEMETRY_READ_ONLY_CONFIRMED"


def test_closeout_separates_single_env_repeatability_from_vector_raw_divergence() -> None:
    closeout = _module(
        "stage16c5_r1_closeout",
        "scripts/rl/isaaclab/assemble_stage16c5a_r1_closeout.py",
    )
    single = {
        "rows": [
            {"errors": [{"raw_state_max_abs": {"state": 0.0}, "termination_exact": True}] * 19}
            for _ in range(8)
        ]
    }
    vector = {
        "rows": [
            {
                "errors": [
                    {
                        "raw_state_max_abs": {
                            "robot_joint_pos": 0.0,
                            "object_170650_root_state": 0.25,
                            "source_env_origins": 99.0,
                        },
                        "derived_state": {"object_position_m": 0.25},
                        "reward_components_max_abs": {"total": 1.0},
                    }
                ]
            }
        ]
    }
    assert closeout._all_errors_zero(single)
    analysis = closeout._raw_derived_analysis(vector)
    assert analysis["classification"] == "RAW_SIMULATOR_STATE_DIVERGENCE"
    assert "source_env_origins" not in analysis["raw_simulator_state_max_abs"]
