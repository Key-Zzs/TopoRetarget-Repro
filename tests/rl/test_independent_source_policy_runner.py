from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py"
    spec = importlib.util.spec_from_file_location("independent_source_policy_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_step_fails_closed_when_zero_exit_omits_artifact(tmp_path: Path) -> None:
    runner = _load_runner()
    expected = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="INDEPENDENT_SOURCE_POLICY_STAGE_FAILED"):
        runner._run_step(
            "zero_exit_missing_output",
            [sys.executable, "-c", "pass"],
            log_root=tmp_path / "logs",
            expected_artifacts=(expected,),
        )

    receipt = runner._json(tmp_path / "logs/zero_exit_missing_output.receipt.json")
    assert receipt["status"] == "FAIL"
    assert receipt["returncode"] == 0
    assert receipt["missing_artifacts"] == [str(expected.resolve())]


def test_run_step_pass_requires_declared_artifact(tmp_path: Path) -> None:
    runner = _load_runner()
    expected = tmp_path / "present.json"

    receipt = runner._run_step(
        "zero_exit_with_output",
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(expected)!r}).write_text('{{}}')",
        ],
        log_root=tmp_path / "logs",
        expected_artifacts=(expected,),
    )

    assert receipt["status"] == "PASS"
    assert receipt["missing_artifacts"] == []


def test_l0_uses_materialized_reference_time_scale() -> None:
    trainer = (
        REPO_ROOT / "scripts/rl/isaaclab/train_stage16d_ppo26d.py"
    ).read_text(encoding="utf-8")
    binding = trainer[trainer.index("configure_independent_clip_runtime(") :]
    binding = binding[: binding.index(")")]

    assert "reference_time_scale=" not in binding
