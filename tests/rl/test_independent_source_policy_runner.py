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


def test_run_step_refuses_to_overwrite_failed_receipt(tmp_path: Path) -> None:
    runner = _load_runner()
    logs = tmp_path / "logs"
    logs.mkdir()
    runner.atomic_write_json(
        logs / "failed.receipt.json",
        {"status": "FAIL", "command": [sys.executable, "-c", "pass"]},
    )

    with pytest.raises(
        FileExistsError, match="INDEPENDENT_SOURCE_POLICY_REFUSES_RECEIPT_OVERWRITE"
    ):
        runner._run_step(
            "failed",
            [sys.executable, "-c", "pass"],
            log_root=logs,
        )


def test_l0_uses_materialized_reference_time_scale() -> None:
    trainer = (REPO_ROOT / "scripts/rl/isaaclab/train_stage16d_ppo26d.py").read_text(
        encoding="utf-8"
    )
    binding = trainer[trainer.index("configure_independent_clip_runtime(") :]
    binding = binding[: binding.index(")")]

    assert "reference_time_scale=" not in binding


def test_l0_uses_qualified_reference_kinematics_v2() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )
    start = runner.index('"train_l0"')
    l0_step = runner[start : runner.index("l0 = _json(l0_result)", start)]

    assert "str(reference_v2)" in l0_step
    assert "str(reference_v1)" not in l0_step


def test_source_policy_runner_contains_no_standalone_strict_v4_training_route() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert '"train_strict_v4"' not in runner
    assert '"--strict-per-finger-contact-reward-v4"' not in runner
    assert '"target-reward-v4-samples"' not in runner


def test_auto_v2_forbids_standalone_strict_v4_and_freezes_grouped_rse() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert '"IndependentSourcePolicyReceiptV4"' in runner
    assert '"selected_route": "ZERO_RESIDUAL"' in runner
    assert '"selected_route": "CORRECTED_L0"' in runner
    assert '"status": "FORBIDDEN_NOT_RUN"' in runner
    assert '"reward_aggregation": "grouped_multiplicative_v1"' in runner
    assert '"interaction_term": "u10_per_finger_pair_contact_primitive_v1"' in runner
    assert '"rse_enabled": True' in runner
    assert '"standalone_strict_v4_ppo": False' in runner


def test_source_controller_auto_v2_is_the_fail_safe_default() -> None:
    parser = _load_runner()._parser()
    action = next(item for item in parser._actions if item.dest == "source_policy_profile")

    assert action.default == "source_controller_auto_v2"
    assert tuple(action.choices) == ("source_controller_auto_v2",)


def test_cpu_authorities_precede_gpu_object_import() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert runner.index('"prepare_reference"') < runner.index('"materialize_source_contact"')
    assert runner.index('"materialize_source_contact"') < runner.index('"import_object_usd"')


def test_semantic_pass_is_required_before_reference_or_source_controller() -> None:
    module = _load_runner()
    semantic_action = next(
        item for item in module._parser()._actions if item.dest == "semantic_qualification"
    )
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert semantic_action.required is True
    assert runner.index("require_semantic_admission(") < runner.index('"prepare_reference"')
    prepare_step = runner[runner.index('"prepare_reference"') :]
    prepare_step = prepare_step[: prepare_step.index("expected_artifacts")]
    assert '"--geometric-receipt"' in prepare_step
    assert '"--semantic-qualification"' in prepare_step


def test_zero_residual_is_qualified_before_bounded_l0_fallback() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert runner.index('"qualify_zero_residual_deterministic_v2"') < runner.index('"train_l0"')
    assert '"ZERO_RESIDUAL_NETWORK"' in runner
    assert "ZERO_RESIDUAL_NETWORK_PARITY_OR_EXECUTABILITY_FAILED" in runner
    assert '"source_controller_executability_v2": "PASS"' in runner


def test_cpu_only_checkpoint_is_explicitly_non_ppo_and_resumable() -> None:
    runner = (REPO_ROOT / "scripts/rl/isaaclab/run_independent_source_policy.py").read_text(
        encoding="utf-8"
    )

    assert '"IndependentSourcePolicyPrerequisitesReceiptV2"' in runner
    assert '"standalone_strict_v4_training": "FORBIDDEN_NOT_RUN"' in runner
    assert '"terminal_scope": "CPU_AUTHORITIES_ONLY"' in runner
    assert '"ppo_optimizer_steps": 0' in runner
    assert runner.index("if args.stop_after_cpu_authorities:") < runner.index('"import_object_usd"')


@pytest.mark.parametrize(
    "script_name",
    ["train_stage16d_ppo26d.py", "train_stage16d_ppo26d_object_twist.py"],
)
def test_training_failure_skips_potentially_hanging_simulation_cleanup(
    script_name: str,
) -> None:
    trainer = (REPO_ROOT / "scripts/rl/isaaclab" / script_name).read_text(encoding="utf-8")
    cleanup = trainer[trainer.rindex("finally:") : trainer.rindex('if __name__ == "__main__":')]

    assert "if active_error is None:" in cleanup
    guarded = cleanup[cleanup.index("if active_error is None:") :]
    assert guarded.index("env.close()") < guarded.index("app.close(")
