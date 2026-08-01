from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from toporetarget.rl.stage16c0 import (
    Stage16C0Failure,
    Stage16C0PlatformConfig,
    Stage16C0RecoveryStateMachine,
    Stage16C0Status,
    build_install_commands,
    classify_stage16c0_status,
)

CONFIG = Path("configs/rl/stage16/isaaclab_platform.yaml")


def test_nvidia_trailing_release_zero_is_equivalent() -> None:
    from scripts.verify_stage16_isaaclab_platform import _versions_equivalent

    assert _versions_equivalent("5.1.0.0", "5.1.0")
    assert not _versions_equivalent("5.0.0.0", "5.1.0")


def _complete_evidence() -> dict[str, bool]:
    return {
        "host_compatible": True,
        "isolated_environment": True,
        "versions_frozen": True,
        "isaac_sim_import": True,
        "isaac_sim_empty_scene": True,
        "isaac_lab_import": True,
        "official_smoke": True,
        "gpu_physx": True,
        "headless": True,
        "vector_128": True,
        "cuda_tensors": True,
        "bootstrap_dry_run": True,
        "verify_script": True,
        "reproduction_files": True,
    }


def test_platform_config_freezes_official_stable_stack_and_c0_scope() -> None:
    config = Stage16C0PlatformConfig.load(CONFIG)
    assert config.python_version == "3.11"
    assert config.python_exact_version == "3.11.15"
    assert config.isaac_sim_version == "5.1.0"
    assert config.isaac_sim_bundle == "all"
    assert config.extension_cache == "runtime_on_demand"
    assert config.preinstall_full_extension_cache is False
    assert config.isaac_lab_tag == "v2.3.2"
    assert config.isaac_lab_commit == "37ddf626871758333d6ed89cf64ad702aef127d0"
    assert config.torch_version == "2.7.0"
    assert config.cuda_runtime == "cu128"
    assert config.smoke_steps == 1000
    assert config.vector_env_counts == (1, 128, 512)
    assert config.raw["scope"]["allow_stage16_c1"] is False
    assert "ppo_training" in config.raw["scope"]["prohibited"]
    assert config.raw["licenses"]["authorization_recorded"] is False


def test_environment_manifest_is_isolated_and_python_311() -> None:
    payload = yaml.safe_load(Path("environment.stage16_isaaclab.yml").read_text())
    assert payload["name"] == "toporetarget-isaaclab"
    assert "python=3.11.15" in payload["dependencies"]
    assert "setuptools=80.9.0" in payload["dependencies"]
    assert "numpy=1.26.0" in payload["dependencies"]
    assert "pyyaml=6.0.2" in payload["dependencies"]
    assert payload["name"] != "toporetarget-rl"


def test_command_construction_uses_fixed_versions_and_parameterized_path(tmp_path: Path) -> None:
    config = Stage16C0PlatformConfig.load(CONFIG)
    external = tmp_path / "external" / "IsaacLab"
    commands = build_install_commands(
        config,
        environment_name="qualification-env",
        external_root=external,
    )
    flattened = "\n".join(" ".join(command) for command in commands)
    assert "isaacsim[all]==5.1.0" in flattened
    assert "extscache" not in flattened
    assert "torch==2.7.0" in flattened
    assert "v2.3.2" in flattened
    assert "--editable" in flattened
    assert "source/isaaclab_tasks" in flattened
    assert "source/isaaclab_rl[none]" in flattened
    assert "isaaclab.sh" not in flattened
    assert str(external) in flattened
    assert "/home/deepcybo" not in Path("scripts/bootstrap_stage16_isaaclab_env.sh").read_text()
    with pytest.raises(ValueError, match="narrow"):
        build_install_commands(config, environment_name="qualification-env", external_root="/")


def test_bootstrap_dry_run_does_not_create_external_root(tmp_path: Path) -> None:
    external = tmp_path / "never-created" / "IsaacLab"
    result = subprocess.run(
        [
            "bash",
            "scripts/bootstrap_stage16_isaaclab_env.sh",
            "--env-name",
            "stage16c0-dry-run-test",
            "--external-root",
            str(external),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DRY_RUN:" in result.stdout
    assert "isaacsim" in result.stdout
    assert "--no-build-isolation flatdict==4.0.1" in result.stdout
    assert "v2.3.2" in result.stdout
    assert not external.exists()
    assert (
        "OMNI_KIT_ACCEPT_EULA" not in Path("scripts/bootstrap_stage16_isaaclab_env.sh").read_text()
    )


def test_recovery_state_machine_enforces_all_budgets() -> None:
    machine = Stage16C0RecoveryStateMachine()
    for _ in range(4):
        transition = machine.record(
            failure=Stage16C0Failure.NETWORK_OR_ASSET_CACHE_FAILURE,
            evidence={"source": "official"},
            fallback="retry_official_source",
            repair="http_1_1",
            rerun="fixed_tag_fetch",
            result="RETRY",
        )
    assert transition.result == "BLOCKED_CLASS_REPAIR_BUDGET_EXHAUSTED"
    assert not machine.as_dict()["bounded"]

    switches = Stage16C0RecoveryStateMachine()
    for failure in (
        Stage16C0Failure.GLIBC_INCOMPATIBLE,
        Stage16C0Failure.ISAAC_SIM_IMPORT_FAILURE,
        Stage16C0Failure.ISAAC_LAB_IMPORT_FAILURE,
    ):
        transition = switches.record(
            failure=failure,
            evidence={},
            fallback="next_official_method",
            repair="switch",
            rerun="installation",
            result="RETRY",
            retry=False,
            method_switch=True,
        )
    assert transition.result == "BLOCKED_INSTALLATION_METHOD_SWITCH_BUDGET_EXHAUSTED"


def test_status_classifier_is_fail_closed_and_viewer_is_soft_gate() -> None:
    evidence = _complete_evidence()
    assert classify_stage16c0_status(evidence, viewer_available=True) is Stage16C0Status.VALIDATED
    assert (
        classify_stage16c0_status(evidence, viewer_available=False)
        is Stage16C0Status.VALIDATED_WITH_LIMITATIONS
    )
    evidence["vector_128"] = False
    assert classify_stage16c0_status(evidence, viewer_available=True) is Stage16C0Status.PARTIAL
    evidence["vector_128_blocked"] = True
    assert classify_stage16c0_status(evidence, viewer_available=True) is Stage16C0Status.BLOCKED


def test_base_import_does_not_load_isaac_or_torch() -> None:
    command = (
        "import sys; import toporetarget.rl.stage16c0; "
        "assert not any(name == 'torch' or name.startswith('isaac') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_runtime_phase_fails_closed_before_isaac_import_without_eula_authorization(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_stage16_isaaclab_platform.py",
            "--phase",
            "full",
            "--output-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    summary = yaml.safe_load((tmp_path / "final_summary.json").read_text())
    assert summary["status"] == "STAGE16C0_ISAACLAB_PLATFORM_BLOCKED"
    assert summary["blocker"] == "ISAACLAB_EULA_ACCEPTANCE_REQUIRED"
    assert summary["eula"]["accepted_for_run"] is False
    assert "OMNI_KIT_ACCEPT_EULA" not in result.stdout
