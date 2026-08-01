"""Fail-closed contracts for Stage 16-C.0 Isaac Lab platform qualification.

This module intentionally imports no Isaac Sim, Isaac Lab, Torch, or CUDA
packages.  The base TopoRetarget package must remain importable when the
optional Isaac stack is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Stage16C0Failure(StrEnum):
    HOST_DRIVER_INCOMPATIBLE = "HOST_DRIVER_INCOMPATIBLE"
    GLIBC_INCOMPATIBLE = "GLIBC_INCOMPATIBLE"
    PYTHON_VERSION_CONFLICT = "PYTHON_VERSION_CONFLICT"
    TORCH_CUDA_CONFLICT = "TORCH_CUDA_CONFLICT"
    ISAAC_SIM_IMPORT_FAILURE = "ISAAC_SIM_IMPORT_FAILURE"
    ISAAC_LAB_IMPORT_FAILURE = "ISAAC_LAB_IMPORT_FAILURE"
    EULA_REQUIRED = "EULA_REQUIRED"
    NETWORK_OR_ASSET_CACHE_FAILURE = "NETWORK_OR_ASSET_CACHE_FAILURE"
    HEADLESS_RENDER_FAILURE = "HEADLESS_RENDER_FAILURE"
    GPU_PHYSX_FAILURE = "GPU_PHYSX_FAILURE"


class Stage16C0Status(StrEnum):
    VALIDATED = "STAGE16C0_ISAACLAB_PLATFORM_VALIDATED"
    VALIDATED_WITH_LIMITATIONS = "STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS"
    PARTIAL = "STAGE16C0_ISAACLAB_PLATFORM_PARTIAL"
    BLOCKED = "STAGE16C0_ISAACLAB_PLATFORM_BLOCKED"


@dataclass(frozen=True)
class Stage16C0Transition:
    failure: str
    attempt: int
    evidence: dict[str, Any]
    fallback: str
    repair: str
    rerun: str
    result: str
    retried: bool
    method_switched: bool
    remaining_class_repairs: int
    remaining_retries: int
    remaining_method_switches: int
    remaining_major_transitions: int


class Stage16C0RecoveryStateMachine:
    """Bound recovery without hiding an exhausted qualification path."""

    repairs_per_class = 3
    retry_budget = 4
    method_switch_budget = 2
    major_transition_budget = 16

    def __init__(self) -> None:
        self.class_attempts: dict[str, int] = {}
        self.retry_count = 0
        self.method_switch_count = 0
        self.major_transition_count = 0
        self.transitions: list[Stage16C0Transition] = []

    def record(
        self,
        *,
        failure: Stage16C0Failure,
        evidence: Mapping[str, Any],
        fallback: str,
        repair: str,
        rerun: str,
        result: str,
        retry: bool = True,
        method_switch: bool = False,
    ) -> Stage16C0Transition:
        key = failure.value
        attempt = self.class_attempts.get(key, 0) + 1
        next_retry_count = self.retry_count + int(retry)
        next_switch_count = self.method_switch_count + int(method_switch)
        next_major_count = self.major_transition_count + 1

        if attempt > self.repairs_per_class:
            result = "BLOCKED_CLASS_REPAIR_BUDGET_EXHAUSTED"
        elif next_retry_count > self.retry_budget:
            result = "BLOCKED_RETRY_BUDGET_EXHAUSTED"
        elif next_switch_count > self.method_switch_budget:
            result = "BLOCKED_INSTALLATION_METHOD_SWITCH_BUDGET_EXHAUSTED"
        elif next_major_count > self.major_transition_budget:
            result = "BLOCKED_MAJOR_TRANSITION_BUDGET_EXHAUSTED"

        self.class_attempts[key] = attempt
        self.retry_count = next_retry_count
        self.method_switch_count = next_switch_count
        self.major_transition_count = next_major_count
        transition = Stage16C0Transition(
            failure=key,
            attempt=attempt,
            evidence=dict(evidence),
            fallback=fallback,
            repair=repair,
            rerun=rerun,
            result=result,
            retried=retry,
            method_switched=method_switch,
            remaining_class_repairs=max(self.repairs_per_class - attempt, 0),
            remaining_retries=max(self.retry_budget - next_retry_count, 0),
            remaining_method_switches=max(self.method_switch_budget - next_switch_count, 0),
            remaining_major_transitions=max(self.major_transition_budget - next_major_count, 0),
        )
        self.transitions.append(transition)
        return transition

    def as_dict(self) -> dict[str, Any]:
        bounded = (
            self.retry_count <= self.retry_budget
            and self.method_switch_count <= self.method_switch_budget
            and self.major_transition_count <= self.major_transition_budget
            and all(count <= self.repairs_per_class for count in self.class_attempts.values())
        )
        return {
            "bounded": bounded,
            "repairs_per_class": self.repairs_per_class,
            "retry_budget": self.retry_budget,
            "method_switch_budget": self.method_switch_budget,
            "major_transition_budget": self.major_transition_budget,
            "class_attempts": dict(self.class_attempts),
            "retry_count": self.retry_count,
            "method_switch_count": self.method_switch_count,
            "major_transition_count": self.major_transition_count,
            "transitions": [asdict(value) for value in self.transitions],
        }


@dataclass(frozen=True)
class Stage16C0PlatformConfig:
    schema_version: int
    stage: str
    environment_name: str
    external_root: str
    install_method: str
    isaac_sim_bundle: str
    extension_cache: str
    preinstall_full_extension_cache: bool
    python_version: str
    python_exact_version: str
    isaac_sim_version: str
    isaac_lab_tag: str
    isaac_lab_commit: str
    torch_version: str
    torchvision_version: str
    cuda_runtime: str
    official_task_id: str
    smoke_steps: int
    vector_env_counts: tuple[int, ...]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Stage16C0PlatformConfig:
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage16-C.0 platform config must be a mapping")
        stack = _mapping(payload, "stack")
        environment = _mapping(payload, "environment")
        smoke = _mapping(payload, "smoke")
        source = _mapping(stack, "isaac_lab_source")
        config = cls(
            schema_version=int(payload.get("schema_version", 0)),
            stage=str(payload.get("stage", "")),
            environment_name=str(environment.get("name", "")),
            external_root=str(environment.get("external_root", "")),
            install_method=str(stack.get("install_method", "")),
            isaac_sim_bundle=str(stack.get("isaac_sim_bundle", "")),
            extension_cache=str(stack.get("extension_cache", "")),
            preinstall_full_extension_cache=bool(
                stack.get("preinstall_full_extension_cache", True)
            ),
            python_version=str(stack.get("python", "")),
            python_exact_version=str(stack.get("python_exact", "")),
            isaac_sim_version=str(stack.get("isaac_sim", "")),
            isaac_lab_tag=str(source.get("tag", "")),
            isaac_lab_commit=str(source.get("commit", "")),
            torch_version=str(stack.get("torch", "")),
            torchvision_version=str(stack.get("torchvision", "")),
            cuda_runtime=str(stack.get("cuda_runtime", "")),
            official_task_id=str(smoke.get("official_task_id", "")),
            smoke_steps=int(smoke.get("steps", 0)),
            vector_env_counts=tuple(int(value) for value in smoke.get("vector_env_counts", [])),
            raw=payload,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.stage != "16-C.0":
            raise ValueError("stage must be exactly 16-C.0")
        if self.install_method != "isaac_sim_pip_plus_isaac_lab_source":
            raise ValueError("unsupported or non-official-first install method")
        if self.isaac_sim_bundle != "all":
            raise ValueError("C0 requires the official Isaac Sim all bundle")
        if self.extension_cache != "runtime_on_demand" or self.preinstall_full_extension_cache:
            raise ValueError(
                "C0 must use bounded runtime extension resolution without full extscache"
            )
        if self.python_version != "3.11":
            raise ValueError("Isaac Sim 5.x requires Python 3.11")
        if not self.python_exact_version.startswith("3.11."):
            raise ValueError("stack.python_exact must freeze an actual Python 3.11 patch")
        if self.isaac_sim_version != "5.1.0":
            raise ValueError("the frozen stable stack requires Isaac Sim 5.1.0")
        if self.isaac_lab_tag != "v2.3.2":
            raise ValueError("the frozen stable stack requires Isaac Lab v2.3.2")
        if len(self.isaac_lab_commit) != 40:
            raise ValueError("Isaac Lab source commit must be a full 40-character SHA")
        if self.torch_version != "2.7.0" or self.cuda_runtime != "cu128":
            raise ValueError("Isaac Sim 5.1 x86_64 requires the frozen Torch 2.7.0 cu128 stack")
        if not self.environment_name:
            raise ValueError("environment.name is required")
        external = Path(self.external_root)
        if external.is_absolute() or "deepcybo" in self.external_root:
            raise ValueError("environment.external_root must be portable and repository-relative")
        if self.smoke_steps < 1000:
            raise ValueError("C0 smoke must execute at least 1000 steps")
        if 128 not in self.vector_env_counts:
            raise ValueError("C0 vector qualification must include 128 environments")
        if not self.official_task_id.startswith("Isaac-"):
            raise ValueError("smoke.official_task_id must identify an official Isaac Lab task")
        scope = _mapping(self.raw, "scope")
        if scope.get("allow_stage16_c1") is not False:
            raise ValueError("Stage16-C.1 must remain fail-closed during C0")
        forbidden = set(scope.get("prohibited", []))
        required_forbidden = {
            "wuji_asset_migration",
            "hocap_object_migration",
            "custom_direct_rl_env",
            "physx_oracle",
            "ppo_training",
        }
        if not required_forbidden.issubset(forbidden):
            raise ValueError("scope.prohibited is missing a Stage16-C.1+ boundary")
        recovery = _mapping(self.raw, "recovery")
        if int(recovery.get("installation_method_switches_used", 0)) > int(
            recovery.get("installation_method_switches", 0)
        ):
            raise ValueError("installation method switch budget is exceeded")
        if int(recovery.get("retries_used_before_runtime", 0)) > int(recovery.get("retries", 0)):
            raise ValueError("retry budget is exceeded")


def classify_stage16c0_status(
    evidence: Mapping[str, bool], *, viewer_available: bool
) -> Stage16C0Status:
    """Classify only from explicit gate evidence; missing evidence is not success."""

    hard_gates = (
        "host_compatible",
        "isolated_environment",
        "versions_frozen",
        "isaac_sim_import",
        "isaac_sim_empty_scene",
        "isaac_lab_import",
        "official_smoke",
        "gpu_physx",
        "headless",
        "vector_128",
        "cuda_tensors",
        "bootstrap_dry_run",
        "verify_script",
        "reproduction_files",
    )
    hard_failures = (
        "unsupported_gpu",
        "incompatible_driver",
        "runtime_blocked",
        "gpu_physx_blocked",
        "headless_blocked",
        "vector_128_blocked",
        "eula_blocked",
    )
    if any(evidence.get(key, False) for key in hard_failures):
        return Stage16C0Status.BLOCKED
    if not all(evidence.get(key, False) for key in hard_gates):
        return Stage16C0Status.PARTIAL
    return (
        Stage16C0Status.VALIDATED
        if viewer_available
        else Stage16C0Status.VALIDATED_WITH_LIMITATIONS
    )


def build_install_commands(
    config: Stage16C0PlatformConfig,
    *,
    environment_name: str,
    external_root: str | Path,
) -> tuple[tuple[str, ...], ...]:
    """Build inspectable argv vectors without executing them or accepting an EULA."""

    root = Path(external_root)
    if not environment_name.strip():
        raise ValueError("environment_name must not be empty")
    if str(root).strip() in {"", "/"}:
        raise ValueError("external_root must be a narrow path")
    return (
        (
            "conda",
            "env",
            "create",
            "--name",
            environment_name,
            "--file",
            "environment.stage16_isaaclab.yml",
        ),
        (
            "conda",
            "run",
            "-n",
            environment_name,
            "python",
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"torch=={config.torch_version}",
            f"torchvision=={config.torchvision_version}",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ),
        (
            "conda",
            "run",
            "-n",
            environment_name,
            "python",
            "-m",
            "pip",
            "install",
            f"isaacsim[all]=={config.isaac_sim_version}",
            "--extra-index-url",
            "https://pypi.nvidia.com",
        ),
        (
            "git",
            "clone",
            "--branch",
            config.isaac_lab_tag,
            "--depth",
            "1",
            "https://github.com/isaac-sim/IsaacLab.git",
            str(root),
        ),
        (
            "conda",
            "run",
            "-n",
            environment_name,
            "python",
            "-m",
            "pip",
            "install",
            "--editable",
            str(root / "source/isaaclab"),
            str(root / "source/isaaclab_assets"),
            str(root / "source/isaaclab_contrib"),
            str(root / "source/isaaclab_tasks"),
            f"{root / 'source/isaaclab_rl'}[none]",
            f"{root / 'source/isaaclab_mimic'}[none]",
        ),
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


__all__ = [
    "Stage16C0Failure",
    "Stage16C0PlatformConfig",
    "Stage16C0RecoveryStateMachine",
    "Stage16C0Status",
    "Stage16C0Transition",
    "build_install_commands",
    "classify_stage16c0_status",
]
