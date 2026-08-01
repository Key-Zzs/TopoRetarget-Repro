#!/usr/bin/env python3
"""Qualify the Stage 16-C.0 Isaac Lab platform without implementing Stage C.1.

Isaac imports are deliberately confined to runtime phase functions after an
AppLauncher has started.  ``--phase static`` is safe in the base repository
environment and never launches Isaac Sim or accepts a license agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toporetarget.rl.stage16c0 import (  # noqa: E402
    Stage16C0PlatformConfig,
    classify_stage16c0_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_platform.yaml",
    )
    parser.add_argument("--external-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c_isaaclab_platform",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "static",
            "imports",
            "empty-scene",
            "official-smoke",
            "vector",
            "viewer",
            "full",
        ),
        default="full",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Set OMNI_KIT_ACCEPT_EULA=YES for this run after recorded user authorization",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_s": time.monotonic() - started,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "duration_s": time.monotonic() - started,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_snapshot() -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = (
        [value.strip() for value in query["stdout"].splitlines()[0].split(",")]
        if query["returncode"] == 0 and query["stdout"]
        else []
    )
    return {
        "query": query,
        "name": fields[0] if len(fields) >= 1 else None,
        "utilization_percent": int(fields[1]) if len(fields) >= 2 else None,
        "memory_used_mib": int(fields[2]) if len(fields) >= 3 else None,
        "memory_total_mib": int(fields[3]) if len(fields) >= 4 else None,
        "performance_state": fields[4] if len(fields) >= 5 else None,
    }


def _git_value(root: Path, *arguments: str) -> str | None:
    result = _run(["git", "-C", str(root), *arguments])
    return result["stdout"] if result["returncode"] == 0 else None


def _host_evidence() -> dict[str, Any]:
    ldd = _run(["ldd", "--version"])
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    disk = shutil.disk_usage(REPO_ROOT)
    os_release = platform.freedesktop_os_release()
    mem_total_kib = 0
    mem_available_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            mem_total_kib = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            mem_available_kib = int(line.split()[1])
    gpu_fields: list[str] = []
    if gpu["returncode"] == 0 and gpu["stdout"]:
        gpu_fields = [value.strip() for value in gpu["stdout"].splitlines()[0].split(",")]
    gpu_name = gpu_fields[0] if len(gpu_fields) >= 1 else None
    driver = gpu_fields[1] if len(gpu_fields) >= 2 else None
    vram_mib = int(gpu_fields[2]) if len(gpu_fields) >= 3 else None
    compute_capability = gpu_fields[3] if len(gpu_fields) >= 4 else None
    bus_id = gpu_fields[4] if len(gpu_fields) >= 5 else None
    glibc_version = platform.libc_ver()[1]
    checks = {
        "os": os_release.get("ID") == "ubuntu"
        and os_release.get("VERSION_ID") in {"22.04", "24.04"},
        "glibc": _version_tuple(glibc_version) >= _version_tuple("2.35"),
        "gpu": bool(gpu_name and "RTX" in gpu_name),
        "driver": bool(driver and _version_tuple(driver) >= _version_tuple("580.65.06")),
        "vram": bool(vram_mib is not None and vram_mib >= 16000),
        "ram": mem_total_kib >= 32 * 1024 * 1024,
        "disk": disk.free >= 50 * 1024**3,
    }
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "os_release": os_release,
        "glibc": platform.libc_ver(),
        "ldd": ldd,
        "gpu_query": gpu,
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": mem_total_kib * 1024,
        "ram_available_bytes": mem_available_kib * 1024,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": {
            "name": gpu_name,
            "driver": driver,
            "vram_mib": vram_mib,
            "compute_capability": compute_capability,
            "bus_id": bus_id,
        },
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "xdg_session_type": os.environ.get("XDG_SESSION_TYPE"),
        "checks": checks,
        "status": (
            "HOST_COMPATIBILITY_PASS" if all(checks.values()) else "HOST_COMPATIBILITY_BLOCKED"
        ),
    }


def _version_tuple(value: str) -> tuple[int, ...]:
    components = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def _write_host_markdown(path: Path, host: dict[str, Any]) -> None:
    gpu = host["gpu"]
    checks = host["checks"]
    rows = (
        ("OS", host["os_release"].get("PRETTY_NAME"), "Ubuntu 22.04/24.04", checks["os"]),
        ("glibc", host["glibc"][1], ">= 2.35", checks["glibc"]),
        ("GPU", gpu["name"], "RTX GPU; RTX 4080 minimum", checks["gpu"]),
        ("Driver", gpu["driver"], ">= 580.65.06", checks["driver"]),
        ("VRAM", f"{gpu['vram_mib']} MiB", ">= 16000 MiB", checks["vram"]),
        (
            "RAM",
            f"{host['ram_total_bytes'] / 1024**3:.1f} GiB",
            ">= 32 GiB",
            checks["ram"],
        ),
        (
            "Free disk",
            f"{host['disk_free_bytes'] / 1024**3:.1f} GiB",
            ">= 50 GiB",
            checks["disk"],
        ),
    )
    lines = [
        "# Stage 16-C.0 Host Compatibility",
        "",
        f"Status: `{host['status']}`",
        "",
        "| Item | Actual | Official requirement | Result |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {item} | {actual} | {requirement} | {'PASS' if passed else 'FAIL'} |"
        for item, actual, requirement, passed in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _external_evidence(root: Path, config: Stage16C0PlatformConfig) -> dict[str, Any]:
    exists = root.is_dir()
    head = _git_value(root, "rev-parse", "HEAD") if exists else None
    tag = _git_value(root, "describe", "--tags", "--exact-match") if exists else None
    remote = _git_value(root, "remote", "get-url", "origin") if exists else None
    status = _git_value(root, "status", "--short") if exists else None
    submodules = _git_value(root, "submodule", "status") if exists else None
    return {
        "path": str(root.resolve()),
        "exists": exists,
        "expected_url": "https://github.com/isaac-sim/IsaacLab.git",
        "remote_url": remote,
        "expected_tag": config.isaac_lab_tag,
        "tag": tag,
        "expected_commit": config.isaac_lab_commit,
        "commit": head,
        "dirty": status not in (None, ""),
        "status": status,
        "submodules": submodules,
        "validated": (
            exists
            and head == config.isaac_lab_commit
            and tag == config.isaac_lab_tag
            and remote == "https://github.com/isaac-sim/IsaacLab.git"
            and status == ""
        ),
    }


def _static_validation(
    config: Stage16C0PlatformConfig, external_root: Path, output_root: Path
) -> dict[str, bool]:
    host = _host_evidence()
    external = _external_evidence(external_root, config)
    packages = {
        name: _package_version(name)
        for name in ("isaacsim", "torch", "torchvision", "isaaclab", "isaaclab_tasks")
    }
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    installation = {
        "install_method": config.install_method,
        "environment_name": config.environment_name,
        "python": platform.python_version(),
        "python_required": config.python_version,
        "python_exact_required": config.python_exact_version,
        "packages": packages,
        "isaac_sim_required": config.isaac_sim_version,
        "isaac_lab_tag": config.isaac_lab_tag,
        "isaac_lab_commit": config.isaac_lab_commit,
        "torch_required": config.torch_version,
        "cuda_runtime_required": config.cuda_runtime,
        "eula_accepted_by_script": False,
    }
    official_sources = {
        **dict(config.raw["official_compatibility"]),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_policy": "latest_stable_non_beta",
    }
    dry_run = _run(
        [
            "bash",
            str(REPO_ROOT / "scripts/bootstrap_stage16_isaaclab_env.sh"),
            "--env-name",
            config.environment_name,
            "--external-root",
            str(external_root),
            "--dry-run",
        ]
    )
    _write_json(output_root / "host_compatibility.json", host)
    _write_host_markdown(output_root / "host_compatibility.md", host)
    _write_json(output_root / "official_compatibility_sources.json", official_sources)
    _write_json(output_root / "installation_manifest.json", installation)
    _write_json(output_root / "external_dependency_manifest.json", external)
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"], timeout=120.0)
    _write_json(
        output_root / "conda_environment.json",
        {
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "pip_freeze": pip_freeze,
            "pip_freeze_sha256": hashlib.sha256(pip_freeze["stdout"].encode("utf-8")).hexdigest(),
        },
    )
    scripts = config.raw["smoke"]
    script_evidence = {
        key: {
            "relative_path": value,
            "exists": (external_root / value).is_file(),
        }
        for key, value in scripts.items()
        if key.endswith("_script")
    }
    reproduction_paths = (
        "environment.stage16_isaaclab.yml",
        "requirements-stage16-isaaclab.txt",
        "scripts/bootstrap_stage16_isaaclab_env.sh",
        "scripts/verify_stage16_isaaclab_platform.py",
        "configs/rl/stage16/isaaclab_platform.yaml",
    )
    _write_json(
        output_root / "reproducibility.json",
        {
            "scripts": script_evidence,
            "files": {
                path: {
                    "bytes": (REPO_ROOT / path).stat().st_size,
                    "sha256": _sha256(REPO_ROOT / path),
                }
                for path in reproduction_paths
            },
            "external_checkout": {
                "path": str(external_root.resolve()),
                "tag": external.get("tag"),
                "commit": external.get("commit"),
                "dirty": external.get("dirty"),
            },
        },
    )
    _write_json(output_root / "bootstrap_dry_run.json", dry_run)
    installed_versions_match = (
        platform.python_version() == config.python_exact_version
        and packages["isaacsim"] == config.isaac_sim_version
        and packages["torch"] is not None
        and packages["torch"].split("+")[0] == config.torch_version
        and packages["torchvision"] is not None
        and packages["torchvision"].split("+")[0] == config.torchvision_version
        and external["validated"] is True
    )
    return {
        "host_compatible": host["status"] == "HOST_COMPATIBILITY_PASS",
        "isolated_environment": os.environ.get("CONDA_DEFAULT_ENV") == config.environment_name,
        "versions_frozen": installed_versions_match,
        "reproduction_files": all((REPO_ROOT / path).is_file() for path in reproduction_paths),
        "bootstrap_dry_run": dry_run["returncode"] == 0,
        "verify_script": False,
        "external_checkout": bool(external["validated"]),
        "python_match": python_minor == config.python_version,
    }


def _launch_app(*, headless: bool) -> tuple[Any, Any]:
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, device="cuda:0")
    return launcher, launcher.app


def _runtime_imports(*, headless: bool) -> dict[str, Any]:
    started = time.monotonic()
    launcher = app = None
    try:
        launcher, app = _launch_app(headless=headless)
        import isaaclab
        import isaaclab_tasks
        import isaacsim
        import torch

        return {
            "result": "PASS",
            "isaacsim_module": str(Path(isaacsim.__file__).resolve()),
            "isaaclab_module": str(Path(isaaclab.__file__).resolve()),
            "isaaclab_tasks_module": str(Path(isaaclab_tasks.__file__).resolve()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "duration_s": time.monotonic() - started,
        }
    finally:
        if app is not None:
            app.close()
        del launcher


def _empty_scene(*, steps: int, headless: bool) -> dict[str, Any]:
    started = time.monotonic()
    launcher = app = simulation_context = None
    try:
        launcher, app = _launch_app(headless=headless)
        import isaaclab.sim as sim_utils
        import torch

        simulation_context = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
        )
        simulation_context.reset()
        torch.cuda.reset_peak_memory_stats()
        gpu_before = _gpu_snapshot()
        tensor = torch.zeros((1,), device="cuda:0")
        for _ in range(steps):
            simulation_context.step(render=not headless)
        torch.cuda.synchronize()
        gpu_after = _gpu_snapshot()
        duration = time.monotonic() - started
        return {
            "result": "PASS",
            "headless": headless,
            "steps": steps,
            "physics_device_requested": "cuda:0",
            "simulation_device": str(simulation_context.device),
            "tensor_device": str(tensor.device),
            "finite": bool(torch.isfinite(tensor).all().item()),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "wall_time_s": duration,
            "physics_steps_per_s": steps / duration,
        }
    finally:
        del simulation_context
        if app is not None:
            app.close()
        del launcher


def _first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for child in value.values():
            result = _first_tensor(child)
            if result is not None:
                return result
    return None


def _official_vector_smoke(
    *, task_id: str, num_envs: int, steps: int, headless: bool
) -> dict[str, Any]:
    started = time.monotonic()
    launcher = app = env = None
    try:
        launcher, app = _launch_app(headless=headless)
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=num_envs)
        env = gym.make(task_id, cfg=env_cfg)
        observations, _ = env.reset()
        first_observation = _first_tensor(observations)
        if first_observation is None:
            raise RuntimeError("official task returned no tensor observation")
        origins = env.unwrapped.scene.env_origins
        done_total = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
        reset_counts = torch.zeros(num_envs, dtype=torch.int64, device="cuda:0")
        partial_reset_events = 0
        independent_action_rows = False
        action_shape = tuple(env.action_space.shape)
        torch.cuda.reset_peak_memory_stats()
        gpu_before = _gpu_snapshot()
        for _ in range(steps):
            actions = 2.0 * torch.rand(action_shape, device="cuda:0") - 1.0
            if num_envs > 1 and actions.shape[0] == num_envs:
                independent_action_rows |= bool(
                    torch.unique(actions.reshape(num_envs, -1), dim=0).shape[0] > 1
                )
            observations, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            done_total |= done
            reset_counts += done.to(dtype=torch.int64)
            done_count = int(done.sum().item())
            partial_reset_events += int(0 < done_count < num_envs)
        final_observation = _first_tensor(observations)
        if final_observation is None:
            raise RuntimeError("official task returned no final tensor observation")
        torch.cuda.synchronize()
        gpu_after = _gpu_snapshot()
        duration = time.monotonic() - started
        origin_rows = torch.unique(origins, dim=0).shape[0]
        return {
            "result": "PASS",
            "task_id": task_id,
            "num_envs": num_envs,
            "steps": steps,
            "headless": headless,
            "physics_device_requested": "cuda:0",
            "environment_device": str(env.unwrapped.device),
            "observation_device": str(first_observation.device),
            "observation_shape": list(first_observation.shape),
            "action_shape": list(action_shape),
            "env_origins_shape": list(origins.shape),
            "unique_env_origins": int(origin_rows),
            "done_env_count": int(done_total.sum().item()),
            "reset_count_total": int(reset_counts.sum().item()),
            "reset_count_min": int(reset_counts.min().item()),
            "reset_count_max": int(reset_counts.max().item()),
            "partial_reset_events": partial_reset_events,
            "per_env_action_independence": num_envs == 1 or independent_action_rows,
            "per_env_reset_independence": num_envs == 1 or partial_reset_events > 0,
            "finite_observation": bool(torch.isfinite(final_observation).all().item()),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "wall_time_s": duration,
            "physics_steps_per_s": num_envs * steps / duration,
            "control_steps_per_s": steps / duration,
            "parallel_envs_proven": bool(origin_rows == num_envs),
        }
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()
        del launcher


def _runtime_failure(phase: str, exc: BaseException) -> dict[str, Any]:
    return {
        "result": "FAIL",
        "phase": phase,
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_closeout_artifacts(
    output_root: Path,
    summary: dict[str, Any],
    config: Stage16C0PlatformConfig,
) -> None:
    status = str(summary["status"])
    c1_authorized = status in {
        "STAGE16C0_ISAACLAB_PLATFORM_VALIDATED",
        "STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS",
    }
    evidence = summary["evidence"]
    lines = [
        "# Stage 16-C.0 Isaac Lab platform qualification",
        "",
        f"Status: `{status}`",
        "",
        f"C.1 asset-migration authorization: `{'YES' if c1_authorized else 'NO'}`",
        "",
        "C.2-C.9 authorization: `NO`",
        "",
        "PPO started: `NO`",
        "",
        "## Gate evidence",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{'PASS' if value else 'NOT_PASS'}`"
        for key, value in sorted(evidence.items())
        if isinstance(value, bool)
    )
    (output_root / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    handoff = [
        "# Stage 16-C handoff",
        "",
        f"- C.0: `{status}`",
        f"- C.1 asset migration authorized: `{'YES' if c1_authorized else 'NO'}`",
        "- C.2 custom DirectRLEnv: `NOT_STARTED`",
        "- C.3 semantic parity: `NOT_STARTED`",
        "- C.4 custom-task GPU vectorization: `NOT_STARTED`",
        "- C.5 PhysX oracle: `NOT_STARTED`",
        "- C.6/C.7 PPO: `NOT_STARTED`",
        "- C.8 randomization: `NOT_STARTED`",
        "- C.9 comparison: `NOT_STARTED`",
        "- MuJoCo role: correctness/debug/regression/contact/action-replay/visualization",
    ]
    (output_root / "handoff.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")

    runtime = summary.get("runtime", {})
    vector = summary.get("vector", [])
    _write_json(
        output_root / "resource_usage.json",
        {
            "empty_scene": runtime.get("empty_scene"),
            "vector_runs": vector,
        },
    )
    _write_json(
        output_root / "preflight.json",
        {
            "stage": "16-C.0",
            "platform_only": True,
            "prohibited": list(config.raw["scope"]["prohibited"]),
            "ppo_started": False,
            "stage16_c1_started": False,
        },
    )
    _write_json(
        output_root / "stage16b_closeout_reference.json",
        {
            "branch": "feature/reference-tracking-ppo",
            "commit": "bde5b98ded6a0064f7db3179fd3968a2b0bc1e66",
            "status": "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL",
            "mujoco_backend": "MUJOCO_CORRECTNESS_BACKEND_CLOSED",
            "ppo_started": False,
            "source_report": (
                ".local/reports/stage16b_adaptive_oracle_single_ppo/final_summary.json"
            ),
        },
    )
    _write_json(output_root / "recovery_limits.json", config.raw["recovery"])
    _write_json(
        output_root / "license_authorization.json",
        {
            **dict(config.raw["licenses"]),
            "accepted_for_run": summary.get("eula", {}).get("accepted_for_run", False),
        },
    )


def _run_full_children(
    *,
    args: argparse.Namespace,
    external_root: Path,
    output_root: Path,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    child_root = output_root / "child_phases"

    def run_child(phase: str, *, num_envs: int | None = None) -> tuple[dict[str, Any], Path]:
        phase_root = child_root / (f"{phase}-{num_envs}" if num_envs is not None else phase)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--external-root",
            str(external_root),
            "--output-root",
            str(phase_root),
            "--phase",
            phase,
            "--steps",
            str(steps),
        ]
        if num_envs is not None:
            command.extend(("--num-envs", str(num_envs)))
        if args.accept_eula:
            command.append("--accept-eula")
        process = _run(command, timeout=1800.0)
        for stream in ("stdout", "stderr"):
            if len(process[stream]) > 8000:
                process[stream] = "...<truncated>...\n" + process[stream][-8000:]
        return process, phase_root

    imports_process, imports_root = run_child("imports")
    empty_process, empty_root = run_child("empty-scene")
    official_process, official_root = run_child("official-smoke", num_envs=1)
    vector_process, vector_root = run_child("vector", num_envs=128)

    def load_or_failure(path: Path, phase: str, process: dict[str, Any]) -> Any:
        if path.is_file():
            return _read_json(path)
        return {
            "result": "FAIL",
            "phase": phase,
            "exception_type": "ChildProcessFailure",
            "error": process,
        }

    imports = load_or_failure(imports_root / "isaac_lab_import.json", "imports", imports_process)
    empty = load_or_failure(empty_root / "isaac_sim_empty_scene.json", "empty-scene", empty_process)
    official = load_or_failure(
        official_root / "isaac_lab_official_smoke.json", "official-smoke", official_process
    )
    vector = load_or_failure(
        vector_root / "isaac_lab_official_smoke.json", "vector-128", vector_process
    )
    vector_results = [
        *(official.get("runs", []) if isinstance(official, dict) else []),
        *(vector.get("runs", []) if isinstance(vector, dict) else []),
    ]
    runtime_results = {
        "imports": imports,
        "empty_scene": empty,
        "child_processes": {
            "imports": imports_process,
            "empty_scene": empty_process,
            "official_smoke": official_process,
            "vector_128": vector_process,
        },
    }
    if args.viewer:
        viewer_process, viewer_root = run_child("viewer")
        viewer = load_or_failure(viewer_root / "viewer_validation.json", "viewer", viewer_process)
        runtime_results["child_processes"]["viewer"] = viewer_process
    else:
        viewer = {"result": "NOT_REQUESTED", "soft_gate": True}
    return runtime_results, vector_results, viewer


def main() -> int:
    args = _parser().parse_args()
    config = Stage16C0PlatformConfig.load(args.config)
    license_config = config.raw["licenses"]
    if args.accept_eula:
        if license_config.get("authorization_recorded") is not True:
            raise SystemExit("EULA_REQUIRED: config has no explicit user authorization record")
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    external_root = args.external_root or (REPO_ROOT / config.external_root)
    output_root = args.output_root
    steps = args.steps or config.smoke_steps
    if steps < config.smoke_steps:
        raise SystemExit(f"--steps must be >= {config.smoke_steps}")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")

    evidence = _static_validation(config, external_root, output_root)
    if args.phase == "static":
        evidence["verify_script"] = True
        status = classify_stage16c0_status(evidence, viewer_available=False)
        summary = {
            "phase": "static",
            "status": status.value,
            "evidence": evidence,
            "eula": {
                "authorized": license_config.get("authorization_recorded") is True,
                "accepted_for_run": args.accept_eula,
            },
        }
        _write_json(output_root / "final_summary.json", summary)
        _write_closeout_artifacts(output_root, summary, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if license_config.get("authorization_recorded") is not True:
        evidence.update({"verify_script": True, "eula_blocked": True})
        not_run = {
            "result": "NOT_RUN",
            "failure": "ISAACLAB_EULA_ACCEPTANCE_REQUIRED",
            "reason": "no explicit user authorization is recorded",
        }
        for report_name in (
            "isaac_sim_import.json",
            "isaac_lab_import.json",
            "isaac_sim_empty_scene.json",
            "headless_validation.json",
            "gpu_physx_evidence.json",
            "isaac_lab_official_smoke.json",
            "vector_env_benchmark.json",
            "viewer_validation.json",
        ):
            _write_json(output_root / report_name, not_run)
        transition = {
            "failure": "EULA_REQUIRED",
            "fallback": "complete_static_audit_only",
            "repair": "wait_for_explicit_user_authorization",
            "rerun": "none",
            "result": "ISAACLAB_EULA_ACCEPTANCE_REQUIRED",
            "retried": False,
            "method_switched": False,
        }
        (output_root / "recovery_transitions.jsonl").write_text(
            json.dumps(transition, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "phase": args.phase,
            "status": "STAGE16C0_ISAACLAB_PLATFORM_BLOCKED",
            "blocker": "ISAACLAB_EULA_ACCEPTANCE_REQUIRED",
            "evidence": evidence,
            "runtime": not_run,
            "vector": [],
            "viewer": not_run,
            "eula": {
                "authorized": False,
                "accepted_for_run": False,
                "environment_variable": None,
            },
        }
        _write_json(output_root / "final_summary.json", summary)
        _write_closeout_artifacts(output_root, summary, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    if args.viewer and args.phase not in {"viewer", "full"}:
        raise SystemExit("--viewer is valid only with --phase viewer or --phase full")

    runtime_results: dict[str, Any] = {}
    vector_results: list[dict[str, Any]] = []
    if args.phase == "full":
        runtime_results, vector_results, viewer = _run_full_children(
            args=args,
            external_root=external_root,
            output_root=output_root,
            steps=steps,
        )
        _write_json(output_root / "isaac_lab_import.json", runtime_results["imports"])
        _write_json(output_root / "isaac_sim_import.json", runtime_results["imports"])
        _write_json(output_root / "isaac_sim_empty_scene.json", runtime_results["empty_scene"])
        _write_json(output_root / "headless_validation.json", runtime_results["empty_scene"])
        _write_json(output_root / "gpu_physx_evidence.json", runtime_results["empty_scene"])
    else:
        if args.phase == "imports":
            try:
                runtime_results["imports"] = _runtime_imports(headless=True)
            except BaseException as exc:
                runtime_results["imports"] = _runtime_failure("imports", exc)
            _write_json(output_root / "isaac_lab_import.json", runtime_results["imports"])
            _write_json(output_root / "isaac_sim_import.json", runtime_results["imports"])

        if args.phase == "empty-scene":
            try:
                runtime_results["empty_scene"] = _empty_scene(steps=steps, headless=True)
            except BaseException as exc:
                runtime_results["empty_scene"] = _runtime_failure("empty-scene", exc)
            _write_json(output_root / "isaac_sim_empty_scene.json", runtime_results["empty_scene"])
            _write_json(output_root / "headless_validation.json", runtime_results["empty_scene"])
            _write_json(output_root / "gpu_physx_evidence.json", runtime_results["empty_scene"])

        vector_counts: list[int]
        if args.phase == "official-smoke":
            vector_counts = [1]
        elif args.phase == "vector":
            vector_counts = [args.num_envs]
        else:
            vector_counts = []
        for count in vector_counts:
            try:
                result = _official_vector_smoke(
                    task_id=config.official_task_id,
                    num_envs=count,
                    steps=steps,
                    headless=True,
                )
            except BaseException as exc:
                result = _runtime_failure(f"vector-{count}", exc)
                result["num_envs"] = count
            vector_results.append(result)

        if args.phase == "viewer":
            try:
                viewer = _empty_scene(steps=steps, headless=False)
            except BaseException as exc:
                viewer = _runtime_failure("viewer", exc)
        else:
            viewer = {"result": "NOT_REQUESTED", "soft_gate": True}

    if vector_results:
        payload = {"task_id": config.official_task_id, "runs": vector_results}
        _write_json(output_root / "isaac_lab_official_smoke.json", payload)
        _write_json(output_root / "vector_env_benchmark.json", payload)
    _write_json(output_root / "viewer_validation.json", viewer)

    imports_pass = runtime_results.get("imports", {}).get("result") == "PASS"
    empty_pass = runtime_results.get("empty_scene", {}).get("result") == "PASS"
    vector_1 = any(
        result.get("num_envs") == 1 and result.get("result") == "PASS" for result in vector_results
    )
    vector_128 = any(
        result.get("num_envs") == 128
        and result.get("result") == "PASS"
        and result.get("parallel_envs_proven") is True
        and result.get("per_env_action_independence") is True
        and result.get("per_env_reset_independence") is True
        for result in vector_results
    )
    evidence.update(
        {
            "isaac_sim_import": imports_pass,
            "isaac_lab_import": imports_pass,
            "isaac_sim_empty_scene": empty_pass,
            "official_smoke": vector_1,
            "gpu_physx": empty_pass
            and runtime_results["empty_scene"].get("simulation_device") == "cuda:0",
            "headless": empty_pass,
            "vector_128": vector_128,
            "cuda_tensors": vector_128
            and any(result.get("observation_device") == "cuda:0" for result in vector_results),
            "verify_script": True,
        }
    )
    viewer_pass = viewer.get("result") == "PASS"
    status = classify_stage16c0_status(evidence, viewer_available=viewer_pass)
    summary = {
        "phase": args.phase,
        "status": status.value,
        "evidence": evidence,
        "runtime": runtime_results,
        "vector": vector_results,
        "viewer": viewer,
        "eula": {
            "authorized": license_config.get("authorization_recorded") is True,
            "accepted_for_run": args.accept_eula,
            "environment_variable": "OMNI_KIT_ACCEPT_EULA=YES" if args.accept_eula else None,
        },
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for result in vector_results
        for value in result.values()
    ):
        summary["status"] = "STAGE16C0_ISAACLAB_PLATFORM_PARTIAL"
        summary["non_finite_metric"] = True
    _write_json(output_root / "final_summary.json", summary)
    _write_closeout_artifacts(output_root, summary, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.value.endswith(("VALIDATED", "VALIDATED_WITH_LIMITATIONS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
