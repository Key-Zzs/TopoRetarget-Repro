#!/usr/bin/env python3
"""Benchmark the C4 retimed explicit-wrist DirectRLEnv on GPU PhysX."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_DEFAULT_ENV_COUNTS = (128, 512, 1024, 2048, 4096)
_OOM_PATTERN = re.compile(
    r"out of memory|cudaerrormemoryallocation|cuda_error_out_of_memory|gpu.*allocation.*fail",
    re.IGNORECASE,
)
_CONTACT_BUFFER_PATTERN = re.compile(
    r"contact.*buffer|gpu_max_rigid_(?:contact|patch)_count|contact.*capacity",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--reference-time-scale", type=int, default=8)
    parser.add_argument("--profile", default="high_authority_bounded")
    parser.add_argument("--env-counts", nargs="+", type=int, default=list(_DEFAULT_ENV_COUNTS))
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--measurement-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-num-envs", type=int)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args()


def _serialize(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _run_child(args: argparse.Namespace) -> int:
    if args.child_num_envs is None or args.child_output is None:
        raise RuntimeError("child mode requires --child-num-envs and --child-output")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_explicit_virtual_wrist,
            configure_uniform_reference_retiming,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = args.child_num_envs
        cfg.scene.lazy_sensor_update = True
        cfg.balanced_clip_assignment = True
        cfg.contact_telemetry = "aggregate"
        configure_uniform_reference_retiming(cfg, time_scale=args.reference_time_scale)
        configure_explicit_virtual_wrist(cfg, profile_identifier=args.profile)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        observation, _ = env.reset(seed=20260804)
        policy = observation["policy"]
        if policy.shape != (args.child_num_envs, 764) or policy.device.type != "cuda":
            raise RuntimeError(
                "C4_OBSERVATION_CONTRACT_FAILURE: "
                f"shape={tuple(policy.shape)}, device={policy.device}"
            )
        action = torch.zeros((args.child_num_envs, 26), device=env.device)
        finite = torch.ones((), dtype=torch.bool, device=env.device)
        for _ in range(args.warmup_steps):
            observation, reward, terminated, timed_out, _ = env.step(action)
            finite &= torch.isfinite(observation["policy"]).all()
            finite &= torch.isfinite(reward).all()
            finite &= torch.isfinite(terminated.to(torch.float32)).all()
            finite &= torch.isfinite(timed_out.to(torch.float32)).all()
        torch.cuda.synchronize(device=env.device)
        reset_count = torch.zeros((), dtype=torch.long, device=env.device)
        reason_count = torch.zeros(
            len(env.extras["stage16"]["termination_reasons"]),
            dtype=torch.long,
            device=env.device,
        )
        started = time.perf_counter()
        for _ in range(args.measurement_steps):
            observation, reward, terminated, timed_out, extras = env.step(action)
            reset_count += (terminated | timed_out).sum()
            reason_count += torch.bincount(
                extras["stage16"]["primary_reason_code"], minlength=reason_count.numel()
            )
            finite &= torch.isfinite(observation["policy"]).all()
            finite &= torch.isfinite(reward).all()
            finite &= torch.isfinite(terminated.to(torch.float32)).all()
            finite &= torch.isfinite(timed_out.to(torch.float32)).all()
        torch.cuda.synchronize(device=env.device)
        elapsed = time.perf_counter() - started
        reset_value = int(reset_count.detach().cpu())
        reason_values = reason_count.detach().cpu().tolist()
        labels = env.extras["stage16"]["termination_reasons"]
        reason_counts = {
            label: int(reason_values[index])
            for index, label in enumerate(labels)
            if reason_values[index]
        }
        contract = env.contract_report()
        contact = env.contact_sensor_contract()
        finite_value = bool(finite.detach().cpu())
        no_hidden_writes = (
            contract["object_rollout_state_writes"] == 0
            and contract["wrist_root_state_writes_during_step"] == 0
        )
        environment_steps = args.child_num_envs * args.measurement_steps
        result = {
            "status": (
                "STAGE16C4_VECTOR_COUNT_VALIDATED"
                if finite_value and no_hidden_writes
                else "STAGE16C4_VECTOR_COUNT_FAILED"
            ),
            "num_envs": args.child_num_envs,
            "warmup_control_steps": args.warmup_steps,
            "measurement_control_steps": args.measurement_steps,
            "measurement_elapsed_s": elapsed,
            "environment_steps_per_s": environment_steps / elapsed,
            "physics_steps_per_s": environment_steps * cfg.decimation / elapsed,
            "samples_per_s": environment_steps / elapsed,
            "action_shape": [args.child_num_envs, 26],
            "observation_shape": [args.child_num_envs, 764],
            "device": str(env.device),
            "physics_dt_s": env.physics_dt,
            "control_dt_s": env.step_dt,
            "decimation": cfg.decimation,
            "reference_time_scale": args.reference_time_scale,
            "retimed_control_steps": env.reference_bank.frame_count,
            "profile": args.profile,
            "finite": finite_value,
            "nan_or_inf": not finite_value,
            "reset_count": reset_value,
            "reset_rate_per_environment_step": reset_value / max(environment_steps, 1),
            "termination_reason_counts": reason_counts,
            "contact_mode": cfg.contact_telemetry,
            "contact_sensor_contract": _serialize(contact),
            "contract": _serialize(contract),
        }
        args.child_output.parent.mkdir(parents=True, exist_ok=True)
        args.child_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0 if result["status"] == "STAGE16C4_VECTOR_COUNT_VALIDATED" else 1
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _gpu_snapshot(worker_pid: int) -> dict[str, int | None]:
    global_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    utilization = None
    global_memory = None
    try:
        utilization, global_memory = (
            int(value.strip()) for value in global_query.stdout.splitlines()[0].split(",")
        )
    except (IndexError, ValueError):
        pass
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    process_memory = None
    for line in process_query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == worker_pid:
                process_memory = int(fields[1])
        except ValueError:
            continue
    return {
        "utilization_percent": utilization,
        "global_memory_used_mib": global_memory,
        "process_memory_used_mib": process_memory,
    }


def _rss_mib(pid: int) -> float | None:
    try:
        rows = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for row in rows:
        if row.startswith("VmRSS:"):
            try:
                return int(row.split()[1]) / 1024.0
            except (IndexError, ValueError):
                return None
    return None


def _worker_command(args: argparse.Namespace, num_envs: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--reference-time-scale",
        str(args.reference_time_scale),
        "--profile",
        args.profile,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--output",
        str(args.output),
        "--child-num-envs",
        str(num_envs),
        "--child-output",
        str(output),
    ]


def _run_monitored_worker(
    args: argparse.Namespace, num_envs: int, worker_root: Path
) -> dict[str, Any]:
    output = worker_root / f"env_{num_envs}.json"
    log_path = worker_root / f"env_{num_envs}.log"
    samples: list[dict[str, float | int | None]] = []
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            _worker_command(args, num_envs, output),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            snapshot = _gpu_snapshot(process.pid)
            samples.append(
                {
                    "elapsed_s": time.monotonic() - started,
                    **snapshot,
                    "rss_mib": _rss_mib(process.pid),
                }
            )
            time.sleep(0.25)
        returncode = process.wait()
    elapsed = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    contact_warning_lines = [
        row
        for row in log_text.splitlines()
        if "warning" in row.lower() and "contact" in row.lower()
    ]
    resource = {
        "sample_count": len(samples),
        "gpu_utilization_mean_percent": (
            sum(
                float(row["utilization_percent"])
                for row in samples
                if row["utilization_percent"] is not None
            )
            / max(sum(row["utilization_percent"] is not None for row in samples), 1)
        ),
        "gpu_utilization_peak_percent": max(
            (
                int(row["utilization_percent"])
                for row in samples
                if row["utilization_percent"] is not None
            ),
            default=None,
        ),
        "global_vram_peak_mib": max(
            (
                int(row["global_memory_used_mib"])
                for row in samples
                if row["global_memory_used_mib"] is not None
            ),
            default=None,
        ),
        "process_vram_peak_mib": max(
            (
                int(row["process_memory_used_mib"])
                for row in samples
                if row["process_memory_used_mib"] is not None
            ),
            default=None,
        ),
        "process_rss_peak_mib": max(
            (float(row["rss_mib"]) for row in samples if row["rss_mib"] is not None),
            default=None,
        ),
    }
    report = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    oom = bool(_OOM_PATTERN.search(log_text))
    contact_buffer_failure = bool(_CONTACT_BUFFER_PATTERN.search(log_text)) and returncode != 0
    if report is not None and returncode == 0:
        outcome = "validated"
    elif oom:
        outcome = "oom"
    elif contact_buffer_failure:
        outcome = "contact_buffer_failure"
    else:
        outcome = "failed"
    return {
        "num_envs": num_envs,
        "outcome": outcome,
        "clean_exit": returncode == 0,
        "returncode": returncode,
        "wall_elapsed_s": elapsed,
        "worker_report": str(output),
        "worker_log": str(log_path),
        "contact_warning_count": len(contact_warning_lines),
        "contact_warning_examples": contact_warning_lines[:10],
        "oom_detected": oom,
        "contact_buffer_failure_detected": contact_buffer_failure,
        "resource": resource,
        "benchmark": report,
    }


def _training_geometry(selected_num_envs: int) -> dict[str, int]:
    target_samples = 65536
    rollout_length = max(8, min(128, target_samples // selected_num_envs))
    samples = selected_num_envs * rollout_length
    return {
        "selected_num_envs": selected_num_envs,
        "rollout_length": rollout_length,
        "shards": max(1, math.ceil(selected_num_envs / 1024)),
        "samples_per_update": samples,
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    if args.reference_time_scale < 1:
        raise SystemExit("--reference-time-scale must be positive")
    if args.warmup_steps != 100 or args.measurement_steps != 500:
        raise SystemExit("formal C4 requires --warmup-steps 100 --measurement-steps 500")
    if args.child_num_envs is not None:
        return _run_child(args)
    env_counts = tuple(args.env_counts)
    if env_counts != _DEFAULT_ENV_COUNTS:
        raise SystemExit(f"formal C4 env counts must be {_DEFAULT_ENV_COUNTS}")
    worker_root = args.output.parent / f"{args.output.stem}.workers"
    if args.output.exists() or worker_root.exists():
        raise FileExistsError(
            f"C4_BENCHMARK_REFUSES_OVERWRITE: output={args.output}, workers={worker_root}"
        )
    worker_root.mkdir(parents=True, exist_ok=False)
    rows = []
    oom_count = 0
    contact_buffer_failure_count = 0
    for num_envs in env_counts:
        row = _run_monitored_worker(args, num_envs, worker_root)
        rows.append(row)
        oom_count += int(row["outcome"] == "oom")
        contact_buffer_failure_count += int(row["outcome"] == "contact_buffer_failure")
        if oom_count >= 2:
            for skipped in env_counts[len(rows) :]:
                rows.append(
                    {
                        "num_envs": skipped,
                        "outcome": "skipped_after_two_ooms",
                        "clean_exit": False,
                    }
                )
            break
    validated = [row for row in rows if row["outcome"] == "validated"]
    non_bounded_failures = [
        row for row in rows if row["outcome"] in {"failed", "contact_buffer_failure"}
    ]
    selected = max((int(row["num_envs"]) for row in validated), default=0)
    passed = (
        selected >= 128
        and oom_count <= 2
        and contact_buffer_failure_count <= 2
        and not non_bounded_failures
        and all(
            row["benchmark"]["finite"]
            and not row["benchmark"]["nan_or_inf"]
            and row["benchmark"]["contract"]["object_rollout_state_writes"] == 0
            and row["benchmark"]["contract"]["wrist_root_state_writes_during_step"] == 0
            for row in validated
        )
    )
    result = {
        "status": (
            "STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED"
            if passed
            else "STAGE16C4_GPU_VECTOR_BACKEND_BLOCKED"
        ),
        "controller": "finite_virtual_6d_wrist_actuator_v1",
        "profile": args.profile,
        "reference_time_scale": args.reference_time_scale,
        "environment_counts": list(env_counts),
        "warmup_control_steps": args.warmup_steps,
        "measurement_control_steps": args.measurement_steps,
        "oom_attempts": oom_count,
        "oom_attempt_limit": 2,
        "contact_buffer_failures": contact_buffer_failure_count,
        "contact_buffer_fix_limit": 2,
        "rows": rows,
        "selection": _training_geometry(selected) if selected else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
