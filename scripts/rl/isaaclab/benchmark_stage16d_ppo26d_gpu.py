#!/usr/bin/env python3
"""Benchmark actual PPO-26D rollout and update capacity on the host GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.gpu_capacity import (  # noqa: E402
    GpuCapacityMeasurement,
    select_ppo26d_environment_capacity,
)
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d/gpu"
CANDIDATES = (512, 1024, 1536, 2048, 3072, 4096)
OOM_PATTERN = re.compile(r"out of memory|cuda.*allocation|cudaerrormemory", re.IGNORECASE)
CONTACT_PATTERN = re.compile(r"contact.*buffer|contact.*capacity", re.IGNORECASE)
PHYSX_PATTERN = re.compile(r"physx.*fatal|physx.*error", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-counts", nargs="+", type=int, default=list(CANDIDATES))
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse same-root completed candidate rows instead of rerunning them.",
    )
    parser.add_argument("--child-num-envs", type=int)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nvidia_text(arguments: list[str]) -> str:
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, check=False)
    return result.stdout + result.stderr


def gpu_snapshot() -> dict[str, float | None]:
    text = nvidia_text(
        [
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        total, used, free, utilization = (
            float(value.strip()) for value in text.splitlines()[0].split(",")
        )
        return {
            "total_vram_mib": total,
            "used_vram_mib": used,
            "free_vram_mib": free,
            "gpu_utilization": utilization,
        }
    except (IndexError, ValueError):
        return {
            "total_vram_mib": None,
            "used_vram_mib": None,
            "free_vram_mib": None,
            "gpu_utilization": None,
        }


def child_main(args: argparse.Namespace) -> int:
    if args.child_num_envs is None or args.child_output is None:
        raise ValueError("child mode requires --child-num-envs and --child-output")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=args.child_num_envs, rsi=True, critical_dr=False
        )
        startup_started = time.perf_counter()
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        startup_s = time.perf_counter() - startup_started
        reset_started = time.perf_counter()
        observation, _ = env.reset(seed=20260808)
        reset_s = time.perf_counter() - reset_started
        if observation["policy"].shape != (args.child_num_envs, 764):
            raise RuntimeError("PPO26D_OBSERVATION_INVALID")
        action = torch.zeros((args.child_num_envs, 26), device=env.device)
        b0_started = time.perf_counter()
        for _ in range(50):
            env.step(action)
        torch.cuda.synchronize(env.device)
        b0_elapsed_s = time.perf_counter() - b0_started
        b1_started = time.perf_counter()
        for _ in range(100):
            env.step(action)
        torch.cuda.synchronize(env.device)
        b1_elapsed_s = time.perf_counter() - b1_started
        torch.cuda.synchronize(env.device)
        start = time.perf_counter()
        finite = torch.ones((), dtype=torch.bool, device=env.device)
        nan_count = torch.zeros((), dtype=torch.long, device=env.device)
        inf_count = torch.zeros((), dtype=torch.long, device=env.device)
        for _ in range(500):
            observation, reward, terminated, timed_out, _ = env.step(action)
            values = (observation["policy"], reward, terminated.float(), timed_out.float())
            finite &= torch.stack(tuple(torch.isfinite(value).all() for value in values)).all()
            nan_count += sum(torch.isnan(value).sum() for value in values)
            inf_count += sum(torch.isinf(value).sum() for value in values)
        torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats(env.device)
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        update = trainer.collect_and_update(env)
        # The last observation is needed only by checkpoint round-trip callers;
        # benchmark evidence must remain plain JSON.
        update.pop("last_policy_observation")
        torch.cuda.synchronize(env.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(env.device)
        result = {
            "num_envs": args.child_num_envs,
            "startup_and_reset_pass": True,
            "startup_s": startup_s,
            "reset_s": reset_s,
            "b0_control_steps": 50,
            "b0_elapsed_s": b0_elapsed_s,
            "b0_control_steps_per_s": 50 / b0_elapsed_s,
            "b1_warmup_control_steps": 100,
            "b1_elapsed_s": b1_elapsed_s,
            "b1_control_steps_per_s": 100 / b1_elapsed_s,
            "b2_measurement_control_steps": 500,
            "b2_elapsed_s": elapsed,
            "control_steps_per_s": 500 / elapsed,
            "env_steps_per_s": args.child_num_envs * 500 / elapsed,
            "samples_per_s": args.child_num_envs * 500 / elapsed,
            "finite": bool(finite.item()),
            "nan_count": int(nan_count.item()),
            "inf_count": int(inf_count.item()),
            "ppo_update": update,
            "ppo_update_ok": update["actor_parameter_changed"]
            and update["critic_parameter_changed"],
            "rollout_storage_mib": update["rollout_storage_mib"],
            "ppo_update_s": update["ppo_update_s"],
            "free_vram_mib": free_bytes / 2**20,
            "total_vram_mib": total_bytes / 2**20,
            "ppo_peak_allocated_mib": torch.cuda.max_memory_allocated(env.device) / 2**20,
            "cpu_rss_mib": _rss_mib(os.getpid()),
            "rsi": env.rsi_report(),
            "contract": env.contract_report(),
        }
        write_json(args.child_output, result)
        return 0
    except Exception as error:
        write_json(
            args.child_output,
            {
                "num_envs": args.child_num_envs,
                "startup_and_reset_pass": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _rss_mib(pid: int) -> float | None:
    try:
        for row in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if row.startswith("VmRSS:"):
                return int(row.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def run_child(args: argparse.Namespace, count: int, root: Path) -> dict[str, Any]:
    child_output = root / "children" / f"{count}.json"
    log_path = root / "children" / f"{count}.log"
    child_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--output-root",
        str(root),
        "--child-num-envs",
        str(count),
        "--child-output",
        str(child_output),
    ]
    peak_used = 0.0
    peak_utilization = 0.0
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True
        )
        while process.poll() is None:
            snapshot = gpu_snapshot()
            peak_used = max(peak_used, float(snapshot["used_vram_mib"] or 0.0))
            peak_utilization = max(peak_utilization, float(snapshot["gpu_utilization"] or 0.0))
            time.sleep(0.5)
        returncode = process.wait()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    child = json.loads(child_output.read_text(encoding="utf-8")) if child_output.is_file() else {}
    snapshot = gpu_snapshot()
    payload = {
        **child,
        "num_envs": count,
        "returncode": returncode,
        "clean_exit": returncode == 0,
        "wall_time_s": time.perf_counter() - start,
        "peak_vram_mib": peak_used,
        "peak_gpu_utilization": peak_utilization,
        "free_vram_after_exit_mib": snapshot["free_vram_mib"],
        "oom": bool(OOM_PATTERN.search(log)),
        "cuda_allocation_failure": bool(OOM_PATTERN.search(log)),
        "contact_buffer_overflow": bool(CONTACT_PATTERN.search(log)),
        "physx_fatal_warning": bool(PHYSX_PATTERN.search(log)),
        "log": str(log_path.resolve()),
    }
    payload["nan_or_inf"] = not bool(payload.get("finite", False))
    return payload


def completed_rows(root: Path) -> dict[int, dict[str, Any]]:
    """Load only finished prior rows for an explicitly requested resume."""

    path = root / "ppo_gpu_capacity_benchmark.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if isinstance(row, dict) and row.get("clean_exit") and row.get("ppo_update_ok"):
            result[int(row["num_envs"])] = row
    return result


def main() -> int:
    args = parse_args()
    if args.child_num_envs is not None:
        return child_main(args)
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    counts = tuple(args.env_counts)
    if not counts or any(count not in (*CANDIDATES, 5120, 6144) for count in counts):
        raise ValueError("supported benchmark candidates are 512..6144; never 8192")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    before = {
        "nvidia_smi": nvidia_text([]),
        "query": nvidia_text(
            [
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "pmon": nvidia_text(["pmon", "-c", "1"]),
    }
    write_json(
        root / "host_gpu_probe.json",
        {"host_detected": "NVIDIA-SMI" in before["nvidia_smi"], **before},
    )
    (root / "gpu_processes.txt").write_text(before["pmon"], encoding="utf-8")
    (root / "gpu_before_benchmark.txt").write_text(before["nvidia_smi"], encoding="utf-8")
    existing = completed_rows(root) if args.reuse_existing else {}
    rows = [
        existing[count] if count in existing else run_child(args, count, root) for count in counts
    ]
    measurements = [
        GpuCapacityMeasurement(
            num_envs=int(row["num_envs"]),
            samples_per_s=float(row["samples_per_s"]) if row.get("samples_per_s") else None,
            total_vram_mib=float(row.get("total_vram_mib") or 0.0),
            peak_vram_mib=float(row["peak_vram_mib"]) if row.get("peak_vram_mib") else None,
            free_vram_mib=float(row["free_vram_mib"]) if row.get("free_vram_mib") else None,
            ppo_update_ok=bool(row.get("ppo_update_ok")),
            clean_exit=bool(row.get("clean_exit")),
            oom=bool(row.get("oom")),
            cuda_allocation_failure=bool(row.get("cuda_allocation_failure")),
            nan_or_inf=bool(row.get("nan_or_inf")),
            physx_fatal_warning=bool(row.get("physx_fatal_warning")),
            contact_buffer_overflow=bool(row.get("contact_buffer_overflow")),
        )
        for row in rows
    ]
    try:
        selection = select_ppo26d_environment_capacity(measurements)
    except RuntimeError as error:
        selection = {
            "selector": "Stage16DPPOEnvCapacitySelectorV1",
            "status": "FAILED",
            "error": str(error),
            "measurements": [measurement.as_dict() for measurement in measurements],
        }
        write_json(root / "ppo_gpu_capacity_benchmark.json", {"rows": rows, "selection": selection})
        write_json(root / "selected_capacity.json", selection)
        raise
    write_json(root / "ppo_gpu_capacity_benchmark.json", {"rows": rows, "selection": selection})
    write_json(root / "selected_capacity.json", selection)
    with (root / "ppo_gpu_capacity_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "num_envs",
                "startup_s",
                "reset_s",
                "control_steps_per_s",
                "env_steps_per_s",
                "samples_per_s",
                "ppo_update_ok",
                "ppo_update_s",
                "rollout_storage_mib",
                "ppo_peak_allocated_mib",
                "peak_vram_mib",
                "free_vram_mib",
                "nan_count",
                "inf_count",
                "clean_exit",
                "eligible",
            ],
        )
        writer.writeheader()
        for row, measurement in zip(rows, measurements, strict=True):
            writer.writerow(
                {
                    "num_envs": measurement.num_envs,
                    "startup_s": row.get("startup_s"),
                    "reset_s": row.get("reset_s"),
                    "control_steps_per_s": row.get("control_steps_per_s"),
                    "env_steps_per_s": row.get("env_steps_per_s"),
                    "samples_per_s": measurement.samples_per_s,
                    "ppo_update_ok": measurement.ppo_update_ok,
                    "ppo_update_s": row.get("ppo_update_s"),
                    "rollout_storage_mib": row.get("rollout_storage_mib"),
                    "ppo_peak_allocated_mib": row.get("ppo_peak_allocated_mib"),
                    "peak_vram_mib": measurement.peak_vram_mib,
                    "free_vram_mib": measurement.free_vram_mib,
                    "nan_count": row.get("nan_count"),
                    "inf_count": row.get("inf_count"),
                    "clean_exit": measurement.clean_exit,
                    "eligible": measurement.eligible,
                }
            )
    print(
        json.dumps({"selected_num_envs": selection["selected_num_envs"], "output_root": str(root)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
