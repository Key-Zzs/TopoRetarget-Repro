#!/usr/bin/env python3
"""Capture immutable preflight evidence for Stage 16-D metric qualification."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/preflight"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _write(path: Path, value: str) -> None:
    path.write_text(value if value.endswith("\n") else value + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "git_status.txt", _run(["git", "status", "--short", "--branch"]).stdout)
    _write(output / "staged.patch", _run(["git", "diff", "--cached", "--binary"]).stdout)
    _write(output / "unstaged.patch", _run(["git", "diff", "--binary"]).stdout)
    _write(
        output / "processes.txt",
        _run(["ps", "-eo", "pid,ppid,pgid,user,stat,pcpu,pmem,nlwp,args"]).stdout,
    )
    gpu = _run(["nvidia-smi"], check=False)
    _write(output / "gpu_before.txt", gpu.stdout + gpu.stderr)
    free = _run(["free", "-h"], check=False)
    disk = _run(["df", "-h"], check=False)
    _write(output / "resources.txt", free.stdout + free.stderr + disk.stdout + disk.stderr)
    stat = REPO_ROOT.stat()
    ownership = {
        "schema_version": "Stage16DPreflightOwnershipV1",
        "generated_at": datetime.now(UTC).isoformat(),
        "effective_uid": os.geteuid(),
        "effective_user": pwd.getpwuid(os.geteuid()).pw_name,
        "repo_root": str(REPO_ROOT),
        "repo_owner_uid": stat.st_uid,
        "repo_owner_user": pwd.getpwuid(stat.st_uid).pw_name,
        "unrelated_processes_terminated": 0,
    }
    _write(output / "ownership.json", json.dumps(ownership, indent=2, sort_keys=True))

    smoke: dict[str, object] = {"run": False, "status": "NOT_REQUESTED"}
    if args.run_smoke:
        if not args.accept_eula:
            raise RuntimeError("explicit --accept-eula is required for the Isaac Lab smoke")
        smoke_root = output / "platform_smoke"
        command = [
            "conda",
            "run",
            "-n",
            "toporetarget-isaaclab",
            "env",
            "OMNI_KIT_ACCEPT_EULA=YES",
            "python",
            "scripts/verify_stage16_isaaclab_platform.py",
            "--phase",
            "full",
            "--steps",
            "100",
            "--output-root",
            str(smoke_root),
            "--accept-eula",
        ]
        result = _run(command, check=False)
        _write(output / "isaac_smoke.log", result.stdout + result.stderr)
        smoke = {
            "run": True,
            "command": command,
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "output_root": str(smoke_root.relative_to(REPO_ROOT)),
        }
        _write(output / "isaac_smoke.json", json.dumps(smoke, indent=2, sort_keys=True))
        if result.returncode != 0:
            raise RuntimeError("STAGE16D_PREFLIGHT_ISAAC_SMOKE_FAILED")
    print(json.dumps({"output": str(output), "smoke": smoke}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
