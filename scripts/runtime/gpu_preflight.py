#!/usr/bin/env python3
"""Run GPURuntimePreflightV1 in the exact target environment."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-context",
        choices=("host-unsandboxed", "sandbox-container-diagnostic"),
        required=True,
        help="Only host-unsandboxed can authorize a GPU_REQUIRED stage.",
    )
    parser.add_argument(
        "--isaac-bootstrap",
        action="store_true",
        help="Launch and close a minimal headless Isaac application.",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Required with --isaac-bootstrap.",
    )
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def _nvidia_smi() -> tuple[dict[str, Any], list[str], str | None]:
    listing = _command(["nvidia-smi", "-L"])
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    rows = [line.strip() for line in str(query["output"]).splitlines() if line.strip()]
    names = [row.rsplit(",", 1)[0].strip() for row in rows]
    drivers = {row.rsplit(",", 1)[1].strip() for row in rows if "," in row}
    driver = next(iter(drivers)) if len(drivers) == 1 else None
    return {"listing": listing, "query": query}, names, driver


def _isaac_bootstrap() -> tuple[str, str | None]:
    # Kit may terminate the interpreter during ``app.close``.  Keep the receipt
    # authority in this parent process and isolate Kit lifecycle in a child.
    marker = "GPU_PREFLIGHT_ISAAC_BOOTSTRAP_PASS"
    # Keep ``python -c`` on one physical line.  Isaac Sim 5.1's command-line
    # bootstrap on this host crashes before app startup when the ``-c`` payload
    # itself contains literal newlines.
    program = (
        "from isaacsim import SimulationApp; "
        "app = SimulationApp({'headless': True}); "
        f"print({marker!r}, flush=True); "
        "app.close()"
    )
    # Kit's crash/signal machinery must own a fresh process group, and its
    # verbose output is written to a real file rather than a nested PIPE.
    with tempfile.TemporaryDirectory(prefix="gpu_preflight_isaac_") as temporary:
        log_path = Path(temporary) / "isaac_bootstrap.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-c", program],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                start_new_session=True,
            )
        output = log_path.read_text(encoding="utf-8", errors="replace")
    if process.returncode == 0 and marker in output:
        return "PASS", None
    tail = output[-4000:]
    return "FAIL", f"ISAAC_BOOTSTRAP_CHILD_FAILED:{process.returncode}:{tail}"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    if args.isaac_bootstrap and not args.accept_eula:
        raise ValueError("GPU_PREFLIGHT_ISAAC_BOOTSTRAP_REQUIRES_EULA")
    if args.accept_eula:
        # AppLauncher/Kit consumes the environment variable, not this CLI flag.
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"GPU_PREFLIGHT_REFUSES_OVERWRITE:{output}")
    smi, gpu_names, driver = _nvidia_smi()
    device_nodes = sorted(glob.glob("/dev/nvidia*"))
    # Bootstrap Kit before importing torch/CUDA in this parent process.  On the
    # target host, initializing a separate CUDA runtime first can make the
    # subsequent minimal Kit child terminate during plugin startup.
    if args.isaac_bootstrap:
        isaac_status, isaac_error = _isaac_bootstrap()
    else:
        isaac_status = "NOT_RUN"
        isaac_error = None
    torch_error = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        torch_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    except BaseException as error:
        torch_version = "IMPORT_FAILED"
        cuda_available = False
        device_count = 0
        torch_names = []
        torch_error = f"{type(error).__name__}:{error}"
    host_evidence_pass = (
        smi["listing"]["returncode"] == 0 and smi["query"]["returncode"] == 0 and bool(gpu_names)
    )
    target_environment_pass = cuda_available and device_count > 0
    isaac_pass = args.isaac_bootstrap and isaac_status == "PASS"
    passed = (
        args.execution_context == "host-unsandboxed"
        and host_evidence_pass
        and target_environment_pass
        and isaac_pass
    )
    status = (
        "PASS"
        if passed
        else "DIAGNOSTIC_NOT_HOST_AUTHORITY"
        if args.execution_context == "sandbox-container-diagnostic"
        else "GPU_REQUIRED_UNAVAILABLE"
    )
    receipt = {
        "schema_version": "GPURuntimePreflightV1",
        "status": status,
        "execution_context": (
            "HOST_UNSANDBOXED"
            if args.execution_context == "host-unsandboxed"
            else "SANDBOX_CONTAINER_DIAGNOSTIC"
        ),
        "host": platform.node(),
        "platform": platform.platform(),
        "driver": driver,
        "gpu_names": gpu_names,
        "nvidia_smi": smi,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "device_nodes": device_nodes,
        "torch_version": torch_version,
        "torch_cuda_available": cuda_available,
        "torch_device_count": device_count,
        "torch_device_names": torch_names,
        "torch_error": torch_error,
        "isaac_bootstrap": isaac_status,
        "isaac_bootstrap_error": isaac_error,
        "timestamp": _utc(),
        "cpu_fallback": False,
        "sandbox_cuda_failure_is_host_authority": False,
        "checks": {
            "host_nvidia_smi": host_evidence_pass,
            "target_environment_torch_cuda": target_environment_pass,
            "isaac_bootstrap": isaac_pass,
        },
    }
    _write(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
