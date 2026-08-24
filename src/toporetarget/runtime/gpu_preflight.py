"""GPURuntimePreflightV1 receipt validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_gpu_preflight_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("GPU_PREFLIGHT_RECEIPT_OBJECT_REQUIRED")
    required = {
        "schema_version",
        "status",
        "execution_context",
        "host",
        "driver",
        "gpu_names",
        "cuda_visible_devices",
        "torch_version",
        "torch_cuda_available",
        "torch_device_count",
        "isaac_bootstrap",
        "timestamp",
        "cpu_fallback",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"GPU_PREFLIGHT_RECEIPT_FIELDS_MISSING:{missing}")
    if value.get("schema_version") != "GPURuntimePreflightV1":
        raise ValueError("GPU_PREFLIGHT_SCHEMA_INVALID")
    if value.get("status") != "PASS":
        raise ValueError(f"GPU_REQUIRED_UNAVAILABLE:{value.get('status')}")
    if value.get("execution_context") != "HOST_UNSANDBOXED":
        raise ValueError("GPU_PREFLIGHT_NOT_HOST_AUTHORITY")
    if value.get("torch_cuda_available") is not True or int(value.get("torch_device_count", 0)) < 1:
        raise ValueError("GPU_PREFLIGHT_TORCH_CUDA_REQUIRED")
    if value.get("isaac_bootstrap") != "PASS":
        raise ValueError("GPU_PREFLIGHT_ISAAC_BOOTSTRAP_REQUIRED")
    if value.get("cpu_fallback") is not False:
        raise ValueError("GPU_PREFLIGHT_CPU_FALLBACK_FORBIDDEN")
    gpu_names = value.get("gpu_names")
    if not isinstance(gpu_names, list) or not gpu_names:
        raise ValueError("GPU_PREFLIGHT_GPU_NAME_REQUIRED")
    return dict(value)


def validate_gpu_preflight_receipt(path: str | Path) -> dict[str, Any]:
    receipt = Path(path).resolve()
    if not receipt.is_file():
        raise FileNotFoundError(f"GPU_PREFLIGHT_RECEIPT_MISSING:{receipt}")
    return validate_gpu_preflight_payload(json.loads(receipt.read_text(encoding="utf-8")))


__all__ = ["validate_gpu_preflight_payload", "validate_gpu_preflight_receipt"]
