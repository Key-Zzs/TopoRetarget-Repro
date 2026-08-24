from __future__ import annotations

import json

import pytest

from toporetarget.runtime.gpu_preflight import (
    validate_gpu_preflight_payload,
    validate_gpu_preflight_receipt,
)


def _receipt() -> dict[str, object]:
    return {
        "schema_version": "GPURuntimePreflightV1",
        "status": "PASS",
        "execution_context": "HOST_UNSANDBOXED",
        "host": "test-host",
        "driver": "600.1",
        "gpu_names": ["test-gpu"],
        "cuda_visible_devices": "0",
        "torch_version": "2.test",
        "torch_cuda_available": True,
        "torch_device_count": 1,
        "isaac_bootstrap": "PASS",
        "timestamp": "2026-08-23T00:00:00Z",
        "cpu_fallback": False,
    }


def test_gpu_preflight_schema_and_visibility_receipt(tmp_path) -> None:
    path = tmp_path / "gpu_preflight.json"
    path.write_text(json.dumps(_receipt()), encoding="utf-8")

    result = validate_gpu_preflight_receipt(path)

    assert result["cuda_visible_devices"] == "0"
    assert result["torch_cuda_available"] is True
    assert result["torch_device_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "GPU_REQUIRED_UNAVAILABLE", "GPU_REQUIRED_UNAVAILABLE"),
        ("execution_context", "SANDBOX_CONTAINER_DIAGNOSTIC", "NOT_HOST_AUTHORITY"),
        ("torch_cuda_available", False, "TORCH_CUDA_REQUIRED"),
        ("torch_device_count", 0, "TORCH_CUDA_REQUIRED"),
        ("isaac_bootstrap", "FAIL", "ISAAC_BOOTSTRAP_REQUIRED"),
        ("cpu_fallback", True, "CPU_FALLBACK_FORBIDDEN"),
    ],
)
def test_gpu_required_preflight_fails_closed(field: str, value: object, message: str) -> None:
    receipt = _receipt()
    receipt[field] = value

    with pytest.raises(ValueError, match=message):
        validate_gpu_preflight_payload(receipt)
