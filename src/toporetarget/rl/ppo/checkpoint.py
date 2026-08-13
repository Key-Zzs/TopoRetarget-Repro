"""Atomic, hash-carrying PPO checkpoints and deterministic RNG restoration."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _cpu_byte_rng_state(value: Any, *, field: str) -> torch.Tensor:
    """Recover a ByteTensor state after a checkpoint was mapped onto CUDA."""

    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise TypeError(f"RNG_STATE_{field}_MUST_BE_TORCH_BYTETENSOR")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch_cpu"], field="TORCH_CPU"))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("RNG_STATE_TORCH_CUDA_MUST_BE_SEQUENCE")
        torch.cuda.set_rng_state_all(
            [_cpu_byte_rng_state(value, field="TORCH_CUDA") for value in cuda_states]
        )


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


__all__ = ["load_checkpoint", "restore_rng_state", "rng_state", "save_checkpoint"]
