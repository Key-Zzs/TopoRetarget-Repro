"""Stage 2B compatibility facade for the formal Stage 5 GRAB adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from toporetarget.data.adapters.grab import (
    GrabAdapterError,
    GrabDatasetAdapter,
    GrabLoadOptions,
)
from toporetarget.data.indexes.grab import resolve_grab_dataset_root


def resolve_mano_model_root(explicit_root: Path | None = None) -> Path:
    """Resolve MANO root in the historical Stage 2B order."""

    candidate = explicit_root
    if candidate is None and os.environ.get("MANO_MODEL_ROOT"):
        candidate = Path(os.environ["MANO_MODEL_ROOT"]).expanduser()
    if candidate is None:
        config = Path(__file__).resolve().parents[4] / ".local" / "config.yaml"
        if config.is_file():
            try:
                import yaml

                value = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
                raw = value.get("mano_model_root") if isinstance(value, dict) else None
                candidate = Path(raw).expanduser() if isinstance(raw, str) and raw else None
            except (ImportError, OSError, ValueError):
                candidate = None
    if candidate is None:
        raise GrabAdapterError(
            "MANO model root is required for hand reconstruction; pass --mano-model-root, "
            "set MANO_MODEL_ROOT, or set mano_model_root in .local/config.yaml"
        )
    return candidate


def resolve_adapter_grab_root(sequence_path: Path, explicit_root: Path | None = None) -> Path:
    return resolve_grab_dataset_root(explicit_root, sequence_path=sequence_path)


class GrabInspectionAdapter(GrabDatasetAdapter):
    """Deprecated Stage 2B API, mapped to one formal single-hand load."""

    adapter_name = "grab_inspect"
    adapter_version = "1"

    def __init__(self, *, sequence_path: str | Path, hand: str = "right", **kwargs: Any) -> None:
        if hand not in {"left", "right"}:
            raise GrabAdapterError("hand must be left or right")
        self.hand_side = hand
        backend = kwargs.pop("backend", None)
        super().__init__(
            sequence_path=sequence_path,
            backend=backend,
            options=GrabLoadOptions(
                hands=hand,
                include_table=False,
                contact_mode="none",
                include_mediapipe21=False,
            ),
            **kwargs,
        )

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        result = super().describe_sequence(sequence, **kwargs)
        result["hand"] = self.hand_side
        result["selected_hand"] = self.hand_side
        result["hand_vtemp"] = result["mano_parameters"][self.hand_side]["vtemp_path"]
        return result

    def load_sequence(self, sequence: str = "", **kwargs: Any):
        result = super().load_sequence(sequence, **kwargs)
        hand = result.hands[0]
        hand.hand_id = f"hand_{self.hand_side[0]}"
        hand.metadata["selected_hand"] = self.hand_side
        hand.metadata["no_mediapipe_mapping"] = True
        result.metadata.sequence_id = Path(self.sequence_path or "").stem
        result.metadata.provenance.source_sequence = result.metadata.sequence_id
        return result


__all__ = [
    "GrabAdapterError",
    "GrabInspectionAdapter",
    "resolve_adapter_grab_root",
    "resolve_mano_model_root",
]
