"""Lazy, sequence-scoped adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from toporetarget.data.schema import HOISequence


@dataclass(frozen=True)
class FrameRange:
    """A contiguous half-open frame selection; it never resamples."""

    start: int = 0
    end: int | None = None

    def resolve(self, num_frames: int) -> tuple[int, int]:
        end = num_frames if self.end is None else self.end
        if self.start < 0 or end < 0 or self.start >= end or end > num_frames:
            raise ValueError(f"invalid frame range [{self.start}, {end}) for {num_frames} frames")
        return self.start, end


class HOIDatasetAdapter(ABC):
    """Base class for adapters that only load an explicitly requested sequence."""

    adapter_name = "base"
    adapter_version = "1"

    @abstractmethod
    def describe_sequence(self, sequence: str, **kwargs: Any) -> dict[str, Any]:
        """Return metadata without loading the entire dataset."""

    @abstractmethod
    def load_sequence(
        self,
        sequence: str,
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        """Load one sequence or contiguous clip into the canonical schema."""

    def load_raw_renderable(
        self,
        sequence: str,
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        """Return a renderable source-side sequence without writing a cache."""

        return self.load_sequence(sequence, frame_range=frame_range, **kwargs)

    @abstractmethod
    def canonicalize(self, sequence: HOISequence, **kwargs: Any) -> HOISequence:
        """Convert an adapter result to canonical representation."""

    @abstractmethod
    def supported_fields(self) -> tuple[str, ...]:
        """List fields supported by this adapter."""


__all__ = ["FrameRange", "HOIDatasetAdapter"]
