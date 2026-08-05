"""Process/scene allocation contract for deterministic C5 candidate sharding.

No state is copied between shards.  A worker receives an assigned candidate
range and must start a fresh frozen frame-zero rollout; this preserves the R3
ban on object pose writes and makes inter-process IPC reporting explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter

from .topology import balanced_shard_sizes


@dataclass(frozen=True)
class CandidateShardV1:
    shard_id: int
    candidate_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.shard_id < 0 or not self.candidate_ids:
            raise ValueError("candidate shards require a nonnegative ID and at least one candidate")
        if tuple(sorted(self.candidate_ids)) != self.candidate_ids:
            raise ValueError("candidate IDs must be sorted")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique within a shard")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "CandidateShardV1",
            "shard_id": self.shard_id,
            "candidate_ids": list(self.candidate_ids),
            "candidate_count": len(self.candidate_ids),
        }


@dataclass(frozen=True)
class ShardDispatchRecordV1:
    shard_id: int
    candidate_count: int
    latency_s: float
    gpu_memory_peak_mib: float | None
    ipc_overhead_s: float
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.shard_id < 0 or self.candidate_count < 1:
            raise ValueError("invalid dispatched shard identity")
        if self.latency_s < 0.0 or self.ipc_overhead_s < 0.0:
            raise ValueError("dispatch timings cannot be negative")
        if self.gpu_memory_peak_mib is not None and self.gpu_memory_peak_mib < 0.0:
            raise ValueError("GPU memory cannot be negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "ShardDispatchRecordV1",
            "shard_id": self.shard_id,
            "candidate_count": self.candidate_count,
            "latency_s": self.latency_s,
            "gpu_memory_peak_mib": self.gpu_memory_peak_mib,
            "ipc_overhead_s": self.ipc_overhead_s,
            "payload": dict(self.payload),
        }


class ShardedCandidatePoolV1:
    """A deterministic, balanced partition of a bounded candidate pool."""

    def __init__(self, *, candidate_count: int = 96, num_shards: int = 4) -> None:
        if candidate_count not in {1, 32, 96, 144}:
            raise ValueError("C5 candidate count must be one of 1, 32, 96, or 144")
        sizes = balanced_shard_sizes(candidate_count, num_shards)
        cursor = 0
        shards: list[CandidateShardV1] = []
        for shard_id, size in enumerate(sizes):
            ids = tuple(range(cursor, cursor + size))
            shards.append(CandidateShardV1(shard_id=shard_id, candidate_ids=ids))
            cursor += size
        self.candidate_count = candidate_count
        self.num_shards = num_shards
        self.shards = tuple(shards)

    def validate_layout(self) -> dict[str, object]:
        flattened = tuple(candidate for shard in self.shards for candidate in shard.candidate_ids)
        return {
            "version": "ShardedCandidatePoolV1",
            "candidate_count": self.candidate_count,
            "num_shards": self.num_shards,
            "shards": [shard.as_dict() for shard in self.shards],
            "all_candidates_assigned_once": flattened == tuple(range(self.candidate_count)),
            "max_min_shard_size_delta": max(map(len, self._candidate_sets))
            - min(map(len, self._candidate_sets)),
            "cross_shard_ids_unique": len(set(flattened)) == len(flattened),
            "state_transfer": "forbidden_fresh_frame_zero_per_shard",
        }

    @property
    def _candidate_sets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(shard.candidate_ids for shard in self.shards)

    def dispatch(
        self,
        worker: Callable[[CandidateShardV1], Mapping[str, object]],
    ) -> tuple[ShardDispatchRecordV1, ...]:
        """Synchronously dispatch isolated shard workers and preserve timing metadata.

        Runtime code may use a worker that launches an Isaac child process.  It
        returns only serializable measurements; no Tensor/PhysX state travels
        between shards.
        """

        rows: list[ShardDispatchRecordV1] = []
        for shard in self.shards:
            started = perf_counter()
            payload = worker(shard)
            elapsed = perf_counter() - started
            if not isinstance(payload, Mapping):
                raise TypeError("shard worker must return a mapping")
            reported_latency = payload.get("latency_s", elapsed)
            reported_ipc = payload.get("ipc_overhead_s", 0.0)
            reported_memory = payload.get("gpu_memory_peak_mib")
            if not isinstance(reported_latency, (int, float)):
                raise TypeError("shard worker latency must be numeric")
            if not isinstance(reported_ipc, (int, float)):
                raise TypeError("shard worker IPC overhead must be numeric")
            if reported_memory is not None and not isinstance(reported_memory, (int, float)):
                raise TypeError("shard worker GPU memory must be numeric or null")
            rows.append(
                ShardDispatchRecordV1(
                    shard_id=shard.shard_id,
                    candidate_count=len(shard.candidate_ids),
                    latency_s=float(reported_latency),
                    gpu_memory_peak_mib=(
                        None if reported_memory is None else float(reported_memory)
                    ),
                    ipc_overhead_s=float(reported_ipc),
                    payload=payload,
                )
            )
        return tuple(rows)

    @staticmethod
    def aggregate(records: Iterable[ShardDispatchRecordV1]) -> dict[str, object]:
        rows = tuple(records)
        if not rows:
            raise ValueError("cannot aggregate no shard dispatch records")
        total_candidates = sum(row.candidate_count for row in rows)
        total_latency = sum(row.latency_s for row in rows)
        return {
            "version": "ShardedCandidatePoolV1.aggregate",
            "shard_count": len(rows),
            "candidate_count": total_candidates,
            "total_latency_s": total_latency,
            "max_shard_latency_s": max(row.latency_s for row in rows),
            "total_ipc_overhead_s": sum(row.ipc_overhead_s for row in rows),
            "max_gpu_memory_peak_mib": max((row.gpu_memory_peak_mib or 0.0) for row in rows),
            "effective_candidates_per_s": (
                total_candidates / total_latency if total_latency > 0.0 else float("inf")
            ),
            "records": [row.as_dict() for row in rows],
        }


__all__ = ["CandidateShardV1", "ShardDispatchRecordV1", "ShardedCandidatePoolV1"]
