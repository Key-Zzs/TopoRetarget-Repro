"""Persistent GPU replica allocation and unbiased slot mapping for R4/C5B."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import torch

from .candidate_state import capture_candidate_state, replicate_candidate_state
from .contracts import Stage16C5CandidateStateV1, Stage16C5WriteAuditV1

_SUPPORTED_LAYOUTS = {(32, 4): 384, (48, 4): 576, (32, 8): 768}


@dataclass(frozen=True, order=True)
class LogicalCandidateSlotV1:
    candidate_id: int
    horizon: int
    replica_id: int

    @property
    def identifier(self) -> str:
        return f"candidate{self.candidate_id:03d}_h{self.horizon:02d}_r{self.replica_id:02d}"


@dataclass(frozen=True)
class CandidateSlotPermutationV1:
    iteration: int
    seed: int
    logical_to_env: Mapping[LogicalCandidateSlotV1, int]

    def as_dict(self) -> dict[str, object]:
        rows = [
            {
                "candidate_id": slot.candidate_id,
                "horizon": slot.horizon,
                "replica_id": slot.replica_id,
                "env_id": env_id,
            }
            for slot, env_id in sorted(self.logical_to_env.items())
        ]
        return {
            "version": "CandidateSlotPermutationV1",
            "iteration": self.iteration,
            "seed": self.seed,
            "mapping": rows,
        }


class ReplicaManagerV1:
    """Construct deterministic per-iteration logical-candidate permutations."""

    def __init__(
        self,
        *,
        candidate_env_ids: Sequence[int],
        population: int = 32,
        horizons: Sequence[int] = (1, 5, 10),
        replicas: int = 4,
        seed: int = 20260806,
    ) -> None:
        if tuple(horizons) != (1, 5, 10):
            raise ValueError("R4 freezes candidate-pool horizons at [1, 5, 10]")
        expected = _SUPPORTED_LAYOUTS.get((population, replicas))
        if expected is None:
            raise ValueError("R4 permits only 32x3x4, 48x3x4, or 32x3x8")
        env_ids = tuple(int(value) for value in candidate_env_ids)
        if len(env_ids) != expected or len(set(env_ids)) != len(env_ids) or min(env_ids) < 0:
            raise ValueError(f"candidate env IDs must be {expected} unique nonnegative values")
        self.candidate_env_ids = env_ids
        self.population = population
        self.horizons = tuple(int(value) for value in horizons)
        self.replicas = replicas
        self.seed = seed
        self.logical_slots = tuple(
            LogicalCandidateSlotV1(candidate_id, horizon, replica_id)
            for horizon in self.horizons
            for candidate_id in range(self.population)
            for replica_id in range(self.replicas)
        )

    def permutation(self, iteration: int) -> CandidateSlotPermutationV1:
        if iteration < 0:
            raise ValueError("candidate iteration cannot be negative")
        mixed_seed = self.seed + iteration * 1_000_003
        shuffled = list(self.candidate_env_ids)
        random.Random(mixed_seed).shuffle(shuffled)
        return CandidateSlotPermutationV1(
            iteration=iteration,
            seed=mixed_seed,
            logical_to_env=dict(zip(self.logical_slots, shuffled, strict=True)),
        )

    def env_ids_for_horizon(
        self, permutation: CandidateSlotPermutationV1, horizon: int
    ) -> tuple[int, ...]:
        if horizon not in self.horizons:
            raise ValueError("unknown robust-CEM horizon")
        return tuple(
            permutation.logical_to_env[slot]
            for slot in self.logical_slots
            if slot.horizon == horizon
        )

    def validate_mapping_invariance(
        self, candidate_replica_scores: Mapping[tuple[int, int], Sequence[float]]
    ) -> dict[str, object]:
        """Prove aggregation/ranking is keyed by logical IDs, not physical slots."""

        expected_keys = {
            (candidate, horizon)
            for candidate in range(self.population)
            for horizon in self.horizons
        }
        if set(candidate_replica_scores) != expected_keys:
            raise ValueError("mapping-invariance scores do not cover every logical candidate")
        for values in candidate_replica_scores.values():
            if len(values) != self.replicas:
                raise ValueError("mapping-invariance scores do not match replica count")
        baseline = sorted(
            (
                fmean(float(value) for value in values),
                candidate,
                horizon,
            )
            for (candidate, horizon), values in candidate_replica_scores.items()
        )
        # Exercise two physically different mappings and reconstruct the same
        # logical score table from their env-keyed payloads.
        reconstructed_rankings: list[list[tuple[float, int, int]]] = []
        permutations = (self.permutation(0), self.permutation(1))
        for permutation in permutations:
            env_scores = {
                permutation.logical_to_env[LogicalCandidateSlotV1(candidate, horizon, replica)]: (
                    float(candidate_replica_scores[(candidate, horizon)][replica])
                )
                for candidate, horizon in expected_keys
                for replica in range(self.replicas)
            }
            reconstructed_rankings.append(
                sorted(
                    (
                        fmean(
                            env_scores[
                                permutation.logical_to_env[
                                    LogicalCandidateSlotV1(candidate, horizon, replica)
                                ]
                            ]
                            for replica in range(self.replicas)
                        ),
                        candidate,
                        horizon,
                    )
                    for candidate, horizon in expected_keys
                )
            )
        return {
            "version": "candidate_slot_mapping_invariance_v1",
            "permutation_seeds": [row.seed for row in permutations],
            "mapping_changed": permutations[0].logical_to_env != permutations[1].logical_to_env,
            "ranking_unchanged": all(ranking == baseline for ranking in reconstructed_rankings),
            "candidate_count": len(expected_keys),
        }


class PersistentRobustCandidatePoolV1:
    """One live DirectRLEnv reused for every CEM candidate and iteration."""

    def __init__(
        self,
        env: Any,
        *,
        execution_env_id: int = 0,
        population: int = 32,
        horizons: Sequence[int] = (1, 5, 10),
        replicas: int = 4,
        seed: int = 20260806,
    ) -> None:
        expected = _SUPPORTED_LAYOUTS.get((population, replicas))
        if expected is None:
            raise ValueError("unsupported persistent candidate-pool layout")
        if execution_env_id < 0 or env.num_envs < expected + 1:
            raise ValueError(
                "persistent pool requires one execution env plus all candidate envs: "
                f"required={expected + 1} available={env.num_envs}"
            )
        candidate_ids = tuple(value for value in range(env.num_envs) if value != execution_env_id)[
            :expected
        ]
        self.env = env
        self.execution_env_id = execution_env_id
        self.manager = ReplicaManagerV1(
            candidate_env_ids=candidate_ids,
            population=population,
            horizons=horizons,
            replicas=replicas,
            seed=seed,
        )
        self.write_audit = Stage16C5WriteAuditV1()
        self.env._stage16c5_write_audit = self.write_audit
        self.created_monotonic_s = time.monotonic()
        self.dispatch_count = 0
        self._last_dispatch_latency_s = 0.0

    @property
    def candidate_ids(self) -> torch.Tensor:
        return torch.as_tensor(
            self.manager.candidate_env_ids, dtype=torch.long, device=self.env.device
        )

    @property
    def execution_ids(self) -> torch.Tensor:
        return torch.tensor([self.execution_env_id], dtype=torch.long, device=self.env.device)

    def capture_execution_state(self) -> Stage16C5CandidateStateV1:
        return capture_candidate_state(self.env, self.execution_ids)

    def dispatch_execution_state(
        self, state: Stage16C5CandidateStateV1 | None = None
    ) -> Stage16C5CandidateStateV1:
        snapshot = state or self.capture_execution_state()
        started = time.perf_counter()
        replicate_candidate_state(
            self.env, snapshot, self.candidate_ids, write_audit=self.write_audit
        )
        if str(self.env.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.env.device))
        self._last_dispatch_latency_s = time.perf_counter() - started
        self.dispatch_count += 1
        return snapshot

    def aggregate_by_logical_candidate(
        self,
        values_by_env: torch.Tensor,
        permutation: CandidateSlotPermutationV1,
    ) -> tuple[dict[tuple[int, int], torch.Tensor], float]:
        """Gather physical-env results into [replica, ...] logical populations."""

        if values_by_env.shape[0] != self.env.num_envs:
            raise ValueError("candidate aggregation tensor must cover the live environment")
        started = time.perf_counter()
        result: dict[tuple[int, int], torch.Tensor] = {}
        for horizon in self.manager.horizons:
            for candidate in range(self.manager.population):
                ids = torch.tensor(
                    [
                        permutation.logical_to_env[
                            LogicalCandidateSlotV1(candidate, horizon, replica)
                        ]
                        for replica in range(self.manager.replicas)
                    ],
                    dtype=torch.long,
                    device=values_by_env.device,
                )
                result[(candidate, horizon)] = values_by_env.index_select(0, ids)
        if str(values_by_env.device).startswith("cuda"):
            torch.cuda.synchronize(values_by_env.device)
        latency = time.perf_counter() - started
        return result, latency

    def benchmark_record(
        self,
        *,
        rollout_latency_s: float,
        reset_latency_s: float,
        aggregation_latency_s: float,
        gpu_memory_mib: float | None,
        control_steps: int,
    ) -> dict[str, object]:
        elapsed = max(rollout_latency_s, 1.0e-12)
        return {
            "version": "PersistentRobustCandidatePoolV1Benchmark",
            "population": self.manager.population,
            "horizons": list(self.manager.horizons),
            "replicas": self.manager.replicas,
            "candidate_env_count": len(self.manager.candidate_env_ids),
            "live_env_count": self.env.num_envs,
            "persistent_environment": True,
            "isaac_recreated_per_candidate": False,
            "gpu_memory_mib": gpu_memory_mib,
            "rollout_latency_s": rollout_latency_s,
            "rollout_control_steps_per_s": control_steps / elapsed,
            "reset_latency_s": reset_latency_s,
            "state_dispatch_latency_s": self._last_dispatch_latency_s,
            "aggregation_latency_s": aggregation_latency_s,
            "dispatch_count": self.dispatch_count,
            "uptime_s": time.monotonic() - self.created_monotonic_s,
        }


__all__ = [
    "CandidateSlotPermutationV1",
    "LogicalCandidateSlotV1",
    "PersistentRobustCandidatePoolV1",
    "ReplicaManagerV1",
]
