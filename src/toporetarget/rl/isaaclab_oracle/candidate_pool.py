"""Isolated candidate-ID allocation for the bounded PhysX Oracle pool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .candidate_state import capture_candidate_state, replicate_candidate_state
from .contracts import Stage16C5CandidateStateV1, Stage16C5WriteAuditV1


@dataclass(frozen=True)
class CandidatePoolLayoutV1:
    execution_env_ids: tuple[int, ...]
    candidate_env_ids: tuple[int, ...]
    guard_env_ids: tuple[int, ...]
    horizons: tuple[int, ...]
    population_per_horizon: int

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_env_ids)

    @property
    def total_env_count(self) -> int:
        return len(self.execution_env_ids) + len(self.candidate_env_ids) + len(self.guard_env_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "PhysXOracleCandidatePoolV1",
            "execution_env_ids": list(self.execution_env_ids),
            "candidate_env_ids": list(self.candidate_env_ids),
            "guard_env_ids": list(self.guard_env_ids),
            "candidate_count": self.candidate_count,
            "total_env_count": self.total_env_count,
            "horizons": list(self.horizons),
            "population_per_horizon": self.population_per_horizon,
        }


class PhysXOracleCandidatePoolV1:
    """Candidate pool over explicitly disjoint DirectRLEnv IDs.

    The pool does not implement CEM, scoring, or horizon selection.  It only
    allocates IDs, snapshots execution state, and restores candidates.
    """

    def __init__(
        self,
        env: Any,
        *,
        execution_env_ids: Sequence[int] = (0,),
        candidate_count: int = 96,
        horizons: Sequence[int] = (1, 5, 10),
        population_per_horizon: int | None = None,
        guard_env_count: int = 0,
    ) -> None:
        if candidate_count not in {1, 32, 96, 144}:
            raise ValueError("C.5A accepts O0 capacities 1, 32, 96, or 144 only")
        execution = tuple(int(value) for value in execution_env_ids)
        if not execution or len(set(execution)) != len(execution) or min(execution) < 0:
            raise ValueError("execution environment IDs must be nonempty and unique")
        if population_per_horizon is None:
            if candidate_count == 96:
                population_per_horizon = 32
            elif candidate_count == 144:
                population_per_horizon = 48
            else:
                population_per_horizon = candidate_count
        candidate_start = max(execution) + 1
        candidate = tuple(range(candidate_start, candidate_start + candidate_count))
        guard = tuple(range(candidate[-1] + 1, candidate[-1] + 1 + guard_env_count))
        layout = CandidatePoolLayoutV1(
            execution_env_ids=execution,
            candidate_env_ids=candidate,
            guard_env_ids=guard,
            horizons=tuple(int(value) for value in horizons),
            population_per_horizon=int(population_per_horizon),
        )
        if layout.total_env_count > env.num_envs:
            raise ValueError(
                "candidate pool exceeds configured DirectRLEnv size: "
                f"required={layout.total_env_count}, available={env.num_envs}"
            )
        if candidate_count in {96, 144} and candidate_count != len(layout.horizons) * int(
            population_per_horizon
        ):
            raise ValueError("C.5A candidate count must equal horizons × population")
        self.env = env
        self.layout = layout
        self.write_audit = Stage16C5WriteAuditV1()
        self.env._stage16c5_write_audit = self.write_audit

    @property
    def execution_ids(self) -> torch.Tensor:
        return torch.tensor(self.layout.execution_env_ids, dtype=torch.long, device=self.env.device)

    @property
    def candidate_ids(self) -> torch.Tensor:
        return torch.tensor(self.layout.candidate_env_ids, dtype=torch.long, device=self.env.device)

    def capture_execution_state(self) -> Stage16C5CandidateStateV1:
        return capture_candidate_state(self.env, self.execution_ids)

    def replicate_execution_state(
        self, state: Stage16C5CandidateStateV1 | None = None
    ) -> Stage16C5CandidateStateV1:
        snapshot = state if state is not None else self.capture_execution_state()
        replicate_candidate_state(
            self.env,
            snapshot,
            self.candidate_ids,
            write_audit=self.write_audit,
        )
        return snapshot

    def validate_layout(self) -> dict[str, object]:
        origins = self.env.scene.env_origins
        all_ids = (
            *self.layout.execution_env_ids,
            *self.layout.candidate_env_ids,
            *self.layout.guard_env_ids,
        )
        selected_origins = origins[torch.tensor(all_ids, dtype=torch.long, device=self.env.device)]
        unique_origins = torch.unique(selected_origins, dim=0).shape[0]
        return {
            **self.layout.as_dict(),
            "cuda_device": str(self.env.device),
            "unique_env_ids": len(set(all_ids)) == len(all_ids),
            "unique_origins": int(unique_origins) == len(all_ids),
            "candidate_execution_disjoint": not set(self.layout.execution_env_ids).intersection(
                self.layout.candidate_env_ids
            ),
            "candidate_guard_disjoint": not set(self.layout.candidate_env_ids).intersection(
                self.layout.guard_env_ids
            ),
        }

    def reset_candidates(self) -> None:
        """Reset candidate IDs only; this is a reset-category write, never a rollout write."""

        self.env._reset_idx(self.candidate_ids)
        self.write_audit.record(
            category="reset",
            operation="candidate_subset_reset",
            env_ids=self.candidate_ids,
            tensor_names=["candidate_reset"],
        )


__all__ = ["CandidatePoolLayoutV1", "PhysXOracleCandidatePoolV1"]
