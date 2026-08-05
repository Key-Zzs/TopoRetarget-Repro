"""Pure-Python contracts for the Stage 16-C.5A-R3 contact-topology gate.

This module intentionally contains no Isaac imports.  It freezes the allowed
topology matrix and makes the classification rule testable without turning a
numerical diagnostic into a solver or tolerance mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_CLASSIFICATIONS = {
    "SINGLE_SCENE_CONTACT_BATCHING_FAILURE",
    "TRUE_CONTACT_SOLVER_NONDETERMINISM",
    "HARNESS_METRIC_FAILURE",
}


def balanced_shard_sizes(total: int, shard_count: int) -> tuple[int, ...]:
    """Return deterministic, nonempty shard sizes whose sum is ``total``."""

    if total < 1:
        raise ValueError("candidate total must be positive")
    if shard_count < 1 or shard_count > total:
        raise ValueError("shard count must be between one and total candidates")
    quotient, remainder = divmod(total, shard_count)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(shard_count))


@dataclass(frozen=True)
class ContactTopologyExperimentV1:
    """One frozen R3 topology cell.

    ``active_contact_count`` is deliberately separate from ``scene_env_count``:
    T1 has one real contact trajectory while 32 environments remain at their
    reset target as no-contact scheduling dummies.  This is diagnostic-only and
    is never a candidate rollout or a replacement for a physical gate.
    """

    identifier: str
    scene_env_count: int
    active_contact_count: int
    shard_sizes: tuple[int, ...]
    schedule: str
    trials: int = 20

    def __post_init__(self) -> None:
        if not self.identifier.startswith("T"):
            raise ValueError("topology experiment identifier must begin with T")
        if self.scene_env_count < 1:
            raise ValueError("scene environment count must be positive")
        if not 1 <= self.active_contact_count <= self.scene_env_count:
            raise ValueError("active contact count must be within scene population")
        if not self.shard_sizes or any(size < 1 for size in self.shard_sizes):
            raise ValueError("topology shards must be nonempty")
        if sum(self.shard_sizes) != self.scene_env_count:
            raise ValueError("topology shard sizes must sum to scene population")
        if self.schedule not in {"all_simultaneous", "one_active", "staggered"}:
            raise ValueError("unknown contact topology schedule")
        if self.trials != 20:
            raise ValueError("Stage16 C5A-R3 topology gate requires exactly 20 trials")
        if (
            self.schedule == "all_simultaneous"
            and self.active_contact_count != self.scene_env_count
        ):
            raise ValueError("simultaneous topology requires every environment to be active")
        if self.schedule == "one_active" and self.active_contact_count != 1:
            raise ValueError("one-active topology requires exactly one contact environment")
        if self.schedule == "staggered" and self.active_contact_count != self.scene_env_count:
            raise ValueError("staggered topology schedules every environment")

    @property
    def process_count(self) -> int:
        return len(self.shard_sizes)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "ContactTopologyExperimentV1",
            "identifier": self.identifier,
            "scene_env_count": self.scene_env_count,
            "active_contact_count": self.active_contact_count,
            "shard_sizes": list(self.shard_sizes),
            "schedule": self.schedule,
            "trials": self.trials,
        }


def r3_topology_matrix() -> dict[str, tuple[ContactTopologyExperimentV1, ...]]:
    """Return the exact T0--T5 matrix required by the R3 objective."""

    return {
        "T0": (ContactTopologyExperimentV1("T0_single", 1, 1, (1,), "all_simultaneous"),),
        "T1": (ContactTopologyExperimentV1("T1_one_active", 33, 1, (33,), "one_active"),),
        "T2": (ContactTopologyExperimentV1("T2_all_contact", 33, 33, (33,), "all_simultaneous"),),
        "T3": (ContactTopologyExperimentV1("T3_staggered", 33, 33, (33,), "staggered"),),
        "T4": (
            ContactTopologyExperimentV1("T4_1x33", 33, 33, (33,), "all_simultaneous"),
            ContactTopologyExperimentV1("T4_2x16_17", 33, 33, (16, 17), "all_simultaneous"),
            ContactTopologyExperimentV1("T4_4x8_8_8_9", 33, 33, (8, 8, 8, 9), "all_simultaneous"),
        ),
        "T5": (
            ContactTopologyExperimentV1("T5_1x96", 96, 96, (96,), "all_simultaneous"),
            ContactTopologyExperimentV1("T5_4x24", 96, 96, (24, 24, 24, 24), "all_simultaneous"),
            ContactTopologyExperimentV1(
                "T5_8x12", 96, 96, (12, 12, 12, 12, 12, 12, 12, 12), "all_simultaneous"
            ),
        ),
    }


def _row_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"topology result lacks boolean {key!r}")
    return value


def classify_contact_topology(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Classify R3 evidence without creating a new numerical acceptance limit.

    Raw-state equality is a byte-level/fingerprint result produced by workers;
    a derived-only mismatch is therefore a harness failure, not a reason to
    adjust the frozen replication tolerance.  Deterministic recovery needs a
    passing T0 plus a pass from at least one genuinely sharded T4/T5 topology.
    Other failed shard sizes define the maximum safe batch size; they do not
    turn a successful smaller independent scene into solver nondeterminism.
    """

    by_identifier: dict[str, Mapping[str, object]] = {}
    for row in rows:
        identifier = row.get("identifier")
        if not isinstance(identifier, str):
            raise ValueError("topology result lacks identifier")
        if identifier in by_identifier:
            raise ValueError(f"duplicate topology result: {identifier}")
        by_identifier[identifier] = row
    if "T0_single" not in by_identifier or "T2_all_contact" not in by_identifier:
        raise ValueError("R3 classification requires T0 and T2 results")

    t0_pass = _row_bool(by_identifier["T0_single"], "passes_frozen_gate")
    t2 = by_identifier["T2_all_contact"]
    raw_stable = _row_bool(t2, "raw_state_stable")
    derived_stable = _row_bool(t2, "derived_state_stable")
    sharded_rows: list[Mapping[str, object]] = []
    all_sharded_pass = False
    passing_sharded_identifiers: tuple[str, ...] = ()
    if raw_stable and not derived_stable:
        classification = "HARNESS_METRIC_FAILURE"
        rationale = "T2 raw state is byte-identical while a derived metric diverges"
    else:
        sharded_rows = [
            row
            for identifier, row in by_identifier.items()
            if identifier.startswith(("T4_2x", "T4_4x", "T5_4x", "T5_8x"))
        ]
        if not sharded_rows:
            raise ValueError("R3 classification requires at least one sharded topology result")
        passing_sharded_identifiers = tuple(
            str(row["identifier"]) for row in sharded_rows if _row_bool(row, "passes_frozen_gate")
        )
        all_sharded_pass = len(passing_sharded_identifiers) == len(sharded_rows)
        if t0_pass and passing_sharded_identifiers:
            classification = "SINGLE_SCENE_CONTACT_BATCHING_FAILURE"
            rationale = "T0 and at least one smaller independent scene batch pass"
        else:
            classification = "TRUE_CONTACT_SOLVER_NONDETERMINISM"
            rationale = "T0 or every supplied sharded topology remains outside the frozen gate"
    if classification not in _CLASSIFICATIONS:
        raise AssertionError("unreachable topology classification")
    return {
        "version": "stage16c5a_r3_contact_topology_classification_v1",
        "classification": classification,
        "rationale": rationale,
        "t0_passes_frozen_gate": t0_pass,
        "t2_raw_state_stable": raw_stable,
        "t2_derived_state_stable": derived_stable,
        "sharded_topology_count": len(sharded_rows),
        "all_supplied_sharded_topologies_pass": all_sharded_pass,
        "passing_sharded_topology_identifiers": (
            list(passing_sharded_identifiers) if not (raw_stable and not derived_stable) else []
        ),
        "result_count": len(rows),
    }


__all__ = [
    "ContactTopologyExperimentV1",
    "balanced_shard_sizes",
    "classify_contact_topology",
    "r3_topology_matrix",
]
