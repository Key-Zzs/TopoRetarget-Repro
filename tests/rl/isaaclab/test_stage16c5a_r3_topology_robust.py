"""CPU-only contracts for the Stage 16-C.5A-R3 topology/robust transition."""

from __future__ import annotations

from pathlib import Path

import pytest

from toporetarget.rl.isaaclab_oracle.robust import (
    RobustCandidateEvaluatorV1,
    RobustCandidateSelector,
    RobustOracleContractV1,
    RobustReplicaResultV1,
    qualify_c5c_independent_replicas,
    upper_cvar,
)
from toporetarget.rl.isaaclab_oracle.sharded_pool import ShardedCandidatePoolV1
from toporetarget.rl.isaaclab_oracle.topology import (
    balanced_shard_sizes,
    classify_contact_topology,
    r3_topology_matrix,
)


def _row(
    identifier: str,
    *,
    passes: bool,
    raw_stable: bool = False,
    derived_stable: bool = False,
) -> dict[str, object]:
    return {
        "identifier": identifier,
        "passes_frozen_gate": passes,
        "raw_state_stable": raw_stable,
        "derived_state_stable": derived_stable,
    }


def _replica(
    *,
    cost: float,
    position: float = 0.01,
    rotation: float = 5.0,
    axis: float = 0.01,
    success: bool = True,
    reach: bool = True,
    stability: float = 0.1,
) -> RobustReplicaResultV1:
    return RobustReplicaResultV1(
        cost=cost,
        object_position_error_m=position,
        object_rotation_error_deg=rotation,
        object_axis_error_m=axis,
        success=success,
        final_reach=reach,
        contact_stability_penalty=stability,
        action_smoothness=0.2,
        effort=0.3,
        termination_reason="time_limit",
    )


def test_r3_matrix_covers_exact_topologies_and_balanced_shards() -> None:
    matrix = r3_topology_matrix()
    assert tuple(matrix) == ("T0", "T1", "T2", "T3", "T4", "T5")
    assert matrix["T0"][0].scene_env_count == 1
    assert matrix["T1"][0].active_contact_count == 1
    assert matrix["T2"][0].active_contact_count == 33
    assert matrix["T3"][0].schedule == "staggered"
    assert [row.shard_sizes for row in matrix["T4"]] == [(33,), (16, 17), (8, 8, 8, 9)]
    assert [row.shard_sizes for row in matrix["T5"]] == [(96,), (24, 24, 24, 24), (12,) * 8]
    assert balanced_shard_sizes(33, 4) == (9, 8, 8, 8)
    with pytest.raises(ValueError, match="between one and total"):
        balanced_shard_sizes(2, 3)


def test_topology_classification_distinguishes_sharding_and_metric_failure() -> None:
    batch = classify_contact_topology(
        [
            _row("T0_single", passes=True, raw_stable=True, derived_stable=True),
            _row("T2_all_contact", passes=False),
            _row("T4_2x16_17", passes=True, raw_stable=True, derived_stable=True),
        ]
    )
    assert batch["classification"] == "SINGLE_SCENE_CONTACT_BATCHING_FAILURE"
    solver = classify_contact_topology(
        [
            _row("T0_single", passes=True, raw_stable=True, derived_stable=True),
            _row("T2_all_contact", passes=False),
            _row("T4_2x16_17", passes=False),
        ]
    )
    assert solver["classification"] == "TRUE_CONTACT_SOLVER_NONDETERMINISM"
    metric = classify_contact_topology(
        [
            _row("T0_single", passes=True, raw_stable=True, derived_stable=True),
            _row("T2_all_contact", passes=False, raw_stable=True, derived_stable=False),
            _row("T4_2x16_17", passes=False),
        ]
    )
    assert metric["classification"] == "HARNESS_METRIC_FAILURE"


def test_topology_classification_reports_the_safe_partial_shard_topology() -> None:
    result = classify_contact_topology(
        [
            _row("T0_single", passes=True, raw_stable=True, derived_stable=True),
            _row("T2_all_contact", passes=False),
            _row("T4_2x16_17", passes=True, raw_stable=True, derived_stable=True),
            _row("T4_4x8_8_8_9", passes=False),
        ]
    )
    assert result["classification"] == "SINGLE_SCENE_CONTACT_BATCHING_FAILURE"
    assert result["all_supplied_sharded_topologies_pass"] is False
    assert result["passing_sharded_topology_identifiers"] == ["T4_2x16_17"]


def test_sharded_candidate_pool_allocates_without_cross_shard_state_transfer() -> None:
    pool = ShardedCandidatePoolV1(candidate_count=96, num_shards=4)
    layout = pool.validate_layout()
    assert layout["all_candidates_assigned_once"]
    assert layout["cross_shard_ids_unique"]
    assert layout["max_min_shard_size_delta"] == 0
    assert layout["state_transfer"] == "forbidden_fresh_frame_zero_per_shard"
    records = pool.dispatch(
        lambda shard: {
            "latency_s": 0.1 + shard.shard_id,
            "gpu_memory_peak_mib": 100.0 + shard.shard_id,
            "ipc_overhead_s": 0.01,
        }
    )
    aggregate = pool.aggregate(records)
    assert aggregate["candidate_count"] == 96
    assert aggregate["shard_count"] == 4
    assert aggregate["max_gpu_memory_peak_mib"] == 103.0


def test_robust_statistics_and_lexical_selector_are_deterministic() -> None:
    contract = RobustOracleContractV1(replica_count=4)
    evaluator = RobustCandidateEvaluatorV1(contract)
    unsafe = evaluator.evaluate(
        "candidate-z",
        [_replica(cost=1.0, success=False, reach=False) for _ in range(4)],
    )
    safe = evaluator.evaluate(
        "candidate-a", [_replica(cost=value) for value in (1.0, 2.0, 3.0, 4.0)]
    )
    assert safe.mean_cost == 2.5
    assert safe.cvar_cost == upper_cvar([1.0, 2.0, 3.0, 4.0], 0.8)
    assert RobustCandidateSelector().select([unsafe, safe]).candidate_id == "candidate-a"
    equal_a = evaluator.evaluate("a", [_replica(cost=1.0) for _ in range(4)])
    equal_b = evaluator.evaluate("b", [_replica(cost=1.0) for _ in range(4)])
    assert RobustCandidateSelector().select([equal_b, equal_a]).candidate_id == "a"


def test_c5c_qualification_requires_twenty_independent_replica_results() -> None:
    passing = qualify_c5c_independent_replicas([_replica(cost=1.0) for _ in range(20)])
    assert passing["passes_frozen_gate"] is True
    assert passing["success_rate"] == 1.0
    failed = qualify_c5c_independent_replicas(
        [_replica(cost=1.0, success=False, reach=False) for _ in range(3)]
        + [_replica(cost=1.0) for _ in range(17)]
    )
    assert failed["passes_frozen_gate"] is False
    with pytest.raises(ValueError, match="exactly 20"):
        qualify_c5c_independent_replicas([_replica(cost=1.0) for _ in range(19)])


def test_r3_source_forbids_snapshot_or_object_rollout_control() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "scripts/rl/isaaclab/diagnose_stage16c5_contact_topology.py").read_text(
        encoding="utf-8"
    )
    assert 'candidate_state_restore_used": False' in source
    assert 'object_pose_write_used": False' in source
    assert 'hidden_force_or_teleport_used": False' in source
    assert "replicate_candidate_state(" not in source
    assert "write_root_state_to_sim(" not in source


def test_robust_runtime_replays_fresh_frame_zero_rollouts_without_state_restore() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "scripts/rl/isaaclab/run_stage16c5_robust_oracle.py").read_text(
        encoding="utf-8"
    )
    assert "one_environment_fresh_frame_zero_per_replica" in source
    assert "reset_frozen_clip_frame_zero" in source
    assert 'candidate_state_restore_used": False' in source
    assert 'object_pose_write_used": False' in source
    assert 'hidden_force_or_teleport_used": False' in source
    assert "replicate_candidate_state(" not in source
    assert "restore_candidate_state(" not in source
    assert "write_root_state_to_sim(" not in source
