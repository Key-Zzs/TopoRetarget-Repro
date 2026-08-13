"""CPU contracts for Stage 16-C.5A-R4/C5B/C5C."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.isaaclab_oracle.distributional_replication import (
    R4_DISTRIBUTION_FIELDS,
    DistributionalCandidateReplicatorV1,
    DistributionalReplicationContractV1,
    DistributionPopulationV1,
    NaturalPhysicsDistributionV1,
)
from toporetarget.rl.isaaclab_oracle.horizon_selector import AdaptiveHorizonSelectorV1
from toporetarget.rl.isaaclab_oracle.recovery import Stage16C5R4RecoveryStateMachine
from toporetarget.rl.isaaclab_oracle.replica_manager import ReplicaManagerV1
from toporetarget.rl.isaaclab_oracle.robust_cem import (
    RobustCEMConfigV1,
    RobustMultiHorizonCEMV1,
)
from toporetarget.rl.isaaclab_oracle.robust_metrics import (
    distribution_distances,
    termination_distribution_divergence,
    wilson_confidence_interval,
)
from toporetarget.rl.isaaclab_oracle.robust_oracle import (
    RobustCandidateEvaluatorV2,
    RobustCandidateReplicaV2,
    RobustLexicographicSelectorV1,
    qualify_two_clip_c5c,
)


def _population(*, shift: float = 0.0, phase: str = "contact-onset") -> DistributionPopulationV1:
    base = np.arange(20, dtype=np.float64)[:, None] / 20.0
    fields = OrderedDict(
        (name, np.concatenate((base + shift, 0.5 * base + shift), axis=1))
        for name in R4_DISTRIBUTION_FIELDS
    )
    return DistributionPopulationV1(
        clip="hocap_170105",
        phase=phase,
        reference_index=100,
        fields=fields,
        terminations=("NONE",) * 20,
        successes=(False,) * 20,
    )


def _replica(
    *,
    position: float = 0.01,
    rotation: float = 5.0,
    axis: float = 0.01,
    tracking: float = 0.1,
    success: bool = False,
    reach: bool = False,
    terminal: bool = False,
) -> RobustCandidateReplicaV2:
    return RobustCandidateReplicaV2(
        object_position_error_m=position,
        object_rotation_error_deg=rotation,
        object_axis_error_m=axis,
        tracking_error=tracking,
        contact_stability=0.2,
        smoothness=0.3,
        effort=0.4,
        success=success,
        final_reach=reach,
        terminal_required=terminal,
    )


def _evaluation(candidate_id: int, horizon: int, tracking: float):
    return RobustCandidateEvaluatorV2().evaluate(
        candidate_id=candidate_id,
        horizon=horizon,
        replicas=[_replica(tracking=tracking) for _ in range(4)],
    )


def test_distribution_metrics_include_all_frozen_distances() -> None:
    first = np.arange(40, dtype=np.float64).reshape(20, 2)
    equal = distribution_distances(first, first.copy()).as_dict()
    assert set(equal) == {
        "mean_difference",
        "variance_difference",
        "p95_difference",
        "wasserstein_distance",
        "mmd",
    }
    assert all(value == pytest.approx(0.0, abs=1.0e-12) for value in equal.values())
    shifted = distribution_distances(first, first + 2.0)
    assert shifted.mean_difference > 0.0
    assert shifted.wasserstein_distance > 0.0
    assert shifted.mmd > 0.0
    assert termination_distribution_divergence(["A", "A"], ["B", "B"]) == 1.0
    interval = wilson_confidence_interval(18, 20)
    assert interval[0] < 0.9 < interval[1]


def test_natural_thresholds_freeze_before_candidate_and_gate_shift() -> None:
    contract = DistributionalReplicationContractV1()
    natural = NaturalPhysicsDistributionV1.freeze(_population(), contract)
    qualifier = DistributionalCandidateReplicatorV1(contract)
    assert qualifier.qualify(natural, _population()).passes
    shifted = qualifier.qualify(natural, _population(shift=10.0))
    assert not shifted.passes
    assert all(not row["passes"] for row in shifted.field_results.values())
    assert natural.thresholds.as_dict()["derivation"].startswith("2x_p95")


@pytest.mark.parametrize(
    ("remaining", "expected"),
    [(10, (1, 5, 10)), (9, (1, 5)), (5, (1, 5)), (4, (1,)), (1, (1,)), (0, ())],
)
def test_adaptive_horizon_contraction_has_no_padding(
    remaining: int, expected: tuple[int, ...]
) -> None:
    assert AdaptiveHorizonSelectorV1().select(remaining) == expected
    assert AdaptiveHorizonSelectorV1().as_dict()["padding"] is False


@pytest.mark.parametrize(
    ("population", "replicas", "env_count"), [(32, 4, 384), (48, 4, 576), (32, 8, 768)]
)
def test_persistent_pool_slot_manager_supports_exact_gpu_layouts(
    population: int, replicas: int, env_count: int
) -> None:
    manager = ReplicaManagerV1(
        candidate_env_ids=range(1, env_count + 1),
        population=population,
        replicas=replicas,
    )
    first = manager.permutation(0)
    second = manager.permutation(1)
    assert len(first.logical_to_env) == env_count
    assert first.logical_to_env != second.logical_to_env
    scores = {
        (candidate, horizon): [float(candidate + horizon + replica) for replica in range(replicas)]
        for candidate in range(population)
        for horizon in (1, 5, 10)
    }
    invariance = manager.validate_mapping_invariance(scores)
    assert invariance["mapping_changed"]
    assert invariance["ranking_unchanged"]


def test_robust_selector_obeys_failure_then_cvar_then_physical_p95_order() -> None:
    evaluator = RobustCandidateEvaluatorV2()
    safe = evaluator.evaluate(
        candidate_id=0,
        horizon=5,
        replicas=[_replica(tracking=100.0) for _ in range(4)],
    )
    unsafe = evaluator.evaluate(
        candidate_id=1,
        horizon=1,
        replicas=[_replica(position=0.03, tracking=0.0) for _ in range(4)],
    )
    selected = RobustLexicographicSelectorV1().select([unsafe, safe])
    assert selected.candidate_id == 0
    assert selected.lexical_key()[:3] == (
        safe.failure_probability,
        safe.cvar_gate_violation,
        safe.worst_normalized_gate_margin,
    )
    assert len(selected.lexical_key()) == 12


def test_robust_cem_shapes_updates_elites_and_preserves_std_floor() -> None:
    config = RobustCEMConfigV1()
    cem = RobustMultiHorizonCEMV1(config)
    samples = cem.ask((1, 5, 10))
    assert samples[1].shape == (32, 1, 26)
    assert samples[5].shape == (32, 5, 26)
    assert samples[10].shape == (32, 10, 26)
    assert all(bool((values.abs() <= 1.0).all()) for values in samples.values())
    evaluations = {
        horizon: [_evaluation(candidate, horizon, float(candidate)) for candidate in range(32)]
        for horizon in (1, 5, 10)
    }
    cem.tell(0, evaluations)
    assert len(cem.records) == 3
    assert all(record.elite_candidate_ids == list(range(8)) for record in cem.records)
    for horizon in (1, 5, 10):
        _, std = cem.distribution(horizon)
        assert float(std.min()) >= 0.05
    cem.warm_start_next_step()
    assert cem.convergence_report()["config"]["iterations"] == 3


def test_only_one_cem_scale_upgrade_is_permitted() -> None:
    assert RobustCEMConfigV1(population=48).as_dict()["upgrade"] == "population_48"
    assert RobustCEMConfigV1(replicas=8).as_dict()["upgrade"] == "replicas_8"
    with pytest.raises(ValueError, match="only one upgrade"):
        RobustCEMConfigV1(population=48, replicas=8)


def test_c5c_requires_two_clips_twenty_independent_terminal_episodes() -> None:
    passing = [
        _replica(success=True, reach=True, terminal=True, position=0.01, rotation=5.0, axis=0.01)
        for _ in range(20)
    ]
    result = qualify_two_clip_c5c({"hocap_170105": passing, "hocap_170650": passing})
    assert result["passes"]
    assert result["status"] == "STAGE16C5_PHYSX_ROBUST_ORACLE_VALIDATED"
    failing = list(passing)
    failing[-3:] = [
        _replica(
            success=False,
            reach=False,
            terminal=True,
            position=0.04,
            rotation=20.0,
            axis=0.04,
        )
        for _ in range(3)
    ]
    partial = qualify_two_clip_c5c({"hocap_170105": passing, "hocap_170650": failing})
    assert not partial["passes"]


def test_r4_recovery_allows_only_frozen_replica_and_pool_reductions() -> None:
    machine = Stage16C5R4RecoveryStateMachine()
    machine.transition("NATURAL_DISTRIBUTION", reason="inputs frozen")
    assert machine.record_failure("HIGH_VARIANCE", evidence="baseline") == (
        "UPGRADE_REPLICAS_4_TO_8_ONCE"
    )
    assert machine.record_failure("HIGH_VARIANCE", evidence="retry") == (
        "RETAIN_HIGH_VARIANCE_FAILURE"
    )
    assert machine.record_failure("POOL_OOM", evidence="768") == ("REDUCE_CANDIDATE_SCALE_TO_384")
    assert machine.record_failure("POOL_OOM", evidence="384") == "BLOCKED_POOL_OOM_AT_384"


def test_r4_modules_contain_no_hidden_object_control_or_clip_branches() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = [
        root / "src/toporetarget/rl/isaaclab_oracle/distributional_replication.py",
        root / "src/toporetarget/rl/isaaclab_oracle/replica_manager.py",
        root / "src/toporetarget/rl/isaaclab_oracle/robust_cem.py",
        root / "src/toporetarget/rl/isaaclab_oracle/robust_oracle.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "if clip ==" not in source
    assert "write_root_state_to_sim(" not in source
    assert "apply_external_force" not in source
    assert "teleport" not in source.lower()


def test_r4_runtime_separates_candidate_setup_from_restore_free_formal_eval() -> None:
    root = Path(__file__).resolve().parents[3]
    runtime = (root / "scripts/rl/isaaclab/run_stage16c5_robust_cem.py").read_text(encoding="utf-8")
    assert "raw_control_step(env, execution_actions)" in runtime
    assert '"candidate_state_restore_used": False' in runtime
    assert '"formal_execution_rollout_writes": 0' in runtime
    assert "write_root_state_to_sim(" not in runtime
    assert "apply_external_force" not in runtime
    assert "if clip ==" not in runtime


def test_r4_distribution_worker_freezes_natural_baseline_before_candidate_gate() -> None:
    root = Path(__file__).resolve().parents[3]
    runtime = (root / "scripts/rl/isaaclab/qualify_stage16c5_r4_distributional.py").read_text(
        encoding="utf-8"
    )
    freeze = runtime.index("NaturalPhysicsDistributionV1.freeze")
    qualify = runtime.index("DistributionalCandidateReplicatorV1(contract).qualify")
    assert freeze < qualify
    assert "write_root_state_to_sim(" not in runtime
    assert "if clip ==" not in runtime
