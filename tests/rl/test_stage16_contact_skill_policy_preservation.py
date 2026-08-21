from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.evaluate_stage16_contact_collapse import _evaluation_milestones
from scripts.evaluation.finalize_stage16_contact_preserving_full_c0 import (
    _candidate_episode_count,
    _core_curve,
    _expected_c0_samples,
    _historical_row,
    _markdown,
)
from scripts.rl.isaaclab.recover_stage16_contact_preserving_c0_ppo_health import (
    _candidate_trainer_from_predecessor,
    _checkpoint_cumulative_samples,
)
from scripts.rl.isaaclab.run_stage16_contact_preserving_full_c0 import (
    _link_or_verify_receipt,
    _load_verified_evaluation,
    _receipt_safe_training_metric,
    _rollout_length_for_remaining_samples,
    _sha256,
)
from toporetarget.rl.ppo.policy_preservation import _configure_actor_lr_scale, validate_exact_batch
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer


def _batch() -> dict[str, object]:
    steps, environments = 2, 32
    return {
        "schema_version": "Stage16ContactCollapseExactPPOBatchV1",
        "rollout_steps": steps,
        "num_envs": environments,
        "observations": torch.zeros(steps, environments, 764),
        "actions": torch.zeros(steps, environments, 26),
        "old_log_probs": torch.zeros(steps, environments),
        "rewards": torch.zeros(steps, environments),
        "dones": torch.zeros(steps, environments),
        "values": torch.zeros(steps, environments),
        "returns": torch.zeros(steps, environments),
        "advantages": torch.zeros(steps, environments),
        "last_value": torch.zeros(environments),
        "reference_indices": torch.zeros(steps, environments, dtype=torch.long),
        "rng_before_optimizer_update": {},
    }


def test_exact_batch_validation_records_frozen_minibatch_semantics() -> None:
    result = validate_exact_batch(_batch())

    assert result["sample_count"] == 64
    assert result["observation_shape"] == [2, 32, 764]
    assert result["minibatch_order"] == "torch.Generator(device).manual_seed(0); randperm per epoch"


def test_exact_batch_validation_rejects_missing_rng_authority() -> None:
    batch = _batch()
    batch.pop("rng_before_optimizer_update")

    try:
        validate_exact_batch(batch)
    except ValueError as error:
        assert str(error) == "POLICY_PRESERVATION_RNG_MISSING"
    else:
        raise AssertionError("missing RNG authority was accepted")


def test_actor_lr_shadow_contract_keeps_critic_lr_at_baseline() -> None:
    trainer = PPO26DTrainer(observation_dim=764, device="cpu")

    result = _configure_actor_lr_scale(trainer, 0.5)

    assert result["actor_critic_shared_parameters"] is False
    assert result["effective_actor_lr"] == 5.0e-5
    assert result["critic_lr"] == 1.0e-4
    assert len(trainer.trainer.optimizer.param_groups) == 2


def test_train_receipt_excludes_only_in_memory_policy_tensor() -> None:
    policy_observation = torch.zeros(1024, 764)
    metric = {"samples": 40960, "last_policy_observation": policy_observation}

    receipt = _receipt_safe_training_metric(metric)

    assert receipt == {"samples": 40960}
    assert metric["last_policy_observation"] is policy_observation


def test_recovery_reuses_matching_verified_evaluation(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"candidate")
    output = tmp_path / "evaluation"
    output.mkdir()
    snapshot = {
        "label": "C0_U2",
        "update": 2,
        "samples": 81920,
        "checkpoint_sha256": _sha256(checkpoint),
        "persistent_grasp_episodes": 10,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps({"snapshots": [snapshot]}), encoding="utf-8"
    )
    (output / "evaluation_contract.json").write_text("{}", encoding="utf-8")

    recovered = _load_verified_evaluation(
        checkpoint=checkpoint,
        output=output,
        label="C0_U2",
        update=2,
        samples=81920,
    )

    assert recovered == snapshot
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == snapshot


def test_single_zero_contact_candidate_has_no_cross_checkpoint_milestone_baseline() -> None:
    rows = [
        {
            "label": "C0_U17",
            "update": 17,
            "samples": 696320,
            "episodes": 10,
            "contact_episodes": 0,
        }
    ]

    assert _evaluation_milestones(rows, source_only=False) == {}


def test_closeout_keeps_no_contact_force_p95_not_applicable() -> None:
    curve = _core_curve(
        [
            {
                "update": "17",
                "samples": "696320",
                "persistent_grasp_episodes": "0",
                "lift_episodes": "0",
                "contact_fraction": "0.0",
                "force_p95_n": "",
                "tip_recall": "0.0",
                "lift_dz_m": "0.0",
            }
        ]
    )
    metric = {
        "persistent_grasp_episodes": 0,
        "lift_episodes": 0,
        "episodes": 10,
        "contact_fraction": 0.0,
        "active_contact_force_p95_n": None,
        "object_lift_dz_mean": 0.0,
    }

    rendered = _markdown(
        source=metric,
        endpoint=metric,
        comparison=[],
        curve=curve,
        classification="CANDIDATE_REGRESSION",
        decision={
            "DID_CANDIDATE_AVOID_ORIGINAL_U26_COLLAPSE": "NO",
            "DID_COLLAPSE_REAPPEAR_LATER": "NO",
        },
        milestones={},
        best={"update": 17, "samples": 696320, "checkpoint_sha256": "x"},
    )

    assert curve[0]["force_p95_n"] is None
    assert "| 17 | 696320 | 0/10 | 0/10 | 0.000000 | N/A |" in rendered


def test_c0_terminal_update_uses_the_exact_remaining_24_rollout_steps() -> None:
    assert (
        _rollout_length_for_remaining_samples(
            remaining_samples=40_960, num_envs=1024, frozen_rollout_length=40
        )
        == 40
    )
    assert (
        _rollout_length_for_remaining_samples(
            remaining_samples=1_048_576 - 25 * 40_960,
            num_envs=1024,
            frozen_rollout_length=40,
        )
        == 24
    )
    assert _expected_c0_samples(25) == 1_024_000
    assert _expected_c0_samples(26) == 1_048_576


def test_closeout_uses_fixed_eval10_count_for_progression_rows() -> None:
    assert _candidate_episode_count(point="U25-equivalent", candidate_metrics={}) == 10
    assert _candidate_episode_count(point="Endpoint", candidate_metrics={"episodes": 20}) == 20


def test_closeout_accepts_the_explicit_c0_continuation_label_alias(tmp_path: Path) -> None:
    historical_path = tmp_path / "grasp_vs_update.csv"
    historical_path.write_text(
        "stage,label,update,samples\nC0,C0_U25,25,1024000\n",
        encoding="utf-8",
    )

    historical = _historical_row("U25", historical_path=historical_path)

    assert historical["label"] == "C0_U25"
    assert historical["update"] == "25"


def test_endpoint_receipt_link_is_idempotent_only_for_the_same_target(tmp_path) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"checkpoint")
    link = tmp_path / "endpoint" / "checkpoint.pt"

    _link_or_verify_receipt(target, link)
    _link_or_verify_receipt(target, link)

    assert link.is_symlink()
    assert link.resolve() == target.resolve()


def test_exact_batch_health_recovery_restores_one_or_two_group_optimizer(tmp_path) -> None:
    source = PPO26DTrainer(observation_dim=764, device="cpu")
    source_path = tmp_path / "source.pt"
    torch.save(
        {
            "actor_critic": source.model.state_dict(),
            "optimizer": source.trainer.optimizer.state_dict(),
            "observation_normalization": source.trainer.normalizer.state_dict(),
            "reward_v3_samples": 2_129_920,
        },
        source_path,
    )
    candidate = PPO26DTrainer(observation_dim=764, device="cpu")
    _configure_actor_lr_scale(candidate, 0.5)
    candidate_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "actor_critic": candidate.model.state_dict(),
            "optimizer": candidate.trainer.optimizer.state_dict(),
            "observation_normalization": candidate.trainer.normalizer.state_dict(),
            "cumulative_samples": 40960,
        },
        candidate_path,
    )

    source_restored, source_contract, _ = _candidate_trainer_from_predecessor(
        source_path, device="cpu"
    )
    candidate_restored, candidate_contract, _ = _candidate_trainer_from_predecessor(
        candidate_path, device="cpu"
    )

    assert len(source_restored.trainer.optimizer.param_groups) == 2
    assert len(candidate_restored.trainer.optimizer.param_groups) == 2
    assert source_restored.cumulative_samples == 2_129_920
    assert source_contract["effective_actor_lr"] == 5.0e-5
    assert candidate_contract["critic_lr"] == 1.0e-4


def test_checkpoint_cumulative_samples_accepts_c0_stage_counter() -> None:
    assert _checkpoint_cumulative_samples({"contact_preservation_stage_samples": 40_960}) == 40_960
