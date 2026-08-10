from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank
from toporetarget.rl.ppo.gpu_capacity import (
    GpuCapacityMeasurement,
    select_ppo26d_environment_capacity,
)
from toporetarget.rl.ppo.ppo26d_continuation import (
    R6ADecision,
    RSICurriculumPhase,
    build_rsi_curriculum_distribution,
    classify_ppo_update_bottleneck,
    classify_r6a,
    decide_r6b_post_16m,
    generate_frozen_seed_set,
    rank_development_checkpoints,
    summarize_episodes,
)
from toporetarget.rl.ppo.ppo26d_contract import (
    Stage16DPPO26DObservationV2,
    Stage16DReferenceResidualAction26DV1,
)
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer
from toporetarget.rl.reference_tracking.ppo26d_reference import (
    export_factor8_reference,
    inspect_source_reference,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV1,
)
from toporetarget.rl.reference_tracking.ppo26d_rsi import (
    rsi_histogram,
    sample_uniform_reference_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _source_reference(path: Path) -> None:
    frames = 41
    metadata = {
        "joint_order": [f"joint_{index}" for index in range(20)],
        "tracked_link_names": [f"link_{index}" for index in range(16)],
    }
    quaternion = np.zeros((frames, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    np.savez_compressed(
        path,
        timestamps=np.arange(frames, dtype=np.float32) / 20.0,
        wrist_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        wrist_pose_quaternion_world_ref_wxyz=quaternion,
        wrist_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        q_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        qdot_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        object_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        object_axis_points_world_ref=np.zeros((frames, 6, 3), dtype=np.float32),
        tracked_link_positions_world_ref=np.zeros((frames, 16, 3), dtype=np.float32),
        object_axis_points_wrist_ref=np.zeros((frames, 6, 3), dtype=np.float32),
        tracked_link_positions_wrist_ref=np.zeros((frames, 16, 3), dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )


def test_action_and_observation_contracts_are_frozen() -> None:
    action = Stage16DReferenceResidualAction26DV1()
    observation = Stage16DPPO26DObservationV2()
    assert action.action_dimension == 26
    assert action.wrist_slice == (0, 6)
    assert action.finger_slice == (6, 26)
    assert not action.direct_articulation_action
    assert observation.dimension == 764
    assert observation.lookahead_offsets == (0, 1, 3, 5)
    assert sum(observation.field_dimensions().values()) == 764


def test_factor8_reference_contract_exports_321_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "derived.npz"
    _source_reference(source)
    result = export_factor8_reference(source, destination)
    assert inspect_source_reference(source)["source_frames"] == 41
    assert result["contract"]["runtime_samples"] == 321
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["timestamps"].shape == (321,)
        assert archive["q_finger_ref"].shape == (321, 20)
        assert archive["tracked_link_positions_world_ref"].shape == (321, 16, 3)


def test_factor8_export_matches_runtime_reference_retiming(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "derived.npz"
    _source_reference(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    ramp = np.linspace(0.0, 1.0, 41, dtype=np.float32)
    payload["object_pose_translation_world_ref"][:, 0] = ramp**2
    payload["object_twist_world_ref"][:, 0] = np.linspace(0.2, 0.8, 41)
    payload["wrist_pose_translation_world_ref"][:, 1] = ramp**3
    payload["wrist_twist_world_ref"][:, 1] = np.linspace(-0.4, 0.6, 41)
    payload["q_finger_ref"][:, 0] = ramp**2
    payload["qdot_finger_ref"][:, 0] = np.linspace(0.1, 0.5, 41)
    np.savez_compressed(source, **payload)
    export_factor8_reference(source, destination)
    bank = WorldWristReferenceBank({"hocap_170105": source, "hocap_170650": source}, device="cpu")
    bank.apply_uniform_time_scale(8)
    with np.load(destination, allow_pickle=False) as archive:
        for field in (
            "wrist_pose_translation_world_ref",
            "wrist_pose_quaternion_world_ref_wxyz",
            "wrist_twist_world_ref",
            "q_finger_ref",
            "qdot_finger_ref",
            "object_pose_translation_world_ref",
            "object_pose_quaternion_world_ref_wxyz",
            "object_twist_world_ref",
            "object_axis_points_world_ref",
            "tracked_link_positions_world_ref",
        ):
            assert np.allclose(archive[field], getattr(bank, field)[0].numpy(), atol=1.0e-7)


def test_fixed_clip_assignment_never_falls_back_to_zero(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _source_reference(source)
    bank = WorldWristReferenceBank({"hocap_170105": source, "hocap_170650": source}, device="cpu")
    assert bank.assignment(4, balanced=False).tolist() == [0, 0, 0, 0]
    assert bank.assignment(4, balanced=False, fixed_clip="hocap_170650").tolist() == [1] * 4
    with pytest.raises(ValueError, match="unknown fixed reference clip"):
        bank.assignment(1, balanced=False, fixed_clip="hocap_missing")


def test_rsi_samples_valid_full_reference_range() -> None:
    values = sample_uniform_reference_indices(np.random.default_rng(8), count=1024, frame_count=321)
    report = rsi_histogram(values, frame_count=321)
    assert values.min() >= 0 and values.max() < 321
    assert report["sample_count"] == 1024
    assert set(report["phase_counts"]) == {
        "approach",
        "first_contact",
        "persistent_contact",
        "terminal",
    }


def test_reward_excludes_post_ppo_bonus_leakage() -> None:
    profile = TopoRetargetReferenceTrackingReward26DV1()
    assert profile.terminal_contact_bonus == 0.0
    assert profile.penetration_reward == 0.0
    assert profile.inter_finger_penalty == 0.0
    with pytest.raises(ValueError, match="post-PPO"):
        TopoRetargetReferenceTrackingReward26DV1(terminal_contact_bonus=1.0)


def test_gpu_capacity_requires_update_headroom_and_95_percent_rule() -> None:
    rows = [
        GpuCapacityMeasurement(512, 800.0, 16000.0, 6000.0, 7000.0, True, True),
        GpuCapacityMeasurement(1024, 960.0, 16000.0, 9000.0, 4000.0, True, True),
        GpuCapacityMeasurement(1536, 1000.0, 16000.0, 13000.0, 1800.0, True, True),
        GpuCapacityMeasurement(2048, 980.0, 16000.0, 11000.0, 3200.0, True, True),
    ]
    result = select_ppo26d_environment_capacity(rows)
    assert result["selected_num_envs"] == 1024
    assert "95 percent" in result["selection_reason"]


def test_ppo26d_runtime_has_no_rollout_state_write_in_action_method() -> None:
    path = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _apply_action" not in source
    assert "PPO26D_ROLLOUT_STATE_WRITE_FORBIDDEN" in source
    assert "def rollout_state_write_report" in source
    assert (
        "direct_articulation_action"
        in (REPO_ROOT / "src/toporetarget/rl/ppo/ppo26d_contract.py").read_text()
    )


def test_trace_capture_does_not_apply_canonical_joint_order_twice() -> None:
    path = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    )
    source = path.read_text(encoding="utf-8")
    capture = source[source.index("def _capture_ppo26d_trace_row") :]
    capture = capture[: capture.index("def ", 4)]
    assert '"finger_q": state["finger_q"]' in capture
    assert 'isaac_to_canonical(state["finger_q"])' not in capture


def test_ppo26d_evaluation_keeps_any_contact_distinct_from_terminal_contact() -> None:
    source = (REPO_ROOT / "scripts/rl/isaaclab/evaluate_stage16d_ppo26d.py").read_text(
        encoding="utf-8"
    )
    assert '"contact": contact_seen' in source
    assert '"terminal_contact": terminal_contact' in source
    assert '"terminal_contact": contact_seen' not in source
    assert '"reached_final_reference": termination_reason == 7' in source


def test_ppo26d_normalizer_uses_full_rollout_after_frozen_update(monkeypatch) -> None:
    class ReferenceBank:
        frame_count = 321

    class FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 2
            self.reference_bank = ReferenceBank()
            self._reference_index = torch.zeros(2, dtype=torch.long)
            self._last_reward_terms = {"total": torch.ones(2)}
            self.step_index = 0

        def reset(self):
            self.step_index = 0
            return {"policy": torch.zeros(2, 764)}, {}

        def step(self, action):
            assert torch.isfinite(action).all()
            assert torch.all(action.abs() <= 1.0)
            self.step_index += 1
            self._reference_index += 1
            observation = torch.full((2, 764), self.step_index * 0.01)
            done = torch.zeros(2, dtype=torch.bool)
            return {"policy": observation}, torch.ones(2), done, done, {}

        def rsi_report(self):
            return {"sample_count": self.num_envs}

    trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    count_during_update: list[float] = []

    def fake_update(storage, last_value):
        del last_value
        count_during_update.append(float(trainer.trainer.normalizer.count))
        return {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "ratio": 1.0,
            "action_std": 1.0,
            "sample_count": float(storage.sample_count),
            "updates": 1.0,
        }

    monkeypatch.setattr(trainer.trainer, "update", fake_update)
    result = trainer.collect_and_update(FakeEnv())
    assert count_during_update == [pytest.approx(1.0e-8)]
    assert result["safety"]["normalizer_frozen_during_rollout_and_update"]
    assert result["safety"]["normalizer_samples_added"] == pytest.approx(80.0)
    assert trainer.trainer.normalizer.mean.mean().item() == pytest.approx(0.195)


def test_torch_is_available_for_ppo26d_pure_contracts() -> None:
    assert torch.isfinite(torch.tensor([0.0])).all()


def _evaluation_summary(
    *, terminal: float, contact_steps: float, last_contact: float, object_error: float
) -> dict[str, object]:
    return {
        "terminal_contact_rate": terminal,
        "contact_steps": {"median": contact_steps},
        "last_contact_index": {"p75": last_contact},
        "final_object_position_error_m": {"median": object_error},
    }


def test_r6a_decision_gives_rsi_branch_precedence_over_improvement() -> None:
    baseline = _evaluation_summary(
        terminal=0.0, contact_steps=2.0, last_contact=3.0, object_error=0.3
    )
    result = classify_r6a(
        baseline_frame_zero=baseline,
        baseline_rsi=baseline,
        four_m_frame_zero=_evaluation_summary(
            terminal=0.15, contact_steps=20.0, last_contact=40.0, object_error=0.2
        ),
        four_m_rsi=_evaluation_summary(
            terminal=0.65, contact_steps=20.0, last_contact=40.0, object_error=0.2
        ),
        late_object_reward_baseline=1.0,
        late_object_reward_four_m=1.3,
        no_new_safety_failure=True,
        no_reward_exploit=True,
    )
    assert result.decision is R6ADecision.RSI_GOOD_FRAME_ZERO_BAD


def test_r6a_plateau_and_one_time_extension_are_bounded() -> None:
    baseline = _evaluation_summary(
        terminal=0.0, contact_steps=2.0, last_contact=3.0, object_error=0.3
    )
    plateau = classify_r6a(
        baseline_frame_zero=baseline,
        baseline_rsi=baseline,
        four_m_frame_zero=baseline,
        four_m_rsi=baseline,
        late_object_reward_baseline=1.0,
        late_object_reward_four_m=1.0,
        no_new_safety_failure=True,
        no_reward_exploit=True,
    )
    assert plateau.decision is R6ADecision.PLATEAU
    ambiguous = classify_r6a(
        baseline_frame_zero=baseline,
        baseline_rsi=baseline,
        four_m_frame_zero=_evaluation_summary(
            terminal=0.11, contact_steps=5.0, last_contact=10.0, object_error=0.28
        ),
        four_m_rsi=baseline,
        late_object_reward_baseline=1.0,
        late_object_reward_four_m=1.05,
        no_new_safety_failure=True,
        no_reward_exploit=True,
        extension_already_used=True,
    )
    assert ambiguous.decision is R6ADecision.AMBIGUOUS_ONE_TIME_EXTENSION
    assert not ambiguous.extension_allowed


def test_r6b_16m_gate_needs_two_improvements_and_safety() -> None:
    baseline = _evaluation_summary(
        terminal=0.20, contact_steps=10.0, last_contact=80.0, object_error=0.50
    )
    improved = _evaluation_summary(
        terminal=0.30, contact_steps=12.0, last_contact=100.0, object_error=0.45
    )
    decision = decide_r6b_post_16m(
        four_m_frame_zero=baseline,
        four_m_rsi=baseline,
        sixteen_m_frame_zero=improved,
        sixteen_m_rsi=baseline,
        no_safety_regression=True,
    )
    assert decision["improvement_count"] == 4
    assert decision["decision"] == "CONTINUE_TO_32M"
    unsafe = decide_r6b_post_16m(
        four_m_frame_zero=baseline,
        four_m_rsi=baseline,
        sixteen_m_frame_zero=improved,
        sixteen_m_rsi=baseline,
        no_safety_regression=False,
    )
    assert unsafe["decision"] == "STOP_AT_BEST_CHECKPOINT"


def test_seed_sets_are_reproducible_and_formal_is_distinct() -> None:
    development = generate_frozen_seed_set("dev", base_seed=7, count=20, purpose="development")
    assert development == generate_frozen_seed_set(
        "dev", base_seed=7, count=20, purpose="development"
    )
    formal = generate_frozen_seed_set("formal", base_seed=8, count=20, purpose="formal")
    assert not set(development.seeds).intersection(formal.seeds)


def test_d6_and_d7_gates_preserve_failed_single_clip_evidence(tmp_path: Path) -> None:
    def write_r7(clip: str, qualified: bool) -> Path:
        path = tmp_path / f"{clip}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "Stage16DPPO26DR7FormalQualificationV1",
                    "clip": clip,
                    "physics_qualified": qualified,
                }
            ),
            encoding="utf-8",
        )
        return path

    transitions = tmp_path / "transitions.jsonl"
    d6 = tmp_path / "d6.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rl/isaaclab/gate_stage16d_ppo26d_d6.py"),
            "--qualification-170650",
            str(write_r7("hocap_170650", True)),
            "--qualification-170105",
            str(write_r7("hocap_170105", False)),
            "--output",
            str(d6),
            "--transitions",
            str(transitions),
        ],
        check=True,
    )
    assert json.loads(d6.read_text(encoding="utf-8"))["multiclip_training_authorized"] is False
    d7 = tmp_path / "d7.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rl/isaaclab/gate_stage16d_ppo26d_d7.py"),
            "--d6-authorization",
            str(d6),
            "--output",
            str(d7),
            "--transitions",
            str(transitions),
        ],
        check=True,
    )
    assert json.loads(d7.read_text(encoding="utf-8"))["d7_export_authorized"] is False


def test_rsi_curriculum_is_telemetry_derived_and_preserves_frame_zero_coverage() -> None:
    rows = [
        {
            "start_reference_index": 20,
            "first_contact_index": 5,
            "last_contact_index": 14,
        },
        {
            "start_reference_index": 50,
            "first_contact_index": 2,
            "last_contact_index": 8,
        },
    ]
    c0 = build_rsi_curriculum_distribution(rows, frame_count=80, phase=RSICurriculumPhase.C0)
    c2 = build_rsi_curriculum_distribution(rows, frame_count=80, phase=RSICurriculumPhase.C2)
    assert c0["contract"] == "Stage16DPPO26DRSICurriculumV1"
    assert c0["regions"]["contact_persistent"] == list(range(25, 35)) + list(range(52, 59))
    assert c0["frame_zero_probability"] == pytest.approx(0.1)
    assert c2["frame_zero_probability"] >= 0.5
    assert sum(c2["probabilities"]) == pytest.approx(1.0)


def test_checkpoint_selection_keeps_frame_zero_lexicographic_precedence() -> None:
    def episode(
        *, completion: bool, terminal: bool, error: float, reward: float
    ) -> dict[str, object]:
        return {
            "contact": terminal,
            "terminal_contact": terminal,
            "contact_step_count": 20 if terminal else 5,
            "first_contact_index": 4,
            "last_contact_index": 30,
            "longest_continuous_contact_window": 20 if terminal else 5,
            "object_tracking_error_m": {"final": error},
            "final_object_rotation_error_rad": error,
            "final_object_axis_error_m": error,
            "terminal_object_linear_speed_mps": 0.01,
            "terminal_object_angular_speed_radps": 0.1,
            "reached_final_reference": completion,
            "total_reward": reward,
        }

    poor_frame_zero = episode(completion=False, terminal=False, error=0.4, reward=100.0)
    good_frame_zero = episode(completion=True, terminal=True, error=0.3, reward=10.0)
    better_rsi = episode(completion=True, terminal=True, error=0.1, reward=100.0)
    ranked = rank_development_checkpoints(
        [
            {
                "checkpoint": "later.pt",
                "cumulative_training_samples": 2_000_000,
                "frame_zero": [poor_frame_zero],
                "rsi": [better_rsi],
            },
            {
                "checkpoint": "earlier.pt",
                "cumulative_training_samples": 1_000_000,
                "frame_zero": [good_frame_zero],
                "rsi": [poor_frame_zero],
            },
        ]
    )
    assert ranked[0]["checkpoint"] == "earlier.pt"


def test_clip_scoped_development_seed_set_is_accepted_but_formal_is_rejected(
    tmp_path: Path,
) -> None:
    development_seed_set = "development_eval_seed_set_170105_v1"

    def episode(*, terminal: bool, error: float) -> dict[str, object]:
        return {
            "contact": terminal,
            "terminal_contact": terminal,
            "terminal_stable": terminal,
            "contact_step_count": 20 if terminal else 5,
            "first_contact_index": 4 if terminal else None,
            "last_contact_index": 30 if terminal else None,
            "longest_continuous_contact_window": 20 if terminal else 5,
            "object_tracking_error_m": {"final": error},
            "final_object_rotation_error_rad": error,
            "final_object_axis_error_m": error,
            "terminal_object_linear_speed_mps": 0.01,
            "terminal_object_angular_speed_radps": 0.1,
            "reached_final_reference": terminal,
            "total_reward": 1.0,
        }

    def evaluation(path: Path, *, samples: int, terminal: bool, error: float) -> None:
        frame_zero = [episode(terminal=terminal, error=error)]
        rsi = [episode(terminal=terminal, error=error)]
        path.write_text(
            json.dumps(
                {
                    "requested_clip": "hocap_170105",
                    "checkpoint": f"checkpoint_{samples}.pt",
                    "checkpoint_sha256": f"hash_{samples}",
                    "cumulative_training_samples": samples,
                    "seed_set": {"identifier": development_seed_set},
                    "frame_zero": frame_zero,
                    "rsi": rsi,
                    "frame_zero_summary": summarize_episodes(frame_zero),
                    "rsi_summary": summarize_episodes(rsi),
                }
            ),
            encoding="utf-8",
        )

    four_m = tmp_path / "four_m.json"
    sixteen_m = tmp_path / "sixteen_m.json"
    evaluation(four_m, samples=4_194_304, terminal=False, error=0.4)
    evaluation(sixteen_m, samples=16_793_600, terminal=True, error=0.3)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "cumulative_samples": 16_793_600,
                "finite": {"reward": True},
                "reference": {
                    "rsi": {
                        "rollout_object_state_writes": 0,
                        "rollout_wrist_root_state_writes": 0,
                    }
                },
                "safety": {
                    "before_update": {
                        "sampled_action_saturation_fraction": 0.0,
                        "action_saturation_fraction_limit": 0.25,
                    },
                    "after_update": {"deterministic_action_saturation_fraction": 0.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rl/isaaclab/select_stage16d_ppo26d_checkpoint.py"),
            "--evaluation",
            str(four_m),
            "--evaluation",
            str(sixteen_m),
            "--training-metrics",
            str(metrics),
            "--development-seed-set",
            development_seed_set,
            "--output",
            str(selection),
        ],
        check=True,
    )
    assert json.loads(selection.read_text(encoding="utf-8"))["seed_set"] == development_seed_set

    decision = tmp_path / "decision.json"
    transitions = tmp_path / "transitions.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rl/isaaclab/decide_stage16d_ppo26d_r6b_16m.py"),
            "--four-m-evaluation",
            str(four_m),
            "--sixteen-m-evaluation",
            str(sixteen_m),
            "--training-metrics",
            str(metrics),
            "--development-seed-set",
            development_seed_set,
            "--output",
            str(decision),
            "--transitions",
            str(transitions),
        ],
        check=True,
    )
    assert (
        json.loads(decision.read_text(encoding="utf-8"))["development_seed_set"]
        == development_seed_set
    )

    rejected = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rl/isaaclab/select_stage16d_ppo26d_checkpoint.py"),
            "--evaluation",
            str(four_m),
            "--training-metrics",
            str(metrics),
            "--development-seed-set",
            "formal_holdout_seed_set_170105_v1",
            "--output",
            str(tmp_path / "forbidden.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "formal holdout evidence is forbidden" in rejected.stderr


def test_episode_summary_and_ppo_update_bottleneck_diagnostic() -> None:
    summary = summarize_episodes(
        [
            {
                "contact": True,
                "terminal_contact": True,
                "contact_step_count": 12,
                "first_contact_index": 4,
                "last_contact_index": 25,
                "longest_continuous_contact_window": 10,
                "object_tracking_error_m": {"final": 0.1},
            },
            {
                "contact": False,
                "terminal_contact": False,
                "contact_step_count": 0,
                "first_contact_index": None,
                "last_contact_index": None,
                "longest_continuous_contact_window": 0,
                "object_tracking_error_m": {"final": 0.3},
            },
        ]
    )
    assert summary["ever_contact_rate"] == pytest.approx(0.5)
    rows = [
        {
            "ppo": {
                "kl_early_stop": True,
                "actual_epochs_executed": 1.0,
                "kl_per_epoch": [0.05],
                "target_kl": 0.03,
            }
        }
        for _ in range(9)
    ] + [
        {
            "ppo": {
                "kl_early_stop": False,
                "actual_epochs_executed": 1.0,
                "kl_per_epoch": [0.05],
                "target_kl": 0.03,
            }
        }
    ]
    diagnosis = classify_ppo_update_bottleneck(rows)
    assert diagnosis["classification"] == "POSSIBLE_PPO_UPDATE_BOTTLENECK"


def test_ppo26d_update_accepts_per_epoch_diagnostic_lists(monkeypatch) -> None:
    class ReferenceBank:
        frame_count = 321

    class FakeEnv:
        num_envs = 2
        reference_bank = ReferenceBank()
        _reference_index = torch.zeros(2, dtype=torch.long)
        _last_reward_terms = {"total": torch.ones(2)}

        def reset(self):
            return {"policy": torch.zeros(2, 764)}, {}

        def step(self, action):
            del action
            done = torch.zeros(2, dtype=torch.bool)
            return {"policy": torch.zeros(2, 764)}, torch.ones(2), done, done, {}

        def rsi_report(self):
            return {}

    trainer = PPO26DTrainer(observation_dim=764, device="cpu")

    def diagnostic_update(storage, last_value):
        del storage, last_value
        return {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "ratio": 1.0,
            "action_std": 1.0,
            "kl_per_epoch": [0.01, 0.02],
            "kl_per_minibatch": [0.01, 0.02, 0.03],
            "minibatches_per_epoch": [2, 1],
        }

    monkeypatch.setattr(trainer.trainer, "update", diagnostic_update)
    result = trainer.collect_and_update(FakeEnv())
    assert result["ppo"]["kl_per_epoch"] == [0.01, 0.02]
