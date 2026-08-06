"""Bounded Stage 16-D recovery state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PHASES = (
    "INPUT_FREEZE",
    "TASK_SEMANTICS",
    "CONTACT_TOPOLOGY",
    "REWARD_CONTRACT",
    "PHYSICS_CORRECTION_ENV",
    "TRAJECTORY_OPTIMIZATION",
    "TRAJECTORY_QUALIFICATION",
    "DEMONSTRATION_EXPORT",
    "BC",
    "SINGLE_PPO",
    "TWO_CLIP_PPO",
    "PHYSICS_DATA_EXPORT",
    "SENSITIVITY",
    "CLOSEOUT",
)
FAIL_CLOSED = {
    "SOURCE_HASH_DRIFT",
    "GENERATED_ASSET_CORRUPTION",
    "HIDDEN_OBJECT_CONTROL",
    "HIDDEN_WRIST_OR_OBJECT_STATE_WRITES",
    "SOURCE_OVERWRITE",
    "REWARD_EXPLOIT",
    "ACTION_CONTRACT_MUTATION",
    "UNAUTHORIZED_PHYSICS_MUTATION",
    "UNRELATED_PROCESS_TERMINATION",
}


@dataclass
class Stage16DRecoveryStateMachine:
    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, Any]] = field(default_factory=list)
    failure_repairs: dict[str, int] = field(default_factory=dict)
    phase_reruns: dict[str, int] = field(default_factory=dict)
    reward_profiles_used: set[str] = field(default_factory=set)
    knot_levels_used: set[int] = field(default_factory=set)
    optimizer_global_upgrades: int = 0
    ppo_seeds: dict[str, set[int]] = field(default_factory=dict)
    ppo_lr_fallbacks: dict[str, int] = field(default_factory=dict)

    def transition(self, target: str, *, reason: str) -> None:
        if target not in PHASES:
            raise ValueError("unknown Stage16D recovery phase")
        if len(self.transitions) >= 48:
            raise RuntimeError("STAGE16D_MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def repair(self, failure_class: str, *, reason: str) -> None:
        if failure_class in FAIL_CLOSED:
            raise RuntimeError(f"STAGE16D_FAIL_CLOSED:{failure_class}:{reason}")
        count = self.failure_repairs.get(failure_class, 0) + 1
        if count > 3:
            raise RuntimeError(f"STAGE16D_REPAIR_BUDGET_EXHAUSTED:{failure_class}")
        self.failure_repairs[failure_class] = count
        self.transitions.append(
            {"phase": self.phase, "failure_class": failure_class, "repair": count, "reason": reason}
        )

    def rerun_phase(self, *, reason: str) -> None:
        count = self.phase_reruns.get(self.phase, 0) + 1
        if count > 5:
            raise RuntimeError(f"STAGE16D_PHASE_RERUN_BUDGET_EXHAUSTED:{self.phase}")
        self.phase_reruns[self.phase] = count
        self.transitions.append({"phase": self.phase, "rerun": count, "reason": reason})

    def use_reward_profile(self, profile: str) -> None:
        self.reward_profiles_used.add(profile)
        if len(self.reward_profiles_used) > 3:
            raise RuntimeError("STAGE16D_REWARD_PROFILE_BUDGET_EXHAUSTED")

    def use_knot_level(self, knots: int) -> None:
        if knots not in {16, 32, 64}:
            raise ValueError("Stage16D knots must be 16, 32, or 64")
        self.knot_levels_used.add(knots)
        if len(self.knot_levels_used) > 3:
            raise RuntimeError("STAGE16D_KNOT_LEVEL_BUDGET_EXHAUSTED")

    def optimizer_upgrade(self, *, reason: str) -> None:
        self.optimizer_global_upgrades += 1
        if self.optimizer_global_upgrades > 2:
            raise RuntimeError("STAGE16D_OPTIMIZER_UPGRADE_BUDGET_EXHAUSTED")
        self.transitions.append(
            {
                "phase": self.phase,
                "optimizer_upgrade": self.optimizer_global_upgrades,
                "reason": reason,
            }
        )

    def register_ppo_seed(self, clip: str, seed: int) -> None:
        seeds = self.ppo_seeds.setdefault(clip, set())
        seeds.add(int(seed))
        if len(seeds) > 2:
            raise RuntimeError(f"STAGE16D_PPO_SEED_BUDGET_EXHAUSTED:{clip}")

    def register_ppo_lr_fallback(self, clip: str) -> None:
        count = self.ppo_lr_fallbacks.get(clip, 0) + 1
        if count > 1:
            raise RuntimeError(f"STAGE16D_PPO_LR_FALLBACK_BUDGET_EXHAUSTED:{clip}")
        self.ppo_lr_fallbacks[clip] = count

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reward_profiles_used"] = sorted(self.reward_profiles_used)
        payload["knot_levels_used"] = sorted(self.knot_levels_used)
        payload["ppo_seeds"] = {key: sorted(value) for key, value in self.ppo_seeds.items()}
        return {"schema_version": "Stage16DRecoveryStateMachine", **payload}


__all__ = ["FAIL_CLOSED", "PHASES", "Stage16DRecoveryStateMachine"]
