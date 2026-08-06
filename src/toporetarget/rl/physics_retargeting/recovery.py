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


GEOMETRY_PPO_PHASES = (
    "INPUT_FREEZE",
    "GEOMETRY_INVENTORY",
    "QUERY_BACKEND",
    "QUERY_VALIDATION",
    "METRIC_FREEZE",
    "SOURCE_GEOMETRY",
    "CORRECTED_GEOMETRY",
    "D4_REQUALIFICATION",
    "TERMINAL_FAILURE_ANALYSIS",
    "TERMINAL_REFINEMENT",
    "GLOBAL_OPTIMIZER_FALLBACK",
    "DEMONSTRATIONS",
    "BC",
    "PPO_BENCHMARK",
    "SINGLE_PPO",
    "TWO_CLIP_PPO",
    "V2_EXPORT",
    "SENSITIVITY",
    "CLOSEOUT",
)


@dataclass
class Stage16DGeometryAndPPORecoveryStateMachine:
    """Fail-closed budgets for the metric-qualification and PPO recovery run."""

    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, Any]] = field(default_factory=list)
    geometry_backends: set[str] = field(default_factory=set)
    geometry_repairs: int = 0
    terminal_refinement_profiles: set[str] = field(default_factory=set)
    global_optimizer_upgrades: int = 0
    ppo_seeds: dict[str, set[int]] = field(default_factory=dict)
    ppo_lr_fallbacks: dict[str, int] = field(default_factory=dict)
    ppo_samples: dict[str, int] = field(default_factory=dict)

    def transition(self, target: str, *, reason: str) -> None:
        if target not in GEOMETRY_PPO_PHASES:
            raise ValueError("unknown Stage16D geometry/PPO recovery phase")
        if len(self.transitions) >= 48:
            raise RuntimeError("STAGE16D_MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def register_geometry_backend(self, backend: str) -> None:
        self.geometry_backends.add(backend)
        if len(self.geometry_backends) > 1:
            raise RuntimeError("STAGE16D_GEOMETRY_BACKEND_BUDGET_EXHAUSTED")

    def register_geometry_repair(self, *, reason: str) -> None:
        self.geometry_repairs += 1
        if self.geometry_repairs > 3:
            raise RuntimeError("STAGE16D_GEOMETRY_REPAIR_BUDGET_EXHAUSTED")
        self.transitions.append(
            {
                "phase": self.phase,
                "geometry_repair": self.geometry_repairs,
                "reason": reason,
            }
        )

    def register_terminal_refinement_profile(self, profile: str) -> None:
        self.terminal_refinement_profiles.add(profile)
        if len(self.terminal_refinement_profiles) > 1:
            raise RuntimeError("STAGE16D_TERMINAL_REFINEMENT_PROFILE_BUDGET_EXHAUSTED")

    def register_global_optimizer_upgrade(self, *, reason: str) -> None:
        self.global_optimizer_upgrades += 1
        if self.global_optimizer_upgrades > 1:
            raise RuntimeError("STAGE16D_GLOBAL_OPTIMIZER_UPGRADE_BUDGET_EXHAUSTED")
        self.transitions.append(
            {
                "phase": self.phase,
                "global_optimizer_upgrade": self.global_optimizer_upgrades,
                "reason": reason,
            }
        )

    def register_ppo_run(self, clip: str, *, seed: int, samples: int) -> None:
        if samples < 0 or samples > 67_108_864:
            raise RuntimeError(f"STAGE16D_PPO_SAMPLE_BUDGET_EXHAUSTED:{clip}")
        seeds = self.ppo_seeds.setdefault(clip, set())
        seeds.add(int(seed))
        if len(seeds) > 2:
            raise RuntimeError(f"STAGE16D_PPO_SEED_BUDGET_EXHAUSTED:{clip}")
        self.ppo_samples[clip] = self.ppo_samples.get(clip, 0) + int(samples)

    def register_ppo_lr_fallback(self, clip: str) -> None:
        count = self.ppo_lr_fallbacks.get(clip, 0) + 1
        if count > 1:
            raise RuntimeError(f"STAGE16D_PPO_LR_FALLBACK_BUDGET_EXHAUSTED:{clip}")
        self.ppo_lr_fallbacks[clip] = count

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry_backends"] = sorted(self.geometry_backends)
        payload["terminal_refinement_profiles"] = sorted(self.terminal_refinement_profiles)
        payload["ppo_seeds"] = {key: sorted(value) for key, value in self.ppo_seeds.items()}
        return {
            "schema_version": "Stage16DGeometryAndPPORecoveryStateMachine",
            **payload,
        }


__all__ += [
    "GEOMETRY_PPO_PHASES",
    "Stage16DGeometryAndPPORecoveryStateMachine",
]


GEOMETRY_AWARE_PPO_PHASES = (
    "INPUT_FREEZE",
    "GATE_ATTAINABILITY",
    "METRIC_V2",
    "FAST_SIGNAL",
    "GEOMETRY_RANKING",
    "OPTIMIZE_170650",
    "OPTIMIZE_170105",
    "TRAJECTORY_QUALIFICATION",
    "DEMONSTRATIONS",
    "BC",
    "PPO_BENCHMARK",
    "PPO_170650",
    "PPO_170105",
    "TWO_CLIP",
    "V2_EXPORT",
    "SENSITIVITY",
    "CLOSEOUT",
)


@dataclass
class Stage16DGeometryAwarePPORecoveryStateMachine:
    """Budgets and prerequisite checks for the D.4R2 through D.7 recovery."""

    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, Any]] = field(default_factory=list)
    gate_versions: set[str] = field(default_factory=lambda: {"V1"})
    fast_geometry_backends: set[str] = field(default_factory=set)
    optimizer_levels: dict[str, set[str]] = field(default_factory=dict)
    formal_evaluations: dict[str, int] = field(default_factory=dict)
    ppo_seeds: dict[str, set[int]] = field(default_factory=dict)
    ppo_lr_fallbacks: dict[str, int] = field(default_factory=dict)
    single_ppo_validated: set[str] = field(default_factory=set)

    def transition(self, target: str, *, reason: str) -> None:
        if target not in GEOMETRY_AWARE_PPO_PHASES:
            raise ValueError("unknown Stage16D geometry-aware recovery phase")
        if len(self.transitions) >= 48:
            raise RuntimeError("STAGE16D_MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        if target == "TWO_CLIP" and self.single_ppo_validated != {
            "hocap_170105",
            "hocap_170650",
        }:
            raise RuntimeError("STAGE16D_TWO_CLIP_PPO_PREREQUISITES_NOT_MET")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def register_gate_version(self, version: str) -> None:
        if version not in {"V1", "V2"}:
            raise ValueError("only V1 and evidence-authorized V2 are supported")
        self.gate_versions.add(version)
        if len(self.gate_versions) > 2:
            raise RuntimeError("STAGE16D_GEOMETRY_GATE_VERSION_BUDGET_EXHAUSTED")

    def register_fast_geometry_backend(self, backend: str) -> None:
        self.fast_geometry_backends.add(backend)
        if len(self.fast_geometry_backends) > 1:
            raise RuntimeError("STAGE16D_FAST_GEOMETRY_BACKEND_BUDGET_EXHAUSTED")

    def register_optimizer_level(self, clip: str, level: str) -> None:
        if level not in {"G1", "G2"}:
            raise ValueError("only G1/G2 optimizer levels are authorized")
        levels = self.optimizer_levels.setdefault(clip, set())
        if level == "G2" and "G1" not in levels:
            raise RuntimeError(f"STAGE16D_G2_REQUIRES_G1:{clip}")
        levels.add(level)
        if len(levels) > 2:
            raise RuntimeError(f"STAGE16D_GEOMETRY_OPTIMIZER_LEVEL_BUDGET_EXHAUSTED:{clip}")

    def register_formal_evaluation(self, clip: str, level: str) -> None:
        key = f"{clip}:{level}"
        evaluations = self.formal_evaluations.get(key, 0) + 1
        self.formal_evaluations[key] = evaluations
        if evaluations > 1:
            raise RuntimeError(f"STAGE16D_FORMAL20_BUDGET_EXHAUSTED:{key}")

    def register_ppo_run(self, clip: str, *, seed: int, samples: int) -> None:
        if samples < 0 or samples > 67_108_864:
            raise RuntimeError(f"STAGE16D_PPO_SAMPLE_BUDGET_EXHAUSTED:{clip}")
        seeds = self.ppo_seeds.setdefault(clip, set())
        seeds.add(int(seed))
        if len(seeds) > 2:
            raise RuntimeError(f"STAGE16D_PPO_SEED_BUDGET_EXHAUSTED:{clip}")

    def register_ppo_lr_fallback(self, clip: str) -> None:
        count = self.ppo_lr_fallbacks.get(clip, 0) + 1
        if count > 1:
            raise RuntimeError(f"STAGE16D_PPO_LR_FALLBACK_BUDGET_EXHAUSTED:{clip}")
        self.ppo_lr_fallbacks[clip] = count

    def authorize_single_ppo_success(self, clip: str) -> None:
        if clip not in {"hocap_170105", "hocap_170650"}:
            raise ValueError("unknown Stage16D clip")
        self.single_ppo_validated.add(clip)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "Stage16DGeometryAwarePPORecoveryStateMachineV1",
            "phase": self.phase,
            "transitions": self.transitions,
            "gate_versions": sorted(self.gate_versions),
            "fast_geometry_backends": sorted(self.fast_geometry_backends),
            "optimizer_levels": {
                key: sorted(value) for key, value in self.optimizer_levels.items()
            },
            "formal_evaluations": dict(self.formal_evaluations),
            "ppo_seeds": {key: sorted(value) for key, value in self.ppo_seeds.items()},
            "ppo_lr_fallbacks": dict(self.ppo_lr_fallbacks),
            "single_ppo_validated": sorted(self.single_ppo_validated),
            "major_transition_budget": 48,
        }


__all__ += [
    "GEOMETRY_AWARE_PPO_PHASES",
    "Stage16DGeometryAwarePPORecoveryStateMachine",
]
