"""Detached Stage16 C1 saturation receipts.

The recorder is deliberately unaware of PPO updates.  It records tensors only
after they have been used by the production action path and persists them
before its caller records the frozen saturation warning.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Stage16SaturationInstrumentationV1:
    """Frozen observability contract; it does not alter the PPO contract."""

    schema_version: str = "Stage16SaturationInstrumentationV1"
    action_dimension: int = 26
    saturation_absolute_threshold: float = 0.98
    saturation_fraction_warning_threshold: float = 0.25
    watch_levels: tuple[float, ...] = (0.15, 0.20, 0.23)
    rolling_full_rollouts: int = 4
    persistence_order: str = "collect->summarize->persist_warning_receipt->update_and_continue"
    tensor_layers: tuple[str, ...] = (
        "actor_location_pre_tanh",
        "actor_mean_tanh",
        "actor_log_std",
        "sampled_action_tanh",
        "scaled_residual",
        "pre_safety_command",
        "post_safety_command",
        "actuator_target",
        "actual_joint_q",
        "actual_joint_qdot",
        "wrist_state",
        "phase_code",
        "hand_object_contact",
        "hand_object_force",
        "table_object_contact",
        "object_tracking_error",
    )

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "metric_semantics": "count(abs(tanh(actor_location)) >= 0.98) / (T*N*26)",
            "detached": True,
            "mutates_policy": False,
            "action_override": False,
            "watch_policy": "flush_only_never_stop",
            "warning_checkpoint_policy": "checkpoint_and_receipt_before_continuing",
        }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class SaturationRecorder:
    """Keep bounded detached rollout telemetry and persist pre-gate receipts."""

    def __init__(
        self, root: Path, *, contract: Stage16SaturationInstrumentationV1 | None = None
    ) -> None:
        self.root = Path(root)
        self.contract = contract or Stage16SaturationInstrumentationV1()
        self._steps: list[dict[str, torch.Tensor]] = []
        self._rollout_index = 0
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.root / "instrumentation_contract.json", self.contract.as_dict())

    @staticmethod
    def _detach(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu").contiguous().clone()

    def record_step(
        self,
        *,
        actor_location: torch.Tensor,
        actor_mean: torch.Tensor,
        actor_log_std: torch.Tensor,
        sampled_action: torch.Tensor,
        environment: Mapping[str, torch.Tensor],
    ) -> None:
        """Record the existing production tensors without a backward edge."""

        required = {
            "scaled_residual",
            "pre_safety_command",
            "post_safety_command",
            "actuator_target",
            "actual_joint_q",
            "actual_joint_qdot",
            "wrist_state",
            "phase_code",
            "hand_object_contact",
            "hand_object_force",
            "table_object_contact",
            "object_tracking_error",
        }
        missing = required.difference(environment)
        if missing:
            raise RuntimeError(f"SATURATION_TELEMETRY_ENV_FIELDS_MISSING:{sorted(missing)}")
        if actor_mean.ndim != 2 or actor_mean.shape[-1] != self.contract.action_dimension:
            raise ValueError("SATURATION_TELEMETRY_ACTOR_SHAPE_INVALID")
        row: dict[str, torch.Tensor] = {
            "actor_location_pre_tanh": self._detach(actor_location),
            "actor_mean_tanh": self._detach(actor_mean),
            "actor_log_std": self._detach(actor_log_std.expand_as(actor_mean)),
            "sampled_action_tanh": self._detach(sampled_action),
        }
        row.update(
            {name: self._detach(value) for name, value in environment.items() if name in required}
        )
        self._steps.append(row)

    def _payload(self) -> dict[str, torch.Tensor]:
        if not self._steps:
            raise RuntimeError("SATURATION_TELEMETRY_EMPTY_ROLLOUT")
        names = self._steps[0].keys()
        return {name: torch.stack([row[name] for row in self._steps]) for name in names}

    def _summary(
        self, payload: Mapping[str, torch.Tensor], *, samples_before: int, samples_after: int
    ) -> dict[str, Any]:
        actor = payload["actor_mean_tanh"]
        positive = actor >= self.contract.saturation_absolute_threshold
        negative = actor <= -self.contract.saturation_absolute_threshold
        saturated = positive | negative
        phase = payload["phase_code"].to(torch.long)
        per_phase: dict[str, dict[str, float | int]] = {}
        phase_names = (
            "PRE_CONTACT",
            "APPROACH",
            "CONTACT",
            "GRASP",
            "LIFT",
            "MANIPULATION",
            "TERMINAL",
        )
        for code, name in enumerate(phase_names):
            mask = phase == code
            count = int(mask.sum().item())
            if count:
                expanded = mask.unsqueeze(-1).expand_as(actor)
                per_phase[name] = {
                    "steps": count,
                    "saturation": float(saturated[expanded].float().mean().item()),
                    "wrist_saturation": float(saturated[..., :6][mask].float().mean().item()),
                    "finger_saturation": float(saturated[..., 6:][mask].float().mean().item()),
                    "command_clamp": float(
                        (payload["pre_safety_command"] != payload["post_safety_command"])[..., 6:][
                            mask
                        ]
                        .float()
                        .mean()
                        .item()
                    ),
                }
        return {
            "schema_version": "Stage16SaturationRolloutSummaryV1",
            "rollout_index": self._rollout_index,
            "samples_before": samples_before,
            "samples_after": samples_after,
            "rollout_length": int(actor.shape[0]),
            "num_envs": int(actor.shape[1]),
            "global_saturation": float(saturated.float().mean().item()),
            "positive_saturation": float(positive.float().mean().item()),
            "negative_saturation": float(negative.float().mean().item()),
            "per_dimension_saturation": saturated.float().mean(dim=(0, 1)).tolist(),
            "per_dimension_positive": positive.float().mean(dim=(0, 1)).tolist(),
            "per_dimension_negative": negative.float().mean(dim=(0, 1)).tolist(),
            "wrist_saturation": float(saturated[..., :6].float().mean().item()),
            "finger_saturation": float(saturated[..., 6:].float().mean().item()),
            "command_clamp_fraction": float(
                (payload["pre_safety_command"] != payload["post_safety_command"])[..., 6:]
                .float()
                .mean()
                .item()
            ),
            "phase_attribution": per_phase,
            "watch_crossed": [
                level
                for level in self.contract.watch_levels
                if float(saturated.float().mean()) >= level
            ],
        }

    def persist_pre_gate(
        self, *, samples_before: int, samples_after: int
    ) -> tuple[dict[str, Any], Path]:
        """Flush summary and full rolling receipt before the PPO update continues."""

        payload = self._payload()
        summary = self._summary(payload, samples_before=samples_before, samples_after=samples_after)
        summaries = self.root / "rollout_summaries"
        full = self.root / "rolling_full_rollouts"
        summaries.mkdir(parents=True, exist_ok=True)
        full.mkdir(parents=True, exist_ok=True)
        summary_path = summaries / f"rollout_{self._rollout_index:04d}.json"
        _atomic_json(summary_path, summary)
        full_path = full / f"rollout_{self._rollout_index:04d}.pt"
        temporary = full_path.with_suffix(".pt.tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, full_path)
        retained = sorted(full.glob("rollout_*.pt"))
        for stale in retained[: -self.contract.rolling_full_rollouts]:
            stale.unlink()
        self._rollout_index += 1
        self._steps.clear()
        return summary, full_path

    def preserve_failure_window(self, *, triggering: Path) -> None:
        destination = self.root / "failure"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(triggering, destination / "failure_rollout.pt")
        prior = sorted((self.root / "rolling_full_rollouts").glob("rollout_*.pt"))
        previous = [path for path in prior if path != triggering]
        if previous:
            shutil.copy2(previous[-1], destination / "previous_full_rollout.pt")
