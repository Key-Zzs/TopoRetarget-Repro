"""Fail-closed trajectory, PPO, and export authorization for Stage 16-D."""

from __future__ import annotations

from typing import Any


def trajectory_entry_decision(
    qualification: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    clip = str(qualification["clip"])
    geometry_pass = (
        geometry.get("formal_pass") is True or geometry.get("formal_geometry_gate") == "PASS"
    )
    success = float(qualification["success_rate"])
    semantic = float(qualification["semantic_reach_rate"])
    contact = float(qualification["contact_topology_pass_rate"])
    causality = float(qualification["contact_causality_pass_rate"])
    stable = float(qualification["terminal_stability_pass_rate"])
    numerical = float(qualification.get("numerical_pass_rate", 0.0))
    complete = float(qualification["complete_trajectory_rate"])
    episodes = qualification.get("episodes", [])
    maximum_episode_progress = max(
        (float(row["semantic_progress"]) for row in episodes),
        default=float(qualification.get("semantic_reach_rate", 0.0)),
    )
    qualified = (
        success >= 0.80
        and semantic >= 0.80
        and contact >= 0.80
        and causality == 1.0
        and stable >= 0.80
        and numerical == 1.0
        and complete == 1.0
        and geometry_pass
    )
    partial = (
        success >= 0.30
        and contact >= 0.50
        and maximum_episode_progress > 0.0
        and numerical == 1.0
        and complete == 1.0
        and geometry_pass
    )
    return {
        "schema_version": "Stage16DTrajectoryEntryDecisionV1",
        "clip": clip,
        "qualified_seed": qualified,
        "nondegenerate_partial_seed": partial and not qualified,
        "geometry_pass": geometry_pass,
        "authorization": (
            "STAGE16D_SINGLE_CLIP_PPO_AUTHORIZED"
            if qualified
            else "STAGE16D_EXPLORATORY_PPO_AUTHORIZED"
            if partial
            else "PPO_NOT_AUTHORIZED_FOR_CLIP"
        ),
        "blockers": [
            reason
            for condition, reason in (
                (success < 0.30, "task success below 30 percent partial gate"),
                (contact < 0.50, "contact topology recall below 50 percent partial gate"),
                (not geometry_pass, "independent geometry gate did not pass"),
                (numerical < 1.0, "numerical failures observed"),
                (complete < 1.0, "incomplete 321-step trajectory observed"),
            )
            if condition
        ],
    }


def two_clip_ppo_authorized(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return all(row.get("status") == "STAGE16D_SINGLE_CLIP_PPO_VALIDATED" for row in (first, second))


__all__ = ["trajectory_entry_decision", "two_clip_ppo_authorized"]
