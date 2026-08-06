from __future__ import annotations

from toporetarget.rl.physics_retargeting.entry_gate import trajectory_entry_decision


def _qualification(success: float = 1.0) -> dict[str, object]:
    return {
        "clip": "hocap_170650",
        "success_rate": success,
        "semantic_reach_rate": 1.0,
        "contact_topology_pass_rate": 1.0,
        "contact_causality_pass_rate": 1.0,
        "terminal_stability_pass_rate": 1.0,
        "numerical_pass_rate": 1.0,
        "complete_trajectory_rate": 1.0,
        "episodes": [{"semantic_progress": 1.0}],
    }


def test_geometry_block_prevents_ppo() -> None:
    decision = trajectory_entry_decision(
        _qualification(), {"formal_geometry_gate": "BLOCKED_METRIC_COMPARABILITY"}
    )
    assert decision["authorization"] == "PPO_NOT_AUTHORIZED_FOR_CLIP"


def test_qualified_seed_authorizes_single_clip_ppo() -> None:
    decision = trajectory_entry_decision(_qualification(), {"formal_geometry_gate": "PASS"})
    assert decision["authorization"] == "STAGE16D_SINGLE_CLIP_PPO_AUTHORIZED"
