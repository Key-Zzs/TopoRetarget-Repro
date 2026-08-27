from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_h3_physicalization_protocol import _stable_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_h3_protocol_config_keeps_scientific_authorities_and_safety_limits() -> None:
    path = REPO_ROOT / "configs/contracts/h3_physicalization_protocol_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "H3PhysicalizationProtocolV1"
    assert value["retarget"]["math_changed"] is False
    assert value["retarget"]["precision"] == "float64"
    assert value["retarget"]["all_frame_independent_validation"] is True
    source = value["source_controller"]
    assert source["admission_contract"] == "SourceControllerExecutableV2"
    assert source["fidelity_contract"] == "SourceControllerFidelityV2"
    assert source["fidelity_is_hard_admission"] is False
    assert source["real_finger_joint_position_limits"] is True
    assert source["actuator_effort_limits"] is True
    assert source["actuator_velocity_limits"] is True
    assert source["unbounded_l0_profile"] == "DIAGNOSTIC_ONLY_FORBIDDEN_IN_PRODUCTION"
    assert value["ppo"]["independent_lineage_per_episode"] is True
    assert value["ppo"]["per_episode_tuning"] is False
    assert value["physical"]["reward"] == "Stage16GroupedMultiplicativeRewardV1"
    assert value["physical"]["rse"] == "Stage16ReferenceScopedExplorationV1"


def test_h3_protocol_hash_is_canonical() -> None:
    assert _stable_hash({"b": [2, 3], "a": 1}) == _stable_hash({"a": 1, "b": [2, 3]})
