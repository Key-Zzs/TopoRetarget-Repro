from __future__ import annotations

from pathlib import Path

import pytest
import torch

from toporetarget.rl.physics_retargeting.contact_topology import body_contact_group
from toporetarget.rl.physics_retargeting.self_collision import (
    InterFingerCapsulePenetrationV1,
    capsule_segment_distance,
    load_self_collision_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml"
LOCAL_AUTHORITIES = (
    REPO_ROOT
    / ".local/generated_assets/isaaclab/wuji_hand2_beta1/configuration/wujihand2_physics.usd",
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json",
)


def test_anatomical_grouping_does_not_confuse_middle_segment_names() -> None:
    assert body_contact_group("r_ring_finger_middle") == "ring"
    assert body_contact_group("r_pinky_middle") == "pinky"
    assert body_contact_group("r_middle_finger_distal") == "middle"


def test_versioned_self_collision_contract_validates_frozen_schema() -> None:
    contract = load_self_collision_contract(
        CONTRACT_PATH,
        repo_root=REPO_ROOT,
        validate_artifacts=False,
    )

    assert contract.enabled_self_collisions
    assert contract.maximum_inter_finger_penetration_m == pytest.approx(0.003)
    assert contract.config_sha256


@pytest.mark.skipif(
    not all(path.is_file() for path in LOCAL_AUTHORITIES),
    reason="requires generated Stage16-D self-collision authorities",
)
def test_versioned_self_collision_contract_validates_local_authorities() -> None:
    contract = load_self_collision_contract(CONTRACT_PATH, repo_root=REPO_ROOT)

    assert contract.config_sha256


def test_capsule_segment_distance_handles_crossing_and_separation() -> None:
    first_start = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    first_end = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second_start = torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 2.0]])
    second_end = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

    distance = capsule_segment_distance(first_start, first_end, second_start, second_end)

    torch.testing.assert_close(distance, torch.tensor([0.0, 2.0]))


def test_inter_finger_capsule_metric_reports_only_configured_pair() -> None:
    metric = InterFingerCapsulePenetrationV1(
        body_names=("index", "middle"),
        endpoint_start_local=torch.tensor([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]),
        endpoint_end_local=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        radii_m=torch.tensor([0.1, 0.1]),
        pair_indices=torch.tensor([[0, 1]]),
        pair_names=("index<->middle",),
    )
    pose = torch.zeros((2, 2, 7))
    pose[..., 3] = 1.0
    pose[1, 1, 2] = 0.5

    result = metric.evaluate(pose)

    torch.testing.assert_close(result["maximum_penetration_m"], torch.tensor([0.2, 0.0]))


def test_self_collision_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
        "3ac2f64a2d80513570329b77d44ba0f8db6e60e971f71deaaec83e0b401f1926",
        "0" * 64,
    )
    path = tmp_path / "contract.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="SELF_COLLISION_CONTRACT_HASH_DRIFT"):
        load_self_collision_contract(path, repo_root=REPO_ROOT)
