from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "configs/contracts/hocap_physicalization_v1.yaml"


def _load() -> dict[str, object]:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_one_current_authority_per_physicalization_domain() -> None:
    protocol = _load()
    authorities = protocol["current_authorities"]
    assert isinstance(authorities, dict)
    assert set(authorities) == {
        "episode",
        "canonical_episode",
        "canonical_hoi",
        "geometric_retarget",
        "source_policy",
        "support",
        "gpu",
        "physical_evaluation",
        "physical_ppo",
        "physical_functionality",
        "demonstration_fidelity",
    }
    for authority in authorities.values():
        assert isinstance(authority, dict)
        assert authority["status"] == "CURRENT"
        assert isinstance(authority["name"], str) and authority["name"]


def test_current_protocol_has_no_stage_or_local_retry_identity() -> None:
    protocol = _load()
    current = {
        "current_authorities": protocol["current_authorities"],
        "workflow": protocol["workflow"],
    }
    serialized = yaml.safe_dump(current).lower()
    for stale in ("stage16", "stage16d", "v14", "v15", ".local/"):
        assert stale not in serialized


def test_current_authority_entrypoints_and_configs_exist() -> None:
    protocol = _load()
    authorities = protocol["current_authorities"]
    assert isinstance(authorities, dict)
    referenced = []
    for authority in authorities.values():
        assert isinstance(authority, dict)
        for key in ("config", "entrypoint", "materializer"):
            if key in authority:
                referenced.append(REPO_ROOT / str(authority[key]))
    assert referenced
    assert all(path.is_file() for path in referenced)


def test_protocol_serialization_hash_is_deterministic() -> None:
    first = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    second = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert first == second


def test_support_collision_and_gpu_fallback_invariants_are_frozen() -> None:
    support = yaml.safe_load(
        (REPO_ROOT / "configs/physics/support_resolution_v1.yaml").read_text(encoding="utf-8")
    )
    collision = support["collision"]
    assert collision["implementation"] == "pairwise_collision_filtering"
    assert collision["global_support_collision_disable"] is False
    assert collision["SOURCE_EXPLICIT_SUPPORT"] == {
        "object_support": True,
        "hand_support": True,
    }
    assert collision["INFERRED_PLANAR_SUPPORT"]["object_support"] is True
    assert collision["INFERRED_PLANAR_SUPPORT"]["hand_support"] is False
    protocol = _load()
    assert protocol["current_authorities"]["gpu"]["name"] == "GPURuntimePreflightV1"
    gpu = yaml.safe_load(
        (REPO_ROOT / "configs/contracts/gpu_runtime_preflight_v1.yaml").read_text(encoding="utf-8")
    )
    assert gpu["execution_context"]["authoritative"] == "HOST_UNSANDBOXED"
    assert gpu["execution_context"]["sandbox_cuda_failure_is_host_authority"] is False
    assert gpu["failure"] == {
        "status": "GPU_REQUIRED_UNAVAILABLE",
        "stop_gpu_required_stage": True,
        "automatic_cpu_fallback": False,
    }
