from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import run_oakink2_o5rd3certr as certr
from toporetarget.retarget.objective_v2 import ProductionObjectiveContextBindingV2


def _binding(local_ordinal: int, *, complete: bool) -> ProductionObjectiveContextBindingV2:
    previous = np.eye(4) if complete else None
    q = np.zeros(20) if complete else None
    return ProductionObjectiveContextBindingV2(
        active_frame_id=101,
        local_ordinal=local_ordinal,
        current_source_frame_id=101,
        previous_source_frame_id=100 if complete else None,
        previous_runtime_base_scene=previous,
        previous_robot_qpos=q,
        continuous_predicted_translation_scene=np.zeros(3) if complete else None,
        continuous_predicted_rotation_scene=np.eye(3) if complete else None,
        continuous_predicted_qpos=q,
        base_correction_reference=np.zeros(6) if complete else None,
        object_pose_scene=np.eye(4),
        object_id="object",
        robot_name="wuji",
        robot_side="right",
        robot_dof_names=tuple(f"q{i}" for i in range(20)),
        robot_mapping_authority="frozen",
        source_hand_id="right",
    )


def test_cert_v1_historical_fail_and_hashes_are_immutable() -> None:
    summary = certr.read_json(certr.CERT_V1_ROOT / "final_summary.json")
    assert summary["REFINEMENT_V2_INDEPENDENT_CERTIFICATION"] == "FAIL"
    assert certr.sha256_file(certr.CERT_V1_ROOT / "sparse/manifest.json") == certr.SPARSE_V1_SHA256
    assert (
        certr.sha256_file(certr.CERT_V1_ROOT / "certification_plan/plan.json") == certr.PLAN_SHA256
    )
    assert certr.DESIGN_SHA256 == "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
    assert (
        certr.GATE_V2_SHA256 == "505af73c9871baa67045449a4908b3259a16116b19bc2817d8b08470855bf6dc"
    )


def test_exact_first_failure_is_recoverable_from_artifacts() -> None:
    manifest = certr.sparse_manifest()
    state = certr.read_json(certr.CERT_V1_ROOT / "sparse/run_state.json")
    first = next(
        (index, item)
        for index, item in enumerate(manifest["frames"])
        if certr.target_key(item) not in set(state["completed"])
    )
    assert first[0] == 1
    assert first[1]["ordinal"] == 1
    assert first[1]["source_frame"] == 2560


def test_nonzero_missing_context_fails_before_optimizer() -> None:
    with pytest.raises(ValueError, match="nonzero frame omitted"):
        _binding(1, complete=False).validate()
    assert _binding(1, complete=True).validate().previous_robot_qpos is not None


def test_ordinal_roles_are_distinct_and_prefix_is_recursive() -> None:
    item = certr.sparse_manifest()["frames"][2]
    authority = {"trajectory_sha256": item["baseline_qold_trajectory_sha256"]}
    receipt = certr._context_plan_receipt(item, authority)
    assert item["source_frame"] != item["ordinal"]
    assert receipt["prefix_sequence_local_ordinals"] == list(range(int(item["ordinal"]) + 1))
    assert receipt["context_only_ordinals"][-1] == int(item["ordinal"]) - 1


def test_dataset_source_frame_and_record_local_graph_frame_are_distinct() -> None:
    item = certr.sparse_manifest()["frames"][1]
    authority = certr.cert._record_authority(certr.CERT_V1_ROOT, str(item["baseline_id"]))
    assert int(item["source_frame"]) == int(authority["source_interval"][0]) + int(item["ordinal"])
    assert int(item["source_frame"]) != int(item["ordinal"])


def test_every_production_input_has_known_authority_and_producer() -> None:
    rows = certr._inventory()
    assert rows
    assert all(row["Producer"] and row["Consumer"] for row in rows)
    assert all(row["Authority"] != "UNKNOWN" for row in rows)
    previous = {row["Field"]: row for row in rows}
    assert previous["previous_robot_qpos"]["Authority"] == "PREVIOUS_ACCEPTED_REFINED_RUNTIME"
    assert (
        previous["previous_runtime_base_scene"]["Authority"] == "PREVIOUS_ACCEPTED_REFINED_RUNTIME"
    )


def test_qold_is_never_previous_refined_runtime() -> None:
    source = Path(certr.__file__).read_text()
    assert '"q_old_not_previous_refined": True' in source
    assert '"previous_refined_authority": "PREVIOUS_ACCEPTED_REFINED_RUNTIME"' in source
    assert "previous_q, previous_base = q, base" in source


def test_standalone_sparse_is_rejected_and_target_is_distinct_from_context() -> None:
    plan = json.loads((certr.ROOT / "repair_plan/plan.json").read_text())
    assert plan["sparse_unit_v2"] == "PREFIX_ANCHORED_TARGET"
    assert plan["impact"] == "CERTIFICATION_PROTOCOL_UNIT_CHANGE"
    assert plan["scientific_payload_change"] == "NO"
    assert "CONTEXT_ONLY_FRAME" in Path(certr.__file__).read_text()
    assert "TARGET_CERTIFICATION_FRAME" in Path(certr.__file__).read_text()


def test_cli_exposes_contract_actions_and_no_fresh_v2_action() -> None:
    required = {
        "preflight",
        "verify-cert-v1-history",
        "locate-cert-v1-first-failure",
        "trace-production-context-callgraph",
        "inventory-production-continuous-inputs",
        "classify-context-authorities",
        "audit-ordinal-semantics",
        "decide-cert-r-root-cause",
        "freeze-cert-r-repair-plan",
        "run-sparse-v1-context-preflight",
        "run-production-context-parity",
        "run-consumed-sparse-v1-regression",
        "run-consumed-regression-determinism",
        "audit-scientific-payload-impact",
        "build-cert-v2-exclusion-ledger",
        "freeze-certification-protocol-v2",
        "authorize-cert-v2",
        "generate-cert-v2-plan",
        "summarize",
    }
    assert required <= set(certr.ACTIONS)
    assert certr.FORBIDDEN_ACTION_NAMES.isdisjoint(certr.ACTIONS)


def test_protocol_rules_are_not_outcome_driven() -> None:
    source = Path(certr.__file__).read_text()
    assert '"outcome_driven_rules": False' in source
    assert '"CROSS_EPISODE_REQUIREMENT": "REQUIRED"' in source
    assert '"D3_V2_AUTHORIZED": "NO"' in source
    assert '"FRESH_SPARSE_V2": "NOT_RUN"' in source


def test_consumed_regression_cannot_be_certification() -> None:
    source = Path(certr.__file__).read_text()
    assert "CONSUMED_PROTOCOL_REGRESSION_ONLY" in source
    assert "NOT_INDEPENDENT_CERTIFICATION" in source
    assert '"SPARSE_V1_ALL_30_CONSUMED_FOR_FUTURE_CERTIFICATION": "YES"' in source
