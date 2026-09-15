from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.data import run_oakink2_o5rd2g as workflow
from toporetarget.retarget.objective_v3_execution import (
    ExecutionFrameInputsV3,
    ExecutionInputAuthorityError,
    RetargetMode,
)
from toporetarget.utils.hashing import sha256_file


def test_frozen_scientific_contract_hashes_remain_exact() -> None:
    paths = workflow.frozen_paths()
    assert sha256_file(paths["objective_v2"]) == workflow.OBJECTIVE_SHA
    assert sha256_file(paths["execution_v2"]) == workflow.EXECUTION_V2_SHA
    assert sha256_file(paths["gate_v2"]) == workflow.GATE_V2_SHA


def test_dev2_literals_are_identity_only_not_algorithm_branches() -> None:
    source = inspect.getsource(workflow.search_cold_start_frame)
    for forbidden in ("C11001", "10704", "scene_01__A003"):
        assert forbidden not in source
    assert workflow._special_case_audit()["DEV2_SPECIAL_CASE_ADDED"] == "NO"


def test_cold_start_rejects_synthetic_qold_before_optimizer() -> None:
    with pytest.raises(ExecutionInputAuthorityError, match="COLD_START_FORBIDS"):
        ExecutionFrameInputsV3(
            mode=RetargetMode.COLD_START,
            runtime_step_index=0,
            old_production_q=np.zeros(20),
            previous_accepted_q=None,
            previous_accepted_base=None,
        ).validate()


def test_same_canonical_source_produces_same_graph_artifact_hash(tmp_path: Path) -> None:
    paths = workflow.frozen_paths()
    source = workflow.load_hoi_sequence(paths["dev1_canonical"])
    samples = workflow.SurfaceSampleSet.load(paths["dev1_object_samples"])
    kwargs = {
        "source_cache": paths["dev1_canonical"],
        "object_sample_path": None,
        "delaunay_profile": workflow.load_delaunay_profile("strict_scipy_qhull_v1"),
        "kappa": workflow.load_paper_kappa(),
        "frame_indices": [0],
    }
    first = workflow.build_source_interaction_graph(
        source, "right_hand", "C10001", samples, **kwargs
    )
    second = workflow.build_source_interaction_graph(
        source, "right_hand", "C10001", samples, **kwargs
    )
    first_path, second_path = tmp_path / "first.zarr", tmp_path / "second.zarr"
    workflow.save_interaction_graph(first, first_path)
    workflow.save_interaction_graph(second, second_path)
    assert first.frames[0].graph_hash == second.frames[0].graph_hash
    assert workflow.interaction_artifact_hash(first_path) == workflow.interaction_artifact_hash(
        second_path
    )


def test_fail_closed_stage_order_rejects_missing_authority(tmp_path: Path) -> None:
    actions = (
        workflow.run_graph_parity,
        workflow.develop_execution_v3_candidates,
        workflow.run_masked_qold_development,
        workflow.run_refinement_regression,
        workflow.run_dev2_frame0_development,
        workflow.freeze_execution_v3,
    )
    for action in actions:
        with pytest.raises(FileNotFoundError):
            action(tmp_path)


def test_masked_development_rejects_explicit_failed_authority(tmp_path: Path) -> None:
    workflow.write_json(
        tmp_path / "input_authority/authority_decision.json",
        {"INPUT_AUTHORITY_AUDIT": "FAIL"},
    )
    with pytest.raises(RuntimeError, match="INPUT_AUTHORITY_AUDIT=FAIL"):
        workflow.run_masked_qold_development(tmp_path)


def test_no_valid_coldstart_candidate_has_exact_failure_semantics() -> None:
    with pytest.raises(RuntimeError, match="^TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE$"):
        workflow._require_coldstart_candidate(None)


def test_masked_profiler_preserves_every_required_field(tmp_path: Path) -> None:
    candidate = workflow.V3_CANDIDATES["V3_A_GENERIC_ASSET_SOURCE_COLD_START"]
    slug = workflow._candidate_slug(candidate)
    workflow.write_csv(
        tmp_path / f"development/{slug}_masked_qold.csv",
        [
            {
                "candidate": candidate.name,
                "window_id": "high",
                "stratum": "HIGH",
                "ordinal": 7,
                "technical_success": True,
            },
            {
                "candidate": candidate.name,
                "window_id": "high",
                "stratum": "HIGH",
                "ordinal": 8,
                "technical_success": False,
            },
        ],
    )
    profiler = {
        "schema_version": "RetargetSolverProfilerV1",
        "seed_count": 2,
        "candidate_probes": 2,
        "probe_nfev": 11,
        "primary_nfev": 3,
        "secondary_nfev": 4,
        "interaction_eval_time_sec": 0.1,
        "fk_time_sec": 0.2,
        "candidate_screening_time_sec": 0.0,
        "primary_retention": "PRIMARY_RETAINED",
        "fallback": True,
        "wall_sec": 0.3,
    }
    workflow.write_json(
        tmp_path / f"development/receipts/{slug}/high/frame_0007.json",
        {"profiler": profiler},
    )
    rows = workflow._write_masked_profiler(tmp_path, (candidate.name,))
    required = {
        "seed_count",
        "candidate_probes",
        "primary_nfev",
        "secondary_nfev",
        "interaction_eval_time_sec",
        "fk_time_sec",
        "candidate_screening_time_sec",
        "primary_retention",
        "fallback",
        "wall_sec",
    }
    assert required <= rows[0].keys()
    assert rows[0]["profiler_status"] == "RECORDED"
    assert rows[1]["profiler_status"] == "NOT_AVAILABLE_TECHNICAL_FAIL_BEFORE_RECEIPT"
    assert rows[1]["candidate_probes"] is None


def test_cli_has_every_contract_action() -> None:
    required = {
        "preflight",
        "verify-frozen-upstream",
        "audit-execution-input-authority",
        "audit-qold-role",
        "audit-interaction-graph-authority",
        "audit-dev2-input-completeness",
        "run-graph-parity",
        "develop-execution-v3-candidates",
        "run-masked-qold-development",
        "run-refinement-regression",
        "select-execution-v3",
        "run-dev2-frame0-development",
        "freeze-execution-input-authority",
        "freeze-coldstart-seed-authority",
        "freeze-interaction-graph-authority",
        "freeze-execution-v3",
        "generate-future-certification-plan",
        "summarize",
    }
    assert required <= set(workflow.ACTIONS)


def test_every_contract_cli_help_command_executes() -> None:
    script = Path(workflow.__file__)
    required = (
        "preflight",
        "verify-frozen-upstream",
        "audit-execution-input-authority",
        "audit-qold-role",
        "audit-interaction-graph-authority",
        "audit-dev2-input-completeness",
        "run-graph-parity",
        "develop-execution-v3-candidates",
        "run-masked-qold-development",
        "run-refinement-regression",
        "select-execution-v3",
        "run-dev2-frame0-development",
        "freeze-execution-input-authority",
        "freeze-coldstart-seed-authority",
        "freeze-interaction-graph-authority",
        "freeze-execution-v3",
        "generate-future-certification-plan",
        "summarize",
    )
    commands = ([sys.executable, str(script), "--help"],) + tuple(
        [sys.executable, str(script), action, "--help"] for action in required
    )
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("usage:")


def test_no_downstream_execution_action_exists() -> None:
    forbidden = ("sparse-v4", "window-v4", "dev2-full", "physx", "ppo", "support")
    assert not any(any(token in action for token in forbidden) for action in workflow.ACTIONS)


def test_current_fail_closed_terminal_has_every_required_artifact() -> None:
    required = {
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "preflight/git.json",
        "preflight/upstream_state.json",
        "preflight/frozen_authorities.json",
        "preflight/frozen_method_integrity.json",
        "input_authority/execution_input_dependency_audit.json",
        "input_authority/q_old_role_audit.json",
        "input_authority/interaction_graph_authority_audit.json",
        "input_authority/dev2_input_completeness.json",
        "input_authority/authority_decision.json",
        "graph_authority/source_interaction_graph_authority.json",
        "graph_authority/graph_parity.csv",
        "graph_authority/graph_parity_summary.json",
        "execution_v3_design/candidate_v3_a.json",
        "execution_v3_design/candidate_v3_b.json",
        "execution_v3_design/candidate_v3_c.json",
        "execution_v3_design/cold_start_seed_authority_draft.json",
        "execution_v3_design/fallback_authority_draft.json",
        "development/masked_qold_high.csv",
        "development/masked_qold_mid.csv",
        "development/masked_qold_low.csv",
        "development/refinement_regression.csv",
        "development/candidate_summary.csv",
        "development/selection_decision.json",
        "development/determinism.json",
        "development/profiler.csv",
        "dev2_frame0_development/role_receipt.json",
        "dev2_frame0_development/run_1.json",
        "dev2_frame0_development/run_2.json",
        "dev2_frame0_development/run_3.json",
        "dev2_frame0_development/determinism.json",
        "dev2_frame0_development/decision.json",
        "frozen_v3/no_execution_v3_ready.json",
        "ledger/execution_v3_evidence_ledger.json",
        "ledger/future_validation_exclusion_ledger.json",
        "future_certification/execution_v3_independent_certification_plan.json",
        "tests.json",
        "validation_results.json",
        "git_commits.json",
        "technical_failures.jsonl",
        "resource_usage.json",
    }
    missing = sorted(path for path in required if not (workflow.ROOT / path).is_file())
    assert missing == []
    summary = workflow.read_json(workflow.ROOT / "final_summary.json")
    assert summary["freeze"]["EXECUTION_V3_DEVELOPMENT_GATE"] == "FAIL"
    assert summary["freeze"]["EXECUTION_V3_STATUS"] == "NO_EXECUTION_V3_READY"
    assert all(value is None for key, value in summary["freeze"].items() if key.endswith("SHA256"))
    assert summary["selection"]["SELECTED_EXECUTION_V3"] is None
    assert summary["refinement_regression"]["REFINEMENT_MODE_REGRESSION"] == "NOT_RUN"
    assert summary["dev2_frame0"]["DEV2_FRAME0_V3_DEVELOPMENT"] == "NOT_RUN"
    assert not (workflow.ROOT / "frozen_v3/objective_v2_execution_contract_v3.json").exists()
    for candidate in workflow.V3_CANDIDATES.values():
        rows = workflow.read_csv(
            workflow.ROOT / f"development/{workflow._candidate_slug(candidate)}_masked_qold.csv"
        )
        assert len(rows) == 60
        assert {
            stratum: sum(row["stratum"] == stratum for row in rows)
            for stratum in ("HIGH", "MID", "LOW")
        } == {"HIGH": 20, "MID": 20, "LOW": 20}
    profiler = workflow.read_csv(workflow.ROOT / "development/profiler.csv")
    assert len(profiler) == 180
    assert {
        "seed_count",
        "candidate_probes",
        "primary_nfev",
        "secondary_nfev",
        "interaction_eval_time_sec",
        "fk_time_sec",
        "candidate_screening_time_sec",
        "primary_retention",
        "fallback",
        "wall_sec",
    } <= profiler[0].keys()
    graph = workflow.read_json(
        workflow.ROOT / "graph_authority/source_interaction_graph_authority.json"
    )
    assert graph["serialization_determinism"]["status"] == "PASS"
    assert len(set(graph["serialization_determinism"]["artifact_hashes"])) == 1
    assert graph["dev1_replay_parity"]["frame_count"] == 82
    assert graph["dev1_replay_parity"]["max_source_vertices_abs"] == 0.0
    ledger = workflow.read_json(workflow.ROOT / "ledger/execution_v3_evidence_ledger.json")
    assert ledger["FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT"] == 295
    assert len(summary["mode_comparison"]) == 6
