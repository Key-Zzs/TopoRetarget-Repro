from __future__ import annotations

import inspect

import numpy as np

from scripts.evaluation import run_oakink2_o5rd3certqoldr as workflow
from toporetarget.retarget import final_refinement


def test_contract_cli_is_real_and_complete() -> None:
    required = {
        "preflight",
        "verify-cert-v2-blocked-history",
        "locate-qold-generation-failure",
        "trace-canonical-wrist-constructor",
        "analyze-failing-canonical-frame",
        "analyze-failure-neighborhood",
        "audit-raw-vs-canonical-source",
        "audit-hand-side-and-frame-binding",
        "trace-canonicalization-git-history",
        "decide-qold-failure-root-cause",
        "freeze-qold-repair-plan-if-authorized",
        "freeze-qold-baseline-eligibility",
        "run-whole-pool-canonical-preflight",
        "repair-canonicalization-if-authorized",
        "run-consumed-qold-failure-regression",
        "audit-existing-qold-artifacts",
        "classify-qold-reusability",
        "freeze-qold-authority-manifest",
        "build-cert-v3-exclusion-ledger",
        "audit-protocol-impact",
        "authorize-cert-v3",
        "generate-cert-v3-plan",
        "summarize",
    }
    assert required <= workflow.COMMANDS.keys()


def test_no_optimizer_order_freezes_eligibility_before_pool_scan() -> None:
    source = inspect.getsource(workflow.run_no_optimizer)
    assert source.index("decide_root_cause") < source.index("freeze_eligibility")
    assert source.index("freeze_eligibility") < source.index("whole_pool_preflight")
    assert "run_consumed_regression" not in source


def test_whole_pool_has_no_failed_sequence_or_object_special_case() -> None:
    source = inspect.getsource(workflow.whole_pool_preflight)
    assert "FAILED_BASELINE" not in source
    assert "window_03" not in source
    assert "O02@0038@00001" not in source


def test_robot_bone_canonicalization_uses_base_keypoints() -> None:
    source = inspect.getsource(final_refinement._FrameContext.breakdown_tensor)
    assert "robot_keypoints_base = self.robot_model.keypoints_base" in source
    bone_block = source[source.index('with self.timers.measure("bone_features")') :]
    assert "extract_bone_features(\n                robot_keypoints_base," in bone_block
    assert "robot_vertices = self.robot_graph_vertices_torch(value, robot_keypoints)" in source


def test_float64_remote_translation_reproduces_old_numeric_degeneracy() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    points[9] = [0.001, -0.008, -0.074]
    points[5] = [0.023, -0.003, -0.070]
    points[17] = [-0.037, -0.001, -0.055]
    base_metrics = workflow._metrics(points)
    remote_metrics = workflow._metrics(points + 1.0e16)
    assert base_metrics["degenerate_existing_guard"] is False
    assert remote_metrics["axis_a_norm_m"] == 0.0
    assert remote_metrics["axis_b_norm_m"] == 0.0
    assert remote_metrics["degenerate_existing_guard"] is True


def test_success_path_never_runs_cert_v3_or_d3v2() -> None:
    source = inspect.getsource(workflow.authorize_cert_v3)
    assert '"FRESH_SPARSE_V3": "NOT_RUN"' in source
    assert '"FRESH_WINDOW_V3": "NOT_RUN"' in source
    assert '"FRESH_CROSS_EPISODE_V3": "NOT_RUN"' in source
    assert '"D3_V2_SCIENTIFIC_RUN_COUNT": 0' in source
