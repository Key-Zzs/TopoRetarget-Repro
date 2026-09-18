from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import run_oakink2_o5rd2l as study
from toporetarget.retarget.objective_v4_execution import ExecutionV4AcceptedRuntimeState


def test_runtime_state_roundtrip_hash_is_stable() -> None:
    state = ExecutionV4AcceptedRuntimeState.from_arrays(
        source_ordinal=12,
        source_frame=34,
        qpos=np.arange(20, dtype=np.float64),
        base_pose_scene=np.eye(4),
        object_id="C10001",
    )
    restored = ExecutionV4AcceptedRuntimeState.from_payload(state.canonical_payload())
    assert restored.sha256 == state.sha256
    assert all(
        np.array_equal(left, right)
        for left, right in zip(state.arrays(), restored.arrays(), strict=True)
    )


def test_frame0_t1_t2_chain_never_aliases_q_old(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(runtime, root, directory, ordinal, run, candidate, **kwargs):
        del runtime, root, directory, run, candidate
        calls.append({"ordinal": ordinal, **kwargs})
        q = np.full(20, float(ordinal))
        base = np.eye(4)
        base[0, 3] = float(ordinal)
        return ({"selected": {"interaction_e_im": 0.0}}, q, base)

    monkeypatch.setattr(study.d2kr, "_run_fresh_v4_frame", fake_run)
    runtime = SimpleNamespace(
        graph=SimpleNamespace(frame_indices=np.arange(100, 200), metadata={"object_id": "C10001"})
    )
    receipt0, _q0, _base0, state0 = study._execute_frame(runtime, tmp_path, "chain", 1, 1, 0, None)
    receipt1, _q1, _base1, state1 = study._execute_frame(
        runtime, tmp_path, "chain", 2, 1, 1, state0
    )
    receipt2, _q2, _base2, _state2 = study._execute_frame(
        runtime, tmp_path, "chain", 3, 1, 2, state1
    )
    assert calls[0]["previous_q"] is None and calls[0]["previous_base"] is None
    assert np.array_equal(calls[1]["previous_q"], state0.arrays()[0])
    assert np.array_equal(calls[2]["previous_q"], state1.arrays()[0])
    assert [receipt0["q_old_field"], receipt1["q_old_field"], receipt2["q_old_field"]] == [
        "ABSENT",
        "ABSENT",
        "ABSENT",
    ]
    assert sum(item["q_old_access_count"] for item in (receipt0, receipt1, receipt2)) == 0


def test_nonzero_step_requires_previous_runtime_state(tmp_path) -> None:
    runtime = SimpleNamespace(
        graph=SimpleNamespace(frame_indices=np.arange(100), metadata={"object_id": "C10001"})
    )
    with pytest.raises(RuntimeError, match="SEQUENTIAL_STATE_CONTRACT_VIOLATION"):
        study._execute_frame(runtime, tmp_path, "chain", 1, 1, 1, None)


def test_frame0_forbids_previous_runtime_state(tmp_path) -> None:
    runtime = SimpleNamespace(
        graph=SimpleNamespace(frame_indices=np.arange(100), metadata={"object_id": "C10001"})
    )
    state = ExecutionV4AcceptedRuntimeState.from_arrays(
        source_ordinal=0,
        source_frame=0,
        qpos=np.zeros(20),
        base_pose_scene=np.eye(4),
        object_id="C10001",
    )
    with pytest.raises(RuntimeError, match="SEQUENTIAL_STATE_CONTRACT_VIOLATION"):
        study._execute_frame(runtime, tmp_path, "chain", 1, 1, 0, state)


def test_window_v5_frozen_manifest_is_hash_bound() -> None:
    assert (
        study.sha256_file(study.WINDOW_V5_MANIFEST)
        == study.WINDOW_V5_MANIFEST.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    )


def test_cli_has_required_fail_closed_actions() -> None:
    required = {
        "preflight",
        "verify-upstream",
        "localize-window-v5-failure",
        "compare-v3-v4-sequence-runtime",
        "repair-v4-runtime-state",
        "run-window-v5-regression",
        "audit-repair-impact",
        "run-sparse-v5-sentinel",
        "run-sparse-v5-regression",
        "freeze-repaired-v4",
        "select-sparse-v6",
        "run-sparse-v6",
        "select-window-v6",
        "run-window-v6",
        "select-cross-episode-v6",
        "run-cross-episode-v6",
        "freeze-certified-repaired-v4",
        "authorize-dev2-full",
        "generate-dev2-d2m-plan",
        "summarize",
    }
    assert required <= study.ACTIONS.keys()
    assert "run-dev2-full" not in study.ACTIONS


def test_runtime_only_retention_requires_sentinel_parity() -> None:
    source = study.inspect.getsource(study.audit_repair_impact)
    assert '"SPARSE_V5_CERTIFICATION_RETAINED": "YES" if parity else "NO"' in source
    assert '"REPAIR_IMPACT": "RUNTIME_ONLY" if parity else' in source


def test_window_v6_selection_excludes_window_v5_and_sparse_v5() -> None:
    excluded, counts = study._historical_exclusions()
    sparse = {int(row["ordinal"]) for row in study.read_json(study.SPARSE_V5_MANIFEST)["frames"]}
    windows = {
        int(value)
        for window in study.read_json(study.WINDOW_V5_MANIFEST)["windows"]
        for value in window["ordinals"]
    }
    assert sparse <= excluded
    assert windows <= excluded
    assert counts["sparse_v5"] == 30
    assert counts["window_v5"] == 128


def test_dev2_count_is_hard_zero() -> None:
    source = study.inspect.getsource(study.authorize_dev2_full)
    assert '"DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0' in source
    assert "run-dev2-full" not in study.ACTIONS


def test_cross_episode_v6_manifest_is_development_only_and_source_disjoint() -> None:
    manifest = study.read_json(study.ROOT / "cross_episode_v6/manifest.json")
    exclusions = study.read_json(study.ROOT / "cross_episode_v6/exclusions.json")
    controls = manifest["controls"]
    assert manifest["split"] == "DEVELOPMENT"
    assert manifest["certification_split_new_consumption"] == 0
    assert manifest["heldout_split_new_consumption"] == 0
    assert len(controls) == 3
    assert len({item["sequence_id"] for item in controls}) == 3
    assert len({item["object_id"] for item in controls}) == 3
    assert len({item["primitive"] for item in controls}) == 3
    assert not (
        {item["sequence_id"] for item in controls} & set(exclusions["known_source_sequences"])
    )


def test_cross_episode_v6_requires_window_v6_pass() -> None:
    source = study.inspect.getsource(study.select_cross_episode_v6)
    assert 'require(root / "window_v6/decision.json", "PASS"' in source
    assert "SELECT_CROSS_EPISODE_V6_REJECTED:WINDOW_V6_FAIL" in source
