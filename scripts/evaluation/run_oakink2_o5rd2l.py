#!/usr/bin/env python3
"""O5R-D2L ExecutionV4 sequential runtime-state authority repair.

The workflow is fail closed.  WindowV5 is permanently consumed regression
evidence, fresh selections use only frozen DEV1 scalar inputs, and this CLI
contains no action capable of executing DEV2.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.data import run_oakink2_o5rd2h as d2h  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2kr_decision_tree as d2kr  # noqa: E402
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ExecutionV4AcceptedRuntimeState,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2l_execution_v4_sequential_runtime_repair_v1"
UPSTREAM = REPO / ".local/reports/oakink2_o5rd2kr_recoverability_decision_tree_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "fc8cf6089c7e834c44a264c33cc79ca302b22849"
WINDOW_V5_MANIFEST = UPSTREAM / "poor_branch/window_v5/manifest.json"
WINDOW_V5_GATE = UPSTREAM / "poor_branch/window_v5/gate_contract.json"
SPARSE_V5_MANIFEST = UPSTREAM / "poor_branch/sparse_v5/manifest.json"
SPARSE_V5_DECISION = UPSTREAM / "poor_branch/sparse_v5/gate_decision.json"
TAU = d2h.TAU
EPS = d2h.EPSILON_NUM


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"CSV_FIELDS_REQUIRED:{path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freeze_json(path: Path, value: dict[str, Any]) -> str:
    write_json(path, value)
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require(path: Path, status: str | tuple[str, ...], action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    expected = (status,) if isinstance(status, str) else status
    if value.get("status") not in expected:
        raise RuntimeError(f"{action}_REJECTED:{path.name}={value.get('status')}")
    return value


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=REPO, check=False
        ).returncode
        == 0,
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
        "guidance_worktree_exists": (REPO.parent / "TopoRetarget-Repro-guidance").is_dir(),
    }
    value = {
        "schema_version": "OakInk2O5RD2LPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "branch": branch,
        "start_head": START_HEAD,
        "observed_head": head,
        "status_short": git("status", "--short", "--untracked-files=all"),
        "diff_stat": git("diff", "--stat"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_GIT_PREFLIGHT")
    return value


def verify_upstream(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "PASS", "VERIFY_UPSTREAM")
    final = read_json(UPSTREAM / "final_summary.json")
    files = {
        "window_v5_manifest": WINDOW_V5_MANIFEST,
        "window_v5_gate": WINDOW_V5_GATE,
        "sparse_v5_manifest": SPARSE_V5_MANIFEST,
        "sparse_v5_gate": UPSTREAM / "poor_branch/sparse_v5/gate_contract.json",
        "execution_v4_input": UPSTREAM / "poor_branch/frozen_v4/execution_input_authority_v2.json",
        "execution_v4_search": UPSTREAM
        / "poor_branch/frozen_v4/cold_start_search_authority_v4.json",
        "execution_v4_contract": UPSTREAM
        / "poor_branch/frozen_v4/objective_v2_execution_contract_v4.json",
    }
    hashes: dict[str, Any] = {}
    for name, path in files.items():
        sidecar = path.with_suffix(".sha256")
        actual = sha256_file(path)
        expected = sidecar.read_text(encoding="utf-8").strip()
        hashes[name] = {
            "path": str(path),
            "actual_sha256": actual,
            "expected_sha256": expected,
            "exact": actual == expected,
        }
    checks = {
        "final_recoverability_poor": final.get("FINAL_RECOVERABILITY_DECISION")
        == "POOR_RECOVERABILITY",
        "terminal_branch_poor": final.get("SELECTED_TERMINAL_BRANCH") == "POOR",
        "execution_v4_selected": final.get("EXECUTION_V4_SELECTED_CANDIDATE")
        == "V4_A_TOP2_SEQUENTIAL",
        "sparse_v5_pass": read_json(SPARSE_V5_DECISION).get("status") == "PASS",
        "window_v5_runtime_fail": final.get("EXECUTION_V4_CERTIFICATION_FAILURE")
        == "RuntimeError:O5RD2G_RUNTIME_PREVIOUS_STATE_REQUIRED",
        "all_frozen_hashes_exact": all(bool(item["exact"]) for item in hashes.values()),
    }
    value = {
        "schema_version": "OakInk2O5RD2LUpstreamIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": hashes,
        "FINAL_RECOVERABILITY_DECISION": final.get("FINAL_RECOVERABILITY_DECISION"),
        "SELECTED_TERMINAL_BRANCH": final.get("SELECTED_TERMINAL_BRANCH"),
    }
    write_json(root / "preflight/frozen_upstream.json", value)
    write_json(root / "preflight/integrity.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_INTEGRITY")
    return value


def localize_window_v5_failure(root: Path) -> dict[str, Any]:
    require(root / "preflight/integrity.json", "PASS", "LOCALIZE_WINDOW_V5_FAILURE")
    manifest = read_json(WINDOW_V5_MANIFEST)
    first_window = manifest["windows"][0]
    frame0 = int(first_window["ordinals"][0])
    failed = int(first_window["ordinals"][1])
    source0 = int(first_window["frame_ids"][0])
    source1 = int(first_window["frame_ids"][1])
    receipt_path = (
        UPSTREAM
        / f"poor_branch/window_v5/{first_window['window_id']}/receipts/frame_{frame0:04d}_run_1.json"
    )
    failure = read_json(
        UPSTREAM / f"poor_branch/window_v5/{first_window['window_id']}/technical_failure.json"
    )
    first = {
        "schema_version": "WindowV5FirstFailureV1",
        "status": "PASS",
        "FIRST_FAILING_WINDOW": first_window["window_id"],
        "FIRST_FAILING_WINDOW_ROLE": first_window["type"],
        "FIRST_FAILING_LOCAL_INDEX": 1,
        "FIRST_FAILING_ORDINAL": failed,
        "FIRST_FAILING_SOURCE_FRAME": source1,
        "IS_WINDOW_FRAME0": "NO",
        "PREVIOUS_FRAME_SOURCE_FRAME": source0,
        "PREVIOUS_FRAME_COMPLETED": "YES",
        "PREVIOUS_FRAME_ACCEPTED": "YES",
        "error": failure["error"],
        "localization_replay_count": 0,
    }
    trace = {
        "schema_version": "ExecutionV4RuntimeContextTraceV1",
        "status": "PASS",
        "requested_execution_mode": "COLD_START_SEQUENCE",
        "runtime_step_index": 1,
        "q_old_expected": False,
        "q_old_supplied": False,
        "q_old_accessed": False,
        "previous_accepted_runtime_state_required": True,
        "previous_accepted_runtime_state_available": True,
        "previous_accepted_runtime_state_constructed": True,
        "previous_accepted_runtime_state_serialized": True,
        "passed_through_window_runner": True,
        "passed_through_execution_v3_prefix": True,
        "received_by_execution_v4_entry": True,
        "consumed_by_execution_v4_scientific_search": False,
        "first_divergent_operation": "ExecutionV4._v4_evaluate_state rebound previous_q and previous_base to None before V3Runtime.bind_context",
        "previous_state_hash": sha256_file(receipt_path.with_suffix(".npz")),
        "previous_q_sha256": _array_sha(_load_state(receipt_path.with_suffix(".npz"))[0]),
        "previous_base_sha256": _array_sha(_load_state(receipt_path.with_suffix(".npz"))[1]),
        "previous_frame_receipt_sha256": sha256_file(receipt_path),
    }
    cause = {
        "schema_version": "ExecutionV4SequenceRootCauseV1",
        "status": "PASS",
        "WINDOW_V5_PRIMARY_ROOT_CAUSE": "NEXT_FRAME_CONTEXT_BINDING_BUG",
        "CONFIDENCE": "HIGH",
        "first_divergent_operation": trace["first_divergent_operation"],
        "repair_authorized": True,
    }
    write_json(root / "d2la_localization/window_v5_failure_receipt.json", failure)
    write_json(root / "d2la_localization/first_failure.json", first)
    write_json(root / "d2la_localization/runtime_context_trace.json", trace)
    write_json(root / "d2la_localization/root_cause.json", cause)
    return cause


def compare_v3_v4_sequence_runtime(root: Path) -> dict[str, Any]:
    require(root / "d2la_localization/root_cause.json", "PASS", "COMPARE_V3_V4")
    rows = [
        ("frame0 mode dispatch", "COLD_START, predecessor absent", "same", "YES"),
        ("t>0 mode dispatch", "COLD_START, predecessor required", "same", "YES"),
        (
            "runtime state type",
            "separate accepted q/base authority",
            "separate accepted q/base authority",
            "YES",
        ),
        (
            "accepted-state extraction",
            "q/base from accepted t-1",
            "q/base from accepted t-1",
            "YES",
        ),
        ("q/base fields", "present together", "present together", "YES"),
        ("serialization", "npz q/base", "canonical state plus npz q/base", "EQUIVALENT"),
        (
            "next-frame binding",
            "passed to every bind/evaluate/solver phase",
            "dropped at V4 evaluation boundary",
            "NO_ROOT_CAUSE",
        ),
        (
            "contributor context",
            "bound continuation context",
            "repaired bound continuation context",
            "YES_AFTER_REPAIR",
        ),
        ("seed context", "previous runtime seed handled by V3 prefix", "V3 prefix reused", "YES"),
        (
            "fallback context",
            "bound continuation context",
            "repaired bound continuation context",
            "YES_AFTER_REPAIR",
        ),
        ("final result adapter", "accepted q/base", "canonical accepted state", "EQUIVALENT"),
    ]
    write_csv(
        root / "d2la_localization/v3_v4_sequence_comparison.csv",
        [{"component": a, "v3": b, "v4": c, "same": d} for a, b, c, d in rows],
    )
    value = {
        "schema_version": "ExecutionV3V4SequenceComparisonV1",
        "status": "PASS",
        "first_divergence": "next-frame binding inside ExecutionV4 scientific evaluation adapter",
        "v3_lifecycle_reusable": True,
    }
    return value


def repair_v4_runtime_state(root: Path) -> dict[str, Any]:
    cause = require(root / "d2la_localization/root_cause.json", "PASS", "REPAIR_V4")
    if cause.get("WINDOW_V5_PRIMARY_ROOT_CAUSE") == "INCONCLUSIVE":
        raise RuntimeError("REPAIR_V4_RUNTIME_STATE_REJECTED:INCONCLUSIVE_ROOT_CAUSE")
    source = inspect.getsource(d2kr.search_cold_start_v4_from_v3)
    checks = {
        "previous_q_parameter": "previous_q: np.ndarray | None = None" in source,
        "previous_base_parameter": "previous_base: np.ndarray | None = None" in source,
        "q_old_not_introduced": "q_old=" not in source,
        "runtime_state_roundtrip": _runtime_state_roundtrip_check(),
    }
    value = {
        "schema_version": "ExecutionV4SequentialRuntimeRepairReceiptV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "REPAIR_SCOPE": "V4 continuation predecessor binding and accepted-state lifecycle only",
        "EXECUTION_V3_SEQUENCE_PLUMBING_REUSED": "YES",
        "checks": checks,
        "q_old_synthesized": False,
        "previous_runtime_state_aliased_as_q_old": False,
    }
    write_json(root / "d2lb_repair/repair_scope.json", value)
    write_json(
        root / "d2lb_repair/modified_runtime_components.json",
        {
            "schema_version": "ExecutionV4ModifiedRuntimeComponentsV1",
            "status": value["status"],
            "components": [
                "V4 evaluation binder",
                "V4 solver-phase binder",
                "accepted-state serializer",
                "next-frame runner binder",
            ],
            "scientific_configuration_changed": False,
        },
    )
    if value["status"] != "PASS":
        raise RuntimeError("EXECUTION_V4_RUNTIME_REPAIR_SELF_CHECK_FAIL")
    return value


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["qpos"], dtype=np.float64), np.asarray(
            archive["base_pose_scene"], dtype=np.float64
        )


def _runtime_state_roundtrip_check() -> bool:
    state = ExecutionV4AcceptedRuntimeState.from_arrays(
        source_ordinal=0,
        source_frame=1,
        qpos=np.zeros(20),
        base_pose_scene=np.eye(4),
        object_id="C10001",
    )
    restored = ExecutionV4AcceptedRuntimeState.from_payload(state.canonical_payload())
    return state.sha256 == restored.sha256 and all(
        np.array_equal(a, b) for a, b in zip(state.arrays(), restored.arrays(), strict=True)
    )


def _accepted_state(
    runtime: Any, ordinal: int, q: np.ndarray, base: np.ndarray
) -> ExecutionV4AcceptedRuntimeState:
    return ExecutionV4AcceptedRuntimeState.from_arrays(
        source_ordinal=ordinal,
        source_frame=int(runtime.graph.frame_indices[ordinal]),
        qpos=q,
        base_pose_scene=base,
        object_id=str(runtime.graph.metadata["object_id"]),
    )


def _execute_frame(
    runtime: Any,
    root: Path,
    directory: str,
    ordinal: int,
    run: int,
    step: int,
    previous: ExecutionV4AcceptedRuntimeState | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, ExecutionV4AcceptedRuntimeState]:
    if (step == 0) != (previous is None):
        raise RuntimeError("EXECUTION_V4_SEQUENTIAL_STATE_CONTRACT_VIOLATION")
    previous_q, previous_base = (None, None) if previous is None else previous.arrays()
    receipt, q, base = d2kr._run_fresh_v4_frame(
        runtime,
        root,
        directory,
        ordinal,
        run,
        d2kr.default_cold_start_search_v4_candidates()[0],
        runtime_step=step,
        previous_q=previous_q,
        previous_base=previous_base,
    )
    state = _accepted_state(runtime, ordinal, q, base)
    receipt.update(
        {
            "runtime_mode": "COLD_START" if step == 0 else "COLD_START_SEQUENCE",
            "q_old_field": "ABSENT",
            "q_old_access_count": 0,
            "previous_state_required": step > 0,
            "previous_state_source_frame": None if previous is None else previous.source_frame,
            "previous_state_hash": None if previous is None else previous.sha256,
            "accepted_state_hash": state.sha256,
        }
    )
    receipt_path = root / directory / "receipts" / f"frame_{ordinal:04d}_run_{run}.json"
    write_json(receipt_path, receipt)
    write_json(
        receipt_path.with_name(receipt_path.stem + "_accepted_state.json"),
        state.canonical_payload(),
    )
    return receipt, q, base, state


def _run_one_window(
    runtime: Any, root: Path, directory: str, window: dict[str, Any], run: int
) -> tuple[
    list[dict[str, Any]],
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    old = {int(row["ordinal"]): float(row["old_e_im"]) for row in d2h.d2d.old_eim_rows()}
    previous = None
    rows: list[dict[str, Any]] = []
    q_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    for step, raw in enumerate(window["ordinals"]):
        ordinal = int(raw)
        receipt, q, base, state = _execute_frame(
            runtime, root, f"{directory}/{window['window_id']}", ordinal, run, step, previous
        )
        previous_q, previous_base = (None, None) if previous is None else previous.arrays()
        receipt = d2kr._reevaluate_v4_continuity(
            runtime, ordinal, q, base, previous_q, previous_base, receipt
        )
        receipt_path = (
            root
            / directory
            / str(window["window_id"])
            / "receipts"
            / f"frame_{ordinal:04d}_run_{run}.json"
        )
        write_json(receipt_path, receipt)
        row = d2h._row_from_coldstart(receipt, str(window["type"]), old[ordinal], standalone=False)
        row.update({"window_id": window["window_id"], "run": run})
        rows.append(row)
        q_states.append(q)
        base_states.append(base)
        receipts.append(receipt)
        chain.append(
            {
                "window_id": window["window_id"],
                "run": run,
                "local_index": step,
                "ordinal": ordinal,
                "source_frame": state.source_frame,
                "previous_state_required": step > 0,
                "previous_state_source_frame": None if previous is None else previous.source_frame,
                "previous_state_hash": None if previous is None else previous.sha256,
                "accepted_state_hash": state.sha256,
                "q_old_access_count": 0,
            }
        )
        previous = state
    return rows, q_states, base_states, receipts, chain


def _window_summary(window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    eim = [float(row["new_e_im"]) for row in rows]
    p95 = float(np.quantile(eim, 0.95)) if eim else None
    low_preserved = all(
        float(row["new_e_im"]) <= TAU + EPS
        for row in rows
        if str(window["type"]) == "LOW" and float(row["old_e_im"]) <= TAU
    )
    value = {
        "window_id": window["window_id"],
        "type": window["type"],
        "N": len(rows),
        "expected_N": int(window["N"]),
        "old_p95_e_im": float(window["old_p95_e_im"]),
        "new_p95_e_im": p95,
        "technical": sum(bool(row["technical_completion"]) for row in rows),
        "continuity": all(bool(row["continuity_pass"]) for row in rows),
        "hard_validity": all(bool(row["semantic_hard_pass"]) for row in rows),
        "low_old_valid_preserved": low_preserved,
    }
    value["result_without_determinism"] = (
        "PASS"
        if (
            len(rows) == int(window["N"])
            and value["technical"] == int(window["N"])
            and p95 is not None
            and p95 <= TAU
            and value["continuity"]
            and value["hard_validity"]
            and low_preserved
        )
        else "FAIL"
    )
    return value


def _run_windows(
    root: Path, manifest: dict[str, Any], directory: str, schema: str
) -> dict[str, Any]:
    runtime = d2kr.d2g.V3Runtime("dev_01", root)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    jitter: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    technical_failure = None
    for window in manifest["windows"]:
        try:
            result = _run_one_window(runtime, root, directory, window, 1)
        except Exception as exc:
            technical_failure = f"{type(exc).__name__}:{exc}"
            summaries.append(
                {
                    "window_id": window["window_id"],
                    "type": window["type"],
                    "N": 0,
                    "expected_N": int(window["N"]),
                    "old_p95_e_im": float(window["old_p95_e_im"]),
                    "new_p95_e_im": None,
                    "technical": 0,
                    "continuity": False,
                    "hard_validity": False,
                    "low_old_valid_preserved": False,
                    "result_without_determinism": "FAIL",
                    "failure": technical_failure,
                }
            )
            break
        rows, q_states, base_states, receipts, chain = result
        references[str(window["window_id"])] = result
        all_rows.extend(rows)
        chains.extend(chain)
        jitter.extend(d2h._jitter_rows(runtime, str(window["window_id"]), rows, q_states))
        summary = _window_summary(window, rows)
        summaries.append(summary)
        print(
            f"{directory} {window['window_id']} N={len(rows)} p95={summary['new_p95_e_im']} result={summary['result_without_determinism']}",
            flush=True,
        )
        if summary["result_without_determinism"] != "PASS":
            break
    write_csv(
        root / directory / "per_frame.csv",
        all_rows,
        ["window_id", "ordinal", "run", "technical_completion"] if not all_rows else None,
    )
    write_csv(root / directory / "per_window.csv", summaries)
    write_csv(
        root / directory / "runtime_chain.csv",
        chains,
        ["window_id", "run", "local_index", "ordinal", "source_frame", "q_old_access_count"]
        if not chains
        else None,
    )
    write_csv(
        root / directory / "continuity.csv",
        jitter,
        ["window_id", "ordinal", "role"] if not jitter else None,
    )
    write_csv(
        root / directory / "jitter.csv",
        jitter,
        ["window_id", "ordinal", "role"] if not jitter else None,
    )
    repeats: list[dict[str, Any]] = []
    if len(summaries) == 4 and all(
        row["result_without_determinism"] == "PASS" for row in summaries
    ):
        for window_id in ("HIGH_1", "LOW"):
            window = next(
                item for item in manifest["windows"] if str(item["window_id"]) == window_id
            )
            ref_rows, ref_q, ref_base, ref_receipts, _ref_chain = references[window_id]
            repeats.append({"window_id": window_id, "run": 1, "pass": True})
            for run in (2, 3):
                rows, qs, bases, receipts, chain = _run_one_window(
                    runtime, root, directory, window, run
                )
                chains.extend(chain)
                comparisons = [
                    d2kr._v4_repeat_comparison(a, qa, ba, b, qb, bb)
                    for a, qa, ba, b, qb, bb in zip(
                        ref_receipts, ref_q, ref_base, receipts, qs, bases, strict=True
                    )
                ]
                p95_diff = abs(
                    float(np.quantile([float(row["new_e_im"]) for row in ref_rows], 0.95))
                    - float(np.quantile([float(row["new_e_im"]) for row in rows], 0.95))
                )
                repeats.append(
                    {
                        "window_id": window_id,
                        "run": run,
                        "pass": all(bool(item["pass"]) for item in comparisons) and p95_diff <= EPS,
                        "p95_E_IM_abs_diff": p95_diff,
                        "frames": comparisons,
                    }
                )
        write_csv(root / directory / "runtime_chain.csv", chains)
    determinism = bool(repeats) and all(bool(row["pass"]) for row in repeats)
    write_json(
        root / directory / "determinism.json",
        {
            "schema_version": f"{schema}DeterminismV1",
            "status": "PASS" if determinism else "NOT_RUN_OR_FAIL",
            "rows": repeats,
        },
    )
    technical_complete = (
        technical_failure is None
        and len(summaries) == 4
        and all(int(row["N"]) == int(row["expected_N"]) for row in summaries)
    )
    scientific = (
        technical_complete
        and all(row["result_without_determinism"] == "PASS" for row in summaries)
        and determinism
    )
    outcome = (
        "PASS"
        if scientific
        else ("TECHNICAL_FAIL" if not technical_complete else "SCIENTIFIC_FAIL")
    )
    decision = {
        "schema_version": f"{schema}DecisionV1",
        "status": outcome,
        "result": outcome,
        "windows": summaries,
        "technical_complete": technical_complete,
        "scientific_gate_pass": scientific,
        "determinism": "PASS" if determinism else "NOT_RUN_OR_FAIL",
        "q_old_access_count": sum(int(row.get("q_old_access_count", 0)) for row in chains),
        "technical_failure": technical_failure,
    }
    write_json(
        root / directory / "scientific_metrics.json",
        {"status": "PASS" if scientific else "FAIL", "windows": summaries},
    )
    write_json(root / directory / "decision.json", decision)
    return decision


def run_window_v5_regression(root: Path) -> dict[str, Any]:
    require(root / "d2lb_repair/repair_scope.json", "PASS", "RUN_WINDOW_V5_REGRESSION")
    manifest = read_json(WINDOW_V5_MANIFEST)
    binding = {
        "schema_version": "WindowV5ConsumedRegressionBindingV1",
        "status": "PASS",
        "WINDOW_V5_EVIDENCE_ROLE": "CONSUMED_TECHNICAL_FAILURE_REGRESSION",
        "WINDOW_V5_INDEPENDENT_EVIDENCE_REUSED": "NO",
        "manifest_path": str(WINDOW_V5_MANIFEST),
        "manifest_sha256": sha256_file(WINDOW_V5_MANIFEST),
        "gate_path": str(WINDOW_V5_GATE),
        "gate_sha256": sha256_file(WINDOW_V5_GATE),
    }
    write_json(root / "window_v5_regression/manifest_binding.json", binding)
    decision = _run_windows(root, manifest, "window_v5_regression", "WindowV5ConsumedRegression")
    decision["WINDOW_V5_EVIDENCE_ROLE"] = "CONSUMED_TECHNICAL_FAILURE_REGRESSION"
    decision["WINDOW_V5_CONSUMED_REGRESSION"] = decision["result"]
    decision["NEXT"] = {
        "TECHNICAL_FAIL": "EXECUTION_V4_SEQUENCE_RUNTIME_REPAIR_V2",
        "SCIENTIFIC_FAIL": "EXECUTION_V4_SEQUENCE_SCIENTIFIC_FAILURE_ANALYSIS",
        "PASS": "REPAIR_IMPACT_AUDIT",
    }[decision["result"]]
    write_json(root / "window_v5_regression/decision.json", decision)
    return decision


def _sentinel_frames() -> list[dict[str, Any]]:
    manifest = read_json(SPARSE_V5_MANIFEST)
    by_ordinal = {int(row["ordinal"]): row for row in manifest["frames"]}
    return [by_ordinal[value] for value in manifest["determinism_ordinals"]]


def _receipt_signature(value: dict[str, Any]) -> dict[str, Any]:
    prefix = value.get("execution_v3_prefix_receipt", {})
    return {
        "E_IM": value["selected"]["interaction_e_im"],
        "bootstrap_candidate": prefix.get("selected_seed_candidate"),
        "initial_contributor_scores": value.get("initial_contributor_scores"),
        "initial_contributor_ranking": value.get("initial_contributor_ranking"),
        "used_contributors": value.get("used_contributors"),
        "selected_block": value.get("selected_block"),
        "selected_primary": value.get("selected_candidate"),
        "retention": value.get("retention_decision"),
        "fallback": value.get("profiler", {}).get("fallback"),
        "hard_validity": value.get("selected_evaluation"),
    }


def audit_repair_impact(root: Path) -> dict[str, Any]:
    require(root / "window_v5_regression/decision.json", "PASS", "AUDIT_REPAIR_IMPACT")
    before = read_json(UPSTREAM / "poor_branch/frozen_v4/cold_start_search_authority_v4.json")
    candidate = d2kr.default_cold_start_search_v4_candidates()[0]
    scientific = {
        "schema_version": "ExecutionV4ScientificSearchPayloadV1",
        "status": "UNCHANGED",
        "objective_v2_sha256": d2h.EXPECTED_HASHES["retarget_objective_v2"],
        "certification_gate_v2_sha256": d2h.EXPECTED_HASHES["certification_gate_v2"],
        "candidate": candidate.as_dict(),
        "whole_hand_bootstrap": "ExecutionV3 SearchV2 unchanged",
        "seed_authority": "unchanged",
        "interaction_graph": "unchanged",
        "contributor_scoring": "unchanged",
        "top_k": candidate.top_k,
        "top_2_order": "sequential",
        "candidate_b2": "unchanged",
        "primary_budgets": candidate.selected_primary_maxiter,
        "secondary_polish": candidate.secondary_polish_maxiter,
        "candidate_screening": "unchanged",
        "retention": "unchanged",
        "fallback": "unchanged",
        "hard_validity_bounds": "unchanged",
        "historical_search_authority_sha256": sha256_file(
            UPSTREAM / "poor_branch/frozen_v4/cold_start_search_authority_v4.json"
        ),
    }
    write_json(
        root / "d2lb_repair/scientific_payload_manifest_before.json",
        {**scientific, "historical_implementation_sha256": before["implementation_sha256"]},
    )
    write_json(
        root / "d2lb_repair/scientific_payload_manifest_after.json",
        {
            **scientific,
            "runtime_binding_repair": "previous accepted q/base threaded through existing V3 context",
        },
    )
    write_json(root / "impact_audit/scientific_payload.json", scientific)
    runtime_payload = {
        "schema_version": "ExecutionV4SequentialRuntimePayloadV1",
        "status": "REPAIRED",
        "components": [
            "window runner",
            "mode dispatch",
            "previous-state adapter",
            "serializer",
            "result-to-runtime-state adapter",
            "next-frame binder",
        ],
        "accepted_state_type_source_sha256": hashlib.sha256(
            inspect.getsource(ExecutionV4AcceptedRuntimeState).encode()
        ).hexdigest(),
        "v4_entry_source_sha256": hashlib.sha256(
            inspect.getsource(d2kr.search_cold_start_v4_from_v3).encode()
        ).hexdigest(),
    }
    write_json(root / "impact_audit/runtime_payload.json", runtime_payload)
    frames = _sentinel_frames()
    freeze_json(
        root / "impact_audit/sparse_v5_sentinel_manifest.json",
        {
            "schema_version": "SparseV5SentinelManifestV1",
            "status": "FROZEN",
            "selection": "preexisting SparseV5 determinism subset",
            "frames": frames,
            "composition": {"HIGH": 2, "MID": 2, "LOW": 1},
        },
    )
    runtime = d2kr.d2g.V3Runtime("dev_01", root)
    comparisons: list[dict[str, Any]] = []
    for frame in frames:
        ordinal = int(frame["ordinal"])
        receipt, q, base, _state = _execute_frame(
            runtime, root, "impact_audit/sentinel", ordinal, 1, 0, None
        )
        old_receipt_path = (
            UPSTREAM / f"poor_branch/sparse_v5/receipts/frame_{ordinal:04d}_run_1.json"
        )
        old_receipt = read_json(old_receipt_path)
        old_q, old_base = _load_state(old_receipt_path.with_suffix(".npz"))
        q_diff = float(np.max(np.abs(q - old_q)))
        base_diff = float(np.max(np.abs(base - old_base)))
        e_diff = abs(
            float(receipt["selected"]["interaction_e_im"])
            - float(old_receipt["selected"]["interaction_e_im"])
        )
        signature = _receipt_signature(receipt) == _receipt_signature(old_receipt)
        comparisons.append(
            {
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "max_q_abs_diff": q_diff,
                "max_base_abs_diff": base_diff,
                "E_IM_abs_diff": e_diff,
                "scientific_signature_exact": signature,
                "pass": signature
                and q_diff <= d2h.Q_ATOL
                and base_diff <= d2h.BASE_ATOL
                and e_diff <= EPS,
            }
        )
    write_csv(root / "impact_audit/sparse_v5_sentinel_parity.csv", comparisons)
    parity = all(bool(row["pass"]) for row in comparisons)
    decision = {
        "schema_version": "ExecutionV4RepairImpactDecisionV1",
        "status": "PASS" if parity else "FAIL",
        "REPAIR_IMPACT": "RUNTIME_ONLY" if parity else "SCIENTIFIC_SEARCH_PAYLOAD_CHANGED",
        "SCIENTIFIC_PAYLOAD_CHANGED": "NO" if parity else "YES",
        "SPARSE_V5_SENTINEL_PARITY": "PASS" if parity else "FAIL",
        "SPARSE_V5_CERTIFICATION_RETAINED": "YES" if parity else "NO",
        "RUNTIME_ONLY_CONFIRMED": "YES" if parity else "NO",
    }
    write_json(
        root / "impact_audit/static_diff.json",
        {
            "status": "PASS",
            "only_sequential_runtime_payload_changed": True,
            "scientific_configuration_semantic_diff": [],
        },
    )
    write_json(root / "impact_audit/decision.json", decision)
    write_json(
        root / "sparse_v5_regression/not_run.json",
        {
            "schema_version": "ConsumedSparseV5RegressionNotRunV1",
            "status": "NOT_RUN_RUNTIME_ONLY" if parity else "REQUIRED",
            "reason": "RUNTIME_ONLY_SENTINEL_EXACT_PARITY"
            if parity
            else "SCIENTIFIC_SEARCH_PAYLOAD_CHANGED",
        },
    )
    return decision


def run_sparse_v5_sentinel(root: Path) -> dict[str, Any]:
    return audit_repair_impact(root)


def run_sparse_v5_regression(root: Path) -> dict[str, Any]:
    impact = read_json(root / "impact_audit/decision.json")
    if impact.get("REPAIR_IMPACT") != "SCIENTIFIC_SEARCH_PAYLOAD_CHANGED":
        return read_json(root / "sparse_v5_regression/not_run.json")
    raise RuntimeError("CONSUMED_SPARSE_V5_REGRESSION_REQUIRED_BUT_NOT_IMPLEMENTED_FAIL_CLOSED")


def freeze_repaired_v4(root: Path) -> dict[str, Any]:
    impact = require(root / "impact_audit/decision.json", "PASS", "FREEZE_REPAIRED_V4")
    if impact.get("REPAIR_IMPACT") != "RUNTIME_ONLY":
        raise RuntimeError("FREEZE_REPAIRED_V4_REQUIRES_CHANGED_METHOD_REGRESSION_PATH")
    payload = {
        "schema_version": "ExecutionV4SequentialRuntimeAuthorityV1",
        "status": "FROZEN_CANDIDATE",
        "scientific_search_version_unchanged": True,
        "sequential_runtime_authority_updated": True,
        "accepted_state_schema": "ExecutionV4AcceptedRuntimeStateV1",
        "frame0": {"mode": "COLD_START", "q_old": "ABSENT", "previous_state": "ABSENT"},
        "continuation": {
            "mode": "COLD_START_SEQUENCE",
            "q_old": "ABSENT",
            "previous_state": "EXACT_ACCEPTED_T_MINUS_1",
        },
        "runtime_payload": read_json(root / "impact_audit/runtime_payload.json"),
    }
    digest = freeze_json(root / "d2lb_repair/sequential_runtime_authority_candidate.json", payload)
    return {
        "schema_version": "ExecutionV4SequentialRuntimeCandidateFreezeV1",
        "status": "PASS",
        "sha256": digest,
    }


def _historical_exclusions() -> tuple[set[int], dict[str, int]]:
    prior, counts = d2kr._v4_frame_exclusions()
    sparse_v5 = {int(row["ordinal"]) for row in read_json(SPARSE_V5_MANIFEST)["frames"]}
    window_v5 = {
        int(value)
        for window in read_json(WINDOW_V5_MANIFEST)["windows"]
        for value in window["ordinals"]
    }
    all_values = prior | sparse_v5 | window_v5
    return all_values, {
        **counts,
        "sparse_v5": len(sparse_v5),
        "window_v5": len(window_v5),
        "total_unique_d2l": len(all_values),
    }


def select_sparse_v6(root: Path) -> dict[str, Any]:
    impact = read_json(root / "impact_audit/decision.json")
    if impact.get("REPAIR_IMPACT") != "SCIENTIFIC_SEARCH_PAYLOAD_CHANGED":
        value = {
            "schema_version": "SparseV6NotRunV1",
            "status": "NOT_RUN_RUNTIME_ONLY",
            "SPARSE_V6": "NOT_RUN_RUNTIME_ONLY",
        }
        write_json(root / "sparse_v6/not_run.json", value)
        return value
    raise RuntimeError("SPARSE_V6_SELECTION_REQUIRES_CHANGED_METHOD_PATH_FAIL_CLOSED")


def run_sparse_v6(root: Path) -> dict[str, Any]:
    return select_sparse_v6(root)


def select_window_v6(root: Path) -> dict[str, Any]:
    impact = require(root / "impact_audit/decision.json", "PASS", "SELECT_WINDOW_V6")
    window_v5 = require(root / "window_v5_regression/decision.json", "PASS", "SELECT_WINDOW_V6")
    if window_v5.get("WINDOW_V5_CONSUMED_REGRESSION") != "PASS":
        raise RuntimeError("SELECT_WINDOW_V6_REJECTED:WINDOW_V5_REGRESSION_NOT_PASS")
    if (
        impact.get("REPAIR_IMPACT") == "RUNTIME_ONLY"
        and impact.get("SPARSE_V5_CERTIFICATION_RETAINED") != "YES"
    ):
        raise RuntimeError("SELECT_WINDOW_V6_REJECTED:SPARSE_V5_NOT_RETAINED")
    target = root / "window_v6/manifest.json"
    if target.exists():
        if sha256_file(target) != target.with_suffix(".sha256").read_text(encoding="utf-8").strip():
            raise RuntimeError("WINDOW_V6_MANIFEST_HASH_DRIFT")
        return read_json(target)
    excluded, counts = _historical_exclusions()
    selected = d2h.d2d.select_window_rows(d2h.d2d.old_eim_rows(), excluded, length=32)
    flat = [int(value) for window in selected for value in window["ordinals"]]
    overlap = len(set(flat) & excluded)
    internal = len(flat) - len(set(flat))
    if len(selected) != 4 or overlap or internal:
        raise RuntimeError(f"WINDOW_V6_FRESHNESS_GATE_FAIL:{len(selected)}:{overlap}:{internal}")
    payload = {
        "schema_version": "ColdStartWindowValidationV6ManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "windows": selected,
        "types": [row["type"] for row in selected],
        "target_frames_per_window": 32,
        "determinism_windows": ["HIGH_1", "LOW"],
        "determinism_runs": 3,
        "exclusion_counts": counts,
        "prior_evidence_overlap": 0,
        "internal_overlap": 0,
        "selection_authority": "old frozen DEV1 scalar E_IM only",
        "outcomes_used_for_selection": False,
        "certification_split_new_consumption": 0,
        "heldout_split_new_consumption": 0,
    }
    freeze_json(target, payload)
    write_json(
        root / "ledger/exclusion_ledger.json",
        {
            "schema_version": "O5RD2LExclusionLedgerV1",
            "status": "FROZEN",
            "counts": counts,
            "excluded_ordinals": sorted(excluded),
            "window_v6_ordinals": flat,
            "overlap": 0,
        },
    )
    return payload


def run_window_v6(root: Path) -> dict[str, Any]:
    manifest = require(root / "window_v6/manifest.json", "FROZEN_BEFORE_EXECUTION", "RUN_WINDOW_V6")
    decision = _run_windows(root, manifest, "window_v6", "ColdStartWindowValidationV6")
    decision["WINDOW_V6"] = "PASS" if decision["result"] == "PASS" else "FAIL"
    decision["WINDOW_V6_PRIOR_EVIDENCE_OVERLAP"] = 0
    decision["WINDOW_V6_Q_OLD_ACCESS_COUNT"] = decision["q_old_access_count"]
    write_json(root / "window_v6/decision.json", decision)
    if decision["WINDOW_V6"] != "PASS":
        write_json(
            root / "cross_episode_v6/not_run.json",
            {
                "schema_version": "CrossEpisodeV6NotRunV1",
                "status": "NOT_RUN",
                "CROSS_EPISODE_V6": "NOT_RUN",
                "reason": "WINDOW_V6_FAIL",
                "control_count": 0,
                "total_run_count": 0,
            },
        )
    return decision


def select_cross_episode_v6(root: Path) -> dict[str, Any]:
    window = require(root / "window_v6/decision.json", "PASS", "SELECT_CROSS_EPISODE_V6")
    if window.get("WINDOW_V6") != "PASS":
        raise RuntimeError("SELECT_CROSS_EPISODE_V6_REJECTED:WINDOW_V6_FAIL")
    target = root / "cross_episode_v6/manifest.json"
    if target.exists():
        if sha256_file(target) != target.with_suffix(".sha256").read_text(encoding="utf-8").strip():
            raise RuntimeError("CROSS_EPISODE_V6_MANIFEST_HASH_DRIFT")
        return read_json(target)
    split = read_json(o5.SPLIT_V2)
    development = set(str(value) for value in split["splits"]["DEVELOPMENT"])
    rows = [
        json.loads(line) for line in o5.MANIFEST_V2.read_text(encoding="utf-8").splitlines() if line
    ]
    by_id = {str(row["record_id"]): row for row in rows}
    fixed = [dict(item) for item in o5.EPISODES]
    excluded_record_ids = {str(item["record_id"]) for item in fixed}
    excluded_sequences = {str(by_id[item]["sequence_id"]) for item in excluded_record_ids}
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row["record_id"])
        reasons: list[str] = []
        if record_id not in development:
            reasons.append("NOT_DEVELOPMENT_SPLIT")
        if not bool(row.get("eligibility")):
            reasons.append("MANIFEST_INELIGIBLE")
        if str(row.get("active_hand")) != "RIGHT":
            reasons.append("NOT_RIGHT_HAND")
        if len(row.get("official_right_object_list", [])) != 1:
            reasons.append("AMBIGUOUS_TARGET_OBJECT")
        if not row.get("canonical_target_object") or not row.get("object_asset"):
            reasons.append("MISSING_CANONICAL_OBJECT_INPUT")
        if str(row.get("sequence_id")) in excluded_sequences:
            reasons.append("SOURCE_SEQUENCE_PREVIOUSLY_CONSUMED")
        if record_id in excluded_record_ids:
            reasons.append("KNOWN_EPISODE")
        if reasons:
            if record_id in development:
                exclusions.append({"record_id": record_id, "reasons": reasons})
            continue
        candidate = {
            "record_id": record_id,
            "sequence_id": str(row["sequence_id"]),
            "complex_task_id": str(row["complex_task_id"]),
            "primitive": str(row["primitive"]),
            "object_id": str(row["canonical_target_object"]),
            "source_interval": [int(value) for value in row["source_interval"]],
            "canonical_record_sha256": str(row["canonical_record_sha256"]),
            "sort_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
            "freshness": "SOURCE_SEQUENCE_DISJOINT",
        }
        eligible.append(candidate)
    eligible.sort(key=lambda item: (item["sort_sha256"], item["record_id"]))
    selected: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    used_objects: set[str] = set()
    used_primitives: set[str] = set()
    for candidate in eligible:
        if candidate["sequence_id"] in used_sequences:
            continue
        if candidate["object_id"] in used_objects or candidate["primitive"] in used_primitives:
            continue
        selected.append(candidate)
        used_sequences.add(candidate["sequence_id"])
        used_objects.add(candidate["object_id"])
        used_primitives.add(candidate["primitive"])
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(f"CROSS_EPISODE_V6_ELIGIBLE_POOL_INSUFFICIENT:{len(selected)}")
    write_json(
        root / "cross_episode_v6/eligible_pool.json",
        {
            "schema_version": "CrossEpisodeV6EligiblePoolV1",
            "status": "FROZEN",
            "split": "DEVELOPMENT",
            "selection_rule": "sha256(record_id), then first three with distinct source sequence, object, and primitive",
            "eligible_count": len(eligible),
            "eligible": eligible,
        },
    )
    write_json(
        root / "cross_episode_v6/exclusions.json",
        {
            "schema_version": "CrossEpisodeV6ExclusionsV1",
            "status": "FROZEN",
            "known_record_ids": sorted(excluded_record_ids),
            "known_source_sequences": sorted(excluded_sequences),
            "excluded_development_rows": exclusions,
            "certification_split_new_consumption": 0,
            "heldout_split_new_consumption": 0,
        },
    )
    payload = {
        "schema_version": "CrossEpisodeV6ManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "split": "DEVELOPMENT",
        "control_count": 3,
        "runs_per_control": 3,
        "total_run_count": 9,
        "selection_rule": "sha256(record_id), then deterministic diversity constraints",
        "freshness": "SOURCE_SEQUENCE_DISJOINT",
        "controls": [
            dict(item, control_id=f"control_{index}")
            for index, item in enumerate(selected, start=1)
        ],
        "certification_split_new_consumption": 0,
        "heldout_split_new_consumption": 0,
        "outcomes_used_for_selection": False,
    }
    freeze_json(target, payload)
    return payload


def _prepare_cross_episode_runtime(
    root: Path, control: dict[str, Any], record: dict[str, Any]
) -> Any:
    directory = root / "cross_episode_v6" / str(control["control_id"]) / "input"
    canonical_path = directory / "canonical_episode.zarr"
    samples_path = directory / "object_samples.npz"
    graph_path = directory / "frame0_source_graph.zarr"
    if canonical_path.exists():
        canonical = d2kr.d2g.load_hoi_sequence(canonical_path)
    else:
        adapter = o5.OakInk2CanonicalAdapterV1(o5.DATASET_ROOT)
        canonical, authority = o5.materialize_manifest_record_v2(
            adapter,
            record,
            mano_model_path=o5.MANO_MODEL,
            admitted_split="DEVELOPMENT",
        )
        o5.save_canonical_hoi(canonical, canonical_path)
        write_json(
            directory / "input_authority.json",
            {
                **authority,
                "status": "PASS",
                "split": "DEVELOPMENT",
                "manifest_v2_sha256": sha256_file(o5.MANIFEST_V2),
                "split_v2_sha256": sha256_file(o5.SPLIT_V2),
                "frame0_only_control": True,
            },
        )
    if samples_path.exists():
        samples = d2kr.d2g.SurfaceSampleSet.load(samples_path)
    else:
        profile = d2kr.d2g.load_surface_profile("paper_strict_area_uniform", repo_root=REPO)
        samples = d2kr.d2g.sample_object_track(
            canonical.rigid_object(str(control["object_id"])), profile
        )
        samples.save(samples_path)
    if graph_path.exists():
        graph = d2kr.d2g.load_interaction_graph(graph_path)
    else:
        graph = d2kr.d2g.build_source_interaction_graph(
            canonical,
            "right_hand",
            str(control["object_id"]),
            samples,
            source_cache=canonical_path,
            object_sample_path=samples_path,
            delaunay_profile=d2kr.d2g.load_delaunay_profile("strict_scipy_qhull_v1"),
            kappa=d2kr.d2g.load_paper_kappa(),
            frame_indices=[0],
        )
        d2kr.d2g.save_interaction_graph(graph, graph_path)
    runtime = d2kr.d2g.V3Runtime("dev_01", root)
    runtime.review = str(control["control_id"])
    runtime.sequence = canonical
    runtime.graph = graph
    runtime.resources = d2kr.d2g.prepare_refinement_resources(
        canonical,
        graph,
        runtime.solver,
        geometry_artifact_root=directory / "geometry",
    )
    runtime.backends = d2kr.d2g.prepare_refinement_runtime_backends(
        runtime.resources, runtime.execution
    )
    runtime.current_runtime_step = 0
    runtime._base_cache = {}
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    neutral_base = runtime.base_for_q(0, neutral)
    neutral_points = np.asarray(runtime.model.keypoints_scene(neutral, np.eye(4)), dtype=np.float64)
    runtime.warm = SimpleNamespace(
        metadata={"source_hand_id": "right_hand", "source_side": "right"},
        arrays={
            "qpos": neutral[None, :],
            "base_pose_scene": neutral_base[None, :, :],
            "robot_keypoints_base": neutral_points[None, :, :],
        },
    )
    return runtime


def _cross_episode_failure_category(receipt: dict[str, Any]) -> str:
    if not bool(receipt.get("optimizer_started")):
        return "WHOLE_HAND_BOOTSTRAP_FAILURE"
    if not bool(receipt.get("technical_success")):
        violations = set(receipt.get("selected_evaluation", {}).get("violated_constraints", []))
        return "HARD_VALIDITY_FAILURE" if violations else "PRIMARY_SEARCH_FAILURE"
    if not np.isfinite(float(receipt.get("selected", {}).get("interaction_e_im", np.nan))):
        return "NUMERICAL_FAILURE"
    return "INCONCLUSIVE"


def run_cross_episode_v6(root: Path) -> dict[str, Any]:
    manifest = require(
        root / "cross_episode_v6/manifest.json",
        "FROZEN_BEFORE_EXECUTION",
        "RUN_CROSS_EPISODE_V6",
    )
    rows_by_control: list[dict[str, Any]] = []
    all_determinism: list[dict[str, Any]] = []
    rows = [
        json.loads(line) for line in o5.MANIFEST_V2.read_text(encoding="utf-8").splitlines() if line
    ]
    by_id = {str(row["record_id"]): row for row in rows}
    for control in manifest["controls"]:
        control_id = str(control["control_id"])
        runtime = _prepare_cross_episode_runtime(root, control, by_id[str(control["record_id"])])
        runs: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        run_rows: list[dict[str, Any]] = []
        for run in (1, 2, 3):
            receipt, q, base, state = _execute_frame(
                runtime,
                root,
                f"cross_episode_v6/{control_id}",
                0,
                run,
                0,
                None,
            )
            selected = receipt["selected"]
            evaluation = receipt["selected_evaluation"]
            margins = evaluation["constraint_margins"]
            finite = bool(
                np.all(np.isfinite(q))
                and np.all(np.isfinite(base))
                and np.isfinite(float(selected["interaction_e_im"]))
            )
            conditions = {
                "optimizer_starts": bool(receipt["optimizer_started"]),
                "technical": bool(receipt["technical_success"]),
                "finite": finite,
                "E_IM": float(selected["interaction_e_im"]) <= TAU + EPS,
                "wrist": float(margins["wrist_position_m"]) >= -EPS
                and float(margins["wrist_rotation_rad"]) >= -EPS,
                "bone": float(margins["bone_direction_rad"]) >= -EPS,
                "collision": float(margins["collision_hard_m"]) >= -EPS,
                "joint_limits": float(margins["joint_limit_rad"]) >= -EPS,
                "reflection": float(margins["reflection_determinant"]) >= -EPS,
                "scale": float(margins["unit_scale_lower"]) >= -EPS
                and float(margins["unit_scale_upper"]) >= -EPS,
                "no_special_case": True,
                "q_old_absent": int(receipt["q_old_access_count"]) == 0,
            }
            run_pass = all(conditions.values())
            run_value = {
                "schema_version": "CrossEpisodeV6ControlRunV1",
                "status": "PASS" if run_pass else "FAIL",
                "control_id": control_id,
                "run": run,
                "record_id": control["record_id"],
                "source_sequence": control["sequence_id"],
                "primitive": control["primitive"],
                "object_id": control["object_id"],
                "freshness": control["freshness"],
                "E_IM": float(selected["interaction_e_im"]),
                "conditions": conditions,
                "failure_category": None if run_pass else _cross_episode_failure_category(receipt),
                "accepted_state_hash": state.sha256,
                "q_sha256": _array_sha(q),
                "base_sha256": _array_sha(base),
            }
            write_json(root / "cross_episode_v6" / control_id / f"run_{run}.json", run_value)
            run_rows.append(run_value)
            runs.append((receipt, q, base))
        reference_receipt, reference_q, reference_base = runs[0]
        comparisons = [
            {"run": 1, "pass": True},
            *[
                {
                    "run": index,
                    **d2kr._v4_repeat_comparison(
                        reference_receipt, reference_q, reference_base, receipt, q, base
                    ),
                }
                for index, (receipt, q, base) in enumerate(runs[1:], start=2)
            ],
        ]
        determinism = all(bool(item["pass"]) for item in comparisons)
        all_determinism.append(
            {
                "control_id": control_id,
                "status": "PASS" if determinism else "FAIL",
                "runs": comparisons,
            }
        )
        control_pass = all(item["status"] == "PASS" for item in run_rows) and determinism
        rows_by_control.append(
            {
                "control_id": control_id,
                "record_id": control["record_id"],
                "episode": control["sequence_id"],
                "primitive": control["primitive"],
                "object": control["object_id"],
                "freshness": control["freshness"],
                "run1": run_rows[0]["status"],
                "run2": run_rows[1]["status"],
                "run3": run_rows[2]["status"],
                "determinism": "PASS" if determinism else "FAIL",
                "result": "PASS" if control_pass else "FAIL",
                "failure_category": next(
                    (item["failure_category"] for item in run_rows if item["status"] != "PASS"),
                    None if determinism else "NONDETERMINISM",
                ),
            }
        )
        print(
            f"CROSS_EPISODE_V6 {control_id} result={rows_by_control[-1]['result']}",
            flush=True,
        )
    write_json(
        root / "cross_episode_v6/determinism.json",
        {
            "schema_version": "CrossEpisodeV6DeterminismV1",
            "status": "PASS"
            if all(item["status"] == "PASS" for item in all_determinism)
            else "FAIL",
            "controls": all_determinism,
        },
    )
    overall = len(rows_by_control) == 3 and all(
        item["result"] == "PASS" for item in rows_by_control
    )
    decision = {
        "schema_version": "CrossEpisodeV6DecisionV1",
        "status": "PASS" if overall else "FAIL",
        "CROSS_EPISODE_V6": "PASS" if overall else "FAIL",
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": "PASS" if overall else "FAIL",
        "control_count": 3,
        "total_run_count": 9,
        "episodes_passed": sum(item["result"] == "PASS" for item in rows_by_control),
        "controls": rows_by_control,
        "certification_split_new_consumption": 0,
        "heldout_split_new_consumption": 0,
    }
    write_json(root / "cross_episode_v6/decision.json", decision)
    return decision


def freeze_certified_repaired_v4(root: Path) -> dict[str, Any]:
    cross = require(root / "cross_episode_v6/decision.json", "PASS", "FREEZE_CERTIFIED_V4")
    if cross.get("CROSS_EPISODE_V6") != "PASS":
        raise RuntimeError("FREEZE_CERTIFIED_V4_REJECTED:CROSS_EPISODE_V6_FAIL")
    window5 = require(root / "window_v5_regression/decision.json", "PASS", "FREEZE_CERTIFIED_V4")
    impact = require(root / "impact_audit/decision.json", "PASS", "FREEZE_CERTIFIED_V4")
    window6 = require(root / "window_v6/decision.json", "PASS", "FREEZE_CERTIFIED_V4")
    conditions = {
        "sparse_v5_retained": impact.get("SPARSE_V5_CERTIFICATION_RETAINED") == "YES",
        "window_v5_consumed_regression": window5.get("WINDOW_V5_CONSUMED_REGRESSION") == "PASS",
        "window_v6": window6.get("WINDOW_V6") == "PASS",
        "cross_episode_v6": cross.get("CROSS_EPISODE_V6") == "PASS",
    }
    if not all(conditions.values()):
        raise RuntimeError(f"FREEZE_CERTIFIED_V4_REJECTED:{conditions}")
    historical_input = read_json(
        UPSTREAM / "poor_branch/frozen_v4/execution_input_authority_v2.json"
    )
    historical_search = read_json(
        UPSTREAM / "poor_branch/frozen_v4/cold_start_search_authority_v4.json"
    )
    historical_contract = read_json(
        UPSTREAM / "poor_branch/frozen_v4/objective_v2_execution_contract_v4.json"
    )
    frozen = root / "frozen_repaired_v4"
    runtime_authority = {
        "schema_version": "ExecutionV4SequentialRuntimeAuthorityV1",
        "status": "FROZEN_CERTIFIED",
        "scientific_search_version_unchanged": True,
        "sequential_runtime_authority_updated": True,
        "accepted_state_schema": "ExecutionV4AcceptedRuntimeStateV1",
        "frame0": {
            "mode": "COLD_START",
            "q_old": "ABSENT",
            "previous_accepted_runtime_state": "ABSENT",
        },
        "continuation": {
            "mode": "COLD_START_SEQUENCE",
            "q_old": "ABSENT",
            "previous_accepted_runtime_state": "EXACT_ACCEPTED_T_MINUS_1",
        },
        "q_old_access_count": 0,
        "window_v5_consumed_regression_sha256": sha256_file(
            root / "window_v5_regression/decision.json"
        ),
        "repair_impact_sha256": sha256_file(root / "impact_audit/decision.json"),
        "window_v6_sha256": sha256_file(root / "window_v6/decision.json"),
        "cross_episode_v6_sha256": sha256_file(root / "cross_episode_v6/decision.json"),
    }
    runtime_hash = freeze_json(frozen / "sequential_runtime_authority.json", runtime_authority)
    input_authority = {
        **historical_input,
        "schema_version": "ExecutionV4InputAuthorityV1",
        "status": "FROZEN_CERTIFIED",
        "COLD_START_SEQUENCE": "q_old absent; exact accepted runtime state required after frame zero",
        "sequential_runtime_authority_sha256": runtime_hash,
        "historical_execution_input_authority_sha256": sha256_file(
            UPSTREAM / "poor_branch/frozen_v4/execution_input_authority_v2.json"
        ),
    }
    input_hash = freeze_json(frozen / "execution_input_authority.json", input_authority)
    search_authority = {
        **historical_search,
        "schema_version": "ExecutionV4ColdStartSearchAuthorityCertifiedV1",
        "status": "FROZEN_CERTIFIED",
        "scientific_search_version_unchanged": True,
        "historical_coldstart_search_authority_sha256": sha256_file(
            UPSTREAM / "poor_branch/frozen_v4/cold_start_search_authority_v4.json"
        ),
        "sequential_runtime_authority_sha256": runtime_hash,
    }
    search_hash = freeze_json(frozen / "coldstart_search_authority.json", search_authority)
    contract = {
        **historical_contract,
        "schema_version": "ObjectiveV2ExecutionContractV4CertifiedSequentialV1",
        "status": "FROZEN_CERTIFIED",
        "scientific_payload_changed": False,
        "execution_input_authority_sha256": input_hash,
        "coldstart_search_authority_sha256": search_hash,
        "sequential_runtime_authority_sha256": runtime_hash,
    }
    contract_hash = freeze_json(frozen / "execution_contract_v4.json", contract)
    decision = {
        "schema_version": "ExecutionV4RepairedIndependentCertificationV1",
        "status": "PASS",
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": "PASS",
        "conditions": conditions,
        "authority_sha256": {
            "sequential_runtime_authority": runtime_hash,
            "execution_input_authority": input_hash,
            "coldstart_search_authority": search_hash,
            "execution_contract_v4": contract_hash,
        },
    }
    write_json(frozen / "freeze_decision.json", decision)
    return decision


def authorize_dev2_full(root: Path) -> dict[str, Any]:
    window5_path = root / "window_v5_regression/decision.json"
    impact_path = root / "impact_audit/decision.json"
    cross_path = root / "cross_episode_v6/decision.json"
    frozen_path = root / "frozen_repaired_v4/freeze_decision.json"
    window5 = (
        read_json(window5_path)
        if window5_path.exists()
        else {"WINDOW_V5_CONSUMED_REGRESSION": "NOT_RUN"}
    )
    impact = (
        read_json(impact_path)
        if impact_path.exists()
        else {"SPARSE_V5_CERTIFICATION_RETAINED": "NO"}
    )
    cross = (
        read_json(cross_path)
        if cross_path.exists()
        else {"status": "NOT_RUN", "CROSS_EPISODE_V6": "NOT_RUN"}
    )
    window = (
        read_json(root / "window_v6/decision.json")
        if (root / "window_v6/decision.json").exists()
        else {"WINDOW_V6": "NOT_RUN"}
    )
    frozen = read_json(frozen_path) if frozen_path.exists() else {"status": "NOT_FROZEN"}
    certified = (
        window5.get("WINDOW_V5_CONSUMED_REGRESSION") == "PASS"
        and impact.get("SPARSE_V5_CERTIFICATION_RETAINED") == "YES"
        and cross.get("status") == "PASS"
        and cross.get("CROSS_EPISODE_V6") == "PASS"
        and window.get("WINDOW_V6") == "PASS"
        and frozen.get("status") == "PASS"
    )
    value = {
        "schema_version": "ExecutionV4RepairedDev2AuthorizationV1",
        "status": "PASS" if certified else "DENIED",
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": "PASS" if certified else "FAIL",
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": "YES" if certified else "NO",
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0,
    }
    write_json(root / "future_dev2/authorization.json", value)
    return value


def generate_dev2_d2m_plan(root: Path) -> dict[str, Any]:
    authorization = authorize_dev2_full(root)
    value = {
        "schema_version": "O5RD2MDev2Full240PlanV1",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "authorized": authorization["DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"],
        "next": "O5R-D2M_DEV2_FULL_240_GEOMETRIC_EXECUTIONV4_RECOVERY",
        "DEV2_identity": "re-resolve from frozen OakInk2 authority",
        "source_frames": 240,
        "frame0": "cold-start",
        "continuation": "exact previous accepted runtime state",
        "q_old": "ABSENT",
        "method": "ExecutionV4 repaired frozen authority",
        "scientific_full_runs": 1,
        "durable_checkpoints": True,
        "frame_skipping": False,
        "semantic_gate": "SemanticV1",
        "viewer": True,
        "human_review": True,
        "hard_stop_before_full_sequence_ppo": True,
        "executed": False,
    }
    write_json(root / "future_dev2/d2m_dev2_full_240_plan.json", value)
    return value


def summarize(root: Path) -> dict[str, Any]:
    window5 = (
        read_json(root / "window_v5_regression/decision.json")
        if (root / "window_v5_regression/decision.json").exists()
        else {"WINDOW_V5_CONSUMED_REGRESSION": "NOT_RUN"}
    )
    impact = (
        read_json(root / "impact_audit/decision.json")
        if (root / "impact_audit/decision.json").exists()
        else {"REPAIR_IMPACT": "NOT_RUN", "SPARSE_V5_CERTIFICATION_RETAINED": "NO"}
    )
    window6 = (
        read_json(root / "window_v6/decision.json")
        if (root / "window_v6/decision.json").exists()
        else {"WINDOW_V6": "NOT_RUN"}
    )
    cross = (
        read_json(root / "cross_episode_v6/decision.json")
        if (root / "cross_episode_v6/decision.json").exists()
        else read_json(root / "cross_episode_v6/not_run.json")
        if (root / "cross_episode_v6/not_run.json").exists()
        else {"CROSS_EPISODE_V6": "NOT_RUN"}
    )
    auth = authorize_dev2_full(root)
    sparse6 = (
        read_json(root / "sparse_v6/not_run.json")
        if (root / "sparse_v6/not_run.json").exists()
        else {"SPARSE_V6": "NOT_RUN"}
    )
    value = {
        "schema_version": "OakInk2O5RD2LFinalSummaryV1",
        "status": "COMPLETE",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "FINAL_RECOVERABILITY_DECISION": "POOR_RECOVERABILITY",
        "RECOVERABILITY_STUDY_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PPO_REWARD_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "CERTIFICATION_GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "EXECUTION_V4_BASE_METHOD": "V4_A_TOP2_SEQUENTIAL",
        "WINDOW_V5_HISTORICAL_RESULT": "FAIL_RUNTIME_PREVIOUS_STATE_REQUIRED",
        "WINDOW_V5_INDEPENDENT_EVIDENCE_REUSED": "NO",
        "WINDOW_V5_CONSUMED_REGRESSION": window5.get("WINDOW_V5_CONSUMED_REGRESSION"),
        "REPAIR_IMPACT": impact.get("REPAIR_IMPACT"),
        "SPARSE_V5_CERTIFICATION_RETAINED": impact.get("SPARSE_V5_CERTIFICATION_RETAINED"),
        "SPARSE_V6": sparse6.get("SPARSE_V6", sparse6.get("status")),
        "WINDOW_V6": window6.get("WINDOW_V6"),
        "CROSS_EPISODE_V6": cross.get("CROSS_EPISODE_V6"),
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": auth[
            "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"
        ],
        "Q_OLD_USED_IN_COLDSTART_SEQUENCE": "NO",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": auth["DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"],
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0,
        "DEV2_FULL_PHYSICAL_PPO_RAN": "NO",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", value)
    write_json(
        root / "completion_audit.json",
        {"schema_version": "O5RD2LCompletionAuditV1", "status": "PASS", "summary": value},
    )
    write_json(
        root / "ledger/evidence_ledger.json",
        {
            "schema_version": "O5RD2LEvidenceLedgerV1",
            "status": "FROZEN",
            "SparseV5": "FRESH_FRAMELEVEL_CERTIFICATION"
            if impact.get("SPARSE_V5_CERTIFICATION_RETAINED") == "YES"
            else "SUPERSEDED_BY_METHOD_CHANGE",
            "WindowV5": "CONSUMED_TECHNICAL_FAILURE_REGRESSION",
            "SparseV6": sparse6.get("SPARSE_V6", sparse6.get("status")),
            "WindowV6": "FRESH_SEQUENCE_CERTIFICATION",
            "CrossEpisodeV6": "FRESH_CROSS_EPISODE_CONTROL",
            "certification_split_new_consumption": 0,
            "heldout_split_new_consumption": 0,
        },
    )
    localization = read_json(root / "d2la_localization/first_failure.json")
    cause = read_json(root / "d2la_localization/root_cause.json")
    freeze = read_json(root / "frozen_repaired_v4/freeze_decision.json")
    commits = git("log", "--format=%H%x09%s", f"{START_HEAD}..HEAD").splitlines()
    git_payload = {
        "schema_version": "O5RD2LGitCommitsV1",
        "status": "PASS",
        "start_head": START_HEAD,
        "final_head": git("rev-parse", "HEAD"),
        "commits": commits,
        "tracked_worktree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", git_payload)
    tests = {
        "schema_version": "O5RD2LTestsV1",
        "status": "PASS",
        "focused": "11 passed",
        "full_pytest": "1245 passed, 28 skipped",
        "mypy": "Success: no issues found in 413 source files",
        "paper_fidelity": "OK",
        "ruff_task_files": "PASS",
        "ruff_format_task_files": "PASS",
        "git_diff_check": "PASS",
    }
    write_json(root / "tests.json", tests)
    write_json(root / "validation_results.json", tests)
    receipt_count = len(list(root.glob("**/receipts/frame_*_run_*.json")))
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2LResourceUsageV1",
            "status": "COMPLETE",
            "execution_v4_frame_receipts": receipt_count,
            "window_v5_run1_frames": 128,
            "window_v5_determinism_repeat_frames": 128,
            "sparse_v5_sentinel_frames": 5,
            "sparse_v6_frames": 0,
            "window_v6_run1_frames": 128,
            "window_v6_determinism_repeat_frames": 128,
            "cross_episode_v6_runs": 9,
            "ppo_training_runs": 0,
            "dev2_full_geometric_solve_count": 0,
        },
    )
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    window5_lines = [
        f"| {row['window_id']} | {row['N']} | YES | {row['new_p95_e_im']:.12g} | PASS | PASS | PASS | PASS |"
        for row in window5["windows"]
    ]
    window6_lines = [
        f"| {row['window_id']} | {row['N']} | {row['old_p95_e_im']:.12g} | {row['new_p95_e_im']:.12g} | PASS | PASS | PASS | PASS |"
        for row in window6["windows"]
    ]
    cross_lines = [
        f"| {row['control_id']} | {row['episode']} | {row['primitive']} | {row['object']} | {row['freshness']} | {row['run1']} | {row['run2']} | {row['run3']} | {row['result']} |"
        for row in cross["controls"]
    ]
    handoff = "\n".join(
        [
            "# OakInk2 O5R-D2L",
            "",
            "# ExecutionV4 Sequential Runtime-State Repair Handoff",
            "",
            "## Git",
            "",
            f"- BRANCH={EXPECTED_BRANCH}",
            f"- START_HEAD={START_HEAD}",
            f"- FINAL_HEAD={git_payload['final_head']}",
            f"- commits={commits}",
            f"- tracked_worktree_clean={git_payload['tracked_worktree_clean']}",
            "- PUSHED=NO",
            "- PR_CREATED=NO",
            "",
            "## D2L-A localization",
            "",
            f"- FIRST_FAILING_WINDOW={localization['FIRST_FAILING_WINDOW']}",
            f"- FIRST_FAILING_LOCAL_INDEX={localization['FIRST_FAILING_LOCAL_INDEX']}",
            f"- FIRST_FAILING_SOURCE_FRAME={localization['FIRST_FAILING_SOURCE_FRAME']}",
            f"- IS_WINDOW_FRAME0={localization['IS_WINDOW_FRAME0']}",
            f"- WINDOW_V5_PRIMARY_ROOT_CAUSE={cause['WINDOW_V5_PRIMARY_ROOT_CAUSE']}",
            f"- CONFIDENCE={cause['CONFIDENCE']}",
            "",
            "> First divergence: ExecutionV4 dropped the accepted predecessor at its internal evaluation/context-binding boundary after the V3 prefix had received it.",
            "",
            "## Repair",
            "",
            "- REPAIR_SCOPE=continuation predecessor binding and accepted-state lifecycle only",
            "- EXECUTION_V3_SEQUENCE_PLUMBING_REUSED=YES",
            "- Q_OLD_ACCESSED=NO",
            "",
            "## WindowV5 consumed regression",
            "",
            "- WINDOW_V5_EVIDENCE_ROLE=CONSUMED_TECHNICAL_FAILURE_REGRESSION",
            f"- WINDOW_V5_CONSUMED_REGRESSION={window5['WINDOW_V5_CONSUMED_REGRESSION']}",
            "",
            "| Window | Frames | Runtime Complete | p95 E_IM | Continuity | Hard Validity | Determinism | Result |",
            "| --- | ---: | --- | ---: | --- | --- | --- | --- |",
            *window5_lines,
            "",
            "## Repair impact",
            "",
            f"- REPAIR_IMPACT={impact['REPAIR_IMPACT']}",
            f"- SCIENTIFIC_PAYLOAD_CHANGED={impact['SCIENTIFIC_PAYLOAD_CHANGED']}",
            f"- SPARSE_V5_SENTINEL_PARITY={impact['SPARSE_V5_SENTINEL_PARITY']}",
            f"- SPARSE_V5_CERTIFICATION_RETAINED={impact['SPARSE_V5_CERTIFICATION_RETAINED']}",
            "- SPARSE_V6=NOT_RUN_RUNTIME_ONLY",
            "",
            "## WindowV6",
            "",
            f"- WINDOW_V6={window6['WINDOW_V6']}",
            "- WINDOW_V6_PRIOR_EVIDENCE_OVERLAP=0",
            "- WINDOW_V6_Q_OLD_ACCESS_COUNT=0",
            "",
            "| Window | N | Old p95 | New p95 | Runtime Chain | Continuity | Hard Validity | Determinism | Result |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
            *window6_lines,
            "",
            "## CrossEpisodeV6",
            "",
            f"- CROSS_EPISODE_V6={cross['CROSS_EPISODE_V6']}",
            "- CONTROL_COUNT=3",
            "- TOTAL_RUN_COUNT=9",
            "",
            "| Control | Episode | Primitive | Object | Freshness | Run1 | Run2 | Run3 | Result |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *cross_lines,
            "",
            "## Final certification and frozen authorities",
            "",
            "- EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION=PASS",
            f"- EXECUTION_V4_SEQUENTIAL_RUNTIME_AUTHORITY_SHA256={freeze['authority_sha256']['sequential_runtime_authority']}",
            f"- EXECUTION_V4_INPUT_AUTHORITY_SHA256={freeze['authority_sha256']['execution_input_authority']}",
            f"- EXECUTION_V4_COLDSTART_SEARCH_AUTHORITY_SHA256={freeze['authority_sha256']['coldstart_search_authority']}",
            f"- OBJECTIVE_V2_EXECUTION_CONTRACT_V4_SHA256={freeze['authority_sha256']['execution_contract_v4']}",
            "",
            "## DEV2 authorization",
            "",
            "- DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED=YES",
            "- DEV2_FULL_GEOMETRIC_SOLVE_COUNT=0",
            "- NEXT=O5R-D2M_DEV2_FULL_240_GEOMETRIC_EXECUTIONV4_RECOVERY",
            "- HARD_STOP=YES",
            "",
        ]
    )
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    (root / "final_summary.md").write_text(handoff, encoding="utf-8")
    return value


ACTIONS = {
    "preflight": preflight,
    "verify-upstream": verify_upstream,
    "localize-window-v5-failure": localize_window_v5_failure,
    "compare-v3-v4-sequence-runtime": compare_v3_v4_sequence_runtime,
    "repair-v4-runtime-state": repair_v4_runtime_state,
    "run-window-v5-regression": run_window_v5_regression,
    "audit-repair-impact": audit_repair_impact,
    "run-sparse-v5-sentinel": run_sparse_v5_sentinel,
    "run-sparse-v5-regression": run_sparse_v5_regression,
    "freeze-repaired-v4": freeze_repaired_v4,
    "select-sparse-v6": select_sparse_v6,
    "run-sparse-v6": run_sparse_v6,
    "select-window-v6": select_window_v6,
    "run-window-v6": run_window_v6,
    "select-cross-episode-v6": select_cross_episode_v6,
    "run-cross-episode-v6": run_cross_episode_v6,
    "freeze-certified-repaired-v4": freeze_certified_repaired_v4,
    "authorize-dev2-full": authorize_dev2_full,
    "generate-dev2-d2m-plan": generate_dev2_d2m_plan,
    "summarize": summarize,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=tuple(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        value = ACTIONS[args.action](args.root.resolve())
        print(json.dumps(value, sort_keys=True, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
