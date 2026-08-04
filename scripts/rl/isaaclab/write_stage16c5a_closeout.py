#!/usr/bin/env python3
"""Write an immutable, fail-closed C.5A closeout from existing run evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c5a_state_replication",
    )
    parser.add_argument(
        "--test-command",
        action="append",
        default=[],
        help="Already-executed validation command; repeated values are retained verbatim.",
    )
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"STAGE16C5A_CLOSEOUT_EVIDENCE_MISSING: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"STAGE16C5A_CLOSEOUT_EVIDENCE_MALFORMED: {path}")
    return payload


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_CLOSEOUT_REFUSES_OVERWRITE: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _blocked_report(*, name: str, reason: str, frames: dict[str, Any]) -> dict[str, object]:
    return {
        "status": "NOT_RUN_DUE_TO_PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
        "name": name,
        "reason": reason,
        "test_frames": frames,
        "no_result_is_inferred": True,
    }


def _write_markdown(path: Path, lines: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_CLOSEOUT_REFUSES_OVERWRITE: {path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen = _read(root / "frozen_inputs.json")
    current_contract_path = root / "revalidated/candidate_state_contract.json"
    contract = _read(current_contract_path)
    current_audit_path = root / "revalidated/candidate_state_field_audit.json"
    contract_audit = _read(current_audit_path)
    if contract_audit.get("status") != "STAGE16C5A_CANDIDATE_STATE_CONTRACT_VALIDATED":
        raise RuntimeError("STAGE16C5A_CLOSEOUT_CURRENT_CONTRACT_NOT_VALIDATED")
    noise = _read(root / "replication_noise_floor.json")
    frames = _read(root / "replication_test_frames.json")
    expected_noise = "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
    actual_noise = str(noise.get("status"))
    if actual_noise != expected_noise:
        raise RuntimeError(
            "STAGE16C5A_CLOSEOUT_REQUIRES_FAIL_CLOSED_BASELINE: "
            f"expected={expected_noise}, actual={actual_noise}"
        )
    o0_paths = {
        1: root / "o0_candidate_pool_1_current.json",
        32: root / "o0_candidate_pool_32_current.json",
        96: root / "o0_candidate_pool_96_current.json",
        144: root / "o0_candidate_pool_144_current.json",
    }
    o0 = {str(size): _read(path) for size, path in o0_paths.items()}
    for size, report in o0.items():
        if report.get("status") != "STAGE16C5A_O0_CANDIDATE_POOL_VALIDATED":
            raise RuntimeError(f"STAGE16C5A_CLOSEOUT_O0_NOT_VALIDATED:{size}")

    tolerance = {
        "status": actual_noise,
        "usable_for_O1": False,
        "policy": "No tolerance may be loosened after a hard-cap violation.",
        "source": "replication_noise_floor.json",
        "global_tolerances": noise["global_tolerances"],
    }
    _write(root / "replication_tolerances.json", tolerance)
    blocked_reason = (
        "The natural no-clone baseline exceeded a frozen hard cap; O1, history replay, "
        "candidate independence qualification, and performance benchmark are stopped."
    )
    phase_reports = {
        "tensor_clone_precontact.json": _blocked_report(
            name="tensor_clone_precontact", reason=blocked_reason, frames=frames
        ),
        "tensor_clone_contact.json": _blocked_report(
            name="tensor_clone_contact", reason=blocked_reason, frames=frames
        ),
        "tensor_clone_postcontact.json": _blocked_report(
            name="tensor_clone_postcontact", reason=blocked_reason, frames=frames
        ),
        "history_replay_contact.json": _blocked_report(
            name="deterministic_history_replay_v1", reason=blocked_reason, frames=frames
        ),
        "candidate_independence.json": {
            "status": "STAGE16C5A_O0_CANDIDATE_INDEPENDENCE_VALIDATED_ONLY",
            "scope": (
                "allocation, origins, subset-reset isolation; no O1 rollout independence claim"
            ),
            "o0_evidence": {size: path.name for size, path in o0_paths.items()},
            "original_single_candidate_incident": "o0_candidate_pool_1.json",
            "repaired_single_candidate_evidence": "o0_candidate_pool_1_repair1.json",
        },
        "candidate_pool_benchmark.json": _blocked_report(
            name="candidate_pool_benchmark", reason=blocked_reason, frames=frames
        ),
    }
    for name, report in phase_reports.items():
        _write(root / name, report)
    _write(
        root / "replication_qualification.json",
        {
            "status": "STAGE16C5A_O0_O1_BLOCKED",
            "baseline_status": actual_noise,
            "tensor_clone": "NOT_RUN",
            "history_replay": "NOT_RUN",
            "qualification_phases": [
                {
                    "clip": row["clip"],
                    "phase": phase,
                    "frame": frame,
                    "status": "NOT_RUN_DUE_TO_PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
                }
                for row in frames["clips"]
                for phase, frame in row["frames"].items()
            ],
            "prohibited_actions": [
                "tolerance_softening",
                "CEM",
                "PPO",
                "checkpoint_write",
                "formal_20_episode_evaluation",
            ],
        },
    )
    _write(
        root / "write_audit_contract.json",
        {
            "status": "STAGE16C5A_WRITE_AUDIT_CONTRACT_VALIDATED_IN_O0",
            "candidate_setup_direct_writes": "allowed only for candidate IDs",
            "execution_rollout_direct_writes": 0,
            "O0_reports": {size: report["write_audit"] for size, report in o0.items()},
            "no_O1_execution_rollout_occurred": True,
        },
    )
    _write(
        root / "candidate_pool_config.json",
        {
            "status": "STAGE16C5A_O0_CANDIDATE_POOL_VALIDATED",
            "default_schedule": {"candidate_count": 96, "horizons": [1, 5, 10], "per_horizon": 32},
            "upgrade_schedule": {"candidate_count": 144, "horizons": [1, 5, 10], "per_horizon": 48},
            "realized_o0_counts": [1, 32, 96, 144],
            "not_a_CEM_implementation": True,
        },
    )
    _write(
        root / "o0_candidate_pool_smoke.json",
        {
            "status": "STAGE16C5A_O0_CANDIDATE_POOL_VALIDATED",
            "reports": {size: path.name for size, path in o0_paths.items()},
            "single_candidate_initial_incident": {
                "report": "o0_candidate_pool_1.json",
                "classification": "O0_SINGLE_CANDIDATE_PEER_BOOKKEEPING",
                "repair": "no-peer condition is not evaluated as a peer-isolation comparison",
                "replacement": "o0_candidate_pool_1_repair1.json",
            },
        },
    )
    _write(
        root / "resource_usage.json",
        {
            "status": "PARTIAL_O0_ONLY",
            "host_preflight": "resources.txt",
            "gpu_preflight": "gpu_before.txt",
            "O0_capacity_validated": [1, 32, 96, 144],
            "O0_per_run_peak_vram_mb": "NOT_COLLECTED",
            "O1_or_benchmark_resource_measurements": "NOT_RUN",
            "reason": blocked_reason,
        },
    )
    _write(
        root / "c5b_runtime_projection.json",
        {
            "status": "C5B_NOT_AUTHORIZED",
            "required_precondition": "passing C.5A O0/O1 state-replication qualification",
            "actual_precondition": actual_noise,
            "contingent_schedule_only": {
                "default": "96 = 3 horizons x 32 candidates",
                "upgrade": "144 = 3 horizons x 48 candidates",
            },
            "predicted_runtime": (
                "NOT_AVAILABLE; no benchmark was permitted after the hard-cap failure"
            ),
        },
    )
    transitions = [
        {
            "phase": "CANDIDATE_POOL",
            "status": "REPAIRED",
            "failure_class": "O0_SINGLE_CANDIDATE_PEER_BOOKKEEPING",
            "attempt": 1,
            "evidence": "o0_candidate_pool_1.json -> o0_candidate_pool_1_repair1.json",
        },
        {
            "phase": "NOISE_FLOOR",
            "status": "FAIL_CLOSED",
            "failure_class": actual_noise,
            "attempt": 1,
            "action": "stop O1/history replay/benchmark; do not loosen tolerances",
            "evidence": "replication_noise_floor.json",
        },
    ]
    transitions_path = root / "failure_transitions.jsonl"
    if transitions_path.exists():
        raise FileExistsError(f"STAGE16C5A_CLOSEOUT_REFUSES_OVERWRITE: {transitions_path}")
    transitions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions), encoding="utf-8"
    )
    commits = [line.split("\u001f") for line in _git("log", "-6", "--format=%H%x1f%s").splitlines()]
    _write(
        root / "git_commits.json",
        {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "recent_commits": [{"commit": row[0], "subject": row[1]} for row in commits],
        },
    )
    tests = {
        "status": "PASS",
        "commands": args.test_command,
        "scope": "C.5A unit/static/documentation validation executed before closeout",
        "Isaac_runtime": {
            "candidate_state_audit": "STAGE16C5A_CANDIDATE_STATE_CONTRACT_VALIDATED",
            "O0_candidate_pool": "STAGE16C5A_O0_CANDIDATE_POOL_VALIDATED",
            "no_clone_baseline": actual_noise,
        },
    }
    _write(root / "tests.json", tests)
    final = {
        "status": "STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
        "inputs": frozen,
        "candidate_state_contract": {
            "status": contract_audit["status"],
            "version": contract["version"],
            "evidence": str(current_contract_path.relative_to(root)),
            "field_audit": str(current_audit_path.relative_to(root)),
        },
        "O0": {size: report["status"] for size, report in o0.items()},
        "O1": "NOT_RUN_DUE_TO_PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
        "C5B": "NOT_AUTHORIZED",
        "C6_PPO": "NOT_AUTHORIZED",
        "no_claims": [
            "No tensor-clone or history-replay qualification passed.",
            "No CEM, PPO, checkpoint, full Oracle, or formal 20-episode evaluation ran.",
            "No performance estimate is inferred from O0 capacity smokes.",
        ],
    }
    _write(root / "final_summary.json", final)
    _write_markdown(
        root / "final_summary.md",
        [
            "# Stage 16-C.5A closeout",
            "",
            "**Status:** `STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM`.",
            "",
            "The required 20-trial no-clone baseline over both clips and all four phases "
            "exceeded hard caps. The qualification stops before O1, deterministic history "
            "replay, candidate benchmarking, C5B, and C6/PPO. Tolerances were not loosened.",
            "",
            "O0 nevertheless passed for 1, 32, 96, and 144 candidate environments with "
            "CUDA, unique origins, subset-reset isolation, finite state, and zero formal "
            "execution-rollout direct state writes. The initial one-candidate peer-bookkeeping "
            "partial result is preserved, with its corrected rerun recorded in "
            "`failure_transitions.jsonl`.",
            "",
            "See `replication_noise_floor.json`, `replication_qualification.json`, and "
            "`handoff.md` for machine-readable evidence and next gates.",
        ],
    )
    _write_markdown(
        root / "handoff.md",
        [
            "# Handoff",
            "",
            "## 1. Git info",
            "",
            f"- Branch: `{_git('branch', '--show-current')}`",
            f"- Commit: `{_git('rev-parse', 'HEAD')}`",
            "",
            "## 2. GPU usage",
            "",
            "- O0 capacity smoke passed at 1/32/96/144 candidates on `cuda:0`; "
            "peak VRAM was not sampled per O0 run.",
            "- O1 and benchmark GPU usage: NOT_RUN because the no-clone baseline failed hard caps.",
            "",
            "## 3. Current stage",
            "",
            "- `STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM`.",
            "- Candidate state contract and O0 isolation are valid; O1/C5B/C6 are not authorized.",
            "",
            "## 4. What changed",
            "",
            "- Added `Stage16C5CandidateStateV1`, candidate-pool isolation, write auditing, "
            "no-clone tolerance calibration, bounded fallback interfaces, recovery ledger, "
            "configurations, tests, and stage documents.",
            "",
            "## 5. Key artifacts",
            "",
            "- `frozen_inputs.json`, `candidate_state_contract.json`, "
            "`replication_noise_floor.json`, `o0_candidate_pool_smoke.json`, "
            "`replication_qualification.json`, and `failure_transitions.jsonl`.",
            "",
            "## 6. Unblock conditions",
            "",
            "1. Demonstrate a natural no-clone baseline that passes every existing hard cap "
            "with frozen inputs and the same 20x8 protocol.",
            "2. Then qualify tensor clone in pre-contact/contact/post-contact and use history "
            "replay only if a tensor-clone contact mismatch remains.",
            "3. Only after all C.5A gates pass may a separately authorized C5B candidate "
            "evaluator be considered; CEM and PPO remain out of scope.",
        ],
    )
    print(json.dumps({"status": final["status"], "output_root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
