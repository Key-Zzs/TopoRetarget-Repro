#!/usr/bin/env python3
"""Audit the pre-repair C.5A baseline against the repaired qualification path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
BASELINE_SCRIPT = Path("scripts/rl/isaaclab/calibrate_stage16c5_replication_noise.py")
REPAIR_HELPERS = Path("src/toporetarget/rl/isaaclab_oracle/history_replay.py")
DIAGNOSTIC_SCRIPT = Path("scripts/rl/isaaclab/diagnose_stage16c5_natural_nondeterminism.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--baseline-revision", default="HEAD")
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_AUDIT_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _historical_source(revision: str, path: Path) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        cwd=REPO_ROOT,
        text=True,
    )


def _contains(source: str, token: str) -> bool:
    return token in source


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    baseline_before = _historical_source(args.baseline_revision, BASELINE_SCRIPT)
    baseline_current = (REPO_ROOT / BASELINE_SCRIPT).read_text(encoding="utf-8")
    repair_helpers = (REPO_ROOT / REPAIR_HELPERS).read_text(encoding="utf-8")
    diagnostic = (REPO_ROOT / DIAGNOSTIC_SCRIPT).read_text(encoding="utf-8")

    audit = {
        "schema_version": "stage16c5_baseline_harness_audit_v1",
        "baseline_revision": args.baseline_revision,
        "source_hashes": {
            "historical_baseline": _sha256_text(baseline_before),
            "current_baseline": _sha256_text(baseline_current),
            "repair_helpers": _sha256_text(repair_helpers),
            "natural_diagnostic": _sha256_text(diagnostic),
        },
        "historical_comparison": {
            "process_scope": "same-process, one 33-environment vector scene",
            "reference_population": "env 0 expanded against envs 1..32",
            "trial_scope": "20 sequential all-environment resets per clip and phase",
            "cross_process_mixed_into_formal_gate": False,
            "evidence": [
                "historical --num-envs default is 33",
                "historical _metrics expands view[:1] over view[1:]",
                "historical script has no subprocess invocation",
            ],
        },
        "current_qualification_boundary": {
            "reset_sync_order": ["write_data_to_sim", "sim.forward", "scene.update"],
            "manual_step_order": [
                "_pre_physics_step",
                "per-substep _apply_action/write/step/update",
                "episode/common counters",
                "_get_dones",
                "_get_rewards",
            ],
            "auto_reset_suppressed_for_measurement": True,
            "cuda_sync_before_capture": _contains(diagnostic, "torch.cuda.synchronize"),
            "state_capture_cloned": _contains(diagnostic, "_clone(raw)"),
            "scene_local_view": _contains(diagnostic, "_origin_invariance"),
            "no_snapshot_restore": _contains(diagnostic, '"snapshot_restore_used": False'),
        },
        "required_measurement_coverage": {
            "raw_state": [
                "26 q/qdot",
                "articulation root",
                "both object pose/twist",
                "drive targets",
                "action history",
                "controller/reference/episode state",
            ],
            "derived_state": [
                "scene-local wrist/object state",
                "observation",
                "reward components and total",
                "termination/reason masks",
            ],
            "implemented_by": "capture_candidate_state plus state_view and _last_reward_terms",
        },
    }
    defects = {
        "schema_version": "stage16c5_harness_repairs_v1",
        "defects": [
            {
                "id": "RESET_BOUNDARY_NOT_MATERIALIZED",
                "class": "trial_boundary",
                "historical_evidence": (
                    "_set_clip_frame_zero called _reset_idx without write_data_to_sim, "
                    "sim.forward, scene.update"
                ),
                "repair": (
                    "reset_frozen_clip_frame_zero and reset_candidates_to_frame_zero "
                    "call synchronize_reset_boundary"
                ),
                "unit_test": "test_reset_boundary_sync_matches_direct_env_reset_order",
                "status": "REPAIRED_AND_RETEST_REQUIRED",
            },
            {
                "id": "MANUAL_STEP_OMITTED_DIRECT_RL_BOOKKEEPING",
                "class": "step_boundary",
                "historical_evidence": (
                    "historical raw_control_step did not update counters or compute done "
                    "masks before reward"
                ),
                "repair": (
                    "raw_control_step now preserves DirectRLEnv ordering through "
                    "done/reward while retaining terminal state"
                ),
                "unit_test": "test_raw_control_step_keeps_terminal_state_but_preserves_step_order",
                "status": "REPAIRED_AND_RETEST_REQUIRED",
            },
            {
                "id": "FIRST_STEP_REWARD_CACHE_UNMATERIALIZED",
                "class": "framework_compatibility",
                "historical_evidence": (
                    "live Isaac Lab environment has no reward_buf before its first public step"
                ),
                "repair": (
                    "raw_control_step materializes output-only reward_buf after _get_rewards "
                    "when absent"
                ),
                "unit_test": "test_raw_control_step_allocates_a_missing_framework_reward_cache",
                "status": "REPAIRED_AND_RETEST_REQUIRED",
            },
            {
                "id": "REWARD_COMPONENTS_NOT_REPORTED",
                "class": "reporting",
                "historical_evidence": (
                    "historical _metrics compared only _last_reward_terms['total']"
                ),
                "repair": "natural diagnostic records every _last_reward_terms component and total",
                "status": "REPAIRED_AND_RETEST_REQUIRED",
            },
        ],
        "explicitly_not_a_defect": {
            "cross_process_scope": (
                "historical formal baseline was already same-process; E4/E5 remain diagnostics only"
            ),
            "hard_caps": "unchanged",
            "physics_or_solver_configuration": "unchanged",
        },
    }
    reward = {
        "schema_version": "stage16c5_reward_component_audit_v1",
        "historical": {"total_only": True, "reduction": "per-environment absolute difference"},
        "repaired_measurement": {
            "components": [
                "object",
                "tracked_links",
                "finger_joints",
                "wrist_position",
                "wrist_rotation",
                "smoothness",
                "total",
            ],
            "comparison": (
                "per-environment component-wise absolute difference; no cross-environment reduction"
            ),
            "reference_index_boundary": "captured after the same manual DirectRLEnv step boundary",
        },
    }
    scope = """# Stage 16-C.5A baseline comparison scope

The historical formal baseline was already a same-process, 33-environment
vector-scene comparison: env 0 was expanded against 32 peers after each frozen
frame-zero reset. It did not invoke child processes, so E4/E5 are retained
strictly as reproducibility diagnostics rather than being removed from or added
to the O1 gate.

The repaired formal gate remains E2: source and candidates share one process,
scene, reset boundary, action history, phase frame, and frozen input contract.
The only scope repair is to record E1 and E4/E5 separately so single-environment
and cross-process results are not misrepresented as candidate-pool equivalence.
Hard caps and tolerance formulas are unchanged.
"""

    _write(report_dir / "baseline_harness_audit.json", audit)
    _write(report_dir / "baseline_comparison_scope.md", scope)
    _write(report_dir / "reward_component_audit.json", reward)
    _write(report_dir / "suspected_defects.json", defects)
    print(json.dumps({"result": "STAGE16C5A_BASELINE_AUDIT_COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
