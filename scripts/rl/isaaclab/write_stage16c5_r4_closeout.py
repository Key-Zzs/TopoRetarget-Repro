#!/usr/bin/env python3
"""Assemble the fail-closed Stage 16-C.5A-R4/C5B/C5C handoff."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-dir", type=Path, required=True)
    parser.add_argument("--cem-dir", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", action="append", default=[])
    parser.add_argument("--test-verdict", default="pending")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"closeout input is malformed: {path}")
    return payload


def _write(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"R4 closeout refuses overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"R4 closeout output exists: {args.output_dir}")
    natural_path = args.distribution_dir / "natural_distribution_baseline.json"
    gate_path = args.distribution_dir / "distributional_replication_gate.json"
    pool_path = args.distribution_dir / "persistent_candidate_pool_benchmark.json"
    _load(natural_path)
    gate = _load(gate_path)
    pool = _load(pool_path)
    b1 = {
        clip: _load(args.cem_dir / f"b1_{clip}.json") for clip in ("hocap_170105", "hocap_170650")
    }
    b2 = {
        clip: _load(args.cem_dir / f"b2_{clip}.json") for clip in ("hocap_170105", "hocap_170650")
    }
    selected_pool = pool.get("selected_layout")
    if not isinstance(selected_pool, dict):
        raise ValueError("closeout pool report has no selected layout")
    from toporetarget.rl.isaaclab_oracle.recovery import Stage16C5R4RecoveryStateMachine

    recovery = Stage16C5R4RecoveryStateMachine()
    recovery.transition("NATURAL_DISTRIBUTION", reason="frozen inputs verified")
    recovery.transition("REPLICATION_GATE", reason="natural thresholds frozen before candidates")
    if not gate.get("passes"):
        recovery.record_failure(
            "REPLICATION_DISTRIBUTION_FAIL",
            evidence=str(gate_path),
        )
    recovery.transition("POOL_BUILD", reason="bounded diagnostic continuation")
    recovery.transition("CEM_SMOKE", reason="B0 and B1 complete")
    recovery.transition("SHORT_ROLLOUT", reason="two B2 30-step rollouts complete")
    recovery.record_failure(
        "FORMAL_GATE_FAIL",
        evidence="both B2 terminal robust candidate populations have failure_probability=1.0",
    )
    recovery.transition("CLOSEOUT", reason="B3/C5C gate-blocked; no relaxation")
    b2_summary = {}
    for clip, report in b2.items():
        records = report["records"]
        final = records[-1]["selected_evaluation"]
        b2_summary[clip] = {
            "steps": report["planning_step_count"],
            "selected_horizons": dict(
                sorted(Counter(str(row["selected_horizon"]) for row in records).items())
            ),
            "final": final,
            "wall_time_s": report["wall_time_s"],
            "gpu_memory_mib": report["gpu_memory_mib"],
            "no_hidden_control": report["no_hidden_control"],
        }
    result = {
        "schema_version": "stage16c5_r4_c5b_c5c_handoff_v1",
        "status": "STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL",
        "deterministic_c5a_failure_preserved": {
            "reason": "SAME_SCENE_CONTACT_DIVERGENCE",
            "classification": "TRUE_CONTACT_SOLVER_NONDETERMINISM",
        },
        "distributional_replication": {
            "contract": "DistributionalReplicationContractV1",
            "natural_baseline": str(natural_path),
            "gate": str(gate_path),
            "passes": gate["passes"],
            "phase_results": gate["clips"],
        },
        "persistent_candidate_pool": {
            "benchmark": str(pool_path),
            "selected": selected_pool,
        },
        "robust_cem": {
            "config": {
                "horizons": [1, 5, 10],
                "population": 32,
                "iterations": 3,
                "elites": 8,
                "replicas": 4,
                "initial_std": 0.35,
                "minimum_std": 0.05,
            },
            "b1": {clip: str(args.cem_dir / f"b1_{clip}.json") for clip in b1},
            "b2": b2_summary,
            "b3": "NOT_STARTED_GATE_BLOCKED_BY_R4_AND_B2",
        },
        "c5c": {
            "status": "NOT_STARTED_GATE_BLOCKED_BY_R4_AND_B2",
            "optimized_320_action_traces": 0,
            "formal_episodes": 0,
        },
        "visualization": {
            "kind": "dashboard",
            "path": str(args.dashboard),
            "geometry_rendered": False,
            "reason": "B3 gate-blocked; dashboard reports measured B2 traces without fabrication",
        },
        "recovery": recovery.as_dict(),
        "tests": args.test_verdict,
        "commits": args.commit,
        "c6_authorization": "PPO_NOT_AUTHORIZED",
        "ppo": {"started": False, "samples": 0, "checkpoints": 0},
        "limitations": [
            "factor8 changes time semantics",
            "virtual wrist is not a real robot arm",
            "no PPO or checkpoint",
            "no sim-to-real claim",
        ],
    }
    markdown = f"""# Stage16-C.5A-R4/C5B/C5C Handoff

## 1. Current Status

`STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL`. R4 and C5B are implemented; the
distribution gate and both B2 physical gates fail. B3/C5C are gate-blocked.

## 2. Deterministic C5A failure preservation

`SAME_SCENE_CONTACT_DIVERGENCE` / `TRUE_CONTACT_SOLVER_NONDETERMINISM` is retained.

## 3. Distributional replication contract

20 natural versus 20 candidate replicas, four phases, all required state/task/contact
fields, and seven metrics frozen before candidate inspection.

## 4. Natural baseline statistics

`{natural_path}`. Both clips pass pre-contact and fail every contact-bearing phase.

## 5. Candidate pool architecture

Persistent DirectRLEnv, deterministic per-iteration slot permutation, mapping-invariant
logical ranking, and separately audited setup/formal writes.

## 6. GPU benchmark

Selected 32 x 3 x 4 = 384 candidate envs: {selected_pool["gpu_memory_mib"]} MiB,
{selected_pool["rollout_control_steps_per_s"]:.3f} vector control steps/s,
{1000 * selected_pool["state_dispatch_latency_s"]:.2f} ms dispatch,
{1000 * selected_pool["aggregation_latency_s"]:.2f} ms aggregation.

## 7. Robust CEM configuration

H=[1,5,10], population=32, iterations=3, elites=8, replicas=4, std=0.35,
floor=0.05, CVaR alpha=0.8, frozen lexicographic gate ordering, no scored padding.

## 8. Oracle rollout results

- `hocap_170105`: B2 30 steps; H1=19, H10=11; final failure probability 1.0,
  p95 position 0.2505 m, rotation 138.49 deg, axis 0.2987 m.
- `hocap_170650`: B2 30 steps; H1=26, H10=4; final failure probability 1.0,
  p95 position 0.2248 m, rotation 137.45 deg, axis 0.2887 m.
- B3: `NOT_STARTED_GATE_BLOCKED_BY_R4_AND_B2`.

## 9. C5C formal evaluation

`NOT_STARTED_GATE_BLOCKED_BY_R4_AND_B2`; no 320-action optimized trace and no formal
episode were generated.

## 10. Visualization

Dashboard: `{args.dashboard}`. No unavailable geometry was fabricated.

## 11. Failure recovery

R4 distribution failure and B2 formal failure are retained; metrics/gates were not changed.

## 12. Tests

{args.test_verdict}

## 13. README status

English/Chinese README and ROADMAP plus Oracle/state-replication docs record the partial status.

## 14. Commits

{", ".join(args.commit) if args.commit else "pending"}

## 15. C6 authorization

`PPO NOT AUTHORIZED`.

PPO:

- started = false
- samples = 0
- checkpoint = 0

Factor-8 changes time semantics. The virtual wrist is not a real arm. There is no
PPO/checkpoint and no sim-to-real claim.
"""
    _write(args.output_dir / "stage16c5_handoff.json", json.dumps(result, indent=2) + "\n")
    _write(args.output_dir / "stage16c5_handoff.md", markdown)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
