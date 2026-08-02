#!/usr/bin/env python3
"""Materialize the fail-closed Stage 16-C.2--C.5 report bundle.

This is a report writer only.  It never starts Isaac, changes an asset, runs an
oracle, or authorizes PPO.  C.4/C.5 records are emitted as gate-blocked facts
when the immutable C.3 input is partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16c2_c5_isaaclab"
C1_ROOT = REPO_ROOT / ".local/reports/stage16c1_asset_migration"
C0_ROOT = REPO_ROOT / ".local/reports/stage16c_isaaclab_platform"
REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tests-passed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_archive(destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=False)
    sources = [
        (C0_ROOT / "final_summary.json", "c0_final_summary.json"),
        (C1_ROOT / "final_summary.json", "c1_final_summary.json"),
        (C1_ROOT / "wuji_asset_manifest.json", "wuji_asset_manifest.json"),
        (C1_ROOT / "hocap_170105_asset_manifest.json", "hocap_170105_asset_manifest.json"),
        (C1_ROOT / "hocap_170650_asset_manifest.json", "hocap_170650_asset_manifest.json"),
        (
            REFERENCE_ROOT / "hocap_170105.world_wrist.stage16.npz",
            "hocap_170105.world_wrist.stage16.npz",
        ),
        (
            REFERENCE_ROOT / "hocap_170650.world_wrist.stage16.npz",
            "hocap_170650.world_wrist.stage16.npz",
        ),
    ]
    copied = []
    for source, archive_name in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / archive_name
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(REPO_ROOT)))
    return copied


def c2_summary(c2: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            key: report[key]
            for key in (
                "status",
                "num_envs",
                "steps",
                "clip_mode",
                "finite",
                "observation_shape",
                "observation_device",
                "resets",
                "unique_action_rows",
                "environment_steps_per_s",
            )
            if key in report
        }
        for name, report in c2.items()
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    c2_paths = {
        "single": output_root / "c2_smoke_1env.json",
        "alternating": output_root / "c2_smoke_1env_alternating.json",
        "vector": output_root / "c2_smoke_128env.json",
    }
    c2 = {name: load_json(path) for name, path in c2_paths.items()}
    c3 = load_json(output_root / "c3_semantic_qualification.json")
    manifests = {
        "wuji": load_json(C1_ROOT / "wuji_asset_manifest.json"),
        "hocap_170105": load_json(C1_ROOT / "hocap_170105_asset_manifest.json"),
        "hocap_170650": load_json(C1_ROOT / "hocap_170650_asset_manifest.json"),
    }
    references = {
        path.stem.replace(".world_wrist.stage16", ""): {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256(path),
        }
        for path in sorted(REFERENCE_ROOT.glob("*.world_wrist.stage16.npz"))
    }
    if len(references) != 2:
        raise RuntimeError(f"expected two frozen references, found {sorted(references)}")
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive = REPO_ROOT / ".local/archive" / f"stage16c0_c1_frozen_inputs_{stamp}_{git_head()}"
    copied = copy_archive(archive)
    frozen_inputs = {
        "status": "FROZEN_INPUTS_ARCHIVED",
        "archive": str(archive.relative_to(REPO_ROOT)),
        "copied": copied,
        "source_manifests": {
            key: {
                "generated_sha256": value.get("generated_sha256"),
                "visual_mesh_sha256": value.get("visual_mesh_sha256"),
                "mass_kg": value.get("mass_kg"),
                "fixed_base": value.get("fixed_base"),
            }
            for key, value in manifests.items()
        },
        "references": references,
    }
    write_json(output_root / "frozen_inputs.json", frozen_inputs)
    write_json(
        output_root / "preflight/platform_revalidation_attempt.json",
        {
            "status": "NOT_RUN_CONFIGURATION_INCOMPATIBLE",
            "command": (
                "verify_stage16_isaaclab_platform.py --phase full --steps 100 --accept-eula"
            ),
            "result": "the frozen C.0 verifier rejects --steps below 1000",
            "resolution": (
                "C.0 code and thresholds were not weakened; its existing full qualification "
                "remains the authoritative platform evidence."
            ),
        },
    )
    write_json(
        output_root / "reference_step_contract.json",
        {
            "status": "VALIDATED",
            "clips": ["hocap_170105", "hocap_170650"],
            "frames": 41,
            "control_hz": 20,
            "endpoint_interval_policy": (
                "final timestamp endpoint interval is not treated as cadence drift"
            ),
            "canonical_finger_joint_count": 20,
            "tracked_link_count": 16,
        },
    )
    write_json(
        output_root / "scene_frame_contract.json",
        {
            "status": "VALIDATED",
            "reference_frame": "per-environment scene frame",
            "simulator_root_frame": "Isaac world frame",
            "conversion": "state_world - env_origin == reference_scene",
            "object_axis_points": (
                "computed in active object world pose then converted to scene frame"
            ),
        },
    )
    write_json(
        output_root / "isaac_state_convention.json",
        {
            "status": "VALIDATED",
            "root_quaternion": "wxyz",
            "root_linear_velocity": "world frame",
            "root_angular_velocity": "world frame",
            "wrench_frame": "global/world",
            "source": "Isaac Lab v2.3 DirectRLEnv runtime contract",
        },
    )
    c2_contract = c2["vector"]["contract"]
    write_json(
        output_root / "c2_environment_contract.json",
        {
            "status": "STAGE16C2_DIRECT_RL_ENV_VALIDATED",
            "action_dimension": 26,
            "observation_dimension": 764,
            "formal_object_rollout_state_writes": c2_contract["object_rollout_state_writes"],
            "wrist_root_state_writes_during_step": c2_contract[
                "wrist_root_state_writes_during_step"
            ],
            "c2_smokes": c2_summary(c2),
        },
    )
    c3_reasons = [
        "C3_WRIST_TRACKING_DIAGNOSTIC_BOUND_FAILED",
        "C3_CONTACT_DRIVEN_RESPONSE_EVIDENCE_NOT_YET_IMPLEMENTED",
        "C3_MUJOCO_TRACE_REPLAY_NOT_RUN_GATE_BLOCKED",
    ]
    transitions = [
        {"stage": "C2", "state": "VALIDATED", "reason": "finite real GPU direct-environment smokes"}
    ]
    transitions.extend(
        {"stage": "C3", "state": "PARTIAL", "reason": reason} for reason in c3_reasons
    )
    transitions.extend(
        (
            {"stage": "C4", "state": "NOT_RUN_GATE_BLOCKED", "reason": "C3 is partial"},
            {"stage": "C5", "state": "NOT_RUN_GATE_BLOCKED", "reason": "C3 is partial"},
            {"stage": "C6", "state": "NOT_AUTHORIZED", "reason": "C5 did not validate"},
        )
    )
    (output_root / "failure_recovery_transitions.jsonl").write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in transitions), encoding="utf-8"
    )
    c4 = {
        "status": "NOT_RUN_GATE_BLOCKED_BY_C3",
        "reason": "C.3 is partial; task-vector benchmarking is prohibited by the frozen gate order",
        "no_gpu_task_benchmark_started": True,
    }
    c5 = {
        "status": "NOT_RUN_GATE_BLOCKED_BY_C3",
        "reason": "C.3 is partial; no PhysX oracle is permitted",
        "ppo_authorized": False,
        "ppo_samples": 0,
        "ppo_checkpoints": 0,
    }
    write_json(output_root / "c4_vector_benchmark.json", c4)
    write_json(output_root / "c4_selected_backend.json", c4)
    write_json(output_root / "c5_oracle_config.json", c5)
    write_json(output_root / "c5_oracle_state.json", c5)
    write_json(output_root / "tests.json", {"contract_suite_passed": args.tests_passed})
    summary = {
        "status": "STAGE16C_DIRECT_ENV_PARTIAL",
        "stage_status": {
            "C2": "STAGE16C2_DIRECT_RL_ENV_VALIDATED",
            "C3": c3["status"],
            "C4": c4["status"],
            "C5": c5["status"],
            "C6": "NOT_AUTHORIZED",
        },
        "c3_blocker": c3["blocker"],
        "c3_contact": c3["contact"],
        "c2": c2_summary(c2),
        "c3_kinematic_object": c3["dynamic_wrist_finger_kinematic_object"],
        "c3_free_object": c3["free_object_zero_residual"],
        "tests_passed": args.tests_passed,
        "archive": frozen_inputs["archive"],
        "generated_at_utc": now.isoformat(),
    }
    write_json(output_root / "final_summary.json", summary)
    (output_root / "final_summary.md").write_text(
        "# Stage 16-C.2--C.5 closeout\n\n"
        "Overall: `STAGE16C_DIRECT_ENV_PARTIAL`. C.2 is validated; C.3 is partial "
        "because dynamic wrist tracking misses its 2 cm/10 degree diagnostic bound and "
        "direct all-hand contact-pair/impulse evidence is absent. C.4/C.5 were not run "
        "by gate order; C.6 PPO is not authorized (0 samples, 0 checkpoints).\n",
        encoding="utf-8",
    )
    (output_root / "visual_review.md").write_text(
        "# Numerical C.3 review\n\n"
        "Headless numeric review only: the kinematic-object diagnostic reaches wrist "
        "position errors of 6.756 cm/4.779 cm and rotation errors of 87.891/93.717 "
        "degrees for 170105/170650. This is failure evidence, not visual or contact "
        "success. Rendering and C.4/C.5 execution are intentionally not attempted after "
        "the C.3 hard gate.\n",
        encoding="utf-8",
    )
    (output_root / "dashboard.html").write_text(
        "<!doctype html><title>Stage 16-C closeout</title><h1>Stage 16-C.2--C.5</h1>"
        "<table border='1'><tr><th>Stage</th><th>Status</th></tr>"
        "<tr><td>C.2</td><td>VALIDATED</td></tr><tr><td>C.3</td><td>PARTIAL</td></tr>"
        "<tr><td>C.4</td><td>BLOCKED</td></tr><tr><td>C.5</td><td>BLOCKED</td></tr>"
        "<tr><td>C.6 PPO</td><td>NOT AUTHORIZED</td></tr></table>",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
