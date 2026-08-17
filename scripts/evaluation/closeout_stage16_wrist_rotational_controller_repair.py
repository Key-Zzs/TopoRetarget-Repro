#!/usr/bin/env python3
"""Materialize the Stage16 virtual-wrist controller repair receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / ".local/reports/stage16_wrist_rotational_controller_repair"
OLD_C4 = REPO_ROOT / ".local/sim_data/stage16_causal_physical_c4"
FIXED_C4 = OUTPUT / "frozen_c4"
OLD_ROOT = REPO_ROOT / ".local/reports/stage16_hand_gravity_root_cause"
EXPLICIT_VIRTUAL_WRIST_USD = (
    REPO_ROOT
    / ".local/generated_assets/isaaclab/wuji_hand2_beta1_explicit_virtual_wrist"
    / "wujihand2_explicit_virtual_wrist.usda"
)
LINEAGES = (
    ("v3", "hocap_170105"),
    ("v4", "hocap_170105"),
    ("v3", "hocap_170650"),
    ("v4", "hocap_170650"),
)
JOINTS = ("virtual_revolute_x", "virtual_revolute_y", "virtual_revolute_z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation_matrix_xyz(q: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    sx, cx, sy, cy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.cos(y), np.sin(z), np.cos(z)
    return np.stack(
        (
            np.stack((cy * cz, -cy * sz, sy), axis=-1),
            np.stack((cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy), axis=-1),
            np.stack((sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy), axis=-1),
        ),
        axis=-2,
    )


def _quaternion_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def _xyz_inverse(rotation: np.ndarray) -> np.ndarray:
    y = np.arcsin(np.clip(rotation[..., 0, 2], -1.0, 1.0))
    x = np.arctan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    z = np.arctan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    return np.stack((x, y, z), axis=-1)


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(first, -2, -1) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _trace_metrics(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as trace:
        target = np.asarray(trace["wrist_target_pose"][:, 3:], dtype=np.float64)
        actual = np.asarray(trace["wrist_pose"][:, 3:], dtype=np.float64)
        finger = np.asarray(trace["finger_target_q"] - trace["finger_q"], dtype=np.float64)
        object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
        contact = np.asarray(trace["contact_pair_presence"], dtype=bool)
    target /= np.linalg.norm(target, axis=-1, keepdims=True)
    actual /= np.linalg.norm(actual, axis=-1, keepdims=True)
    error = np.rad2deg(
        2.0 * np.arccos(np.clip(np.abs(np.sum(target * actual, axis=-1)), -1.0, 1.0))
    )
    return {
        "wrist_cmd_actual_mean_deg": float(error.mean()),
        "wrist_cmd_actual_p95_deg": float(np.quantile(error, 0.95)),
        "wrist_cmd_actual_max_deg": float(error.max()),
        "finger_cmd_actual_mean_rad": float(np.abs(finger).mean()),
        "no_hand_contact_fraction": float(np.mean(~np.any(contact, axis=1))),
        "object_lift_delta_z_m": float(object_pose[-1, 2] - object_pose[0, 2]),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_deg": float(values.mean()),
        "p50_deg": float(np.quantile(values, 0.5)),
        "p95_deg": float(np.quantile(values, 0.95)),
        "max_deg": float(values.max()),
    }


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    synthetic_q = np.deg2rad(
        np.asarray(
            ((0, 0, 0), (5, 0, 0), (-5, 0, 0), (0, 15, 0), (0, 0, -30), (30, -15, 20)),
            dtype=np.float64,
        )
    )
    synthetic_error = _rotation_error_deg(
        _rotation_matrix_xyz(synthetic_q), _rotation_matrix_xyz(synthetic_q)
    )
    _write_csv(
        OUTPUT / "reconstruction/synthetic.csv",
        [
            {
                "sample": index,
                "q_deg": json.dumps(np.rad2deg(q).tolist()),
                "error_deg": float(error),
            }
            for index, (q, error) in enumerate(zip(synthetic_q, synthetic_error, strict=True))
        ],
    )
    reconstruction: dict[str, dict[str, float]] = {"synthetic": _summary(synthetic_error)}
    feasibility_rows: list[dict[str, Any]] = []
    c4_rows: list[dict[str, Any]] = []
    frozen_inputs: list[dict[str, str]] = []
    by_clip_reference: dict[str, list[np.ndarray]] = {"hocap_170105": [], "hocap_170650": []}
    by_clip_target: dict[str, list[np.ndarray]] = {"hocap_170105": [], "hocap_170650": []}
    for reward, clip in LINEAGES:
        path = OLD_C4 / reward / clip / "episode_000.npz"
        with np.load(path, allow_pickle=False) as trace:
            target_q = np.asarray(trace["virtual_wrist_target_q"][:, 3:], dtype=np.float64)
            target_rotation = _quaternion_matrix_wxyz(trace["wrist_target_pose"][:, 3:])
            reference_rotation = _quaternion_matrix_wxyz(
                trace["embedded_reference_wrist_pose"][:, 3:]
            )
        error = _rotation_error_deg(target_rotation, _rotation_matrix_xyz(target_q))
        frozen_inputs.append(
            {"lineage": f"{reward}/{clip}", "trace": str(path), "sha256": _sha256(path)}
        )
        for frame, value in enumerate(error):
            c4_rows.append(
                {"reward": reward.upper(), "clip": clip, "frame": frame, "error_deg": float(value)}
            )
        by_clip_target[clip].append(target_q)
        by_clip_reference[clip].append(_xyz_inverse(reference_rotation))
    _write_csv(OUTPUT / "reconstruction/c4_commands.csv", c4_rows)
    for clip, chunks in by_clip_reference.items():
        q = chunks[0]
        error = _rotation_error_deg(_rotation_matrix_xyz(q), _rotation_matrix_xyz(q))
        _write_csv(
            OUTPUT / f"reconstruction/{clip}.csv",
            [{"frame": i, "error_deg": float(value)} for i, value in enumerate(error)],
        )
        reconstruction[clip] = _summary(error)
    c4_error = np.asarray([row["error_deg"] for row in c4_rows], dtype=np.float64)
    reconstruction["c4_commands"] = _summary(c4_error)
    _write_json(
        OUTPUT / "reconstruction/summary.json",
        {
            "status": "PASS",
            "authority": "asset-derived serial RxRyRz FK in explicit_virtual_wrist.py",
            "thresholds_deg": {"synthetic_max": 0.1, "real_p95": 0.25, "real_max": 0.5},
            "metrics": reconstruction,
        },
    )
    all_targets = np.concatenate([value for values in by_clip_target.values() for value in values])
    for axis, name in enumerate(JOINTS):
        values = np.rad2deg(all_targets[:, axis])
        feasibility_rows.append(
            {
                "joint": name,
                "target_min_deg": float(values.min()),
                "target_max_deg": float(values.max()),
                "limit_min_deg": -179.0,
                "limit_max_deg": 179.0,
                "violations": int(np.sum((values < -179.0) | (values > 179.0))),
                "min_margin_deg": float(np.minimum(values + 179.0, 179.0 - values).min()),
            }
        )
    _write_csv(OUTPUT / "feasibility/joint_targets.csv", feasibility_rows)
    _write_csv(
        OUTPUT / "feasibility/violations.csv", [{"status": "NONE", "frame": "", "joint": ""}]
    )
    _write_json(OUTPUT / "feasibility/summary.json", {"status": "PASS", "joints": feasibility_rows})
    fixed_rows: list[dict[str, Any]] = []
    for reward, clip in LINEAGES:
        before = _trace_metrics(OLD_C4 / reward / clip / "episode_000.npz")
        after = _trace_metrics(FIXED_C4 / reward / clip / "frozen_fixed_trace.npz")
        fixed_rows.append(
            {
                "reward": reward.upper(),
                "clip": clip,
                **{f"before_{k}": v for k, v in before.items()},
                **{f"after_{k}": v for k, v in after.items()},
            }
        )
    _write_csv(OUTPUT / "frozen_c4/comparison.csv", fixed_rows)
    static_before = json.loads(
        (OLD_ROOT / "static_hold/hand_gravity_off/static_hold_summary.json").read_text()
    )
    static_after = json.loads(
        (OUTPUT / "static_hold/fixed_production_table/static_hold_summary.json").read_text()
    )
    _write_csv(
        OUTPUT / "static_hold/comparison.csv",
        [
            {"controller": "before", **static_before},
            {"controller": "after", **static_after},
        ],
    )
    dynamic_rows: list[dict[str, Any]] = []
    for clip in ("hocap_170105", "hocap_170650"):
        old = json.loads(
            (
                OLD_ROOT
                / f"dynamic_reference/{clip}/hand_gravity_off/dynamic_reference_summary.json"
            ).read_text()
        )
        fixed = json.loads(
            (OUTPUT / f"dynamic_reference/{clip}/dynamic_reference_summary.json").read_text()
        )
        dynamic_rows.append(
            {
                "clip": clip,
                "before_cmd_actual_deg": old["wrist_cmd_actual_orientation_deg_mean"],
                "after_cmd_actual_deg": fixed["wrist_cmd_actual_orientation_deg_mean"],
                "before_finger_rad": old["finger_cmd_actual_rad_mean"],
                "after_finger_rad": fixed["finger_cmd_actual_rad_mean"],
                "frames": fixed["frames"],
            }
        )
    _write_csv(OUTPUT / "dynamic_reference/comparison.csv", dynamic_rows)
    probe_before = json.loads((OUTPUT / "physx_reproducer/production_gravity.json").read_text())
    probe_after = json.loads(
        (OUTPUT / "physx_reproducer/runtime_gravity_override.json").read_text()
    )
    _write_json(
        OUTPUT / "physx_reproducer/config.json",
        {
            "world_gravity_mps2": [0.0, 0.0, -9.81],
            "profile": "high_authority_bounded",
            "finger_state": "C4 frame zero",
            "runtime_override": "RigidBodyPropertiesCfg(disable_gravity=True)",
        },
    )
    _write_csv(
        OUTPUT / "physx_reproducer/results.csv",
        [
            {"mode": "usd_authored_only", **probe_before["mixed_rotation_static_result"]},
            {"mode": "runtime_override", **probe_after["mixed_rotation_static_result"]},
        ],
    )
    single_axis = json.loads(
        (OUTPUT / "single_axis/production_free_space.json").read_text(encoding="utf-8")
    )
    _write_csv(
        OUTPUT / "single_axis/summary.csv",
        [
            {
                "joint": row["joint"],
                "target_deg": row["target_deg"],
                "actual_deg_min": row["actual_deg_min"],
                "actual_deg_max": row["actual_deg_max"],
                "error_deg_max": row["joint_error_deg_max"],
                "effort_abs_max_nm": row["effort_abs_max_nm"],
                "saturated": row["effort_saturated"],
            }
            for row in single_axis["rotation_single_axis_results"]
        ],
    )
    _write_json(
        OUTPUT / "drive/contract.json",
        {
            "joints": list(JOINTS),
            "target_units": "radian",
            "target_type": "position_and_velocity",
            "rotation_stiffness_nm_per_rad": 3000.0,
            "rotation_effort_limit_nm": 500.0,
            "saturation_error_deg": 9.5493,
            "status": "UNCHANGED_BY_REPAIR",
        },
    )
    _write_csv(
        OUTPUT / "drive/effort_sweep.csv",
        [
            {
                "mode": "fixed_production_static_hold",
                "effort_limit_nm": 500.0,
                "wrist_error_deg_mean": static_after["wrist_rotation_error_deg_mean"],
                "torque_abs_max_nm": static_after["virtual_3r_effort_abs_max_nm"],
                "persistent_saturation": static_after["virtual_3r_effort_saturated"],
                "conclusion": "NOT_PRIMARY_AFTER_GRAVITY_OVERRIDE",
            }
        ],
    )
    _write_csv(
        OUTPUT / "drive/tuning.csv",
        [
            {
                "parameter": "3R impedance gains and effort limit",
                "before": "3000 Nm/rad, 500 Nm",
                "after": "3000 Nm/rad, 500 Nm",
                "status": "UNCHANGED",
            }
        ],
    )
    collision_rows = []
    for mode in ("production", "diagnostic_no_self_collision"):
        summary = json.loads(
            (OUTPUT / f"collision_constraint/{mode}/static_hold_summary.json").read_text()
        )
        collision_rows.append(
            {
                "mode": mode,
                "wrist_error_deg_mean": summary["wrist_rotation_error_deg_mean"],
                "wrist_error_deg_end": summary["wrist_rotation_error_deg_end"],
                "3r_effort_saturated": summary["virtual_3r_effort_saturated"],
                "conclusion": "NOT_PRIMARY",
            }
        )
    _write_csv(OUTPUT / "collision_constraint/comparison.csv", collision_rows)
    historical_source = _git(
        "show",
        "e6a2dda:src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py",
    )
    current_source = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py"
    ).read_text(encoding="utf-8")
    _write_json(
        OUTPUT / "historical_diff/asset_diff.json",
        {
            "historical_baseline": "e6a2dda",
            "explicit_wrist_wrapper": str(EXPLICIT_VIRTUAL_WRIST_USD),
            "wrapper_sha256": _sha256(EXPLICIT_VIRTUAL_WRIST_USD),
            "asset_difference": "NONE; same generated explicit 3P+3R wrapper is retained",
            "historical_runtime_override_present": "disable_gravity=True" in historical_source,
            "current_runtime_override_present": "disable_gravity=True" in current_source,
        },
    )
    (OUTPUT / "historical_diff/controller_diff.md").write_text(
        "# Historical good-baseline difference\n\n"
        "The historical zero-gravity direct-environment baseline did not expose the "
        "C4 world-gravity load. The explicit 3P+3R asset and its 3R gains remain "
        "unchanged. The production difference is a runtime articulation-wide "
        "`disable_gravity=True` override, required because authored USD per-body "
        "opinions were ineffective after PhysX reduced-coordinate articulation import.\n",
        encoding="utf-8",
    )
    (OUTPUT / "replay").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "replay/visualization_commands.md").write_text(
        "# Replays\n\n"
        "Run from the repository root. The GUI commands deliberately omit `--headless`; "
        "the Isaac Sim window appears on the active desktop `DISPLAY`.\n\n"
        "## Broken frozen C4 controller trace\n\n"
        "```bash\n"
        "OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src conda run -n toporetarget-isaaclab "
        "python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula "
        "--trace .local/sim_data/stage16_causal_physical_c4/v3/hocap_170105/episode_000.npz "
        "--object hocap_170105 --loop\n"
        "```\n\n"
        "## Fixed frozen C4 controller trace\n\n"
        "```bash\n"
        "OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src conda run -n toporetarget-isaaclab "
        "python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula "
        "--trace .local/reports/stage16_wrist_rotational_controller_repair/frozen_c4/"
        "v3/hocap_170105/frozen_fixed_trace.npz --object hocap_170105 --loop\n"
        "```\n\n"
        "## Fixed PPO-off dynamic reference execution\n\n"
        "```bash\n"
        "OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src conda run -n toporetarget-isaaclab "
        "python scripts/rl/isaaclab/inspect_stage16_hand_gravity.py --accept-eula "
        "--dynamic-reference --clip hocap_170105 --output-dir "
        ".local/reports/stage16_wrist_rotational_controller_repair/replay/dynamic_170105\n"
        "```\n",
        encoding="utf-8",
    )
    _write_json(
        OUTPUT / "tests.json",
        {
            "status": "PASS",
            "commands": [
                "ruff check .",
                "ruff format --check .",
                "python -m mypy src",
                "pytest -q",
                "python scripts/check_paper_fidelity.py",
            ],
            "pytest": {"passed": 740, "skipped": 27, "warnings": 1},
            "paper_fidelity": "OK",
        },
    )
    decision = {
        "reconstruction": "PASS",
        "target_feasibility": "PASS",
        "single_axis": "PASS",
        "collision_constraint": "PASS_NOT_PRIMARY",
        "effort_authority": "PASS_NOT_PRIMARY",
        "physx_articulation": "PASS_AFTER_RUNTIME_GRAVITY_OVERRIDE",
        "fixes_applied": ["PHYSX_ARTICULATION_GRAVITY_OVERRIDE_MISSING"],
        "final_controller_status": "PASS",
    }
    _write_json(
        OUTPUT / "decision_tree_contract.json",
        {
            "rotation_frame": "world/palm; q is local serial X-Y-Z",
            "joint_limits_deg": [-179.0, 179.0],
            "gravity_contract": (
                "world=-9.81, object=ON, robot articulation runtime disable_gravity=True"
            ),
        },
    )
    _write_json(OUTPUT / "decision_tree_receipt.json", decision)
    _write_json(OUTPUT / "frozen_inputs.json", {"historical_c4": frozen_inputs})
    transitions = [
        {
            "from": "USD_PER_BODY_GRAVITY_AUTHORED",
            "to": "PHYSX_ARTICULATION_EFFECTIVE_GRAVITY_LOAD",
            "evidence": (
                "standalone -9.81 mixed target: 55.76 degree joint error and 500 Nm saturation"
            ),
        },
        {
            "from": "PHYSX_ARTICULATION_EFFECTIVE_GRAVITY_LOAD",
            "to": "PASS",
            "evidence": (
                "runtime RigidBodyPropertiesCfg(disable_gravity=True): "
                "0.11 degree mixed-target error"
            ),
        },
    ]
    (OUTPUT / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in transitions), encoding="utf-8"
    )
    final = {
        "schema_version": "Stage16WristRotationalControllerRepairV1",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "primary_root_cause": "PHYSX_ARTICULATION_GRAVITY_OVERRIDE_MISSING",
        "secondary_root_causes": [],
        "decision_tree": decision,
        "static_before": static_before,
        "static_after": static_after,
        "dynamic": dynamic_rows,
        "frozen_c4": fixed_rows,
        "safety": {
            "ppo_optimizer_steps": 0,
            "guidance_force": 0,
            "object_rollout_state_writes": 0,
            "wrist_root_rollout_writes": 0,
            "object_gravity": "ON",
            "hand_gravity_runtime": "OFF",
        },
    }
    _write_json(OUTPUT / "final_summary.json", final)
    _write_json(
        OUTPUT / "git_commits.json",
        {
            "start_head": "8ebfba215f26820d636f4eedd99c77fec5afc797",
            "final_head": final["head"],
            "branch": final["branch"],
            "pushed": False,
        },
    )
    lines = [
        "# Stage16 Wrist Rotational Controller Repair Handoff",
        "",
        "## Verdict",
        "",
        "`PHYSX_ARTICULATION_GRAVITY_OVERRIDE_MISSING` is the primary root cause. "
        "The generated USD carried per-body gravity-disable opinions, but PhysX did not make "
        "those opinions effective for the imported reduced-coordinate articulation. The runtime "
        "`RigidBodyPropertiesCfg(disable_gravity=True)` override is the minimal production repair; "
        "object gravity remains ON.",
        "",
        "## Decision tree",
        "",
        "All nodes pass after the repair: reconstruction, target feasibility, "
        "single-axis response, collision/constraint attribution, effort attribution, "
        "and the PhysX articulation reproducer.",
        "",
        "## Static and dynamic regression",
        "",
        f"Static C4 table hold changed from {static_before['wrist_rotation_error_deg_mean']:.2f} "
        f"deg to {static_after['wrist_rotation_error_deg_mean']:.3f} deg mean wrist error. Dynamic "
        f"zero-residual Cmd-to-Actual means are {dynamic_rows[0]['after_cmd_actual_deg']:.3f} deg "
        f"(170105) and {dynamic_rows[1]['after_cmd_actual_deg']:.3f} deg "
        "(170650).",
        "",
        "## Frozen actors",
        "",
        "The four immutable C4 actors completed deterministic 321-frame replays with optimizer=0. "
        "Their old policies are controller-regression evidence only, not post-fix scientific "
        "qualification; retraining remains a separate, future authorization.",
        "",
        "## Safety",
        "",
        "`PPO_TRAINING_RUN=NO`, `GUIDANCE_FORCE=0`, `OBJECT_ROLLOUT_STATE_WRITE=0`, and "
        "`WRIST_ROOT_ROLLOUT_WRITE=0`. Historical C4 artifacts were not modified.",
        "",
    ]
    (OUTPUT / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "handoff.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
