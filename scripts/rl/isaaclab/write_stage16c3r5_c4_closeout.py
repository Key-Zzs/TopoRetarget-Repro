#!/usr/bin/env python3
"""Write the fail-closed retimed C3 and GPU-vector C4 closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16c3r5_reference_retiming_c4"
EXPECTED_COUNTS = [128, 512, 1024, 2048, 4096]
EXPECTED_REFERENCE_HASHES = {
    "hocap_170105": "63bb0b630d54a92c45bf9d306bf2111ff1d4c070d112706d21970aef452db162",
    "hocap_170650": "c153f5071ffd81414ef7837cde625597e46c753afad8966598649cae9b3fac5c",
}
STATUS_DOCUMENTS = (
    "README.md",
    "README.zh-CN.md",
    "docs/ROADMAP.md",
    "docs/ROADMAP.zh-CN.md",
    "docs/PAPER_FIDELITY.md",
    "docs/PAPER_FIDELITY.yaml",
    "docs/rl/PAPER_FIDELITY_LEDGER.md",
    "docs/rl/PAPER_FIDELITY_LEDGER.yaml",
    "docs/rl/ISAACLAB_DIRECT_RL_ENV.md",
)
SUPPORTING_DOCUMENTS = (
    "docs/ASSUMPTIONS.md",
    "docs/rl/FAILURE_RECOVERY_STATE_MACHINE.md",
    "docs/rl/ISAACLAB_CONTACT_CAUSALITY.md",
    "docs/rl/ISAACLAB_CONTACT_CAUSALITY.zh-CN.md",
    "docs/rl/ISAACLAB_WRIST_DYNAMICS.md",
    "docs/rl/ISAACLAB_WRIST_DYNAMICS.zh-CN.md",
    "docs/stages/STAGE16_REFERENCE_TRACKING_PPO.md",
)
IMPLEMENTATION_PATHS = (
    "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
    "scripts/rl/isaaclab/benchmark_stage16c4_vector_env.py",
    "scripts/rl/isaaclab/qualify_stage16c3_finite_virtual_wrist.py",
    "scripts/rl/isaaclab/qualify_stage16c3_retimed_contact_causality.py",
    "scripts/rl/isaaclab/qualify_stage16c3_retimed_semantics.py",
    "scripts/rl/isaaclab/write_stage16c3r5_c4_closeout.py",
    "scripts/rl/materialize_stage16b_selected_action_traces.py",
    "src/toporetarget/rl/environments/isaaclab_backend/explicit_wrist_reference.py",
    "src/toporetarget/rl/environments/isaaclab_backend/reference_bank.py",
    "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py",
    "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-report",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r4_mpc_holdout_c4/final_summary.json",
    )
    parser.add_argument(
        "--c3-report",
        type=Path,
        default=DEFAULT_ROOT / "c3_full_qualification_scale8_final.json",
    )
    parser.add_argument(
        "--contact-report",
        type=Path,
        default=DEFAULT_ROOT / "contact_causality_scale8.json",
    )
    parser.add_argument(
        "--c4-report",
        type=Path,
        default=DEFAULT_ROOT / "c4_gpu_vector_benchmark_scale8.json",
    )
    parser.add_argument("--test-report", type=Path, default=DEFAULT_ROOT / "test_summary.json")
    parser.add_argument(
        "--active-config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_world_wrist_env.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"C3R5_CLOSEOUT_INPUT_MISSING: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"C3R5_CLOSEOUT_INPUT_NOT_OBJECT: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"C3R5_CLOSEOUT_INPUT_MISSING: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"C3R5_CLOSEOUT_INPUT_NOT_OBJECT: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _validate_documentation() -> dict[str, dict[str, str]]:
    artifacts = {}
    for relative in (*STATUS_DOCUMENTS, *SUPPORTING_DOCUMENTS):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"C3R5_CLOSEOUT_DOCUMENT_MISSING: {path}")
        text = path.read_text(encoding="utf-8")
        if "STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED" not in text:
            raise RuntimeError(f"C3R5_CLOSEOUT_DOCUMENT_C3_STATUS_MISSING: {relative}")
        if relative in STATUS_DOCUMENTS and "STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED" not in text:
            raise RuntimeError(f"C3R5_CLOSEOUT_DOCUMENT_C4_STATUS_MISSING: {relative}")
        artifacts[relative] = _artifact(path)
    return artifacts


def _implementation_artifacts() -> dict[str, dict[str, str]]:
    artifacts = {}
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"C3R5_CLOSEOUT_IMPLEMENTATION_MISSING: {path}")
        artifacts[relative] = _artifact(path)
    return artifacts


def _validate(
    historical: dict[str, Any],
    c3: dict[str, Any],
    contact: dict[str, Any],
    c4: dict[str, Any],
    tests: dict[str, Any],
    active_config: dict[str, Any],
) -> None:
    if historical.get("overall") != "STAGE16C_BLOCKED_WITH_BOUNDED_EVIDENCE":
        raise RuntimeError("C3R5_CLOSEOUT_HISTORICAL_BLOCKER_UNEXPECTED")
    if c3.get("status") != "STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED":
        raise RuntimeError("C3R5_CLOSEOUT_C3_NOT_VALIDATED")
    if set(c3.get("passes", {})) != {f"C3-{index}" for index in range(6)} or not all(
        c3["passes"].values()
    ):
        raise RuntimeError("C3R5_CLOSEOUT_C3_MODE_GATE_FAILURE")
    scale = c3.get("reference_time_scale")
    if scale != 8 or c3.get("source_keyframes") != 41 or c3.get("retimed_control_steps") != 321:
        raise RuntimeError("C3R5_CLOSEOUT_RETIMING_CONTRACT_FAILURE")
    reference = c3.get("contract", {}).get("reference_bank", {})
    if (
        reference.get("identifier") != "world_wrist_reference_bank_uniform_retimed_v1"
        or reference.get("source_frame_count") != 41
        or reference.get("frame_count") != 321
        or reference.get("source_control_hz") != 20.0
        or reference.get("control_hz") != 20.0
        or reference.get("reference_time_scale") != scale
        or reference.get("hashes") != EXPECTED_REFERENCE_HASHES
    ):
        raise RuntimeError("C3R5_CLOSEOUT_REFERENCE_MANIFEST_FAILURE")
    if c3.get("process_isolation", {}).get("worker_exits") != {
        "dynamic": 0,
        "kinematic": 0,
    }:
        raise RuntimeError("C3R5_CLOSEOUT_C3_WORKER_EXIT_FAILURE")
    for mode in ("C3-1", "C3-2"):
        for row in c3.get(mode, []):
            if (
                not row.get("pass")
                or row.get("formal_object_rollout_state_writes") != 0
                or row.get("wrist_root_state_writes_during_step") != 0
            ):
                raise RuntimeError(f"C3R5_CLOSEOUT_{mode}_ROLLOUT_CONTRACT_FAILURE")
    basis = c3.get("C3-3", {})
    if basis.get("basis_count") != 26 or not basis.get("all_pass"):
        raise RuntimeError("C3R5_CLOSEOUT_C3_ACTION_BASIS_FAILURE")
    traces = c3.get("C3-4", [])
    if [(row.get("clip"), row.get("source_horizon")) for row in traces] != [
        ("hocap_170105", 5),
        ("hocap_170650", 10),
    ]:
        raise RuntimeError("C3R5_CLOSEOUT_C3_TRACE_SET_FAILURE")
    for trace in traces:
        if (
            trace.get("source_action_shape") != [40, 26]
            or trace.get("retimed_action_shape") != [320, 26]
            or not trace.get("action_bounds_pass")
            or not trace.get("classified")
            or trace.get("formal_object_rollout_state_writes") != 0
        ):
            raise RuntimeError("C3R5_CLOSEOUT_C3_TRACE_CONTRACT_FAILURE")
    if contact.get("status") != "C3_CONTACT_CAUSALITY_VALIDATED":
        raise RuntimeError("C3R5_CLOSEOUT_CONTACT_NOT_VALIDATED")
    if contact.get("reference_time_scale") != scale:
        raise RuntimeError("C3R5_CLOSEOUT_CONTACT_TIMING_MISMATCH")
    baseline = contact.get("no_contact_baseline", {})
    if (
        not baseline.get("pass")
        or baseline.get("max_force_n") != 0.0
        or baseline.get("max_delta_v_mps") != 0.0
    ):
        raise RuntimeError("C3R5_CLOSEOUT_NO_CONTACT_BASELINE_FAILURE")
    if [clip.get("clip") for clip in contact.get("clips", [])] != [
        "hocap_170105",
        "hocap_170650",
    ]:
        raise RuntimeError("C3R5_CLOSEOUT_CONTACT_CLIP_SET_FAILURE")
    for clip in contact["clips"]:
        contract = clip.get("contract", {})
        if (
            clip.get("status") != "C3_CONTACT_CAUSALITY_VALIDATED"
            or not all(clip.get("passes", {}).values())
            or clip.get("contact_record_count", 0) < 1
            or clip.get("causal_record_count", 0) < 1
            or clip.get("peak_contact_force_n", 0.0) <= 0.0
            or clip.get("peak_impulse_ns", 0.0) <= 0.0
            or contract.get("object_rollout_state_writes") != 0
            or contract.get("wrist_root_state_writes_during_step") != 0
        ):
            raise RuntimeError("C3R5_CLOSEOUT_CONTACT_CLIP_CONTRACT_FAILURE")
    if c4.get("status") != "STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED":
        raise RuntimeError("C3R5_CLOSEOUT_C4_NOT_VALIDATED")
    if (
        c4.get("controller") != "finite_virtual_6d_wrist_actuator_v1"
        or c4.get("profile") != "high_authority_bounded"
        or c4.get("reference_time_scale") != scale
    ):
        raise RuntimeError("C3R5_CLOSEOUT_C4_TIMING_MISMATCH")
    if c4.get("environment_counts") != EXPECTED_COUNTS:
        raise RuntimeError("C3R5_CLOSEOUT_C4_ENV_COUNTS_MISMATCH")
    if c4.get("warmup_control_steps") != 100 or c4.get("measurement_control_steps") != 500:
        raise RuntimeError("C3R5_CLOSEOUT_C4_STEP_BUDGET_MISMATCH")
    rows = c4.get("rows", [])
    if [row.get("num_envs") for row in rows] != EXPECTED_COUNTS:
        raise RuntimeError("C3R5_CLOSEOUT_C4_ROWS_INCOMPLETE")
    if any(
        row.get("outcome") not in {"validated", "oom", "skipped_after_two_ooms"} for row in rows
    ):
        raise RuntimeError("C3R5_CLOSEOUT_C4_UNBOUNDED_FAILURE")
    for row in rows:
        if row.get("outcome") == "oom" and (row.get("clean_exit") or not row.get("oom_detected")):
            raise RuntimeError("C3R5_CLOSEOUT_C4_OOM_CLASSIFICATION_FAILURE")
    validated = [row for row in rows if row.get("outcome") == "validated"]
    if not validated:
        raise RuntimeError("C3R5_CLOSEOUT_C4_NO_VALIDATED_COUNT")
    for row in validated:
        benchmark = row.get("benchmark", {})
        contract = benchmark.get("contract", {})
        num_envs = row["num_envs"]
        force_shapes = benchmark.get("contact_sensor_contract", {}).get("force_matrix_shapes", {})
        resource = row.get("resource", {})
        if (
            not row.get("clean_exit")
            or benchmark.get("status") != "STAGE16C4_VECTOR_COUNT_VALIDATED"
            or not benchmark.get("finite")
            or benchmark.get("nan_or_inf")
            or benchmark.get("action_shape") != [num_envs, 26]
            or benchmark.get("observation_shape") != [num_envs, 764]
            or benchmark.get("warmup_control_steps") != 100
            or benchmark.get("measurement_control_steps") != 500
            or benchmark.get("reference_time_scale") != scale
            or benchmark.get("retimed_control_steps") != 321
            or benchmark.get("profile") != "high_authority_bounded"
            or not str(benchmark.get("device", "")).startswith("cuda")
            or benchmark.get("decimation") != 6
            or not math.isclose(benchmark.get("physics_dt_s", 0.0), 1.0 / 120.0)
            or not math.isclose(benchmark.get("control_dt_s", 0.0), 1.0 / 20.0)
            or benchmark.get("contact_mode") != "aggregate"
            or force_shapes.get("Object170105") != [num_envs, 1, 21, 3]
            or force_shapes.get("Object170650") != [num_envs, 1, 21, 3]
            or benchmark.get("environment_steps_per_s", 0.0) <= 0.0
            or benchmark.get("physics_steps_per_s", 0.0) <= 0.0
            or benchmark.get("samples_per_s", 0.0) <= 0.0
            or resource.get("sample_count", 0) < 1
            or resource.get("process_vram_peak_mib", 0) <= 0
            or resource.get("process_rss_peak_mib", 0.0) <= 0.0
            or row.get("contact_warning_count", -1) < 0
            or contract.get("object_rollout_state_writes") != 0
            or contract.get("wrist_root_state_writes_during_step") != 0
        ):
            raise RuntimeError("C3R5_CLOSEOUT_C4_VALIDATED_ROW_CONTRACT_FAILURE")
    selected = c4.get("selection", {})
    highest = max(int(row["num_envs"]) for row in validated)
    rollout_length = max(8, min(128, 65536 // highest))
    expected_selection = {
        "selected_num_envs": highest,
        "rollout_length": rollout_length,
        "shards": max(1, math.ceil(highest / 1024)),
        "samples_per_update": highest * rollout_length,
    }
    if selected != expected_selection:
        raise RuntimeError("C3R5_CLOSEOUT_C4_SELECTION_FAILURE")
    if c4.get("oom_attempts", 0) > c4.get("oom_attempt_limit", 2):
        raise RuntimeError("C3R5_CLOSEOUT_C4_OOM_BUDGET_EXCEEDED")
    if c4.get("contact_buffer_failures", 0) > c4.get("contact_buffer_fix_limit", 2):
        raise RuntimeError("C3R5_CLOSEOUT_C4_CONTACT_BUDGET_EXCEEDED")
    if (
        tests.get("status") != "PASS"
        or not tests.get("checks")
        or not all(check.get("pass") for check in tests["checks"])
    ):
        raise RuntimeError("C3R5_CLOSEOUT_TEST_FAILURE")
    retiming = active_config.get("reference_bank", {}).get("active_retiming", {})
    configured_reference = active_config.get("reference_bank", {})
    active_wrist = active_config.get("active_wrist", {})
    if (
        active_config.get("schema_version") != "toporetarget.stage16c3r5.direct_env.v1"
        or retiming.get("time_scale") != scale
        or retiming.get("runtime_frames") != 321
        or retiming.get("runtime_control_hz") != 20
        or retiming.get("runtime_key_span_s") != 16.0
        or retiming.get("episode_length_s") != 16.05
        or not retiming.get("shared_across_both_clips")
        or configured_reference.get("source_hashes") != EXPECTED_REFERENCE_HASHES
        or active_wrist.get("controller") != "finite_virtual_6d_wrist_actuator_v1"
        or active_wrist.get("profile") != "high_authority_bounded"
    ):
        raise RuntimeError("C3R5_CLOSEOUT_ACTIVE_CONFIG_FAILURE")


def _c3_rows(c3: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "clip": row["clip"],
            "finger_rmse_final_rad": row["finger_rmse_rad"]["final"],
            "tracked_link_rmse_final_m": row["tracked_link_rmse_m"]["final"],
            "wrist": row["wrist"],
            "formal_object_rollout_state_writes": row["formal_object_rollout_state_writes"],
            "wrist_root_state_writes_during_step": row["wrist_root_state_writes_during_step"],
            "pass": row["pass"],
        }
        for row in c3["C3-1"]
    ]


def _c4_rows(c4: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in c4["rows"]:
        benchmark = row.get("benchmark") or {}
        resource = row.get("resource") or {}
        rows.append(
            {
                "num_envs": row["num_envs"],
                "outcome": row["outcome"],
                "clean_exit": row.get("clean_exit", False),
                "environment_steps_per_s": benchmark.get("environment_steps_per_s"),
                "physics_steps_per_s": benchmark.get("physics_steps_per_s"),
                "samples_per_s": benchmark.get("samples_per_s"),
                "gpu_utilization_mean_percent": resource.get("gpu_utilization_mean_percent"),
                "gpu_utilization_peak_percent": resource.get("gpu_utilization_peak_percent"),
                "process_vram_peak_mib": resource.get("process_vram_peak_mib"),
                "process_rss_peak_mib": resource.get("process_rss_peak_mib"),
                "reset_rate_per_environment_step": benchmark.get("reset_rate_per_environment_step"),
                "contact_warning_count": row.get("contact_warning_count"),
                "nan_or_inf": benchmark.get("nan_or_inf"),
            }
        )
    return rows


def _c4_throughput_peak(c4: dict[str, Any]) -> dict[str, Any]:
    validated = [row for row in c4["rows"] if row["outcome"] == "validated"]
    row = max(validated, key=lambda item: item["benchmark"]["samples_per_s"])
    return {
        "num_envs": row["num_envs"],
        "samples_per_s": row["benchmark"]["samples_per_s"],
        "selection_policy": (
            "diagnostic only; formal selection uses highest validated count for rollout geometry"
        ),
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _handoff(summary: dict[str, Any]) -> str:
    historical = summary["historical_blocker"]
    contact_lines = []
    for row in summary["contact"]["clips"]:
        contact_lines.append(
            (
                "| {clip} | {contacts} | {causal} | {force} | {impulse} | {dv} | {domega} | PASS |"
            ).format(
                clip=row["clip"],
                contacts=row["contact_record_count"],
                causal=row["causal_record_count"],
                force=_format_number(row["peak_contact_force_n"], 6),
                impulse=_format_number(row["peak_impulse_ns"], 8),
                dv=_format_number(row["peak_delta_v_mps"], 8),
                domega=_format_number(row["peak_delta_omega_radps"], 8),
            )
        )
    c3_lines = []
    for row in summary["c3"]["clips"]:
        wrist = row["wrist"]
        c3_lines.append(
            "| {clip} | {pos} | {rot} | {finger} | {link} | PASS |".format(
                clip=row["clip"],
                pos=_format_number(wrist["max_position_m"], 6),
                rot=_format_number(wrist["max_rotation_deg"], 3),
                finger=_format_number(row["finger_rmse_final_rad"], 6),
                link=_format_number(row["tracked_link_rmse_final_m"], 6),
            )
        )
    c4_lines = []
    for row in summary["c4"]["rows"]:
        c4_lines.append(
            (
                "| {num} | {outcome} | {envps} | {physps} | {samples} | {gpu} | "
                "{gpu_peak} | {vram} | {rss} | {reset} | {warn} | {clean} |"
            ).format(
                num=row["num_envs"],
                outcome=row["outcome"],
                envps=_format_number(row["environment_steps_per_s"], 2),
                physps=_format_number(row["physics_steps_per_s"], 2),
                samples=_format_number(row["samples_per_s"], 2),
                gpu=_format_number(row["gpu_utilization_mean_percent"], 1),
                gpu_peak=_format_number(row["gpu_utilization_peak_percent"], 1),
                vram=_format_number(row["process_vram_peak_mib"], 0),
                rss=_format_number(row["process_rss_peak_mib"], 0),
                reset=_format_number(row["reset_rate_per_environment_step"], 6),
                warn=_format_number(row["contact_warning_count"], 0),
                clean=str(row["clean_exit"]),
            )
        )
    tests = "\n".join(
        f"- {check['name']}: `{check['result']}`" for check in summary["tests"]["checks"]
    )
    return "\n".join(
        [
            "# Stage 16-C.3 Reference Retiming through C.4 GPU Handoff",
            "",
            "## 1. Final Status",
            "",
            f"`{summary['status']}` on `{summary['git']['branch']}` at evidence HEAD "
            f"`{summary['git']['head']}`. C.3 and C.4 validate; C.5 is outside this "
            "closeout and C.6 PPO remains unauthorized.",
            "",
            "## 2. Frozen PD Baseline",
            "",
            "The unretimed bounded PD baseline retained 1.128/1.089 cm position maximum, "
            "17.587/19.570 degree rotation maximum, and 21.25%/18.75% torque saturation.",
            "",
            "## 3. Previous Blocker",
            "",
            f"The repaired reporter false exit is `{historical['false_blocker_status']}`. The "
            f"actual prior result was `{historical['status']}`: independent multi-step holdout "
            "and both fixed-timeline 41-frame gates failed after computed-torque and MPC.",
            "",
            "## 4. Authorized Reference Retiming",
            "",
            "One global factor 8 is shared by both clips. The immutable 41 source keys and NPZ "
            "hashes are preserved at stride 8 in a derived 321-sample, 20 Hz runtime view. "
            "Position/finger values use Hermite interpolation, quaternions use normalized "
            "shortest-arc interpolation, and reference rates are scaled by 1/8.",
            "",
            "## 5. Explicit 3P+3R Joint Reference",
            "",
            "Joint order remains Px/Py/Pz/Rx/Ry/Rz followed by the frozen 20-finger order. "
            "The SE(3)-to-serial-XYZ mapping, limits, FK and singularity checks are unchanged.",
            "",
            "## 6. Articulation Dynamics and Coupling History",
            "",
            "C3R4 validated the GPU PhysX 26x26 generalized mass matrix, M_ww/M_wf coupling, "
            "and live Coriolis/centrifugal plus gravity tensors. Its fixed-timeline Path A/B "
            "failures remain immutable historical evidence; retiming does not relabel them.",
            "",
            "## 7. Active Controller Decision",
            "",
            f"`{summary['controller']}` with shared `{summary['profile']}` is active for the "
            "authorized retimed task. Gains, effort bounds, action/observation contracts and "
            "qualification gates were not relaxed.",
            "",
            "## 8. Contact and Causality",
            "",
            f"`{summary['contact']['status']}`. The collision-disabled baseline has "
            f"max force {summary['contact']['no_contact_max_force_n']} N and max delta-v "
            f"{summary['contact']['no_contact_max_delta_v_mps']} m/s; both clips prove ordered "
            "finite contact to subsequent object momentum response.",
            "",
            "| Clip | Contact records | Causal records | Peak force (N) | Peak impulse (Ns) | "
            "Peak delta-v (m/s) | Peak delta-omega (rad/s) | Result |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            *contact_lines,
            "",
            "## 9. Stage 16-C.3 Qualification",
            "",
            "| Clip | Wrist pos max (m) | Wrist rot max (deg) | "
            "Finger final RMSE (rad) | Link final RMSE (m) | Result |",
            "|---|---:|---:|---:|---:|---|",
            *c3_lines,
            "",
            f"All C3-0 through C3-5 gates pass: `{summary['c3']['status']}`.",
            "",
            "## 10. Stage 16-C.4 Benchmark",
            "",
            "| Envs | Outcome | Env-steps/s | Physics-steps/s | Samples/s | GPU mean % | "
            "GPU peak % | VRAM MiB | RSS MiB | Reset rate | Contact warnings | Clean exit |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *c4_lines,
            "",
            f"Status: `{summary['c4']['status']}`. Selection: "
            f"`{json.dumps(summary['c4']['selection'], sort_keys=True)}`. Peak measured "
            f"throughput: `{json.dumps(summary['c4']['throughput_peak'], sort_keys=True)}`.",
            "",
            "## 11. Stage 16-C.5 Oracle",
            "",
            "`NOT_RUN_OUTSIDE_ACTIVE_C3_C4_GOAL`; no PhysX-oracle or PPO authorization claim.",
            "",
            "## 12. Commands",
            "",
            *[f"- `{command}`" for command in summary["commands"]],
            "",
            "## 13. Tests",
            "",
            tests,
            "",
            "## 14. README and Roadmap",
            "",
            "English and Chinese status surfaces record the factor-8 C3 validation and actual "
            "C4 benchmark rather than the superseded gate-blocked state.",
            "",
            "## 15. Local Commit Boundary",
            "",
            "Local commit permitted. PUSHED=NO; PR_CREATED=NO; MAIN_MERGED=NO; "
            "TAG_CREATED=NO; RELEASE_CREATED=NO.",
            "",
            "## 16. Stage 16-C.6 Entry Decision",
            "",
            "`STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED`; samples=0; checkpoints=0.",
            "",
            "## 17. Remaining Limitations",
            "",
            "No PPO/checkpoint; the explicit wrist is an abstract engineering actuator, not a "
            "real arm; physical provenance remains unresolved; Isaac Sim 5.1 is legacy; no "
            "linear-scaling or training-optimal-throughput claim. Unrelated CUDA workloads "
            "shared the GPU during C4, so global utilization and throughput are conservative "
            "shared-load observations; process VRAM was sampled separately. No real-world-"
            "dynamics or sim-to-real claim is made.",
            "",
            "## 18. Recommended Next Action",
            "",
            "Treat the selected C4 geometry as infrastructure evidence only. C.5 requires a "
            "separate explicit goal and must pass before any C.6/PPO authorization.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    inputs = {
        "historical": args.historical_report,
        "c3": args.c3_report,
        "contact": args.contact_report,
        "c4": args.c4_report,
        "tests": args.test_report,
        "active_config": args.active_config,
    }
    loaded = {
        name: (_load_yaml(path) if name == "active_config" else _load(path))
        for name, path in inputs.items()
    }
    documentation = _validate_documentation()
    implementation = _implementation_artifacts()
    _validate(
        loaded["historical"],
        loaded["c3"],
        loaded["contact"],
        loaded["c4"],
        loaded["tests"],
        loaded["active_config"],
    )
    historical = loaded["historical"]
    c3 = loaded["c3"]
    contact = loaded["contact"]
    c4 = loaded["c4"]
    summary = {
        "status": "STAGE16C3_C4_REFERENCE_RETIMING_VALIDATED",
        "git": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "start_head": "41e1e185d2ee38dbf9c47ba0097e8e2c80ddf47f",
        },
        "controller": "finite_virtual_6d_wrist_actuator_v1",
        "profile": c3["profile"],
        "reference_retiming": {
            "identifier": "world_wrist_reference_bank_uniform_retimed_v1",
            "time_scale": c3["reference_time_scale"],
            "source_keyframes": c3["source_keyframes"],
            "retimed_control_steps": c3["retimed_control_steps"],
            "control_hz": 20.0,
            "source_npz_modified": False,
            "source_keys_and_hashes_preserved": True,
            "shared_across_both_clips": True,
            "gains_effort_limits_and_gates_changed": False,
            "source_hashes": c3["contract"]["reference_bank"]["hashes"],
        },
        "historical_blocker": {
            "status": historical["overall"],
            "actual_blocker": historical["diagnosis"]["actual_blocker"],
            "false_blocker_status": historical["diagnosis"]["false_blocker"]["status"],
            "path_a_status": historical["path_a"]["status"],
            "path_b_status": historical["path_b"]["status"],
            "active_controller": historical["active_wrist_controller"],
            "path_a": historical["path_a"],
            "path_b": historical["path_b"],
        },
        "frozen_pd_baseline": [
            {
                "clip": "hocap_170105",
                "max_position_m": 0.01128,
                "position_rmse_m": 0.00636,
                "max_rotation_deg": 17.587,
                "rotation_rmse_deg": 7.291,
                "torque_saturation_ratio": 0.2125,
                "pass": False,
            },
            {
                "clip": "hocap_170650",
                "max_position_m": 0.01089,
                "position_rmse_m": 0.00546,
                "max_rotation_deg": 19.570,
                "rotation_rmse_deg": 7.550,
                "torque_saturation_ratio": 0.1875,
                "pass": False,
            },
        ],
        "contact": {
            "status": contact["status"],
            "reference_time_scale": contact["reference_time_scale"],
            "no_contact_max_force_n": contact["no_contact_baseline"]["max_force_n"],
            "no_contact_max_delta_v_mps": contact["no_contact_baseline"]["max_delta_v_mps"],
            "clips": contact["clips"],
        },
        "c3": {
            "status": c3["status"],
            "passes": c3["passes"],
            "process_isolation": c3["process_isolation"],
            "clips": _c3_rows(c3),
        },
        "c4": {
            "status": c4["status"],
            "rows": _c4_rows(c4),
            "oom_attempts": c4["oom_attempts"],
            "contact_buffer_failures": c4["contact_buffer_failures"],
            "selection": c4["selection"],
            "throughput_peak": _c4_throughput_peak(c4),
        },
        "c5": "NOT_RUN_OUTSIDE_ACTIVE_C3_C4_GOAL",
        "c6": {
            "status": "STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED",
            "samples": 0,
            "checkpoints": 0,
        },
        "tests": loaded["tests"],
        "artifacts": {name: _artifact(path) for name, path in inputs.items()},
        "documentation": documentation,
        "implementation": implementation,
        "commands": [
            "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python "
            "scripts/rl/isaaclab/qualify_stage16c3_retimed_contact_causality.py "
            "--accept-eula --reference-time-scale 8 --output "
            ".local/reports/stage16c3r5_reference_retiming_c4/"
            "contact_causality_scale8.json",
            "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python "
            "scripts/rl/isaaclab/qualify_stage16c3_retimed_semantics.py --accept-eula "
            "--reference-time-scale 8 --contact-report "
            ".local/reports/stage16c3r5_reference_retiming_c4/"
            "contact_causality_scale8.json --output "
            ".local/reports/stage16c3r5_reference_retiming_c4/"
            "c3_full_qualification_scale8_final.json",
            "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python "
            "scripts/rl/isaaclab/benchmark_stage16c4_vector_env.py --accept-eula "
            "--reference-time-scale 8 --profile high_authority_bounded --env-counts "
            "128 512 1024 2048 4096 --warmup-steps 100 --measurement-steps 500 "
            "--output .local/reports/stage16c3r5_reference_retiming_c4/"
            "c4_gpu_vector_benchmark_scale8.json",
        ],
        "prohibited_actions": {
            "ppo_started": False,
            "ppo_samples": 0,
            "ppo_checkpoints": 0,
            "push": False,
            "pull_request": False,
            "merge": False,
            "tag": False,
            "release": False,
        },
        "limitations": [
            "no PPO and no PPO checkpoint",
            "explicit serial 3P3R wrist is an abstract engineering actuator",
            "no real arm model",
            "physical provenance remains unresolved",
            "Isaac Sim 5.1 is a legacy unsupported stack",
            "C4 stability does not imply linear scaling or training-optimal throughput",
            "C4 global utilization and throughput were observed under unrelated concurrent "
            "CUDA workload; process VRAM was sampled separately",
            "no real-world dynamics or sim-to-real claim",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "final_summary.json"
    handoff_path = args.output_root / "handoff.md"
    for path in (summary_path, handoff_path):
        if path.exists():
            raise FileExistsError(f"C3R5_CLOSEOUT_REFUSES_OVERWRITE: {path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    handoff_path.write_text(_handoff(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
