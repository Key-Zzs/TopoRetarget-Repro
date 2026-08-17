#!/usr/bin/env python3
"""Run the fixed-wrist Stage16 C0--C4 PPO lineages sequentially and recoverably.

This runner deliberately stops after training C4 for all four lineages.  It
does not select checkpoints, run Formal20, or overwrite historical evidence.
It must be launched from the Isaac Lab conda environment so each child shares
the already-validated runtime and owns the GPU serially.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.fixed_wrist_causal_queue import (  # noqa: E402
    QUEUE_LINEAGES,
    STAGES,
    all_four_c4_complete,
    atomic_write_json,
    initial_lineage_state,
    stage_budget_samples,
    stage_sequence,
)

HISTORICAL_ROOT = REPO_ROOT / ".local/reports/stage16_causal_physical_c0_c4"
DEFAULT_RUN_DIR = REPO_ROOT / ".local/runs/stage16_fixed_wrist_causal_ppo_rerun"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_fixed_wrist_causal_ppo_rerun"
TRAIN_SCRIPT = REPO_ROOT / "scripts/rl/isaaclab/train_stage16_full_trajectory_p3.py"
CONTROLLER_SOURCE = (
    REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _torch_hash(value: Any) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _historical_c0(lineage: dict[str, str]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = HISTORICAL_ROOT / "training" / lineage["directory"] / lineage["clip"] / "c0"
    result_path = root / "training_result.json"
    config_path = root / "training_config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"FIXED_WRIST_HISTORICAL_C0_MISSING:{root}")
    return root, _read_json(result_path), _read_json(config_path)


def _component_hashes(checkpoint: Path) -> dict[str, str]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_critic = payload["actor_critic"]
    if not isinstance(actor_critic, dict):
        raise ValueError("FIXED_WRIST_C0_ACTOR_CRITIC_INVALID")
    actor = {
        key: value
        for key, value in actor_critic.items()
        if key.startswith("actor") or key == "log_std_parameter"
    }
    critic = {key: value for key, value in actor_critic.items() if key.startswith("critic")}
    return {
        "checkpoint": _sha256(checkpoint),
        "actor": _torch_hash(actor),
        "critic": _torch_hash(critic),
        "optimizer": _torch_hash(payload["optimizer"]),
        "normalizer": _torch_hash(payload["observation_normalization"]),
        "action26": _canonical_hash(payload["action_contract"]),
        "ppo_config": _canonical_hash(payload["ppo_config"]),
    }


def audit_c0_reuse(*, report_root: Path) -> dict[str, bool]:
    """Audit C0 independently and fail closed when exact replay evidence is absent."""

    decisions: dict[str, bool] = {}
    controller_hash = _sha256(CONTROLLER_SOURCE)
    for lineage in QUEUE_LINEAGES:
        root, result, _config = _historical_c0(lineage)
        checkpoint = Path(str(result["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FIXED_WRIST_C0_CHECKPOINT_MISSING:{checkpoint}")
        environment = result.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("FIXED_WRIST_C0_ENVIRONMENT_MISSING")
        physics = environment.get("gravity_friction_curriculum")
        ppo = environment.get("ppo26d")
        if not isinstance(physics, dict) or not isinstance(ppo, dict):
            raise ValueError("FIXED_WRIST_C0_CONTRACT_INVALID")
        effective_zero_g = float(physics.get("gravity_scale", -1.0)) == 0.0 and list(
            physics.get("gravity_world_mps2", [])
        ) == [0.0, 0.0, -0.0]
        # The old result predates the runtime override.  Gravity being zero is
        # mathematically encouraging but cannot replace a same-seed
        # optimizer=0 trace in the repaired runtime.  No such trace was saved.
        reusable = False
        reason = "EMPIRICAL_EQUIVALENCE_EVIDENCE_INCOMPLETE"
        receipt = {
            "schema_version": "Stage16FixedWristC0ReuseAuditV1",
            "lineage": lineage,
            "historical_c0_root": str(root.resolve()),
            "checkpoint": str(checkpoint),
            "hashes": _component_hashes(checkpoint),
            "reward_hash": _canonical_hash(ppo.get("reward")),
            "reference_hash": result.get("reference_hash"),
            "support_hash": result.get("support_contract_hash"),
            "episode_start_hash": _canonical_hash(result.get("episode_start")),
            "controller_config_hash": _canonical_hash(
                environment.get("finite_virtual_6d_wrist_actuator")
            ),
            "current_fixed_wrist_controller_source_hash": controller_hash,
            "sample_count": result.get("stage_samples"),
            "gravity_contract": {
                "C0_WORLD_GRAVITY_SCALE": physics.get("gravity_scale"),
                "C0_WORLD_GRAVITY_MPS2": physics.get("gravity_world_mps2"),
                "C0_OBJECT_GRAVITY_SCALE": physics.get("gravity_scale"),
                "zero_g_effective": effective_zero_g,
                "repair_difference": "runtime_articulation_disable_gravity_override",
            },
            "equivalence_metrics": {
                "historical_same_seed_optimizer_zero_trace": "NOT_AVAILABLE",
                "current_same_seed_optimizer_zero_trace": "NOT_RUN_NO_HISTORICAL_COMPARATOR",
                "tolerance": {
                    "wrist_position_mean_m": 0.002,
                    "wrist_orientation_mean_deg": 1.0,
                    "finger_q_mean_rad": 0.02,
                },
            },
            "C0_REUSABLE": "YES" if reusable else "NO",
            "reason": reason,
            "old_c1_reused": False,
            "old_c2_reused": False,
            "old_c3_reused": False,
            "old_c4_reused": False,
            "created_at": _now(),
        }
        receipt_name = f"{lineage['directory']}_{lineage['clip'].removeprefix('hocap_')}.json"
        _write_receipt(report_root / "c0_reuse" / receipt_name, receipt)
        decisions[lineage["id"]] = reusable
    return decisions


def _new_state(decisions: dict[str, bool]) -> dict[str, Any]:
    per_lineage = {
        lineage["id"]: initial_lineage_state(c0_reusable=decisions[lineage["id"]])
        for lineage in QUEUE_LINEAGES
    }
    return {
        "schema_version": "Stage16FixedWristCausalPPOQueueStateV1",
        "status": "RUNNING",
        "active_lineage": None,
        "active_stage": None,
        "lineage_index": 0,
        "lineage_total": len(QUEUE_LINEAGES),
        "stage_samples_done": 0,
        "stage_samples_total": 0,
        "cumulative_samples": 0,
        "latest_checkpoint": None,
        "warnings": [],
        "technical_retries": {},
        "process_alive": True,
        "start_time": _now(),
        "last_update_time": _now(),
        "per_lineage": per_lineage,
        "ALL_FOUR_C4_COMPLETE": "NO",
    }


def _load_or_initialize_state(run_dir: Path, decisions: dict[str, bool]) -> dict[str, Any]:
    path = run_dir / "queue_state.json"
    if not path.is_file():
        return _new_state(decisions)
    state = _read_json(path)
    if not isinstance(state.get("per_lineage"), dict):
        raise ValueError("FIXED_WRIST_QUEUE_STATE_INVALID")
    return state


def _persist_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["cumulative_samples"] = sum(
        stage_budget_samples(stage)
        for lineage in state["per_lineage"].values()
        for stage in STAGES
        if lineage.get(stage.lower()) == "COMPLETE"
    )
    state["last_update_time"] = _now()
    state["ALL_FOUR_C4_COMPLETE"] = "YES" if all_four_c4_complete(state["per_lineage"]) else "NO"
    atomic_write_json(run_dir / "queue_state.json", state)


def _tail_metrics_samples(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0
    with metrics_path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 131_072))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples = value.get("stage_samples")
        if isinstance(samples, int) and samples >= 0:
            return samples
    return 0


def _stage_output(run_dir: Path, lineage: dict[str, str], stage: str) -> Path:
    return run_dir / "training" / lineage["directory"] / lineage["clip"] / stage.lower()


def _stage_checkpoint(run_dir: Path, lineage: dict[str, str], stage: str) -> Path | None:
    result_path = _stage_output(run_dir, lineage, stage) / "training_result.json"
    if not result_path.is_file():
        return None
    result = _read_json(result_path)
    checkpoint = Path(str(result.get("checkpoint", "")))
    if result.get("status") != "P3_FULL_TRAJECTORY_STAGE_COMPLETE" or not checkpoint.is_file():
        return None
    return checkpoint.resolve()


def _record_technical_failure(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "technical_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": _now(), **payload}, sort_keys=True) + "\n")


def _run_stage(
    *, run_dir: Path, state: dict[str, Any], lineage: dict[str, str], stage: str
) -> Path:
    lineage_id = lineage["id"]
    stage_root = _stage_output(run_dir, lineage, stage)
    metrics_path = stage_root / "training_metrics.jsonl"
    log_path = run_dir / "lineages" / lineage_id / f"{stage.lower()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = (
        None
        if stage == "C0"
        else _stage_checkpoint(run_dir, lineage, STAGES[STAGES.index(stage) - 1])
    )
    if stage != "C0" and previous is None:
        raise RuntimeError(f"FIXED_WRIST_QUEUE_PREDECESSOR_MISSING:{lineage_id}:{stage}")
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--accept-eula",
        "--clip",
        lineage["clip"],
        "--contact-mode",
        lineage["mode"],
        "--stage",
        stage,
        "--num-envs",
        "1024",
        "--output-root",
        str((run_dir / "training").resolve()),
        "--saturation-instrumentation-root",
        str((stage_root / "saturation").resolve()),
    ]
    if previous is not None:
        command.extend(("--resume-checkpoint", str(previous)))
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    state["active_lineage"] = lineage_id
    state["active_stage"] = stage
    state["stage_samples_done"] = _tail_metrics_samples(metrics_path)
    state["stage_samples_total"] = stage_budget_samples(stage)
    state["per_lineage"][lineage_id][stage.lower()] = "RUNNING"
    _persist_state(run_dir, state)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] COMMAND {json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            state["stage_samples_done"] = _tail_metrics_samples(metrics_path)
            _persist_state(run_dir, state)
            time.sleep(5)
        returncode = process.returncode
    checkpoint = _stage_checkpoint(run_dir, lineage, stage) if returncode == 0 else None
    if checkpoint is None:
        raise RuntimeError(f"FIXED_WRIST_QUEUE_STAGE_FAILED:{lineage_id}:{stage}:exit={returncode}")
    return checkpoint


def _queue_contract(*, decisions: dict[str, bool]) -> dict[str, Any]:
    return {
        "schema_version": "Stage16FixedWristCausalPhysicalC4V1",
        "start_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "queue_order": list(QUEUE_LINEAGES),
        "stage_order": list(STAGES),
        "stage_budgets": {stage: stage_budget_samples(stage) for stage in STAGES},
        "c0_reuse": decisions,
        "reward_v3_changed": False,
        "reward_v4_changed": False,
        "reference_changed": False,
        "action26_changed": False,
        "action_range_changed": False,
        "ppo_hyperparameters_changed": False,
        "wrist_controller_repair_source": str(CONTROLLER_SOURCE),
        "wrist_controller_repair_hash": _sha256(CONTROLLER_SOURCE),
        "old_c1_reused": False,
        "old_c2_reused": False,
        "old_c3_reused": False,
        "old_c4_reused": False,
        "formal20_automatic": False,
        "created_at": _now(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--audit-c0-only",
        action="store_true",
        help="Write the four immutable-input C0 reuse receipts without launching PPO.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    report_root = args.report_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_path = run_dir / "runner.pid"
    if pid_path.is_file():
        previous_pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        if previous_pid != os.getpid() and _pid_alive(previous_pid):
            print("DO_NOT_START_DUPLICATE_QUEUE")
            return 0
    decisions = audit_c0_reuse(report_root=report_root)
    if args.audit_c0_only:
        print(json.dumps({"C0_REUSE": decisions}, sort_keys=True))
        return 0
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    atomic_write_json(run_dir / "queue_contract.json", _queue_contract(decisions=decisions))
    state = _load_or_initialize_state(run_dir, decisions)
    state["process_alive"] = True
    _persist_state(run_dir, state)
    try:
        for index, lineage in enumerate(QUEUE_LINEAGES, start=1):
            lineage_state = state["per_lineage"][lineage["id"]]
            state["lineage_index"] = index
            for stage in stage_sequence(c0_reusable=bool(lineage_state["c0_reuse"])):
                checkpoint = _stage_checkpoint(run_dir, lineage, stage)
                if checkpoint is not None:
                    lineage_state[stage.lower()] = "COMPLETE"
                    lineage_state["latest_checkpoint"] = str(checkpoint)
                    state["latest_checkpoint"] = str(checkpoint)
                    continue
                retries = 0
                while retries < 3:
                    try:
                        checkpoint = _run_stage(
                            run_dir=run_dir, state=state, lineage=lineage, stage=stage
                        )
                    except BaseException as error:
                        retries += 1
                        state["technical_retries"][f"{lineage['id']}:{stage}"] = retries
                        lineage_state[stage.lower()] = "TECHNICAL_RETRY"
                        _record_technical_failure(
                            run_dir,
                            {
                                "lineage": lineage["id"],
                                "stage": stage,
                                "attempt": retries,
                                "exception_type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        _persist_state(run_dir, state)
                        continue
                    lineage_state[stage.lower()] = "COMPLETE"
                    lineage_state["latest_checkpoint"] = str(checkpoint)
                    state["latest_checkpoint"] = str(checkpoint)
                    state["stage_samples_done"] = stage_budget_samples(stage)
                    _persist_state(run_dir, state)
                    break
                else:
                    lineage_state[stage.lower()] = "TECHNICALLY_INCOMPLETE"
                    lineage_state["warnings"].append("LINEAGE_TECHNICALLY_INCOMPLETE")
                    _persist_state(run_dir, state)
                    break
        state["active_lineage"] = None
        state["active_stage"] = None
        state["process_alive"] = False
        state["status"] = "COMPLETE" if all_four_c4_complete(state["per_lineage"]) else "INCOMPLETE"
        _persist_state(run_dir, state)
        return 0 if state["status"] == "COMPLETE" else 2
    finally:
        if state.get("status") == "RUNNING":
            state["process_alive"] = False
            state["status"] = "TECHNICAL_CRASH"
            _persist_state(run_dir, state)


if __name__ == "__main__":
    raise SystemExit(main())
