#!/usr/bin/env python3
"""G3 calibration using frozen V3/V4 action traces in fresh PhysX workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.guidance import ObjectGuidanceContractV1  # noqa: E402

MATRIX_PATH = REPO_ROOT / "configs/physics/object_guidance_candidates_v1.yaml"
FROZEN_INPUTS = REPO_ROOT / ".local/reports/stage16_guidance_g0_g5/g3/frozen_baseline_inputs.json"
REFERENCE_ROOT = REPO_ROOT / ".local/frozen_baselines/reference_kinematics_v2"
OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_guidance_g0_g5/g3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--frozen-inputs", type=Path, default=FROZEN_INPUTS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-version", choices=("v3", "v4"))
    parser.add_argument("--worker-clip", choices=("hocap_170105", "hocap_170650"))
    parser.add_argument("--worker-trace", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def load_matrix(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ObjectGuidanceCandidateMatrixV1":
        raise ValueError("GUIDANCE_G3_CANDIDATE_MATRIX_SCHEMA_INVALID")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise ValueError("GUIDANCE_G3_CANDIDATE_MATRIX_SIZE_INVALID")
    return payload, candidates


def worker(args: argparse.Namespace) -> int:
    if None in (
        args.worker_candidate,
        args.worker_version,
        args.worker_clip,
        args.worker_trace,
        args.worker_output,
    ):
        raise ValueError("GUIDANCE_G3_WORKER_ARGUMENTS_INCOMPLETE")
    matrix, candidates = load_matrix(args.matrix.resolve())
    candidate = next((item for item in candidates if item["id"] == args.worker_candidate), None)
    if candidate is None:
        raise ValueError("GUIDANCE_G3_CANDIDATE_UNKNOWN")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as cfg_module,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        contract = ObjectGuidanceContractV1(
            mode="reference_wrench_v1", **{k: v for k, v in candidate.items() if k != "id"}
        )
        cfg = cfg_module.IsaacPPO26DReferenceTrackingEnvCfg()
        cfg_module.configure_stage16d_ppo26d(
            cfg, num_envs=1, clip=args.worker_clip, rsi=False, critical_dr=False
        )
        cfg_module.configure_stage16d_reference_kinematics_v2(cfg, reference_root=REFERENCE_ROOT)
        cfg_module.configure_stage16d_object_guidance(cfg, contract=contract)
        actions = np.load(args.worker_trace, allow_pickle=False)["action"].astype(
            np.float32, copy=False
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        env.reset(seed=20260816)
        tracking: list[float] = []
        impulse = 0.0
        work = 0.0
        active = False
        finite = True
        force_bound = True
        torque_bound = True
        for action in actions:
            observation, reward, _, _, _ = env.step(
                torch.as_tensor(action[None], device=env.device)
            )
            wrench = env._last_object_guidance
            if wrench is None:
                raise RuntimeError("GUIDANCE_G3_WRENCH_MISSING")
            finite = (
                finite
                and bool(torch.isfinite(observation["policy"]).all())
                and bool(torch.isfinite(reward).all())
            )
            force = wrench.force_world[0]
            torque = wrench.torque_world[0]
            tracking.append(float(torch.linalg.vector_norm(wrench.position_error_world[0])))
            impulse += float(torch.linalg.vector_norm(force)) * env.step_dt
            velocity = env._state()["object_twist_world"][0, :3]
            work += abs(float((force * velocity).sum())) * env.step_dt
            active = active or bool(wrench.guidance_active[0])
            force_bound = force_bound and bool(
                torch.linalg.vector_norm(force) <= wrench.force_limit_n[0] + 1e-6
            )
            torque_bound = torque_bound and bool(
                torch.linalg.vector_norm(torque) <= wrench.torque_limit_nm[0] + 1e-6
            )
        result = {
            "candidate_id": candidate["id"],
            "version": args.worker_version,
            "clip": args.worker_clip,
            "trace_path": str(args.worker_trace),
            "trace_sha256": sha256(args.worker_trace),
            "steps": int(actions.shape[0]),
            "finite": finite,
            "force_bound_pass": force_bound,
            "torque_bound_pass": torque_bound,
            "guidance_active_any": active,
            "mean_object_tracking_error_m": float(np.mean(tracking)),
            "guidance_impulse_ns": impulse,
            "guidance_absolute_work_j": work,
            "rollout_state_writes": env.rollout_state_write_report(),
            "guidance_contract": {**contract.as_dict(), "sha256": contract.sha256()},
            "calibration_input": "frozen_historical_policy_action_trace",
            "reward_mode_origin": args.worker_version,
            "candidate_matrix_sha256": sha256(args.matrix.resolve()),
        }
        write_json(args.worker_output, result)
    finally:
        if "env" in locals():
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close()
    return 0


def invoke_worker(
    args: argparse.Namespace, candidate: str, version: str, clip: str, trace: Path
) -> dict[str, Any]:
    output = args.output_root / "worker_receipts" / f"{candidate}_{version}_{clip}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--matrix",
        str(args.matrix.resolve()),
        "--worker-candidate",
        candidate,
        "--worker-version",
        version,
        "--worker-clip",
        clip,
        "--worker-trace",
        str(trace),
        "--worker-output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode or not output.is_file():
        raise RuntimeError(
            f"GUIDANCE_G3_WORKER_FAILED:{candidate}:{version}:{clip}\n{completed.stderr[-4000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if args.worker_candidate is not None:
        return worker(args)
    matrix, candidates = load_matrix(args.matrix.resolve())
    frozen = json.loads(args.frozen_inputs.read_text(encoding="utf-8"))
    try:
        rows = []
        for candidate in candidates:
            for version in ("v3", "v4"):
                for clip in ("hocap_170105", "hocap_170650"):
                    trace = Path(
                        frozen["records"][f"{version}_{clip}"]["action_trace"]["copied_path"]
                    )
                    rows.append(invoke_worker(args, candidate["id"], version, clip, trace))
        eligible = [
            r
            for r in rows
            if r["finite"]
            and r["force_bound_pass"]
            and r["torque_bound_pass"]
            and r["rollout_state_writes"]["object_rollout_state_writes"] == 0
            and r["rollout_state_writes"]["wrist_root_state_writes_during_step"] == 0
        ]
        scores = []
        for index, candidate in enumerate(candidates):
            group = [r for r in eligible if r["candidate_id"] == candidate["id"]]
            if len(group) != 4:
                continue
            scores.append(
                {
                    "id": candidate["id"],
                    "tracking": float(np.mean([r["mean_object_tracking_error_m"] for r in group])),
                    "work": float(np.mean([r["guidance_absolute_work_j"] for r in group])),
                    "impulse": float(np.mean([r["guidance_impulse_ns"] for r in group])),
                    "strength_rank": index,
                }
            )
        if not scores:
            raise RuntimeError("GUIDANCE_G3_NO_ELIGIBLE_GLOBAL_PROFILE")
        selected = min(
            scores, key=lambda x: (x["tracking"], x["work"], x["impulse"], x["strength_rank"])
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "guidance_candidate_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            args.output_root / "guidance_candidate_matrix.json",
            {"matrix": matrix, "sha256": sha256(args.matrix.resolve())},
        )
        write_json(
            args.output_root / "global_guidance_profile_selection.json",
            {
                "schema_version": "Stage16GuidanceGlobalProfileSelectionV1",
                "status": "GUIDANCE_PROFILE_EXPERIMENTAL",
                "selected_global_guidance_profile": selected["id"],
                "selection_rule": matrix["selection_rule"],
                "candidate_scores": scores,
                "all_rows": str(args.output_root / "guidance_candidate_results.csv"),
                "shared_for": ["hocap_170105", "hocap_170650", "v3", "v4"],
            },
        )
        print(f"SELECTED_GLOBAL_GUIDANCE_PROFILE={selected['id']}")
    except Exception as error:
        write_json(
            args.output_root / "technical_failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
