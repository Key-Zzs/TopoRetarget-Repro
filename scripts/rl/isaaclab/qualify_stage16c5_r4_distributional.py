#!/usr/bin/env python3
"""Qualify R4 distributions and benchmark persistent GPU candidate layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_CLIPS = ("hocap_170105", "hocap_170650")
_LAYOUTS = ((32, 4), (48, 4), (32, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("distribution", "pool"), help=argparse.SUPPRESS)
    parser.add_argument("--clip", choices=_CLIPS, help=argparse.SUPPRESS)
    parser.add_argument("--population", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--replicas", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5_R4_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_for_clip(path: Path, clip: str) -> tuple[Path, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("traces") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("R4 trace manifest has no traces list")
    selected: Mapping[str, object] | None = None
    for row in rows:
        if isinstance(row, Mapping) and row.get("clip") == clip:
            selected = row
    if selected is None:
        raise ValueError(f"R4 trace manifest lacks {clip}")
    trace = selected.get("action_trace")
    expected_hash = selected.get("action_trace_sha256")
    if not isinstance(trace, str) or not isinstance(expected_hash, str):
        raise ValueError("R4 trace manifest row is malformed")
    trace_path = Path(trace)
    if not trace_path.is_file() or _sha256(trace_path) != expected_hash:
        raise ValueError(f"R4 trace hash mismatch: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as source:
        actions = np.asarray(source["actions"], dtype=np.float32)
    if actions.shape != (40, 26) or not np.isfinite(actions).all():
        raise ValueError("R4 source action trace must be finite [40,26]")
    retimed = np.repeat(actions, 8, axis=0)
    if retimed.shape != (320, 26):
        raise AssertionError("factor-8 runtime action shape mismatch")
    return trace_path, retimed


def _frames_for_clip(path: Path, clip: str) -> OrderedDict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("R4 frame manifest has no clips list")
    names = (
        ("pre-contact", "pre_contact"),
        ("contact-onset", "contact_onset"),
        ("sustained-contact", "sustained_contact"),
        ("post-contact", "post_contact"),
    )
    for row in rows:
        if isinstance(row, Mapping) and row.get("clip") == clip:
            values = row.get("frames")
            if not isinstance(values, Mapping):
                break
            result = OrderedDict((phase, int(values[key])) for phase, key in names)
            if any(value < 0 or value > 320 for value in result.values()):
                raise ValueError("R4 phase frame is outside the factor-8 runtime")
            return result
    raise ValueError(f"R4 frame manifest lacks {clip}")


def _gpu_memory_mib() -> float | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    own_pid = os.getpid()
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2 and values[0] == str(own_pid):
            try:
                return float(values[1])
            except ValueError:
                return None
    return None


def _canonical_quaternion(value: Any) -> Any:
    import torch

    sign = torch.where(value[:, :1] < 0.0, -torch.ones_like(value[:, :1]), 1.0)
    return value * sign


def _contact_features(env: Any, env_ids: Any, *, frame: int) -> Any:
    import torch

    ids = [int(value) for value in env_ids.detach().cpu().tolist()]
    if frame == 0:
        return torch.zeros((len(ids), 3), dtype=torch.float32, device=env.device)
    required = env.num_envs * env.cfg.decimation
    rows = env.contact_substep_records[-required:]
    by_env: dict[int, list[Mapping[str, object]]] = {env_id: [] for env_id in ids}
    for row in rows:
        if isinstance(row, Mapping) and int(row.get("env_id", -1)) in by_env:
            by_env[int(row["env_id"])].append(row)
    result: list[list[float]] = []
    for env_id in ids:
        selected = by_env[env_id]
        contact_count = sum(int(row.get("contact_count", 0)) for row in selected)
        forces = [
            math.sqrt(
                sum(float(value) ** 2 for value in row["net_contact_force_world_on_object_n"])
            )
            for row in selected
            if isinstance(row.get("net_contact_force_world_on_object_n"), list)
        ]
        impulses = [
            row["impulse_world_on_object_ns"]
            for row in selected
            if isinstance(row.get("impulse_world_on_object_ns"), list)
        ]
        impulse_sum = [sum(float(row[index]) for row in impulses) for index in range(3)]
        result.append(
            [
                float(contact_count),
                sum(forces) / len(forces) if forces else 0.0,
                math.sqrt(sum(value * value for value in impulse_sum)),
            ]
        )
    return torch.tensor(result, dtype=torch.float32, device=env.device)


def _reward_components(env: Any, state: Mapping[str, Any], env_ids: Any) -> Any:
    import torch

    from toporetarget.rl.environments.isaaclab_backend.reward_terms import (
        world_wrist_reward_terms,
    )

    index = env._target_reference_index
    terms = world_wrist_reward_terms(
        object_axis_points=state["object_axis_points_scene"],
        object_axis_points_ref=env.reference_bank.gather(
            "object_axis_points_world_ref", env._clip_index, index
        ),
        tracked_links=state["tracked_links_scene"],
        tracked_links_ref=env.reference_bank.gather(
            "tracked_link_positions_world_ref", env._clip_index, index
        ),
        finger_q=state["finger_q"],
        finger_q_ref=env.reference_bank.gather("q_finger_ref", env._clip_index, index),
        joint_lower=env.joint_lower,
        joint_upper=env.joint_upper,
        wrist_position=state["wrist_position_scene"],
        wrist_quaternion_wxyz=state["wrist_quaternion_wxyz"],
        wrist_position_ref=env.reference_bank.gather(
            "wrist_pose_translation_world_ref", env._clip_index, index
        ),
        wrist_quaternion_ref_wxyz=env.reference_bank.gather(
            "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, index
        ),
        action=env._actions,
        previous_action=env._previous_actions,
        second_previous_action=env._second_previous_actions,
        profile=env.reward_profile,
    )
    order = (
        "object",
        "tracked_links",
        "finger_joints",
        "wrist_position",
        "wrist_rotation",
        "smoothness",
        "total",
    )
    return torch.stack([terms[name].index_select(0, env_ids) for name in order], dim=-1)


def _population(env: Any, env_ids: Any, *, clip: str, phase: str, frame: int) -> Any:
    import torch

    from toporetarget.rl.environments.isaaclab_backend.termination_terms import (
        TERMINATION_REASONS,
    )
    from toporetarget.rl.isaaclab_oracle.distributional_replication import (
        DistributionPopulationV1,
    )

    state = env._state()
    selected = {name: value.index_select(0, env_ids) for name, value in state.items()}
    object_pose = torch.cat(
        (
            selected["object_position_scene"],
            _canonical_quaternion(selected["object_quaternion_wxyz"]),
        ),
        dim=-1,
    )
    wrist_state = torch.cat(
        (
            selected["wrist_position_scene"],
            _canonical_quaternion(selected["wrist_quaternion_wxyz"]),
            selected["wrist_twist_world"],
        ),
        dim=-1,
    )
    fields = OrderedDict(
        (
            ("object_pose", object_pose),
            ("object_velocity", selected["object_twist_world"]),
            ("wrist_state", wrist_state),
            ("finger_state", torch.cat((selected["finger_q"], selected["finger_qdot"]), dim=-1)),
            ("tracked_links", selected["tracked_links_scene"].flatten(1)),
            ("reward_components", _reward_components(env, state, env_ids)),
            ("contact", _contact_features(env, env_ids, frame=frame)),
        )
    )
    reasons = tuple(
        TERMINATION_REASONS[int(value)]
        if 0 <= int(value) < len(TERMINATION_REASONS)
        else f"CODE_{int(value)}"
        for value in env._reason_codes.index_select(0, env_ids).detach().cpu().tolist()
    )
    return DistributionPopulationV1(
        clip=clip,
        phase=phase,
        reference_index=frame,
        fields=OrderedDict(
            (name, value.detach().cpu().numpy().astype(np.float64, copy=False))
            for name, value in fields.items()
        ),
        terminations=reasons,
        successes=tuple(
            bool(value) for value in env._success.index_select(0, env_ids).detach().cpu().tolist()
        ),
    )


def _run_distribution_worker(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.clip is None or args.output is None:
        raise SystemExit("distribution worker needs --accept-eula, --clip, and --output")
    trace_path, actions = _trace_for_clip(args.trace_manifest, args.clip)
    frames = _frames_for_clip(args.frames, args.clip)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    started = time.perf_counter()
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.candidate_state import (
            capture_candidate_state,
            replicate_candidate_state,
        )
        from toporetarget.rl.isaaclab_oracle.contracts import Stage16C5WriteAuditV1
        from toporetarget.rl.isaaclab_oracle.distributional_replication import (
            DistributionalCandidateReplicatorV1,
            DistributionalReplicationContractV1,
            NaturalPhysicsDistributionV1,
        )
        from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
        from toporetarget.rl.isaaclab_oracle.runtime import (
            make_stage16c5_env,
            reset_frozen_clip_frame_zero,
        )

        env = make_stage16c5_env(num_envs=21, contact_telemetry="aggregate")
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        candidate_ids = torch.arange(1, 21, dtype=torch.long, device=env.device)
        execution_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        contract = DistributionalReplicationContractV1()
        baseline_populations: dict[str, Any] = {}
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=all_ids)
        if 0 in frames.values():
            phase = next(name for name, frame in frames.items() if frame == 0)
            baseline_populations[phase] = _population(
                env, candidate_ids, clip=args.clip, phase=phase, frame=0
            )
        for step in range(1, max(frames.values()) + 1):
            action = torch.as_tensor(actions[step - 1], device=env.device).expand(env.num_envs, -1)
            raw_control_step(env, action)
            for phase, frame in frames.items():
                if frame == step:
                    baseline_populations[phase] = _population(
                        env, candidate_ids, clip=args.clip, phase=phase, frame=frame
                    )
        baselines = {
            phase: NaturalPhysicsDistributionV1.freeze(baseline_populations[phase], contract)
            for phase in frames
        }
        write_audit = Stage16C5WriteAuditV1()
        gates: dict[str, Any] = {}
        for phase, frame in frames.items():
            reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=all_ids)
            for step in range(1, frame):
                action = torch.as_tensor(actions[step - 1], device=env.device).expand(
                    env.num_envs, -1
                )
                raw_control_step(env, action)
            snapshot = capture_candidate_state(env, execution_ids)
            replicate_candidate_state(env, snapshot, candidate_ids, write_audit=write_audit)
            if frame > 0:
                action = torch.as_tensor(actions[frame - 1], device=env.device).expand(
                    env.num_envs, -1
                )
                raw_control_step(env, action)
            candidate = _population(env, candidate_ids, clip=args.clip, phase=phase, frame=frame)
            gates[phase] = DistributionalCandidateReplicatorV1(contract).qualify(
                baselines[phase], candidate
            )
        env_contract = env.contract_report()
        payload = {
            "schema_version": "stage16c5a_r4_distributional_worker_v1",
            "clip": args.clip,
            "source_action_trace": str(trace_path.resolve()),
            "source_action_trace_sha256": _sha256(trace_path),
            "frames": dict(frames),
            "contract": contract.as_dict(),
            "natural_baselines": {phase: row.as_dict() for phase, row in baselines.items()},
            "replication_gates": {phase: row.as_dict() for phase, row in gates.items()},
            "passes": all(row.passes for row in gates.values()),
            "candidate_setup_write_audit": write_audit.as_dict(),
            "formal_execution_rollout_writes": 0,
            "no_hidden_control": bool(
                env_contract["object_rollout_state_writes"] == 0
                and env_contract["wrist_root_state_writes_during_step"] == 0
            ),
            "inaccessible_physx_state_retained": list(snapshot.inaccessible_physx_state),
            "gpu_memory_mib": _gpu_memory_mib(),
            "wall_time_s": time.perf_counter() - started,
        }
        _write(args.output, payload)
        print(json.dumps({"clip": args.clip, "passes": payload["passes"]}))
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _run_pool_worker(args: argparse.Namespace) -> int:
    if (
        not args.accept_eula
        or args.population is None
        or args.replicas is None
        or args.output is None
    ):
        raise SystemExit("pool worker needs --population, --replicas, and --output")
    if (args.population, args.replicas) not in _LAYOUTS:
        raise ValueError("unsupported R4 pool layout")
    _, actions = _trace_for_clip(args.trace_manifest, "hocap_170650")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
        from toporetarget.rl.isaaclab_oracle.replica_manager import (
            PersistentRobustCandidatePoolV1,
        )
        from toporetarget.rl.isaaclab_oracle.runtime import (
            make_stage16c5_env,
            reset_frozen_clip_frame_zero,
        )

        candidate_count = args.population * 3 * args.replicas
        env = make_stage16c5_env(num_envs=candidate_count + 1, contact_telemetry="aggregate")
        clip_index = env.reference_bank.clip_ids.index("hocap_170650")
        pool = PersistentRobustCandidatePoolV1(
            env, population=args.population, replicas=args.replicas
        )
        reset_started = time.perf_counter()
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=pool.candidate_ids)
        torch.cuda.synchronize(torch.device(env.device))
        reset_latency = time.perf_counter() - reset_started
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=all_ids)
        for step in range(106):
            action = torch.as_tensor(actions[step], device=env.device).expand(env.num_envs, -1)
            raw_control_step(env, action)
        pool.dispatch_execution_state()
        permutation = pool.manager.permutation(0)
        rollout_started = time.perf_counter()
        for step in range(106, 116):
            action = torch.as_tensor(actions[step], device=env.device).expand(env.num_envs, -1)
            raw_control_step(env, action)
        torch.cuda.synchronize(torch.device(env.device))
        rollout_latency = time.perf_counter() - rollout_started
        state = env._state()["object_position_scene"]
        _, aggregation_latency = pool.aggregate_by_logical_candidate(state, permutation)
        scores = {
            (candidate, horizon): [
                float(candidate + horizon) + replica * 1.0e-3 for replica in range(args.replicas)
            ]
            for candidate in range(args.population)
            for horizon in (1, 5, 10)
        }
        mapping_gate = pool.manager.validate_mapping_invariance(scores)
        payload = pool.benchmark_record(
            rollout_latency_s=rollout_latency,
            reset_latency_s=reset_latency,
            aggregation_latency_s=aggregation_latency,
            gpu_memory_mib=_gpu_memory_mib(),
            control_steps=10,
        )
        payload.update(
            {
                "workload": "hocap_170650_contact_onset_frames_106_to_116",
                "mapping_invariance": mapping_gate,
                "candidate_slot_permutation": permutation.as_dict(),
                "no_hidden_control": bool(
                    env.contract_report()["object_rollout_state_writes"] == 0
                    and env.contract_report()["wrist_root_state_writes_during_step"] == 0
                ),
            }
        )
        _write(args.output, payload)
        print(
            json.dumps(
                {
                    "population": args.population,
                    "replicas": args.replicas,
                    "rollout_steps_per_s": payload["rollout_control_steps_per_s"],
                }
            )
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _launch(
    args: argparse.Namespace, worker_args: Sequence[str], output: Path
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--trace-manifest",
        str(args.trace_manifest.resolve()),
        "--frames",
        str(args.frames.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        *worker_args,
        "--output",
        str(output.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        check=False,
        capture_output=True,
        text=True,
    )
    log = output.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"R4 worker failed: returncode={completed.returncode} log={log}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("R4 worker output is malformed")
    return payload


def _run_parent(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.output_dir.exists():
        raise SystemExit("parent needs --accept-eula and a fresh --output-dir")
    distribution_rows = {}
    for clip in _CLIPS:
        output = args.output_dir / "workers" / f"distribution_{clip}.json"
        distribution_rows[clip] = _launch(
            args, ["--worker", "distribution", "--clip", clip], output
        )
    natural = {
        "schema_version": "natural_distribution_baseline_v1",
        "frozen_before_candidate_results": True,
        "clips": {clip: row["natural_baselines"] for clip, row in distribution_rows.items()},
    }
    gate = {
        "schema_version": "distributional_replication_gate_v1",
        "deterministic_c5a_failure_preserved": "SAME_SCENE_CONTACT_DIVERGENCE",
        "clips": {clip: row["replication_gates"] for clip, row in distribution_rows.items()},
        "passes": all(bool(row["passes"]) for row in distribution_rows.values()),
        "no_hidden_control": all(
            bool(row["no_hidden_control"]) for row in distribution_rows.values()
        ),
    }
    _write(args.output_dir / "natural_distribution_baseline.json", natural)
    _write(args.output_dir / "distributional_replication_gate.json", gate)
    pool_rows = []
    for population, replicas in _LAYOUTS:
        output = args.output_dir / "workers" / f"pool_p{population}_r{replicas}.json"
        pool_rows.append(
            _launch(
                args,
                [
                    "--worker",
                    "pool",
                    "--population",
                    str(population),
                    "--replicas",
                    str(replicas),
                ],
                output,
            )
        )
    eligible = [
        row
        for row in pool_rows
        if row.get("mapping_invariance", {}).get("ranking_unchanged")
        and row.get("no_hidden_control")
    ]
    selected = (
        max(eligible, key=lambda row: float(row["rollout_control_steps_per_s"]))
        if eligible
        else None
    )
    pool_report = {
        "schema_version": "persistent_robust_candidate_pool_benchmark_v1",
        "selection_rule": "CEM-compatible first, then highest rollout speed",
        "rows": pool_rows,
        "selected_layout": selected,
    }
    _write(args.output_dir / "persistent_candidate_pool_benchmark.json", pool_report)
    print(
        json.dumps(
            {
                "distributional_replication_passes": gate["passes"],
                "selected_pool": (
                    None if selected is None else [selected["population"], selected["replicas"]]
                ),
            }
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.worker == "distribution":
        return _run_distribution_worker(args)
    if args.worker == "pool":
        return _run_pool_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
