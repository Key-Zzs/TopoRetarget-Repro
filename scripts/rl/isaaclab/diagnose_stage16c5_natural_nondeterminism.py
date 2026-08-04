#!/usr/bin/env python3
"""Run one bounded Stage 16-C.5A natural-baseline diagnostic cell.

The script never captures/restores a state.  It follows the repaired manual
control boundary used by C.5A so terminal buffers are measured before any
automatic reset.  ``single`` is E1, ``vector`` is E2/E3, and the telemetry
argument provides an E6 worker.  Cross-process E4/E5 use this same script in
fresh child processes, keeping those results out of the same-process gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--mode", choices=("single", "vector"), required=True)
    parser.add_argument(
        "--telemetry", choices=("off", "aggregate", "diagnostic"), default="aggregate"
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--cross-process-worker",
        action="store_true",
        help="permit exactly one trial for an E4/E5 child process only",
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_DIAGNOSTIC_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
        raise ValueError(f"malformed C5A frame selection: {path}")
    return value


def _clone(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone(child) for name, child in value.items()}
    raise TypeError(f"unsupported diagnostic value: {type(value)!r}")


def _fingerprint(value: Any) -> str:
    """Return a stable byte-level fingerprint without retaining live tensors."""

    import torch

    digest = hashlib.sha256()

    def visit(child: Any, prefix: str) -> None:
        if isinstance(child, dict):
            for name in sorted(child):
                visit(child[name], f"{prefix}.{name}")
            return
        if not isinstance(child, torch.Tensor):
            raise TypeError(f"diagnostic tensors required, got {type(child)!r}")
        cpu = child.detach().contiguous().cpu()
        digest.update(prefix.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())

    visit(value, "root")
    return digest.hexdigest()


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contact_telemetry(env: Any, ids: Any) -> dict[str, Any]:
    """Capture read-only per-environment telemetry identities at this boundary."""

    requested_ids = [int(value) for value in ids.detach().cpu().tolist()]
    latest: dict[int, dict[str, object]] = {}
    for record in getattr(env, "contact_substep_records", []):
        if not isinstance(record, dict):
            continue
        env_id = record.get("env_id")
        if isinstance(env_id, int) and env_id in requested_ids:
            latest[env_id] = record
    per_environment = {
        str(env_id): _json_fingerprint(latest[env_id]) if env_id in latest else None
        for env_id in requested_ids
    }
    return {
        "mode": str(env.cfg.contact_telemetry),
        "record_total": int(getattr(env, "_contact_substep_record_total", 0)),
        "latest_record_count": len(latest),
        "per_environment": per_environment,
        "fingerprint": _json_fingerprint(per_environment),
    }


def _max_abs(first: Any, second: Any) -> float:
    import torch

    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("diagnostic tensors required")
    if first.shape != second.shape:
        raise ValueError(f"diagnostic tensor shape mismatch: {first.shape} != {second.shape}")
    if first.dtype == torch.bool or first.dtype == torch.long:
        return float((first != second).to(torch.float32).amax().detach().cpu())
    return float((first - second).abs().amax().detach().cpu())


def _measurement(env: Any, ids: Any) -> dict[str, Any]:
    import torch

    from toporetarget.rl.isaaclab_oracle.candidate_state import capture_candidate_state
    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    if str(env.device).startswith("cuda"):
        torch.cuda.synchronize(env.device)
    raw = capture_candidate_state(env, ids).tensors
    derived = {
        "state": state_view(env, ids),
        "observation": env._get_observations()["policy"].index_select(0, ids),
        "reward_components": {
            name: value.index_select(0, ids) for name, value in env._last_reward_terms.items()
        },
        "terminated": env.reset_terminated.index_select(0, ids),
        "timed_out": env.reset_time_outs.index_select(0, ids),
        "reason_codes": env._reason_codes.index_select(0, ids),
    }
    measurement = {"raw": _clone(raw), "derived": _clone(derived)}
    return {
        **measurement,
        "fingerprints": {
            "raw": _fingerprint(measurement["raw"]),
            "derived": _fingerprint(measurement["derived"]),
        },
        "contact_telemetry": _contact_telemetry(env, ids),
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    import torch

    from toporetarget.rl.isaaclab_oracle.metrics import state_differences

    raw = {
        name: _max_abs(value, candidate["raw"][name])
        for name, value in reference["raw"].items()
        if name in candidate["raw"]
    }
    reference_state = reference["derived"]["state"]
    candidate_state = candidate["derived"]["state"]
    state = state_differences(reference_state, candidate_state)
    reward_components = {
        name: _max_abs(value, candidate["derived"]["reward_components"][name])
        for name, value in reference["derived"]["reward_components"].items()
    }
    termination_exact = bool(
        torch.equal(reference["derived"]["terminated"], candidate["derived"]["terminated"])
        and torch.equal(reference["derived"]["timed_out"], candidate["derived"]["timed_out"])
        and torch.equal(reference["derived"]["reason_codes"], candidate["derived"]["reason_codes"])
    )
    return {
        "raw_state_max_abs": raw,
        "derived_state": state,
        "observation_max_abs": _max_abs(
            reference["derived"]["observation"], candidate["derived"]["observation"]
        ),
        "reward_components_max_abs": reward_components,
        "termination_exact": termination_exact,
    }


def _source_and_peers(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    def select(mapping: dict[str, Any], index: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, child in mapping.items():
            if isinstance(child, dict):
                result[name] = select(child, index)
            else:
                result[name] = child[index : index + 1]
        return result

    def peers(mapping: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, child in mapping.items():
            if isinstance(child, dict):
                result[name] = peers(child)
            else:
                result[name] = child[1:]
        return result

    source = select(value, 0)
    peer = peers(value)
    for name, child in source["raw"].items():
        source["raw"][name] = child.expand_as(peer["raw"][name])
    for name, child in source["derived"]["state"].items():
        source["derived"]["state"][name] = child.expand_as(peer["derived"]["state"][name])
    source["derived"]["observation"] = source["derived"]["observation"].expand_as(
        peer["derived"]["observation"]
    )
    for name, child in source["derived"]["reward_components"].items():
        source["derived"]["reward_components"][name] = child.expand_as(
            peer["derived"]["reward_components"][name]
        )
    for name in ("terminated", "timed_out", "reason_codes"):
        source["derived"][name] = source["derived"][name].expand_as(peer["derived"][name])
    return source, peer


def _origin_invariance(value: dict[str, Any]) -> dict[str, Any]:
    """Measure world-origin offsets separately from local physical deltas."""

    import torch

    raw = value["raw"]
    origins = raw["source_env_origins"]
    result: dict[str, Any] = {
        "unique_origin_count": int(torch.unique(origins, dim=0).shape[0]),
    }
    for name in (
        "robot_root_state",
        "object_170105_root_state",
        "object_170650_root_state",
    ):
        world_position = raw[name][:, :3]
        local_position = world_position - origins
        source_local = local_position[:1].expand_as(local_position[1:])
        source_world = world_position[:1].expand_as(world_position[1:])
        source_delta = (world_position[1:] - world_position[:1]).contiguous()
        origin_delta = (origins[1:] - origins[:1]).contiguous()
        result[name] = {
            "world_position_max_abs": _max_abs(source_world, world_position[1:]),
            "scene_local_max_abs": _max_abs(source_local, local_position[1:]),
            "world_minus_origin_delta_max_abs": _max_abs(source_delta, origin_delta),
        }
    return result


def _compare_contact_telemetry(
    reference: dict[str, Any], candidate: dict[str, Any], *, vector: bool
) -> dict[str, Any]:
    reference_mode = reference["mode"]
    candidate_mode = candidate["mode"]
    reference_per_env = reference["per_environment"]
    candidate_per_env = candidate["per_environment"]
    if not isinstance(reference_per_env, dict) or not isinstance(candidate_per_env, dict):
        raise TypeError("C5A contact telemetry fingerprints are malformed")
    if vector:
        source = reference_per_env.get("0")
        peers = [candidate_per_env.get(str(index)) for index in range(1, len(candidate_per_env))]
        exact = all(value == source for value in peers)
        return {
            "mode_exact": reference_mode == candidate_mode,
            "source_to_peer_exact": exact,
            "peer_unique_fingerprint_count": len(set(peers)),
            "source_fingerprint_present": source is not None,
        }
    return {
        "mode_exact": reference_mode == candidate_mode,
        "exact": reference["fingerprint"] == candidate["fingerprint"],
        "reference_fingerprint_present": reference["fingerprint"] is not None,
    }


def main() -> int:
    args = parse_args()
    expected_trials = 1 if args.cross_process_worker else 20
    if not args.accept_eula or args.trials != expected_trials:
        raise SystemExit(
            "C.5A natural diagnostics require --accept-eula and "
            f"exactly {expected_trials} trials for this execution mode"
        )
    if args.output.exists():
        raise FileExistsError(f"STAGE16C5A_DIAGNOSTIC_REFUSES_OVERWRITE: {args.output}")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    frames = _load(args.frames)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
        from toporetarget.rl.isaaclab_oracle.runtime import (
            make_stage16c5_env,
            reset_frozen_clip_frame_zero,
        )

        num_envs = 1 if args.mode == "single" else 33
        env = make_stage16c5_env(num_envs=num_envs, contact_telemetry=args.telemetry)
        ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        rows: list[dict[str, Any]] = []
        for clip_index, clip_row in enumerate(frames["clips"]):
            assert isinstance(clip_row, dict)
            phase_frames = clip_row["frames"]
            assert isinstance(phase_frames, dict)
            for phase, frame_value in phase_frames.items():
                frame = int(frame_value)
                baseline: dict[str, Any] | None = None
                baseline_telemetry: dict[str, Any] | None = None
                errors: list[dict[str, Any]] = []
                fingerprints: list[dict[str, str]] = []
                origin_checks: list[dict[str, Any]] = []
                for _trial in range(args.trials):
                    reset_frozen_clip_frame_zero(env, clip_index=clip_index)
                    actions = torch.zeros(
                        (env.num_envs, 26), dtype=torch.float32, device=env.device
                    )
                    for step in range(frame + 1):
                        terminated, timed_out = raw_control_step(env, actions)
                        if bool((terminated | timed_out).any()) and step < frame:
                            raise RuntimeError(
                                "C5A_NATURAL_DIAGNOSTIC_EARLY_TERMINATION: "
                                f"clip={clip_row['clip']} phase={phase} step={step}"
                            )
                    measured = _measurement(env, ids)
                    fingerprints.append(measured["fingerprints"])
                    if args.mode == "vector":
                        origin_checks.append(_origin_invariance(measured))
                    if args.mode == "vector":
                        reference, candidate = _source_and_peers(
                            {"raw": measured["raw"], "derived": measured["derived"]}
                        )
                    elif baseline is None:
                        baseline = measured
                        baseline_telemetry = measured["contact_telemetry"]
                        continue
                    else:
                        reference, candidate = baseline, measured
                    comparison = _compare(reference, candidate)
                    comparison["contact_telemetry"] = _compare_contact_telemetry(
                        measured["contact_telemetry"]
                        if args.mode == "vector"
                        else baseline_telemetry,
                        measured["contact_telemetry"],
                        vector=args.mode == "vector",
                    )
                    errors.append(comparison)
                rows.append(
                    {
                        "clip": clip_row["clip"],
                        "phase": phase,
                        "frame": frame,
                        "comparison": (
                            "same_process_sequential_single_env"
                            if args.mode == "single"
                            else "same_process_33env"
                        ),
                        "errors": errors,
                        "measurement_fingerprints": fingerprints,
                        "origin_invariance": origin_checks,
                    }
                )
        report = {
            "schema_version": "stage16c5_natural_nondeterminism_v1",
            "mode": args.mode,
            "process_mode": (
                "cross_process_worker" if args.cross_process_worker else "same_process"
            ),
            "num_envs": num_envs,
            "telemetry": args.telemetry,
            "trials": args.trials,
            "snapshot_restore_used": False,
            "rows": rows,
            "result": "DIAGNOSTIC_COMPLETE",
        }
        _write(args.output, report)
        print(json.dumps({"result": report["result"], "mode": args.mode}, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
