#!/usr/bin/env python3
"""Run the Stage16 frozen zero-g actor contact-ready/full-start A/B audit.

This is deliberately an inference-only driver.  It loads the already selected
V3/V4 checkpoints and frozen normalizers, builds the repaired fixed-wrist C0
table runtime, and records deterministic actions only.  It never calls PPO
collection/update APIs and leaves every checkpoint untouched.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.evaluate_stage16d_ppo26d import (
    _device_trace_to_numpy,
    _initial_trace_snapshot,
    _prepend_initial_trace,
    apply_episode_seed,
    model_from_checkpoint,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".local/reports/stage16_zero_g_contract_fidelity_and_frozen_contact"
)
PAIR_CONTRACT = (
    REPO_ROOT / "configs/rl/stage16/stage16_p3_p4_contact_ready_evaluation_pairs_v1.yaml"
)
START_ROOT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart/episode_start"
TOPOLOGY_PATH = (
    REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting/contact_topology.json"
)
V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("FROZEN_ACTOR_AUDIT_EMPTY_ROWS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _torch_hash(value: object) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _component_hashes(trainer: Any) -> dict[str, str]:
    actor_critic = trainer.model.state_dict()
    actor = {
        key: value
        for key, value in actor_critic.items()
        if key.startswith("actor") or key == "log_std_parameter"
    }
    return {
        "actor": _torch_hash(actor),
        "normalizer": _torch_hash(trainer.trainer.normalizer.state_dict()),
    }


def _selection(mode: str, clip: str) -> dict[str, object]:
    if mode == "aggregate_v3":
        path = V3_ROOT / clip / "dev/checkpoint_selection.json"
    elif clip == "hocap_170650":
        path = V4_ROOT / clip / "final_checkpoint_selection.json"
    else:
        path = V4_ROOT / clip / "checkpoint_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranked = payload.get("ranked")
    if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
        raise ValueError(f"FROZEN_ACTOR_AUDIT_SELECTION_INVALID:{path}")
    checkpoint = Path(str(ranked[0].get("checkpoint", ""))).resolve()
    expected_hash = ranked[0].get("checkpoint_sha256")
    if not checkpoint.is_file() or not isinstance(expected_hash, str):
        raise FileNotFoundError(f"FROZEN_ACTOR_AUDIT_CHECKPOINT_MISSING:{checkpoint}")
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_hash:
        raise RuntimeError("FROZEN_ACTOR_AUDIT_CHECKPOINT_HASH_DRIFT")
    return {
        "selection": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "samples": ranked[0].get("reward_v3_samples", ranked[0].get("reward_v4_samples")),
    }


def _load_pairs(clip: str, count: int) -> list[dict[str, int]]:
    # YAML is intentionally simple but use the project's available PyYAML rather
    # than infer a new reset state from source geometry.
    import yaml

    payload = yaml.safe_load(PAIR_CONTRACT.read_text(encoding="utf-8"))
    pairs = payload["clips"][clip]["development"]["pairs"]
    if not isinstance(pairs, list) or len(pairs) < count:
        raise ValueError("FROZEN_ACTOR_AUDIT_CONTACT_READY_PAIRS_INSUFFICIENT")
    result = [
        {"seed": int(item["seed"]), "reset_index": int(item["reset_index"])}
        for item in pairs[:count]
    ]
    if len({row["reset_index"] for row in result}) < 2:
        raise ValueError("FROZEN_ACTOR_AUDIT_CONTACT_READY_DIVERSITY_INSUFFICIENT")
    return result


def _full_start(clip: str) -> int:
    payload = json.loads((START_ROOT / f"{clip}.json").read_text(encoding="utf-8"))
    index = payload.get("start_index")
    if not isinstance(index, int) or index < 0:
        raise ValueError("FROZEN_ACTOR_AUDIT_FULL_START_INVALID")
    return index


def _reference_phase(clip: str, indices: np.ndarray) -> np.ndarray:
    """Export repository-topology phase labels alongside the reference index."""
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))["clips"][clip]
    onset = topology["source_onset_window"]
    hold = topology["final_hold_window"]
    result = np.full(indices.shape, "MANIPULATION", dtype="U24")
    result[indices >= 280] = "TERMINAL"
    result[indices < int(onset["start"]) - 16] = "PRE_CONTACT"
    result[(indices >= int(onset["start"]) - 16) & (indices < int(onset["start"]))] = "APPROACH"
    result[(indices >= int(onset["start"])) & (indices <= int(onset["end"]))] = "CONTACT"
    result[(indices > int(onset["end"])) & (indices <= int(hold["end"]))] = "GRASP"
    return result


def _longest_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _persistent(values: np.ndarray, steps: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    result = np.zeros_like(values)
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop]:
            stop += 1
        if stop - start >= steps:
            result[start:stop] = True
        start = stop
    return result


def _frozen_formula_reward(trace: dict[str, np.ndarray], *, mode: str) -> np.ndarray:
    """Recompute the frozen contact term from exported, exact pair telemetry."""
    mask = np.asarray(trace["reference_contact_mask"], dtype=bool)
    magnitude = np.linalg.norm(
        np.asarray(trace["fingertip_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    result = np.zeros(mask.shape[0], dtype=np.float64)
    if mode == "aggregate_v3":
        scale = (magnitude * mask).sum(axis=-1)
        present = mask.any(axis=-1)
        result[present] = np.exp(-1.2498435974121094 / (scale[present] + 1.0e-5))
        return result
    presence = np.asarray(trace.get("tip_pair_presence", trace["actual_contact_mask"]), dtype=bool)
    valid_force = presence & (magnitude > 1.0e-4)
    per_tip = np.where(valid_force, np.exp(-0.5766498904285564 / (magnitude + 1.0e-5)), 0.0)
    required = (per_tip * mask).sum(axis=-1)
    count = mask.sum(axis=-1)
    nonempty = count > 0
    result[nonempty] = required[nonempty] / count[nonempty]
    return result


def _trace_metrics(trace: dict[str, np.ndarray], *, mode: str) -> dict[str, object]:
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    if valid.ndim != 1 or valid.size < 2 or valid[0] or not valid[1:].all():
        raise ValueError("FROZEN_ACTOR_AUDIT_PAIR_FORCE_VALIDITY_INVALID")
    hand_contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    tip_contact = np.asarray(trace["actual_contact_mask"], dtype=bool)
    expected = np.asarray(trace["reference_contact_mask"], dtype=bool)
    if expected.shape != tip_contact.shape or expected.shape[0] != valid.size:
        raise ValueError("FROZEN_ACTOR_AUDIT_CONTACT_MASK_SHAPE_INVALID")
    force = np.linalg.norm(
        np.asarray(trace["fingertip_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    contact_reward_key = "contact_reward" if mode == "aggregate_v3" else "r_contact_v4"
    reward = np.asarray(trace[contact_reward_key], dtype=np.float64)
    # Reward callbacks begin after the reset state while the contact telemetry
    # deliberately contains an invalid frame-zero row.  Restore that matching
    # zero reward explicitly instead of shifting a post-physics reward.
    if reward.shape == (valid.size - 1,):
        reward = np.concatenate((np.zeros(1, dtype=np.float64), reward))
    if reward.shape != valid.shape:
        raise ValueError("FROZEN_ACTOR_AUDIT_CONTACT_REWARD_LENGTH_INVALID")
    active = valid
    formula_reward = _frozen_formula_reward(trace, mode=mode)
    formula_error = np.abs(reward[active] - formula_reward[active])
    if not np.all(formula_error <= 5.0e-5):
        raise ValueError(
            "FROZEN_ACTOR_AUDIT_CONTACT_REWARD_FORMULA_MISMATCH:"
            f"{float(formula_error.max(initial=0.0))}"
        )
    expected_active = expected & active[:, None]
    actual_active = tip_contact & active[:, None]
    expected_count = int(expected_active.sum())
    source_recall = (
        None
        if expected_count == 0
        else float((expected_active & actual_active).sum() / expected_count)
    )
    expected_persistent = np.stack(
        [_persistent(expected_active[:, finger]) for finger in range(expected.shape[1])], axis=-1
    )
    persistent_count = int(expected_persistent.sum())
    persistent_recall = (
        None
        if persistent_count == 0
        else float((expected_persistent & actual_active).sum() / persistent_count)
    )
    contacts = hand_contact & active
    first = np.flatnonzero(contacts)
    object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
    object_delta = object_pose[-1, :3] - object_pose[0, :3]
    table_support = np.asarray(trace["table_object_contact"], dtype=bool) & active
    support_transition = bool(np.any(table_support[1:] != table_support[:-1]))
    return {
        "any_hand_object_contact": bool(contacts.any()),
        "first_contact_step": None if not first.size else int(first[0]),
        "hand_object_contact_fraction": float(contacts.sum() / active.sum()),
        "tip_contact_fraction": float(actual_active.any(axis=-1).sum() / active.sum()),
        "source_tip_recall": source_recall,
        "persistent_tip_recall": persistent_recall,
        "max_contact_force_n": float(force[active].max(initial=0.0)),
        "nonzero_contact_reward_fraction": float((reward[active] > 0.0).mean()),
        "contact_reward_mean": float(reward[active].mean()),
        "contact_reward_max": float(reward[active].max(initial=0.0)),
        "contact_reward_formula_max_abs_error": float(formula_error.max(initial=0.0)),
        "longest_no_contact_gap": int(_longest_run(~contacts[active])),
        "grasp_persistence_steps": int(_longest_run(contacts[active])),
        "object_displacement_m": float(np.linalg.norm(object_delta)),
        "object_lift_dz_m": float(object_delta[2]),
        "object_support_transition": support_transition,
        "object_drop": bool(table_support.any() and not table_support[-1]),
        "contact_reward_activates_when_actual_contact": bool(
            np.any((reward > 0.0) & actual_active.any(axis=-1))
        ),
    }


def _run_episode(
    *, env: Any, trainer: Any, clip: str, seed: int, reset_index: int
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    import torch

    apply_episode_seed(seed)
    env.cfg.evaluation_reset_reference_indices = (reset_index,)
    observation, _ = env.reset(seed=seed)
    start = int(env._reference_index[0].item())
    if start != reset_index:
        raise RuntimeError("FROZEN_ACTOR_AUDIT_RESET_INDEX_NOT_APPLIED")
    initial = _initial_trace_snapshot(
        env,
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
    )
    initial = {
        name: (
            value.detach().cpu().numpy().copy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value).copy()
        )
        for name, value in initial.items()
    }
    # The production table scene exports this sensor only after a physics step.
    # Frame zero is deliberately marked invalid for all force-derived telemetry.
    initial["table_object_contact"] = np.zeros(1, dtype=bool)
    env.start_trace_capture(
        capacity=env.reference_bank.frame_count,
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
    )
    steps = 0
    final_reason = None
    for _ in range(env.reference_bank.frame_count):
        with torch.no_grad():
            action = trainer.trainer.distribution(observation["policy"]).mean
        observation, _, terminated, timed_out, extras = env.step(action)
        steps += 1
        if bool(terminated[0] | timed_out[0]):
            final_reason = int(extras["ppo26d"]["primary_reason_code"][0].detach().cpu())
            break
    if final_reason is None:
        raise RuntimeError("FROZEN_ACTOR_AUDIT_EPISODE_DID_NOT_TERMINATE")
    trace = _prepend_initial_trace(
        _device_trace_to_numpy(env.finish_trace_capture()), initial, all_replicas=False
    )
    if "phase_code" in trace:
        phase_names = np.asarray(
            ("PRE_CONTACT", "APPROACH", "CONTACT", "GRASP", "LIFT", "MANIPULATION", "TERMINAL")
        )
        trace["phase"] = phase_names[np.asarray(trace["phase_code"], dtype=np.int64)]
    else:
        trace["phase"] = _reference_phase(
            clip, np.asarray(trace["reference_index"], dtype=np.int64)
        )
    return {
        "seed": seed,
        "reset_index": reset_index,
        "start_reference_index": start,
        "steps": steps,
        "termination_reason": final_reason,
        "reached_reference_end": final_reason == 7,
        "rollout_state_writes": env.rollout_state_write_report(),
    }, trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes-per-condition", type=int, default=10)
    parser.add_argument("--mode", choices=("aggregate_v3", "strict_per_finger_v4"))
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.episodes_per_condition < 10 or args.episodes_per_condition > 20:
        raise ValueError("FROZEN_ACTOR_AUDIT_EPISODES_MUST_BE_10_TO_20")
    output = args.output_root.resolve()
    selected_directory = {"aggregate_v3": "v3", "strict_per_finger_v4": "v4"}.get(args.mode)
    if args.mode is not None and args.clip is None:
        raise ValueError("FROZEN_ACTOR_AUDIT_MODE_REQUIRES_CLIP")
    if output.exists() and args.mode is None:
        recovery_only = {"technical_failure.json", "frozen_inputs.json"}
        existing = {entry.name for entry in output.iterdir()}
        if existing - recovery_only:
            raise FileExistsError(f"FROZEN_ACTOR_AUDIT_OUTPUT_EXISTS:{output}")
    if (
        selected_directory is not None
        and (output / "frozen_actor" / selected_directory / str(args.clip)).exists()
    ):
        raise FileExistsError("FROZEN_ACTOR_AUDIT_CONDITION_OUTPUT_EXISTS")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app
    env = None
    try:
        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        rows: list[dict[str, object]] = []
        frozen: dict[str, object] = {
            "schema_version": "Stage16FrozenActorContactAuditInputsV1",
            "C0": {"gravity_scale": 0.0, "friction_scale": 2.0},
            "optimizer_steps": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "normalizer_updates": 0,
            "contact_ready_pairs": {
                clip: _load_pairs(clip, args.episodes_per_condition)
                for clip in ("hocap_170105", "hocap_170650")
            },
            "full_start": {clip: _full_start(clip) for clip in ("hocap_170105", "hocap_170650")},
            "topology": {"path": str(TOPOLOGY_PATH.resolve()), "sha256": _sha256(TOPOLOGY_PATH)},
        }
        frozen_path = output / "frozen_inputs.json"
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
                raise RuntimeError("FROZEN_ACTOR_AUDIT_INPUT_CONTRACT_DRIFT")
        else:
            _write_json(frozen_path, frozen)
        modes = (
            ((args.mode, selected_directory),)
            if args.mode is not None
            else (
                ("aggregate_v3", "v3"),
                ("strict_per_finger_v4", "v4"),
            )
        )
        clips = (str(args.clip),) if args.clip is not None else ("hocap_170105", "hocap_170650")
        for mode, directory in modes:
            assert mode is not None and directory is not None
            selected_mode = ContactRewardMode.parse(mode)
            for clip in clips:
                selection = _selection(mode, clip)
                env = _make_table_env(
                    clip=clip,
                    num_envs=1,
                    start_index=int(frozen["full_start"][clip]),
                    mode=selected_mode,
                    stage="C0",
                )
                contract = env.contract_report()
                physics = contract.get("gravity_friction_curriculum", {})
                if (
                    not isinstance(physics, dict)
                    or physics.get("gravity_scale") != 0.0
                    or physics.get("friction_scale") != 2.0
                ):
                    raise RuntimeError("FROZEN_ACTOR_AUDIT_C0_PHYSICS_DRIFT")
                trainer, payload = model_from_checkpoint(
                    Path(str(selection["checkpoint"])), str(env.device), expected_clip=clip
                )
                before = _component_hashes(trainer)
                for start_kind, cases in (
                    ("contact_ready", frozen["contact_ready_pairs"][clip]),
                    (
                        "full_start",
                        [
                            {"seed": item["seed"], "reset_index": frozen["full_start"][clip]}
                            for item in frozen["contact_ready_pairs"][clip]
                        ],
                    ),
                ):
                    for episode, case in enumerate(cases):
                        rollout, trace = _run_episode(
                            env=env,
                            trainer=trainer,
                            clip=clip,
                            seed=int(case["seed"]),
                            reset_index=int(case["reset_index"]),
                        )
                        metrics = _trace_metrics(trace, mode=mode)
                        after = _component_hashes(trainer)
                        if before != after:
                            raise RuntimeError("FROZEN_ACTOR_AUDIT_MODEL_OR_NORMALIZER_MUTATED")
                        writes = rollout["rollout_state_writes"]
                        if writes != {
                            "object_rollout_state_writes": 0,
                            "wrist_root_state_writes_during_step": 0,
                        }:
                            raise RuntimeError("FROZEN_ACTOR_AUDIT_ROLLOUT_STATE_WRITE")
                        trace_path = (
                            output
                            / "frozen_actor"
                            / directory
                            / clip
                            / start_kind
                            / f"episode_{episode:02d}.npz"
                        )
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(trace_path, **trace)
                        record = {
                            "reward": "V3" if mode == "aggregate_v3" else "V4",
                            "mode": mode,
                            "clip": clip,
                            "start": start_kind,
                            "episode": episode,
                            "checkpoint": selection["checkpoint"],
                            "checkpoint_sha256": selection["checkpoint_sha256"],
                            "checkpoint_samples": selection["samples"],
                            "actor_hash_before": before["actor"],
                            "actor_hash_after": after["actor"],
                            "normalizer_hash_before": before["normalizer"],
                            "normalizer_hash_after": after["normalizer"],
                            "trace": str(trace_path),
                            **rollout,
                            **metrics,
                        }
                        _write_json(trace_path.with_suffix(".json"), record)
                        rows.append(record)
                env.close()
                env = None
        if args.mode is not None:
            _write_csv(
                output / "frozen_actor" / f"comparison_{selected_directory}_{args.clip}.csv", rows
            )
        else:
            _write_csv(output / "frozen_actor" / "comparison.csv", rows)
        _write_json(
            output / "tests.json",
            {
                "schema_version": "Stage16FrozenActorContactAuditTestsV1",
                "status": "PASS",
                "assertions": {
                    "optimizer_steps": 0,
                    "actor_updates": 0,
                    "critic_updates": 0,
                    "normalizer_updates": 0,
                    "actor_hash_before_equals_after": True,
                    "normalizer_hash_before_equals_after": True,
                    "C0_gravity_scale": 0.0,
                    "C0_friction_scale": 2.0,
                    "object_rollout_state_writes": 0,
                    "wrist_root_rollout_state_writes": 0,
                },
            },
        )
        print(json.dumps({"status": "FROZEN_ACTOR_CONTACT_AUDIT_COMPLETE", "rows": len(rows)}))
        return 0
    except BaseException as error:
        _write_json(
            output / "technical_failure.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
