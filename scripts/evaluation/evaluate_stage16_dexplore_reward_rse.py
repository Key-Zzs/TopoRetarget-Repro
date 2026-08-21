#!/usr/bin/env python3
"""Evaluate one grouped-multiplicative/RSE checkpoint under frozen C4 gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.audit_stage16_zero_g_frozen_actor_contact import _full_start
from scripts.rl.isaaclab.evaluate_stage16d_ppo26d import model_from_checkpoint
from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
    FROZEN_GATES,
    _evaluate_geometry_with_exact_broadphase,
    _inter_finger_penetration,
    _load_gate,
    _parallel_rollouts,
    _reconstruct_hand,
    _seeds,
    _valid_rows,
)
from toporetarget.evaluation import hand_metric_series, object_metric_series
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.stage16_authority_v2 import angular_velocity_semantic_alignment
from toporetarget.rl.stage16_pf_df import (
    DemonstrationFidelityContract,
    PhysicalFunctionalityContract,
    evaluate_demonstration_fidelity,
    evaluate_physical_functionality,
    first_true,
    persistent_finger_mask,
    terminal_threshold_pass,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, choices=(10, 20), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("DEXPLORE_EVALUATION_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows]))


def _first_multi_contact(actual: np.ndarray, valid: np.ndarray) -> tuple[int | None, int | None]:
    masked = np.asarray(actual, dtype=bool) & np.asarray(valid, dtype=bool)[:, None]
    first = first_true(masked.any(axis=-1))
    persistent = persistent_finger_mask(masked, minimum_steps=3)
    ready = first_true(persistent.sum(axis=-1) >= 2)
    return first, ready


def _episode_row(
    *,
    episode: int,
    seed: int,
    trace: dict[str, np.ndarray],
    rollout: dict[str, object],
    gate: dict[str, object],
    tracked_link_names: list[str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    valid = _valid_rows(trace)
    actual = np.asarray(trace["tip_pair_presence"], dtype=bool)
    first_contact, persistent_contact = _first_multi_contact(actual, valid)
    phase = np.asarray(trace["phase"]).astype("U24")
    lift_rows = np.flatnonzero(phase == "LIFT")
    if not len(lift_rows):
        raise ValueError("DEXPLORE_EVALUATION_LIFT_PHASE_MISSING")
    lift_frame = int(lift_rows[0])
    trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
    trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
    geometry, _ = _evaluate_geometry_with_exact_broadphase(
        clip="hocap_170105",
        object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
        hand_collision_body_pose=np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[
            :, None
        ],
        hand_collision_body_names=tuple(str(value) for value in trace["hand_collision_body_names"]),
    )
    inter_finger = _inter_finger_penetration(trace["hand_collision_body_pose"])
    geometry_safe = bool(
        float(geometry["max_penetration_m"]) < float(gate["catastrophic_penetration_m"])
        and float(geometry["p95_penetration_m"]) <= float(gate["p95_penetration_m"])
        and float(inter_finger.max(initial=0.0))
        <= float(gate["maximum_inter_finger_penetration_m"])
    )
    object_series = object_metric_series(trace["object_pose"], trace["object_reference"])
    hand_series = hand_metric_series(
        trace["hand_collision_body_pose"],
        [str(value) for value in trace["hand_collision_body_names"]],
        trace["tracked_link_reference"],
        tracked_link_names,
    )
    means = {
        "E_r_mean_deg": float(object_series["e_r_deg"].mean()),
        "E_t_mean_cm": float(object_series["e_t_cm"].mean()),
        "E_j_mean_cm": float(hand_series["e_j_cm"].mean()),
        "E_ft_mean_cm": float(hand_series["e_ft_cm"].mean()),
    }
    contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
    linear_error = np.linalg.norm(
        np.asarray(trace["object_twist"], dtype=np.float64)[:, :3]
        - np.asarray(trace["object_twist_reference"], dtype=np.float64)[:, :3],
        axis=-1,
    )
    linear_pass = terminal_threshold_pass(
        linear_error,
        contact=contact,
        valid=valid,
        contact_limit=float(gate["terminal_linear_speed_mps"]),
        free_limit=float(gate["terminal_free_object_linear_speed_mps"]),
        terminal_steps=int(gate["terminal_window_control_steps"]),
    )
    timestamps = np.arange(len(trace["object_pose"]), dtype=np.float64) * 0.05
    angular = angular_velocity_semantic_alignment(
        object_pose_wxyz=trace["object_pose"],
        trace_angular_velocity=np.asarray(trace["object_twist"], dtype=np.float64)[:, 3:],
        timestamps_s=timestamps,
        valid=valid,
    )
    authority_omega = np.asarray(angular["authority_omega_world"], dtype=np.float64)
    angular_error = np.linalg.norm(
        authority_omega - np.asarray(trace["object_twist_reference"], dtype=np.float64)[:, 3:],
        axis=-1,
    )
    angular_pass = terminal_threshold_pass(
        angular_error,
        contact=contact,
        valid=valid,
        contact_limit=float(gate["terminal_angular_speed_radps"]),
        free_limit=float(gate["terminal_free_object_angular_speed_radps"]),
        terminal_steps=int(gate["terminal_window_control_steps"]),
    )
    writes = rollout["rollout_state_writes"]
    causal = bool(
        int(writes["object_rollout_state_writes"]) == 0
        and int(writes["wrist_root_state_writes_during_step"]) == 0
    )
    action_safe = bool(
        np.isfinite(trace["action"]).all() and (np.abs(trace["action"]) <= 1.0).all()
    )
    lift_dz = float(trace["object_pose"][-1, 2] - trace["object_pose"][0, 2])
    pf = evaluate_physical_functionality(
        causal_execution=causal,
        geometry_safe=geometry_safe,
        action_bounds_safe=action_safe,
        prelift_multifinger_grasp_ready=bool(
            persistent_contact is not None and persistent_contact <= lift_frame
        ),
        lift_dz_m=lift_dz,
        no_hidden_control=causal,
        contract=PhysicalFunctionalityContract(),
    )
    df = evaluate_demonstration_fidelity(
        e_r_mean_deg=means["E_r_mean_deg"],
        e_t_mean_cm=means["E_t_mean_cm"],
        e_j_mean_cm=means["E_j_mean_cm"],
        e_ft_mean_cm=means["E_ft_mean_cm"],
        linear_pass_under_v1=linear_pass,
        angular_trace_pass_under_v1=angular_pass,
        angular_pose_pass_under_v1=angular_pass,
        contract=DemonstrationFidelityContract(),
    )
    accepted = bool(
        pf["pf"]
        and df["df_pose"]
        and df["df_linear"]
        and df["df_angular_pose_derived"]
        and causal
        and geometry_safe
    )
    table = np.asarray(trace["table_object_contact"], dtype=bool)
    prelift = np.arange(len(phase)) < lift_frame
    lift = phase == "LIFT"
    row = {
        "episode": episode,
        "seed": seed,
        "PF": bool(pf["pf"]),
        "PF_failure_reasons": ";".join(pf["pf_failure_reasons"]),
        "lift": bool(pf["lift_success"]),
        "DF_pose": bool(df["df_pose"]),
        "DF_linear": bool(df["df_linear"]),
        "DF_angular_v2": bool(df["df_angular_pose_derived"]),
        "causality": causal,
        "geometry": geometry_safe,
        "PHYSICAL_HOI_ACCEPTED": accepted,
        **means,
        "Delta_v_mean_mps": float(linear_error[valid].mean()),
        "Delta_v_p95_mps": float(np.quantile(linear_error[valid], 0.95)),
        "Delta_omega_v2_mean_radps": float(angular_error[valid].mean()),
        "Delta_omega_v2_p95_radps": float(np.quantile(angular_error[valid], 0.95)),
        "first_contact": first_contact,
        "persistent_multi_contact": persistent_contact,
        "LIFT": lift_frame,
        "pre_LIFT_margin": (
            None if persistent_contact is None else int(lift_frame - persistent_contact)
        ),
        "lift_dz_m": lift_dz,
        "support_transfer": bool(table[prelift].any() and not table[lift].any()),
        "table_object_contact_fraction": float(table.mean()),
        "R_obj": float(np.asarray(trace["reward_group_object"])[valid].mean()),
        "R_hand": float(np.asarray(trace["reward_group_hand"])[valid].mean()),
        "R_int": float(np.asarray(trace["reward_group_interaction"])[valid].mean()),
        "R_reg": float(np.asarray(trace["reward_group_regularization"])[valid].mean()),
        "R_total": float(np.asarray(trace["reward_total"])[valid].mean()),
    }
    return row, trace


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("DEXPLORE_EVALUATION_REQUIRES_EULA")
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"DEXPLORE_EVALUATION_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        seeds = _seeds("hocap_170105", count=args.episodes)
        start = _full_start("hocap_170105")
        env = _make_table_env(
            clip="hocap_170105",
            num_envs=args.episodes,
            start_index=start,
            mode=ContactRewardMode.STRICT_PER_FINGER_V4,
            stage="C4",
            training_rsi=False,
            reward_aggregation_mode="grouped_multiplicative_v1",
            rse_enabled=True,
            full_horizon_evaluation=True,
        )
        trainer, payload = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip="hocap_170105"
        )
        contract = env.contract_report()
        physics = contract["gravity_friction_curriculum"]
        if (
            physics["stage"] != "C4"
            or physics["gravity_scale"] != 1.0
            or physics["friction_scale"] != 1.0
            or contract["ppo26d"]["reward_aggregation"]["mode"] != "grouped_multiplicative_v1"
            or contract["ppo26d"]["rse"]["enabled"] is not True
        ):
            raise RuntimeError("DEXPLORE_EVALUATION_RUNTIME_CONTRACT_DRIFT")
        gate = _load_gate(FROZEN_GATES, clip="hocap_170105")
        rollouts = _parallel_rollouts(
            env=env,
            trainer=trainer,
            clip="hocap_170105",
            seeds=seeds,
            start=start,
        )
        rows: list[dict[str, object]] = []
        trace_receipts: list[dict[str, object]] = []
        for episode, (rollout, trace) in enumerate(rollouts):
            row, enriched = _episode_row(
                episode=episode,
                seed=seeds[episode],
                trace=trace,
                rollout=rollout,
                gate=gate,
                tracked_link_names=list(env.reference_bank.tracked_link_names),
            )
            trace_path = output / "traces" / f"episode_{episode:02d}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, **enriched)
            row["trace"] = str(trace_path.resolve())
            row["trace_sha256"] = _sha256(trace_path)
            rows.append(row)
            trace_receipts.append(
                {
                    "episode": episode,
                    "path": str(trace_path.resolve()),
                    "sha256": row["trace_sha256"],
                }
            )
        counts = {
            name: sum(bool(row[name]) for row in rows)
            for name in (
                "PF",
                "lift",
                "DF_pose",
                "DF_linear",
                "DF_angular_v2",
                "causality",
                "geometry",
                "PHYSICAL_HOI_ACCEPTED",
            )
        }
        required = int(np.ceil(0.8 * args.episodes))
        summary = {
            "schema_version": "Stage16DexploreRewardRSEEvaluationV1",
            "clip": "hocap_170105",
            "update": args.update,
            "samples": args.samples,
            "episodes": args.episodes,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_schema": payload["schema_version"],
            "evaluation_reset": "FRAME0_DETERMINISTIC_FULL_TRAJECTORY",
            "optimizer_steps": 0,
            "counts": counts,
            "acceptance_required": required,
            "accepted": counts["PHYSICAL_HOI_ACCEPTED"] >= required,
            "group_means": {
                name: _mean(rows, name) for name in ("R_obj", "R_hand", "R_int", "R_reg", "R_total")
            },
            "timing": {
                "first_contact_median": float(
                    np.median(
                        [
                            int(row["first_contact"])
                            for row in rows
                            if row["first_contact"] is not None
                        ]
                    )
                )
                if any(row["first_contact"] is not None for row in rows)
                else None,
                "persistent_multi_contact_median": float(
                    np.median(
                        [
                            int(row["persistent_multi_contact"])
                            for row in rows
                            if row["persistent_multi_contact"] is not None
                        ]
                    )
                )
                if any(row["persistent_multi_contact"] is not None for row in rows)
                else None,
                "LIFT": int(rows[0]["LIFT"]),
                "pre_LIFT_margin_median": float(
                    np.median(
                        [
                            int(row["pre_LIFT_margin"])
                            for row in rows
                            if row["pre_LIFT_margin"] is not None
                        ]
                    )
                )
                if any(row["pre_LIFT_margin"] is not None for row in rows)
                else None,
            },
            "support_transfer_episodes": sum(bool(row["support_transfer"]) for row in rows),
            "lift_dz_mean_m": _mean(rows, "lift_dz_m"),
            "environment": contract,
            "traces": trace_receipts,
        }
        _write_csv(output / "per_episode.csv", rows)
        _write_json(output / "summary.json", summary)
        print(json.dumps({"PF": counts["PF"], "accepted": summary["accepted"]}))
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
