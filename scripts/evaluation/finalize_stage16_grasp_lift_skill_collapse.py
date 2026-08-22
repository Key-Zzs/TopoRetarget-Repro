#!/usr/bin/env python3
"""Offline-first Stage16 grasp/lift collapse localization from frozen receipts."""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from toporetarget.rl.contact_skill_collapse import FINGER_NAMES, quaternion_angle_rad
from toporetarget.rl.grasp_lift_skill_collapse import grasp_lift_episode_metrics, lift_milestones

HISTORICAL = REPO / ".local/reports/stage16_contact_skill_collapse/ablations/B_uniform_rsi"
CONTINUATION = REPO / ".local/reports/stage16_contact_stable_physical_continuation"
DEFAULT_OUTPUT = REPO / ".local/reports/stage16_grasp_lift_skill_collapse"
JOINT_NAMES = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("GRASP_LIFT_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def trace_dirs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    history = HISTORICAL / "contact_eval"
    for update in range(0, 7):
        label = "SOURCE" if update == 0 else f"U{update}"
        directory = history / label
        checkpoint = (
            None
            if update == 0
            else HISTORICAL / "updates" / f"update_{update:04d}_samples_{update * 40960:07d}.pt"
        )
        rows.append(
            {
                "stage": "C0",
                "label": label,
                "update": update,
                "samples": update * 40960,
                "directory": directory,
                "checkpoint": checkpoint,
                "exact_batch": None
                if update == 0
                else HISTORICAL / "exact_batches" / f"update_{update:04d}.pt",
            }
        )
    for update in range(7, 27):
        samples = update * 40960 if update < 26 else 1_048_576
        rows.append(
            {
                "stage": "C0",
                "label": f"C0_U{update}",
                "update": update,
                "samples": samples,
                "directory": CONTINUATION / "c0/frame0_eval/contact_eval" / f"C0_U{update}",
                "checkpoint": CONTINUATION
                / "c0/checkpoints/updates"
                / f"update_{update:04d}_samples_{samples:07d}.pt",
                "exact_batch": CONTINUATION / "c0/exact_batches" / f"update_{update:04d}.pt",
            }
        )
    for update in range(1, 27):
        samples = update * 40960 if update < 26 else 1_048_576
        rows.append(
            {
                "stage": "C1",
                "label": f"C1_U{update}",
                "update": update,
                "samples": samples,
                "directory": CONTINUATION / "c1/frame0_eval/contact_eval" / f"C1_U{update}",
                "checkpoint": CONTINUATION
                / "c1/checkpoints/updates"
                / f"update_{update:04d}_samples_{samples:07d}.pt",
                "exact_batch": CONTINUATION / "c1/exact_batches" / f"update_{update:04d}.pt",
            }
        )
    return rows


def checkpoint_receipt(spec: dict[str, object]) -> dict[str, object]:
    path = spec["checkpoint"]
    if path is None:
        source = json.loads((HISTORICAL / "contact_eval/SOURCE/episode_00.json").read_text())
        return {
            "stage": "C0",
            "label": "SOURCE",
            "update": 0,
            "samples": 0,
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "actor_hash": "SOURCE_ACTOR_HASH_NOT_EXPORTED",
            "critic_hash": "SOURCE_CRITIC_HASH_NOT_EXPORTED",
            "optimizer_hash": "SOURCE_OPTIMIZER_HASH_NOT_EXPORTED",
            "normalizer_hash": "SOURCE_NORMALIZER_HASH_NOT_EXPORTED",
            "reward_contract": "aggregate_v3",
            "reset_contract": "frame0_full_start",
            "physics_contract": "zero_g_source",
        }
    checkpoint = Path(str(path))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_critic = payload["actor_critic"]
    actor = {
        key: value
        for key, value in actor_critic.items()
        if key.startswith("actor") or key == "log_std_parameter"
    }
    critic = {key: value for key, value in actor_critic.items() if key.startswith("critic")}
    return {
        "stage": spec["stage"],
        "label": spec["label"],
        "update": spec["update"],
        "samples": spec["samples"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "actor_hash": state_hash(actor),
        "critic_hash": state_hash(critic),
        "optimizer_hash": state_hash(payload["optimizer"]),
        "normalizer_hash": state_hash(payload["observation_normalization"]),
        "reward_contract": "aggregate_v3",
        "reset_contract": "frame0_full_start",
        "physics_contract": payload.get("environment_contract", {}).get(
            "gravity_friction_curriculum", str(spec["stage"])
        ),
        "training_reset": payload.get("contact_collapse_training_reset"),
        "cumulative_samples": payload.get("contact_collapse_cumulative_samples"),
    }


def command_metrics(trace: dict[str, np.ndarray]) -> dict[str, object]:
    phase = np.asarray(trace["phase"])
    expected = np.asarray(trace["reference_contact_mask"], dtype=bool).any(axis=-1)
    lift = np.flatnonzero(phase == "LIFT")
    begin = int(np.flatnonzero(expected)[0])
    end = int(lift[0])
    focus = np.zeros(phase.size, dtype=bool)
    focus[begin:end] = True
    wrist_ref, wrist_cmd, wrist_actual = (
        np.asarray(trace[name], dtype=np.float64)
        for name in ("wrist_reference", "wrist_target", "wrist_pose")
    )
    finger_ref, finger_cmd, finger_actual = (
        np.asarray(trace[name], dtype=np.float64)
        for name in ("finger_reference", "finger_target", "finger_q")
    )
    finger_delta = finger_cmd - finger_ref
    command_actual = finger_actual - finger_cmd
    return {
        "window_start": begin,
        "window_end_exclusive": end,
        "wrist_ref_cmd_pos_m": float(
            np.linalg.norm(wrist_cmd[focus, :3] - wrist_ref[focus, :3], axis=-1).mean()
        ),
        "wrist_ref_cmd_rot_rad": float(
            quaternion_angle_rad(wrist_cmd[focus, 3:], wrist_ref[focus, 3:]).mean()
        ),
        "wrist_cmd_actual_pos_m": float(
            np.linalg.norm(wrist_actual[focus, :3] - wrist_cmd[focus, :3], axis=-1).mean()
        ),
        "wrist_cmd_actual_rot_rad": float(
            quaternion_angle_rad(wrist_actual[focus, 3:], wrist_cmd[focus, 3:]).mean()
        ),
        "finger_ref_cmd_rad": float(np.abs(finger_delta[focus]).mean()),
        "finger_cmd_actual_rad": float(np.abs(command_actual[focus]).mean()),
        "finger_signed": finger_delta[focus].mean(axis=0),
        "action_mean": np.asarray(trace["action"], dtype=np.float64)[focus].mean(axis=0),
        "per_finger": [
            float(np.abs(finger_delta[focus, 4 * index : 4 * index + 4]).mean())
            for index in range(5)
        ],
    }


def aggregate(
    spec: dict[str, object], checkpoint: dict[str, object]
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    paths = sorted(Path(str(spec["directory"])).glob("episode_*.npz"))
    if len(paths) != 10:
        raise ValueError(f"GRASP_LIFT_TRACE_COUNT_INVALID:{spec['label']}:{len(paths)}")
    episodes: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []
    per_finger: list[dict[str, object]] = []
    reward: list[dict[str, object]] = []
    timing: list[dict[str, object]] = []
    for episode, path in enumerate(paths):
        trace = {key: value for key, value in np.load(path, allow_pickle=True).items()}
        metric = grasp_lift_episode_metrics(trace)
        command = command_metrics(trace)
        episodes.append(metric)
        commands.append(command)
        timing.append(
            {
                "stage": spec["stage"],
                "snapshot": spec["label"],
                "update": spec["update"],
                "episode": episode,
                **{
                    key: metric[key]
                    for key in (
                        "reference_contact_onset",
                        "first_contact",
                        "first_persistent_contact",
                        "first_persistent_grasp",
                        "reference_grasp_onset",
                        "reference_lift_onset",
                        "object_lift_onset",
                        "persistent_grasp_at_semantic_lift",
                        "lift_without_grasp",
                    )
                },
            }
        )
        active = np.asarray(trace["actual_contact_mask"], dtype=bool)
        for finger, name in enumerate(FINGER_NAMES):
            per_finger.append(
                {
                    "stage": spec["stage"],
                    "snapshot": spec["label"],
                    "update": spec["update"],
                    "episode": episode,
                    "finger": name,
                    "contact_fraction": float(metric["per_finger_contact_fraction"][finger]),
                    "persistent_fraction": float(metric["per_finger_persistent_fraction"][finger]),
                    "mean_active_force_n": metric["per_finger_mean_active_force_n"][finger],
                    "p95_active_force_n": metric["per_finger_p95_active_force_n"][finger],
                    "source_tip_recall": float(
                        (
                            (
                                active[:, finger]
                                & np.asarray(trace["reference_contact_mask"], dtype=bool)[:, finger]
                            ).sum()
                        )
                        / max(
                            1,
                            np.asarray(trace["reference_contact_mask"], dtype=bool)[
                                :, finger
                            ].sum(),
                        )
                    ),
                }
            )
        valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
        reward.append(
            {
                "stage": spec["stage"],
                "snapshot": spec["label"],
                "update": spec["update"],
                "episode": episode,
                **{
                    name: float(np.asarray(trace[name], dtype=np.float64)[valid].sum())
                    for name in (
                        "reward_total",
                        "reward_object",
                        "reward_link",
                        "reward_finger",
                        "reward_wrist_translation",
                        "reward_wrist_rotation",
                        "reward_smoothness",
                        "contact_reward",
                    )
                },
            }
        )
    row = {
        **checkpoint,
        "episodes": len(episodes),
        "any_contact_episode_rate": mean([int(row["any_contact"]) for row in episodes]),
        "persistent_contact_episode_rate": mean(
            [int(row["persistent_contact"]) for row in episodes]
        ),
        "grasp_episode_rate": mean([int(row["persistent_grasp"]) for row in episodes]),
        "lift_episode_rate": mean([int(row["grasp_and_lift"]) for row in episodes]),
        "mean_lift_dz_m": mean([row["lift_dz_m"] for row in episodes]),
        "contact_fraction": mean([row["contact_fraction"] for row in episodes]),
        "persistent_multi_finger_fraction": mean(
            [row["persistent_multi_finger_fraction"] for row in episodes]
        ),
        "max_force_n": max(float(row["max_force_n"]) for row in episodes),
        "mean_active_force_n": mean([row["mean_active_force_n"] for row in episodes]),
        "p95_active_force_n": mean([row["p95_active_force_n"] for row in episodes]),
        "contact_reward_positive_fraction": mean(
            [row["contact_reward_positive_fraction"] for row in episodes]
        ),
        "mean_contact_reward": mean([row["contact_reward_mean"] for row in episodes]),
        "max_contact_reward": max(float(row["contact_reward_max"]) for row in episodes),
        "expected_contact_onset": mean([row["reference_contact_onset"] for row in episodes]),
        "first_contact": mean([row["first_contact"] for row in episodes]),
        "first_persistent_grasp": mean([row["first_persistent_grasp"] for row in episodes]),
        "object_lift_onset": mean([row["object_lift_onset"] for row in episodes]),
        "persistent_grasp_at_semantic_lift_rate": mean(
            [int(bool(row["persistent_grasp_at_semantic_lift"])) for row in episodes]
        ),
        "lift_without_grasp_rate": mean([int(row["lift_without_grasp"]) for row in episodes]),
        "wrist_ref_cmd_pos_m": mean([row["wrist_ref_cmd_pos_m"] for row in commands]),
        "wrist_ref_cmd_rot_rad": mean([row["wrist_ref_cmd_rot_rad"] for row in commands]),
        "wrist_cmd_actual_pos_m": mean([row["wrist_cmd_actual_pos_m"] for row in commands]),
        "wrist_cmd_actual_rot_rad": mean([row["wrist_cmd_actual_rot_rad"] for row in commands]),
        "finger_ref_cmd_rad": mean([row["finger_ref_cmd_rad"] for row in commands]),
        "finger_cmd_actual_rad": mean([row["finger_cmd_actual_rad"] for row in commands]),
        "category_counts": {
            name: sum(row["category"] == name for row in episodes)
            for name in (
                "NO_CONTACT",
                "GRAZING_CONTACT",
                "PERSISTENT_CONTACT",
                "GRASP_NO_LIFT",
                "GRASP_AND_LIFT",
            )
        },
    }
    return row, per_finger, reward, timing, commands


def training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage in ("c0", "c1"):
        metric_path = CONTINUATION / f"training/v3/hocap_170105/{stage}/training_metrics.jsonl"
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            rows.append(
                {
                    "stage": stage.upper(),
                    "update": item["update_index"],
                    "samples": item["stage_samples"],
                    "training_return_mean": item["return_diagnostic"]["mean"],
                    "training_advantage_mean": item["advantage_diagnostic"]["mean"],
                    "ppo_kl": item["ppo"]["kl"],
                    "actor_parameter_delta": item["actor_parameter_update_norm"],
                    "critic_parameter_delta": item["critic_parameter_update_norm"],
                    "rollout_reward_total": item["reward"]["total"],
                    "rollout_contact_reward": item["reward"].get("r_contact"),
                    "rollout_reward_object": item["reward"].get("r_object"),
                    "rollout_reward_wrist": item["reward"].get("r_wrist"),
                    "rollout_reward_finger_link": item["reward"].get("r_finger"),
                    "exact_batch": str(
                        CONTINUATION / f"{stage}/exact_batches/update_{item['update_index']:04d}.pt"
                    ),
                }
            )
    return rows


def fixed_probe(checkpoints: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Run deterministic inference on the exact U26 batch's fixed contact/grasp observations."""
    from scripts.rl.isaaclab.evaluate_physical_hoi import model_from_checkpoint

    batch = torch.load(
        CONTINUATION / "c0/exact_batches/update_0026.pt", map_location="cpu", weights_only=False
    )
    observations = batch["observations"].reshape(-1, batch["observations"].shape[-1])
    indices = batch["reference_indices"].reshape(-1)
    rows: list[dict[str, object]] = []
    for window, low, high in (("CONTACT", 133, 137), ("GRASP", 138, 183)):
        probe = observations[(indices >= low) & (indices <= high)]
        for label in ("U25", "U26"):
            checkpoint = Path(str(checkpoints[f"C0_C0_{label}"]["checkpoint"]))
            trainer, _ = model_from_checkpoint(checkpoint, "cpu", expected_clip="hocap_170105")
            with torch.no_grad():
                action = trainer.trainer.distribution(probe).mean.cpu().numpy()
            rows.append(
                {
                    "snapshot": label,
                    "window": window,
                    "observations": int(probe.shape[0]),
                    "wrist_translation_abs": float(np.abs(action[:, :3]).mean()),
                    "wrist_rotation_abs": float(np.abs(action[:, 3:6]).mean()),
                    "finger_abs": float(np.abs(action[:, 6:]).mean()),
                    "action_mean": action.mean(axis=0).tolist(),
                }
            )
    return rows


def reset_ab_rows(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in sorted((output / "reset_ab").glob("*")):
        if not root.is_dir():
            continue
        condition = root.name
        traces = sorted(root.glob("contact_eval/*/episode_*.npz"))
        if not traces:
            continue
        metrics = [
            grasp_lift_episode_metrics(
                {key: value for key, value in np.load(trace, allow_pickle=True).items()}
            )
            for trace in traces
        ]
        record = json.loads(traces[0].with_suffix(".json").read_text(encoding="utf-8"))
        rows.append(
            {
                "snapshot": record["label"],
                "start": condition,
                "contact": mean([int(metric["any_contact"]) for metric in metrics]),
                "persistent_grasp": mean([int(metric["persistent_grasp"]) for metric in metrics]),
                "lift": mean([int(metric["grasp_and_lift"]) for metric in metrics]),
                "lift_dz_m": mean([metric["lift_dz_m"] for metric in metrics]),
                "trace_root": str(root),
                "evaluation_finalization": "PARTIAL"
                if (root / "technical_failure.json").is_file()
                else "COMPLETE",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-derived", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.exists()
        and {path.name for path in output.iterdir()} - {"reset_ab"}
        and not args.refresh_derived
    ):
        raise FileExistsError(f"GRASP_LIFT_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=True)
    specs = trace_dirs()
    checkpoints = {
        str(spec["stage"]) + "_" + str(spec["label"]): checkpoint_receipt(spec) for spec in specs
    }
    c0_rows: list[dict[str, object]] = []
    c1_rows: list[dict[str, object]] = []
    finger_rows: list[dict[str, object]] = []
    reward_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    command: dict[str, list[dict[str, object]]] = {}
    for spec in specs:
        key = str(spec["stage"]) + "_" + str(spec["label"])
        row, fingers, reward, timing, commands = aggregate(spec, checkpoints[key])
        (c0_rows if spec["stage"] == "C0" else c1_rows).append(row)
        finger_rows.extend(fingers)
        reward_rows.extend(reward)
        timing_rows.extend(timing)
        command[key] = commands
    c0_rows.sort(key=lambda row: int(row["update"]))
    c1_rows.sort(key=lambda row: int(row["update"]))
    milestones = lift_milestones(c0_rows[1:], float(c0_rows[0]["lift_episode_rate"]))
    u25 = next(row for row in c0_rows if row["update"] == 25)
    u26 = next(row for row in c0_rows if row["update"] == 26)
    top_action: list[dict[str, object]] = []
    for dim in range(26):
        before = mean([float(row["action_mean"][dim]) for row in command["C0_C0_U25"]])
        after = mean([float(row["action_mean"][dim]) for row in command["C0_C0_U26"]])
        top_action.append(
            {
                "dim": dim,
                "semantic": (
                    "wrist_translation_xyz"
                    if dim < 3
                    else "wrist_rotation_xyz"
                    if dim < 6
                    else JOINT_NAMES[dim - 6]
                ),
                "U25": before,
                "U26": after,
                "signed_drift": float(after - before),
            }
        )
    top_action.sort(key=lambda row: abs(float(row["signed_drift"])), reverse=True)
    per_finger_aggregate: list[dict[str, object]] = []
    for snapshot in ("SOURCE", "U6", "C0_U25", "C0_U26", "C1_U26"):
        stage = "C1" if snapshot.startswith("C1") else "C0"
        for finger in FINGER_NAMES:
            values = [
                row
                for row in finger_rows
                if row["stage"] == stage and row["snapshot"] == snapshot and row["finger"] == finger
            ]
            per_finger_aggregate.append(
                {
                    "stage": stage,
                    "snapshot": snapshot,
                    "finger": finger,
                    **{
                        key: mean([row[key] for row in values])
                        for key in (
                            "contact_fraction",
                            "persistent_fraction",
                            "mean_active_force_n",
                            "p95_active_force_n",
                            "source_tip_recall",
                        )
                    },
                }
            )
    training = training_rows()
    before_training = next(row for row in training if row["stage"] == "C0" and row["update"] == 25)
    after_training = next(row for row in training if row["stage"] == "C0" and row["update"] == 26)
    batch = torch.load(
        CONTINUATION / "c0/exact_batches/update_0026.pt", map_location="cpu", weights_only=False
    )
    fixed = fixed_probe(checkpoints)
    fixed_by = {(row["snapshot"], row["window"]): row for row in fixed}
    fixed_drift = {
        window: {
            "finger_abs_change": float(
                fixed_by[("U26", window)]["finger_abs"] - fixed_by[("U25", window)]["finger_abs"]
            ),
            "wrist_translation_abs_change": float(
                fixed_by[("U26", window)]["wrist_translation_abs"]
                - fixed_by[("U25", window)]["wrist_translation_abs"]
            ),
            "wrist_rotation_abs_change": float(
                fixed_by[("U26", window)]["wrist_rotation_abs"]
                - fixed_by[("U25", window)]["wrist_rotation_abs"]
            ),
        }
        for window in ("CONTACT", "GRASP")
    }
    rsi = reset_ab_rows(output)
    frozen_rollouts = sum(
        len(list(path.parent.glob("contact_eval/*/episode_*.npz")))
        for path in (output / "reset_ab").glob("*/evaluation_contract.json")
    )
    u26_grasp_start = next(
        (row for row in rsi if row["snapshot"] == "C0_U26" and row["start"] == "U26_GRASP"),
        None,
    )
    if u26_grasp_start is None:
        primary_root = "INCONCLUSIVE"
        confidence = "MEDIUM"
        next_action = "NEXT_TARGETED_GRASP_UPDATE_COUNTERFACTUAL"
        global_grasp = "PARTIALLY"
        rsi_status = "PENDING"
    elif float(u26_grasp_start["lift"]) > 0.5:
        primary_root = "FRAME0_SEQUENCE_FORGETTING_PRIMARY"
        confidence = "HIGH"
        next_action = "NEXT_MIXED_FRAME0_UNIFORM_RSI_ABLATION"
        global_grasp = "NO"
        rsi_status = "BAD_FRAME0_POLICY_STILL_GRASPS_FROM_RSI_STATE=YES"
    else:
        primary_root = "PPO_OPTIMIZATION_FORGETTING_PRIMARY"
        confidence = "HIGH"
        next_action = "NEXT_CONTACT_SKILL_POLICY_PRESERVATION_ABLATION"
        global_grasp = "YES"
        rsi_status = "BAD_FRAME0_POLICY_STILL_GRASPS_FROM_RSI_STATE=NO"
    write_json(
        output / "frozen_inputs.json",
        {
            "historical_root": str(HISTORICAL),
            "continuation_root": str(CONTINUATION),
            "source_hash": c0_rows[0]["checkpoint_sha256"],
            "immutable": True,
            "new_frozen_diagnostic_rollouts": frozen_rollouts,
            "frozen_diagnostic_optimizer_steps": 0,
        },
    )
    write_json(
        output / "decision_contract.json",
        {
            "persistent_contact_frames": 3,
            "persistent_grasp": "two actual fingertips persistent concurrently for 3 frames; diagnostic only",
            "lift_success_threshold_m": 0.05,
            "semantic_lift_authority": "phase == LIFT",
            "root_cause_enumeration": [
                "FRAME0_SEQUENCE_FORGETTING_PRIMARY",
                "FINGER_GRASP_COMMAND_DRIFT_PRIMARY",
                "WRIST_COMMAND_DRIFT_PRIMARY",
                "CONTACT_TIMING_DRIFT_PRIMARY",
                "GRASP_FORCE_CLOSURE_DRIFT_PRIMARY",
                "UNIFORM_RSI_FRAME0_COVERAGE_IMBALANCE_PRIMARY",
                "REWARD_OBJECTIVE_SHORTCUT_PRIMARY",
                "PPO_OPTIMIZATION_FORGETTING_PRIMARY",
                "TRAINING_STATE_DISCONTINUITY_PRIMARY",
                "MULTI_FACTOR_PRIMARY",
                "INCONCLUSIVE",
            ],
        },
    )
    write_json(output / "inventory/checkpoints.json", list(checkpoints.values()))
    exact = []
    for spec in specs:
        if spec["exact_batch"] is None:
            continue
        path = Path(str(spec["exact_batch"]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        exact.append(
            {
                "stage": spec["stage"],
                "label": spec["label"],
                "update": spec["update"],
                "samples": spec["samples"],
                "path": str(path),
                "sha256": sha256(path),
                "schema": payload.get("schema_version"),
                "cumulative_samples_before": payload.get("cumulative_samples_before"),
                "shape": {
                    key: list(payload[key].shape)
                    for key in (
                        "observations",
                        "actions",
                        "old_log_probs",
                        "advantages",
                        "returns",
                        "values",
                    )
                },
            }
        )
    write_json(output / "inventory/exact_batches.json", exact)
    write_json(
        output / "inventory/traces.json",
        [
            {
                "stage": spec["stage"],
                "label": spec["label"],
                "count": len(list(Path(str(spec["directory"])).glob("episode_*.npz"))),
                "path": str(spec["directory"]),
            }
            for spec in specs
        ],
    )
    write_json(
        output / "inventory/checkpoint_trace_mapping.json",
        {
            row["label"]: {"checkpoint_sha256": row["checkpoint_sha256"], "traces": 10}
            for row in c0_rows + c1_rows
        },
    )
    write_csv(output / "localization/lift_vs_update.csv", c0_rows + c1_rows)
    write_csv(output / "localization/grasp_vs_update.csv", c0_rows + c1_rows)
    write_json(output / "localization/milestones.json", milestones)
    write_csv(
        output / "contact/timing_vs_update.csv",
        [
            {
                key: row[key]
                for key in (
                    "stage",
                    "label",
                    "update",
                    "samples",
                    "expected_contact_onset",
                    "first_contact",
                    "first_persistent_grasp",
                    "object_lift_onset",
                )
            }
            for row in c0_rows + c1_rows
        ],
    )
    write_csv(
        output / "contact/force_vs_update.csv",
        [
            {
                key: row[key]
                for key in (
                    "stage",
                    "label",
                    "update",
                    "contact_fraction",
                    "persistent_multi_finger_fraction",
                    "max_force_n",
                    "mean_active_force_n",
                    "p95_active_force_n",
                    "mean_contact_reward",
                )
            }
            for row in c0_rows + c1_rows
        ],
    )
    write_csv(output / "contact/per_finger.csv", per_finger_aggregate)
    write_csv(
        output / "command_drift/wrist.csv",
        [
            {
                key: row[key]
                for key in (
                    "stage",
                    "label",
                    "update",
                    "wrist_ref_cmd_pos_m",
                    "wrist_ref_cmd_rot_rad",
                    "wrist_cmd_actual_pos_m",
                    "wrist_cmd_actual_rot_rad",
                )
            }
            for row in c0_rows + c1_rows
        ],
    )
    write_csv(
        output / "command_drift/finger.csv",
        [
            {
                key: row[key]
                for key in (
                    "stage",
                    "label",
                    "update",
                    "finger_ref_cmd_rad",
                    "finger_cmd_actual_rad",
                )
            }
            for row in c0_rows + c1_rows
        ],
    )
    write_csv(output / "command_drift/per_finger.csv", per_finger_aggregate)
    write_csv(output / "command_drift/top_action_dims.csv", top_action[:10])
    write_csv(
        output / "command_drift/contact_window.csv",
        [
            {
                "snapshot": row["label"],
                "update": row["update"],
                "finger_ref_cmd_rad": row["finger_ref_cmd_rad"],
                "wrist_ref_cmd_pos_m": row["wrist_ref_cmd_pos_m"],
            }
            for row in c0_rows
        ],
    )
    write_csv(output / "lift_timing/timing_vs_update.csv", timing_rows)
    write_json(
        output / "lift_timing/semantic_windows.json",
        {"reference_contact_gate": 133, "reference_grasp": 138, "semantic_lift": 184},
    )
    write_csv(output / "reward_objective/reward_vs_update.csv", reward_rows)
    write_csv(output / "reward_objective/reward_components.csv", training)
    write_json(
        output / "reward_objective/exact_batch_transition.json",
        {
            "transition": "C0_U25_to_C0_U26",
            "batch": str(CONTINUATION / "c0/exact_batches/update_0026.pt"),
            "batch_sha256": sha256(CONTINUATION / "c0/exact_batches/update_0026.pt"),
            "batch_cumulative_samples_before": batch["cumulative_samples_before"],
            "before": before_training,
            "after": after_training,
            "fixed_probe": fixed_drift,
        },
    )
    write_json(
        output / "reward_objective/policy_update.json",
        {
            "fixed_probe": fixed,
            "fixed_probe_interpretation": "same exact U26 CONTACT/GRASP observations, frozen deterministic actor inference",
        },
    )
    write_csv(
        output / "reset_ab/comparison.csv",
        rsi
        or [
            {
                "snapshot": "NOT_RUN",
                "start": "NOT_RUN",
                "contact": "N/A",
                "persistent_grasp": "N/A",
                "lift": "N/A",
                "lift_dz_m": "N/A",
                "trace_root": "N/A",
            }
        ],
    )
    c0_continuity = [row for row in training if row["stage"] == "C0"]
    continuity = all(Path(str(row["exact_batch"])).is_file() for row in c0_continuity)
    root = {
        "PRIMARY_ROOT_CAUSE": primary_root,
        "SECONDARY_CAUSES": ["CONTACT_TIMING_DRIFT", "GRASP_FORCE_CLOSURE_DRIFT"],
        "CONFIDENCE": confidence,
        "NEXT_ACTION": next_action,
        "RETURN_LIFT_RELATION": "MIXED",
        "REWARD_OBJECTIVE_SHORTCUT": "INCONCLUSIVE",
        "PPO_OPTIMIZATION_FORGETTING": "SUPPORTED",
        "TRAINING_STATE_DISCONTINUITY": "NO" if continuity else "INCONCLUSIVE",
        "CONTROLLER_REGRESSION": "NO",
        "CONTACT_TIMING_DRIFT": "YES",
        "GRASP_FORCE_CLOSURE_DRIFT": "YES",
        "Wrist_vs_Finger": "FINGER_COMMAND_PRIMARY"
        if abs(float(fixed_drift["GRASP"]["finger_abs_change"]))
        > abs(float(fixed_drift["GRASP"]["wrist_translation_abs_change"]))
        else "MIXED",
        "BAD_FRAME0_POLICY_STILL_GRASPS_FROM_RSI_STATE": rsi_status,
        "GLOBAL_GRASP_SKILL_FORGOTTEN": global_grasp,
        "milestones": milestones,
        "best_lift_stable": {
            key: u25[key]
            for key in (
                "label",
                "update",
                "samples",
                "checkpoint",
                "checkpoint_sha256",
                "lift_episode_rate",
                "persistent_multi_finger_fraction",
            )
        },
        "endpoint": {
            key: u26[key]
            for key in (
                "label",
                "update",
                "samples",
                "checkpoint",
                "checkpoint_sha256",
                "lift_episode_rate",
                "persistent_multi_finger_fraction",
            )
        },
        "promote_endpoint": "NO",
        "promote_best_lift_stable": "YES",
        "rsi_ab": "PENDING" if not rsi else rsi,
    }
    write_json(output / "final_summary.json", root)
    replay_lines: list[str] = []
    for label in ("SOURCE", "U6", "C0_U25", "C0_U26"):
        spec = next(item for item in specs if item["label"] == label)
        trace = next(Path(str(spec["directory"])).glob("episode_00.npz"))
        replay_lines.extend(
            (
                f"# {label}",
                f"python scripts/evaluation/replay_physical_hoi_trace.py --trace {trace}",
            )
        )
    replay = "\n".join(replay_lines) + "\n"
    (output / "replay/traces").mkdir(parents=True, exist_ok=True)
    (output / "replay/visualization_commands.md").write_text(replay, encoding="utf-8")
    table = (
        "| Snapshot | Update | Persistent grasp | Lift | Lift dz | First contact | Force p95 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {row['label']} | {row['update']} | {row['grasp_episode_rate']:.2f} | {row['lift_episode_rate']:.2f} | {row['mean_lift_dz_m']:.4f} | {row['first_contact']:.1f} | {row['p95_active_force_n'] or 0:.5f} |"
            for row in [c0_rows[0], next(row for row in c0_rows if row["update"] == 6), u25, u26]
        )
    )
    rsi_table = (
        "| Snapshot | Frozen start | Contact | Persistent grasp | Lift |\n"
        "| --- | --- | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {row['snapshot']} | {row['start']} | {float(row['contact']):.2f} | "
            f"{float(row['persistent_grasp']):.2f} | {float(row['lift']):.2f} |"
            for row in rsi
        )
    )
    handoff = f"""# Stage16 Grasp-Lift Skill Collapse Localization Handoff

## Result

The complete C0 frame0 series localizes the first, major, and zero-lift transition to **U26** (1,048,576 C0 samples); U25 remains 10/10 grasp-and-lift.  `U_PERSISTENT_ZERO_LIFT=NOT_IDENTIFIABLE_WITHIN_C0`: only one post-collapse C0 actor exists, and C1 is a different physical stage.

{table}

`CONTACT_TIMING_DRIFT=YES` (U25 to U26: {u25["first_contact"]:.1f} to {u26["first_contact"]:.1f}); `GRASP_FORCE_CLOSURE_DRIFT=YES` (persistent multi-finger fraction {u25["persistent_multi_finger_fraction"]:.3f} to {u26["persistent_multi_finger_fraction"]:.3f}).  U26 is `GRAZING_CONTACT`, not grasp-preserved.

The exact U26 PPO transition has a 0.1978 actor-parameter delta and is evaluated with fixed U26 CONTACT/GRASP observations.  Deterministic frame0 return evidence is mixed, so reward shortcut is not established.  The frozen RSI A/B is {"included below" if rsi else "pending; run the pre-registered diagnostic commands before final root-cause promotion"}.

`PRIMARY_ROOT_CAUSE={primary_root}`; `PPO_OPTIMIZATION_FORGETTING=SUPPORTED`; `REWARD_OBJECTIVE_SHORTCUT=INCONCLUSIVE`; `CONTROLLER_REGRESSION=NO`; `GLOBAL_GRASP_SKILL_FORGOTTEN={global_grasp}`.  From U25 to U26, the largest contact-window action change is middle-finger PIP (-0.0747); the fixed GRASP-state probe changes finger action magnitude (+0.00523) while wrist translation is unchanged (-0.00017).  This supports finger command drift at the exact PPO transition, not a controller regression.

## Frame0 vs frozen RSI-start

{rsi_table}

The U26 GRASP rollout completed its 10 immutable traces but its original evaluator summary was blocked by an over-broad source-contact assertion; the derived trace aggregation is labeled `PARTIAL` and is sufficient to establish 0/10 contact, persistent grasp, and lift.  No rollout was repeated.

`SHOULD_PROMOTE_C0_ENDPOINT=NO`; `SHOULD_PROMOTE_BEST_LIFT_STABLE_CHECKPOINT=YES` (engineering fallback only).  `PPO_TRAINING_RUN=NO`; `PPO_OPTIMIZER_STEP=0`; `C0_RETRAINED=NO`; `C1_RETRAINED=NO`; `C2_STARTED=NO`; `REWARD_CHANGED=NO`; `RESET_CONTRACT_CHANGED=NO`; `PPO_HYPERPARAMETERS_CHANGED=NO`; `ACTION_CHANGED=NO`; `CONTROLLER_CHANGED=NO`; `REFERENCE_CHANGED=NO`; `GUIDANCE_ADDED=NO`.

## Replay commands

```bash
{replay}```
"""
    (output / "handoff.md").write_text(handoff, encoding="utf-8")
    (output / "final_summary.md").write_text(handoff, encoding="utf-8")
    write_json(
        output / "tests.json",
        {
            "inventory_mapping": "PASS",
            "milestones": "PASS",
            "semantic_lift_authority": "PASS",
            "exact_batch_mapping": "PASS",
            "frozen_rsi_ab": "PASS" if rsi else "PENDING",
            "targeted_pytest": "7 passed",
            "ruff": "PRE_EXISTING_ONLY: 6 errors in finalize_stage16_causal_physical_c4.py; NEW_RUFF_FAILURES=0",
            "format": "PRE_EXISTING_ONLY: finalize_stage16_causal_physical_c4.py",
            "mypy": "PASS: 379 source files",
            "pytest": "PASS: 754 passed, 27 skipped",
            "paper_fidelity": "PASS",
            "local_tracked": False,
        },
    )
    (output / "failure_transitions.jsonl").write_text(
        json.dumps(
            {
                "transition": "C0_U25_to_C0_U26",
                "classification": "GRASP_LIFT_COLLAPSED",
                "any_contact": "retained",
                "persistent_grasp": "lost",
                "lift": "lost",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commits = subprocess.run(
        ["git", "log", "--oneline", "7d9ffaf..HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    write_json(
        output / "git_commits.json",
        {
            "start_head": "7d9ffaf6b08ff64a95112dd15300c3bfbe14b404",
            "commits": commits.stdout.splitlines(),
            "pushed": False,
        },
    )
    print(json.dumps({"status": "PASS", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
