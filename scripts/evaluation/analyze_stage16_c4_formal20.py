#!/usr/bin/env python3
"""Analyze and export one causal table-supported C4 Formal20 trace.

This is deliberately Isaac-free.  It consumes the all-replica physical trace
written by ``evaluate_stage16d_ppo26d.py`` and never reruns a policy or alters
the frozen C4 checkpoint.  Failed episodes are retained alongside successful
ones in both the metrics and the generic simulation-data export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation import (  # noqa: E402
    PhysicsEpisodeEvidence,
    aggregate_rollouts,
    hand_metric_series,
    object_metric_series,
    trajectory_success,
)
from toporetarget.rl.geometry_audit.exact_evaluator import (  # noqa: E402
    evaluate_runtime_proxy_state,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    load_runtime_geometry_manifest,
)
from toporetarget.rl.physical_evaluation import (  # noqa: E402
    contact_metrics,
    flight_metrics,
    twist_metrics,
)
from toporetarget.rl.physical_scene_rsi import (  # noqa: E402
    TABLE_HAND_MAX_PENETRATION_M,
    _table_center_pose,
    _table_query,
    load_table_proxy,
)
from toporetarget.rl.physics_retargeting.self_collision import (  # noqa: E402
    InterFingerCapsulePenetrationV1,
    load_self_collision_contract,
)

FRAME_COUNT = 321
EPISODE_COUNT = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"C4_FORMAL20_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("C4_FORMAL20_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _array(archive: np.lib.npyio.NpzFile, name: str, suffix: tuple[int, ...]) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"C4_FORMAL20_TRACE_FIELD_MISSING:{name}")
    value = np.asarray(archive[name])
    expected = (FRAME_COUNT, EPISODE_COUNT, *suffix)
    if value.shape != expected:
        raise ValueError(f"C4_FORMAL20_TRACE_SHAPE_INVALID:{name}:{value.shape}!={expected}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"C4_FORMAL20_TRACE_NONFINITE:{name}")
    return value


def _load_trace(path: Path, *, mode: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    common_shapes = {
        "replica_object_pose": (7,),
        "replica_object_twist": (6,),
        "replica_hand_collision_body_pose": (21, 7),
        "replica_wrist_pose": (7,),
        "replica_wrist_twist_world": (6,),
        "replica_virtual_wrist_q": (6,),
        "replica_virtual_wrist_qdot": (6,),
        "replica_virtual_wrist_target_q": (6,),
        "replica_virtual_wrist_target_qdot": (6,),
        "replica_finger_q": (20,),
        "replica_finger_qdot": (20,),
        "replica_wrist_target_pose": (7,),
        "replica_finger_target_q": (20,),
        "replica_tracked_link_positions": (16, 3),
        "replica_embedded_reference_object_pose": (7,),
        "replica_embedded_reference_wrist_pose": (7,),
        "replica_embedded_reference_finger_q": (20,),
        "replica_embedded_reference_tracked_links": (16, 3),
        "replica_action": (26,),
        "replica_contact_force_world": (3,),
        "replica_actuator_effort": (26,),
        "replica_clip_index": (),
        "replica_reference_index": (),
        "replica_reason_code": (),
        "replica_terminated": (),
        "replica_timed_out": (),
        "replica_hand_object_pair_presence": (21,),
        "replica_hand_object_pair_force_world": (21, 3),
        "replica_hand_object_pair_force_valid": (),
        "replica_fingertip_object_pair_force_world": (5, 3),
        "replica_fingertip_object_pair_force_valid": (),
        "replica_table_object_contact": (),
        "replica_object_twist_reference": (6,),
    }
    mode_shapes = (
        {
            "replica_reference_contact_mask": (5,),
            "replica_actual_contact_mask": (5,),
        }
        if mode == "aggregate_v3"
        else {
            "replica_source_contact_mask": (5,),
            "replica_tip_pair_presence": (5,),
        }
    )
    with np.load(path, allow_pickle=False) as archive:
        values = {
            name: _array(archive, name, suffix)
            for name, suffix in {**common_shapes, **mode_shapes}.items()
        }
        names = tuple(
            str(value) for value in np.asarray(archive["hand_collision_body_names"]).tolist()
        )
        if names != HAND_COLLISION_BODY_NAMES:
            raise ValueError("C4_FORMAL20_HAND_COLLISION_BODY_ORDER_DRIFT")
        metadata = {
            "clip": str(np.asarray(archive["clip"]).item()),
            "checkpoint_path": str(np.asarray(archive["checkpoint_path"]).item()),
            "checkpoint_sha256": str(np.asarray(archive["checkpoint_sha256"]).item()),
            "reference_hash": str(np.asarray(archive["reference_hash"]).item()),
            "action_contract": str(np.asarray(archive["action_contract"]).item()),
        }
    valid = values["replica_hand_object_pair_force_valid"].astype(bool)
    if valid[0].any() or not valid[1:].all():
        raise ValueError("C4_FORMAL20_PAIR_FORCE_VALIDITY_MUST_EXCLUDE_ONLY_RESET")
    if not values["replica_table_object_contact"].any():
        raise ValueError("C4_FORMAL20_TABLE_SUPPORT_CONTACT_NEVER_OBSERVED")
    if metadata["action_contract"] != "26D_reference_residual":
        raise ValueError("C4_FORMAL20_ACTION_CONTRACT_DRIFT")
    return values, metadata


def _gate(path: Path, *, clip: str) -> dict[str, Any]:
    frozen = _read_json(path)
    value = frozen.get("task_gates", {}).get("clips", {}).get(clip)
    if not isinstance(value, dict):
        raise ValueError("C4_FORMAL20_FROZEN_GATE_MISSING")
    return value


def _interfinger(hand_pose: np.ndarray) -> np.ndarray:
    contract = load_self_collision_contract(
        REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml", repo_root=REPO_ROOT
    )
    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        REPO_ROOT / contract.runtime_collision_manifest_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=contract.capsule_radius_scale,
        device="cpu",
    )
    import torch

    with torch.no_grad():
        result = metric.evaluate(torch.as_tensor(hand_pose, dtype=torch.float32))[
            "maximum_penetration_m"
        ]
    return np.asarray(result.numpy(), dtype=np.float64)


def _table_penetration(
    *, clip: str, hand_pose: np.ndarray, object_pose: np.ndarray, geometry_manifest: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Query the same frozen finite table box used by the live C4 environment."""

    from toporetarget.rl.geometry_audit.convex_query import PythonFCLConvexQueryBackend

    table_path = (
        REPO_ROOT
        / ".local/reports/stage16_support_reconstruction/inference"
        / clip
        / "table_proxy.json"
    )
    table_proxy = load_table_proxy(table_path)
    backend = PythonFCLConvexQueryBackend()
    hand_proxies, object_by_clip = load_runtime_geometry_manifest(geometry_manifest)
    table_shape = backend.box(
        (
            float(table_proxy["table_extent"][0]),
            float(table_proxy["table_extent"][1]),
            float(table_proxy["table_thickness"]),
        )
    )
    table_pose = _table_center_pose(table_proxy)
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_proxies = object_by_clip[clip]
    object_shapes = [backend.proxy_shape(proxy) for proxy in object_proxies]
    frames, episodes = hand_pose.shape[:2]
    hand_table = np.empty((frames, episodes), dtype=np.float64)
    hand_signed = np.empty_like(hand_table)
    object_table = np.empty_like(hand_table)
    object_signed = np.empty_like(hand_table)
    for episode in range(episodes):
        hand_table[:, episode], hand_signed[:, episode] = _table_query(
            backend=backend,
            shapes=hand_shapes,
            proxies=hand_proxies,
            poses=hand_pose[:, episode],
            table_shape=table_shape,
            table_pose=table_pose,
        )
        object_table[:, episode], object_signed[:, episode] = _table_query(
            backend=backend,
            shapes=object_shapes,
            proxies=object_proxies,
            poses=object_pose[:, episode],
            table_shape=table_shape,
            table_pose=table_pose,
        )
    return (
        {
            "hand_table_max_penetration_m": hand_table,
            "hand_table_min_signed_distance_m": hand_signed,
            "object_table_max_penetration_m": object_table,
            "object_table_min_signed_distance_m": object_signed,
        },
        {
            "table_proxy": str(table_path.resolve()),
            "table_proxy_sha256": _sha256(table_path),
            "query_contract": "PythonFCLConvexQueryBackend",
            "hand_table_max_limit_m": TABLE_HAND_MAX_PENETRATION_M,
        },
    )


def _episode_penetration(values: np.ndarray) -> dict[str, float]:
    active = values[values > 0.0]
    return {
        "max_penetration_m": float(values.max()),
        "p95_penetration_m": float(np.quantile(active, 0.95)) if active.size else 0.0,
        "active_p95_penetration_m": float(np.quantile(active, 0.95)) if active.size else 0.0,
    }


def _terminal_stability(
    *, twist: np.ndarray, hand_contact: np.ndarray, valid: np.ndarray, gate: dict[str, Any]
) -> bool:
    indices = np.flatnonzero(valid)[
        -min(int(gate["terminal_window_control_steps"]), int(valid.sum())) :
    ]
    linear = np.linalg.norm(twist[indices, :3], axis=-1)
    angular = np.linalg.norm(twist[indices, 3:], axis=-1)
    linear_limit = np.where(
        hand_contact[indices],
        float(gate["terminal_linear_speed_mps"]),
        float(gate["terminal_free_object_linear_speed_mps"]),
    )
    angular_limit = np.where(
        hand_contact[indices],
        float(gate["terminal_angular_speed_radps"]),
        float(gate["terminal_free_object_angular_speed_radps"]),
    )
    return bool(np.all(linear <= linear_limit) and np.all(angular <= angular_limit))


def _export_simulation_data(
    *,
    root: Path,
    values: dict[str, np.ndarray],
    metadata: dict[str, Any],
    rows: list[dict[str, object]],
) -> None:
    """Export every formal episode, including failures, as generic reloadable NPZ."""

    if root.exists():
        raise FileExistsError(f"C4_FORMAL20_SIMULATION_EXPORT_ALREADY_EXISTS:{root}")
    root.mkdir(parents=True)
    replica_values = {name: value for name, value in values.items() if name.startswith("replica_")}
    for episode, row in enumerate(rows):
        episode_values = {
            name.removeprefix("replica_"): value[:, episode]
            for name, value in replica_values.items()
        }
        # The replay contract predates the explicit hand-object telemetry name.
        # Preserve that source field and export its lossless compatibility alias.
        episode_values["contact_pair_presence"] = episode_values[
            "hand_object_pair_presence"
        ].astype(bool, copy=False)
        episode_values["phase"] = np.minimum(
            (episode_values["reference_index"].astype(np.int64) * 7) // FRAME_COUNT,
            6,
        ).astype(np.int64)
        np.savez_compressed(
            root / f"episode_{episode:03d}.npz",
            **episode_values,
            metadata=np.asarray(
                json.dumps({**metadata, "episode": episode, "result": row}, sort_keys=True)
            ),
            hand_collision_body_names=np.asarray(HAND_COLLISION_BODY_NAMES),
            trace_type=np.asarray("stage16d_ppo26d"),
            clip=np.asarray(str(metadata["clip"])),
            action_contract=np.asarray(str(metadata["action_contract"])),
            checkpoint_path=np.asarray(str(metadata["checkpoint"]["path"])),
            checkpoint_sha256=np.asarray(str(metadata["checkpoint"]["sha256"])),
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "Stage16CausalPhysicalC4SimulationDataV1",
            "formal_episode_count": EPISODE_COUNT,
            "frame_count": FRAME_COUNT,
            "includes_successes_and_failures": True,
            "causal_physics": True,
            "inferred_table_support": True,
            "metadata": metadata,
            "episodes": rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--frozen-gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulation-output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = _read_json(args.evaluation.resolve())
    training = _read_json(args.training_config.resolve())
    if (
        not evaluation.get("full_trajectory_table")
        or evaluation.get("curriculum_stage") != "C4"
        or training.get("curriculum_stage") != "C4"
    ):
        raise ValueError("C4_FORMAL20_REQUIRES_TABLE_SUPPORTED_C4_ARTIFACTS")
    mode = str(evaluation.get("contact_mode"))
    if mode not in {"aggregate_v3", "strict_per_finger_v4"} or training.get("contact_mode") != mode:
        raise ValueError("C4_FORMAL20_CONTACT_MODE_PROVENANCE_DRIFT")
    clip = str(evaluation.get("clip"))
    if training.get("clip") != clip:
        raise ValueError("C4_FORMAL20_CLIP_PROVENANCE_DRIFT")
    values, metadata = _load_trace(args.trace.resolve(), mode=mode)
    if metadata["clip"] != clip or metadata["checkpoint_sha256"] != evaluation.get(
        "checkpoint_sha256"
    ):
        raise ValueError("C4_FORMAL20_CHECKPOINT_TRACE_PROVENANCE_DRIFT")
    if Path(metadata["checkpoint_path"]).resolve() != Path(str(evaluation["checkpoint"])).resolve():
        raise ValueError("C4_FORMAL20_CHECKPOINT_PATH_DRIFT")
    if str(training.get("reference_hash")) not in metadata["reference_hash"]:
        raise ValueError("C4_FORMAL20_REFERENCE_PROVENANCE_DRIFT")
    records = evaluation.get("frame_zero")
    if not isinstance(records, list) or len(records) != EPISODE_COUNT:
        raise ValueError("C4_FORMAL20_REQUIRES_EXACTLY_20_EVALUATION_RECORDS")
    gate = _gate(args.frozen_gates.resolve(), clip=clip)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"C4_FORMAL20_OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True)

    aggregate_geometry, raw_geometry = evaluate_runtime_proxy_state(
        manifest_path=args.geometry_manifest.resolve(),
        clip=clip,
        object_pose=values["replica_object_pose"],
        hand_collision_body_pose=values["replica_hand_collision_body_pose"],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    table_geometry, table_geometry_contract = _table_penetration(
        clip=clip,
        hand_pose=values["replica_hand_collision_body_pose"],
        object_pose=values["replica_object_pose"],
        geometry_manifest=args.geometry_manifest.resolve(),
    )
    raw_geometry.update(table_geometry)
    (output / "geometry").mkdir()
    np.savez_compressed(output / "geometry" / "all_replicas_pairs.npz", **raw_geometry)
    inter = _interfinger(
        values["replica_hand_collision_body_pose"].reshape(FRAME_COUNT * EPISODE_COUNT, 21, 7)
    ).reshape(FRAME_COUNT, EPISODE_COUNT)
    expected_key, actual_key = (
        ("replica_reference_contact_mask", "replica_actual_contact_mask")
        if mode == "aggregate_v3"
        else ("replica_source_contact_mask", "replica_tip_pair_presence")
    )
    rows: list[dict[str, object]] = []
    finger_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    ppo = evaluation.get("physics_contract", {}).get("ppo26d", {})
    reference_names = (
        evaluation.get("physics_contract", {}).get("reference_bank", {}).get("tracked_link_names")
    )
    if not isinstance(reference_names, list) or len(reference_names) != 16:
        raise ValueError("C4_FORMAL20_REFERENCE_LINK_MANIFEST_MISSING")
    if (
        int(ppo.get("object_rollout_state_writes", -1)) != 0
        or int(ppo.get("wrist_root_state_writes_during_step", -1)) != 0
    ):
        raise ValueError("C4_FORMAL20_CAUSAL_ROLLOUT_WRITE_DETECTED")
    for episode, record in enumerate(records):
        valid = values["replica_hand_object_pair_force_valid"][:, episode].astype(bool)
        hand_contact = values["replica_hand_object_pair_presence"][:, episode].any(axis=-1)
        expected = values[expected_key][:, episode].astype(bool)
        actual = values[actual_key][:, episode].astype(bool)
        interaction, per_finger = contact_metrics(expected=expected, actual=actual, valid=valid)
        flight = flight_metrics(
            tip_contact=actual.any(axis=-1),
            hand_contact=hand_contact,
            valid=valid,
            object_pose=values["replica_object_pose"][:, episode],
            object_twist=values["replica_object_twist"][:, episode],
        )
        twist = twist_metrics(
            actual=values["replica_object_twist"][:, episode],
            reference=values["replica_object_twist_reference"][:, episode],
            valid=valid,
            terminal_steps=int(gate["terminal_window_control_steps"]),
        )
        penetration = _episode_penetration(raw_geometry["frame_worst_penetration_m"][:, episode])
        hand_table_max = float(table_geometry["hand_table_max_penetration_m"][:, episode].max())
        inter_max = float(inter[:, episode].max())
        terminal_contact = bool(hand_contact[np.flatnonzero(valid)[-1]])
        stability = _terminal_stability(
            twist=values["replica_object_twist"][:, episode],
            hand_contact=hand_contact,
            valid=valid,
            gate=gate,
        )
        causality = bool(
            np.any(
                hand_contact[1:]
                & (
                    np.linalg.norm(
                        np.diff(values["replica_object_twist"][:, episode], axis=0), axis=-1
                    )
                    > 1.0e-7
                )
            )
        )
        absolute_geometry = bool(
            penetration["max_penetration_m"] < float(gate["catastrophic_penetration_m"])
            and penetration["p95_penetration_m"] <= float(gate["p95_penetration_m"])
        )
        evidence = PhysicsEpisodeEvidence(
            terminal_contact_pass=terminal_contact,
            terminal_stability_pass=stability,
            contact_causality_pass=causality,
            inter_finger_penetration_pass=inter_max
            <= float(gate["maximum_inter_finger_penetration_m"]),
            absolute_hand_object_penetration_pass=absolute_geometry,
            action_bounds_pass=bool(
                np.max(np.abs(values["replica_action"][:, episode])) <= float(gate["action_limit"])
            ),
            no_hidden_force=not bool(ppo.get("hidden_force_or_attachment", True)),
            no_object_rollout_state_write=True,
            no_wrist_root_teleport=True,
        )
        metrics = object_metric_series(
            values["replica_object_pose"][:, episode],
            values["replica_embedded_reference_object_pose"][:, episode],
        )
        metrics.update(
            hand_metric_series(
                values["replica_hand_collision_body_pose"][:, episode],
                list(HAND_COLLISION_BODY_NAMES),
                values["replica_embedded_reference_tracked_links"][:, episode],
                [str(name) for name in reference_names],
            )
        )
        complete = bool(record.get("reached_final_reference"))
        suite = trajectory_success(metrics, complete=complete, physics=evidence)
        row = {
            "episode": episode,
            "seed": int(record["seed"]),
            "steps": int(record["steps"]),
            "reached_reference_end": complete,
            "E_r_mean_deg": float(suite["E_r_mean_deg"]),
            "E_t_mean_cm": float(suite["E_t_mean_cm"]),
            "E_j_mean_cm": float(suite["E_j_mean_cm"]),
            "E_ft_mean_cm": float(suite["E_ft_mean_cm"]),
            "kinematic_success": bool(suite["kinematic_success"]),
            "physics_success": bool(suite["physics_success"]),
            "qualified_success": bool(suite["qualified_success"]),
            "terminal_contact": terminal_contact,
            "terminal_stability": stability,
            "contact_causality": causality,
            **interaction,
            "no_tip_contact_fraction": flight["no_tip_contact_fraction"],
            "no_hand_object_contact_fraction": flight["no_hand_object_contact_fraction"],
            "longest_flight_gap": int(flight["longest_flight_gap"]),
            "recontact_count": int(flight["recontact_count"]),
            "Delta_v_mean_mps": float(twist["Delta_v_mps"]["mean"]),
            "Delta_v_p95_mps": float(twist["Delta_v_mps"]["p95"]),
            "Delta_v_terminal_mps": float(twist["Delta_v_mps"]["terminal"]),
            "Delta_omega_mean_radps": float(twist["Delta_omega_radps"]["mean"]),
            "Delta_omega_p95_radps": float(twist["Delta_omega_radps"]["p95"]),
            "Delta_omega_terminal_radps": float(twist["Delta_omega_radps"]["terminal"]),
            "terminal_abs_v_mps": float(twist["terminal_abs_v_mps"]),
            "terminal_abs_omega_radps": float(twist["terminal_abs_omega_radps"]),
            "hand_object_max_penetration_mm": penetration["max_penetration_m"] * 1000.0,
            "hand_object_p95_penetration_mm": penetration["p95_penetration_m"] * 1000.0,
            "hand_table_max_penetration_mm": hand_table_max * 1000.0,
            "interfinger_max_penetration_mm": inter_max * 1000.0,
            "absolute_geometry_pass": absolute_geometry,
            "table_object_contact_fraction": float(
                values["replica_table_object_contact"][:, episode].mean()
            ),
        }
        rows.append(row)
        finger_rows.extend({"episode": episode, **item} for item in per_finger)
        details.append(
            {
                "episode": episode,
                "interaction": interaction,
                "flight": flight,
                "twist": twist,
                "penetration": penetration,
            }
        )
    suite = {
        "schema_version": "TopoRetargetEvaluationSuiteV2ResultV1",
        "aggregate": aggregate_rollouts(rows),
    }
    interaction = {
        "schema_version": "Stage16CausalPhysicalInteractionMetricsV1",
        "aggregate": {
            key: float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows)
            else None
            for key in (
                "source_tip_recall",
                "source_persistent_tip_recall",
                "cross_finger_compensation",
                "persistent_cross_finger_compensation",
                "fully_missing_source_contact",
                "source_contact_full_coverage",
            )
        },
        "per_finger": finger_rows,
    }
    flight = {
        "schema_version": "Stage16CausalPhysicalFlightMetricsV1",
        "no_tip_contact_fraction": float(np.mean([row["no_tip_contact_fraction"] for row in rows])),
        "no_hand_object_contact_fraction": float(
            np.mean([row["no_hand_object_contact_fraction"] for row in rows])
        ),
        "longest_flight_gap": max(int(row["longest_flight_gap"]) for row in rows),
        "recontact_count": sum(int(row["recontact_count"]) for row in rows),
    }
    twist = {
        "schema_version": "Stage16CausalPhysicalTwistMetricsV1",
        "aggregation": "equal_weight_per_episode",
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "Delta_v_mean_mps",
                "Delta_v_p95_mps",
                "Delta_v_terminal_mps",
                "Delta_omega_mean_radps",
                "Delta_omega_p95_radps",
                "Delta_omega_terminal_radps",
                "terminal_abs_v_mps",
                "terminal_abs_omega_radps",
            )
        },
        "terminal_stability_rate": float(np.mean([row["terminal_stability"] for row in rows])),
    }
    penetration = {
        "schema_version": "Stage16CausalPhysicalExactGeometryMetricsV1",
        "aggregate": aggregate_geometry,
        "hand_table": {
            **table_geometry_contract,
            "max_penetration_m": float(table_geometry["hand_table_max_penetration_m"].max()),
            "p95_penetration_m": float(
                np.quantile(table_geometry["hand_table_max_penetration_m"], 0.95)
            ),
            "absolute_geometry_pass": bool(
                table_geometry["hand_table_max_penetration_m"].max() <= TABLE_HAND_MAX_PENETRATION_M
            ),
        },
        "interfinger_max_penetration_m": float(inter.max()),
        "absolute_geometry_pass": bool(all(row["absolute_geometry_pass"] for row in rows)),
    }
    provenance = {
        "trace": {"path": str(args.trace.resolve()), "sha256": _sha256(args.trace.resolve())},
        "evaluation": {
            "path": str(args.evaluation.resolve()),
            "sha256": _sha256(args.evaluation.resolve()),
        },
        "training_config": {
            "path": str(args.training_config.resolve()),
            "sha256": _sha256(args.training_config.resolve()),
        },
        "geometry_manifest": {
            "path": str(args.geometry_manifest.resolve()),
            "sha256": _sha256(args.geometry_manifest.resolve()),
        },
        "frozen_gates": {
            "path": str(args.frozen_gates.resolve()),
            "sha256": _sha256(args.frozen_gates.resolve()),
        },
        "checkpoint": {
            "path": metadata["checkpoint_path"],
            "sha256": metadata["checkpoint_sha256"],
        },
        "reference_hash": str(training["reference_hash"]),
        "support_contract_hash": str(training["support_contract_hash"]),
        "action_contract": evaluation["physics_contract"]["ppo26d"]["action"],
        "action_contract_sha256": _canonical_hash(
            evaluation["physics_contract"]["ppo26d"]["action"]
        ),
        "controller_contract": evaluation["physics_contract"].get(
            "finite_virtual_6d_wrist_actuator"
        ),
        "controller_contract_sha256": _canonical_hash(
            evaluation["physics_contract"].get("finite_virtual_6d_wrist_actuator")
        ),
        "curriculum_stage": "C4",
        "gravity_friction": evaluation["physics_contract"]["gravity_friction_curriculum"],
        "contact_mode": mode,
        "clip": clip,
        "table_support": "finite_inferred_table_proxy_v1",
    }
    qualification = {
        "schema_version": "Stage16CausalPhysicalC4Formal20QualificationV1",
        "status": "C4_FORMAL20_COMPLETE",
        "clip": clip,
        "contact_mode": mode,
        "formal_episode_count": EPISODE_COUNT,
        "inferred_table_support": True,
        "causal_contract": {
            "external_guidance": False,
            "object_rollout_state_writes": 0,
            "wrist_root_rollout_writes": 0,
        },
        "provenance": provenance,
        "evaluation_suite_v2": suite,
        "interaction": interaction,
        "flight": flight,
        "twist": twist,
        "penetration": penetration,
        "episodes": rows,
    }
    _write_csv(output / "per_episode.csv", rows)
    _write_csv(output / "per_finger.csv", finger_rows)
    for name, value in (
        ("evaluation_suite_v2.json", suite),
        ("interaction.json", interaction),
        ("flight.json", flight),
        ("twist.json", twist),
        ("penetration.json", penetration),
        ("provenance.json", provenance),
        ("qualification.json", qualification),
    ):
        _write_json(output / name, value)
    _export_simulation_data(
        root=args.simulation_output.resolve(),
        values=values,
        metadata={**provenance, "action_contract": metadata["action_contract"]},
        rows=rows,
    )
    print(json.dumps({"status": qualification["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
