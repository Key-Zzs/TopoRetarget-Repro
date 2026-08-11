#!/usr/bin/env python3
"""Export one Reward V3 Formal20 trace as a complete, reloadable simulation dataset.

The exporter only serializes a completed frozen Formal20 trace.  It does not
launch IsaacLab, rerun a policy, infer pair force from aggregate contact force,
or discard failed episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DT_S = 1.0 / 20.0
FORMAL_FRAME_COUNT = 321
FORMAL_EPISODE_COUNT = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_SIM_DATA_JSON_OBJECT_REQUIRED:{path}")
    return value


def _string_scalar(archive: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in archive.files:
        raise ValueError(f"V3_SIM_DATA_TRACE_METADATA_MISSING:{name}")
    return str(np.asarray(archive[name]).item())


def _trace_array(
    archive: np.lib.npyio.NpzFile, name: str, *, suffix: tuple[int, ...]
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"V3_SIM_DATA_TRACE_FIELD_MISSING:{name}")
    value = np.asarray(archive[name])
    expected = (FORMAL_FRAME_COUNT, FORMAL_EPISODE_COUNT, *suffix)
    if value.shape != expected:
        raise ValueError(f"V3_SIM_DATA_TRACE_FIELD_SHAPE:{name}:{value.shape}!={expected}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"V3_SIM_DATA_TRACE_FIELD_NONFINITE:{name}")
    return value


def _load_trace(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    field_shapes = {
        "replica_object_pose": (7,),
        "replica_object_twist": (6,),
        "replica_object_axis_points": (6, 3),
        "replica_wrist_pose": (7,),
        "replica_wrist_twist_world": (6,),
        "replica_virtual_wrist_q": (6,),
        "replica_virtual_wrist_qdot": (6,),
        "replica_virtual_wrist_target_q": (6,),
        "replica_virtual_wrist_target_qdot": (6,),
        "replica_finger_q": (20,),
        "replica_finger_qdot": (20,),
        "replica_tracked_link_positions": (16, 3),
        "replica_action": (26,),
        "replica_wrist_residual": (6,),
        "replica_finger_residual": (20,),
        "replica_wrist_target_pose": (7,),
        "replica_finger_target_q": (20,),
        "replica_actuator_effort": (26,),
        "replica_reference_index": (),
        "replica_embedded_reference_object_pose": (7,),
        "replica_embedded_reference_wrist_pose": (7,),
        "replica_embedded_reference_finger_q": (20,),
        "replica_embedded_reference_tracked_links": (16, 3),
        "replica_object_twist_reference": (6,),
        "replica_reference_contact_mask": (5,),
        "replica_actual_contact_mask": (5,),
        "replica_fingertip_object_pair_force_world": (5, 3),
        "replica_fingertip_object_pair_force_valid": (),
        "replica_contact_reward": (),
        "replica_contact_force_scale": (),
        "replica_reward_total": (),
        "replica_reward_object": (),
        "replica_reward_link": (),
        "replica_reward_finger": (),
        "replica_reward_wrist_translation": (),
        "replica_reward_wrist_rotation": (),
        "replica_reward_smoothness": (),
        "replica_reward_obj_vel": (),
        "replica_reward_obj_ang_vel": (),
        "replica_error_obj_vel": (),
        "replica_error_obj_ang_vel": (),
    }
    with np.load(path, allow_pickle=False) as archive:
        values = {
            name: _trace_array(archive, name, suffix=suffix)
            for name, suffix in field_shapes.items()
        }
        if (
            values["replica_fingertip_object_pair_force_valid"][0].any()
            or not values["replica_fingertip_object_pair_force_valid"][1:].all()
        ):
            raise ValueError("V3_SIM_DATA_PAIR_FORCE_VALIDITY_MUST_EXCLUDE_ONLY_RESET")
        if not np.array_equal(
            values["replica_reference_contact_mask"],
            np.broadcast_to(
                values["replica_reference_contact_mask"][:, :1],
                values["replica_reference_contact_mask"].shape,
            ),
        ):
            raise ValueError("V3_SIM_DATA_REFERENCE_CONTACT_MASK_REPLICA_DRIFT")
        metadata = {
            "clip": _string_scalar(archive, "clip"),
            "checkpoint_path": _string_scalar(archive, "checkpoint_path"),
            "checkpoint_sha256": _string_scalar(archive, "checkpoint_sha256"),
            "reference_hash": _string_scalar(archive, "reference_hash"),
            "reference_kinematics_version": int(
                np.asarray(archive["reference_kinematics_version"]).item()
            ),
            "reward_v3_samples": int(np.asarray(archive["reward_v3_samples"]).item()),
            "pair_force_frame": _string_scalar(archive, "pair_force_frame"),
            "pair_force_units": _string_scalar(archive, "pair_force_units"),
            "pair_force_semantics": _string_scalar(archive, "pair_force_semantics"),
            "fingertip_link_names": [
                str(value) for value in np.asarray(archive["fingertip_link_names"]).tolist()
            ],
        }
    if metadata["pair_force_frame"] != "world" or metadata["pair_force_units"] != "N":
        raise ValueError("V3_SIM_DATA_PAIR_FORCE_PROVENANCE_INVALID")
    return values, metadata


def _read_suite_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != FORMAL_EPISODE_COUNT:
        raise ValueError("V3_SIM_DATA_EVALUATION_SUITE_REQUIRES_FORMAL20")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if int(row["replica"]) != index:
            raise ValueError("V3_SIM_DATA_EVALUATION_SUITE_REPLICA_ORDER_INVALID")
        result.append(row)
    return result


def _create_dataset(group: Any, name: str, value: np.ndarray) -> None:
    chunks = (min(64, value.shape[0]), *value.shape[1:])
    group.create_dataset(name, data=value, chunks=chunks, overwrite=False)


def _write_zarr(output: Path, *, values: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    try:
        import zarr
    except ModuleNotFoundError as error:  # pragma: no cover - environment contract
        raise RuntimeError("V3_SIM_DATA_REQUIRES_ZARR") from error
    if output.exists():
        raise FileExistsError(f"V3_SIM_DATA_ZARR_ALREADY_EXISTS:{output}")
    root = zarr.open_group(str(output), mode="w-")
    root.attrs.update(
        {
            "schema_version": "Stage16DRewardV3FormalSimulationDataV1",
            "control_dt_s": CONTROL_DT_S,
            "formal_frame_count": FORMAL_FRAME_COUNT,
            "formal_episode_count": FORMAL_EPISODE_COUNT,
            "causal_physics": True,
            "external_guidance": False,
            **metadata,
        }
    )
    episodes = root.create_group("episodes")
    field_groups = {
        "robot": (
            "wrist_pose",
            "wrist_twist_world",
            "virtual_wrist_q",
            "virtual_wrist_qdot",
            "virtual_wrist_target_q",
            "virtual_wrist_target_qdot",
            "finger_q",
            "finger_qdot",
            "tracked_link_positions",
            "action",
            "wrist_residual",
            "finger_residual",
            "wrist_target_pose",
            "finger_target_q",
            "actuator_effort",
        ),
        "object": ("object_pose", "object_twist", "object_axis_points"),
        "reference": (
            "reference_index",
            "embedded_reference_object_pose",
            "embedded_reference_wrist_pose",
            "embedded_reference_finger_q",
            "embedded_reference_tracked_links",
            "object_twist_reference",
        ),
        "contact": (
            "reference_contact_mask",
            "actual_contact_mask",
            "fingertip_object_pair_force_world",
            "fingertip_object_pair_force_valid",
            "contact_reward",
            "contact_force_scale",
        ),
        "reward": (
            "reward_total",
            "reward_object",
            "reward_link",
            "reward_finger",
            "reward_wrist_translation",
            "reward_wrist_rotation",
            "reward_smoothness",
            "reward_obj_vel",
            "reward_obj_ang_vel",
            "error_obj_vel",
            "error_obj_ang_vel",
        ),
    }
    pair_force = values["replica_fingertip_object_pair_force_world"]
    expected = values["replica_reference_contact_mask"]
    magnitude = np.linalg.norm(pair_force, axis=-1).astype(np.float32)
    s_contact = (magnitude * expected).sum(axis=-1, dtype=np.float32)
    impulse = np.cumsum(s_contact * CONTROL_DT_S, axis=0, dtype=np.float32)
    for episode in range(FORMAL_EPISODE_COUNT):
        group = episodes.create_group(f"episode_{episode:03d}")
        group.attrs.update({"episode": episode, "causal_physics": True, "external_guidance": False})
        for group_name, names in field_groups.items():
            child = group.create_group(group_name)
            for name in names:
                _create_dataset(
                    child,
                    name.removeprefix("embedded_reference_"),
                    values[f"replica_{name}"][:, episode],
                )
        contact = group["contact"]
        _create_dataset(contact, "fingertip_object_force_magnitude_n", magnitude[:, episode])
        _create_dataset(contact, "S_contact_n", s_contact[:, episode])
        _create_dataset(contact, "contact_impulse_ns", impulse[:, episode])


def _require_parquet() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("V3_SIM_DATA_REQUIRES_PYARROW_FOR_REAL_PARQUET") from error
    return pa, parquet


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("V3_SIM_DATA_PARQUET_ROWS_EMPTY")
    pa, parquet = _require_parquet()
    parquet.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value not in {"True", "False"}:
        raise ValueError(f"V3_SIM_DATA_BOOLEAN_INVALID:{value}")
    return value == "True"


def _episode_rows(
    *,
    values: dict[str, np.ndarray],
    qualification: dict[str, Any],
    contact: dict[str, Any],
    suite_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    qualification_rows = qualification.get("episodes")
    contact_rows = contact.get("per_replica")
    if (
        not isinstance(qualification_rows, list)
        or not isinstance(contact_rows, list)
        or len(qualification_rows) != FORMAL_EPISODE_COUNT
        or len(contact_rows) != FORMAL_EPISODE_COUNT
    ):
        raise ValueError("V3_SIM_DATA_FORMAL20_QUALIFICATION_OR_CONTACT_MISSING")
    force = values["replica_fingertip_object_pair_force_world"].astype(np.float64)
    expected = values["replica_reference_contact_mask"].astype(bool)
    actual = values["replica_actual_contact_mask"].astype(bool)
    valid = values["replica_fingertip_object_pair_force_valid"].astype(bool)
    magnitude = np.linalg.norm(force, axis=-1)
    s_contact = (magnitude * expected).sum(axis=-1)
    per_episode: list[dict[str, Any]] = []
    contact_metrics: list[dict[str, Any]] = []
    qualified: list[int] = []
    failed: list[int] = []
    for episode in range(FORMAL_EPISODE_COUNT):
        q = qualification_rows[episode]
        c = contact_rows[episode]
        suite = suite_rows[episode]
        if int(q["replica"]) != episode or int(c["replica"]) != episode:
            raise ValueError("V3_SIM_DATA_REPLICA_ORDER_INVALID")
        is_qualified = _as_bool(suite["qualified_success"])
        (qualified if is_qualified else failed).append(episode)
        per_episode.append(
            {
                "episode": episode,
                "seed": int(q["seed"]),
                "kinematic_success": _as_bool(suite["kinematic_success"]),
                "physics_success": _as_bool(suite["physics_success"]),
                "qualified_success": is_qualified,
                "physics_qualified": is_qualified,
                **{key: float(suite[key]) for key in suite if key.startswith("E_")},
                "terminal_delta_v_mps": float(q["terminal_delta_v_mps"]),
                "terminal_delta_omega_radps": float(q["terminal_delta_omega_radps"]),
                "max_inter_finger_penetration_m": float(q["max_inter_finger_penetration_m"]),
                "expected_contact_recall": c["expected_contact_recall"],
                "persistent_contact_recall": c["persistent_contact_recall"],
                "unexpected_contact_rate": c["unexpected_contact_rate"],
                "actual_contact_fraction": c["actual_contact_fraction"],
                "longest_contact_loss_gap": c["longest_contact_loss_gap"],
                "contact_loss_event_count": c["contact_loss_event_count"],
                "recontact_event_count": c["recontact_event_count"],
                "terminal_contact": c["terminal_contact"],
                "contact_force_mean_n": c["contact_force"]["mean"],
                "contact_force_p95_n": c["contact_force"]["p95"],
                "contact_force_max_n": c["contact_force"]["max"],
                "total_contact_impulse_ns": c["total_contact_impulse_ns"],
            }
        )
        for frame in range(FORMAL_FRAME_COUNT):
            for finger in range(5):
                contact_metrics.append(
                    {
                        "episode": episode,
                        "seed": int(q["seed"]),
                        "frame": frame,
                        "finger_index": finger,
                        "reference_expected_contact": bool(expected[frame, episode, finger]),
                        "actual_contact": bool(actual[frame, episode, finger]),
                        "pair_force_valid": bool(valid[frame, episode]),
                        "pair_force_x_n": float(force[frame, episode, finger, 0]),
                        "pair_force_y_n": float(force[frame, episode, finger, 1]),
                        "pair_force_z_n": float(force[frame, episode, finger, 2]),
                        "pair_force_norm_n": float(magnitude[frame, episode, finger]),
                        "S_contact_n": float(s_contact[frame, episode]),
                        "r_contact": float(values["replica_contact_reward"][frame, episode]),
                        "contact_impulse_ns": float(
                            (s_contact[: frame + 1, episode] * CONTROL_DT_S).sum()
                        ),
                    }
                )
    return per_episode, contact_metrics, qualified, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--evaluation-suite", type=Path, required=True)
    parser.add_argument("--per-episode", type=Path, required=True)
    parser.add_argument("--contact-summary", type=Path, required=True)
    parser.add_argument("--reward-v3-contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    qualification_path = args.qualification.resolve()
    suite_path = args.evaluation_suite.resolve()
    per_episode_path = args.per_episode.resolve()
    contact_path = args.contact_summary.resolve()
    contract_path = args.reward_v3_contract.resolve()
    reference_path = args.reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"V3_SIM_DATA_OUTPUT_ALREADY_EXISTS:{output}")
    values, trace_metadata = _load_trace(trace_path)
    qualification = _read_json(qualification_path)
    contact = _read_json(contact_path)
    suite = _read_json(suite_path)
    contract = _read_json(contract_path)
    if (
        qualification.get("status") != "STAGE16D_REWARD_V3_FORMAL_COMPLETE"
        or contact.get("status") != "REFERENCE_CONTACT_EVALUATION_COMPLETE"
        or suite.get("schema_version") != "TopoRetargetEvaluationSuiteV2ResultV1"
        or contract.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN"
    ):
        raise ValueError("V3_SIM_DATA_REQUIRES_COMPLETED_FROZEN_FORMAL_INPUTS")
    if qualification.get("trace_sha256") != _sha256(trace_path) or contact.get(
        "trace_sha256"
    ) != _sha256(trace_path):
        raise ValueError("V3_SIM_DATA_TRACE_PROVENANCE_MISMATCH")
    if trace_metadata["clip"] != qualification.get("clip"):
        raise ValueError("V3_SIM_DATA_CLIP_MISMATCH")
    if not reference_path.is_file():
        raise FileNotFoundError(f"V3_SIM_DATA_REFERENCE_MISSING:{reference_path}")
    suite_rows = _read_suite_rows(per_episode_path)
    per_episode, contact_metrics, qualified, failed = _episode_rows(
        values=values,
        qualification=qualification,
        contact=contact,
        suite_rows=suite_rows,
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        **trace_metadata,
        "reward_contract_sha256": _sha256(contract_path),
        "lambda_c_n": float(contract["frozen_parameters"]["lambda_c_n"]),
        "physics_contract_sha256": str(qualification["physics_contract_sha256"]),
        "causal_physics": True,
        "external_guidance": False,
    }
    _write_zarr(output / "rollouts.zarr", values=values, metadata=metadata)
    _write_parquet(output / "per_episode_metrics.parquet", per_episode)
    _write_parquet(output / "contact_metrics.parquet", contact_metrics)
    shutil.copyfile(reference_path, output / "reference.npz")
    manifest = {
        "schema_version": "Stage16DRewardV3FormalSimulationManifestV1",
        "status": "STAGE16D_REWARD_V3_FORMAL_SIM_DATA_EXPORTED",
        "clip": trace_metadata["clip"],
        "episode_count": FORMAL_EPISODE_COUNT,
        "frame_count": FORMAL_FRAME_COUNT,
        "all_episode_indices": list(range(FORMAL_EPISODE_COUNT)),
        "qualified_episode_indices": qualified,
        "failed_episode_indices": failed,
        "metadata": metadata,
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "qualification": {
                "path": str(qualification_path),
                "sha256": _sha256(qualification_path),
            },
            "evaluation_suite": {"path": str(suite_path), "sha256": _sha256(suite_path)},
            "per_episode": {"path": str(per_episode_path), "sha256": _sha256(per_episode_path)},
            "contact_summary": {"path": str(contact_path), "sha256": _sha256(contact_path)},
            "reward_v3_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
        },
        "files": {
            name: _sha256(output / name)
            for name in ("per_episode_metrics.parquet", "contact_metrics.parquet", "reference.npz")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evaluation_summary = {
        "schema_version": "Stage16DRewardV3FormalEvaluationSummaryV1",
        "clip": trace_metadata["clip"],
        "evaluation_suite": suite,
        "contact": contact["aggregate"],
        "qualification": {
            key: qualification[key]
            for key in (
                "status",
                "reference_completion_rate",
                "terminal_contact_rate",
                "terminal_stability_rate",
                "ppo_task_success_rate",
                "physics_qualified",
                "geometry_absolute_pass",
                "twist_residuals",
            )
        },
        "all_episode_indices": manifest["all_episode_indices"],
        "qualified_episode_indices": qualified,
        "failed_episode_indices": failed,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output),
                "qualified_episode_count": len(qualified),
                "failed_episode_count": len(failed),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
