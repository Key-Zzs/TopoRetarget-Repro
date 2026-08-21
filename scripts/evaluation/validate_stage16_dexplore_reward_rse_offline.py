#!/usr/bin/env python3
"""Offline hard gate for Stage16 grouped multiplicative reward and RSE V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toporetarget.rl.dynamic_physical_qualification import phase_labels_from_reference_index
from toporetarget.rl.reference_tracking.grouped_multiplicative_reward import (
    GroupedMultiplicativeRewardV1,
    grouped_multiplicative_reward_v1_terms,
    parse_obj_triangles,
    point_to_triangle_surface_distance,
    reference_scope_weight,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV4,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (
    EVALUATION_FINGERTIP_LINKS,
)
from toporetarget.rl.reference_tracking.reference_scoped_exploration import (
    AdaptiveScopeStateV1,
    ReferenceScopedExplorationV1,
    adaptive_kappa,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse"
POSITIVE_ROOT = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
NEGATIVE_ROOT = REPO_ROOT / ".local/sim_data/stage16_fixed_wrist_causal_physical_c4/v4/hocap_170105"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DISTANCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_contact"
MESH_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
V4_CONTRACT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json"
MJCF = REPO_ROOT / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
GROUP_NAMES = ("R_obj", "R_hand", "R_int", "R_reg", "total")
LIFT_FRAME = 184
EARLY_LIFT_END = 225
CONTACT_PHASE_START = 92
LIFT_PHASE_END = 230


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"OFFLINE_CSV_EMPTY:{path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion_wxyz, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(values, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*values.shape[:-1], 3, 3)


def _joint_bounds(expected_order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(model.njnt)
    )
    if names != expected_order:
        raise RuntimeError("OFFLINE_JOINT_ORDER_MISMATCH")
    return model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()


def _source_line(function: object) -> str:
    source, line = inspect.getsourcelines(function)
    del source
    path = Path(inspect.getsourcefile(function) or "").resolve().relative_to(REPO_ROOT)
    return f"{path}:{line}"


def _baseline_contract(output: Path, profile: TopoRetargetReferenceTrackingReward26DV4) -> None:
    from toporetarget.rl.environments.isaaclab_backend.reward_terms import world_wrist_reward_terms
    from toporetarget.rl.reference_tracking.ppo26d_reward import (
        ppo26d_reward_v2_object_twist_terms,
        ppo26d_reward_v4_strict_per_finger_contact_terms,
    )
    from toporetarget.rl.reference_tracking.strict_per_finger_contact import (
        strict_per_finger_contact_reward,
    )

    components = [
        {
            "component": "object_axis_point_tracking",
            "mathematical_definition": "exp(-(mean_j ||axis_j-axis_ref_j|| / 0.04 m)^2)",
            "range": "(0,1]",
            "weight": profile.object_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "tracked_link_tracking",
            "mathematical_definition": "mean_j exp(-(||link_j-link_ref_j|| / 0.025 m)^2)",
            "range": "(0,1]",
            "weight": profile.link_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "finger_joint_tracking",
            "mathematical_definition": "mean_j exp(-(((q-q_ref)/(q_hi-q_lo))/0.10)^2)",
            "range": "(0,1]",
            "weight": profile.finger_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "wrist_position_tracking",
            "mathematical_definition": "exp(-(||p-p_ref|| / 0.02 m)^2)",
            "range": "(0,1]",
            "weight": profile.wrist_position_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "wrist_rotation_tracking",
            "mathematical_definition": "exp(-(geodesic(q,q_ref) / 0.174532925 rad)^2)",
            "range": "(0,1]",
            "weight": profile.wrist_rotation_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "action_smoothness_26d",
            "mathematical_definition": "sum ||a_t-a_t-1||^2 + ||a_t-2a_t-1+a_t-2||^2",
            "range": "[0,inf) cost",
            "weight": profile.smoothness_weight,
            "already_exponential": False,
            "bounded_0_1": False,
            "source": _source_line(world_wrist_reward_terms),
        },
        {
            "component": "object_linear_velocity_tracking",
            "mathematical_definition": "exp(-(||v-v_ref|| / 0.075 m/s)^2)",
            "range": "(0,1]",
            "weight": profile.object_velocity_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(ppo26d_reward_v2_object_twist_terms),
        },
        {
            "component": "object_angular_velocity_tracking_authority_v2",
            "mathematical_definition": "exp(-(||omega_world-omega_world_ref_v2|| / 0.125 rad/s)^2)",
            "range": "(0,1]",
            "weight": profile.object_angular_velocity_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(ppo26d_reward_v2_object_twist_terms),
        },
        {
            "component": "strict_per_finger_contact_v4",
            "mathematical_definition": (
                "mean_required exp(-lambda_tip/(named_pair_force+epsilon)); absent pair=0"
            ),
            "range": "[0,1]",
            "weight": profile.contact_weight,
            "already_exponential": True,
            "bounded_0_1": True,
            "source": _source_line(strict_per_finger_contact_reward),
        },
    ]
    _write_json(
        output / "contracts/baseline_reward_contract.json",
        {
            "schema_version": "Stage16BaselineRewardContractAuditV1",
            "branch_point": "6b3851fd66b95e3f5ca76638b8bf3d04d019f789",
            "legacy_formula_source": _source_line(ppo26d_reward_v4_strict_per_finger_contact_terms),
            "legacy_formula": "RewardV2 additive total + strict_per_finger_contact_v4",
            "components": components,
        },
    )


def _load_authority(clip: str) -> dict[str, Any]:
    reference_path = REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz"
    distance_path = DISTANCE_ROOT / f"reference_contact_mask_{clip}.npz"
    mesh_path = MESH_ROOT / f"{clip}.obj"
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {
            name: np.asarray(archive[name]) for name in archive.files if name != "metadata"
        }
        metadata = json.loads(str(archive["metadata"].item()))
    with np.load(distance_path, allow_pickle=False) as archive:
        distance = np.asarray(archive["reference_fingertip_to_object_distance_m"], dtype=np.float32)
    triangles = parse_obj_triangles(mesh_path)
    joint_order = tuple(str(value) for value in metadata["joint_order"])
    lower, upper = _joint_bounds(joint_order)
    link_names = tuple(str(value) for value in metadata["tracked_link_names"])
    tip_indices = tuple(link_names.index(name) for name in EVALUATION_FINGERTIP_LINKS)
    pose = np.concatenate(
        (
            reference["object_pose_translation_world_ref"],
            reference["object_pose_quaternion_world_ref_wxyz"],
        ),
        axis=-1,
    )
    rotation = _quaternion_matrix(pose[:, 3:7])
    local_axes = np.einsum(
        "tji,tkj->tki", rotation, reference["object_axis_points_world_ref"] - pose[:, None, :3]
    )
    return {
        "reference": reference,
        "reference_distance": distance,
        "triangles": triangles,
        "joint_lower": lower.astype(np.float32),
        "joint_upper": upper.astype(np.float32),
        "tip_indices": tip_indices,
        # Match WorldWristReferenceBank._local_axis_points exactly: the runtime
        # freezes frame-0 object-local axes and compares them with the retimed
        # reference-axis series, whose interpolation has small expected drift.
        "object_axis_local": local_axes[0].astype(np.float32),
        "paths": {
            "reference": str(reference_path),
            "reference_sha256": _sha256(reference_path),
            "distance": str(distance_path),
            "distance_sha256": _sha256(distance_path),
            "mesh": str(mesh_path),
            "mesh_sha256": _sha256(mesh_path),
        },
    }


def _reward_inputs(
    path: Path, authority: dict[str, Any], profile: TopoRetargetReferenceTrackingReward26DV4
) -> dict[str, torch.Tensor | TopoRetargetReferenceTrackingReward26DV4]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    index = arrays["reference_index"].astype(np.int64)
    reference = authority["reference"]
    object_pose = arrays["object_pose"].astype(np.float32)
    object_rotation = _quaternion_matrix(object_pose[:, 3:7]).astype(np.float32)
    object_axes = object_pose[:, None, :3] + np.einsum(
        "tij,kj->tki", object_rotation, authority["object_axis_local"]
    )
    tips = arrays["tracked_link_positions"][:, authority["tip_indices"]].astype(np.float32)
    tips_local = np.einsum("tji,tkj->tki", object_rotation, tips - object_pose[:, None, :3])
    source_mask = arrays["source_contact_mask"].astype(bool)
    actual_distance = np.full(source_mask.shape, 0.03, dtype=np.float32)
    active = source_mask
    if np.any(active):
        actual_distance[active] = (
            point_to_triangle_surface_distance(
                torch.from_numpy(tips_local[active]), authority["triangles"]
            )
            .numpy()
            .astype(np.float32)
        )
    action = arrays["action"].astype(np.float32)
    previous = np.concatenate((np.zeros_like(action[:1]), action[:-1]), axis=0)
    second_previous = np.concatenate((np.zeros_like(action[:2]), action[:-2]), axis=0)
    return {
        "reference_fingertip_surface_distance_m": torch.from_numpy(
            authority["reference_distance"][index]
        ),
        "actual_fingertip_surface_distance_m": torch.from_numpy(actual_distance),
        "source_contact_mask": torch.from_numpy(source_mask),
        "fingertip_object_pair_force_world": torch.from_numpy(
            arrays["fingertip_object_pair_force_world"].astype(np.float32)
        ),
        "fingertip_object_pair_presence": torch.from_numpy(
            arrays["tip_pair_presence"].astype(bool)
        ),
        "object_twist_world": torch.from_numpy(arrays["object_twist"].astype(np.float32)),
        "object_twist_world_ref": torch.from_numpy(
            arrays["object_twist_reference"].astype(np.float32)
        ),
        "object_axis_points": torch.from_numpy(object_axes),
        "object_axis_points_ref": torch.from_numpy(
            reference["object_axis_points_world_ref"][index].astype(np.float32)
        ),
        "tracked_links": torch.from_numpy(arrays["tracked_link_positions"].astype(np.float32)),
        "tracked_links_ref": torch.from_numpy(
            arrays["embedded_reference_tracked_links"].astype(np.float32)
        ),
        "finger_q": torch.from_numpy(arrays["finger_q"].astype(np.float32)),
        "finger_q_ref": torch.from_numpy(arrays["embedded_reference_finger_q"].astype(np.float32)),
        "joint_lower": torch.from_numpy(authority["joint_lower"]),
        "joint_upper": torch.from_numpy(authority["joint_upper"]),
        "wrist_position": torch.from_numpy(arrays["wrist_pose"][:, :3].astype(np.float32)),
        "wrist_quaternion_wxyz": torch.from_numpy(arrays["wrist_pose"][:, 3:7].astype(np.float32)),
        "wrist_position_ref": torch.from_numpy(
            arrays["embedded_reference_wrist_pose"][:, :3].astype(np.float32)
        ),
        "wrist_quaternion_ref_wxyz": torch.from_numpy(
            arrays["embedded_reference_wrist_pose"][:, 3:7].astype(np.float32)
        ),
        "action": torch.from_numpy(action),
        "previous_action": torch.from_numpy(previous),
        "second_previous_action": torch.from_numpy(second_previous),
        "profile": profile,
        "_reference_index": torch.from_numpy(index),
    }


def _evaluate(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    values = {name: value for name, value in inputs.items() if not name.startswith("_")}
    return grouped_multiplicative_reward_v1_terms(**values)


def _frame_rows(
    clip: str, episode: int, inputs: dict[str, Any], terms: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    index = inputs["_reference_index"].numpy()
    phase = phase_labels_from_reference_index(index)
    return [
        {
            "clip": clip,
            "episode": episode,
            "frame": frame,
            "reference_index": int(index[frame]),
            "phase": str(phase[frame]),
            "R_obj": float(terms["R_obj"][frame]),
            "R_hand": float(terms["R_hand"][frame]),
            "R_int": float(terms["R_int"][frame]),
            "R_reg": float(terms["R_reg"][frame]),
            "R_total": float(terms["total"][frame]),
            "D_ref": float(terms["D_ref"][frame]),
            "w_scope": float(terms["w_scope"][frame]),
            "reference_interaction_active": bool(inputs["source_contact_mask"][frame].any()),
        }
        for frame in range(len(index))
    ]


def _window_mask(rows: list[dict[str, Any]], window: str) -> np.ndarray:
    if window in {"PRE_CONTACT", "CONTACT", "GRASP", "LIFT"}:
        return np.asarray([row["phase"] == window for row in rows], dtype=bool)
    if window == "CONTACT_TO_LIFT":
        return np.asarray(
            [CONTACT_PHASE_START <= row["reference_index"] < LIFT_PHASE_END for row in rows],
            dtype=bool,
        )
    if window == "CONTACT_TO_EARLY_LIFT":
        return np.asarray(
            [CONTACT_PHASE_START <= row["reference_index"] < EARLY_LIFT_END for row in rows],
            dtype=bool,
        )
    if window == "REFERENCE_INTERACTION":
        return np.asarray([row["reference_interaction_active"] for row in rows], dtype=bool)
    if window == "ALL":
        return np.ones(len(rows), dtype=bool)
    raise ValueError(window)


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("OFFLINE_STATISTICS_INVALID")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p5": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "fraction_lt_1e-2": float(np.mean(array < 1.0e-2)),
        "fraction_lt_1e-4": float(np.mean(array < 1.0e-4)),
        "fraction_lt_1e-6": float(np.mean(array < 1.0e-6)),
    }


def _counterfactual_inputs(original: dict[str, Any], name: str) -> dict[str, Any]:
    result = dict(original)
    if name == "CF0_ORIGINAL_ACCEPTED":
        return result
    if name == "CF1_DELAYED_CONTACT":
        delay = 17
        for field, fill in (
            ("fingertip_object_pair_force_world", 0.0),
            ("fingertip_object_pair_presence", False),
            ("actual_fingertip_surface_distance_m", 0.20),
        ):
            source = original[field]
            target = torch.full_like(source, fill)
            target[delay:] = source[:-delay]
            result[field] = target
        return result
    if name == "CF2_MISSING_CONTACT":
        result["fingertip_object_pair_force_world"] = torch.zeros_like(
            original["fingertip_object_pair_force_world"]
        )
        result["fingertip_object_pair_presence"] = torch.zeros_like(
            original["fingertip_object_pair_presence"]
        )
        return result
    if name == "CF3_HAND_TRACKING_DEGRADATION":
        result["tracked_links"] = original["tracked_links"] + 0.05
        result["finger_q"] = original["finger_q"] + 0.50
        result["wrist_position"] = original["wrist_position"] + 0.05
        return result
    if name == "CF4_OBJECT_TRACKING_DEGRADATION":
        result["object_axis_points"] = original["object_axis_points"] + 0.05
        result["object_twist_world"] = original["object_twist_world"] + 0.10
        return result
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".local/runs/stage16_dexplore_reward_rse").mkdir(parents=True, exist_ok=True)

    v4 = json.loads(V4_CONTRACT.read_text(encoding="utf-8"))["frozen_parameters"]
    profile = TopoRetargetReferenceTrackingReward26DV4(
        contact_force_scale_lambda_tip_n=float(v4["lambda_tip_n"])
    )
    grouped = GroupedMultiplicativeRewardV1()
    rse = ReferenceScopedExplorationV1()
    _baseline_contract(output, profile)
    _write_json(
        output / "contracts/grouped_multiplicative_reward_v1.json",
        {
            "schema_version": "Stage16GroupedMultiplicativeRewardContractV1",
            "status": "PREREGISTERED_BEFORE_170105_REWARD_EVALUATION",
            "parameters": grouped.as_dict(),
            "formula": "R_total=exp(log(R_obj)+log(R_hand)+log(R_int)+log(R_reg))",
            "object": (
                "weighted mean normalized squared existing pose/twist errors then exp(-E_obj)"
            ),
            "hand": (
                "weighted mean normalized squared existing link/finger/wrist errors then "
                "exp(-w_scope*E_hand)"
            ),
            "interaction": "0.5*(strict_V4_contact + visual_surface_proximity)",
            "regularization": "exp(-0.01*smoothness_26d)",
            "proximity_scale_authority": (
                "GLOBAL reciprocal of existing frozen 0.03 m tolerance; "
                "NOT_PER_OBJECT; NOT_OUTCOME_TUNED"
            ),
            "phase_hard_gate": False,
        },
    )
    _write_json(
        output / "contracts/rse_v1.json",
        {
            "schema_version": "Stage16ReferenceScopedExplorationContractV1",
            "status": "PREREGISTERED_BEFORE_170105_REWARD_EVALUATION",
            "parameters": rse.as_dict(),
            "kappa": "clip(N_fail/N_total,kappa_min,1)",
            "threshold": "T_g(kappa)=kappa*T_g_base",
            "counter_policy": (
                "N_fail counts only primary RSE deviation terminations; normal completion "
                "increments N_total; technical failures excluded"
            ),
            "uniform_rsi_preserved": True,
            "start_only_binding": False,
        },
    )
    _write_json(
        output / "contracts/bounded_training_contract.json",
        {
            "schema_version": "Stage16DexploreStyleBoundedTrainingContractV1",
            "lineage": "V4/hocap_170105/C4",
            "maximum_updates": 10,
            "evaluation": "Eval10 deterministic frame0 full trajectory after every update",
            "confirm20_trigger": "PF Eval10 == 10/10",
            "positive_control_ppo": False,
            "sweeps": False,
            "offline_gate_required": "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED",
        },
    )
    _write_json(
        output / "git/branch_creation.json",
        {
            "original_branch": "feature/ppo-physical",
            "original_head": "2a679a594ee5618a08d17f1ef3ed577482d7376c",
            "base_commit": "6b3851fd66b95e3f5ca76638b8bf3d04d019f789",
            "new_branch": _git("branch", "--show-current"),
            "branch_head_at_creation": "6b3851fd66b95e3f5ca76638b8bf3d04d019f789",
            "direct_switch": True,
            "new_worktree_created": False,
            "source_profile_failed_commit_in_ancestry": subprocess.call(
                ("git", "merge-base", "--is-ancestor", "76a7f16", "HEAD"), cwd=REPO_ROOT
            )
            == 0,
        },
    )

    authorities = {clip: _load_authority(clip) for clip in ("hocap_170105", "hocap_170650")}
    all_rows: dict[str, list[dict[str, Any]]] = {clip: [] for clip in authorities}
    input_sets: dict[str, list[dict[str, Any]]] = {clip: [] for clip in authorities}
    roots = {"hocap_170105": NEGATIVE_ROOT, "hocap_170650": POSITIVE_ROOT}
    expected_counts = {"hocap_170105": 20, "hocap_170650": 20}
    for clip, root in roots.items():
        paths = sorted(root.glob("episode_*.npz"))
        if len(paths) != expected_counts[clip]:
            raise RuntimeError(f"OFFLINE_TRACE_COUNT_INVALID:{clip}:{len(paths)}")
        for episode, path in enumerate(paths):
            inputs = _reward_inputs(path, authorities[clip], profile)
            terms = _evaluate(inputs)
            input_sets[clip].append(inputs)
            all_rows[clip].extend(_frame_rows(clip, episode, inputs, terms))
        _write_csv(output / f"offline/{clip.removeprefix('hocap_')}_reward.csv", all_rows[clip])

    statistics_rows: list[dict[str, Any]] = []
    collapse_payload: dict[str, Any] = {}
    windows = (
        "ALL",
        "PRE_CONTACT",
        "CONTACT",
        "GRASP",
        "LIFT",
        "CONTACT_TO_EARLY_LIFT",
        "CONTACT_TO_LIFT",
        "REFERENCE_INTERACTION",
    )
    for clip, rows in all_rows.items():
        collapse_payload[clip] = {}
        for window in windows:
            mask = _window_mask(rows, window)
            if not np.any(mask):
                continue
            for metric in GROUP_NAMES:
                field = "R_total" if metric == "total" else metric
                values = np.asarray([row[field] for row in rows], dtype=np.float64)[mask]
                stats = _stats(values)
                statistics_rows.append({"clip": clip, "window": window, "metric": field, **stats})
                if field == "R_total":
                    collapse_payload[clip][window] = stats
    _write_csv(output / "offline/group_statistics.csv", statistics_rows)

    positive_contact_to_lift = collapse_payload["hocap_170650"]["CONTACT_TO_LIFT"]
    collapsed = positive_contact_to_lift["fraction_lt_1e-6"] > 0.80
    _write_json(
        output / "offline/reward_collapse.json",
        {
            "schema_version": "Stage16GroupedRewardCollapseAuditV1",
            "hard_stop": ">80% of accepted 170650 CONTACT_TO_LIFT frames below 1e-6",
            "classification": "MULTIPLICATIVE_REWARD_COLLAPSED" if collapsed else "NOT_COLLAPSED",
            "statistics": collapse_payload,
        },
    )

    counterfactual_names = (
        "CF0_ORIGINAL_ACCEPTED",
        "CF1_DELAYED_CONTACT",
        "CF2_MISSING_CONTACT",
        "CF3_HAND_TRACKING_DEGRADATION",
        "CF4_OBJECT_TRACKING_DEGRADATION",
    )
    counterfactual_rows: list[dict[str, Any]] = []
    for name in counterfactual_names:
        values = {group: [] for group in GROUP_NAMES}
        for inputs in input_sets["hocap_170650"]:
            cf_inputs = _counterfactual_inputs(inputs, name)
            terms = _evaluate(cf_inputs)
            mask = cf_inputs["source_contact_mask"].any(dim=-1)
            for group in GROUP_NAMES:
                values[group].extend(terms[group][mask].tolist())
        counterfactual_rows.append(
            {
                "counterfactual": name,
                **{group: float(np.mean(values[group])) for group in GROUP_NAMES},
            }
        )
    _write_csv(output / "offline/counterfactuals.csv", counterfactual_rows)
    cf = {row["counterfactual"]: row for row in counterfactual_rows}
    cf_ordering = (
        cf["CF0_ORIGINAL_ACCEPTED"]["total"] > cf["CF1_DELAYED_CONTACT"]["total"]
        and cf["CF0_ORIGINAL_ACCEPTED"]["total"] > cf["CF2_MISSING_CONTACT"]["total"]
    )

    lag_rows = [
        row
        for row in all_rows["hocap_170105"]
        if 181 <= row["reference_index"] < 198 and row["reference_interaction_active"]
    ]
    lag_medians = {
        name: float(np.median([row[name] for row in lag_rows]))
        for name in ("R_obj", "R_hand", "R_int", "R_total")
    }
    interaction_load_bearing = (
        bool(lag_rows)
        and lag_medians["R_int"] < min(lag_medians["R_obj"], lag_medians["R_hand"])
        and lag_medians["R_total"] <= lag_medians["R_int"] + 1.0e-7
        and cf_ordering
    )

    distances = torch.linspace(0.0, 0.40, 401)[:, None].expand(-1, 5)
    _, weights = reference_scope_weight(distances, rse.distance_scope_m)
    monotonic = bool(torch.all(weights[1:] >= weights[:-1]))
    kappa_high = float(adaptive_kappa(9, 10, kappa_min=rse.kappa_min))
    kappa_low = float(adaptive_kappa(1, 10, kappa_min=rse.kappa_min))
    rse_valid = (
        monotonic
        and weights[0].item() == 0.0
        and weights[200].item() == 1.0
        and kappa_high > kappa_low
    )
    _write_json(
        output / "offline/rse_sanity.json",
        {
            "schema_version": "Stage16RSESanityV1",
            "distance_scope_m": rse.distance_scope_m,
            "near_weight": float(weights[0]),
            "at_scope_weight": float(weights[200]),
            "far_weight": float(weights[-1]),
            "monotonic_non_decreasing_with_distance": monotonic,
            "high_fail_rate_kappa": kappa_high,
            "low_fail_rate_kappa": kappa_low,
            "initial_state": AdaptiveScopeStateV1().as_dict(),
            "uniform_rsi_preserved": True,
            "passed": rse_valid,
        },
    )

    positive_interaction = [
        row["R_int"]
        for row in all_rows["hocap_170650"]
        if CONTACT_PHASE_START <= row["reference_index"] < LIFT_PHASE_END
    ]
    positive_valid = (
        np.isfinite([row["R_total"] for row in all_rows["hocap_170650"]]).all()
        and float(np.mean(np.asarray(positive_interaction) < 1.0e-6)) <= 0.80
    )
    numerical_valid = all(
        np.isfinite([row[name] for row in rows]).all()
        for rows in all_rows.values()
        for name in ("R_obj", "R_hand", "R_int", "R_reg", "R_total", "D_ref", "w_scope")
    )
    if not numerical_valid:
        classification = "NUMERICALLY_INVALID"
    elif collapsed:
        classification = "MULTIPLICATIVE_REWARD_COLLAPSED"
    elif not interaction_load_bearing:
        classification = "INTERACTION_NOT_LOAD_BEARING"
    elif not rse_valid:
        classification = "RSE_SCOPE_INVALID"
    elif not positive_valid:
        classification = "INCONCLUSIVE"
    else:
        classification = "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED"
    _write_json(
        output / "offline/offline_gate.json",
        {
            "schema_version": "Stage16DexploreStyleOfflineGateV1",
            "classification": classification,
            "passed": classification == "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED",
            "positive_control_valid": bool(positive_valid),
            "numerically_valid": bool(numerical_valid),
            "interaction_load_bearing": bool(interaction_load_bearing),
            "counterfactual_ordering": bool(cf_ordering),
            "rse_scope_valid": bool(rse_valid),
            "reward_collapsed": bool(collapsed),
            "170105_lag_window_medians": lag_medians,
            "authorities": {clip: value["paths"] for clip, value in authorities.items()},
            "ppo_training_run_authorized": classification == "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED",
        },
    )
    print(classification)


if __name__ == "__main__":
    main()
