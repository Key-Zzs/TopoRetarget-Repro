#!/usr/bin/env python3
"""Run the no-training Stage16 frozen-source physical-curriculum sweep.

The driver deliberately evaluates the four historical selected zero-g actors
from the full frame-zero start under every authoritative C0--C4 physics stage.
It never collects PPO batches or calls an optimizer.  Each condition produces
ten deterministic, 321-frame physical traces and a self-contained receipt.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.audit_stage16_zero_g_frozen_actor_contact import (
    _full_start,
    _reference_phase,
    _selection,
    _trace_metrics,
)
from scripts.rl.isaaclab.evaluate_physical_hoi import (
    model_from_checkpoint,
    run_parallel_episodes,
)
from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
from toporetarget.evaluation import (
    PhysicsEpisodeEvidence,
    aggregate_rollouts,
    hand_metric_series,
    object_metric_series,
    trajectory_success,
)
from toporetarget.rl.contact_skill_collapse import command_tracking_metrics
from toporetarget.rl.geometry_audit.convex_query import PythonFCLConvexQueryBackend
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.geometry_audit.metrics import aggregate_penetration
from toporetarget.rl.geometry_audit.runtime_geometry import load_runtime_geometry_manifest
from toporetarget.rl.geometry_audit.transforms import compose_poses, transform_points
from toporetarget.rl.grasp_lift_skill_collapse import grasp_lift_episode_metrics
from toporetarget.rl.gravity_friction_curriculum import load_gravity_friction_curriculum
from toporetarget.rl.physical_evaluation import contact_metrics, flight_metrics, twist_metrics
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

DEFAULT_RUN_ROOT = REPO_ROOT / ".local/runs/stage16_frozen_source_policy_gravity_sweep"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_frozen_source_policy_gravity_sweep"
DEFAULT_TRACE_ROOT = REPO_ROOT / ".local/sim_data/stage16_frozen_source_policy_gravity_sweep"
# These roots are configured once by ``main``. Keeping the default preserves
# the historical sweep command, while closure work writes separate evidence.
RUN_ROOT = DEFAULT_RUN_ROOT
REPORT_ROOT = DEFAULT_REPORT_ROOT
TRACE_ROOT = DEFAULT_TRACE_ROOT
CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
GEOMETRY = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)
FROZEN_GATES = (
    REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/frozen_evaluation_gates.json"
)
STATIC_HOLD = REPORT_ROOT / "targeted_preflight/static_wrist_hold/static_hold_summary.json"
STAGES = ("C0", "C1", "C2", "C3", "C4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_value(value: object) -> object:
    """Convert evaluator metadata to JSON without changing saved trace arrays."""

    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("FROZEN_SOURCE_SWEEP_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cpu_value(value: object) -> object:
    """Make source and CUDA evaluator hashes identical without mutating either."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    return value


def _component_hashes(trainer: Any) -> dict[str, str]:
    import io

    actor = {
        key: value
        for key, value in trainer.model.state_dict().items()
        if key.startswith("actor") or key == "log_std_parameter"
    }

    def digest(value: object) -> str:
        stream = io.BytesIO()
        torch.save(_cpu_value(value), stream)
        return hashlib.sha256(stream.getvalue()).hexdigest()

    return {
        "actor": digest(actor),
        "normalizer": digest(trainer.trainer.normalizer.state_dict()),
    }


def _append_failure(payload: dict[str, object]) -> None:
    path = REPORT_ROOT / "technical_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _configure_output_roots(*, run_root: Path, report_root: Path, trace_root: Path) -> None:
    """Set one explicit evidence namespace before any run-side effects."""

    global RUN_ROOT, REPORT_ROOT, TRACE_ROOT
    RUN_ROOT = run_root.resolve()
    REPORT_ROOT = report_root.resolve()
    TRACE_ROOT = trace_root.resolve()


def _gpu_usage() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        "stderr": completed.stderr.strip(),
    }


def _mode_label(mode: ContactRewardMode) -> str:
    return "v3" if mode is ContactRewardMode.AGGREGATE_V3 else "v4"


def _source_specs() -> tuple[tuple[ContactRewardMode, str], ...]:
    return (
        (ContactRewardMode.AGGREGATE_V3, "hocap_170105"),
        (ContactRewardMode.STRICT_PER_FINGER_V4, "hocap_170105"),
        (ContactRewardMode.AGGREGATE_V3, "hocap_170650"),
        (ContactRewardMode.STRICT_PER_FINGER_V4, "hocap_170650"),
    )


def _selection_path(mode: ContactRewardMode, clip: str) -> Path:
    if mode is ContactRewardMode.AGGREGATE_V3:
        return (
            REPO_ROOT
            / ".local/reports/stage16d_reward_v3_pairforce_unblock"
            / clip
            / "dev/checkpoint_selection.json"
        )
    root = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4" / clip
    return root / (
        "final_checkpoint_selection.json" if clip == "hocap_170650" else "checkpoint_selection.json"
    )


def _source_authority(mode: ContactRewardMode, clip: str) -> dict[str, object]:
    """Resolve by receipt, validate the full checkpoint, and hash frozen components."""

    selection = _selection(mode.value, clip)
    selection_path = _selection_path(mode, clip)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    ranked = selection_payload.get("ranked")
    if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
        raise ValueError(f"SOURCE_ACTOR_MISSING:{mode.value}:{clip}:selection")
    selected = ranked[0]
    checkpoint = Path(str(selection["checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SOURCE_ACTOR_MISSING:{mode.value}:{clip}:{checkpoint}")
    trainer, payload = model_from_checkpoint(checkpoint, "cpu", expected_clip=clip)
    expected_schema = (
        "Stage16DRewardV3CheckpointV1"
        if mode is ContactRewardMode.AGGREGATE_V3
        else "Stage16DStrictPerFingerV4CheckpointV1"
    )
    expected_reward = (
        "TopoRetargetReferenceTrackingReward26DV3"
        if mode is ContactRewardMode.AGGREGATE_V3
        else "TopoRetargetReferenceTrackingReward26DV4"
    )
    reward = payload.get("environment_contract", {}).get("ppo26d", {}).get("reward", {})
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("clip") != clip
        or reward.get("identifier") != expected_reward
        or _sha256(checkpoint) != selection["checkpoint_sha256"]
    ):
        raise ValueError(f"SOURCE_ACTOR_CONTRACT_INVALID:{mode.value}:{clip}")
    component_hashes = _component_hashes(trainer)
    qualification = selected.get("qualification")
    qualification_path = None if qualification is None else Path(str(qualification)).resolve()
    if qualification_path is None or not qualification_path.is_file():
        raise FileNotFoundError(f"SOURCE_ACTOR_QUALIFICATION_MISSING:{mode.value}:{clip}")
    sample_key = (
        "reward_v3_samples" if mode is ContactRewardMode.AGGREGATE_V3 else "reward_v4_samples"
    )
    samples = payload.get(sample_key)
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError(f"SOURCE_ACTOR_SAMPLE_MARKER_INVALID:{mode.value}:{clip}")
    return {
        "id": f"{_mode_label(mode)}_{clip}",
        "reward": _mode_label(mode),
        "contact_mode": mode.value,
        "clip": clip,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "actor_hash": component_hashes["actor"],
        "normalizer_path": f"{checkpoint}::observation_normalization",
        "normalizer_hash": component_hashes["normalizer"],
        "schema_version": payload["schema_version"],
        "reward_identifier": reward["identifier"],
        "reference_hash": payload.get("reference_hash"),
        "source_sample_marker": {sample_key: samples},
        "selection_receipt": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
        },
        "historical_qualification_receipt": {
            "path": str(qualification_path),
            "sha256": _sha256(qualification_path),
        },
        "historical_evaluation_suite": selected.get("evaluation_suite"),
        "historical_source_audit": selected.get("source_audit"),
    }


def _load_sources(source_manifest: Path | None = None) -> dict[str, dict[str, object]]:
    """Load historical sources or one explicitly prepared adapted checkpoint."""

    if source_manifest is not None:
        payload = json.loads(source_manifest.resolve().read_text(encoding="utf-8"))
        declared = payload.get("sources")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("SOURCE_ACTOR_MANIFEST_INVALID")
        sources: dict[str, dict[str, object]] = {}
        required = {
            "id",
            "reward",
            "contact_mode",
            "clip",
            "checkpoint",
            "checkpoint_sha256",
            "actor_hash",
            "normalizer_hash",
        }
        for key, item in declared.items():
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError("SOURCE_ACTOR_MANIFEST_ENTRY_INVALID")
            source = dict(item)
            checkpoint = Path(str(source["checkpoint"])).resolve()
            mode = ContactRewardMode.parse(str(source["contact_mode"]))
            if (
                str(key) != source["id"]
                or source["reward"] != _mode_label(mode)
                or source["clip"] not in {"hocap_170105", "hocap_170650"}
                or not checkpoint.is_file()
                or _sha256(checkpoint) != source["checkpoint_sha256"]
            ):
                raise ValueError("SOURCE_ACTOR_MANIFEST_ENTRY_DRIFT")
            sources[str(key)] = source
        return sources
    sources = {
        f"{_mode_label(mode)}_{clip}": _source_authority(mode, clip)
        for mode, clip in _source_specs()
    }
    if len(sources) != 4 or len({item["checkpoint"] for item in sources.values()}) != 4:
        raise ValueError("SOURCE_ACTOR_SET_INVALID")
    return sources


def _seeds(clip: str, *, count: int) -> list[int]:
    """Use the pre-existing deterministic development seed set, but frame zero only."""

    from scripts.evaluation.audit_stage16_zero_g_frozen_actor_contact import _load_pairs

    pairs = _load_pairs(clip, count)
    return [int(pair["seed"]) for pair in pairs]


def _physics_contract() -> dict[str, object]:
    contract = load_gravity_friction_curriculum(CURRICULUM)
    raw = yaml.safe_load(CURRICULUM.read_text(encoding="utf-8"))
    rows = []
    for stage in STAGES:
        physics = contract.physics(stage)
        rows.append(
            {
                "stage": stage,
                "gravity_scale": float(physics["gravity_scale"]),
                "friction_scale": float(physics["friction_scale"]),
                "gravity_world_mps2": physics["gravity_world_mps2"],
                "table": "finite_inferred_table_proxy_v1",
                "hand_gravity": "disabled",
            }
        )
    return {
        "path": str(CURRICULUM.resolve()),
        "sha256": _sha256(CURRICULUM),
        "scientific_label": "FROZEN_SOURCE_POLICY_GRAVITY_PHYSICS_SWEEP_NOT_PURE_GRAVITY",
        "raw": raw,
        "stages": rows,
    }


def _require_static_wrist_receipt() -> dict[str, object]:
    if not STATIC_HOLD.is_file():
        raise FileNotFoundError(f"FROZEN_SOURCE_SWEEP_STATIC_WRIST_PREFLIGHT_MISSING:{STATIC_HOLD}")
    payload = json.loads(STATIC_HOLD.read_text(encoding="utf-8"))
    mean = float(payload.get("wrist_rotation_error_deg_mean", float("inf")))
    if payload.get("ppo_optimizer_steps") != 0 or mean >= 20.0:
        raise RuntimeError("FROZEN_SOURCE_SWEEP_WRIST_TECHNICAL_REGRESSION")
    return {"path": str(STATIC_HOLD.resolve()), "sha256": _sha256(STATIC_HOLD), "summary": payload}


def _runtime_contract(
    report: Mapping[str, Any], *, source: Mapping[str, object], stage_row: Mapping[str, object]
) -> dict[str, object]:
    ppo = report.get("ppo26d")
    physics = report.get("gravity_friction_curriculum")
    wrist = report.get("finite_virtual_6d_wrist_actuator")
    if (
        not isinstance(ppo, Mapping)
        or not isinstance(physics, Mapping)
        or not isinstance(wrist, Mapping)
    ):
        raise ValueError("FROZEN_SOURCE_SWEEP_RUNTIME_CONTRACT_MISSING")
    expected = {
        "stage": stage_row["stage"],
        "gravity_scale": stage_row["gravity_scale"],
        "friction_scale": stage_row["friction_scale"],
        "support": "finite_inferred_table_proxy_v1",
        "table_actor_active": True,
        "external_guidance": False,
        "fixed_clip": source["clip"],
        "active_clip_ids": [source["clip"]],
        "object_rollout_state_writes": 0,
        "wrist_root_state_writes_during_step": 0,
    }
    observed = {
        "stage": physics.get("stage"),
        "gravity_scale": physics.get("gravity_scale"),
        "friction_scale": physics.get("friction_scale"),
        "support": physics.get("support"),
        "table_actor_active": physics.get("table_actor_active"),
        "external_guidance": physics.get("external_guidance"),
        "fixed_clip": ppo.get("fixed_clip"),
        "active_clip_ids": ppo.get("active_clip_ids"),
        "object_rollout_state_writes": ppo.get("object_rollout_state_writes"),
        "wrist_root_state_writes_during_step": ppo.get("wrist_root_state_writes_during_step"),
    }
    if observed != expected or wrist.get("authority_enabled") is not True:
        raise ValueError(f"FROZEN_SOURCE_SWEEP_RUNTIME_CONTRACT_DRIFT:{observed}")
    return {"environment": report, "expected": expected, "wrist": wrist}


def _mean(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return None if not numeric else float(np.mean(numeric))


def _load_gate(path: Path, *, clip: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("task_gates", {}).get("clips", {}).get(clip)
    if payload.get("status") != "STRICT_V4_EVALUATION_GATES_FROZEN" or not isinstance(gate, dict):
        raise ValueError("FROZEN_SOURCE_SWEEP_GATE_INVALID")
    return gate


def _valid_rows(trace: Mapping[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(trace["fingertip_object_pair_force_valid"], dtype=bool)
    hand_valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    if valid.ndim != 1 or not np.array_equal(valid, hand_valid) or valid[0] or not valid[1:].all():
        raise ValueError("FROZEN_SOURCE_SWEEP_PAIR_FORCE_VALIDITY_INVALID")
    return valid


def _reference_contact(
    trace: Mapping[str, np.ndarray], mode: ContactRewardMode
) -> tuple[np.ndarray, np.ndarray]:
    if mode is ContactRewardMode.AGGREGATE_V3:
        expected = np.asarray(trace["reference_contact_mask"], dtype=bool)
        actual = np.asarray(trace["actual_contact_mask"], dtype=bool)
    else:
        expected = np.asarray(trace["source_contact_mask"], dtype=bool)
        actual = np.asarray(trace["tip_pair_presence"], dtype=bool)
    if expected.shape != actual.shape or expected.shape[1:] != (5,):
        raise ValueError("FROZEN_SOURCE_SWEEP_TIP_CONTACT_TRACE_INVALID")
    return expected, actual


def _inter_finger_penetration(hand_pose: np.ndarray) -> np.ndarray:
    from toporetarget.rl.physics_retargeting.self_collision import (
        InterFingerCapsulePenetrationV1,
        load_self_collision_contract,
    )

    contract = load_self_collision_contract(
        REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml", repo_root=REPO_ROOT
    )
    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        REPO_ROOT / contract.runtime_collision_manifest_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=contract.capsule_radius_scale,
        device="cpu",
    )
    with torch.no_grad():
        values = metric.evaluate(torch.as_tensor(hand_pose, dtype=torch.float32))[
            "maximum_penetration_m"
        ]
    return np.asarray(values.numpy(), dtype=np.float64)


def _reconstruct_hand(trace: Mapping[str, np.ndarray]) -> np.ndarray:
    return reconstruct_hand_collision_body_pose(
        trace["wrist_pose"], trace["finger_q"], repo_root=REPO_ROOT
    ).astype(np.float32)


def _proxy_radius(proxy: Any) -> float:
    """Radius about the proxy-local origin of one authored convex hull."""

    vertices = np.asarray(proxy.scaled_vertices, dtype=np.float64)
    return float(np.linalg.norm(vertices, axis=1).max())


def _world_aabb(vertices: np.ndarray, pose_xyz_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact world-axis-aligned bounds of one convex proxy."""

    world_vertices = transform_points(vertices, pose_xyz_wxyz)
    return world_vertices.min(axis=0), world_vertices.max(axis=0)


def _aabb_separation_lower_bound(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> float:
    """A zero/nonzero lower bound that certifies disjoint AABBs."""

    first_min, first_max = first
    second_min, second_max = second
    gaps = np.maximum(np.maximum(first_min - second_max, second_min - first_max), 0.0)
    return float(np.linalg.norm(gaps))


def _evaluate_geometry_with_exact_broadphase(
    *,
    clip: str,
    object_pose: np.ndarray,
    hand_collision_body_pose: np.ndarray,
    hand_collision_body_names: tuple[str, ...],
    geometry_path: Path = GEOMETRY,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Keep exact FCL for every potentially colliding pair.

    A pair whose *proxy-centre* distance exceeds the sum of two precomputed
    enclosing-sphere radii is mathematically non-penetrating.  The proxy
    centres are the authored local proxy poses transformed into world space;
    they are tighter than articulation-root spheres but retain the same strict
    guarantee. Such pairs are certified as zero penetration before FCL; every
    remaining pair is queried by the frozen python-fcl backend. A second exact
    world-AABB separation test certifies additional non-overlapping proxy pairs
    before FCL. This is a conservative query reduction, not a change to
    penetration thresholds or the physical metric.
    """

    objects = np.asarray(object_pose, dtype=np.float64)
    hands = np.asarray(hand_collision_body_pose, dtype=np.float64)
    hand_proxies, objects_by_clip = load_runtime_geometry_manifest(geometry_path)
    object_proxies = objects_by_clip.get(clip)
    if object_proxies is None or hand_collision_body_names != tuple(
        item.body_name for item in hand_proxies
    ):
        raise ValueError("FROZEN_SOURCE_SWEEP_GEOMETRY_MANIFEST_DRIFT")
    if objects.ndim != 3 or hands.shape != (*objects.shape[:2], len(hand_proxies), 7):
        raise ValueError("FROZEN_SOURCE_SWEEP_GEOMETRY_SHAPE_INVALID")
    hand_radius = np.asarray([_proxy_radius(item) for item in hand_proxies])
    object_radius = np.asarray([_proxy_radius(item) for item in object_proxies])
    hand_vertices = [np.asarray(item.scaled_vertices, dtype=np.float64) for item in hand_proxies]
    object_vertices = [
        np.asarray(item.scaled_vertices, dtype=np.float64) for item in object_proxies
    ]
    pair_ids = tuple(
        f"{hand.shape_id}<->{obj.shape_id}" for hand in hand_proxies for obj in object_proxies
    )
    frames, replicas = objects.shape[:2]
    signed = np.empty((frames, replicas, len(pair_ids)), dtype=np.float64)
    penetration = np.zeros_like(signed)
    direction = np.zeros((*signed.shape, 3), dtype=np.float64)
    backend: PythonFCLConvexQueryBackend | None = None
    hand_shapes: list[Any] | None = None
    object_shapes: list[Any] | None = None
    exact_query_count = 0
    certified_pair_count = 0
    aabb_certified_pair_count = 0
    for frame in range(frames):
        for replica in range(replicas):
            hand_world_poses = [
                compose_poses(hands[frame, replica, hand_index], proxy.local_pose_xyz_wxyz)
                for hand_index, proxy in enumerate(hand_proxies)
            ]
            object_world_poses = [
                compose_poses(objects[frame, replica], proxy.local_pose_xyz_wxyz)
                for proxy in object_proxies
            ]
            hand_aabbs = [
                _world_aabb(vertices, pose)
                for vertices, pose in zip(hand_vertices, hand_world_poses, strict=True)
            ]
            object_aabbs = [
                _world_aabb(vertices, pose)
                for vertices, pose in zip(object_vertices, object_world_poses, strict=True)
            ]
            pair_index = 0
            for hand_index, _hand_proxy in enumerate(hand_proxies):
                hand_world = hand_world_poses[hand_index]
                center_hand = hand_world[:3]
                for object_index, _object_proxy in enumerate(object_proxies):
                    object_world = object_world_poses[object_index]
                    center_object = object_world[:3]
                    delta = center_object - center_hand
                    distance = float(np.linalg.norm(delta))
                    lower_bound = distance - hand_radius[hand_index] - object_radius[object_index]
                    if lower_bound > 0.0:
                        signed[frame, replica, pair_index] = lower_bound
                        if distance > 0.0:
                            direction[frame, replica, pair_index] = delta / distance
                        certified_pair_count += 1
                    elif (
                        aabb_lower_bound := _aabb_separation_lower_bound(
                            hand_aabbs[hand_index], object_aabbs[object_index]
                        )
                    ) > 0.0:
                        signed[frame, replica, pair_index] = aabb_lower_bound
                        if distance > 0.0:
                            direction[frame, replica, pair_index] = delta / distance
                        aabb_certified_pair_count += 1
                    else:
                        if backend is None:
                            backend = PythonFCLConvexQueryBackend()
                            hand_shapes = [backend.proxy_shape(item) for item in hand_proxies]
                            object_shapes = [backend.proxy_shape(item) for item in object_proxies]
                        assert hand_shapes is not None and object_shapes is not None
                        result = backend.query(
                            hand_shapes[hand_index],
                            hand_world,
                            object_shapes[object_index],
                            object_world,
                            collision_mtd_only=True,
                        )
                        if not result.converged:
                            raise RuntimeError("FROZEN_SOURCE_SWEEP_FCL_NONCONVERGENCE")
                        signed[frame, replica, pair_index] = result.signed_separation_m
                        penetration[frame, replica, pair_index] = result.penetration_depth_m
                        direction[frame, replica, pair_index] = (
                            result.depenetration_direction_for_second
                        )
                        exact_query_count += 1
                    pair_index += 1
    worst_pair = np.argmax(penetration, axis=2)
    worst = np.take_along_axis(penetration, worst_pair[..., None], axis=2)[..., 0]
    aggregate = aggregate_penetration(worst, worst_pair, pair_ids)
    aggregate.update(
        {
            "schema_version": "RuntimeCollisionProxyPenetrationResultV1",
            "clip": clip,
            "complete_frames": frames,
            "pair_ids": list(pair_ids),
            "all_queries_converged": True,
            "query_execution": "EXACT_FCL_OR_CONSERVATIVE_SPHERE_SEPARATION_CERTIFICATE",
            "exact_fcl_query_count": exact_query_count,
            "sphere_separation_certified_pair_count": certified_pair_count,
            "aabb_separation_certified_pair_count": aabb_certified_pair_count,
        }
    )
    return aggregate, {
        "signed_separation_m": signed,
        "penetration_depth_m": penetration,
        "depenetration_direction_for_object": direction,
        "frame_worst_penetration_m": worst,
        "frame_worst_pair_index": worst_pair,
        "pair_ids": np.asarray(pair_ids),
    }


def _v4_safe_grasp_metrics(
    trace: dict[str, np.ndarray], mode: ContactRewardMode
) -> dict[str, object]:
    local = dict(trace)
    if mode is ContactRewardMode.STRICT_PER_FINGER_V4:
        local["contact_reward"] = np.asarray(trace["r_contact_v4"], dtype=np.float64)
    return grasp_lift_episode_metrics(local)


def _terminal_evidence(
    *,
    trace: dict[str, np.ndarray],
    rollout: Mapping[str, object],
    gate: Mapping[str, object],
    geometry: Mapping[str, object],
    inter_finger: np.ndarray,
    environment: Mapping[str, Any],
) -> tuple[PhysicsEpisodeEvidence, dict[str, object]]:
    valid = _valid_rows(trace)
    hand = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    indices = np.flatnonzero(valid)[
        -min(int(gate["terminal_window_control_steps"]), int(valid.sum())) :
    ]
    twist = np.asarray(trace["object_twist"], dtype=np.float64)
    linear = np.linalg.norm(twist[indices, :3], axis=-1)
    angular = np.linalg.norm(twist[indices, 3:], axis=-1)
    hand_terminal = hand[indices]
    linear_limit = np.where(
        hand_terminal,
        float(gate["terminal_linear_speed_mps"]),
        float(gate["terminal_free_object_linear_speed_mps"]),
    )
    angular_limit = np.where(
        hand_terminal,
        float(gate["terminal_angular_speed_radps"]),
        float(gate["terminal_free_object_angular_speed_radps"]),
    )
    terminal_contact = bool(rollout.get("terminal_contact", False))
    terminal_stability = bool(
        terminal_contact and np.all(linear <= linear_limit) and np.all(angular <= angular_limit)
    )
    action = np.asarray(trace["action"], dtype=np.float64)
    finite = all(
        np.isfinite(np.asarray(trace[name])).all()
        for name in ("object_pose", "object_twist", "wrist_pose", "finger_q", "action")
    )
    absolute_geometry = bool(
        float(geometry["max_penetration_m"]) < float(gate["catastrophic_penetration_m"])
        and float(geometry["p95_penetration_m"]) <= float(gate["p95_penetration_m"])
    )
    inter_pass = bool(
        inter_finger.max(initial=0.0) <= float(gate["maximum_inter_finger_penetration_m"])
    )
    ppo = environment["ppo26d"]
    contact_causality = bool(
        np.any(hand[1:] & (np.linalg.norm(np.diff(twist, axis=0), axis=-1) > 1.0e-7))
    )
    evidence = PhysicsEpisodeEvidence(
        terminal_contact_pass=terminal_contact,
        terminal_stability_pass=terminal_stability,
        contact_causality_pass=contact_causality,
        inter_finger_penetration_pass=inter_pass,
        absolute_hand_object_penetration_pass=absolute_geometry,
        action_bounds_pass=bool(np.all(np.abs(action) <= float(gate["action_limit"]))),
        no_hidden_force=not bool(ppo["hidden_force_or_attachment"]),
        no_object_rollout_state_write=int(ppo["object_rollout_state_writes"]) == 0,
        no_wrist_root_teleport=int(ppo["wrist_root_state_writes_during_step"]) == 0,
    )
    return evidence, {
        "finite": finite,
        "terminal_stability": terminal_stability,
        "terminal_contact": terminal_contact,
        "absolute_geometry_pass": absolute_geometry,
        "interfinger_pass": inter_pass,
        "contact_causality": contact_causality,
    }


def _condition_paths(source: Mapping[str, object], stage: str) -> tuple[Path, Path]:
    mode = str(source["reward"])
    clip = str(source["clip"])
    return (
        REPORT_ROOT / "sweep" / mode / clip / stage.lower(),
        TRACE_ROOT / mode / clip / stage.lower(),
    )


def _slice_parallel_trace(
    trace: Mapping[str, np.ndarray],
    *,
    replica: int,
    replicas: int,
    clip: str,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    """Select one replica through its authoritative terminal row only.

    Isaac Lab resets vectorized environments immediately after a terminal
    condition. The raw all-replica buffer can therefore contain rows from a
    subsequent reset for a replica that ended early. Those rows are neither
    part of the physical episode nor admissible terminal evidence.
    """

    indices = np.asarray(trace["reference_index"])
    if indices.ndim < 2 or indices.shape[1] != replicas:
        raise RuntimeError("FROZEN_SOURCE_SWEEP_PARALLEL_REFERENCE_INDEX_INVALID")
    captured_frames = int(indices.shape[0])
    if expected_frames <= 0 or expected_frames > captured_frames:
        raise RuntimeError("FROZEN_SOURCE_SWEEP_CAPTURED_TRACE_SHORTER_THAN_EXECUTION")

    result: dict[str, np.ndarray] = {}
    for name, value in trace.items():
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[0] == captured_frames and array.shape[1] == replicas:
            result[name] = array[:expected_frames, replica].copy()
        elif array.ndim >= 1 and array.shape[0] == captured_frames:
            result[name] = array[:expected_frames].copy()
        else:
            result[name] = array.copy()
    if "phase_code" in result:
        names = np.asarray(
            ("PRE_CONTACT", "APPROACH", "CONTACT", "GRASP", "LIFT", "MANIPULATION", "TERMINAL")
        )
        result["phase"] = names[np.asarray(result["phase_code"], dtype=np.int64)]
    else:
        result["phase"] = _reference_phase(
            clip, np.asarray(result["reference_index"], dtype=np.int64)
        )
    return result


def _completion_layers(
    *, trace: Mapping[str, np.ndarray], rollout: Mapping[str, object], start: int
) -> dict[str, object]:
    """Report simulation, trace, and qualification preconditions separately."""

    reference_index = np.asarray(trace["reference_index"], dtype=np.int64)
    expected_rows = int(rollout["steps"]) + 1
    expected_index = np.arange(start, start + expected_rows, dtype=np.int64)
    rows_match_execution = bool(
        len(reference_index) == expected_rows and np.array_equal(reference_index, expected_index)
    )
    reached_terminal_reference = bool(
        rollout["reached_reference_end"]
        and expected_rows == 321
        and len(reference_index) == 321
        and int(reference_index[-1]) == 320
    )
    finite_fields = ("object_pose", "object_twist", "wrist_pose", "finger_q", "action")
    required_fields_present = all(name in trace for name in finite_fields)
    finite = bool(
        required_fields_present
        and all(np.isfinite(np.asarray(trace[name])).all() for name in finite_fields)
    )
    phase = set(np.asarray(trace.get("phase", ())).tolist())
    terminal_semantic = bool(reached_terminal_reference and "TERMINAL" in phase)
    simulation_completed = bool(reached_terminal_reference and rows_match_execution and finite)
    trace_completed = bool(simulation_completed and required_fields_present)
    return {
        "SIMULATION_COMPLETED": simulation_completed,
        "TRACE_COMPLETED": trace_completed,
        "QUALIFICATION_COMPLETED": False,
        "valid_physics_frames": int(len(reference_index)),
        "expected_physics_frames": 321,
        "reference_index_matches_execution": rows_match_execution,
        "reference_reaches_terminal": reached_terminal_reference,
        "terminal_semantic_recorded": terminal_semantic,
        "finite_state": finite,
        "required_trace_fields_present": required_fields_present,
        "termination_reason": int(rollout["termination_reason"]),
    }


def _parallel_rollouts(
    *, env: Any, trainer: Any, clip: str, seeds: list[int], start: int
) -> list[tuple[dict[str, object], dict[str, np.ndarray]]]:
    env.cfg.evaluation_reset_reference_indices = (start,) * len(seeds)
    results = run_parallel_episodes(
        env,
        trainer,
        capture=True,
        capture_all_replicas=True,
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
        expected_clip=clip,
        seeds=seeds,
    )
    all_trace = results[0].pop("all_replica_trace", None)
    if not isinstance(all_trace, dict) or len(results) != len(seeds):
        raise RuntimeError("FROZEN_SOURCE_SWEEP_PARALLEL_TRACE_MISSING")
    output: list[tuple[dict[str, object], dict[str, np.ndarray]]] = []
    for replica, (seed, result) in enumerate(zip(seeds, results, strict=True)):
        if result.get("seed") != seed or result.get("start_reference_index") != start:
            raise RuntimeError("FROZEN_SOURCE_SWEEP_PARALLEL_SEED_OR_RESET_DRIFT")
        rollout = dict(result)
        rollout["reached_reference_end"] = bool(rollout.pop("reached_final_reference"))
        rollout["rollout_state_writes"] = env.rollout_state_write_report()
        output.append(
            (
                rollout,
                _slice_parallel_trace(
                    all_trace,
                    replica=replica,
                    replicas=len(seeds),
                    clip=clip,
                    expected_frames=int(rollout["steps"]) + 1,
                ),
            )
        )
    return output


def _run_condition(
    *,
    source: Mapping[str, object],
    mode: ContactRewardMode,
    stage_row: Mapping[str, object],
    seeds: list[int],
    gate: Mapping[str, object],
) -> dict[str, object]:
    stage = str(stage_row["stage"])
    report_dir, trace_dir = _condition_paths(source, stage)
    checkpoint = Path(str(source["checkpoint"])).resolve()
    start = _full_start(str(source["clip"]))
    started = time.monotonic()
    env = _make_table_env(
        clip=str(source["clip"]),
        num_envs=len(seeds),
        start_index=start,
        mode=mode,
        stage=stage,
    )
    try:
        runtime = _runtime_contract(env.contract_report(), source=source, stage_row=stage_row)
        trainer, _ = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip=str(source["clip"])
        )
        before = _component_hashes(trainer)
        if (
            before["actor"] != source["actor_hash"]
            or before["normalizer"] != source["normalizer_hash"]
        ):
            raise RuntimeError(f"FROZEN_SOURCE_HASH_RESTORE_DRIFT:{source['id']}:{stage}")
        rows: list[dict[str, object]] = []
        detailed: list[dict[str, object]] = []
        per_finger_rows: list[dict[str, object]] = []
        rollouts = _parallel_rollouts(
            env=env,
            trainer=trainer,
            clip=str(source["clip"]),
            seeds=seeds,
            start=start,
        )
        for episode, (rollout, trace) in enumerate(rollouts):
            seed = seeds[episode]
            completion = _completion_layers(trace=trace, rollout=rollout, start=start)
            captured_path = trace_dir / "captured_pre_geometry" / f"episode_{episode:02d}.npz"
            captured_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                captured_path,
                **trace,
                frozen_source_actor_sha256=np.asarray(str(source["checkpoint_sha256"])),
                frozen_source_actor_id=np.asarray(str(source["id"])),
                curriculum_stage=np.asarray(stage),
            )
            _write_json(
                report_dir / "captured_pre_geometry" / f"episode_{episode:02d}.json",
                {
                    "episode": episode,
                    "seed": seed,
                    "rollout": rollout,
                    "completion": completion,
                    "trace": {
                        "path": str(captured_path.resolve()),
                        "sha256": _sha256(captured_path),
                    },
                },
            )
            phase = set(np.asarray(trace["phase"]).tolist())
            required_phases = {
                "PRE_CONTACT",
                "APPROACH",
                "CONTACT",
                "GRASP",
                "LIFT",
                "MANIPULATION",
                "TERMINAL",
            }
            # An early physical termination is an evaluated outcome, not a
            # runner crash.  Preserve its trimmed trace and finish all ten
            # replicas, but never manufacture the missing terminal phase.
            # A missing non-terminal reference phase remains a trace defect.
            missing_phases = required_phases.difference(phase)
            if missing_phases.difference({"TERMINAL"}):
                raise RuntimeError(f"FROZEN_SOURCE_SWEEP_REFERENCE_PHASE_MISSING:{phase}")
            frozen_contact = _trace_metrics(trace, mode=mode.value)
            grasp = _v4_safe_grasp_metrics(trace, mode)
            command = command_tracking_metrics(trace)
            valid = _valid_rows(trace)
            expected, actual = _reference_contact(trace, mode)
            trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
            trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
            geometry, raw_geometry = _evaluate_geometry_with_exact_broadphase(
                clip=str(source["clip"]),
                object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
                hand_collision_body_pose=np.asarray(
                    trace["hand_collision_body_pose"], dtype=np.float64
                )[:, None],
                hand_collision_body_names=tuple(
                    str(value) for value in trace["hand_collision_body_names"]
                ),
            )
            inter = _inter_finger_penetration(trace["hand_collision_body_pose"])
            evidence, diagnostic = _terminal_evidence(
                trace=trace,
                rollout=rollout,
                gate=gate,
                geometry=geometry,
                inter_finger=inter,
                environment=runtime["environment"],
            )
            series = object_metric_series(
                np.asarray(trace["object_pose"], dtype=np.float64),
                np.asarray(trace["object_reference"], dtype=np.float64),
            )
            series.update(
                hand_metric_series(
                    np.asarray(trace["hand_collision_body_pose"], dtype=np.float64),
                    [str(value) for value in trace["hand_collision_body_names"]],
                    np.asarray(trace["tracked_link_reference"], dtype=np.float64),
                    [str(value) for value in env.reference_bank.tracked_link_names],
                )
            )
            suite = trajectory_success(
                series,
                complete=bool(rollout["reached_reference_end"]),
                physics=evidence,
            )
            interaction, per_finger = contact_metrics(expected=expected, actual=actual, valid=valid)
            flight = flight_metrics(
                tip_contact=actual.any(axis=-1),
                hand_contact=np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(
                    axis=-1
                ),
                valid=valid,
                object_pose=np.asarray(trace["object_pose"], dtype=np.float64),
                object_twist=np.asarray(trace["object_twist"], dtype=np.float64),
            )
            twist = twist_metrics(
                actual=np.asarray(trace["object_twist"], dtype=np.float64),
                reference=np.asarray(trace["object_twist_reference"], dtype=np.float64),
                valid=valid,
                terminal_steps=int(gate["terminal_window_control_steps"]),
            )
            table = np.asarray(trace["table_object_contact"], dtype=bool) & valid
            hand = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
            trace_path = trace_dir / f"episode_{episode:02d}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trace_path,
                **trace,
                frozen_source_actor_sha256=np.asarray(str(source["checkpoint_sha256"])),
                frozen_source_actor_id=np.asarray(str(source["id"])),
                curriculum_stage=np.asarray(stage),
            )
            raw_path = report_dir / "geometry" / f"episode_{episode:02d}_pairs.npz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_path, **raw_geometry)
            row = {
                "reward": source["reward"],
                "clip": source["clip"],
                "stage": stage,
                "episode": episode,
                "seed": seed,
                "trace": str(trace_path.resolve()),
                "steps": int(rollout["steps"]),
                "reached_reference_end": bool(rollout["reached_reference_end"]),
                "any_contact": bool(grasp["any_contact"]),
                "persistent_grasp": bool(grasp["persistent_grasp"]),
                "grasp_and_lift": bool(grasp["grasp_and_lift"]),
                "category": grasp["category"],
                "tip_contact_fraction": float(frozen_contact["tip_contact_fraction"]),
                "persistent_multi_finger_fraction": float(
                    grasp["persistent_multi_finger_fraction"]
                ),
                "source_tip_recall": interaction["source_tip_recall"],
                "persistent_tip_recall": interaction["source_persistent_tip_recall"],
                "active_force_mean_n": grasp["mean_active_force_n"],
                "active_force_p95_n": grasp["p95_active_force_n"],
                "max_contact_force_n": float(grasp["max_force_n"]),
                "contact_reward_activation_fraction": float(
                    grasp["contact_reward_positive_fraction"]
                ),
                "object_lift_dz_m": float(grasp["lift_dz_m"]),
                "object_lift_onset": grasp["object_lift_onset"],
                "drop": bool(frozen_contact["object_drop"]),
                "table_object_contact_fraction": float(table.sum() / valid.sum()),
                "hand_supported_fraction": float((table & hand).sum() / valid.sum()),
                "cross_finger_compensation": interaction["cross_finger_compensation"],
                "persistent_cross_finger_compensation": interaction[
                    "persistent_cross_finger_compensation"
                ],
                "fully_missing_source_contact": interaction["fully_missing_source_contact"],
                "no_tip_contact_fraction": flight["no_tip_contact_fraction"],
                "no_hand_object_contact_fraction": flight["no_hand_object_contact_fraction"],
                "longest_no_hand_contact_gap": int(flight["longest_flight_gap"]),
                "recontact_count": int(flight["recontact_count"]),
                "Delta_v_mean_mps": float(twist["Delta_v_mps"]["mean"]),
                "Delta_v_p95_mps": float(twist["Delta_v_mps"]["p95"]),
                "Delta_v_terminal_mps": float(twist["Delta_v_mps"]["terminal"]),
                "Delta_omega_mean_radps": float(twist["Delta_omega_radps"]["mean"]),
                "Delta_omega_p95_radps": float(twist["Delta_omega_radps"]["p95"]),
                "Delta_omega_terminal_radps": float(twist["Delta_omega_radps"]["terminal"]),
                "terminal_stability": bool(diagnostic["terminal_stability"]),
                "hand_object_max_mm": float(geometry["max_penetration_m"]) * 1000.0,
                "hand_object_p95_mm": float(geometry["p95_penetration_m"]) * 1000.0,
                "active_p95_mm": float(geometry["active_p95_penetration_m"]) * 1000.0,
                "interfinger_max_mm": float(inter.max(initial=0.0)) * 1000.0,
                "hand_table_max_mm": None,
                "hand_table_metric_status": "NOT_IDENTIFIABLE_WITH_CURRENT_TABLE_TRACE",
                "absolute_geometry_pass": bool(diagnostic["absolute_geometry_pass"]),
                "E_r_mean_deg": float(suite["E_r_mean_deg"]),
                "E_t_mean_cm": float(suite["E_t_mean_cm"]),
                "E_j_mean_cm": float(suite["E_j_mean_cm"]),
                "E_ft_mean_cm": float(suite["E_ft_mean_cm"]),
                "kinematic_success": bool(suite["kinematic_success"]),
                "physics_success": bool(suite["physics_success"]),
                "qualified_success": bool(suite["qualified_success"]),
                "SRkin": bool(suite["kinematic_success"]),
                "SRphysics": bool(suite["physics_success"]),
                "SRqualified": bool(suite["qualified_success"]),
                "wrist_ref_command_m": float(command["wrist_position_ref_to_command_m"]["mean"]),
                "wrist_command_actual_m": float(
                    command["wrist_position_command_to_actual_m"]["mean"]
                ),
                "wrist_ref_command_rad": float(
                    command["wrist_rotation_ref_to_command_rad"]["mean"]
                ),
                "wrist_command_actual_rad": float(
                    command["wrist_rotation_command_to_actual_rad"]["mean"]
                ),
                "finger_ref_command_rad": float(command["finger_ref_to_command_rad"]["mean"]),
                "finger_command_actual_rad": float(command["finger_command_to_actual_rad"]["mean"]),
                "contact_formula_max_abs_error": float(
                    frozen_contact["contact_reward_formula_max_abs_error"]
                ),
                "finite": bool(diagnostic["finite"]),
                "simulation_completed": bool(completion["SIMULATION_COMPLETED"]),
                "trace_completed": bool(completion["TRACE_COMPLETED"]),
                "terminal_phase_reached": bool(completion["terminal_semantic_recorded"]),
            }
            detail = {
                "episode": episode,
                "seed": seed,
                "rollout": rollout,
                "row": row,
                "interaction": interaction,
                "per_finger": per_finger,
                "flight": flight,
                "twist": twist,
                "geometry": geometry,
                "diagnostic": diagnostic,
                "physics_evidence": evidence.as_dict(),
                "trace": {"path": str(trace_path.resolve()), "sha256": _sha256(trace_path)},
                "geometry_raw": {"path": str(raw_path.resolve()), "sha256": _sha256(raw_path)},
            }
            _write_json(report_dir / "episodes" / f"episode_{episode:02d}.json", detail)
            rows.append(row)
            detailed.append(detail)
            per_finger_rows.extend([{"episode": episode, **item} for item in per_finger])
        after = _component_hashes(trainer)
        if after != before:
            raise RuntimeError(f"FROZEN_SOURCE_ACTOR_OR_NORMALIZER_MUTATED:{source['id']}:{stage}")
        _write_csv(report_dir / "per_episode.csv", rows)
        _write_csv(report_dir / "per_finger.csv", per_finger_rows)
        suite_aggregate = aggregate_rollouts(rows)
        all_simulation_completed = all(bool(row["simulation_completed"]) for row in rows)
        all_trace_completed = all(bool(row["trace_completed"]) for row in rows)
        all_terminal_phase_reached = all(bool(row["terminal_phase_reached"]) for row in rows)
        qualification_completed = bool(
            all_simulation_completed and all_trace_completed and all_terminal_phase_reached
        )
        summary = {
            "schema_version": "Stage16FrozenSourcePolicyGravitySweepConditionV1",
            "status": (
                "COMPLETE_DIAGNOSTIC_SWEEP"
                if qualification_completed
                else "COMPLETE_DIAGNOSTIC_SWEEP_WITH_PHYSICAL_FAILURE"
            ),
            "source": source,
            "stage": stage,
            "physics": stage_row,
            "runtime": runtime,
            "evaluation_reset": "FRAME0_ONLY_FULL_TRAJECTORY",
            "seeds": seeds,
            "episodes": len(rows),
            "optimizer_steps": 0,
            "actor_hash_before": before["actor"],
            "actor_hash_after": after["actor"],
            "normalizer_hash_before": before["normalizer"],
            "normalizer_hash_after": after["normalizer"],
            "any_contact_episodes": int(sum(bool(row["any_contact"]) for row in rows)),
            "persistent_grasp_episodes": int(sum(bool(row["persistent_grasp"]) for row in rows)),
            "lift_episodes": int(sum(bool(row["grasp_and_lift"]) for row in rows)),
            "contact_fraction": _mean([row["tip_contact_fraction"] for row in rows]),
            "persistent_multi_finger_fraction": _mean(
                [row["persistent_multi_finger_fraction"] for row in rows]
            ),
            "source_tip_recall": _mean([row["source_tip_recall"] for row in rows]),
            "persistent_tip_recall": _mean([row["persistent_tip_recall"] for row in rows]),
            "active_force_mean_n": _mean([row["active_force_mean_n"] for row in rows]),
            "active_force_p95_n": _mean([row["active_force_p95_n"] for row in rows]),
            "max_contact_force_n": max(float(row["max_contact_force_n"]) for row in rows),
            "contact_reward_activation_fraction": _mean(
                [row["contact_reward_activation_fraction"] for row in rows]
            ),
            "object_lift_dz_m": _mean([row["object_lift_dz_m"] for row in rows]),
            "object_lift_onset": _mean([row["object_lift_onset"] for row in rows]),
            "drop_fraction": float(np.mean([bool(row["drop"]) for row in rows])),
            "table_object_contact_fraction": _mean(
                [row["table_object_contact_fraction"] for row in rows]
            ),
            "hand_supported_fraction": _mean([row["hand_supported_fraction"] for row in rows]),
            "interaction": {
                key: _mean([row[key] for row in rows])
                for key in (
                    "cross_finger_compensation",
                    "persistent_cross_finger_compensation",
                    "fully_missing_source_contact",
                    "no_tip_contact_fraction",
                    "no_hand_object_contact_fraction",
                )
            },
            "longest_no_hand_contact_gap": max(
                int(row["longest_no_hand_contact_gap"]) for row in rows
            ),
            "recontact_count": int(sum(int(row["recontact_count"]) for row in rows)),
            "twist": {
                key: _mean([row[key] for row in rows])
                for key in (
                    "Delta_v_mean_mps",
                    "Delta_v_p95_mps",
                    "Delta_v_terminal_mps",
                    "Delta_omega_mean_radps",
                    "Delta_omega_p95_radps",
                    "Delta_omega_terminal_radps",
                )
            },
            "terminal_stability_rate": float(
                np.mean([bool(row["terminal_stability"]) for row in rows])
            ),
            "penetration": {
                "hand_object_max_mm": max(float(row["hand_object_max_mm"]) for row in rows),
                "hand_object_p95_mm": _mean([row["hand_object_p95_mm"] for row in rows]),
                "active_p95_mm": _mean([row["active_p95_mm"] for row in rows]),
                "interfinger_max_mm": max(float(row["interfinger_max_mm"]) for row in rows),
                "hand_table_max_mm": None,
                "hand_table_metric_status": "NOT_IDENTIFIABLE_WITH_CURRENT_TABLE_TRACE",
                "absolute_geometry_pass": all(bool(row["absolute_geometry_pass"]) for row in rows),
            },
            "evaluation_suite_v2": {"aggregate": suite_aggregate},
            "controller_tracking": {
                key: _mean([row[key] for row in rows])
                for key in (
                    "wrist_ref_command_m",
                    "wrist_command_actual_m",
                    "wrist_ref_command_rad",
                    "wrist_command_actual_rad",
                    "finger_ref_command_rad",
                    "finger_command_actual_rad",
                )
            },
            "full_321_frame_traces": all(int(row["steps"]) == 320 for row in rows),
            "completion": {
                "SIMULATION_COMPLETED": all_simulation_completed,
                "TRACE_COMPLETED": all_trace_completed,
                "QUALIFICATION_COMPLETED": qualification_completed,
            },
            "terminal_phase_reached": all_terminal_phase_reached,
            "runtime_s": time.monotonic() - started,
            "trace_root": str(trace_dir.resolve()),
        }
        _write_json(report_dir / "qualification.json", summary)
        _write_json(report_dir / "evaluation_suite_v2.json", summary["evaluation_suite_v2"])
        _write_json(report_dir / "interaction.json", summary["interaction"])
        _write_json(report_dir / "twist.json", summary["twist"])
        _write_json(report_dir / "penetration.json", summary["penetration"])
        return summary
    finally:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only missing conditions after validating completed frozen receipts.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help=(
            "Run only this independent frozen or prepared adapted source; repeat to select several."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Prepared one-or-more adapted-source manifest; never changes checkpoint contents.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Run only this independent physics stage; repeat to select several.",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    isolation = parser.add_mutually_exclusive_group()
    isolation.add_argument(
        "--isolate-conditions",
        action="store_true",
        default=True,
        help="Run every reward/clip/stage condition in its own bounded child process.",
    )
    isolation.add_argument(
        "--no-isolate-conditions",
        action="store_false",
        dest="isolate_conditions",
        help="Internal diagnostic mode only; do not use for formal evidence.",
    )
    parser.add_argument(
        "--condition-timeout-s",
        type=float,
        default=360.0,
        help="Bound for one isolated condition, including Kit startup and finalization.",
    )
    parser.add_argument("--child-condition", action="store_true", help=argparse.SUPPRESS)
    return parser


def _completed_condition(source: Mapping[str, object], stage: str, *, episodes: int) -> bool:
    report_dir, trace_dir = _condition_paths(source, stage)
    path = report_dir / "qualification.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("status")
        in {
            "COMPLETE_DIAGNOSTIC_SWEEP",
            "COMPLETE_DIAGNOSTIC_SWEEP_WITH_PHYSICAL_FAILURE",
        }
        and payload.get("episodes") == episodes
        and payload.get("actor_hash_before") == source["actor_hash"]
        and payload.get("actor_hash_after") == source["actor_hash"]
        and payload.get("normalizer_hash_before") == source["normalizer_hash"]
        and payload.get("normalizer_hash_after") == source["normalizer_hash"]
        and all((trace_dir / f"episode_{episode:02d}.npz").is_file() for episode in range(episodes))
    )


def _prepare_namespace(*, resume: bool) -> None:
    if not resume:
        if any(path.exists() for path in (RUN_ROOT, TRACE_ROOT)):
            raise FileExistsError("FROZEN_SOURCE_SWEEP_NAMESPACE_EXISTS")
        if REPORT_ROOT.exists() and (REPORT_ROOT / "sweep").exists():
            raise FileExistsError("FROZEN_SOURCE_SWEEP_REPORT_NAMESPACE_EXISTS")
        RUN_ROOT.mkdir(parents=True)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        TRACE_ROOT.mkdir(parents=True)
    elif not (RUN_ROOT.is_dir() and REPORT_ROOT.is_dir() and TRACE_ROOT.is_dir()):
        raise FileNotFoundError("FROZEN_SOURCE_SWEEP_RESUME_NAMESPACE_MISSING")


def _terminate_own_child(process: subprocess.Popen[str]) -> None:
    """Gracefully terminate only the isolated condition's process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30.0)


def _run_isolated_conditions(args: argparse.Namespace) -> int:
    """Coordinate child Kit processes without allowing one condition to contaminate another."""

    if args.condition_timeout_s <= 0.0:
        raise ValueError("FROZEN_SOURCE_SWEEP_CONDITION_TIMEOUT_MUST_BE_POSITIVE")
    source_filter = None if args.source is None else set(args.source)
    stage_filter = None if args.stage is None else set(args.stage)
    sources = _load_sources(args.source_manifest)
    physics = _physics_contract()
    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.monotonic()
    script = Path(__file__).resolve()
    environment = dict(os.environ)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONUNBUFFERED"] = "1"
    for source in sorted(sources.values(), key=lambda item: str(item["id"])):
        if source_filter is not None and source["id"] not in source_filter:
            continue
        for stage_row in physics["stages"]:
            stage = str(stage_row["stage"])
            if stage_filter is not None and stage not in stage_filter:
                continue
            if _completed_condition(source, stage, episodes=args.episodes):
                completed.append({"source": source["id"], "stage": stage, "reused": True})
                continue
            command = [
                sys.executable,
                str(script),
                "--accept-eula",
                "--episodes",
                str(args.episodes),
                "--resume",
                "--child-condition",
                "--source",
                str(source["id"]),
                "--stage",
                stage,
                "--run-root",
                str(RUN_ROOT),
                "--report-root",
                str(REPORT_ROOT),
                "--trace-root",
                str(TRACE_ROOT),
            ]
            if args.source_manifest is not None:
                command.extend(["--source-manifest", str(args.source_manifest.resolve())])
            attempt_root = REPORT_ROOT / "condition_subprocess" / str(source["id"]) / stage.lower()
            attempt_root.mkdir(parents=True, exist_ok=True)
            attempt = len(list(attempt_root.glob("attempt_*.json"))) + 1
            child_started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            timeout = False
            try:
                stdout, stderr = process.communicate(timeout=args.condition_timeout_s)
            except subprocess.TimeoutExpired:
                timeout = True
                _terminate_own_child(process)
                stdout, stderr = process.communicate()
            elapsed = time.monotonic() - child_started
            (attempt_root / f"attempt_{attempt:02d}.stdout.txt").write_text(
                stdout, encoding="utf-8"
            )
            (attempt_root / f"attempt_{attempt:02d}.stderr.txt").write_text(
                stderr, encoding="utf-8"
            )
            receipt = {
                "source": source["id"],
                "stage": stage,
                "attempt": attempt,
                "command": command,
                "condition_timeout_s": args.condition_timeout_s,
                "wall_clock_s": elapsed,
                "returncode": process.returncode,
                "timeout": timeout,
                "timeout_attribution": (
                    "WALL_CLOCK_BUDGET_TOO_SHORT_OR_REAL_SIMULATION_HANG" if timeout else None
                ),
                "stdout": str((attempt_root / f"attempt_{attempt:02d}.stdout.txt").resolve()),
                "stderr": str((attempt_root / f"attempt_{attempt:02d}.stderr.txt").resolve()),
            }
            _write_json(attempt_root / f"attempt_{attempt:02d}.json", receipt)
            if (
                timeout
                or process.returncode != 0
                or not _completed_condition(source, stage, episodes=args.episodes)
            ):
                failures.append(receipt)
                _append_failure({"scope": "isolated_condition", **receipt})
            else:
                completed.append({"source": source["id"], "stage": stage, "reused": False})
            _write_json(
                RUN_ROOT / "isolated_progress.json",
                {
                    "completed": completed,
                    "failures": failures,
                    "full_scope": source_filter is None and stage_filter is None,
                },
            )
    payload = {
        "status": "FROZEN_SOURCE_SWEEP_ISOLATED_COMPLETE"
        if not failures
        else "FROZEN_SOURCE_SWEEP_ISOLATED_INCOMPLETE",
        "condition_timeout_s": args.condition_timeout_s,
        "completed": completed,
        "failures": failures,
        "PPO_TRAINING_RUN": False,
        "PPO_OPTIMIZER_STEP": 0,
        "total_runtime_s": time.monotonic() - started,
    }
    _write_json(RUN_ROOT / "isolated_complete.json", payload)
    print(
        json.dumps(
            {"status": payload["status"], "completed": len(completed), "failures": len(failures)}
        )
    )
    return 0 if not failures else 1


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.episodes not in {10, 20}:
        raise ValueError("FROZEN_SOURCE_SWEEP_REQUIRES_EULA_AND_10_OR_20_EPISODES")
    _configure_output_roots(
        run_root=args.run_root, report_root=args.report_root, trace_root=args.trace_root
    )
    _prepare_namespace(resume=args.resume)
    if args.isolate_conditions and not args.child_condition:
        return _run_isolated_conditions(args)
    if args.child_condition and (
        args.source is None or len(args.source) != 1 or args.stage is None or len(args.stage) != 1
    ):
        raise ValueError("FROZEN_SOURCE_SWEEP_CHILD_REQUIRES_ONE_SOURCE_AND_ONE_STAGE")
    started = time.monotonic()
    source_filter = None if args.source is None else set(args.source)
    stage_filter = None if args.stage is None else set(args.stage)
    full_scope = source_filter is None and stage_filter is None
    app = None
    try:
        sources = _load_sources(args.source_manifest)
        physics = _physics_contract()
        static_wrist = _require_static_wrist_receipt()
        seed_manifest = {
            "schema_version": "Stage16FrozenSourcePolicyGravitySweepSeedsV1",
            "protocol": "FRAME0_ONLY_DETERMINISTIC",
            "episodes_per_condition": args.episodes,
            "by_clip": {
                clip: _seeds(clip, count=args.episodes) for clip in ("hocap_170105", "hocap_170650")
            },
        }
        _write_json(REPORT_ROOT / "frozen_inputs.json", {"sources": sources, "physics": physics})
        _write_json(REPORT_ROOT / "physics_contract.json", physics)
        _write_json(REPORT_ROOT / "seed_manifest.json", seed_manifest)
        for item in sources.values():
            _write_json(REPORT_ROOT / "sources" / f"{item['id']}.json", item)
        _write_json(
            REPORT_ROOT / "targeted_preflight.json",
            {
                "source_actor_loading": "PASS",
                "normalizer_loading": "PASS",
                "physics_stage_selection": "PASS",
                "optimizer_steps": 0,
                "fixed_wrist_static_hold": static_wrist,
                "frame0_reset": "PASS",
                "matched_seed_contract": "PASS",
            },
        )
        _write_json(
            RUN_ROOT / "run_contract.json",
            {
                "PPO_TRAINING_RUN": False,
                "PPO_OPTIMIZER_STEP": 0,
                "actor_update": False,
                "critic_update": False,
                "normalizer_update": False,
                "evaluation_reset": "FRAME0_ONLY_FULL_TRAJECTORY",
                "all_stages_attempted_even_after_scientific_failure": True,
            },
        )
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True).app
        gate_by_clip = {
            clip: _load_gate(FROZEN_GATES, clip=clip) for clip in seed_manifest["by_clip"]
        }
        completed: list[dict[str, object]] = []
        for source in sorted(sources.values(), key=lambda item: str(item["id"])):
            mode = ContactRewardMode.parse(str(source["contact_mode"]))
            clip = str(source["clip"])
            if source_filter is not None and source["id"] not in source_filter:
                continue
            seeds = seed_manifest["by_clip"][clip]
            for stage_row in physics["stages"]:
                stage = str(stage_row["stage"])
                if stage_filter is not None and stage not in stage_filter:
                    continue
                if not _completed_condition(source, stage, episodes=args.episodes):
                    _run_condition(
                        source=source,
                        mode=mode,
                        stage_row=stage_row,
                        seeds=seeds,
                        gate=gate_by_clip[clip],
                    )
                completed.append(
                    {
                        "source": source["id"],
                        "stage": stage,
                        "qualification": str(
                            (_condition_paths(source, stage)[0] / "qualification.json").resolve()
                        ),
                    }
                )
                progress_path = (
                    RUN_ROOT / "progress.json"
                    if full_scope
                    else RUN_ROOT / "scopes" / "progress.json"
                )
                _write_json(progress_path, {"completed": completed, "full_scope": full_scope})
        receipt = {
            "conditions": len(completed),
            "episodes": len(completed) * args.episodes,
            "PPO_TRAINING_RUN": False,
            "PPO_OPTIMIZER_STEP": 0,
            "sources": sorted(source_filter) if source_filter is not None else "ALL",
            "stages": sorted(stage_filter) if stage_filter is not None else "ALL",
            "resource_after": _gpu_usage(),
            "total_runtime_s": time.monotonic() - started,
        }
        if full_scope:
            _write_json(REPORT_ROOT / "resource_usage.json", {"before": _gpu_usage(), **receipt})
            _write_json(
                RUN_ROOT / "complete.json",
                {"status": "FROZEN_SOURCE_POLICY_GRAVITY_SWEEP_COMPLETE", **receipt},
            )
        else:
            _write_json(
                RUN_ROOT / "scopes" / "partial_scope_complete.json",
                {"status": "FROZEN_SOURCE_POLICY_GRAVITY_SWEEP_PARTIAL_SCOPE_COMPLETE", **receipt},
            )
        print(
            json.dumps(
                {
                    "status": "PASS" if full_scope else "PARTIAL_SCOPE_PASS",
                    "conditions": len(completed),
                    "episodes": len(completed) * args.episodes,
                }
            )
        )
        return 0
    except BaseException as error:
        _append_failure(
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
