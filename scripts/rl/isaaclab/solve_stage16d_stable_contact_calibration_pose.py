#!/usr/bin/env python3
"""Solve a source-only, low-overlap runtime-proxy pose for stable-contact calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.convex_query import (  # noqa: E402
    PythonFCLConvexQueryBackend,
)
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    load_runtime_geometry_manifest,
)
from toporetarget.rl.geometry_audit.transforms import compose_poses  # noqa: E402

BASELINE_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
TOPOLOGY_PATH = (
    REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting/contact_topology.json"
)
MANIFEST_PATH = BASELINE_ROOT / "runtime_collision_geometry_manifest.json"
GROUP_INDICES = {
    "palm": (0,),
    "index": (1, 2, 3, 4),
    "middle": (5, 6, 7, 8),
    "pinky": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "thumb": (17, 18, 19, 20),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-frame", type=int)
    parser.add_argument("--target-depth-mm", type=float, default=0.5)
    parser.add_argument("--translation-bound-mm", type=float, default=30.0)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_frame(
    signed: np.ndarray, required_groups: tuple[str, ...], explicit: int | None
) -> int:
    if explicit is not None:
        if not 0 <= explicit < signed.shape[0]:
            raise ValueError("source frame is outside [0,321)")
        return explicit
    group_gap = []
    for group in required_groups:
        group_gap.append(signed[:, GROUP_INDICES[group]].min(axis=1))
    worst_required_gap = np.stack(group_gap, axis=1).max(axis=1)
    return int(np.argmin(worst_required_gap))


def main() -> int:
    args = _parser().parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0.0 < args.target_depth_mm <= 1.0:
        raise ValueError("stable-contact target depth must be in (0,1] mm")
    if not 1.0 <= args.translation_bound_mm <= 50.0:
        raise ValueError("stable-contact translation bound must be in [1,50] mm")
    suffix = args.clip.removeprefix("hocap_")
    source_state = BASELINE_ROOT / f"source_collision_state_{suffix}.npz"
    source_pairs = BASELINE_ROOT / f"source_runtime_penetration_pairs_{suffix}.npz"
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))["clips"][args.clip]
    required_groups = tuple(str(value) for value in topology["required_body_groups"])
    with np.load(source_state, allow_pickle=False) as state:
        object_pose = np.asarray(state["object_pose"], dtype=np.float64)[:, 0]
        hand_pose = np.asarray(state["hand_collision_body_pose"], dtype=np.float64)[:, 0]
        hand_names = tuple(str(value) for value in state["hand_collision_body_names"])
    with np.load(source_pairs, allow_pickle=False) as pairs:
        source_signed = np.asarray(pairs["signed_separation_m"], dtype=np.float64)[:, 0]
    frame = _source_frame(source_signed, required_groups, args.source_frame)

    hand_proxies, object_proxies = load_runtime_geometry_manifest(MANIFEST_PATH)
    if hand_names != tuple(proxy.body_name for proxy in hand_proxies):
        raise RuntimeError("STAGE16D_CALIBRATION_HAND_BODY_ORDER_DRIFT")
    backend = PythonFCLConvexQueryBackend()
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_proxy = object_proxies[args.clip][0]
    object_shape = backend.proxy_shape(object_proxy)
    target_m = args.target_depth_mm / 1000.0

    def evaluate(delta_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = object_pose[frame].copy()
        pose[:3] += np.asarray(delta_xyz, dtype=np.float64)
        object_world = compose_poses(pose, object_proxy.local_pose_xyz_wxyz)
        signed = []
        for index, (proxy, shape) in enumerate(zip(hand_proxies, hand_shapes, strict=True)):
            hand_world = compose_poses(hand_pose[frame, index], proxy.local_pose_xyz_wxyz)
            signed.append(
                backend.query(shape, hand_world, object_shape, object_world).signed_separation_m
            )
        values = np.asarray(signed, dtype=np.float64)
        group_values = np.asarray(
            [values[list(GROUP_INDICES[group])].min() for group in required_groups]
        )
        return values, group_values

    def objective(delta_xyz: np.ndarray) -> float:
        values, group_values = evaluate(delta_xyz)
        contact_error = np.square(group_values + target_m).sum()
        excessive = max(0.0, float((-values).max()) - 0.003)
        missing = np.maximum(group_values, 0.0)
        translation_regularizer = 1.0e-4 * float(np.square(delta_xyz).sum())
        return float(
            contact_error
            + 100.0 * excessive**2
            + 10.0 * np.square(missing).sum()
            + translation_regularizer
        )

    bound = args.translation_bound_mm / 1000.0
    result = differential_evolution(
        objective,
        bounds=((-bound, bound),) * 3,
        popsize=args.population,
        maxiter=args.iterations,
        seed=args.seed,
        polish=True,
        workers=1,
        updating="immediate",
    )
    values, group_values = evaluate(result.x)
    solved_pose = object_pose[frame].copy()
    solved_pose[:3] += result.x
    maximum_penetration = float(np.maximum(-values, 0.0).max())
    group_pass = bool(np.all(group_values < 0.0) and np.all(group_values >= -0.001))
    absolute_pass = maximum_penetration <= 0.003
    status = (
        "STAGE16D_STABLE_CONTACT_CALIBRATION_POSE_SOLVED"
        if group_pass and absolute_pass and np.isfinite(result.fun)
        else "STAGE16D_STABLE_CONTACT_CALIBRATION_POSE_BLOCKED"
    )
    payload: dict[str, Any] = {
        "schema_version": "Stage16DStableContactCalibrationPoseV1",
        "status": status,
        "clip": args.clip,
        "source_frame": frame,
        "required_groups": list(required_groups),
        "provenance": "source collision state plus frozen runtime proxies only",
        "corrected_candidate_used": False,
        "source_state": str(source_state.relative_to(REPO_ROOT)),
        "source_state_sha256": _sha256(source_state),
        "runtime_geometry_manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "runtime_geometry_manifest_sha256": _sha256(MANIFEST_PATH),
        "target_depth_m": target_m,
        "translation_delta_m": result.x.tolist(),
        "object_pose_scene_xyz_wxyz": solved_pose.tolist(),
        "required_group_signed_separation_m": {
            group: float(value) for group, value in zip(required_groups, group_values, strict=True)
        },
        "maximum_all_pair_penetration_m": maximum_penetration,
        "optimizer": {
            "algorithm": "scipy differential_evolution translation-only",
            "population_multiplier": args.population,
            "iterations": args.iterations,
            "seed": args.seed,
            "translation_bound_m": bound,
            "success": bool(result.success),
            "termination_is_not_geometry_gate": True,
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "objective": float(result.fun),
        },
        "group_contact_pass": group_pass,
        "absolute_3mm_pass": absolute_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output)}))
    if "BLOCKED" in status:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
