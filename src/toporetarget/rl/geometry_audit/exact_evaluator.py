"""Exact python-fcl evaluation over runtime collision-proxy state arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .convex_query import PythonFCLConvexQueryBackend
from .metrics import aggregate_penetration
from .runtime_geometry import load_runtime_geometry_manifest
from .transforms import compose_poses


def evaluate_runtime_proxy_state(
    *,
    manifest_path: Any,
    clip: str,
    object_pose: np.ndarray,
    hand_collision_body_pose: np.ndarray,
    hand_collision_body_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate every hand/object proxy pair and return formal aggregate plus raw arrays."""

    objects = np.asarray(object_pose, dtype=np.float64)
    hands = np.asarray(hand_collision_body_pose, dtype=np.float64)
    if objects.ndim != 3 or objects.shape[2] != 7:
        raise ValueError("object pose must be [frames,replicas,7]")
    hand_proxies, object_proxies_by_clip = load_runtime_geometry_manifest(manifest_path)
    if clip not in object_proxies_by_clip:
        raise ValueError(f"runtime geometry manifest has no clip: {clip}")
    object_proxies = object_proxies_by_clip[clip]
    expected_names = tuple(proxy.body_name for proxy in hand_proxies)
    if hand_collision_body_names != expected_names:
        raise ValueError("hand collision body order does not match runtime geometry manifest")
    if hands.shape != (*objects.shape[:2], len(hand_proxies), 7):
        raise ValueError("hand collision poses must be [frames,replicas,bodies,7]")
    if not np.isfinite(objects).all() or not np.isfinite(hands).all():
        raise ValueError("formal geometry states must be finite")

    backend = PythonFCLConvexQueryBackend()
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_shapes = [backend.proxy_shape(proxy) for proxy in object_proxies]
    pair_ids = tuple(
        f"{hand.shape_id}<->{obj.shape_id}" for hand in hand_proxies for obj in object_proxies
    )
    frames, replicas = objects.shape[:2]
    signed = np.empty((frames, replicas, len(pair_ids)), dtype=np.float64)
    penetration = np.empty_like(signed)
    direction = np.empty((*signed.shape, 3), dtype=np.float64)
    for frame in range(frames):
        for replica in range(replicas):
            pair_index = 0
            for hand_index, (hand_proxy, hand_shape) in enumerate(
                zip(hand_proxies, hand_shapes, strict=True)
            ):
                hand_world = compose_poses(
                    hands[frame, replica, hand_index], hand_proxy.local_pose_xyz_wxyz
                )
                for object_proxy, object_shape in zip(object_proxies, object_shapes, strict=True):
                    object_world = compose_poses(
                        objects[frame, replica], object_proxy.local_pose_xyz_wxyz
                    )
                    result = backend.query(hand_shape, hand_world, object_shape, object_world)
                    if not result.converged:
                        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_NONCONVERGENCE")
                    signed[frame, replica, pair_index] = result.signed_separation_m
                    penetration[frame, replica, pair_index] = result.penetration_depth_m
                    direction[frame, replica, pair_index] = (
                        result.depenetration_direction_for_second
                    )
                    pair_index += 1
    worst_pair = np.argmax(penetration, axis=2)
    worst = np.take_along_axis(penetration, worst_pair[..., None], axis=2)[..., 0]
    aggregate = aggregate_penetration(worst, worst_pair, pair_ids)
    aggregate.update(
        {
            "schema_version": "RuntimeCollisionProxyPenetrationResultV1",
            "clip": clip,
            "complete_frames": frames,
            "all_queries_converged": True,
            "pair_ids": list(pair_ids),
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


__all__ = ["evaluate_runtime_proxy_state"]
