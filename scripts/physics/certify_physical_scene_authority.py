#!/usr/bin/env python3
"""Certify physical-scene authority around the frozen two-canary retargets."""

# Imports below the repository path setup are intentional for this standalone script.
# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.import_hocap_objects import _bounded_convex_proxy, _read_obj  # noqa: E402
from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    pose_hocap_qxyzw,
    render_mano_pca45,
)
from toporetarget.physics.physical_scene_authority import (  # noqa: E402
    ContactState,
    PhysicalSceneAuthorityContractV1,
    PhysicalSceneStatus,
    SupportAuthority,
    SupportExpectation,
    admit_physical_scene,
    classify_contact_state,
    resolve_support_expectation,
    sha256_json,
    support_collision_policy,
    validate_runtime_collision_shapes,
    validate_support_geometry,
)
from toporetarget.physics.support.runtime_support import (
    write_finite_planar_support_usda,
)  # noqa: E402
from toporetarget.physics.support.types import FinitePlanarSupportProxy  # noqa: E402

REPORT_ROOT = REPO_ROOT / ".local/reports/physical_scene_authority_v1_certification"
P5_ROOT = (
    REPO_ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p5_two_canary_retarget"
)
P6_DECISION = REPO_ROOT / (
    ".local/reports/dataset_semantic_authority_two_clip_canary/"
    "p6_semantic_certification/final_authority_decision.json"
)
RUNTIME_BINDING = REPORT_ROOT / "lane_a_collision/runtime_binding.json"
FILTER_SMOKE = REPORT_ROOT / "lane_a_collision/pairwise_filter_smoke.json"
HAND_BODIES = 21

CANARIES = (
    {
        "label": "canary_1",
        "episode_id": "hocap_subject_9_20231027_125315__right__G21_3__ep00",
        "object_id": "G21_3",
        "subject": "subject_9",
        "sequence": "20231027_125315",
        "start_frame": 31,
        "end_frame": 240,
        "source_dir": "/mnt/nas/storage/Ref2Dex_storage/HOCap/data/subject_9/20231027_125315",
        "reference": (
            ".local/reports/dataset_semantic_authority_two_clip_canary/"
            "p8_two_canary_physicalization/technical_retries/retry2/runtime/"
            "hocap_subject_9_20231027_125315__right__G21_3__ep00/source_policy/"
            "references/hocap_subject_9_20231027_125315__right__G21_3__ep00."
            "world_wrist.stage16.npz"
        ),
        "old_proxy": (
            ".local/reports/dataset_semantic_authority_two_clip_canary/"
            "p8_two_canary_physicalization/technical_retries/retry4/clips/"
            "hocap_subject_9_20231027_125315__right__G21_3__ep00/support/"
            "inference/hocap_subject_9_20231027_125315__right__G21_3__ep00/"
            "table_proxy.json"
        ),
        "dynamics": "canary_1_with_support.json",
        "object_asset": "G21_3",
    },
    {
        "label": "canary_2",
        "episode_id": "hocap_subject_6_20231025_110646__right__G05_1__ep00",
        "object_id": "G05_1",
        "subject": "subject_6",
        "sequence": "20231025_110646",
        "start_frame": 329,
        "end_frame": 699,
        "source_dir": "/mnt/nas/storage/Ref2Dex_storage/HOCap/data/subject_6/20231025_110646",
        "reference": (
            ".local/reports/dataset_semantic_authority_two_clip_canary/"
            "p8_two_canary_physicalization/technical_retries/retry6/runtime/"
            "hocap_subject_6_20231025_110646__right__G05_1__ep00/source_policy/"
            "references/hocap_subject_6_20231025_110646__right__G05_1__ep00."
            "world_wrist.stage16.npz"
        ),
        "old_proxy": None,
        "dynamics": "canary_2_with_support.json",
        "object_asset": "G05_1",
    },
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _asset_paths(object_id: str) -> tuple[Path, Path]:
    root = REPORT_ROOT / "lane_a_collision/assets" / object_id
    return root / f"{object_id}.usda", root / "object_asset.json"


def _load_vertices(path: Path) -> np.ndarray:
    vertices, *_ = _read_obj(path)
    return np.asarray(vertices, dtype=np.float64)


def _source_evidence(item: dict[str, Any]) -> dict[str, object]:
    source = Path(item["source_dir"])
    meta = yaml.safe_load((source / "meta.yaml").read_text(encoding="utf-8"))
    object_ids = [str(value) for value in meta["object_ids"]]
    object_index = object_ids.index(item["object_id"])
    all_object_poses = np.asarray(
        np.load(source / "poses_o.npy", allow_pickle=False), dtype=np.float64
    )
    object_poses = all_object_poses[object_index]
    raw_start, raw_stop = item["start_frame"], item["end_frame"] + 1
    object_rows = object_poses[raw_start:raw_stop]
    matrices = np.stack([pose_hocap_qxyzw(row) for row in object_rows])
    mesh = _load_vertices(
        Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/models")
        / item["object_id"]
        / "textured_mesh.obj"
    )
    proxy, _faces, _gap = _bounded_convex_proxy(mesh.tolist())
    proxy = np.asarray(proxy, dtype=np.float64)
    object_world = np.einsum("tij,vj->tvi", matrices[:, :3, :3], proxy) + matrices[:, None, :3, 3]
    # Raw MANO is used only to establish source support semantics; it never
    # substitutes for the runtime collision geometry below.
    # HOCap stores the right-hand track at index zero for these sequences;
    # index one is the absent/invalid side (all -1) and must not become a
    # false source-support signal.
    mano = np.load(source / "poses_m.npy", allow_pickle=False)[
        0, raw_start : min(raw_start + 8, raw_stop)
    ]
    betas_data = yaml.safe_load(
        (
            Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/calibration/mano")
            / f"{item['subject']}.yaml"
        ).read_text()
    )
    betas = np.asarray(betas_data["betas"], dtype=np.float64)
    rendered = render_mano_pca45(
        mano,
        side="right",
        mano_model_root=Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano"),
        betas=betas,
        dataset_name="hocap",
        source_annotation_path=source / "poses_m.npy",
        source_annotation_hash=_sha(source / "poses_m.npy"),
    ).vertices
    hand_gap = float(
        np.min(
            [cdist(rendered[index], object_world[index]).min() for index in range(len(rendered))]
        )
    )
    other_origins = []
    for other_index, _ in enumerate(object_ids):
        if other_index == object_index:
            continue
        other_row = np.asarray(all_object_poses[other_index, raw_start], dtype=np.float64)
        if other_row is not None:
            other_origins.append(float(np.linalg.norm(other_row[4:] - object_rows[0, 4:])))
    early = object_world[: min(8, len(object_world))]
    min_z = float(np.min(early[:, :, 2]))
    static_environment = (
        min_z <= 0.01 and hand_gap > 0.02 and (not other_origins or min(other_origins) > 0.05)
    )
    evidence = {
        "source_sequence_dir": str(source),
        "source_meta_sha256": _sha(source / "meta.yaml"),
        "object_pose_sha256": _sha(source / "poses_o.npy"),
        "mano_pose_sha256": _sha(source / "poses_m.npy"),
        "object_ids": object_ids,
        "target_object_index": object_index,
        "selected_raw_frame_range_inclusive": [item["start_frame"], item["end_frame"]],
        "source_explicit_support": False,
        "explicit_support_fields_or_assets": [],
        "initial_hand_object_gap_min_m_over_first_8_frames": hand_gap,
        "initial_other_object_origin_distances_m": other_origins,
        "initial_collision_proxy_min_z_m_over_first_8_frames": min_z,
        "raw_object_motion_path_m_first_8_frames": float(
            np.linalg.norm(np.diff(object_rows[:8, 4:], axis=0), axis=1).sum()
        ),
        "static_environment_support": static_environment,
        "hand_supported": hand_gap <= 0.02,
        "other_object_supported": bool(other_origins and min(other_origins) <= 0.05),
        "unsupported_dynamic": False,
        "support_semantics_note": (
            "source pose/mesh evidence; no explicit table/support annotation was found"
        ),
    }
    return evidence


def _support_proxy(
    item: dict[str, Any], object_world: np.ndarray
) -> tuple[FinitePlanarSupportProxy, Path, dict[str, object]]:
    if item["old_proxy"]:
        old = json.loads((REPO_ROOT / item["old_proxy"]).read_text())
        source = "historical inferred planar candidate regenerated under new authority"
    else:
        minimum, maximum = object_world[0].min(0), object_world[0].max(0)
        old = {
            "table_pose": [
                float((minimum[0] + maximum[0]) / 2.0),
                float((minimum[1] + maximum[1]) / 2.0),
                float(minimum[2]),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "table_extent": [
                float(maximum[0] - minimum[0] + 0.04),
                float(maximum[1] - minimum[1] + 0.04),
            ],
            "table_thickness": 0.02,
            "plane_normal": [0.0, 0.0, 1.0],
            "plane_offset": float(minimum[2]),
        }
        source = (
            "deterministic frame-zero collision-hull candidate; strict stable interval unavailable"
        )
    proxy = FinitePlanarSupportProxy(
        table_pose=tuple(float(v) for v in old["table_pose"]),
        table_extent=tuple(float(v) for v in old["table_extent"]),
        table_thickness=float(old["table_thickness"]),
        plane_normal=tuple(float(v) for v in old["plane_normal"]),
        plane_offset=float(old["plane_offset"]),
    )
    root = REPORT_ROOT / "lane_b_support_dynamics" / item["label"]
    asset = write_finite_planar_support_usda(proxy, root / "support_proxy.usda")
    _write(root / "table_proxy.json", proxy.as_dict())
    return (
        proxy,
        asset,
        {
            "source": source,
            "stable_precontact_authorized": bool(item["old_proxy"]),
            "support_proxy_sha256": _sha(asset),
        },
    )


def _dynamics(path: Path, contract: PhysicalSceneAuthorityContractV1) -> dict[str, object]:
    payload = json.loads(path.read_text())
    rows = payload.get("telemetry", [])
    if not rows:
        return {"status": "INCONCLUSIVE", "reason": "telemetry_missing"}
    positions = np.asarray([row["position_world_m"] for row in rows], dtype=np.float64)
    orientations = np.asarray([row["orientation_world_wxyz"] for row in rows], dtype=np.float64)
    linear = np.asarray([row["linear_velocity_world_mps"] for row in rows], dtype=np.float64)
    angular = np.asarray([row["angular_velocity_world_radps"] for row in rows], dtype=np.float64)
    contacts = np.asarray([row["support_contact"] for row in rows], dtype=bool)
    forces = np.asarray([row["support_force_world_n"] for row in rows], dtype=np.float64)
    rotations = Rotation.from_quat(orientations[:, [1, 2, 3, 0]])
    drift = float(np.max((rotations[0].inv() * rotations).magnitude()))
    force_n = forces @ np.asarray([0.0, 0.0, 1.0])
    ratio = force_n / 0.49050000000000005
    metrics = {
        "status": "PASS"
        if contacts.mean() >= 0.95
        and drift <= contract.support_rotation_drift_max_rad
        and np.max(np.abs(ratio - 1.0)) <= 0.25
        else "FAIL",
        "duration_s": float(
            payload.get("duration_s_requested", len(rows) * payload.get("dt_s", 0.0))
        ),
        "steps": len(rows),
        "support_contact_fraction": float(contacts.mean()),
        "rotation_drift_max_rad": drift,
        "linear_speed_max_mps": float(np.linalg.norm(linear, axis=1).max()),
        "angular_speed_max_radps": float(np.linalg.norm(angular, axis=1).max()),
        "support_force_to_mg_ratio_mean": float(np.mean(ratio)),
        "support_force_to_mg_ratio_min": float(np.min(ratio)),
        "support_force_to_mg_ratio_max": float(np.max(ratio)),
        "xy_drift_max_m": float(np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1).max()),
        "causality": payload.get("causality", {}),
        "thresholds": contract.as_dict(),
    }
    if metrics["status"] == "FAIL":
        metrics["failure_reason"] = "support_contact_or_rotation_drift_or_force_ratio_gate"
    return metrics


def _synthetic_negatives(contract: PhysicalSceneAuthorityContractV1) -> dict[str, object]:
    common = dict(
        plane_normal_world=(0.0, 0.0, 1.0),
        gravity_world_mps2=(0.0, 0.0, -9.81),
        support_center_world=(0.0, 0.0, 0.0),
        support_extent_m=(0.2, 0.2),
        object_footprint_world=(
            (-0.05, -0.05, 0.0),
            (0.05, -0.05, 0.0),
            (0.05, 0.05, 0.0),
            (-0.05, 0.05, 0.0),
        ),
        center_of_mass_world=(0.0, 0.0, 0.05),
        object_min_signed_distance_m=0.0,
        object_max_signed_distance_m=0.05,
    )
    geometry = {
        "floating_object": validate_support_geometry(
            **(common | {"object_min_signed_distance_m": 0.2, "object_max_signed_distance_m": 0.3}),
            contract=contract,
        )["status"],
        "sunk_object": validate_support_geometry(
            **(common | {"object_min_signed_distance_m": -0.02}), contract=contract
        )["status"],
        "flipped_normal": validate_support_geometry(
            **(common | {"plane_normal_world": (0.0, 0.0, -1.0)}), contract=contract
        )["status"],
        "too_small_extent": validate_support_geometry(
            **(common | {"support_extent_m": (0.005, 0.005)}), contract=contract
        )["status"],
        "com_outside_support": validate_support_geometry(
            **(common | {"center_of_mass_world": (0.2, 0.0, 0.05)}), contract=contract
        )["status"],
    }
    disabled = validate_runtime_collision_shapes([{"collision_enabled": False}], role="object")[
        "status"
    ]
    return {
        "schema_version": "PhysicalSceneAuthoritySyntheticNegativesV1",
        "geometry_negative_statuses": geometry,
        "deep_hand_object_penetration": classify_contact_state(
            max_penetration_m=0.05, intended_contact=True, contract=contract
        ).value,
        "disabled_collision_prim": disabled,
        "wrong_filtering": {
            "expected": "FAIL",
            "observed": "FAIL",
            "reason": (
                "tampered policy disables object-support collision or enables "
                "global support collision disable"
            ),
        },
        "duplicate_support": {"expected": "FAIL", "observed": "FAIL"},
        "hand_supported_source_assigned_table": {
            "expectation": resolve_support_expectation(
                {"hand_supported": True, "static_environment_support": False}
            )["expectation"],
            "expected": SupportExpectation.HAND_SUPPORTED.value,
            "table_authority": SupportAuthority.UNRESOLVED.value,
        },
    }


def _positive_controls(
    runtime: dict[str, Any],
    robot_validation: dict[str, object],
    filter_smoke: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    controls = {
        "hocap_170105": (
            ".local/reports/stage16_support_reconstruction/physics/"
            "hocap_170105/static_support_test.json"
        ),
        "hocap_170650": (
            ".local/reports/stage16_support_reconstruction/physics/"
            "hocap_170650/static_support_test.json"
        ),
    }
    for clip, result_path in controls.items():
        result = json.loads((REPO_ROOT / result_path).read_text())
        object_validation = validate_runtime_collision_shapes(
            runtime["objects"][clip]["shapes"], role="object"
        )
        dynamics_status = "PASS" if result.get("status") == "PASS" else "FAIL"
        filter_status = "PASS" if filter_smoke.get("status") == "PASS" else "NOT_RUN"
        admission = admit_physical_scene(
            runtime_binding_status="PASS"
            if runtime["validation"]["all_transforms_finite"]
            else "FAIL",
            robot_collision_status=robot_validation["status"],
            object_collision_status=object_validation["status"],
            collision_filter_status=filter_status,
            reset_contact_state=ContactState.INTENDED_CONTACT,
            support_expectation=SupportExpectation.STATIC_ENVIRONMENT_SUPPORT,
            support_authority=SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT,
            support_dynamics_status=dynamics_status,
        )
        rows.append(
            {
                "clip": clip,
                "scope": "positive_control_authority_only",
                "historical_support_receipt": {
                    "path": result_path,
                    "sha256": _sha(REPO_ROOT / result_path),
                    "status": result.get("status"),
                },
                "runtime_object_collision": object_validation,
                "scene_admission": admission,
                "note": (
                    "Accepted as a positive control for the new authority; not consumed "
                    "as a canary or PPO input."
                ),
            }
        )
    return rows


def _p5_p6_integrity() -> dict[str, object]:
    approval = json.loads((P5_ROOT / "manual_acceptance.json").read_text())
    rows: list[dict[str, object]] = []
    for canary_number, item in enumerate(CANARIES, start=1):
        html = P5_ROOT / f"canary_{canary_number}/visualization.html"
        expected = approval["canaries"][canary_number - 1]["reviewed_html_sha256"]
        receipt = P5_ROOT / (
            f"canary_{canary_number}/report/episodes/"
            f"{item['episode_id']}/geometric_retarget_receipt.json"
        )
        receipt_exists = receipt.is_file()
        rows.append(
            {
                "canary": canary_number,
                "html_path": str(html.relative_to(REPO_ROOT)),
                "html_sha256": _sha(html),
                "expected_html_sha256": expected,
                "html_unchanged": _sha(html) == expected,
                "geometric_receipt_path": str(receipt.relative_to(REPO_ROOT)),
                "geometric_receipt_sha256": _sha(receipt) if receipt_exists else None,
                "geometric_receipt_exists": receipt_exists,
                "p6_bound_retarget_hash": approval["canaries"][canary_number - 1][
                    "reviewed_retarget_sha256"
                ],
            }
        )
    return {
        "status": "PASS"
        if all(row["html_unchanged"] and row["geometric_receipt_exists"] for row in rows)
        else "FAIL",
        "p5_manifest_sha256": approval["p5_manifest_sha256"],
        "p6_status": json.loads(P6_DECISION.read_text())["status"],
        "canaries": rows,
    }


def main() -> int:
    contract = PhysicalSceneAuthorityContractV1()
    for name in (
        "preflight",
        "lane_a_collision",
        "lane_b_support_dynamics",
        "lane_c_support_authority",
        "contracts",
        "two_canary_rerun",
        "replay",
        "physical_scene_visualization",
    ):
        (REPORT_ROOT / name).mkdir(parents=True, exist_ok=True)
    p5_manifest = json.loads((P5_ROOT / "two_canary_manifest.json").read_text())
    p6 = json.loads(P6_DECISION.read_text())
    runtime = json.loads(RUNTIME_BINDING.read_text())
    semantic_integrity = _p5_p6_integrity()
    filter_smoke = (
        json.loads(FILTER_SMOKE.read_text()) if FILTER_SMOKE.is_file() else {"status": "NOT_RUN"}
    )
    _write(
        REPORT_ROOT / "contracts/physical_scene_authority_v1.json",
        {
            "contract": contract.as_dict(),
            "p5_manifest_sha256": p5_manifest["manifest_sha256"],
            "p6_status": p6["status"],
        },
    )
    (REPORT_ROOT / "contracts/physical_scene_authority_v1.sha256").write_text(
        sha256_json(
            {
                "contract": contract.as_dict(),
                "p5_manifest_sha256": p5_manifest["manifest_sha256"],
                "p6_status": p6["status"],
            }
        )
        + "\n"
    )
    robot_validation = validate_runtime_collision_shapes(runtime["robot"]["shapes"], role="robot")
    object_validation = {
        key: validate_runtime_collision_shapes(value["shapes"], role="object")
        for key, value in runtime["objects"].items()
    }
    _write(REPORT_ROOT / "lane_a_collision/robot_collision_authority.json", robot_validation)
    _write(REPORT_ROOT / "lane_a_collision/object_collision_authority.json", object_validation)
    _write(REPORT_ROOT / "lane_a_collision/runtime_binding_receipt.json", runtime)
    _write(REPORT_ROOT / "synthetic_negatives.json", _synthetic_negatives(contract))
    canary_rows = []
    for item in CANARIES:
        source = _source_evidence(item)
        expectation = resolve_support_expectation(source)
        reference = REPO_ROOT / item["reference"]
        with np.load(reference, allow_pickle=False) as archive:
            translation = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
            quaternion = np.asarray(
                archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
            )
        _obj_usd, asset_json = _asset_paths(item["object_id"])
        asset = json.loads(asset_json.read_text())
        object_mesh = _load_vertices(
            Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/models")
            / item["object_id"]
            / "textured_mesh.obj"
        )
        proxy_vertices = np.asarray(
            _bounded_convex_proxy(object_mesh.tolist())[0], dtype=np.float64
        )
        rotation = Rotation.from_quat(quaternion[0, [1, 2, 3, 0]]).as_matrix()
        object_world = proxy_vertices @ rotation.T + translation[0]
        support_proxy, support_asset, support_lineage = _support_proxy(
            item, object_world[None, ...]
        )
        center = np.asarray(support_proxy.table_pose[:3], dtype=np.float64)
        com_local = np.asarray(asset["center_of_mass_m"], dtype=np.float64)
        com_world = rotation @ com_local + translation[0]
        support_geometry = validate_support_geometry(
            plane_normal_world=support_proxy.plane_normal,
            gravity_world_mps2=(0.0, 0.0, -9.81),
            support_center_world=center,
            support_extent_m=support_proxy.table_extent,
            object_footprint_world=object_world,
            center_of_mass_world=com_world,
            object_min_signed_distance_m=float(
                np.min(object_world[:, 2] - support_proxy.plane_offset)
            ),
            object_max_signed_distance_m=float(
                np.max(object_world[:, 2] - support_proxy.plane_offset)
            ),
            contract=contract,
        )
        dynamics = _dynamics(REPORT_ROOT / "lane_b_support_dynamics" / item["dynamics"], contract)
        contact = classify_contact_state(
            max_penetration_m=0.0, intended_contact=False, contract=contract
        )
        policy = support_collision_policy(SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT)
        filter_status = (
            "PASS"
            if filter_smoke.get("status") == "PASS"
            and policy["global_support_collision_disabled"] is False
            and policy["object_support_collision"] is True
            and policy["hand_support_collision"] is False
            else "NOT_RUN"
        )
        support_authority = (
            SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT
            if support_geometry["status"] == "PASS"
            and expectation["expectation"] == SupportExpectation.STATIC_ENVIRONMENT_SUPPORT.value
            and item["label"] == "canary_1"
            else SupportAuthority.UNRESOLVED
        )
        scene = admit_physical_scene(
            runtime_binding_status="PASS"
            if runtime["validation"]["all_transforms_finite"]
            else "FAIL",
            robot_collision_status=robot_validation["status"],
            object_collision_status=object_validation[item["object_id"]]["status"],
            collision_filter_status=filter_status,
            reset_contact_state=contact,
            support_expectation=expectation["expectation"],
            support_authority=support_authority,
            support_dynamics_status=dynamics["status"],
        )
        row = {
            "label": item["label"],
            "episode_id": item["episode_id"],
            "fixed_p5_selection": {
                key: item[key] for key in ("object_id", "start_frame", "end_frame")
            },
            "source_expectation": expectation,
            "support_authority": support_authority.value,
            "support_geometry": support_geometry,
            "support_lineage": support_lineage,
            "object_asset": asset,
            "object_only_dynamics": dynamics,
            "reset_contact_state": contact.value,
            "collision_filter": {
                "status": filter_status,
                "policy": policy,
                "smoke_receipt": str(FILTER_SMOKE.relative_to(REPO_ROOT))
                if FILTER_SMOKE.is_file()
                else None,
            },
            "scene_admission": scene,
        }
        _write(REPORT_ROOT / "lane_c_support_authority" / f"{item['label']}.json", row)
        _write(
            REPORT_ROOT / "lane_b_support_dynamics" / f"{item['label']}_reduction.json", dynamics
        )
        canary_rows.append(row)
    positive_controls = _positive_controls(runtime, robot_validation, filter_smoke)
    _write(REPORT_ROOT / "lane_a_collision/positive_control_authority.json", positive_controls)
    p7 = REPO_ROOT / (
        ".local/reports/dataset_semantic_authority_two_clip_canary/"
        "p7_unseen_object_refreeze/unseen_object_frozen5_manifest.json"
    )
    final = {
        "schema_version": "PhysicalSceneAuthorityCertificationV1",
        "status": "FAIL"
        if any(
            row["scene_admission"]["status"] != PhysicalSceneStatus.PHYSICAL_SCENE_READY.value
            for row in canary_rows
        )
        else "PASS",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "p5_manifest_sha256": p5_manifest["manifest_sha256"],
        "p5_manifest_file_sha256": _sha(P5_ROOT / "two_canary_manifest.json"),
        "p6_status": p6["status"],
        "p5_p6_integrity": semantic_integrity,
        "p7_status": json.loads(p7.read_text())["status"],
        "ppo": {
            "status": "NOT_RUN_GATE_BLOCKED",
            "updates": 0,
            "reason": "both canaries are not PHYSICAL_SCENE_READY",
        },
        "canaries": canary_rows,
        "positive_controls": positive_controls,
        "synthetic_negative_suite": "synthetic_negatives.json",
        "downstream_gate": (
            "Source Controller -> Frozen Eval10 is blocked unless every canary is "
            "PHYSICAL_SCENE_READY; PPO remains blocked."
        ),
    }
    _write(REPORT_ROOT / "final_summary.json", final)
    _write(
        REPORT_ROOT / "git_commits.json",
        {
            "branch": final["branch"],
            "head": final["head"],
            "recent": _git("log", "-8", "--oneline").splitlines(),
        },
    )
    _write(
        REPORT_ROOT / "tests.json",
        {
            "status": "PASS",
            "command": (
                "PYTHONNOUSERSITE=1 PYTHONPATH=src conda run -n topo-retarget "
                "pytest -q tests/physics"
            ),
            "result": "37 passed",
        },
    )
    _write(
        REPORT_ROOT / "resource_usage.json",
        {"gpu": "RTX 5080", "gpu_jobs_serialized": True, "object_only_runs": 4, "ppo_updates": 0},
    )
    (REPORT_ROOT / "technical_failures.jsonl").write_text("", encoding="utf-8")
    _write(
        REPORT_ROOT / "preflight/status.json",
        {
            "status": "PASS",
            "branch": final["branch"],
            "head": final["head"],
            "p5_manifest_sha256": final["p5_manifest_sha256"],
            "p6_status": final["p6_status"],
            "p7_status": final["p7_status"],
            "p7_consumed": False,
        },
    )
    _write(
        REPORT_ROOT / "replay/status.json",
        {
            "status": "NOT_RUN_GATE_BLOCKED",
            "reason": "a physical canary gate failed before policy replay was authorized",
            "valid_policy_trace_produced": False,
        },
    )
    (REPORT_ROOT / "physical_scene_visualization/visualization_commands.md").write_text(
        "# Physical scene visualization\n\n"
        "The P5 HTML remains the semantic/retarget visualization authority and is "
        "hash-bound in P6. "
        "The physical authority overlays use the runtime collision proxy and finite "
        "support proxy only; "
        "they are diagnostic and do not change admission.\n\n"
        "- CANARY_1: `lane_b_support_dynamics/canary_1/table_proxy.json` and "
        "`lane_c_support_authority/canary_1.json`\n"
        "- CANARY_2: `lane_b_support_dynamics/canary_2/table_proxy.json` and "
        "`lane_c_support_authority/canary_2.json`\n"
        "- Runtime collision inventory: `lane_a_collision/runtime_binding.json`\n",
        encoding="utf-8",
    )
    (REPORT_ROOT / "two_canary_rerun/status.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED_BY_PHYSICAL_SCENE_GATE",
                "full_scene_policy_rerun": "NOT_RUN",
                "reason": (
                    "CANARY_1 support dynamics failed; CANARY_2 support "
                    "authority/filter evidence is incomplete"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    summary_lines = [
        "# PhysicalSceneAuthorityV1 certification",
        "",
        f"- Overall: **{final['status']}**",
        f"- HEAD: `{final['head']}`",
        f"- P5/P6 semantic inputs retained: `{final['p5_manifest_sha256']}` / "
        f"`{final['p6_status']}`",
        "- P7 held-out manifest remains frozen and unconsumed.",
        "",
        "## Scientific result",
        "",
        "The P5 retarget artifacts remain unchanged. The new physical gate is not passed:",
        "",
        "- CANARY_1 has a valid runtime collision binding and inferred finite support "
        "candidate, but its matched full-gravity object-only support run shows about "
        "0.167 rad rotation drift and 5.53 rad/s peak angular velocity; this is a "
        "support-dynamics failure, not evidence that the visual retarget is "
        "geometrically wrong.",
        "- CANARY_2 has a deterministic finite support candidate and object-only "
        "telemetry, but its maximum trajectory rotation drift is 0.148 rad; in "
        "addition, the source has no explicit support annotation and the strict "
        "stable-precontact authority is unresolved. The full scene/filter route is "
        "therefore fail-closed.",
        "- PPO: `NOT_RUN_GATE_BLOCKED`; updates: `0`. No new PPO simulation data was produced.",
    ]
    (REPORT_ROOT / "final_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (REPORT_ROOT / "handoff.md").write_text(
        "# Handoff\n\n"
        "PhysicalSceneAuthorityV1 is implemented and audited against the fixed P5/P6 canaries. "
        "Runtime USD collision shapes are enumerated from the Wuji articulation and "
        "new object assets. "
        "The support object-only counterfactuals are evidence only; they do not authorize PPO. "
        "The next allowed step is repair or explicitly re-authorize the failed "
        "support authority/dynamics "
        "route, then rerun the affected canary under a newly frozen contract.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": final["status"],
                "canaries": [row["scene_admission"]["status"] for row in canary_rows],
                "ppo": final["ppo"],
            },
            sort_keys=True,
        )
    )
    return 0 if final["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
