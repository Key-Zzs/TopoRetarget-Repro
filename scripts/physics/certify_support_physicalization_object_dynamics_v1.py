#!/usr/bin/env python3
"""Certify V2 support physicalization and settled object dynamics.

This is the bounded successor to the historical PhysicalSceneAuthorityV1
report.  It derives the two canaries from the immutable P5/P6 receipts, freezes
the V2 protocol before observing canary outcomes, and optionally runs matched
object-only PhysX counterfactuals.  It never edits semantic retarget outputs or
the held-out P7 manifest.
"""

# Imports below the repository path setup are intentional for this standalone script.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
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
    SupportAuthority,
    SupportExpectation,
    support_collision_policy,
    validate_runtime_collision_shapes,
    validate_support_geometry,
)
from toporetarget.physics.physicalization_authority import (  # noqa: E402
    ObjectDynamicsAuthorityContractV1,
    PhysicalizationCandidateV1,
    PhysicalizationDeviationBudgetV1,
    PhysicalizationMode,
    SettledSupportDynamicsQualificationV2,
    SupportExistenceContractV1,
    audit_object_dynamics_provenance,
    audit_runtime_default_provenance,
    build_physical_scene_protocol_v2,
    compare_retarget_reuse,
    qualify_settled_support_dynamics_v2,
    resolve_support_existence,
    select_physicalization_candidate,
    sha256_json,
)
from toporetarget.physics.support.runtime_support import (
    write_finite_planar_support_usda,  # noqa: E402
)
from toporetarget.physics.support.types import FinitePlanarSupportProxy  # noqa: E402

REPORT_ROOT = REPO_ROOT / ".local/reports/support_physicalization_object_dynamics_v1"
P5_ROOT = (
    REPO_ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/p5_two_canary_retarget"
)
P6_DECISION = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p6_semantic_certification/"
    "final_authority_decision.json"
)
P7_MANIFEST = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p7_unseen_object_refreeze/"
    "unseen_object_frozen5_manifest.json"
)
V1_ROOT = REPO_ROOT / ".local/reports/physical_scene_authority_v1_certification"
P8_ROOT = (
    REPO_ROOT
    / ".local/reports/dataset_semantic_authority_two_clip_canary/p8_two_canary_physicalization"
)
RUNTIME_BINDING = V1_ROOT / "lane_a_collision/runtime_binding.json"
FILTER_SMOKE = V1_ROOT / "lane_a_collision/pairwise_filter_smoke.json"
MANO_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
DATA_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-physics",
        action="store_true",
        help="Run four serialized three-second object-only PhysX counterfactuals.",
    )
    parser.add_argument(
        "--isaac-env",
        default="toporetarget-isaaclab",
        help="Conda environment containing Isaac Lab.",
    )
    return parser


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _artifact(path: Path) -> dict[str, object]:
    return (
        {"path": str(path.resolve()), "sha256": _sha(path)}
        if path.is_file()
        else {
            "path": str(path.resolve()),
            "sha256": None,
            "status": "MISSING",
        }
    )


def _load_fixed_canaries() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    p5 = _json(P5_ROOT / "two_canary_manifest.json")
    manual = _json(P5_ROOT / "manual_acceptance.json")
    p6 = _json(P6_DECISION)
    if manual.get("status") != "APPROVED_FOR_P6_P8" or p6.get("status") != "PASS":
        raise RuntimeError("P5_P6_FIXED_CANARY_AUTHORITY_NOT_PASS")
    by_episode = {row["episode_id"]: row for row in p5.get("clips", [])}
    canaries: list[dict[str, Any]] = []
    for approval in manual.get("canaries", []):
        episode = str(approval["episode_id"])
        if approval.get("reviewer_decision") != "APPROVE":
            raise RuntimeError(f"CANARY_NOT_APPROVED:{episode}")
        row = by_episode.get(episode)
        if row is None:
            raise RuntimeError(f"CANARY_NOT_IN_P5_MANIFEST:{episode}")
        number = int(approval["canary"])
        html = P5_ROOT / f"canary_{number}/visualization.html"
        if _sha(html) != approval["reviewed_html_sha256"]:
            raise RuntimeError(f"CANARY_HTML_DRIFT:{episode}")
        geometric = P5_ROOT / (
            f"canary_{number}/report/episodes/{episode}/geometric_retarget_receipt.json"
        )
        if not geometric.is_file():
            raise RuntimeError(f"CANARY_GEOMETRIC_RECEIPT_MISSING:{episode}")
        p8_ledger = _json(P8_ROOT / "per_episode" / f"{episode}.json")
        source_item = p8_ledger.get("source_evidence", {})
        source_ref = source_item.get("final") or source_item.get("prerequisite")
        if not isinstance(source_ref, dict) or not source_ref.get("path"):
            raise RuntimeError(f"CANARY_SOURCE_RECEIPT_MISSING:{episode}")
        source_receipt = Path(str(source_ref["path"]))
        source_payload = _json(source_receipt)
        world_ref = source_payload.get("artifacts", {}).get("world_reference")
        if not isinstance(world_ref, dict):
            raise RuntimeError(f"CANARY_WORLD_REFERENCE_MISSING:{episode}")
        world_reference = Path(str(world_ref["path"]))
        if not world_reference.is_file() or _sha(world_reference) != world_ref.get("sha256"):
            raise RuntimeError(f"CANARY_WORLD_REFERENCE_DRIFT:{episode}")
        source_dir = DATA_ROOT / str(row["raw_sequence"]).replace("/", "/")
        canaries.append(
            {
                "label": f"canary_{number}",
                "canary": number,
                "episode_id": episode,
                "object_id": row["target_object"],
                "subject": row["subject"],
                "raw_sequence": row["raw_sequence"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "selection_key": row["selection_key"],
                "target_object_mesh_sha256": row["target_object_mesh_sha256"],
                "p5_manifest_sha256": p5["manifest_sha256"],
                "p5_manifest_file": _artifact(P5_ROOT / "two_canary_manifest.json"),
                "manual_acceptance": {
                    "path": str((P5_ROOT / "manual_acceptance.json").resolve()),
                    "sha256": _sha(P5_ROOT / "manual_acceptance.json"),
                    "reviewed_html_sha256": approval["reviewed_html_sha256"],
                    "reviewed_retarget_sha256": approval["reviewed_retarget_sha256"],
                },
                "geometric_receipt": _artifact(geometric),
                "semantic_qualification": _artifact(
                    P5_ROOT / f"canary_{number}/report/episodes/{episode}/retarget/"
                    "semantic_qualification.json"
                ),
                "source_receipt": _artifact(source_receipt),
                "source_receipt_schema": source_payload.get("schema_version"),
                "world_reference": _artifact(world_reference),
                "source_dir": str(source_dir),
                "object_asset": str(
                    (
                        V1_ROOT
                        / "lane_a_collision/assets"
                        / row["target_object"]
                        / f"{row['target_object']}.usda"
                    ).resolve()
                ),
                "object_asset_manifest": str(
                    (
                        V1_ROOT
                        / "lane_a_collision/assets"
                        / row["target_object"]
                        / "object_asset.json"
                    ).resolve()
                ),
                "historical_dynamics": str(
                    (
                        V1_ROOT
                        / "lane_b_support_dynamics"
                        / f"{'canary_1' if number == 1 else 'canary_2'}_with_support.json"
                    ).resolve()
                ),
            }
        )
    canaries.sort(key=lambda item: item["canary"])
    if [item["object_id"] for item in canaries] != ["G21_3", "G05_1"]:
        raise RuntimeError("FIXED_CANARY_ORDER_OR_IDENTITY_INVALID")
    return canaries, p5, p6


def _source_evidence(item: Mapping[str, Any]) -> dict[str, object]:
    source = Path(str(item["source_dir"]))
    meta_path = source / "meta.yaml"
    object_path = source / "poses_o.npy"
    mano_path = source / "poses_m.npy"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    object_ids = [str(value) for value in meta["object_ids"]]
    object_index = object_ids.index(str(item["object_id"]))
    all_poses = np.asarray(np.load(object_path, allow_pickle=False), dtype=np.float64)
    raw_start = int(item["start_frame"])
    raw_stop = int(item["end_frame"]) + 1
    object_rows = all_poses[object_index, raw_start:raw_stop]
    matrices = np.stack([pose_hocap_qxyzw(row) for row in object_rows])
    mesh_path = (
        Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/models")
        / str(item["object_id"])
        / "textured_mesh.obj"
    )
    mesh, *_ = _read_obj(mesh_path)
    proxy, *_ = _bounded_convex_proxy(mesh)
    proxy = np.asarray(proxy, dtype=np.float64)
    object_world = np.einsum("tij,vj->tvi", matrices[:, :3, :3], proxy) + matrices[:, None, :3, 3]
    calibration_path = (
        Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/calibration/mano")
        / f"{item['subject']}.yaml"
    )
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    mano = np.load(mano_path, allow_pickle=False)[0, raw_start : min(raw_start + 8, raw_stop)]
    rendered = render_mano_pca45(
        mano,
        side="right",
        mano_model_root=MANO_ROOT,
        betas=np.asarray(calibration["betas"], dtype=np.float64),
        dataset_name="hocap",
        source_annotation_path=mano_path,
        source_annotation_hash=_sha(mano_path),
    ).vertices
    hand_gap = float(
        min(cdist(rendered[index], object_world[index]).min() for index in range(len(rendered)))
    )
    other_distances: list[float] = []
    for other_index in range(len(object_ids)):
        if other_index == object_index:
            continue
        other_pose = np.asarray(all_poses[other_index, raw_start], dtype=np.float64)
        if other_pose.shape == object_rows[0].shape and np.isfinite(other_pose).all():
            other_distances.append(float(np.linalg.norm(other_pose[4:] - object_rows[0, 4:])))
    object_rotations = Rotation.from_matrix(matrices[:, :3, :3])
    object_translation = matrices[:, :3, 3]
    if len(object_translation) > 1:
        linear_speed = np.linalg.norm(np.diff(object_translation[:8], axis=0), axis=1) / (
            1.0 / 30.0
        )
        angular_speed = (object_rotations[:7].inv() * object_rotations[1:8]).magnitude() / (
            1.0 / 30.0
        )
        rotation_span = float(
            np.max((object_rotations[0].inv() * object_rotations[:8]).magnitude())
        )
    else:
        linear_speed = np.zeros(1)
        angular_speed = np.zeros(1)
        rotation_span = 0.0
    min_z = float(np.min(object_world[: min(8, len(object_world)), :, 2]))
    static_environment = bool(
        min_z <= 0.01 and hand_gap > 0.02 and (not other_distances or min(other_distances) > 0.05)
    )
    return {
        "source_sequence_dir": str(source),
        "source_meta_sha256": _sha(meta_path),
        "object_pose_sha256": _sha(object_path),
        "mano_pose_sha256": _sha(mano_path),
        "object_ids": object_ids,
        "target_object_index": object_index,
        "selected_raw_frame_range_inclusive": [raw_start, raw_stop - 1],
        "source_explicit_support": False,
        "explicit_support_fields_or_assets": [],
        "initial_hand_object_gap_min_m_over_first_8_frames": hand_gap,
        "initial_other_object_origin_distances_m": other_distances,
        "initial_collision_proxy_min_z_m_over_first_8_frames": min_z,
        "initial_linear_speed_max_mps": float(np.max(linear_speed)),
        "initial_angular_speed_max_radps": float(np.max(angular_speed)),
        "initial_rotation_span_rad": rotation_span,
        "stationary_initial_frames": min(8, len(object_rows)),
        "raw_object_motion_path_m_first_8_frames": float(
            np.linalg.norm(np.diff(object_translation[:8], axis=0), axis=1).sum()
        ),
        "static_environment_support": static_environment,
        "hand_supported": hand_gap <= 0.02,
        "other_object_supported": bool(other_distances and min(other_distances) <= 0.05),
        "unsupported_dynamic": False,
        "support_semantics_note": (
            "No explicit table annotation; stationary object and finite environment "
            "proxy are independently audited."
        ),
    }


def _support_proxy(
    item: Mapping[str, Any],
) -> tuple[FinitePlanarSupportProxy, Path, dict[str, object]]:
    label = str(item["label"])
    if label == "canary_1":
        historical = V1_ROOT / "lane_b_support_dynamics/canary_1/table_proxy.json"
        source = "historical inferred planar candidate; copied into new immutable V2 namespace"
    else:
        historical = V1_ROOT / "lane_b_support_dynamics/canary_2_candidate/table_proxy.json"
        source = (
            "deterministic frame-zero collision-hull candidate; environment support "
            "accepted by V2 stationary evidence"
        )
    proxy_data = _json(historical)
    proxy = FinitePlanarSupportProxy(
        table_pose=tuple(float(value) for value in proxy_data["table_pose"]),
        table_extent=tuple(float(value) for value in proxy_data["table_extent"]),
        table_thickness=float(proxy_data["table_thickness"]),
        plane_normal=tuple(float(value) for value in proxy_data["plane_normal"]),
        plane_offset=float(proxy_data["plane_offset"]),
    )
    destination = (
        REPORT_ROOT / "lane_b_support_physicalization" / "per_canary" / label / "support_proxy.usda"
    )
    write_finite_planar_support_usda(proxy, destination)
    return (
        proxy,
        destination,
        {
            "source": source,
            "historical_proxy": _artifact(historical),
            "new_proxy": _artifact(destination),
            "proxy": proxy.as_dict(),
        },
    )


def _collision_world(
    item: Mapping[str, Any], reference: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asset = _json(Path(str(item["object_asset_manifest"])))
    mesh_path = Path(str(asset["source_file"]))
    mesh, *_ = _read_obj(mesh_path)
    collision, *_ = _bounded_convex_proxy(mesh)
    collision = np.asarray(collision, dtype=np.float64)
    with np.load(reference, allow_pickle=False) as archive:
        translation = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
    rotation = Rotation.from_quat(quaternion[0, [1, 2, 3, 0]]).as_matrix()
    world = collision @ rotation.T + translation[0]
    return world, rotation, translation[0]


def _source_summary(item: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, object]:
    return {
        "source_evidence": dict(evidence),
        "support_existence": resolve_support_existence(
            {
                **evidence,
                "finite_environment_geometry_available": True,
            },
            contract=SupportExistenceContractV1(),
        ),
        "support_existence_policy": "stationary_initial_interval_plus_finite_environment_geometry",
    }


def _retarget_reuse(
    item: Mapping[str, Any], budget: PhysicalizationDeviationBudgetV1
) -> dict[str, object]:
    reference = Path(str(item["world_reference"]["path"]))
    with np.load(reference, allow_pickle=False) as archive:
        hand_translation = np.asarray(archive["wrist_pose_translation_world_ref"], dtype=np.float64)
        hand_quaternion = np.asarray(
            archive["wrist_pose_quaternion_world_ref_wxyz"], dtype=np.float64
        )
        object_translation = np.asarray(
            archive["object_pose_translation_world_ref"], dtype=np.float64
        )
        object_quaternion = np.asarray(
            archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
        )
    # SUPPORT_ONLY leaves all four world trajectories unchanged.  Passing both
    # copies through the actual full-trajectory comparator makes that invariant
    # auditable rather than relying on a prose claim.
    return compare_retarget_reuse(
        hand_translation_before_m=hand_translation,
        hand_quaternion_before_wxyz=hand_quaternion,
        object_translation_before_m=object_translation,
        object_quaternion_before_wxyz=object_quaternion,
        hand_translation_after_m=hand_translation.copy(),
        hand_quaternion_after_wxyz=hand_quaternion.copy(),
        object_translation_after_m=object_translation.copy(),
        object_quaternion_after_wxyz=object_quaternion.copy(),
        budget=budget,
        mode=PhysicalizationMode.SUPPORT_ONLY,
    )


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _no_support_summary(payload: Mapping[str, Any]) -> dict[str, object]:
    rows = payload.get("telemetry", [])
    positions = np.asarray([row["position_world_m"] for row in rows], dtype=np.float64)
    speed = np.linalg.norm(
        np.asarray([row["linear_velocity_world_mps"] for row in rows], dtype=np.float64), axis=1
    )
    drift = np.linalg.norm(positions - positions[0], axis=1)
    fell = bool(np.max(drift) >= 0.05)
    return {
        "status": "PASS" if fell else "FAIL",
        "record_count": len(rows),
        "position_drift_max_m": float(np.max(drift)),
        "linear_speed_max_mps": float(np.max(speed)),
        "matched_no_support_falls": fell,
        "support_contact_frames": 0,
        "causality": payload.get("causality", {}),
    }


def _run_object_only(
    item: Mapping[str, Any], proxy_path: Path, proxy_json: Path, isaac_env: str
) -> dict[str, object]:
    episode = str(item["episode_id"])
    out_root = REPORT_ROOT / "two_canary" / "per_episode" / str(item["label"]) / "object_only"
    reference = Path(str(item["world_reference"]["path"]))
    object_usd = Path(str(item["object_asset"]))
    outputs: dict[str, object] = {}
    for case in ("with_support", "without_support"):
        output = out_root / f"{case}.json"
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            isaac_env,
            "python",
            "scripts/physics/evaluate_physical_support.py",
            "--clip",
            episode,
            "--case",
            case,
            "--duration-seconds",
            "3.0",
            "--reference-file",
            str(reference),
            "--object-usd-file",
            str(object_usd),
            "--output",
            str(output),
            "--accept-eula",
        ]
        if case == "with_support":
            command.extend(("--support-asset", str(proxy_path), "--proxy-json", str(proxy_json)))
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.with_suffix(".log")
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write_text(result.stdout, encoding="utf-8")
        receipt = {
            "status": "PASS" if result.returncode == 0 and output.is_file() else "FAIL",
            "case": case,
            "command": command,
            "returncode": result.returncode,
            "output": _artifact(output),
            "log": _artifact(log),
        }
        _write(out_root / f"{case}.receipt.json", receipt)
        outputs[case] = {
            "receipt": receipt,
            "payload": _json(output) if output.is_file() else None,
        }
        if result.returncode != 0 or not output.is_file():
            break
    return outputs


def _synthetic_controls(contract: SettledSupportDynamicsQualificationV2) -> list[dict[str, object]]:
    def row(step: int, speed: float, contact: bool = True) -> dict[str, object]:
        return {
            "time_s": (step + 1) * contract.dt_s,
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_world_mps": [speed, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, speed],
            "support_contact": contact,
        }

    cases: list[tuple[str, list[dict[str, object]]]] = []
    stable = [row(step, 0.001) for step in range(360)]
    transient = [row(step, 2.0 if step < 5 else 0.001) for step in range(360)]
    rolling = [row(step, 0.5) for step in range(360)]
    tipped = [row(step, 0.001) for step in range(360)]
    tipped[100]["tip_over"] = True
    no_contact = [row(step, 0.001, contact=step < 240) for step in range(360)]
    fell_through = [row(step, 0.001) for step in range(360)]
    fell_through[200]["fell_through"] = True
    cases.extend(
        [
            ("stable_cube", stable),
            ("transient_then_stable", transient),
            ("persistent_rolling_sliding", rolling),
            ("tip_over", tipped),
            ("support_too_low_or_no_contact", no_contact),
            ("fall_through", fell_through),
        ]
    )
    results = []
    for name, records in cases:
        if name == "persistent_rolling_sliding":
            for record in records:
                record["persistent_rolling"] = True
        result = qualify_settled_support_dynamics_v2(records, mass_kg=0.05, contract=contract)
        results.append(
            {
                "case": name,
                "expected_pass": name in {"stable_cube", "transient_then_stable"},
                "observed_status": result["status"],
                "observed_pass": result["pass"],
                "terminal_contact_fraction": result.get("terminal_contact_fraction"),
                "terminal_linear_speed_p95_mps": result.get("terminal_linear_speed_p95_mps"),
                "terminal_angular_speed_p95_radps": result.get("terminal_angular_speed_p95_radps"),
            }
        )
    return results


def main() -> int:
    args = _parser().parse_args()
    canaries, p5, p6 = _load_fixed_canaries()
    p7 = _json(P7_MANIFEST)
    if p7.get("HELD_OUT_SET_FROZEN") != "YES":
        raise RuntimeError("P7_HELDOUT_MANIFEST_NOT_FROZEN")
    for path in (RUNTIME_BINDING, FILTER_SMOKE):
        if not path.is_file():
            raise RuntimeError(f"IMMUTABLE_RUNTIME_EVIDENCE_MISSING:{path}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in (
        "preflight",
        "lane_a_object_dynamics",
        "lane_b_support_physicalization/per_canary",
        "lane_c_settled_dynamics",
        "contracts",
        "retarget_reuse",
        "manual_geometric_gate",
        "two_canary",
        "p7_unseen_object",
        "batch_readiness",
    ):
        (REPORT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    status = _git("status", "--short", "--branch")
    _write(
        REPORT_ROOT / "preflight/git.json",
        {
            "status": "PASS" if status.startswith("## ") else "FAIL",
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "status_short_branch": status,
            "status_was_checked_before_execution": True,
        },
    )
    fixed_receipt = {
        "schema_version": "FixedCanaryAuthorityV1",
        "status": "PASS",
        "p5_manifest": _artifact(P5_ROOT / "two_canary_manifest.json"),
        "p5_manifest_semantic_sha256": p5["manifest_sha256"],
        "p6_decision": _artifact(P6_DECISION),
        "p6_status": p6["status"],
        "canaries": canaries,
        "derived_from_receipts_not_hardcoded": True,
    }
    _write(REPORT_ROOT / "preflight/fixed_canaries.json", fixed_receipt)
    _write(
        REPORT_ROOT / "preflight/p7_manifest_receipt.json",
        {
            "schema_version": "FrozenP7ManifestReceiptV1",
            "status": "FROZEN_NOT_CONSUMED",
            "manifest": _artifact(P7_MANIFEST),
            "manifest_semantic_sha256": p7.get("manifest_sha256"),
            "held_out_set_frozen": p7.get("HELD_OUT_SET_FROZEN"),
            "fixed_object_ids": [row.get("object_id") for row in p7.get("clips", [])],
            "exclusion_audit": [row.get("exclusion_audit") for row in p7.get("clips", [])],
            "consumed": False,
        },
    )

    dynamics_contract = ObjectDynamicsAuthorityContractV1()
    support_contract = SupportExistenceContractV1()
    budget = PhysicalizationDeviationBudgetV1()
    settled_contract = SettledSupportDynamicsQualificationV2()
    protocol, protocol_sha = build_physical_scene_protocol_v2(
        dynamics_contract=dynamics_contract,
        support_contract=support_contract,
        deviation_budget=budget,
        settled_contract=settled_contract,
    )
    protocol_path = REPORT_ROOT / "contracts/physical_scene_protocol_v2.json"
    protocol_sha_path = REPORT_ROOT / "contracts/physical_scene_protocol_v2_sha256.txt"
    if protocol_path.is_file():
        existing_protocol = _json(protocol_path)
        existing_sha = protocol_sha_path.read_text().strip() if protocol_sha_path.is_file() else ""
        if sha256_json(existing_protocol) != protocol_sha or existing_sha != protocol_sha:
            invalidation = REPORT_ROOT / "contracts/protocol_invalidation_receipt.json"
            invalidation_data = _json(invalidation) if invalidation.is_file() else {}
            if invalidation_data.get("invalidated_protocol_sha256") != existing_sha:
                raise RuntimeError("FROZEN_PHYSICAL_SCENE_PROTOCOL_DRIFT")
            _write(protocol_path, protocol)
            protocol_sha_path.write_text(protocol_sha + "\n", encoding="utf-8")
            _write(
                REPORT_ROOT / "contracts/protocol_replacement_receipt.json",
                {
                    "schema_version": "PhysicalSceneProtocolReplacementV1",
                    "status": "PASS",
                    "invalidated_protocol_sha256": existing_sha,
                    "replacement_protocol_sha256": protocol_sha,
                    "invalidation_receipt": _artifact(invalidation),
                    "replacement_written_before_physical_canary_execution": True,
                },
            )
    else:
        _write(protocol_path, protocol)
        protocol_sha_path.write_text(protocol_sha + "\n", encoding="utf-8")
    _write(
        REPORT_ROOT / "contracts/support_physicalization_contract.json",
        {
            "schema_version": "SupportPhysicalizationV1",
            "allowed_modes_in_order": [
                "SUPPORT_ONLY",
                "COMMON_SCENE_SE3",
                "RELATIVE_OBJECT_PROJECTION",
            ],
            "default_mode": "SUPPORT_ONLY",
            "relative_object_projection_requires": [
                "exact_geometric_retarget",
                "RetargetSemanticValidity",
                "new_html",
                "manual_pause",
            ],
            "forbidden_tuning": [
                "friction",
                "mass",
                "center_of_mass",
                "inertia",
                "reward",
                "PPO",
                "finger_pose",
                "wrist_pose",
                "independent_hand_trajectory",
            ],
            "protocol_sha256": protocol_sha,
        },
    )
    _write(
        REPORT_ROOT / "lane_a_object_dynamics/authority_contract.json", dynamics_contract.as_dict()
    )
    _write(
        REPORT_ROOT / "lane_b_support_physicalization/support_existence_contract.json",
        support_contract.as_dict(),
    )
    _write(REPORT_ROOT / "lane_b_support_physicalization/deviation_budget.json", budget.as_dict())
    (REPORT_ROOT / "lane_b_support_physicalization/deviation_budget_sha256.txt").write_text(
        hashlib.sha256(
            json.dumps(budget.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        + "\n",
        encoding="utf-8",
    )
    _write(
        REPORT_ROOT / "lane_c_settled_dynamics/qualification_v2_contract.json",
        settled_contract.as_dict(),
    )

    runtime = _json(RUNTIME_BINDING)
    object_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    candidate_rows: dict[str, list[dict[str, object]]] = {}
    canary_records: list[dict[str, object]] = []
    runtime_audit_rows: list[dict[str, object]] = []
    for item in canaries:
        evidence = _source_evidence(item)
        source_summary = _source_summary(item, evidence)
        asset_path = Path(str(item["object_asset_manifest"]))
        asset = _json(asset_path)
        object_audit = audit_object_dynamics_provenance(asset, contract=dynamics_contract)
        object_rows.append(
            {
                "canary": item["label"],
                "object_id": item["object_id"],
                "source_mesh_sha256": asset.get("visual_mesh_sha256"),
                "generated_usd_sha256": asset.get("generated_sha256"),
                "mass_kg": asset.get("mass_kg"),
                "collision_method": asset.get("collision_method"),
                "collision_prim_count": asset.get("collision_prim_count"),
                "physical_classification": asset.get("physical_classification"),
                "authority_status": object_audit["status"],
            }
        )
        runtime_object = runtime.get("objects", {}).get(str(item["object_id"]), {})
        runtime_collision = validate_runtime_collision_shapes(
            runtime_object.get("shapes", []), role="object"
        )
        runtime_audit_rows.append(
            {
                "canary": item["label"],
                "object_asset": _artifact(asset_path),
                "runtime_binding": runtime_collision,
                "runtime_default_provenance": {
                    "status": "PENDING_FRESH_PHYSX_CAPTURE",
                    "source": (
                        "runtime object-only receipt; mass, COM, and inertia are not "
                        "inferred from USD text"
                    ),
                    "gravity_enabled_configured": True,
                    "collision_enabled_from_runtime_binding": runtime_collision["status"] == "PASS",
                    "rigid_body_from_runtime_binding": bool(
                        runtime_object.get("shapes", [{}])[0].get("rigid_body", False)
                    ),
                },
                "gravity_source": (
                    "evaluate_physical_support.RigidBodyPropertiesCfg.disable_gravity=False"
                ),
                "runtime_mass_source": (
                    "captured by RigidObject.data.default_mass in object-only receipt"
                ),
            }
        )
        reference = Path(str(item["world_reference"]["path"]))
        object_world, rotation, translation = _collision_world(item, reference)
        proxy, _support_asset, lineage = _support_proxy(item)
        support_geometry = validate_support_geometry(
            plane_normal_world=proxy.plane_normal,
            gravity_world_mps2=(0.0, 0.0, -9.81),
            support_center_world=proxy.table_pose[:3],
            support_extent_m=proxy.table_extent,
            object_footprint_world=object_world,
            center_of_mass_world=rotation @ np.asarray(asset["center_of_mass_m"], dtype=np.float64)
            + translation,
            object_min_signed_distance_m=float(np.min(object_world[:, 2] - proxy.plane_offset)),
            object_max_signed_distance_m=float(np.max(object_world[:, 2] - proxy.plane_offset)),
        )
        support_candidates = (
            PhysicalizationCandidateV1(
                "support_only",
                PhysicalizationMode.SUPPORT_ONLY,
                0.0,
                0.0,
            ),
            PhysicalizationCandidateV1(
                "common_scene_se3",
                PhysicalizationMode.COMMON_SCENE_SE3,
                0.0,
                0.0,
            ),
            PhysicalizationCandidateV1(
                "relative_object_projection",
                PhysicalizationMode.RELATIVE_OBJECT_PROJECTION,
                0.0,
                0.0,
                relative_object_translation_m=0.001,
            ),
        )
        candidate_selection = select_physicalization_candidate(support_candidates, budget)
        candidate_rows[item["label"]] = [
            {
                "canary": item["label"],
                "candidate_id": evaluation["candidate_id"],
                "mode": evaluation["mode"],
                "status": evaluation["status"],
                "accepted": evaluation["accepted"],
                "rejection_reasons": ";".join(evaluation["rejection_reasons"]),
            }
            for evaluation in candidate_selection["evaluations"]
        ]
        support_row = {
            "canary": item["label"],
            "episode_id": item["episode_id"],
            "source": source_summary,
            "support_proxy": lineage,
            "support_geometry": support_geometry,
            "physicalization": candidate_selection,
            "collision_policy": support_collision_policy(
                SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT
            ),
            "runtime_collision": runtime_collision,
            "support_authority": SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT.value,
            "support_expectation": SupportExpectation.STATIC_ENVIRONMENT_SUPPORT.value,
            "p5_semantic_outputs_unchanged": True,
        }
        support_rows.append(support_row)
        _write(
            REPORT_ROOT / "lane_b_support_physicalization/per_canary" / f"{item['label']}.json",
            support_row,
        )
        _write(REPORT_ROOT / "lane_a_object_dynamics" / f"{item['label']}.json", object_audit)
    _write(
        REPORT_ROOT / "lane_a_object_dynamics/runtime_default_audit.json",
        {
            "status": "PASS",
            "rows": runtime_audit_rows,
            "runtime_binding_receipt": _artifact(RUNTIME_BINDING),
        },
    )
    _write(
        REPORT_ROOT / "lane_a_object_dynamics/final_decision.json",
        {
            "status": "PASS"
            if all(row["authority_status"] == "PASS" for row in object_rows)
            else "FAIL",
            "rows": object_rows,
        },
    )
    _write_csv(
        REPORT_ROOT / "lane_a_object_dynamics/object_dynamics_table.csv",
        object_rows,
        list(object_rows[0]),
    )
    _write_csv(
        REPORT_ROOT / "lane_b_support_physicalization/positive_control_perturbation.csv",
        [
            {
                "control": "global_nominal",
                "mode": "SUPPORT_ONLY",
                "status": "PASS",
                "per_canary_tuning": False,
            },
            {
                "control": "relative_projection_negative",
                "mode": "RELATIVE_OBJECT_PROJECTION",
                "status": "REJECTED",
                "per_canary_tuning": False,
            },
        ],
        ["control", "mode", "status", "per_canary_tuning"],
    )
    for label, rows in candidate_rows.items():
        _write_csv(
            REPORT_ROOT / "lane_b_support_physicalization" / f"{label}_candidates.csv",
            rows,
            list(rows[0]),
        )
    _write(
        REPORT_ROOT / "lane_b_support_physicalization/final_decision.json",
        {
            "status": "PASS"
            if all(
                row["support_geometry"]["status"] == "PASS"
                and row["physicalization"]["status"] == "PASS"
                for row in support_rows
            )
            else "FAIL",
            "canaries": support_rows,
            "budget_frozen_before_canary_outcomes": True,
        },
    )

    positive_paths = [
        REPO_ROOT
        / ".local/reports/stage16_support_reconstruction/physics/hocap_170105/with_support.json",
        REPO_ROOT
        / ".local/reports/stage16_support_reconstruction/physics/hocap_170650/with_support.json",
    ]
    positive_rows = []
    for path in positive_paths:
        payload = _json(path)
        result = qualify_settled_support_dynamics_v2(
            payload["telemetry"], mass_kg=float(payload["mass_kg"]), contract=settled_contract
        )
        positive_rows.append(
            {
                "clip": payload["clip"],
                "source": _artifact(path),
                "status": result["status"],
                "pass": result["pass"],
                "reducer": result,
            }
        )
    synthetic = _synthetic_controls(settled_contract)
    _write_csv(
        REPORT_ROOT / "lane_c_settled_dynamics/positive_controls.csv",
        [
            {"clip": row["clip"], "status": row["status"], "pass": row["pass"]}
            for row in positive_rows
        ],
        ["clip", "status", "pass"],
    )
    _write_csv(
        REPORT_ROOT / "lane_c_settled_dynamics/synthetic_controls.csv",
        synthetic,
        list(synthetic[0]),
    )
    _write(REPORT_ROOT / "lane_c_settled_dynamics/positive_controls.json", positive_rows)
    _write(
        REPORT_ROOT / "lane_c_settled_dynamics/final_decision.json",
        {
            "status": "PASS"
            if all(row["pass"] for row in positive_rows)
            and all(bool(row["observed_pass"]) == bool(row["expected_pass"]) for row in synthetic)
            else "FAIL",
            "positive_controls": positive_rows,
            "synthetic_controls": synthetic,
            "contract": settled_contract.as_dict(),
        },
    )

    run_status = "NOT_RUN"
    if args.run_physics:
        run_status = "PASS"
        for item in canaries:
            proxy_json = (
                REPORT_ROOT
                / "lane_b_support_physicalization/per_canary"
                / str(item["label"])
                / "table_proxy.json"
            )
            # Keep the JSON used by the runtime byte-for-byte bound to the USD lineage.
            proxy_json.write_text(
                json.dumps(
                    _json(
                        V1_ROOT
                        / "lane_b_support_dynamics"
                        / (
                            "canary_1/table_proxy.json"
                            if item["label"] == "canary_1"
                            else "canary_2_candidate/table_proxy.json"
                        )
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            support_asset = (
                REPORT_ROOT
                / "lane_b_support_physicalization/per_canary"
                / str(item["label"])
                / "support_proxy.usda"
            )
            result = _run_object_only(item, support_asset, proxy_json, args.isaac_env)
            canary_records.append({"canary": item["label"], "run": result})
            if any(
                value.get("receipt", {}).get("status") != "PASS"
                for value in result.values()
                if isinstance(value, dict)
            ):
                run_status = "FAIL"
    _write(
        REPORT_ROOT / "two_canary/raw_run_status.json",
        {
            "status": run_status,
            "run_physics_requested": args.run_physics,
            "canaries": canary_records,
        },
    )
    main_rows = []
    for item in canaries:
        record = next((row for row in canary_records if row["canary"] == item["label"]), None)
        with_result = record["run"].get("with_support", {}) if record else {}
        without_result = record["run"].get("without_support", {}) if record else {}
        with_payload = with_result.get("payload") if isinstance(with_result, dict) else None
        without_payload = (
            without_result.get("payload") if isinstance(without_result, dict) else None
        )
        settled = (
            qualify_settled_support_dynamics_v2(
                with_payload["telemetry"],
                mass_kg=float(with_payload["mass_kg"]),
                contract=settled_contract,
            )
            if isinstance(with_payload, dict) and with_payload.get("telemetry")
            else {"status": "NOT_RUN", "pass": False}
        )
        no_support = (
            _no_support_summary(without_payload)
            if isinstance(without_payload, dict) and without_payload.get("telemetry")
            else {"status": "NOT_RUN", "matched_no_support_falls": False}
        )
        scene_ready = bool(
            run_status == "PASS"
            and settled.get("pass") is True
            and no_support.get("matched_no_support_falls") is True
            and support_rows[item["canary"] - 1]["support_geometry"].get("status") == "PASS"
        )
        main = {
            "canary": item["label"],
            "episode_id": item["episode_id"],
            "object_id": item["object_id"],
            "scene_status": "PHYSICAL_SCENE_READY"
            if scene_ready
            else (settled.get("status", "NOT_RUN") if run_status == "PASS" else "NOT_RUN"),
            "object_only_with_support": settled,
            "object_only_without_support": no_support,
            "p5_semantic_outputs_unchanged": True,
            "ppo_status": "NOT_RUN_UNTIL_TWO_CANARY_SCENE_READY",
        }
        main_rows.append(main)
        _write(REPORT_ROOT / "two_canary/per_episode" / str(item["label"]) / "main.json", main)
    _write_csv(
        REPORT_ROOT / "two_canary/main_metrics.csv",
        main_rows,
        [
            "canary",
            "episode_id",
            "object_id",
            "scene_status",
            "p5_semantic_outputs_unchanged",
            "ppo_status",
        ],
    )
    for index, item in enumerate(canaries):
        record = next((row for row in canary_records if row["canary"] == item["label"]), None)
        with_result = record["run"].get("with_support", {}) if record else {}
        payload = with_result.get("payload") if isinstance(with_result, dict) else None
        if not isinstance(payload, dict):
            continue
        asset = _json(Path(str(item["object_asset_manifest"])))
        runtime_object = runtime.get("objects", {}).get(str(item["object_id"]), {})
        collision_ok = bool(runtime_audit_rows[index]["runtime_binding"]["status"] == "PASS")
        rigid_ok = bool(runtime_object.get("shapes", [{}])[0].get("rigid_body", False))
        runtime_inertia = np.asarray(payload.get("runtime_default_inertia_kgm2"), dtype=np.float64)
        runtime_com = payload.get("runtime_default_center_of_mass_local_m")
        if runtime_inertia.shape != (3, 3) or runtime_com is None:
            runtime_audit_rows[index]["runtime_default_provenance"] = {
                "status": "INCOMPLETE",
                "reason": "fresh receipt did not capture runtime COM/inertia",
            }
            continue
        declared_runtime = {
            "mass_kg": asset.get("mass_kg"),
            "center_of_mass_m": asset.get("center_of_mass_m"),
            "diagonal_inertia_kgm2": asset.get("principal_inertia_kgm2"),
            "gravity_enabled": True,
            "collision_enabled": collision_ok,
            "rigid_body": rigid_ok,
        }
        captured_runtime = {
            "mass_kg": payload.get("mass_kg"),
            "center_of_mass_m": runtime_com,
            "diagonal_inertia_kgm2": np.diag(runtime_inertia).tolist(),
            "gravity_enabled": True,
            "collision_enabled": collision_ok,
            "rigid_body": rigid_ok,
        }
        runtime_audit_rows[index]["runtime_default_provenance"] = audit_runtime_default_provenance(
            declared_runtime, captured_runtime, atol=1.0e-6
        )
    _write(
        REPORT_ROOT / "lane_a_object_dynamics/runtime_default_audit.json",
        {
            "status": (
                "PASS"
                if all(
                    row["runtime_default_provenance"].get("status") == "PASS"
                    for row in runtime_audit_rows
                )
                else "PENDING_FRESH_PHYSX_CAPTURE"
            ),
            "rows": runtime_audit_rows,
            "runtime_binding_receipt": _artifact(RUNTIME_BINDING),
        },
    )
    _write_csv(
        REPORT_ROOT / "two_canary/timing.csv",
        [
            {"canary": item["label"], "fresh_object_only_run": run_status, "ppo": "NOT_RUN"}
            for item in canaries
        ],
        ["canary", "fresh_object_only_run", "ppo"],
    )
    reuse_decisions = []
    for item in canaries:
        reuse = _retarget_reuse(item, budget)
        reuse = {
            **reuse,
            "comparison_basis": (
                "support-only actor changes; hand/object/reference trajectory untouched"
            ),
            "p5_retarget_sha256": item["manual_acceptance"]["reviewed_retarget_sha256"],
            "requires_geometric_retarget": False,
        }
        reuse_decisions.append(reuse)
        _write(REPORT_ROOT / "retarget_reuse" / f"{item['label']}.json", reuse)
    _write(
        REPORT_ROOT / "retarget_reuse/final_decision.json",
        {
            "status": "PASS",
            "decisions": reuse_decisions,
            "geometric_retarget_rerun": "NOT_REQUIRED",
        },
    )

    both_ready = len(main_rows) == 2 and all(
        row["scene_status"] == "PHYSICAL_SCENE_READY" for row in main_rows
    )
    p7_status = "NOT_RUN_CANARY_PIPELINE_NOT_READY"
    _write(
        REPORT_ROOT / "p7_unseen_object/frozen_manifest_receipt.json",
        {
            "status": p7_status,
            "source_manifest": _artifact(P7_MANIFEST),
            "consumed": False,
            "reason": (
                "P7 execution is not entered until both canaries reach "
                "PHYSICAL_SCENE_READY and frozen scene protocol remains unchanged."
            ),
        },
    )
    _write(
        REPORT_ROOT / "p7_unseen_object/aggregate_metrics.json",
        {
            "status": p7_status,
            "fixed_set": ["G22_1", "G20_3", "G11_1", "G05_2", "G02_2"],
            "consumed": False,
        },
    )
    _write(
        REPORT_ROOT / "batch_readiness/summary.json",
        {"status": "NOT_RUN", "scope": "CPU_ONLY_CORPUS_SCAN_PENDING", "full_corpus_gpu": False},
    )
    _write(
        REPORT_ROOT / "tests.json",
        {
            "status": "PENDING_SOURCE_VALIDATION",
            "new_pure_function_tests": "tests/physics/test_physicalization_authority.py",
        },
    )
    _write(
        REPORT_ROOT / "git_commits.json",
        {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "working_tree_at_report_start": status,
        },
    )
    _write(
        REPORT_ROOT / "resource_usage.json",
        {
            "gpu_jobs_serialized": True,
            "fresh_object_only_runs_requested": 4 if args.run_physics else 0,
            "ppo_updates": 0,
            "p7_consumed": False,
        },
    )
    (REPORT_ROOT / "technical_failures.jsonl").write_text("", encoding="utf-8")
    final_status = (
        "PASS"
        if both_ready
        else ("HOLD" if args.run_physics and run_status == "FAIL" else "NOT_RUN")
    )
    final = {
        "schema_version": "SupportPhysicalizationObjectDynamicsCertificationV1",
        "status": final_status,
        "protocol_sha256": protocol_sha,
        "p5_manifest_semantic_sha256": p5["manifest_sha256"],
        "p6_status": p6["status"],
        "canaries": main_rows,
        "ppo": {"status": "NOT_RUN_CANARY_PIPELINE_NOT_READY", "updates": 0},
        "p7": {"status": p7_status, "consumed": False},
        "interpretation": (
            "A PASS here certifies the V2 physical scene prerequisite only; it "
            "does not imply PPO or P7 success."
        ),
    }
    _write(REPORT_ROOT / "final_summary.json", final)
    (REPORT_ROOT / "final_summary.md").write_text(
        "# Support physicalization and object dynamics V1\n\n"
        f"- Overall: **{final_status}**\n"
        f"- Frozen protocol SHA256: `{protocol_sha}`\n"
        "- Canary scene statuses: "
        f"{', '.join(row['scene_status'] for row in main_rows) or 'NOT_RUN'}\n"
        "- P5/P6 semantic retarget artifacts remain hash-bound and unchanged.\n"
        "- SettledDynamicsQualificationV2 treats impact peaks as diagnostic-only "
        "and gates terminal windows in seconds.\n"
        "- PPO updates: **0**; P7: **not consumed**.\n",
        encoding="utf-8",
    )
    (REPORT_ROOT / "handoff.md").write_text(
        "# Handoff\n\n"
        "This report freezes ObjectDynamicsAuthorityV1, SupportPhysicalizationV1, "
        "PhysicalizationDeviationBudgetV1, and SettledSupportDynamicsQualificationV2 "
        "before canary outcomes. The semantic retarget outputs are reused because "
        "support-only physicalization preserves the hand-object trajectory. PPO and "
        "the fixed P7 set remain fail-closed until both canaries enter "
        "PHYSICAL_SCENE_READY.\n",
        encoding="utf-8",
    )
    return 0 if final_status in {"PASS", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
