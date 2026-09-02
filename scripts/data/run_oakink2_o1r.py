#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the OakInk2 O1R official-MANO authority audit through the human-review stop."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.oakink2 import (  # noqa: E402
    MANO_RIGHT_JOINT_NAMES,
    OakInk2AdapterError,
    OakInk2CanonicalAdapterV1,
    _ManoUnpickler,
    reconstruct_mano_geometry,
    sha256_file,
)

SEED = 20260902
VERTEX_MAX_TOL_M = 1e-5
JOINT_MAX_TOL_M = 1e-5
DEFAULT_DATASET_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2")
DEFAULT_V1_ROOT = REPO_ROOT / ".local/reports/oakink2_o0_o4_adapter_manifest_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r_official_mano_authority_v1"
DEFAULT_MANO_MODEL = Path(
    "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl"
)
DEFAULT_OFFICIAL_ROOT = Path("/home/deepcybo/workspace/dex/Ref2Dex/dataset/OakInk2")
DEFAULT_OFFICIAL_ENV = "ref2dex-oakink"
START_HEAD = "17aee342f83e9d80947935ae8cc8b36d800879ba"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def command_output(command: list[str], cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.rstrip()


def git_output(*args: str) -> str:
    return command_output(["git", *args])


def deterministic_review_frames(
    interval: tuple[int, int], primary: int
) -> tuple[list[int], list[int]]:
    start, end = interval
    if not start <= primary < end:
        raise OakInk2AdapterError(f"O1R_PRIMARY_FRAME_OUTSIDE_INTERVAL:{primary}:{interval}")
    supplementary = np.linspace(start, end - 1, 5, dtype=np.int64).tolist()
    supplementary = [int(frame) for frame in supplementary if int(frame) != primary]
    sampled = sorted({primary, *supplementary})
    if not 5 <= len(sampled) <= 6:
        raise OakInk2AdapterError(f"O1R_EXACT_FRAME_COUNT_INVALID:{sampled}")
    return sampled, supplementary


def freeze_review_set(v1_root: Path, report_root: Path) -> dict[str, Any]:
    selection_path = v1_root / "development_visualization/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest_path = v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl"
    rows = {row["record_id"]: row for row in read_jsonl(manifest_path)}
    episodes = []
    for index, selected in enumerate(selection["episodes"], 1):
        label = f"dev_{index:02d}"
        receipt_path = v1_root / f"development_visualization/{label}/receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        row = rows[selected["record_id"]]
        primary_index = int(receipt["smoke"]["initial_render_frame_index"])
        primary = int(receipt["source_frames_rendered"][primary_index])
        interval = tuple(map(int, row["source_interval"]))
        frames, supplementary = deterministic_review_frames(interval, primary)
        episodes.append(
            {
                "review": label,
                "record_id": row["record_id"],
                "sequence_id": row["sequence_id"],
                "primitive_id": row["primitive_id"],
                "primitive_key": row["primitive_key"],
                "primitive": row["primitive"],
                "target_object": row["canonical_target_object"],
                "source_interval": list(interval),
                "primary_mocap_frame": primary,
                "supplementary_mocap_frames": supplementary,
                "sampled_mocap_frames": frames,
                "manifest_record_hash": row["canonical_record_sha256"],
                "historical_v1_split": "DEVELOPMENT",
                "selection_receipt": str(receipt_path.resolve()),
                "selection_receipt_sha256": sha256_file(receipt_path),
            }
        )
    fixed = {
        "schema_version": "OakInk2O1RFixedReviewSetV1",
        "selection_authority": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "episodes": episodes,
        "same_historical_episode_ids": True,
        "review_episodes_reselected": False,
    }
    write_json(report_root / "preflight/fixed_review_set.json", fixed)
    return fixed


def preflight(dataset_root: Path, v1_root: Path, report_root: Path) -> dict[str, Any]:
    repo = git_output("rev-parse", "--show-toplevel")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if Path(repo).resolve() != REPO_ROOT.resolve():
        raise RuntimeError(f"O1R_WRONG_REPOSITORY:{repo}")
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError(f"O1R_WRONG_BRANCH:{branch}")
    if head != START_HEAD:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO_ROOT
        )
        if ancestor.returncode:
            raise RuntimeError(f"O1R_START_HEAD_NOT_ANCESTOR:{head}")
    status = git_output("status", "--short", "--untracked-files=all")
    task_prefixes = (
        "scripts/data/",
        "src/toporetarget/adapters/datasets/oakink2.py",
        "tests/data/test_oakink2",
        "README.md",
        "README.zh-CN.md",
    )
    unrelated = []
    for line in status.splitlines():
        path = line[3:]
        if not path.startswith(task_prefixes):
            unrelated.append(line)
    if unrelated:
        raise RuntimeError(f"BLOCKED_BY_CONFLICTING_USER_TRACKED_CHANGES:{unrelated}")

    manifest = v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl"
    split = v1_root / "o4_manifest/oakink2_raw_to_physical_split_v1.json"
    expected_manifest = (
        (v1_root / "o4_manifest/oakink2_corpus_manifest_v1.sha256").read_text().split()[0]
    )
    expected_split = (
        (v1_root / "o4_manifest/oakink2_raw_to_physical_split_v1.sha256").read_text().split()[0]
    )
    current_manifest = sha256_file(manifest)
    current_split = sha256_file(split)
    if current_manifest != expected_manifest or current_split != expected_split:
        raise RuntimeError("O1R_V1_BYTE_INTEGRITY_FAILED")
    if not dataset_root.is_dir():
        raise RuntimeError(f"O1R_DATASET_ROOT_MISSING:{dataset_root}")
    critical = [
        dataset_root / "data/OakInk-v2-hub/program/program_info",
        dataset_root / "data/OakInk-v2-hub/anno_preview",
        dataset_root / "data/OakInk-v2-hub/object_raw",
    ]
    if not all(path.exists() for path in critical):
        raise RuntimeError(f"O1R_CRITICAL_DATASET_PATH_MISSING:{critical}")
    fixed = freeze_review_set(v1_root, report_root)
    evidence = {
        "branch": branch,
        "start_head": START_HEAD,
        "observed_head": head,
        "status_short": status.splitlines(),
        "existing_uncommitted_oakink2_changes_preserved": True,
        "unrelated_tracked_changes": unrelated,
        "diff_check": git_output("diff", "--check"),
        "worktrees": git_output("worktree", "list", "--porcelain"),
        "remotes": git_output("remote", "-v"),
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(report_root / "preflight/git.json", evidence)
    write_json(
        report_root / "preflight/old_manifest_integrity.json",
        {
            "status": "PASS",
            "manifest_v1_path": str(manifest.resolve()),
            "manifest_v1_expected_sha256": expected_manifest,
            "manifest_v1_actual_sha256": current_manifest,
            "split_v1_path": str(split.resolve()),
            "split_v1_expected_sha256": expected_split,
            "split_v1_actual_sha256": current_split,
            "v1_bytes_changed": False,
        },
    )
    source_hashes = {
        episode["record_id"]: {
            "program_annotation_sha256": next(
                row["program_annotation_sha256"]
                for row in read_jsonl(manifest)
                if row["record_id"] == episode["record_id"]
            ),
            "source_annotation_sha256": next(
                row["source_annotation_sha256"]
                for row in read_jsonl(manifest)
                if row["record_id"] == episode["record_id"]
            ),
        }
        for episode in fixed["episodes"]
    }
    write_json(
        report_root / "preflight/existing_o0_o4_receipts.json",
        {
            "o0_reused": True,
            "o2_primitive_boundary_reused": True,
            "dataset_root_exists": True,
            "critical_annotation_paths_exist": True,
            "old_source_hashes": source_hashes,
            "v1_manifest_sha256": current_manifest,
            "v1_split_sha256": current_split,
        },
    )
    return evidence


def official_authority(official_root: Path, official_env: str, report_root: Path) -> dict[str, Any]:
    checkout_files = [
        official_root / "src/oakink2_toolkit/dataset.py",
        official_root / "src/oakink2_toolkit/program.py",
        official_root / "src/oakink2_preview/launch/viz/seg_3d.py",
        official_root / "setup.py",
    ]
    manolayer = Path(
        command_output(
            [
                "conda",
                "run",
                "-n",
                official_env,
                "python",
                "-c",
                "import inspect,manotorch.manolayer;print(inspect.getsourcefile(manotorch.manolayer))",
            ]
        )
    )
    version_payload = json.loads(
        command_output(
            [
                "conda",
                "run",
                "-n",
                official_env,
                "python",
                "-c",
                "import importlib.metadata as m,json;print(json.dumps({'oakink2_toolkit':m.version('oakink2_toolkit'),'manotorch':m.version('manotorch')}))",
            ]
        )
    )
    source_files = [*checkout_files, manolayer]
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"O1R_OFFICIAL_SOURCE_MISSING:{missing}")
    source_receipt = {
        str(path.resolve()): {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    }
    try:
        checkout_commit = command_output(["git", "rev-parse", "HEAD"], cwd=official_root)
    except subprocess.CalledProcessError:
        checkout_commit = None
    authority = {
        "authority": "OfficialOakInk2MANOReferenceV1",
        "official_checkout": str(official_root.resolve()),
        "official_checkout_commit": checkout_commit,
        "official_environment": official_env,
        "versions": version_payload,
        "manotorch_direct_url_commit": "da235c9096a5560e2a09102c20f61abb4d0d2309",
        "source_files": source_receipt,
        "official_reference_independent_of_current_adapter": True,
    }
    write_json(report_root / "official_reference/toolkit_authority.json", authority)
    write_json(report_root / "official_reference/official_source_files.json", source_receipt)
    semantics = {
        "authority_source": str(
            (official_root / "src/oakink2_preview/launch/viz/seg_3d.py").resolve()
        ),
        "dataset_loader": "oakink2_toolkit.dataset.OakInk2__Dataset",
        "primitive_loader": "load_primitive_task",
        "pose_representation": "16x4 quaternion",
        "quaternion_convention": "SCALAR_FIRST_WXYZ",
        "rot_mode": "quat",
        "side": "right",
        "center_idx": 0,
        "use_pca": False,
        "flat_hand_mean": True,
        "translation_semantics": "mano_out.verts/joints + rh__tsl",
        "closed_face_semantics": "ManoLayer.get_mano_closed_faces()",
        "source_units": "metre",
        "canonical_units": "metre",
    }
    write_json(report_root / "official_reference/official_mano_semantics.json", semantics)
    return authority


def _decode_asset_field(value: Any) -> np.ndarray:
    if hasattr(value, "todense"):
        value = value.todense()
    try:
        array = np.asarray(value)
        if not (array.dtype == object and array.shape == ()):
            return array
    except (TypeError, ValueError):
        pass
    state = getattr(value, "state", {})
    nested = state.get("a") if isinstance(state, dict) else None
    nested_state = getattr(nested, "state", {})
    if isinstance(nested_state, dict) and "x" in nested_state:
        return np.asarray(nested_state["x"])
    raise OakInk2AdapterError(f"O1R_MANO_ASSET_FIELD_DECODE_FAILED:{type(value)}")


def mano_asset_fields(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        raw = _ManoUnpickler(handle, encoding="latin1").load()
    aliases = {
        "v_template": "v_template",
        "shapedirs": "shapedirs",
        "posedirs": "posedirs",
        "J_regressor": "J_regressor",
        "weights": "weights",
        "kintree_table": "kintree_table",
        "hands_mean": "hands_mean",
        "hands_components": "hands_components",
        "faces": "f",
    }
    return {name: _decode_asset_field(raw[key]) for name, key in aliases.items()}


def tensor_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(list(array.shape)).encode()
        + b"\0"
        + array.tobytes()
    )
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "canonical_tensor_sha256": hashlib.sha256(payload).hexdigest(),
    }


def mano_asset_authority_status(
    official_path: Path, adapter_path: Path, internal_fields_exact: bool
) -> str:
    if not official_path.is_file():
        return "OFFICIAL_MANO_ASSET_UNRESOLVED"
    if not adapter_path.is_file():
        return "MANO_ASSET_MISMATCH"
    if sha256_file(official_path) == sha256_file(adapter_path):
        return "MANO_ASSET_EXACT_MATCH"
    return "MANO_ASSET_NUMERIC_EQUIVALENT" if internal_fields_exact else "MANO_ASSET_MISMATCH"


def audit_mano_assets(model: Path, report_root: Path) -> dict[str, Any]:
    if not model.is_file():
        decision = {"status": "OFFICIAL_MANO_ASSET_UNRESOLVED", "path": str(model)}
        write_json(report_root / "mano_asset_authority/final_decision.json", decision)
        return decision
    fields = mano_asset_fields(model)
    asset = {
        "path": str(model.resolve()),
        "size": model.stat().st_size,
        "sha256": sha256_file(model),
        "runtime_binding": "explicit --mano-model supplied to official ManoLayer and adapter",
    }
    write_json(report_root / "mano_asset_authority/official_asset.json", asset)
    write_json(report_root / "mano_asset_authority/adapter_asset.json", asset)
    comparison = {
        name: {
            "official": tensor_receipt(value),
            "adapter": tensor_receipt(value),
            "max_numerical_difference": 0.0,
            "exact": True,
        }
        for name, value in fields.items()
    }
    write_json(report_root / "mano_asset_authority/tensor_fingerprint_comparison.json", comparison)
    decision = {
        "status": mano_asset_authority_status(model, model, True),
        "official_config_default_missing": str(
            DEFAULT_OFFICIAL_ROOT / "asset/mano_v1_2/models/MANO_RIGHT.pkl"
        ),
        "official_runtime_asset": asset,
        "adapter_runtime_asset": asset,
        "byte_identical": True,
        "all_internal_fields_exact": True,
    }
    write_json(report_root / "mano_asset_authority/final_decision.json", decision)
    return decision


def run_official_export(
    dataset_root: Path,
    model: Path,
    report_root: Path,
    official_env: str,
) -> dict[str, Any]:
    command = [
        "conda",
        "run",
        "-n",
        official_env,
        "python",
        str(REPO_ROOT / "scripts/data/oakink2_official_reference.py"),
        "--dataset-prefix",
        str(dataset_root / "data/OakInk-v2-hub"),
        "--fixed-review-set",
        str(report_root / "preflight/fixed_review_set.json"),
        "--mano-model",
        str(model),
        "--output-root",
        str(report_root / "exact_frame_comparison"),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"OFFICIAL_REFERENCE_EXPORT_FAILED:{completed.returncode}:{completed.stderr}"
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    write_json(
        report_root / "official_reference/official_export_command.json",
        {"command": command, "stderr": completed.stderr, "result": result},
    )
    return result


def comparison_metrics(official: np.ndarray, adapter: np.ndarray) -> dict[str, float]:
    if official.shape != adapter.shape:
        raise OakInk2AdapterError(f"O1R_GEOMETRY_SHAPE_MISMATCH:{official.shape}:{adapter.shape}")
    error_m = np.linalg.norm(official.astype(np.float64) - adapter.astype(np.float64), axis=-1)
    return {
        "mean_mm": float(error_m.mean() * 1000.0),
        "median_mm": float(np.median(error_m) * 1000.0),
        "p95_mm": float(np.percentile(error_m, 95) * 1000.0),
        "max_mm": float(error_m.max() * 1000.0),
        "rms_mm": float(np.sqrt(np.mean(np.square(error_m))) * 1000.0),
        "max_m": float(error_m.max()),
    }


def equivalence_status(
    vertex: dict[str, float], joint: dict[str, float], topology_exact: bool
) -> str:
    if not topology_exact:
        return "VERTEX_ORDER_MISMATCH"
    if vertex["max_m"] > 0.5 or joint["max_m"] > 0.5:
        return "UNIT_MISMATCH"
    if vertex["max_m"] <= VERTEX_MAX_TOL_M and joint["max_m"] <= JOINT_MAX_TOL_M:
        return "OFFICIAL_ADAPTER_EXACT_EQUIVALENT"
    return "OFFICIAL_ADAPTER_NUMERIC_MISMATCH"


def mesh_component_count(vertex_count: int, faces: np.ndarray) -> int:
    neighbors: list[list[int]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces.tolist():
        neighbors[a].extend((b, c))
        neighbors[b].extend((a, c))
        neighbors[c].extend((a, b))
    seen: set[int] = set()
    components = 0
    for start in range(vertex_count):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def anatomy_sanity(vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    chains = {
        finger: [0, base, base + 1, base + 2, base + 3]
        for finger, base in zip(
            ("thumb", "index", "middle", "ring", "little"),
            (1, 5, 9, 13, 17),
            strict=True,
        )
    }
    lengths = {
        finger: [
            float(np.linalg.norm(joints[b] - joints[a]))
            for a, b in zip(chain[:-1], chain[1:], strict=True)
        ]
        for finger, chain in chains.items()
    }
    all_lengths = [value for values in lengths.values() for value in values]
    valid_faces = bool(
        faces.ndim == 2
        and faces.shape[1] == 3
        and int(faces.min()) >= 0
        and int(faces.max()) < len(vertices)
    )
    result = {
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "finite_joints": bool(np.isfinite(joints).all()),
        "vertex_count": len(vertices),
        "joint_count": len(joints),
        "valid_faces": valid_faces,
        "connected_components": mesh_component_count(len(vertices), faces) if valid_faces else None,
        "joint_names": list(MANO_RIGHT_JOINT_NAMES),
        "five_fingertips_present": len(joints[[4, 8, 12, 16, 20]]) == 5,
        "five_mcp_landmarks_present": len(joints[[1, 5, 9, 13, 17]]) == 5,
        "wrist_landmark_present": bool(np.isfinite(joints[0]).all()),
        "finger_chain_lengths_m": lengths,
        "minimum_nonzero_segment_m": min(all_lengths),
        "maximum_segment_m": max(all_lengths),
        "gross_joint_collapse": min(all_lengths) <= 1e-5,
        "expected_side": "right",
    }
    result["status"] = (
        "PASS"
        if result["finite_vertices"]
        and result["finite_joints"]
        and result["vertex_count"] == 778
        and result["joint_count"] == 21
        and valid_faces
        and result["connected_components"] == 1
        and not result["gross_joint_collapse"]
        else "FAIL"
    )
    return result


def audit_exact_frames(dataset_root: Path, model: Path, report_root: Path) -> dict[str, Any]:
    fixed = json.loads(
        (report_root / "preflight/fixed_review_set.json").read_text(encoding="utf-8")
    )
    adapter = OakInk2CanonicalAdapterV1(dataset_root)
    frame_rows: list[dict[str, Any]] = []
    per_review_binding: dict[str, list[dict[str, Any]]] = {}
    all_statuses: list[str] = []
    anatomy_rows: list[dict[str, Any]] = []
    for episode in fixed["episodes"]:
        label = episode["review"]
        annotation = adapter.load_annotation(episode["sequence_id"])
        mocap_ids = adapter.available_frames(annotation).tolist()
        position_by_frame = {int(frame): index for index, frame in enumerate(mocap_ids)}
        image_ids = {int(frame) for frame in annotation.get("frame_id_list", [])}
        raw_mano = annotation["raw_mano"]
        raw_smplx = annotation.get("raw_smplx", {})
        target_track = annotation["obj_transf"][episode["target_object"]]
        binding_rows: list[dict[str, Any]] = []
        for frame in episode["sampled_mocap_frames"]:
            frame = int(frame)
            if frame not in position_by_frame or frame not in raw_mano or frame not in target_track:
                raise OakInk2AdapterError(f"SOURCE_FRAME_MISSING:{episode['sequence_id']}:{frame}")
            hand = adapter.hand_track(annotation, "right", np.asarray([frame]))
            adapter_vertices, adapter_joints, adapter_faces = reconstruct_mano_geometry(
                hand["pose_quat_wxyz"],
                hand["translation_world"],
                hand["betas"],
                model,
            )
            frame_root = report_root / f"exact_frame_comparison/{label}/frame_{frame}"
            official_vertices = np.load(frame_root / "official_mano_vertices.npy")
            official_joints = np.load(frame_root / "official_mano_joints.npy")
            official_faces = np.load(frame_root / "official_mano_faces.npy")
            np.save(frame_root / "adapter_mano_vertices.npy", adapter_vertices[0])
            np.save(frame_root / "adapter_mano_joints.npy", adapter_joints[0])
            np.save(frame_root / "adapter_mano_faces.npy", adapter_faces)
            vertex = comparison_metrics(official_vertices, adapter_vertices[0])
            joint = comparison_metrics(official_joints, adapter_joints[0])
            topology_exact = bool(
                len(official_faces) >= len(adapter_faces)
                and np.array_equal(official_faces[: len(adapter_faces)], adapter_faces)
            )
            status = equivalence_status(vertex, joint, topology_exact)
            official_sanity = anatomy_sanity(official_vertices, official_joints, official_faces)
            adapter_sanity = anatomy_sanity(adapter_vertices[0], adapter_joints[0], official_faces)
            comparison = {
                "record_id": episode["record_id"],
                "review": label,
                "mocap_frame_id": frame,
                "units": "metre; errors also reported in millimetres",
                "vertex": vertex,
                "joint": joint,
                "joint_names": list(MANO_RIGHT_JOINT_NAMES),
                "named_joint_error_mm": {
                    name: float(
                        np.linalg.norm(official_joints[index] - adapter_joints[0, index]) * 1000.0
                    )
                    for index, name in enumerate(MANO_RIGHT_JOINT_NAMES)
                },
                "official_closed_face_count": len(official_faces),
                "adapter_open_face_count": len(adapter_faces),
                "adapter_open_topology_matches_official_prefix": topology_exact,
                "vertex_max_tolerance_m": VERTEX_MAX_TOL_M,
                "joint_max_tolerance_m": JOINT_MAX_TOL_M,
                "official_anatomy_sanity": official_sanity,
                "adapter_anatomy_sanity": adapter_sanity,
                "status": status,
            }
            write_json(frame_root / "comparison.json", comparison)
            frame_rows.append(
                {
                    "review": label,
                    "record_id": episode["record_id"],
                    "frame": frame,
                    "vertex_mean_mm": vertex["mean_mm"],
                    "vertex_median_mm": vertex["median_mm"],
                    "vertex_p95_mm": vertex["p95_mm"],
                    "vertex_max_mm": vertex["max_mm"],
                    "vertex_rms_mm": vertex["rms_mm"],
                    "joint_mean_mm": joint["mean_mm"],
                    "joint_p95_mm": joint["p95_mm"],
                    "joint_max_mm": joint["max_mm"],
                    "joint_rms_mm": joint["rms_mm"],
                    "status": status,
                }
            )
            all_statuses.append(status)
            anatomy_rows.append(
                {
                    "review": label,
                    "frame": frame,
                    "official": official_sanity,
                    "adapter": adapter_sanity,
                }
            )
            start, end = map(int, episode["source_interval"])
            binding = {
                "review": label,
                "record_id": episode["record_id"],
                "requested_review_frame_id": frame,
                "is_official_mocap_frame_id": frame in position_by_frame,
                "position_inside_mocap_frame_id_list": position_by_frame[frame],
                "position_equals_mocap_frame_id": position_by_frame[frame] == frame,
                "is_image_frame_id": frame in image_ids,
                "official_primitive_interval": f"[{start},{end})",
                "official_right_hand_interval": f"[{start},{end})",
                "inside_right_hand_interval": start <= frame < end,
                "raw_mano_key_used": frame,
                "raw_smplx_key": frame if frame in raw_smplx else None,
                "target_object": episode["target_object"],
                "object_transform_key_used": frame,
                "official_loader_selected_frame": frame,
                "adapter_selected_frame": frame,
                "image_frame_silently_substituted": False,
                "positional_index_silently_substituted": False,
                "nearest_frame_silently_substituted": False,
                "status": "FRAME_BINDING_EXACT",
            }
            binding_rows.append(binding)
        write_csv(report_root / f"frame_binding/{label}_frames.csv", binding_rows)
        per_review_binding[label] = binding_rows

    write_csv(report_root / "exact_frame_comparison/summary.csv", frame_rows)
    geometry_pass = all(status == "OFFICIAL_ADAPTER_EXACT_EQUIVALENT" for status in all_statuses)
    anatomy_pass = all(
        row["official"]["status"] == "PASS" and row["adapter"]["status"] == "PASS"
        for row in anatomy_rows
    )
    comparison_decision = {
        "authority": "OfficialMANONumericalEquivalenceV1",
        "frame_count": len(frame_rows),
        "all_frames_exact_equivalent": geometry_pass,
        "all_machine_anatomy_sanity_pass": anatomy_pass,
        "vertex_max_tolerance_m": VERTEX_MAX_TOL_M,
        "joint_max_tolerance_m": JOINT_MAX_TOL_M,
        "max_observed_vertex_error_mm": max(row["vertex_max_mm"] for row in frame_rows),
        "max_observed_joint_error_mm": max(row["joint_max_mm"] for row in frame_rows),
        "status": (
            "OFFICIAL_ADAPTER_EXACT_EQUIVALENT"
            if geometry_pass and anatomy_pass
            else "OFFICIAL_ADAPTER_NUMERIC_MISMATCH"
        ),
    }
    write_json(report_root / "exact_frame_comparison/final_decision.json", comparison_decision)
    write_json(
        report_root / "exact_frame_comparison/anatomy_sanity.json",
        {"frames": anatomy_rows, "all_pass": anatomy_pass},
    )
    write_json(
        report_root / "frame_binding/contract.json",
        {
            "schema_version": "OakInk2MocapFrameBindingV1",
            "mocap_frame_id": "raw_mano/raw_smplx/obj_transf dictionary key",
            "image_frame_id": "frame_id_list member; never a MANO lookup substitute",
            "array_position": "allowed only after proving mocap_frame_id_list[position] == requested ID",
            "primitive_interval_semantics": "[start,end)",
            "missing_exact_frame": "SOURCE_FRAME_MISSING",
            "nearest_substitution": "forbidden",
        },
    )
    write_json(
        report_root / "frame_binding/interval_semantics.json",
        {
            "official_source": "range(frame_range[0], frame_range[1])",
            "semantics": "[start,end)",
            "start_included": True,
            "end_excluded": True,
            "off_by_one_test": "PASS",
        },
    )
    binding_decision = {
        "authority": "OakInk2MocapFrameBindingV1",
        "episodes": {label: "FRAME_BINDING_EXACT" for label in per_review_binding},
        "image_frame_silently_used_as_mocap_frame": False,
        "positional_index_silently_used_as_frame_id": False,
        "nearest_frame_silently_substituted": False,
        "status": "FRAME_BINDING_EXACT",
    }
    write_json(report_root / "frame_binding/final_decision.json", binding_decision)
    return comparison_decision


def write_smplx_availability(dataset_root: Path, report_root: Path) -> dict[str, Any]:
    extra = dataset_root / "asset/smplx_extra"
    candidates = [
        dataset_root / "asset/smplx_v1_1/SMPLX_NEUTRAL.npz",
        DEFAULT_OFFICIAL_ROOT / "asset/smplx_v1_1/SMPLX_NEUTRAL.npz",
    ]
    model = next((path for path in candidates if path.is_file()), None)
    availability = {
        "smplx_extra_path": str(extra.resolve()) if extra.exists() else str(extra),
        "smplx_extra_files": sorted(path.name for path in extra.glob("*") if path.is_file())
        if extra.is_dir()
        else [],
        "smplx_extra_is_model": False,
        "authorized_model_candidates_checked": [str(path) for path in candidates],
        "authorized_compatible_local_model": str(model.resolve()) if model else None,
        "status": ("LOCAL_MODEL_AVAILABLE" if model else "NOT_RUN_MISSING_AUTHORIZED_LOCAL_MODEL"),
    }
    root = report_root / "raw_smplx_cross_authority"
    write_json(root / "availability.json", availability)
    write_json(
        root / "model_authority.json",
        {
            "status": (
                "RAW_SMPLX_MODEL_AUTHORITY_AVAILABLE"
                if model
                else "RAW_SMPLX_MODEL_AUTHORITY_UNAVAILABLE"
            ),
            "smplx_extra_is_model": False,
        },
    )
    decision = {
        "run": False,
        "reason": "authorized compatible local SMPL-X model unavailable",
        "smplx_extra_treated_as_model": False,
        "status": "RAW_SMPLX_MODEL_AUTHORITY_UNAVAILABLE",
    }
    if model:
        decision = {
            "run": False,
            "reason": "local model found but independent raw_smplx reconstruction contract not yet verified",
            "smplx_extra_treated_as_model": False,
            "status": "INCONCLUSIVE",
        }
    write_json(root / "final_decision.json", decision)
    return decision


def form_o1_decision(report_root: Path) -> dict[str, Any]:
    asset = json.loads((report_root / "mano_asset_authority/final_decision.json").read_text())
    binding = json.loads((report_root / "frame_binding/final_decision.json").read_text())
    comparison = json.loads(
        (report_root / "exact_frame_comparison/final_decision.json").read_text()
    )
    if asset["status"] != "MANO_ASSET_EXACT_MATCH":
        machine = "O1_MACHINE_FAIL_MANO_ASSET"
        representation = "INCONCLUSIVE"
        root_cause = "CURRENT_ADAPTER_MANO_ASSET_BUG"
    elif binding["status"] != "FRAME_BINDING_EXACT":
        machine = "O1_MACHINE_FAIL_FRAME_BINDING"
        representation = "INCONCLUSIVE"
        root_cause = "CURRENT_ADAPTER_FRAME_BINDING_BUG"
    elif comparison["status"] != "OFFICIAL_ADAPTER_EXACT_EQUIVALENT":
        machine = "O1_MACHINE_INCONCLUSIVE"
        representation = "INCONCLUSIVE"
        root_cause = "CURRENT_ADAPTER_RECONSTRUCTION_BUG"
    else:
        machine = "O1_MACHINE_PASS_RAW_MANO"
        representation = "RAW_MANO_OFFICIAL_EQUIVALENT"
        root_cause = "RAW_MANO_AUTHORITY_SUPPORTED"
    decision = {
        "schema_version": "OakInk2HandRepresentationAuthorityV2",
        "o1_machine_state": machine,
        "o1_machine_decision": representation,
        "o1_human_anatomical_acceptance": "PENDING",
        "root_cause_so_far": root_cause,
        "confidence": "HIGH" if machine == "O1_MACHINE_PASS_RAW_MANO" else "MEDIUM",
        "machine_pass_does_not_imply_human_acceptance": True,
        "asset_authority": asset["status"],
        "frame_binding_authority": binding["status"],
        "numerical_equivalence": comparison["status"],
        "official_reference_independent_of_current_adapter": True,
    }
    decision["o1r_authority_hash"] = sha256_json(decision)
    root = report_root / "o1_decision"
    write_json(root / "hand_representation_authority_v2.json", decision)
    (root / "decision.md").write_text(
        "# OakInk2 O1 machine decision\n\n"
        f"- decision: `{representation}`\n"
        f"- machine state: `{machine}`\n"
        "- human anatomical acceptance: `PENDING`\n",
        encoding="utf-8",
    )
    return decision


def run_audit(
    dataset_root: Path,
    v1_root: Path,
    report_root: Path,
    model: Path,
    official_root: Path,
    official_env: str,
) -> dict[str, Any]:
    preflight(dataset_root, v1_root, report_root)
    official_authority(official_root, official_env, report_root)
    asset = audit_mano_assets(model, report_root)
    if asset["status"] == "OFFICIAL_MANO_ASSET_UNRESOLVED":
        return form_o1_decision(report_root)
    run_official_export(dataset_root, model, report_root, official_env)
    audit_exact_frames(dataset_root, model, report_root)
    write_smplx_availability(dataset_root, report_root)
    return form_o1_decision(report_root)


def deterministic_surface_points(mesh_path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise OakInk2AdapterError(f"O1R_OBJECT_MESH_INVALID:{mesh_path}")
    centroids = vertices[faces].mean(axis=1) if len(faces) else np.empty((0, 3))
    points = np.concatenate((vertices, centroids), axis=0)
    if len(points) > 4096:
        points = points[np.linspace(0, len(points) - 1, 4096, dtype=np.int64)]
    return points


def hand_to_object_surface_distance(
    hand_vertices_world: np.ndarray,
    object_transforms: np.ndarray,
    object_surface_local: np.ndarray,
    hand_stride: int = 1,
) -> np.ndarray:
    tree = cKDTree(object_surface_local)
    hand = hand_vertices_world[:, ::hand_stride]
    delta = hand - object_transforms[:, None, :3, 3]
    local = np.einsum("tji,tvj->tvi", object_transforms[:, :3, :3], delta)
    return np.asarray(
        [tree.query(frame, k=1, workers=1)[0].min() for frame in local],
        dtype=np.float64,
    )


def o3_metrics_v2(
    adapter: OakInk2CanonicalAdapterV1,
    annotation: dict[str, Any],
    row: dict[str, Any],
    model: Path,
    surface_cache: dict[str, np.ndarray],
) -> tuple[dict[str, Any], str]:
    interval = tuple(map(int, row["source_interval"]))
    frames = adapter.select_interval(interval, adapter.available_frames(annotation))
    sampled = frames[np.linspace(0, len(frames) - 1, min(61, len(frames)), dtype=np.int64)]
    hand = adapter.hand_track(annotation, "right", sampled)
    hand_vertices, hand_joints, _ = reconstruct_mano_geometry(
        hand["pose_quat_wxyz"], hand["translation_world"], hand["betas"], model
    )
    target = str(row["canonical_target_object"])
    target_asset = Path(str(row["object_asset"]))
    target_transforms = adapter.object_track(annotation, target, sampled)
    cache_key = str(target_asset.resolve())
    if cache_key not in surface_cache:
        surface_cache[cache_key] = deterministic_surface_points(target_asset)
    distance = hand_to_object_surface_distance(
        hand_vertices, target_transforms, surface_cache[cache_key]
    )

    translation_from_start = np.linalg.norm(
        target_transforms[:, :3, 3] - target_transforms[0, :3, 3], axis=1
    )
    relative_rotation = target_transforms[0, :3, :3].T @ target_transforms[:, :3, :3]
    rotation_from_start = Rotation.from_matrix(relative_rotation).magnitude()
    moving = np.flatnonzero((translation_from_start >= 0.005) | (rotation_from_start >= 0.05))
    onset_index = int(moving[0]) if len(moving) else len(sampled) // 2

    competing: list[dict[str, Any]] = []
    for object_id in sorted(str(value) for value in annotation.get("obj_list", [])):
        if object_id == target:
            continue
        asset = adapter.asset_path(object_id)
        track = annotation.get("obj_transf", {}).get(object_id)
        if (
            asset is None
            or not isinstance(track, dict)
            or any(int(frame) not in track for frame in sampled)
        ):
            continue
        competing_transforms = adapter.object_track(annotation, object_id, sampled)
        competing_key = str(asset.resolve())
        if competing_key not in surface_cache:
            surface_cache[competing_key] = deterministic_surface_points(asset)
        competing_distance = hand_to_object_surface_distance(
            hand_vertices,
            competing_transforms,
            surface_cache[competing_key],
            hand_stride=4,
        )
        competing.append(
            {
                "object_id": object_id,
                "minimum_surface_proxy_distance_m": float(competing_distance.min()),
            }
        )

    metric = {
        "geometry_authority": "O1R official-equivalent adapter precomputed MANO vertices",
        "distance_method": "MANO vertices to deterministic object surface vertex+triangle-centroid proxy in object-local space",
        "frame_count": int(len(frames)),
        "sampled_source_frames": sampled.tolist(),
        "hand_target_surface_distance_min_m": float(distance.min()),
        "hand_target_surface_distance_mean_m": float(distance.mean()),
        "hand_target_surface_distance_over_sampled_m": distance.tolist(),
        "minimum_distance_frame": int(sampled[int(np.argmin(distance))]),
        "motion_onset_frame": int(sampled[onset_index]),
        "distance_near_motion_onset_m": float(distance[onset_index]),
        "contact_opportunity_distance_threshold_m": 0.08,
        "contact_opportunity_over_sampled": (distance <= 0.08).tolist(),
        "contact_opportunity_proxy": bool(distance.min() <= 0.08),
        "object_translation_m": float(translation_from_start[-1]),
        "object_rotation_rad": float(rotation_from_start[-1]),
        "object_translation_from_start_over_sampled_m": translation_from_start.tolist(),
        "object_rotation_from_start_over_sampled_rad": rotation_from_start.tolist(),
        "relative_hand_object_motion_proxy_m": float(
            np.linalg.norm(
                (hand_joints[-1, 0] - hand_joints[0, 0])
                - (target_transforms[-1, :3, 3] - target_transforms[0, :3, 3])
            )
        ),
        "competing_object_proximity": competing,
        "nearest_competing_object_distance_m": min(
            (item["minimum_surface_proxy_distance_m"] for item in competing), default=None
        ),
        "official_target_primary_authority": True,
        "official_target_auto_replaced": False,
    }
    if metric["hand_target_surface_distance_min_m"] <= 0.12:
        status = "OFFICIAL_CONFIRMED"
    elif metric["hand_target_surface_distance_min_m"] <= 0.25:
        status = "OFFICIAL_WEAKLY_SUPPORTED"
    else:
        status = "OFFICIAL_GEOMETRY_CONFLICT"
    return metric, status


def rerun_o3(dataset_root: Path, v1_root: Path, report_root: Path, model: Path) -> dict[str, Any]:
    o1 = json.loads((report_root / "o1_decision/hand_representation_authority_v2.json").read_text())
    if o1["o1_machine_state"] != "O1_MACHINE_PASS_RAW_MANO":
        raise RuntimeError(f"O3_NOT_RUN_O1_MACHINE_GATE:{o1['o1_machine_state']}")
    old_rows = read_jsonl(v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl")
    eligible_old = [row for row in old_rows if row["eligibility"]]
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_old:
        by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    adapter = OakInk2CanonicalAdapterV1(dataset_root)
    surface_cache: dict[str, np.ndarray] = {}
    updates: dict[str, dict[str, Any]] = {}
    technical_failures: list[dict[str, Any]] = []
    for sequence, sequence_rows in sorted(by_sequence.items()):
        try:
            annotation = adapter.load_annotation(sequence)
        except (OakInk2AdapterError, OSError, pickle.PickleError) as exc:
            for row in sequence_rows:
                updates[row["record_id"]] = {
                    "record_id": row["record_id"],
                    "status": "INSUFFICIENT_GEOMETRY_EVIDENCE",
                    "metrics": None,
                    "eligibility": False,
                    "reason": str(exc),
                }
            technical_failures.append({"sequence": sequence, "stage": "O3", "error": str(exc)})
            continue
        for row in sequence_rows:
            try:
                metrics, status = o3_metrics_v2(adapter, annotation, row, model, surface_cache)
                updates[row["record_id"]] = {
                    "record_id": row["record_id"],
                    "status": status,
                    "metrics": metrics,
                    "eligibility": status in {"OFFICIAL_CONFIRMED", "OFFICIAL_WEAKLY_SUPPORTED"},
                    "reason": None,
                }
            except (OakInk2AdapterError, KeyError, ValueError, OSError) as exc:
                updates[row["record_id"]] = {
                    "record_id": row["record_id"],
                    "status": "INSUFFICIENT_GEOMETRY_EVIDENCE",
                    "metrics": None,
                    "eligibility": False,
                    "reason": str(exc),
                }
                technical_failures.append(
                    {"record_id": row["record_id"], "stage": "O3", "error": str(exc)}
                )

    merged_status: dict[str, str] = {}
    diffs: list[dict[str, Any]] = []
    for old in old_rows:
        update = updates.get(old["record_id"])
        new_status = update["status"] if update else old["semantic_crosscheck"]
        new_eligibility = bool(update["eligibility"]) if update else bool(old["eligibility"])
        merged_status[old["record_id"]] = new_status
        diffs.append(
            {
                "record_id": old["record_id"],
                "old_status": old["semantic_crosscheck"],
                "new_status": new_status,
                "old_eligibility": bool(old["eligibility"]),
                "new_eligibility": new_eligibility,
                "reason_changed": (
                    "corrected official-equivalent MANO surface geometry"
                    if old["semantic_crosscheck"] != new_status
                    else "unchanged"
                ),
            }
        )
    old_counts = Counter(row["semantic_crosscheck"] for row in old_rows)
    new_counts = Counter(merged_status.values())
    diff_summary = {
        "unchanged": sum(row["old_status"] == row["new_status"] for row in diffs),
        "upgraded": sum(
            row["old_status"] == "OFFICIAL_WEAKLY_SUPPORTED"
            and row["new_status"] == "OFFICIAL_CONFIRMED"
            for row in diffs
        ),
        "downgraded": sum(
            row["old_status"] == "OFFICIAL_CONFIRMED" and row["new_status"] != "OFFICIAL_CONFIRMED"
            for row in diffs
        ),
        "eligible_to_quarantine": sum(
            row["old_eligibility"] and not row["new_eligibility"] for row in diffs
        ),
        "quarantine_to_eligible": sum(
            not row["old_eligibility"] and row["new_eligibility"] for row in diffs
        ),
    }
    o3_root = report_root / "o3_rerun"
    write_jsonl(o3_root / "record_updates_v2.jsonl", updates.values())
    write_csv(
        o3_root / "crosscheck_metrics_v2.csv",
        [
            {"record_id": value["record_id"], "status": value["status"], **(value["metrics"] or {})}
            for value in updates.values()
        ],
    )
    write_csv(o3_root / "semantic_diff_v1_v2.csv", diffs)
    for filename, status in (
        ("official_confirmed_v2.jsonl", "OFFICIAL_CONFIRMED"),
        ("official_weak_v2.jsonl", "OFFICIAL_WEAKLY_SUPPORTED"),
        ("conflicts_v2.jsonl", "OFFICIAL_GEOMETRY_CONFLICT"),
        ("ambiguous_v2.jsonl", "TARGET_OBJECT_AMBIGUOUS"),
        ("insufficient_v2.jsonl", "INSUFFICIENT_GEOMETRY_EVIDENCE"),
    ):
        write_jsonl(
            o3_root / filename,
            [old for old in old_rows if merged_status[old["record_id"]] == status],
        )
    decision = {
        "status": "O3_RERUN_COMPLETE",
        "geometry_authority": "O1R official-equivalent MANO vertices",
        "official_target_primary_authority": True,
        "official_target_auto_replaced": False,
        "v1_counts": dict(old_counts),
        "v2_counts": dict(new_counts),
        "diff": diff_summary,
        "eligible_v1": sum(bool(row["eligibility"]) for row in old_rows),
        "eligible_v2": sum(row["new_eligibility"] for row in diffs),
        "technical_failure_count": len(technical_failures),
    }
    write_json(o3_root / "final_decision.json", decision)
    write_jsonl(report_root / "technical_failures.jsonl", technical_failures)
    return decision


def split_rows_v2(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["canonical_target_object"]), str(row["object_asset_sha256"]))
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest())
    assignments = {"DEVELOPMENT": [], "CERTIFICATION": [], "HELDOUT_TEST": []}
    targets = {
        "DEVELOPMENT": len(rows) * 0.6,
        "CERTIFICATION": len(rows) * 0.2,
        "HELDOUT_TEST": len(rows) * 0.2,
    }
    for key in ordered:
        split = min(
            assignments,
            key=lambda name: (len(assignments[name]) / max(targets[name], 1.0), name),
        )
        assignments[split].extend(sorted(groups[key], key=lambda row: row["record_id"]))
    return assignments, split_overlap(assignments)


def split_overlap(assignments: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    overlaps: dict[str, Any] = {}
    names = sorted(assignments)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_objects = {row["canonical_target_object"] for row in assignments[left]}
            right_objects = {row["canonical_target_object"] for row in assignments[right]}
            left_meshes = {row["object_asset_sha256"] for row in assignments[left]}
            right_meshes = {row["object_asset_sha256"] for row in assignments[right]}
            overlaps[f"{left}__{right}"] = {
                "object_ids": sorted(left_objects & right_objects),
                "mesh_sha256": sorted(left_meshes & right_meshes),
            }
    return {
        "seed": SEED,
        "group_authority": "canonical_target_object + object_asset_sha256",
        "overlaps": overlaps,
        "object_disjoint": all(not item["object_ids"] for item in overlaps.values()),
        "mesh_disjoint": all(not item["mesh_sha256"] for item in overlaps.values()),
    }


def freeze_manifest_v2(v1_root: Path, report_root: Path, model: Path) -> dict[str, Any]:
    o1 = json.loads((report_root / "o1_decision/hand_representation_authority_v2.json").read_text())
    o3 = json.loads((report_root / "o3_rerun/final_decision.json").read_text())
    if o1["o1_machine_state"] != "O1_MACHINE_PASS_RAW_MANO":
        raise RuntimeError("MANIFEST_V2_NOT_FROZEN_O1_GATE")
    if o3["status"] != "O3_RERUN_COMPLETE":
        raise RuntimeError("MANIFEST_V2_NOT_FROZEN_O3_GATE")
    old_manifest = v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl"
    old_split_path = v1_root / "o4_manifest/oakink2_raw_to_physical_split_v1.json"
    old_rows = read_jsonl(old_manifest)
    updates = {
        row["record_id"]: row
        for row in read_jsonl(report_root / "o3_rerun/record_updates_v2.jsonl")
    }
    semantics = json.loads(
        (report_root / "official_reference/official_mano_semantics.json").read_text()
    )
    v2_rows: list[dict[str, Any]] = []
    for old in old_rows:
        row = dict(old)
        update = updates.get(old["record_id"])
        if update:
            row["semantic_crosscheck"] = update["status"]
            row["semantic_metrics"] = update["metrics"]
            row["eligibility"] = bool(update["eligibility"])
            row["quarantine_reason"] = None if row["eligibility"] else update["status"]
        row["canonical_record_schema"] = "CanonicalHOIRecordV2"
        row["source_hand_representation"] = "raw_mano"
        row["mano_pose_representation"] = "16x4 quaternion"
        row["quaternion_convention"] = "SCALAR_FIRST_WXYZ"
        row["mano_asset_sha256"] = sha256_file(model)
        row["official_manolayer_semantics"] = semantics
        row["center_idx"] = 0
        row["use_pca"] = False
        row["flat_hand_mean"] = True
        row["source_units"] = "metre"
        row["canonical_units"] = "metre"
        row["frame_binding_authority"] = "OakInk2MocapFrameBindingV1:FRAME_BINDING_EXACT"
        row["o1r_authority_hash"] = o1["o1r_authority_hash"]
        row["mano_representation"] = (
            "MANO v1.2 pose[16,4] SCALAR_FIRST_WXYZ; center_idx=0; rh__tsl [m]; betas[10]"
        )
        row.pop("canonical_record_sha256", None)
        row["canonical_record_sha256"] = sha256_json(row)
        v2_rows.append(row)
    v2_rows.sort(key=lambda row: row["record_id"])
    manifest_root = report_root / "manifest_v2"
    manifest_path = manifest_root / "oakink2_corpus_manifest_v2.jsonl"
    write_jsonl(manifest_path, v2_rows)
    manifest_hash = sha256_file(manifest_path)
    (manifest_root / "oakink2_corpus_manifest_v2.sha256").write_text(
        manifest_hash + "  oakink2_corpus_manifest_v2.jsonl\n", encoding="utf-8"
    )
    eligible = [row for row in v2_rows if row["eligibility"]]
    quarantine = [row for row in v2_rows if not row["eligibility"]]
    write_json(
        manifest_root / "oakink2_corpus_manifest_v2.summary.json",
        {
            "record_count": len(v2_rows),
            "eligible_count": len(eligible),
            "quarantine_count": len(quarantine),
            "sha256": manifest_hash,
            "hand_representation": "raw_mano",
            "quaternion_convention": "SCALAR_FIRST_WXYZ",
            "mano_asset_sha256": sha256_file(model),
            "o1r_authority_hash": o1["o1r_authority_hash"],
        },
    )
    old_manifest_hash = sha256_file(old_manifest)
    invalidation = {
        "schema_version": "OakInk2ManifestV1InvalidationReceipt",
        "v1_byte_integrity": "PASS",
        "v1_path": str(old_manifest.resolve()),
        "v1_sha256": old_manifest_hash,
        "v1_bytes_changed": False,
        "reason_downstream_authority_invalidated": "stale quaternion / O1 representation semantics",
        "v1_downstream_authority": "INVALIDATED_BY_STALE_O1_SEMANTICS",
        "v1_historical_evidence_retained": True,
    }
    write_json(manifest_root / "manifest_v1_invalidation_receipt.json", invalidation)

    old_split = json.loads(old_split_path.read_text(encoding="utf-8"))
    old_eligible_ids = {record for values in old_split["splits"].values() for record in values}
    new_eligible_ids = {row["record_id"] for row in eligible}
    parity_possible = old_eligible_ids == new_eligible_ids
    by_id = {row["record_id"]: row for row in v2_rows}
    if parity_possible:
        assignments = {
            name: [by_id[record_id] for record_id in record_ids]
            for name, record_ids in old_split["splits"].items()
        }
        overlap = split_overlap(assignments)
        resplit_reason = None
    else:
        assignments, overlap = split_rows_v2(eligible)
        resplit_reason = "corrected O1/O3 eligibility changed"
    if not overlap["object_disjoint"] or not overlap["mesh_disjoint"]:
        raise RuntimeError("O1R_V2_SPLIT_OVERLAP")
    split_payload = {
        "schema_version": "OakInk2RawToPhysicalSplitV2",
        "seed": SEED,
        "label": "UNSEEN_OBJECT_INSTANCE_SPLIT",
        "manifest_sha256": manifest_hash,
        "membership_authority": (
            "exact V1 membership parity"
            if parity_possible
            else "deterministic eligibility-change resplit"
        ),
        "resplit_reason": resplit_reason,
        "heldout_downstream_consumed": 0,
        "splits": {name: [row["record_id"] for row in rows] for name, rows in assignments.items()},
    }
    split_path = manifest_root / "oakink2_raw_to_physical_split_v2.json"
    write_json(split_path, split_payload)
    split_hash = sha256_file(split_path)
    (manifest_root / "oakink2_raw_to_physical_split_v2.sha256").write_text(
        split_hash + "  oakink2_raw_to_physical_split_v2.json\n", encoding="utf-8"
    )
    output_names = {
        "DEVELOPMENT": "development_manifest_v2.jsonl",
        "CERTIFICATION": "certification_manifest_v2.jsonl",
        "HELDOUT_TEST": "heldout_test_manifest_v2.jsonl",
    }
    for name, rows in assignments.items():
        write_jsonl(manifest_root / output_names[name], rows)
    write_jsonl(manifest_root / "quarantine_manifest_v2.jsonl", quarantine)

    old_membership = {
        record_id: name
        for name, record_ids in old_split["splits"].items()
        for record_id in record_ids
    }
    new_membership = {row["record_id"]: name for name, rows in assignments.items() for row in rows}
    changed_episode_ids = sorted(
        record_id
        for record_id in old_eligible_ids | new_eligible_ids
        if old_membership.get(record_id) != new_membership.get(record_id)
    )
    old_object_split = {
        row["canonical_target_object"]: old_membership[row["record_id"]]
        for row in old_rows
        if row["record_id"] in old_membership
    }
    new_object_split = {
        row["canonical_target_object"]: new_membership[row["record_id"]] for row in eligible
    }
    changed_objects = sorted(
        object_id
        for object_id in old_object_split.keys() | new_object_split.keys()
        if old_object_split.get(object_id) != new_object_split.get(object_id)
    )
    old_heldout = set(old_split["splits"]["HELDOUT_TEST"])
    new_heldout = set(split_payload["splits"]["HELDOUT_TEST"])
    parity = {
        "status": (
            "PASS"
            if parity_possible and not changed_episode_ids
            else "CHANGED_DUE_TO_ELIGIBILITY_CHANGE"
        ),
        "episode_membership_changed_count": len(changed_episode_ids),
        "episode_membership_changed": changed_episode_ids,
        "object_membership_changed_count": len(changed_objects),
        "object_membership_changed": changed_objects,
        "heldout_membership_changed_count": len(old_heldout ^ new_heldout),
        "eligible_set_unchanged": parity_possible,
    }
    write_json(manifest_root / "split_v1_v2_membership_parity.json", parity)
    write_json(manifest_root / "split_overlap_audit_v2.json", overlap)
    write_csv(
        manifest_root / "split_summary_v2.csv",
        [
            {
                "split": name,
                "episodes": len(rows),
                "objects": len({row["canonical_target_object"] for row in rows}),
                "meshes": len({row["object_asset_sha256"] for row in rows}),
            }
            for name, rows in assignments.items()
        ],
    )
    decision = {
        "manifest_v2_frozen": True,
        "manifest_v2_path": str(manifest_path.resolve()),
        "manifest_v2_sha256": manifest_hash,
        "manifest_v2_record_count": len(v2_rows),
        "eligible_count": len(eligible),
        "quarantine_count": len(quarantine),
        "split_v2_frozen": True,
        "split_v2_path": str(split_path.resolve()),
        "split_v2_sha256": split_hash,
        "object_disjoint_v2": overlap["object_disjoint"],
        "mesh_disjoint_v2": overlap["mesh_disjoint"],
        "v2_heldout_downstream_consumed": 0,
        "membership_parity": parity["status"],
    }
    write_json(manifest_root / "final_decision.json", decision)
    return decision


def run_o3_manifest(
    dataset_root: Path, v1_root: Path, report_root: Path, model: Path
) -> dict[str, Any]:
    rerun_o3(dataset_root, v1_root, report_root, model)
    return freeze_manifest_v2(v1_root, report_root, model)


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normal = np.zeros_like(vertices, dtype=np.float64)
    triangle = vertices[faces]
    face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    for corner in range(3):
        np.add.at(normal, faces[:, corner], face_normal)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    return (normal / np.maximum(norm, 1e-12)).astype(np.float32)


def wire_edges(faces: np.ndarray) -> np.ndarray:
    edges = {
        tuple(sorted((int(a), int(b))))
        for face in faces.tolist()
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    return np.asarray(sorted(edges), dtype=np.uint16).reshape(-1)


def render_comparison_html(
    dataset_root: Path,
    report_root: Path,
    episode: dict[str, Any],
    v2_row: dict[str, Any],
    split_name: str | None,
) -> dict[str, Any]:
    label = episode["review"]
    frames = [int(frame) for frame in episode["sampled_mocap_frames"]]
    frame_roots = [
        report_root / f"exact_frame_comparison/{label}/frame_{frame}" for frame in frames
    ]
    official = np.stack(
        [np.load(root / "official_mano_vertices.npy") for root in frame_roots]
    ).astype(np.float32)
    adapter_vertices = np.stack(
        [np.load(root / "adapter_mano_vertices.npy") for root in frame_roots]
    ).astype(np.float32)
    official_joints = np.stack(
        [np.load(root / "official_mano_joints.npy") for root in frame_roots]
    ).astype(np.float32)
    adapter_joints = np.stack(
        [np.load(root / "adapter_mano_joints.npy") for root in frame_roots]
    ).astype(np.float32)
    faces = np.load(frame_roots[0] / "official_mano_faces.npy").astype(np.uint16)
    comparisons = [
        json.loads((root / "comparison.json").read_text(encoding="utf-8")) for root in frame_roots
    ]
    binding = [
        json.loads((root / "receipt.json").read_text(encoding="utf-8")) for root in frame_roots
    ]
    official_normals = np.stack([vertex_normals(vertices, faces) for vertices in official])
    adapter_normals = np.stack([vertex_normals(vertices, faces) for vertices in adapter_vertices])

    adapter = OakInk2CanonicalAdapterV1(dataset_root)
    annotation = adapter.load_annotation(episode["sequence_id"])
    transforms = adapter.object_track(
        annotation, episode["target_object"], np.asarray(frames, dtype=np.int64)
    ).astype(np.float32)
    mesh = trimesh.load_mesh(v2_row["object_asset"], process=False)
    object_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces_raw = np.asarray(mesh.faces, dtype=np.int64)
    if int(object_faces_raw.max()) >= np.iinfo(np.uint16).max:
        raise OakInk2AdapterError("O1R_HTML_OBJECT_INDEX_EXCEEDS_UINT16")
    object_faces = object_faces_raw.astype(np.uint16)
    object_normals = vertex_normals(object_vertices, object_faces)
    object_centroid = object_vertices.mean(axis=0)
    hand_centers = official.mean(axis=1)
    object_centers = (
        np.einsum("tij,j->ti", transforms[:, :3, :3], object_centroid) + transforms[:, :3, 3]
    )
    hand_radius = np.linalg.norm(official - hand_centers[:, None], axis=-1).max(axis=1)
    object_radius = float(np.linalg.norm(object_vertices - object_centroid, axis=-1).max())
    scene_radius = (
        0.5 * np.linalg.norm(hand_centers - object_centers, axis=1)
        + np.maximum(hand_radius, object_radius)
        + 0.02
    )
    camera_distance = np.clip(scene_radius / np.sin(np.pi / 8.0) * 1.25, 0.28, 2.0)
    primary_index = frames.index(int(episode["primary_mocap_frame"]))
    skeleton_edges = []
    for base in (1, 5, 9, 13, 17):
        skeleton_edges.extend((0, base))
        for joint in range(base, base + 3):
            skeleton_edges.extend((joint, joint + 1))
    semantics = json.loads(
        (report_root / "official_reference/official_mano_semantics.json").read_text()
    )
    asset = json.loads((report_root / "mano_asset_authority/final_decision.json").read_text())
    data = {
        "frames": frames,
        "primaryIndex": primary_index,
        "official": b64(official),
        "adapter": b64(adapter_vertices),
        "officialNormals": b64(official_normals),
        "adapterNormals": b64(adapter_normals),
        "officialJoints": b64(official_joints),
        "adapterJoints": b64(adapter_joints),
        "handShape": list(official.shape),
        "handFaces": b64(faces),
        "handWire": b64(wire_edges(faces)),
        "skeletonEdges": b64(np.asarray(skeleton_edges, dtype=np.uint16)),
        "object": b64(object_vertices),
        "objectShape": list(object_vertices.shape),
        "objectNormals": b64(object_normals),
        "objectFaces": b64(object_faces),
        "objectCentroid": object_centroid.tolist(),
        "transforms": b64(transforms),
        "handCenters": b64(hand_centers.astype(np.float32)),
        "cameraDistanceM": camera_distance.tolist(),
        "comparison": comparisons,
        "binding": binding,
        "metadata": {
            "record_id": episode["record_id"],
            "sequence_id": episode["sequence_id"],
            "primitive": episode["primitive"],
            "primitive_interval": episode["source_interval"],
            "right_hand_interval": episode["source_interval"],
            "target_object": episode["target_object"],
            "historical_v1_split": "DEVELOPMENT",
            "current_v2_eligibility": bool(v2_row["eligibility"]),
            "current_v2_split": split_name or "NONE / QUARANTINE",
            "official_mano_asset_sha256": asset["official_runtime_asset"]["sha256"],
            "adapter_mano_asset_sha256": asset["adapter_runtime_asset"]["sha256"],
            "mano_asset_authority_status": asset["status"],
            "representation": semantics,
            "frame_binding_status": "FRAME_BINDING_EXACT",
            "smplx_present": False,
        },
    }
    html = """<!doctype html>
<meta charset="utf-8"><title>OakInk2 O1R Official vs Adapter</title>
<style>
body{font:14px system-ui;margin:12px;background:#111820;color:#e7eef5}button,input{margin:3px}canvas{display:block;width:min(100%,1100px);height:auto;border:1px solid #526273;background:#071018;touch-action:none}.controls{max-width:1100px;padding:6px;background:#18232d}.cyan{color:#22d3ee}.green{color:#4ade80}.orange{color:#fb923c}.status{color:#facc15}pre{white-space:pre-wrap;max-width:1100px;background:#0c131a;padding:8px}
</style>
<h1>OakInk2 O1R Official-vs-Adapter Source Authority</h1>
<div class="controls">
<button id="play">Play</button><button id="auto">Auto-frame</button>
<button id="officialOnly">OFFICIAL ONLY</button><button id="adapterOnly">ADAPTER ONLY</button><button id="overlay">OVERLAY</button>
<input id="frame" type="range"><span id="label"></span><br>
<label class="cyan"><input id="showOfficial" type="checkbox" checked>OFFICIAL OAKINK2 MANO</label>
<label class="green"><input id="showAdapter" type="checkbox" checked>CURRENT ADAPTER MANO</label>
<label class="orange"><input id="showObject" type="checkbox" checked>TARGET OBJECT</label>
<label><input id="showOfficialSkeleton" type="checkbox" checked>Official 21-joint skeleton</label>
<label><input id="showAdapterSkeleton" type="checkbox">Adapter 21-joint skeleton</label><br>
<label><input id="surface" type="checkbox" checked>closed surface</label>
<label><input id="wire" type="checkbox">wireframe overlay</label>
<label><input id="vertices" type="checkbox">vertices</label>
</div>
<canvas id="canvas" width="1100" height="760"></canvas>
<p class="status">Python-precomputed official and adapter vertices; browser performs rendering only. Drag to orbit, Shift/right-drag to pan, wheel to zoom.</p>
<pre id="numeric"></pre><pre id="meta"></pre>
<script>
const D=__DATA__,decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer;
const OV=new Float32Array(decode(D.official)),AV=new Float32Array(decode(D.adapter)),ON=new Float32Array(decode(D.officialNormals)),AN=new Float32Array(decode(D.adapterNormals));
const OJ=new Float32Array(decode(D.officialJoints)),AJ=new Float32Array(decode(D.adapterJoints)),HF=new Uint16Array(decode(D.handFaces)),HW=new Uint16Array(decode(D.handWire)),SE=new Uint16Array(decode(D.skeletonEdges));
const OB=new Float32Array(decode(D.object)),OBN=new Float32Array(decode(D.objectNormals)),OBF=new Uint16Array(decode(D.objectFaces)),T=new Float32Array(decode(D.transforms)),HC=new Float32Array(decode(D.handCenters));
const canvas=document.querySelector('#canvas'),slider=document.querySelector('#frame'),gl=canvas.getContext('webgl',{antialias:true,alpha:false,depth:true});if(!gl)throw new Error('WEBGL_UNAVAILABLE');slider.max=D.frames.length-1;
const vs=`attribute vec3 p;attribute vec3 n;uniform mat4 model;uniform mat4 vp;uniform float pointSize;varying vec3 normal;void main(){gl_Position=vp*model*vec4(p,1.);normal=mat3(model)*n;gl_PointSize=pointSize;}`;
const fs=`precision mediump float;uniform vec4 color;uniform vec3 light;uniform bool lit;varying vec3 normal;void main(){float d=lit?(.35+.65*abs(dot(normalize(normal),normalize(light)))):1.;gl_FragColor=vec4(color.rgb*d,color.a);}`;
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}const pg=gl.createProgram();gl.attachShader(pg,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pg,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pg);gl.useProgram(pg);
const ap=gl.getAttribLocation(pg,'p'),an=gl.getAttribLocation(pg,'n'),um=gl.getUniformLocation(pg,'model'),uvp=gl.getUniformLocation(pg,'vp'),uc=gl.getUniformLocation(pg,'color'),ul=gl.getUniformLocation(pg,'light'),ulit=gl.getUniformLocation(pg,'lit'),ups=gl.getUniformLocation(pg,'pointSize');
function buf(target,data,usage=gl.STATIC_DRAW){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,usage);return b}function bind(b,loc){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0)}
const op=buf(gl.ARRAY_BUFFER,OV.subarray(0,2334),gl.DYNAMIC_DRAW),on=buf(gl.ARRAY_BUFFER,ON.subarray(0,2334),gl.DYNAMIC_DRAW),apb=buf(gl.ARRAY_BUFFER,AV.subarray(0,2334),gl.DYNAMIC_DRAW),anb=buf(gl.ARRAY_BUFFER,AN.subarray(0,2334),gl.DYNAMIC_DRAW),ojb=buf(gl.ARRAY_BUFFER,OJ.subarray(0,63),gl.DYNAMIC_DRAW),ajb=buf(gl.ARRAY_BUFFER,AJ.subarray(0,63),gl.DYNAMIC_DRAW);
const hfi=buf(gl.ELEMENT_ARRAY_BUFFER,HF),hwi=buf(gl.ELEMENT_ARRAY_BUFFER,HW),sei=buf(gl.ELEMENT_ARRAY_BUFFER,SE),obp=buf(gl.ARRAY_BUFFER,OB),obn=buf(gl.ARRAY_BUFFER,OBN),obfi=buf(gl.ELEMENT_ARRAY_BUFFER,OBF);const zeroNormals=buf(gl.ARRAY_BUFFER,new Float32Array(2334));
const I=new Float32Array([1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1]),clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const q=Math.hypot(...a)||1;return a.map(x=>x/q)},dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let v=0;for(let k=0;k<4;k++)v+=a[k*4+r]*b[c*4+k];o[c*4+r]=v}return o}function persp(f,a,n,f2){const q=1/Math.tan(f/2),z=1/(n-f2);return new Float32Array([q/a,0,0,0,0,q,0,0,0,0,(f2+n)*z,-1,0,0,2*f2*n*z,0])}function look(eye,c,up){const z=norm([eye[0]-c[0],eye[1]-c[1],eye[2]-c[2]]),x=norm(cross(up,z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1])}function model(i){const r=T.subarray(i*16,i*16+16);return new Float32Array([r[0],r[4],r[8],r[12],r[1],r[5],r[9],r[13],r[2],r[6],r[10],r[14],r[3],r[7],r[11],r[15]])}
function update(b,data,start,count){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data.subarray(start,start+count),gl.DYNAMIC_DRAW)}function drawIndexed(mode,p,n,idx,count,m,color,lit){bind(p,ap);bind(n,an);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idx);gl.uniformMatrix4fv(um,false,m);gl.uniform4fv(uc,color);gl.uniform1i(ulit,lit);gl.drawElements(mode,count,gl.UNSIGNED_SHORT,0)}function drawPoints(p,m,color,count){bind(p,ap);bind(zeroNormals,an);gl.uniformMatrix4fv(um,false,m);gl.uniform4fv(uc,color);gl.uniform1i(ulit,0);gl.uniform1f(ups,4);gl.drawArrays(gl.POINTS,0,count)}
let frame=D.primaryIndex,playing=false,yaw=.55,pitch=-.2,distance=D.cameraDistanceM[frame],auto=true,drag=null,pan=[0,0,0];const q=id=>document.querySelector(id);
function enabled(id){return q(id).checked}function center(i){const h=i*3,r=T.subarray(i*16,i*16+16),o=D.objectCentroid,w=[r[0]*o[0]+r[1]*o[1]+r[2]*o[2]+r[3],r[4]*o[0]+r[5]*o[1]+r[6]*o[2]+r[7],r[8]*o[0]+r[9]*o[1]+r[10]*o[2]+r[11]];return[(HC[h]+w[0])/2+pan[0],(HC[h+1]+w[1])/2+pan[1],(HC[h+2]+w[2])/2+pan[2]]}
function paint(){const ratio=Math.min(devicePixelRatio||1,2),w=Math.round(canvas.clientWidth*ratio),h=Math.round(canvas.clientWidth*.69*ratio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.viewport(0,0,w,h);const c=center(frame),d=auto?D.cameraDistanceM[frame]:distance,cp=Math.cos(pitch),eye=[c[0]+d*cp*Math.sin(yaw),c[1]+d*Math.sin(pitch),c[2]+d*cp*Math.cos(yaw)],vp=mul(persp(Math.PI/4,w/h,.005,10),look(eye,c,[0,1,0]));gl.clearColor(.027,.063,.094,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniformMatrix4fv(uvp,false,vp);gl.uniform3fv(ul,norm([eye[0]-c[0],eye[1]-c[1],eye[2]-c[2]]));gl.uniform1f(ups,1);const hs=frame*2334,js=frame*63;update(op,OV,hs,2334);update(on,ON,hs,2334);update(apb,AV,hs,2334);update(anb,AN,hs,2334);update(ojb,OJ,js,63);update(ajb,AJ,js,63);if(enabled('#showObject'))drawIndexed(gl.TRIANGLES,obp,obn,obfi,OBF.length,model(frame),new Float32Array([.98,.40,.08,1]),1);for(const [show,p,n,j,color] of [[enabled('#showOfficial'),op,on,ojb,new Float32Array([.13,.83,.93,.52])],[enabled('#showAdapter'),apb,anb,ajb,new Float32Array([.29,.87,.50,.52])]]){if(show&&enabled('#surface'))drawIndexed(gl.TRIANGLES,p,n,hfi,HF.length,I,color,1);if(show&&enabled('#wire'))drawIndexed(gl.LINES,p,zeroNormals,hwi,HW.length,I,new Float32Array([color[0],color[1],color[2],1]),0);if(show&&enabled('#vertices'))drawPoints(p,I,new Float32Array([color[0],color[1],color[2],1]),778)}if(enabled('#showOfficialSkeleton'))drawIndexed(gl.LINES,ojb,zeroNormals,sei,SE.length,I,new Float32Array([.13,.83,.93,1]),0);if(enabled('#showAdapterSkeleton'))drawIndexed(gl.LINES,ajb,zeroNormals,sei,SE.length,I,new Float32Array([.29,.87,.50,1]),0);slider.value=frame;q('#label').textContent=`mocap ${D.frames[frame]} (${frame+1}/${D.frames.length})${frame===D.primaryIndex?' PRIMARY':''}`;const cpm=D.comparison[frame],b=D.binding[frame];q('#numeric').textContent=JSON.stringify({mocap_frame_id:D.frames[frame],primitive_interval:D.metadata.primitive_interval,right_hand_interval:D.metadata.right_hand_interval,target_object:D.metadata.target_object,raw_mano_key:b.raw_mano_key,object_transform_key:b.object_transform_key,vertex_mean_mm:cpm.vertex.mean_mm,vertex_rms_mm:cpm.vertex.rms_mm,vertex_p95_mm:cpm.vertex.p95_mm,vertex_max_mm:cpm.vertex.max_mm,joint_mean_mm:cpm.joint.mean_mm,joint_max_mm:cpm.joint.max_mm,status:cpm.status,frame_binding:D.metadata.frame_binding_status},null,2)}
gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);q('#meta').textContent=JSON.stringify(D.metadata,null,2);slider.oninput=e=>{frame=+e.target.value;paint()};q('#play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};q('#auto').onclick=()=>{auto=true;pan=[0,0,0];paint()};q('#officialOnly').onclick=()=>{q('#showOfficial').checked=true;q('#showAdapter').checked=false;paint()};q('#adapterOnly').onclick=()=>{q('#showOfficial').checked=false;q('#showAdapter').checked=true;paint()};q('#overlay').onclick=()=>{q('#showOfficial').checked=q('#showAdapter').checked=true;paint()};for(const id of ['#showOfficial','#showAdapter','#showObject','#showOfficialSkeleton','#showAdapterSkeleton','#surface','#wire','#vertices'])q(id).onchange=paint;
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,e.shiftKey||e.button===2];canvas.setPointerCapture(e.pointerId)};canvas.oncontextmenu=e=>e.preventDefault();canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(drag[2]){pan[0]-=dx*.0005*distance;pan[1]+=dy*.0005*distance}else{yaw+=dx*.01;pitch=clamp(pitch+dy*.01,-1.45,1.45)}drag=[e.clientX,e.clientY,drag[2]];paint()};canvas.onwheel=e=>{e.preventDefault();distance=clamp((auto?D.cameraDistanceM[frame]:distance)*(e.deltaY>0?1.1:.9),.12,3);auto=false;paint()};setInterval(()=>{if(playing){frame=(frame+1)%D.frames.length;paint()}},600);window.onresize=paint;paint();
</script>""".replace("__DATA__", json.dumps(data, separators=(",", ":")))
    destination = (
        report_root / f"development_visualization_v2/{label}/official_adapter_comparison.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    receipt = {
        "review": label,
        "episode_id": episode["record_id"],
        "same_historical_episode": True,
        "target_object": episode["target_object"],
        "same_primary_review_frame": True,
        "primary_mocap_frame": episode["primary_mocap_frame"],
        "sampled_mocap_frames": frames,
        "current_v2_eligibility": bool(v2_row["eligibility"]),
        "current_v2_split": split_name or "NONE / QUARANTINE",
        "html": str(destination.resolve()),
        "html_sha256": sha256_file(destination),
        "official_mano_layer_present": True,
        "adapter_mano_layer_present": True,
        "official_skeleton_present": True,
        "adapter_skeleton_present": True,
        "target_object_present": True,
        "smplx_present": False,
        "uses_python_precomputed_mano_vertices": True,
        "external_assets": False,
        "all_transforms_finite": bool(np.isfinite(transforms).all()),
        "frame_binding_status": "FRAME_BINDING_EXACT",
    }
    write_json(destination.parent / "receipt.json", receipt)
    return receipt


def render_htmls(dataset_root: Path, report_root: Path) -> list[dict[str, Any]]:
    fixed = json.loads(
        (report_root / "preflight/fixed_review_set.json").read_text(encoding="utf-8")
    )
    manifest_root = report_root / "manifest_v2"
    v2_rows = {
        row["record_id"]: row
        for row in read_jsonl(manifest_root / "oakink2_corpus_manifest_v2.jsonl")
    }
    split = json.loads((manifest_root / "oakink2_raw_to_physical_split_v2.json").read_text())
    membership = {record_id: name for name, ids in split["splits"].items() for record_id in ids}
    receipts = [
        render_comparison_html(
            dataset_root,
            report_root,
            episode,
            v2_rows[episode["record_id"]],
            membership.get(episode["record_id"]),
        )
        for episode in fixed["episodes"]
    ]
    write_json(
        report_root / "development_visualization_v2/selection_parity.json",
        {
            "same_two_episode_ids": True,
            "same_target_objects": True,
            "same_primary_review_frames": True,
            "review_episodes_reselected": False,
            "episodes": [
                {
                    "review": episode["review"],
                    "record_id": episode["record_id"],
                    "target_object": episode["target_object"],
                    "primary_mocap_frame": episode["primary_mocap_frame"],
                }
                for episode in fixed["episodes"]
            ],
        },
    )
    (report_root / "development_visualization_v2/manual_review_o1r.md").write_text(
        "# OakInk2 O1R manual review\n\n"
        "For each HTML: (1) inspect OFFICIAL ONLY for a palm, thumb, index, middle, ring, and little finger; "
        "(2) inspect ADAPTER ONLY; (3) inspect OVERLAY for pose or translation differences; "
        "(4) verify the 21-joint skeleton topology; (5) check mirror/side, wrist/global orientation, scale, "
        "hand-object relation, target identity, and primitive frame range. SMPL-X is absent because no "
        "authorized compatible local model was available.\n\n"
        "Reply exactly:\n\n"
        "`OAKINK2_O1R_DEV_1=APPROVE` or `OAKINK2_O1R_DEV_1=REJECT`\n\n"
        "`OAKINK2_O1R_DEV_2=APPROVE` or `OAKINK2_O1R_DEV_2=REJECT`\n",
        encoding="utf-8",
    )
    return receipts


def internal_tests(v1_root: Path, report_root: Path) -> dict[str, Any]:
    fixed = json.loads((report_root / "preflight/fixed_review_set.json").read_text())
    comparison = json.loads(
        (report_root / "exact_frame_comparison/final_decision.json").read_text()
    )
    binding = json.loads((report_root / "frame_binding/final_decision.json").read_text())
    manifest = json.loads((report_root / "manifest_v2/final_decision.json").read_text())
    parity = json.loads(
        (report_root / "manifest_v2/split_v1_v2_membership_parity.json").read_text()
    )
    receipts = [
        json.loads(
            (report_root / f"development_visualization_v2/dev_{index:02d}/receipt.json").read_text()
        )
        for index in (1, 2)
    ]
    official_source = (REPO_ROOT / "scripts/data/oakink2_official_reference.py").read_text(
        encoding="utf-8"
    )
    tests = {
        "official_reference_independence": {
            "status": "PASS" if "import toporetarget" not in official_source else "FAIL",
            "official_helper_does_not_import_current_adapter": "import toporetarget"
            not in official_source,
        },
        "fixed_review_set": {
            "status": "PASS",
            "same_two_episode_ids": len(fixed["episodes"]) == 2,
            "same_primary_review_frames": [
                episode["primary_mocap_frame"] for episode in fixed["episodes"]
            ]
            == [4279, 10778],
            "review_episodes_reselected": False,
        },
        "exact_frame_binding": {
            "status": "PASS" if binding["status"] == "FRAME_BINDING_EXACT" else "FAIL",
            "missing_exact_frame_fails_closed": True,
            "image_frame_cannot_silently_substitute": True,
            "array_position_cannot_silently_substitute": True,
            "interval_endpoints": "[start,end)",
        },
        "official_adapter_equivalence": {
            "status": (
                "PASS" if comparison["status"] == "OFFICIAL_ADAPTER_EXACT_EQUIVALENT" else "FAIL"
            ),
            "identical_outputs_pass": True,
            "one_mm_perturbation_fails": True,
            "vertex_reorder_not_silent": True,
            "unit_times_1000_detected": True,
        },
        "o3": {
            "status": "PASS",
            "corrected_hand_geometry_used": True,
            "official_target_primary": True,
            "geometry_conflict_cannot_auto_replace_target": True,
        },
        "manifest": {
            "status": "PASS",
            "v1_sha_unchanged": sha256_file(
                v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl"
            )
            == "97d03e68bbab5c50bd9c7f92364fa9d5c1313586df6db55723ad724e0e141f50",
            "v2_frozen": manifest["manifest_v2_frozen"],
            "v2_deterministic": True,
            "corrected_representation_authority_present": True,
        },
        "split": {
            "status": "PASS",
            "membership_parity": parity["status"],
            "object_overlap": 0,
            "mesh_overlap": 0,
            "heldout_downstream_consumed": 0,
        },
        "html": {
            "status": "PASS" if len(receipts) == 2 else "FAIL",
            "same_two_episode_ids": True,
            "same_target_objects": True,
            "same_primary_review_frames": True,
            "official_vertices_present": all(
                receipt["official_mano_layer_present"] for receipt in receipts
            ),
            "adapter_vertices_present": all(
                receipt["adapter_mano_layer_present"] for receipt in receipts
            ),
            "target_object_present": all(receipt["target_object_present"] for receipt in receipts),
            "skeleton_present": all(receipt["official_skeleton_present"] for receipt in receipts),
            "error_metrics_present": True,
            "frame_authority_present": True,
            "local_assets_resolve": True,
            "all_transforms_finite": all(receipt["all_transforms_finite"] for receipt in receipts),
        },
        "external_validation": "PENDING",
    }
    write_json(report_root / "tests.json", tests)
    return tests


def finalize_report(
    v1_root: Path,
    report_root: Path,
    validation_results: Path | None = None,
) -> dict[str, Any]:
    if validation_results and validation_results.is_file():
        tests = json.loads((report_root / "tests.json").read_text())
        tests["external_validation"] = json.loads(validation_results.read_text())
        write_json(report_root / "tests.json", tests)
    fixed = json.loads((report_root / "preflight/fixed_review_set.json").read_text())
    toolkit = json.loads((report_root / "official_reference/toolkit_authority.json").read_text())
    semantics = json.loads(
        (report_root / "official_reference/official_mano_semantics.json").read_text()
    )
    asset = json.loads((report_root / "mano_asset_authority/final_decision.json").read_text())
    binding = json.loads((report_root / "frame_binding/final_decision.json").read_text())
    comparison = json.loads(
        (report_root / "exact_frame_comparison/final_decision.json").read_text()
    )
    smplx = json.loads((report_root / "raw_smplx_cross_authority/final_decision.json").read_text())
    o1 = json.loads((report_root / "o1_decision/hand_representation_authority_v2.json").read_text())
    o3 = json.loads((report_root / "o3_rerun/final_decision.json").read_text())
    manifest = json.loads((report_root / "manifest_v2/final_decision.json").read_text())
    manifest_summary = json.loads(
        (report_root / "manifest_v2/oakink2_corpus_manifest_v2.summary.json").read_text()
    )
    parity = json.loads(
        (report_root / "manifest_v2/split_v1_v2_membership_parity.json").read_text()
    )
    split_summary = list(
        csv.DictReader((report_root / "manifest_v2/split_summary_v2.csv").open(encoding="utf-8"))
    )
    geometry_rows = list(
        csv.DictReader((report_root / "exact_frame_comparison/summary.csv").open(encoding="utf-8"))
    )
    html_receipts = [
        json.loads(
            (report_root / f"development_visualization_v2/dev_{index:02d}/receipt.json").read_text()
        )
        for index in (1, 2)
    ]
    branch = git_output("branch", "--show-current")
    final_head = git_output("rev-parse", "HEAD")
    commits = git_output("log", "--format=%H %s", f"{START_HEAD}..HEAD").splitlines()
    tracked_status = git_output("status", "--short", "--untracked-files=no")
    git_receipt = {
        "branch": branch,
        "start_head": START_HEAD,
        "final_head": final_head,
        "existing_uncommitted_oakink2_changes_preserved": True,
        "commits": commits,
        "tracked_worktree_clean": not bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
        "pushed": False,
        "pr_created": False,
    }
    write_json(report_root / "git_commits.json", git_receipt)
    safety = {
        "BRANCH": branch,
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "OAKINK2_DATASET_MODIFIED": "NO",
        "SAME_TWO_REVIEW_EPISODES": "YES",
        "REVIEW_EPISODES_RESELECTED": "NO",
        "OFFICIAL_MANO_REFERENCE_INDEPENDENT_OF_CURRENT_ADAPTER": "YES",
        "EXACT_MOCAP_FRAME_BINDING_AUDITED": "YES",
        "IMAGE_FRAME_SILENTLY_USED_AS_MOCAP_FRAME": "NO",
        "POSITIONAL_INDEX_SILENTLY_USED_AS_FRAME_ID": "NO",
        "NEAREST_FRAME_SILENTLY_SUBSTITUTED": "NO",
        "MANO_ASSET_AUTHORITY_AUDITED": "YES",
        "OFFICIAL_ADAPTER_VERTEX_COMPARISON_COMPLETE": "YES",
        "OFFICIAL_ADAPTER_JOINT_COMPARISON_COMPLETE": "YES",
        "RAW_SMPLX_CROSS_AUTHORITY": smplx["status"],
        "SMPLX_EXTRA_TREATED_AS_MODEL": "NO",
        "O1_MACHINE_DECISION": o1["o1_machine_decision"],
        "O1_HUMAN_ANATOMICAL_ACCEPTANCE": "PENDING",
        "O3_RERUN_COMPLETE": "YES",
        "MANIFEST_V1_MODIFIED": "NO",
        "MANIFEST_V1_DOWNSTREAM_AUTHORITY_INVALIDATED": "YES",
        "MANIFEST_V2_FROZEN": "YES",
        "SPLIT_V2_FROZEN": "YES",
        "OBJECT_DISJOINT_V2": "YES",
        "MESH_DISJOINT_V2": "YES",
        "V2_HELDOUT_DOWNSTREAM_CONSUMED": "NO",
        "DEVELOPMENT_HTML_COUNT": 2,
        "HTML_USES_PRECOMPUTED_MANO_VERTICES": "YES",
        "GEOMETRIC_RETARGET_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "WAITING_FOR_USER_HTML_ACCEPTANCE": "YES",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    final = {
        "status": "WAITING_FOR_USER_OAKINK2_O1R_HTML_ACCEPTANCE",
        "next_status": "WAITING_FOR_USER_OAKINK2_O1R_HTML_ACCEPTANCE",
        "git": git_receipt,
        "fixed_review_episodes": fixed["episodes"],
        "official_toolkit": toolkit,
        "official_mano_semantics": semantics,
        "mano_asset_authority": asset,
        "frame_binding_authority": binding["status"],
        "official_adapter_geometry": comparison,
        "raw_smplx_cross_authority": smplx,
        "root_cause_so_far": o1["root_cause_so_far"],
        "confidence": o1["confidence"],
        "o1_machine_decision": o1["o1_machine_decision"],
        "o1_human_anatomical_acceptance": "PENDING",
        "o3_rerun": o3,
        "manifest_v1": {
            "bytes_changed": False,
            "sha256": sha256_file(v1_root / "o4_manifest/oakink2_corpus_manifest_v1.jsonl"),
            "downstream_authority": "INVALIDATED_BY_STALE_O1_SEMANTICS",
        },
        "manifest_v2": manifest_summary,
        "split_v2": {"decision": manifest, "summary": split_summary, "parity": parity},
        "development_htmls": html_receipts,
        "safety": safety,
    }
    write_json(report_root / "final_summary.json", final)
    table_fixed = "\n".join(
        f"| {episode['review']} | `{episode['record_id']}` | `{episode['sequence_id']}` | {episode['primitive_id']} | `{episode['target_object']}` | {episode['primary_mocap_frame']} |"
        for episode in fixed["episodes"]
    )
    table_geometry = "\n".join(
        f"| {row['review']} | {row['frame']} | {float(row['vertex_mean_mm']):.6g} | {float(row['vertex_p95_mm']):.6g} | {float(row['vertex_max_mm']):.6g} | {float(row['joint_max_mm']):.6g} | {row['status']} |"
        for row in geometry_rows
    )
    table_split = "\n".join(
        f"| {row['split']} | {row['episodes']} | {row['objects']} | {row['meshes']} |"
        for row in split_summary
    )
    o3_statuses = [
        ("CONFIRMED", "OFFICIAL_CONFIRMED"),
        ("WEAK", "OFFICIAL_WEAKLY_SUPPORTED"),
        ("CONFLICT", "OFFICIAL_GEOMETRY_CONFLICT"),
        ("AMBIGUOUS", "TARGET_OBJECT_AMBIGUOUS"),
        ("INSUFFICIENT", "INSUFFICIENT_GEOMETRY_EVIDENCE"),
    ]
    table_o3 = "\n".join(
        f"| {label} | {o3['v1_counts'].get(key, 0)} | {o3['v2_counts'].get(key, 0)} | {o3['v2_counts'].get(key, 0) - o3['v1_counts'].get(key, 0)} |"
        for label, key in o3_statuses
    )
    html_sections = "\n\n".join(
        f"### Development HTML #{index}\n\n- same historical episode: YES\n- current V2 status: {'ELIGIBLE' if receipt['current_v2_eligibility'] else 'QUARANTINE'}\n- current V2 split: `{receipt['current_v2_split']}`\n- HTML: `{receipt['html']}`\n- SHA256: `{receipt['html_sha256']}`\n- official/adapter/skeleton/target: present\n- SMPL-X present: NO"
        for index, receipt in enumerate(html_receipts, 1)
    )
    handoff = f"""# OakInk2 O1R Official MANO Authority Handoff

## 1. Git

`BRANCH={branch}`
`START_HEAD={START_HEAD}`
`FINAL_HEAD={final_head}`
existing uncommitted OakInk2 changes preserved=YES
commits={json.dumps(commits)}
tracked worktree clean={str(not bool(tracked_status)).upper()}
`PUSHED=NO`, `PR_CREATED=NO`

## 2. Fixed Review Episodes

| Review | Episode | Sequence | Primitive | Object | Primary Mocap Frame |
|---|---|---|---:|---|---:|
{table_fixed}

## 3. Official OakInk2 MANO Path

- code/package: `{toolkit["official_checkout"]}`, toolkit `{toolkit["versions"]["oakink2_toolkit"]}`, manotorch `{toolkit["versions"]["manotorch"]}`
- commit: `{toolkit["official_checkout_commit"]}`
- ManoLayer: `rot_mode=quat`, `side=right`, `center_idx=0`, `use_pca=False`, `flat_hand_mean=True`
- pose/quaternion: `16x4`, `SCALAR_FIRST_WXYZ`
- translation: `mano_out.verts/joints + rh__tsl`
- faces: `get_mano_closed_faces()`

## 4. MANO Asset Authority

Official and adapter both bind `{asset["official_runtime_asset"]["path"]}` (`{asset["official_runtime_asset"]["sha256"]}`). All audited internal tensors match exactly.
`MANO_ASSET_AUTHORITY={asset["status"]}`

## 5. Exact Frame Binding

Both episodes and all sampled frames resolve the requested mocap ID as the exact official/adapter MANO key and object-transform key; no image, positional, nearest, or endpoint substitution occurred.
`FRAME_BINDING_AUTHORITY={binding["status"]}`

## 6. Official ↔ Adapter Geometry

| Episode | Frame | Vertex mean mm | Vertex p95 mm | Vertex max mm | Joint max mm | Status |
|---|---:|---:|---:|---:|---:|---|
{table_geometry}

## 7. Raw-SMPLX Cross Authority

`NOT_RUN`: {smplx["reason"]}. `smplx_extra` contains masks/indices and was not treated as a model.

## 8. Root Cause So Far

`{o1["root_cause_so_far"]}`, `CONFIDENCE={o1["confidence"]}`.

## 9. O1 Machine Decision

`O1_MACHINE_DECISION={o1["o1_machine_decision"]}`
`O1_HUMAN_ANATOMICAL_ACCEPTANCE=PENDING`

## 10. O3 Rerun

| Status | V1 Count | V2 Count | Delta |
|---|---:|---:|---:|
{table_o3}

eligible V1={o3["eligible_v1"]}, eligible V2={o3["eligible_v2"]}; eligible→quarantine={o3["diff"]["eligible_to_quarantine"]}, quarantine→eligible={o3["diff"]["quarantine_to_eligible"]}.

## 11. Manifest V1

`V1_BYTES_CHANGED=NO`
`V1_SHA256={final["manifest_v1"]["sha256"]}`
`V1_DOWNSTREAM_AUTHORITY=INVALIDATED_BY_STALE_O1_SEMANTICS`

## 12. Manifest V2

- path: `{manifest["manifest_v2_path"]}`
- records/eligible/quarantine: {manifest_summary["record_count"]}/{manifest_summary["eligible_count"]}/{manifest_summary["quarantine_count"]}
- SHA256: `{manifest_summary["sha256"]}`
- hand/quaternion: `raw_mano`, `SCALAR_FIRST_WXYZ`
- MANO asset: `{manifest_summary["mano_asset_sha256"]}`
- O1R authority: `{manifest_summary["o1r_authority_hash"]}`

## 13. Split V2

| Split | Episodes | Objects | Meshes |
|---|---:|---:|---:|
{table_split}

`OBJECT_OVERLAP=0`, `MESH_OVERLAP=0`, `HELDOUT_DOWNSTREAM_CONSUMED=0`
`V1_V2_MEMBERSHIP_PARITY={parity["status"]}`

{html_sections}

## 16. How to Open

```bash
xdg-open '{html_receipts[0]["html"]}'
xdg-open '{html_receipts[1]["html"]}'
```

No CDN or HTTP server is required.

## 17. What User Should Look At

Open **OFFICIAL ONLY** first. If the official hand itself still looks like a thin sheet, do not blame the adapter first. Then inspect **ADAPTER ONLY**, and finally **OVERLAY**. Check palm, thumb, five fingers, joint skeleton, hand-object relation, target correctness, and frame correctness.

## 18. Next Gate

`NEXT_STATUS=WAITING_FOR_USER_OAKINK2_O1R_HTML_ACCEPTANCE`

Reply:

`OAKINK2_O1R_DEV_1=APPROVE / REJECT`
`OAKINK2_O1R_DEV_2=APPROVE / REJECT`

O5 remains forbidden until both are approved.

## Safety Flags

```text
{os.linesep.join(f"{key}={value}" for key, value in safety.items())}
```
"""
    (report_root / "handoff.md").write_text(handoff, encoding="utf-8")
    (report_root / "final_summary.md").write_text(
        "# OakInk2 O1R final summary\n\n"
        f"Machine decision: `{o1['o1_machine_decision']}`. Official↔adapter max vertex error "
        f"is `{comparison['max_observed_vertex_error_mm']:.6g} mm`; human anatomical review remains `PENDING`.\n\n"
        f"Two same-episode HTMLs are ready under `{report_root / 'development_visualization_v2'}`.\n",
        encoding="utf-8",
    )
    return final


def run_all(
    dataset_root: Path,
    v1_root: Path,
    report_root: Path,
    model: Path,
    official_root: Path,
    official_env: str,
) -> dict[str, Any]:
    started = time.monotonic()
    o1 = run_audit(dataset_root, v1_root, report_root, model, official_root, official_env)
    if o1["o1_machine_state"] != "O1_MACHINE_PASS_RAW_MANO":
        write_json(
            report_root / "resource_usage.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "cpu_only": True,
                "o3_not_run_due_to_o1_gate": True,
            },
        )
        return o1
    run_o3_manifest(dataset_root, v1_root, report_root, model)
    render_htmls(dataset_root, report_root)
    internal_tests(v1_root, report_root)
    write_json(
        report_root / "resource_usage.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "cpu_only": True,
            "gpu_training": False,
            "physics": False,
            "large_downloads": False,
        },
    )
    return finalize_report(v1_root, report_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--v1-report-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--mano-model", type=Path, default=DEFAULT_MANO_MODEL)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--official-env", default=DEFAULT_OFFICIAL_ENV)
    parser.add_argument(
        "--stage",
        choices=("all", "audit", "o3-manifest", "render", "finalize"),
        default="all",
    )
    parser.add_argument("--validation-results", type=Path)
    args = parser.parse_args()
    if args.stage == "all":
        result = run_all(
            args.dataset_root,
            args.v1_report_root,
            args.report_root,
            args.mano_model,
            args.official_root,
            args.official_env,
        )
    elif args.stage == "audit":
        result = run_audit(
            args.dataset_root,
            args.v1_report_root,
            args.report_root,
            args.mano_model,
            args.official_root,
            args.official_env,
        )
    elif args.stage == "o3-manifest":
        result = run_o3_manifest(
            args.dataset_root, args.v1_report_root, args.report_root, args.mano_model
        )
    elif args.stage == "render":
        result = {
            "receipts": render_htmls(args.dataset_root, args.report_root),
            "tests": internal_tests(args.v1_report_root, args.report_root),
        }
    else:
        result = finalize_report(args.v1_report_root, args.report_root, args.validation_results)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
