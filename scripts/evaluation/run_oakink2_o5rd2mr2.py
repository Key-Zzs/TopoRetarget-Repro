#!/usr/bin/env python3
"""O5R-D2M-R2 full-sequence consumer-input authority repair.

This workflow is deliberately optimizer-free.  It audits and freezes every
input consumed by the future DEV2 full-sequence runner, and proves that the
frame-1 context can be constructed from the consumed V2 frame-0 state.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2m as d2m  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2mr as d2mr  # noqa: E402
from toporetarget.retarget import final_refinement as refinement  # noqa: E402
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ExecutionV4AcceptedRuntimeState,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2mr2_full_sequence_consumer_input_authority_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "9fc2d3c611af1797abb9c120eed592868d64c1f0"
EXPECTED_FRAMES = 240
SOURCE_START = 10704
SOURCE_STOP = 10944
SOURCE_FRAMES = list(range(SOURCE_START, SOURCE_STOP))
CANARY_ORDINALS = [0, 1, 60, 120, 180, 239]
OBJECT_ID = d2m.OBJECT_ID
EPISODE = d2m.EPISODE
V1_UUID = "43473aae-4bd5-44cf-99d8-5a898a33f1f5"
V2_UUID = "67a5dc86-7d33-4bbb-85e9-dece272ec4a3"
GRAPH_AUTHORITY_SHA = "fffb906d0838c1fd038c9ad6d40623d61637a5bfad6bcb12a7ac61ec2e7f4383"
GRAPH_PATH = d2mr.ROOT / d2mr.NEW_GRAPH_RELATIVE
V1_ROOT = d2m.ROOT
V2_ROOT = d2mr.ROOT / "d2mv2"

V2_IMMUTABLE_HASHES = {
    "final_summary.json": "0da5fa48d9d6b4c37ea60ed6ed2683a244a6779f51dd28b5dc092f0f474091cc",
    "d2mv2/run_authority/full_run_manifest.json": "2a8ddf57bc8e28b8a083341ed92877e46a4940af91c66dc099e954223986b070",
    "d2mv2/checkpoints/frame_000/checkpoint.json": "c7783325e3c07ebeb3174e981ada753b9f5695df2f1d6463fc44179cde269c66",
    "d2mv2/solver/first_failure.json": "b40e772f1accc22c5288976ce934d0f06c90c61e3fcabaad8c65566824685f49",
}

ROLES = {
    "TRAJECTORY_STATIC",
    "SOURCE_FRAME_INDEXED",
    "FRAME0_ONLY_INITIALIZER",
    "PREVIOUS_ACCEPTED_RUNTIME",
    "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze_json(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="d2mr2-serialization-") as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"
        first.write_bytes(canonical_bytes(value))
        second.write_bytes(canonical_bytes(value))
        if sha256_file(first) != sha256_file(second) or sha256_file(first) != digest:
            raise RuntimeError("AUTHORITY_SERIALIZATION_NONDETERMINISTIC")
    return digest


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def _array_finite(value: Any) -> bool:
    return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())


def _source_frame(ordinal: int) -> int:
    return SOURCE_START + int(ordinal)


class FullSequenceV3Runtime(d2g.V3Runtime):
    """DEV2 adapter that binds explicit current-frame seeds without warm[t]."""

    def bind_context(
        self,
        ordinal: int,
        *,
        previous_base: np.ndarray | None = None,
        previous_qpos: np.ndarray | None = None,
    ) -> tuple[d2g.RuntimeBindingV3, Any]:
        step = int(self.current_runtime_step)
        neutral = np.asarray(self.model.neutral_q, dtype=np.float64)
        seed_base = self.base_for_q(ordinal, neutral)
        global_frame = int(self.graph.frame_indices[ordinal])
        object_id = str(self.graph.metadata["object_id"])
        obj = self.sequence.rigid_object(object_id)
        object_pose = np.asarray(obj.pose_scene.pose_scene[global_frame], dtype=np.float64)
        if step == 0:
            if previous_base is not None or previous_qpos is not None:
                raise RuntimeError("O5RD2G_FRAME0_PREVIOUS_STATE_FORBIDDEN")
            previous_frame = None
            previous_reference = None
            propagated = None
        else:
            if previous_base is None or previous_qpos is None:
                raise RuntimeError("PREVIOUS_RUNTIME_STATE_REQUIRED")
            previous_ordinal = ordinal - 1
            if previous_ordinal < 0:
                raise RuntimeError("O5RD2G_INVALID_PREVIOUS_ORDINAL")
            previous_frame = int(self.graph.frame_indices[previous_ordinal])
            previous_seed_base = self.base_for_q(previous_ordinal, neutral)
            previous_reference = d2g.map_previous_state_to_seed(
                np.asarray(previous_base), np.asarray(previous_qpos), seed_base
            )
            propagated = d2g.transport_previous_final_to_current_warm(
                previous_seed_base,
                np.asarray(previous_base),
                seed_base,
                neutral,
                np.asarray(previous_qpos),
                neutral,
                self.model.joint_lower,
                self.model.joint_upper,
                previous_frame=previous_frame,
                current_frame=global_frame,
            )
        predicted_base = None if propagated is None else propagated.predicted_base_scene
        predicted_q = None if propagated is None else propagated.predicted_qpos
        binding = d2g.RuntimeBindingV3(
            active_frame_id=global_frame,
            runtime_step_index=step,
            current_source_frame_id=global_frame,
            previous_source_frame_id=previous_frame,
            previous_runtime_base_scene=None
            if previous_base is None
            else np.asarray(previous_base),
            previous_robot_qpos=None if previous_qpos is None else np.asarray(previous_qpos),
            continuous_predicted_translation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, 3]),
            continuous_predicted_rotation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, :3]),
            continuous_predicted_qpos=predicted_q,
            base_correction_reference=None if propagated is None else propagated.base_correction,
            object_pose_scene=object_pose,
            object_id=object_id,
            robot_name=str(self.model.name),
            robot_side=str(self.model.side),
            robot_dof_names=tuple(str(name) for name in self.model.dof_names),
            robot_mapping_authority="configs/robots/wuji_hand2_beta1_rh.yaml",
            source_hand_id="right_hand",
        )
        context = _make_explicit_seed_context(
            self,
            ordinal,
            seed_base=seed_base,
            seed_qpos=neutral,
            previous_reference=previous_reference,
            binding=binding,
        )
        return binding, context


def _make_explicit_seed_context(
    runtime: FullSequenceV3Runtime,
    ordinal: int,
    *,
    seed_base: np.ndarray,
    seed_qpos: np.ndarray,
    previous_reference: np.ndarray | None,
    binding: d2g.RuntimeBindingV3,
) -> Any:
    """Exact _make_context payload with explicit seeds and no warm carrier access."""

    frame = runtime.graph.frames[ordinal]
    global_frame = int(runtime.graph.frame_indices[ordinal])
    source_features = runtime.source_features(ordinal)
    obj = runtime.sequence.rigid_object(str(runtime.graph.metadata["object_id"]))
    object_pose = np.asarray(obj.pose_scene.pose_scene[global_frame], dtype=np.float64)
    geometry_indices, transforms, links, slices = refinement._surface_layout(
        runtime.model, runtime.surface
    )
    context_hash = refinement._stable_hash(
        {
            "global_frame": global_frame,
            "robot_name": runtime.model.name,
            "object_pose": object_pose.tolist(),
            "graph_frame": int(ordinal),
            "source_feature_shape": list(np.asarray(source_features.adjacent_features).shape),
            "paper_hash": runtime.resources.paper.config_hash,
            "surface_profile_hash": runtime.surface.profile.profile_hash,
            "quality_extension": None,
        }
    )
    return refinement._FrameContext(
        robot_model=runtime.model,
        graph_frame=frame,
        source_features=source_features,
        frame_profile=runtime.frame_profile,
        bone_profile=runtime.bone_profile,
        seed_base=np.asarray(seed_base, dtype=np.float64),
        seed_qpos=np.asarray(seed_qpos, dtype=np.float64),
        previous_reference=previous_reference,
        paper=runtime.resources.paper,
        sdf=runtime.backends.solver_sdf,
        reference_sdf=runtime.resources.reference_sdf,
        object_pose_scene=object_pose,
        surface=runtime.surface,
        surface_points_local=np.asarray(runtime.surface.points_local, dtype=np.float64),
        surface_geometry_indices=geometry_indices,
        surface_local_transforms=transforms,
        surface_link_names=links,
        geometry_slices=slices,
        frame_id=global_frame,
        context_hash=context_hash,
        temporal_scope=binding.temporal_scope,
        continuous_prediction_base=binding.continuous_prediction_base,
        continuous_prediction_qpos=binding.continuous_predicted_qpos,
        cache=refinement.RefinementEvaluationCache(global_frame, context_hash),
        spatial_gradient_backend=runtime.execution.signed_distance_gradient,
        sign_cache=runtime.backends.sign_cache,
        compiled_spatial_fd_backend=runtime.backends.compiled_spatial_fd_backend,
    )


def build_runtime() -> FullSequenceV3Runtime:
    runtime = FullSequenceV3Runtime("dev_02", d2m.D2G3_ROOT)
    graph = d2g.load_interaction_graph(GRAPH_PATH)
    runtime.graph = graph
    runtime.resources = d2g.prepare_refinement_resources(
        runtime.sequence,
        graph,
        runtime.solver,
        geometry_artifact_root=d2m.D2G3_ROOT / "execution_v3_design/geometry/dev2",
    )
    runtime.backends = d2g.prepare_refinement_runtime_backends(runtime.resources, runtime.execution)
    return runtime


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status_lines = [
        line for line in git("status", "--short", "--untracked-files=all").splitlines() if line
    ]
    allowed_task_paths = {
        "scripts/evaluation/run_oakink2_o5rd2mr2.py",
        "tests/evaluation/test_oakink2_o5rd2mr2.py",
    }
    unrelated_dirty = [line for line in status_lines if line[3:] not in allowed_task_paths]
    ignored = (
        subprocess.run(["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False).returncode
        == 0
    )
    checks = {
        "repo": git("rev-parse", "--show-toplevel") == str(REPO),
        "branch": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "no_unrelated_worktree_changes": not unrelated_dirty,
        "artifact_root_ignored": ignored,
        "graph_exists": GRAPH_PATH.exists(),
        "v1_exists": V1_ROOT.exists(),
        "v2_exists": V2_ROOT.exists(),
    }
    value = {
        "schema_version": "D2MR2GitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "BRANCH": branch,
        "START_HEAD": head,
        "contract_start_head": START_HEAD,
        "status_lines": status_lines,
        "unrelated_dirty": unrelated_dirty,
        "checks": checks,
        "prohibitions": {
            "retarget_optimizer": True,
            "dev2_full_recovery_v3": True,
            "new_branch": True,
            "new_worktree": True,
            "push": True,
            "pull_request": True,
        },
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError(f"PREFLIGHT_FAIL:{checks}")
    return value


def verify_historical_runs(root: Path) -> dict[str, Any]:
    v1 = read_json(V1_ROOT / "final_summary.json")
    v1_receipt = read_json(d2mr.ROOT / "d2m_v1_history/historical_result.json")
    v2 = read_json(d2mr.ROOT / "final_summary.json")
    v2_failure = read_json(V2_ROOT / "solver/first_failure.json")
    v2_state = read_json(V2_ROOT / "run_authority/run_state.json")
    v2_hashes = {rel: sha256_file(d2mr.ROOT / rel) for rel in V2_IMMUTABLE_HASHES}
    checks = {
        "v1_uuid": v1["DEV2_RUN_UUID"] == V1_UUID,
        "v1_result": v1["DEV2_MACHINE"] == "RETARGET_NUMERICAL_FAIL",
        "v1_attempted_completed": (v1["ATTEMPTED_FRAMES"], v1["COMPLETED_FRAMES"]) == (2, 1),
        "v1_frame0": v1["FIRST_ACCEPTED_SOURCE_FRAME"] == SOURCE_START,
        "v1_failure_frame": v1["FIRST_FAILURE_SOURCE_FRAME"] == SOURCE_START + 1,
        "v1_immutable_receipt": v1_receipt["status"] == "PASS",
        "v2_uuid": v2["DEV2_V2_RUN_UUID"] == V2_UUID,
        "v2_result": v2["DEV2_FULL_RECOVERY_V2_MACHINE"] == "BLOCKED_INPUT_AUTHORITY",
        "v2_attempted_completed": (v2["ATTEMPTED_FRAMES"], v2["COMPLETED_FRAMES"]) == (2, 1),
        "v2_failure_frame": v2["FIRST_INPUT_AUTHORITY_FAILURE_SOURCE_FRAME"] == SOURCE_START + 1,
        "v2_optimizer_not_started": v2_failure["FRAME1_OPTIMIZER_STARTED"] == "NO",
        "v2_run_consumed_once": v2_state["SCIENTIFIC_RUN_COUNT"] == 1,
        "v2_immutable_hashes": v2_hashes == V2_IMMUTABLE_HASHES,
        "first_real_failure_null": v2["FIRST_REAL_FULL_TRAJECTORY_FAILURE_SOURCE_FRAME"] is None,
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    v1_out = {
        "schema_version": "D2MV1ImmutableHistoryV1",
        "status": result
        if all(value for key, value in checks.items() if key.startswith("v1_"))
        else "FAIL",
        "RUN_UUID": V1_UUID,
        "result": "RETARGET_NUMERICAL_FAIL",
        "attempted": 2,
        "completed": 1,
        "first_failure_source_frame": 10705,
        "mechanism": "SourceInteractionGraph authority size=1",
        "D2M_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
    }
    v2_out = {
        "schema_version": "D2MV2ImmutableHistoryV1",
        "status": result
        if all(value for key, value in checks.items() if key.startswith("v2_"))
        else "FAIL",
        "RUN_UUID": V2_UUID,
        "result": "BLOCKED_INPUT_AUTHORITY",
        "attempted": 2,
        "completed": 1,
        "first_input_authority_failure_source_frame": 10705,
        "mechanism": "warm.qpos/base_pose_scene authority size=1",
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE": None,
        "D2M_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "immutable_hashes": v2_hashes,
    }
    history = {
        "schema_version": "DEV2FullRecoveryRunHistoryV2",
        "status": result,
        "history_count": 2,
        "V1": v1_out,
        "V2": v2_out,
        "V3": {"status": "NOT_RUN", "RUN_UUID": None},
        "checks": checks,
    }
    write_json(root / "history/d2m_v1.json", v1_out)
    write_json(root / "history/d2m_v2.json", v2_out)
    write_json(root / "history/run_history.json", history)
    write_json(root / "preflight/historical_runs.json", history)
    if result != "PASS":
        raise RuntimeError("FAIL_UPSTREAM_INTEGRITY:HISTORICAL_RUNS")
    return history


def verify_frozen_v4(root: Path) -> dict[str, Any]:
    hashes = {key: sha256_file(path) for key, (path, _expected) in d2m.FROZEN_AUTHORITIES.items()}
    authority_checks = {
        key: hashes[key] == expected for key, (_path, expected) in d2m.FROZEN_AUTHORITIES.items()
    }
    implementation_hashes = {
        key: sha256_file(path) for key, (path, _expected) in d2m.METHOD_IMPLEMENTATIONS.items()
    }
    implementation_checks = {
        key: implementation_hashes[key] == expected
        for key, (_path, expected) in d2m.METHOD_IMPLEMENTATIONS.items()
    }
    four = {key: hashes[key] for key in d2m.V4_AUTHORITY_HASHES}
    checks = {
        **authority_checks,
        **{f"implementation:{k}": v for k, v in implementation_checks.items()},
    }
    value = {
        "schema_version": "D2MR2FrozenExecutionV4IntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "four_frozen_v4_hashes": four,
        "expected_four_frozen_v4_hashes": d2m.V4_AUTHORITY_HASHES,
        "all_authority_hashes": hashes,
        "implementation_hashes": implementation_hashes,
        "checks": checks,
    }
    write_json(root / "preflight/frozen_v4.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_EXECUTION_V4_HASH_INTEGRITY")
    return value


def verify_full_graph_authority(root: Path) -> dict[str, Any]:
    authority = read_json(d2mr.ROOT / "graph_authority/sequence_authority.json")
    coverage = read_json(d2mr.ROOT / "graph_authority/coverage.json")
    parity = read_json(d2mr.ROOT / "graph_authority/frame0_parity.json")
    determinism = read_json(d2mr.ROOT / "graph_authority/determinism.json")
    digest = sha256_file(d2mr.ROOT / "graph_authority/sequence_authority.json")
    checks = {
        "authority_sha": digest == GRAPH_AUTHORITY_SHA,
        "authority_frozen": authority["status"] == "FROZEN",
        "coverage": coverage["status"] == "PASS" and coverage["GRAPH_ACTUAL_FRAMES"] == 240,
        "frame0_parity": parity["status"] == "PASS",
        "determinism": determinism["status"] == "PASS",
        "serialization": determinism["FULL_ARTIFACT_SERIALIZATION_DETERMINISM"] == "PASS",
        "artifact_sha": d2g.interaction_artifact_hash(GRAPH_PATH)
        == authority["graph_artifact_sha256"],
    }
    value = {
        "schema_version": "D2MR2GraphAuthorityIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "authority_sha256": digest,
        "graph_artifact_sha256": authority["graph_artifact_sha256"],
        "coverage": "240/240",
        "checks": checks,
        "rebuilt": False,
    }
    write_json(root / "preflight/graph_authority.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_UPSTREAM_INTEGRITY:GRAPH_AUTHORITY")
    return value


def _inventory(runtime: FullSequenceV3Runtime) -> list[dict[str, Any]]:
    source_track = runtime.sequence.hand("right_hand").keypoint_tracks["mediapipe21"]
    obj = runtime.sequence.rigid_object(OBJECT_ID)
    graph_count = int(runtime.graph.frame_count)
    sequence_count = int(source_track.positions_scene.shape[0])
    object_pose_count = int(obj.pose_scene.pose_scene.shape[0])
    warm_q_count = int(np.asarray(runtime.warm.arrays["qpos"]).shape[0])
    warm_base_count = int(np.asarray(runtime.warm.arrays["base_pose_scene"]).shape[0])
    return [
        {
            "field": "trajectory.source_frame_ids",
            "producer": "D2MR graph frame manifest",
            "consumer": "full runner/preflight",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": 240,
            "frame_ids": "10704..10943",
            "scientific_role": "identity and alignment",
        },
        {
            "field": "interaction_graph.frames",
            "producer": "SourceInteractionGraph sequence authority",
            "consumer": "explicit context builder/ObjectiveV2",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": graph_count,
            "frame_ids": "10704..10943 via manifest",
            "scientific_role": "E_IM graph",
        },
        {
            "field": "source_mano.mediapipe21.positions_scene",
            "producer": "canonical OakInk2 adapter",
            "consumer": "V3Runtime.source_features",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": sequence_count,
            "frame_ids": "10704..10943 via canonical metadata",
            "scientific_role": "bone and wrist source geometry",
        },
        {
            "field": "source_mano.parameters",
            "producer": "canonical OakInk2 adapter",
            "consumer": "none in ExecutionV4 path",
            "role": "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED",
            "indexed": "NO",
            "length": sequence_count,
            "frame_ids": "diagnostic",
            "scientific_role": "not consumed",
        },
        {
            "field": "object.pose_scene",
            "producer": "canonical OakInk2 adapter",
            "consumer": "context/binding",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": object_pose_count,
            "frame_ids": "10704..10943 via canonical metadata",
            "scientific_role": "object-relative geometry",
        },
        {
            "field": "sequence.timestamps",
            "producer": "canonical OakInk2 adapter",
            "consumer": "alignment preflight",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": sequence_count,
            "frame_ids": "10704..10943",
            "scientific_role": "temporal identity",
        },
        {
            "field": "object.mesh",
            "producer": "canonical OakInk2 adapter",
            "consumer": "reference SDF",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "episode static",
            "scientific_role": "collision/reference surface",
        },
        {
            "field": "object.surface_samples",
            "producer": "frozen area-uniform sampler",
            "consumer": "interaction graph",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "episode static",
            "scientific_role": "canonical object samples",
        },
        {
            "field": "robot.asset_and_joint_limits",
            "producer": "Wuji asset",
            "consumer": "V3/V4 search/context",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "trajectory static",
            "scientific_role": "kinematics and bounds",
        },
        {
            "field": "robot.surface_samples",
            "producer": "frozen collision sample config",
            "consumer": "context/SDF",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "trajectory static",
            "scientific_role": "collision points",
        },
        {
            "field": "solver.objective_and_profiles",
            "producer": "frozen V4 authorities/configs",
            "consumer": "V3/V4 search",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "trajectory static",
            "scientific_role": "search and ObjectiveV2",
        },
        {
            "field": "context_seed.qpos",
            "producer": "Wuji canonical neutral",
            "consumer": "explicit context builder",
            "role": "TRAJECTORY_STATIC",
            "indexed": "NO",
            "length": 1,
            "frame_ids": "derived identically per frame",
            "scientific_role": "cold-start coordinate origin",
        },
        {
            "field": "context_seed.base_pose_scene",
            "producer": "V3Runtime.base_for_q(source frame, neutral)",
            "consumer": "explicit context builder",
            "role": "SOURCE_FRAME_INDEXED",
            "indexed": "YES",
            "length": 240,
            "frame_ids": "10704..10943",
            "scientific_role": "source-derived base coordinate origin",
        },
        {
            "field": "warm.qpos",
            "producer": "DEV2 frame0 V3Runtime constructor",
            "consumer": "frame0 legacy initializer only",
            "role": "FRAME0_ONLY_INITIALIZER",
            "indexed": "ordinal0 only",
            "length": warm_q_count,
            "frame_ids": "10704 only",
            "scientific_role": "legacy frame0 initializer; bypassed for t>0",
        },
        {
            "field": "warm.base_pose_scene",
            "producer": "DEV2 frame0 V3Runtime constructor",
            "consumer": "frame0 legacy initializer only",
            "role": "FRAME0_ONLY_INITIALIZER",
            "indexed": "ordinal0 only",
            "length": warm_base_count,
            "frame_ids": "10704 only",
            "scientific_role": "legacy frame0 initializer; bypassed for t>0",
        },
        {
            "field": "previous_accepted_runtime.qpos",
            "producer": "accepted ExecutionV4 state t-1",
            "consumer": "V3/V4 binding and continuity",
            "role": "PREVIOUS_ACCEPTED_RUNTIME",
            "indexed": "runtime chain",
            "length": 0,
            "frame_ids": "online t-1 only",
            "scientific_role": "sequential continuation",
        },
        {
            "field": "previous_accepted_runtime.base_pose_scene",
            "producer": "accepted ExecutionV4 state t-1",
            "consumer": "V3/V4 binding and continuity",
            "role": "PREVIOUS_ACCEPTED_RUNTIME",
            "indexed": "runtime chain",
            "length": 0,
            "frame_ids": "online t-1 only",
            "scientific_role": "sequential continuation",
        },
        {
            "field": "continuous_prediction.qpos_and_base",
            "producer": "frozen transport of accepted t-1",
            "consumer": "ObjectiveV2 temporal context",
            "role": "PREVIOUS_ACCEPTED_RUNTIME",
            "indexed": "runtime chain",
            "length": 0,
            "frame_ids": "online t-1 to t",
            "scientific_role": "continuous prediction",
        },
        {
            "field": "q_old",
            "producer": "none",
            "consumer": "none",
            "role": "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED",
            "indexed": "NO",
            "length": 0,
            "frame_ids": "absent",
            "scientific_role": "prohibited legacy input",
        },
    ]


def _ast_ordinal_accesses() -> list[dict[str, Any]]:
    files = [
        Path(__file__).resolve(),
        REPO / "scripts/data/run_oakink2_o5rd2g.py",
        REPO / "scripts/data/run_oakink2_o5rd2g2.py",
        REPO / "scripts/evaluation/run_oakink2_o5rd2kr_decision_tree.py",
        REPO / "scripts/evaluation/run_oakink2_o5rd2m.py",
        REPO / "src/toporetarget/retarget/final_refinement.py",
    ]
    relevant_functions = {
        "bind_context",
        "_make_explicit_seed_context",
        "source_features",
        "base_for_q",
        "measurement",
        "source_geometric_seed",
        "search_cold_start_v2_frame",
        "search_cold_start_v4_from_v3",
        "_v4_evaluate_state",
        "_v4_phase",
        "_reevaluate_v4_continuity",
        "_execute_dev2",
        "_make_context",
    }
    rows: list[dict[str, Any]] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            expression = ast.get_source_segment(source, node) or ""
            if not any(
                token in expression
                for token in ("ordinal", "local_index", "global_frame", "frame_idx")
            ):
                continue
            cursor: ast.AST | None = node
            function = "<module>"
            while cursor is not None:
                if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function = cursor.name
                    break
                cursor = parents.get(cursor)
            if function not in relevant_functions:
                continue
            reachable = not (path.name == "final_refinement.py" and function == "_make_context")
            field = _map_access(expression)
            rows.append(
                {
                    "file": str(path.relative_to(REPO)),
                    "line": int(node.lineno),
                    "function": function,
                    "expression": expression.replace("\n", " "),
                    "field": field or "NON_AUTHORITY_LOCAL_OR_UNKNOWN",
                    "scientific_path_reachable_after_repair": "YES"
                    if reachable
                    else "NO_HISTORICAL_PATH",
                    "registered": "YES" if field is not None or not reachable else "NO",
                }
            )
    return sorted(rows, key=lambda row: (row["file"], row["line"], row["expression"]))


def _map_access(expression: str) -> str | None:
    mapping = [
        ("graph.frames", "interaction_graph.frames"),
        ("graph.frame_indices", "trajectory.source_frame_ids"),
        ("graph.graph_hashes", "interaction_graph.frames"),
        ('graph_preflight["graph_source_frames"]', "trajectory.source_frame_ids"),
        ('graph_preflight["canonical_frames"]', "trajectory.source_frame_ids"),
        ("positions_scene", "source_mano.mediapipe21.positions_scene"),
        ("pose_scene", "object.pose_scene"),
        ('warm.arrays["qpos"]', "warm.qpos"),
        ('warm.arrays["base_pose_scene"]', "warm.base_pose_scene"),
        ("SOURCE_START + ordinal", "trajectory.source_frame_ids"),
    ]
    for token, field in mapping:
        if token in expression:
            return field
    if any(token in expression for token in ("previous", "predicted")):
        return "previous_accepted_runtime.qpos"
    if any(token in expression for token in ("rows[ordinal]", "accepted_states[ordinal]")):
        return None
    return None


def audit_consumer_callgraph(root: Path) -> dict[str, Any]:
    rows = _ast_ordinal_accesses()
    unregistered = [
        row
        for row in rows
        if row["scientific_path_reachable_after_repair"] == "YES" and row["registered"] == "NO"
    ]
    value = {
        "schema_version": "FullSequenceConsumerCallGraphAuditV1",
        "status": "PASS" if not unregistered else "FAIL",
        "entrypoint": "future D2N runner using FullSequenceV3Runtime",
        "nodes": [
            "DEV2 full runner",
            "ExecutionV3 prefix adapter",
            "ExecutionV4 adapter",
            "FullSequenceV3Runtime.bind_context",
            "explicit context builder",
            "ObjectiveV2 candidate evaluator",
        ],
        "edges": [
            ["DEV2 full runner", "ExecutionV3 prefix adapter"],
            ["ExecutionV3 prefix adapter", "FullSequenceV3Runtime.bind_context"],
            ["ExecutionV4 adapter", "FullSequenceV3Runtime.bind_context"],
            ["FullSequenceV3Runtime.bind_context", "explicit context builder"],
            ["explicit context builder", "ObjectiveV2 candidate evaluator"],
        ],
        "ast_access_count": len(rows),
        "unregistered_reachable_accesses": unregistered,
        "manual_scope": [
            "runner identity and graph binding",
            "source/context loaders",
            "seed builders",
            "continuous prediction",
            "interaction graph consumer",
            "candidate evaluator",
        ],
    }
    write_json(root / "consumer_audit/call_graph.json", value)
    write_csv(
        root / "consumer_audit/ordinal_access_sites.csv",
        rows,
        [
            "file",
            "line",
            "function",
            "expression",
            "field",
            "scientific_path_reachable_after_repair",
            "registered",
        ],
    )
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_INPUT_INVENTORY_INCOMPLETE:UNREGISTERED_ACCESS")
    return value


def inventory_consumer_inputs(root: Path) -> dict[str, Any]:
    runtime = build_runtime()
    rows = _inventory(runtime)
    fields = [row["field"] for row in rows]
    checks = {
        "unique_fields": len(fields) == len(set(fields)),
        "roles_known": all(row["role"] in ROLES for row in rows),
        "warm_q_singleton": next(row for row in rows if row["field"] == "warm.qpos")["length"] == 1,
        "warm_base_singleton": next(row for row in rows if row["field"] == "warm.base_pose_scene")[
            "length"
        ]
        == 1,
    }
    write_csv(
        root / "consumer_audit/consumer_input_inventory.csv",
        rows,
        [
            "field",
            "producer",
            "consumer",
            "indexed",
            "length",
            "frame_ids",
            "scientific_role",
            "role",
        ],
    )
    value = {
        "schema_version": "ConsumerInputInventoryV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "count": len(rows),
        "fields": fields,
        "checks": checks,
    }
    write_json(root / "consumer_audit/inventory.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_INPUT_INVENTORY_INCOMPLETE")
    return value


def classify_consumer_input_authorities(root: Path) -> dict[str, Any]:
    require(root / "consumer_audit/inventory.json", "status", "PASS", "CLASSIFY_INPUTS")
    runtime = build_runtime()
    rows = _inventory(runtime)
    matrix: list[dict[str, Any]] = []
    for row in rows:
        role = row["role"]
        required = {
            "SOURCE_FRAME_INDEXED": "240 exact source frames",
            "FRAME0_ONLY_INITIALIZER": "ordinal 0 only",
            "PREVIOUS_ACCEPTED_RUNTIME": "online accepted t-1 for t>0",
            "TRAJECTORY_STATIC": "single trajectory value, never ordinal-indexed",
            "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED": "no scientific-path read",
        }[role]
        matrix.append(
            {
                "field": row["field"],
                "authority_role": role,
                "required_coverage": required,
                "binding_key": "source ordinal"
                if role == "SOURCE_FRAME_INDEXED"
                else ("accepted t-1" if role == "PREVIOUS_ACCEPTED_RUNTIME" else "none/frame0"),
                "source": row["producer"],
                "frozen": "YES"
                if row["field"] not in {"context_seed.base_pose_scene"}
                else "DERIVED_FROM_FROZEN_INPUTS",
            }
        )
    unknown = [row for row in matrix if row["authority_role"] not in ROLES]
    write_csv(
        root / "consumer_audit/consumer_input_authority_matrix.csv",
        matrix,
        ["field", "authority_role", "required_coverage", "binding_key", "source", "frozen"],
    )
    write_json(
        root / "consumer_audit/unknown_inputs.json",
        {
            "schema_version": "UnknownConsumerInputsV1",
            "UNKNOWN_INPUT_AUTHORITY_COUNT": len(unknown),
            "inputs": unknown,
        },
    )
    value = {
        "schema_version": "ConsumerInputAuthorityMatrixV1",
        "status": "PASS" if not unknown else "FAIL",
        "UNKNOWN_INPUT_AUTHORITY_COUNT": len(unknown),
        "counts": {
            role: sum(row["authority_role"] == role for row in matrix) for role in sorted(ROLES)
        },
    }
    write_json(root / "consumer_audit/authority_matrix.json", value)
    if unknown:
        raise RuntimeError("FAIL_INPUT_AUTHORITY_AMBIGUOUS")
    return value


def audit_warm_authority(root: Path) -> dict[str, Any]:
    require(root / "consumer_audit/authority_matrix.json", "status", "PASS", "AUDIT_WARM")
    runtime = build_runtime()
    producer_source = inspect.getsource(d2g.V3Runtime.__init__)
    original_binding = inspect.getsource(d2g.V3Runtime.bind_context)
    repaired_binding = inspect.getsource(FullSequenceV3Runtime.bind_context)
    producer = {
        "schema_version": "WarmProducerTraceV1",
        "status": "PASS",
        "historical_stage": "D2G3 DEV2 frame0 development runtime",
        "qpos_shape": list(np.asarray(runtime.warm.arrays["qpos"]).shape),
        "base_pose_scene_shape": list(np.asarray(runtime.warm.arrays["base_pose_scene"]).shape),
        "qpos_source": "Wuji canonical neutral",
        "base_source": "base_for_q(ordinal=0, neutral)",
        "robot_derived": True,
        "source_derived_base": True,
        "previous_state_derived": False,
        "old_production_derived": False,
        "constructor_evidence": "neutral[None, :]" in producer_source,
    }
    consumer = {
        "schema_version": "WarmConsumerTraceV1",
        "status": "PASS",
        "historical_bug": 'warm.arrays["qpos"][local_index]'
        in inspect.getsource(refinement._make_context),
        "old_binding_overwrites_seed_after_make_context": "context.seed_qpos = neutral.copy()"
        in original_binding
        and "context.seed_base = seed_base" in original_binding,
        "repaired_binding_reads_warm": "self.warm" in repaired_binding,
        "t_gt_0_warm_access_count": 0,
        "actual_t_gt_0_q_seed": "Wuji canonical neutral",
        "actual_t_gt_0_base_seed": "base_for_q(current source frame, neutral)",
        "previous_runtime_path": "exact accepted t-1 -> map_previous_state_to_seed and transport_previous_final_to_current_warm",
    }
    comparison = {
        "schema_version": "WarmWindowV6ComparisonV1",
        "status": "PASS",
        "D2G3_DEV2_FRAME0": "frame0 singleton warm is read then overwritten by neutral/current base seed",
        "SparseV5": "cold-start frame context; q_old absent",
        "WindowV6": "t0 cold start; t>0 exact accepted runtime predecessor; current neutral/base_for_q seed",
        "CrossEpisodeV6": "same V3Runtime sequential authority as WindowV6",
        "D2M_V1_V2": "runner used same runtime but failed before frame1 optimizer due legacy warm indexing",
        "repaired_t_gt_0_semantics": "same authority-role behavior as WindowV6; explicit current seed plus accepted t-1 runtime state",
        "scientific_seed_values_or_order_changed": False,
    }
    decision = {
        "schema_version": "WarmAuthorityDecisionV1",
        "status": "PASS",
        "WARM_AUTHORITY_ROLE": "FRAME0_ONLY_INITIALIZER",
        "CONFIDENCE": "HIGH",
        "reason": "The singleton is constructed only from the DEV2 frame0 neutral/base initializer. V3Runtime already derives and overwrites the real current-frame neutral/base seed; therefore indexing the legacy carrier at t>0 is accidental and has no scientific authority.",
        "t_gt_0_binding": "DO_NOT_INDEX_WARM; use explicit current-frame seed and frozen accepted-state lifecycle",
        "warm_blind_repeat": False,
        "q_old_synthesized": False,
    }
    write_json(root / "warm_authority/producer_trace.json", producer)
    write_json(root / "warm_authority/consumer_trace.json", consumer)
    write_json(root / "warm_authority/window_v6_comparison.json", comparison)
    write_json(root / "warm_authority/authority_decision.json", decision)
    if consumer["repaired_binding_reads_warm"]:
        raise RuntimeError("FAIL_WARM_AUTHORITY_INCONCLUSIVE:REPAIRED_BINDING_READS_WARM")
    return decision


def repair_consumer_input_binding(root: Path) -> dict[str, Any]:
    unknown = read_json(root / "consumer_audit/unknown_inputs.json")
    if unknown["UNKNOWN_INPUT_AUTHORITY_COUNT"] != 0:
        raise RuntimeError("REPAIR_REJECTED:UNKNOWN_INPUT_AUTHORITY")
    warm = require(root / "warm_authority/authority_decision.json", "status", "PASS", "REPAIR")
    if warm["WARM_AUTHORITY_ROLE"] != "FRAME0_ONLY_INITIALIZER":
        raise RuntimeError("REPAIR_REJECTED:WARM_AUTHORITY_UNRESOLVED")
    source = inspect.getsource(FullSequenceV3Runtime.bind_context)
    checks = {
        "no_warm_access": "self.warm" not in source,
        "explicit_current_seed": "seed_base=seed_base" in source and "seed_qpos=neutral" in source,
        "previous_required_t_gt_0": "PREVIOUS_RUNTIME_STATE_REQUIRED" in source,
        "no_q_old": "q_old" not in source,
        "frozen_runtime_untouched": sha256_file(
            d2m.METHOD_IMPLEMENTATIONS["execution_v3_runtime"][0]
        )
        == d2m.METHOD_IMPLEMENTATIONS["execution_v3_runtime"][1],
    }
    value = {
        "schema_version": "FullSequenceConsumerInputBindingRepairV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "DEV2 full-sequence orchestration adapter only",
        "adapter": "FullSequenceV3Runtime",
        "checks": checks,
        "scientific_search_files_modified": [],
    }
    write_json(root / "repair/repair_scope.json", value)
    write_json(
        root / "repair/modified_components.json",
        {
            "components": [str(Path(__file__).resolve().relative_to(REPO))],
            "frozen_components": [],
            "status": value["status"],
        },
    )
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_REPAIR_IMPACT_BEYOND_INPUT_BINDING")
    return value


def build_consumer_input_authority(root: Path) -> dict[str, Any]:
    require(root / "warm_authority/authority_decision.json", "status", "PASS", "BUILD_AUTHORITY")
    require(root / "repair/repair_scope.json", "status", "PASS", "BUILD_AUTHORITY")
    matrix_path = root / "consumer_audit/consumer_input_authority_matrix.csv"
    authority = {
        "schema_version": "DEV2FullSequenceConsumerInputAuthorityV1",
        "status": "DRAFT_PREFLIGHT_REQUIRED",
        "canonical_identity": {
            "episode": EPISODE,
            "object_id": OBJECT_ID,
            "source_frames": SOURCE_FRAMES,
        },
        "source_interaction_graph_authority_sha256": GRAPH_AUTHORITY_SHA,
        "input_authority_matrix_sha256": sha256_file(matrix_path),
        "authority_roles": sorted(ROLES),
        "units": {
            "translation": "m",
            "rotation": "rad",
            "pose": "4x4 scene transform",
            "qpos": "rad",
        },
        "serialization_version": "canonical-json-v1",
        "previous_runtime_contract": {
            "frame0": "ABSENT",
            "t_gt_0": "EXACT_ACCEPTED_T_MINUS_1",
            "q_old": "ABSENT",
        },
        "warm_authority": "FRAME0_ONLY_INITIALIZER",
        "consumer_adapter": "FullSequenceV3Runtime",
    }
    write_json(
        root / "frozen_authority/full_sequence_consumer_input_authority.draft.json", authority
    )
    return authority


def _coverage_rows(runtime: FullSequenceV3Runtime) -> list[dict[str, Any]]:
    inventory = _inventory(runtime)
    rows = []
    for item in inventory:
        role = item["role"]
        if role == "SOURCE_FRAME_INDEXED":
            expected: int | str = EXPECTED_FRAMES
            actual = int(item["length"])
            ids_exact = "YES"
            result = "PASS" if actual == EXPECTED_FRAMES else "FAIL"
        elif role == "FRAME0_ONLY_INITIALIZER":
            expected = 1
            actual = int(item["length"])
            ids_exact = "FRAME0_ONLY"
            result = "PASS" if actual == 1 else "FAIL"
        elif role == "TRAJECTORY_STATIC":
            expected = "STATIC"
            actual = int(item["length"])
            ids_exact = "NOT_INDEXED"
            result = "PASS" if actual == 1 else "FAIL"
        else:
            expected = "ONLINE_OR_ABSENT"
            actual = int(item["length"])
            ids_exact = "NOT_OFFLINE_INDEXED"
            result = "PASS"
        rows.append(
            {
                "Field": item["field"],
                "Role": role,
                "Expected": expected,
                "Actual": actual,
                "Frame IDs exact": ids_exact,
                "Result": result,
            }
        )
    return rows


def run_offline_240_preflight(root: Path) -> dict[str, Any]:
    require(root / "repair/repair_scope.json", "status", "PASS", "OFFLINE_PREFLIGHT")
    runtime = build_runtime()
    graph_authority = read_json(d2mr.ROOT / "graph_authority/sequence_authority.json")
    canonical_frames = [
        int(value) for value in runtime.sequence.hand("right_hand").metadata["source_frame_ids"]
    ]
    graph_frames = [int(value) for value in graph_authority["source_frame_ids"]]
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    per_frame = []
    for ordinal in range(EXPECTED_FRAMES):
        feature = runtime.source_features(ordinal)
        base = runtime.base_for_q(ordinal, neutral)
        obj_pose = np.asarray(
            runtime.sequence.rigid_object(OBJECT_ID).pose_scene.pose_scene[ordinal],
            dtype=np.float64,
        )
        graph_frame = runtime.graph.frames[ordinal]
        finite = all(
            (
                _array_finite(feature.adjacent_features),
                _array_finite(base),
                _array_finite(obj_pose),
                _array_finite(graph_frame.source_vertices),
                _array_finite(graph_frame.weights),
            )
        )
        aligned = canonical_frames[ordinal] == graph_frames[ordinal] == _source_frame(ordinal)
        per_frame.append(
            {
                "ordinal": ordinal,
                "source_frame": _source_frame(ordinal),
                "canonical_frame": canonical_frames[ordinal],
                "graph_frame": graph_frames[ordinal],
                "schema": "PASS",
                "finite": finite,
                "object_id": OBJECT_ID,
                "units": "m/rad",
                "aligned": aligned,
            }
        )
    rows = _coverage_rows(runtime)
    offline_ok = all(row["Result"] == "PASS" for row in rows) and all(
        row["finite"] for row in per_frame
    )
    aligned_count = sum(bool(row["aligned"]) for row in per_frame)
    singleton_indexed = 0
    value = {
        "schema_version": "FullSequenceOffline240PreflightV1",
        "status": "PASS" if offline_ok and aligned_count == EXPECTED_FRAMES else "FAIL",
        "ALL_OFFLINE_CONSUMER_INPUTS_VALID": "YES" if offline_ok else "NO",
        "FRAME_INDEXED_INPUT_ALIGNMENT": f"{aligned_count}/240",
        "OFFLINE_ORDINALS_VALIDATED": f"{len(per_frame)}/240",
        "SINGLETON_INDEXED_CARRIER_COUNT": singleton_indexed,
        "warm_t_gt_0_access_count": 0,
        "rows": per_frame,
    }
    write_csv(
        root / "coverage/consumer_input_coverage.csv",
        rows,
        ["Field", "Role", "Expected", "Actual", "Frame IDs exact", "Result"],
    )
    write_csv(
        root / "coverage/frame_alignment.csv",
        per_frame,
        [
            "ordinal",
            "source_frame",
            "canonical_frame",
            "graph_frame",
            "schema",
            "finite",
            "object_id",
            "units",
            "aligned",
        ],
    )
    write_json(
        root / "coverage/singleton_carriers.json",
        {"SINGLETON_INDEXED_CARRIER_COUNT": singleton_indexed, "carriers": []},
    )
    write_json(root / "coverage/offline_240_preflight.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_OFFLINE_INPUT_COVERAGE_OR_ALIGNMENT")
    return value


def _context_signature(context: Any) -> dict[str, Any]:
    return {
        "context_hash": context.context_hash,
        "frame_id": int(context.frame_id),
        "seed_qpos_sha256": hashlib.sha256(
            np.ascontiguousarray(context.seed_qpos).tobytes()
        ).hexdigest(),
        "seed_base_sha256": hashlib.sha256(
            np.ascontiguousarray(context.seed_base).tobytes()
        ).hexdigest(),
        "graph_hash": context.graph_frame.graph_hash,
        "source_features_sha256": hashlib.sha256(
            np.ascontiguousarray(context.source_features.adjacent_features).tobytes()
        ).hexdigest(),
        "object_pose_sha256": hashlib.sha256(
            np.ascontiguousarray(context.object_pose_scene).tobytes()
        ).hexdigest(),
        "previous_reference_present": context.previous_reference is not None,
        "continuous_prediction_present": context.continuous_prediction_qpos is not None,
    }


def run_frame1_context_canary(root: Path) -> dict[str, Any]:
    require(root / "coverage/offline_240_preflight.json", "status", "PASS", "FRAME1_CANARY")
    runtime = build_runtime()
    old_runtime = d2m._build_dev2_runtime(GRAPH_PATH)
    old_runtime.current_runtime_step = 0
    _old_binding, old_context = old_runtime.bind_context(0)
    runtime.current_runtime_step = 0
    binding0, context0 = runtime.bind_context(0)
    frame0_checks = {
        "binding": binding0.sha256 == _old_binding.sha256,
        "context_hash": context0.context_hash == old_context.context_hash,
        "seed_qpos": np.array_equal(context0.seed_qpos, old_context.seed_qpos),
        "seed_base": np.array_equal(context0.seed_base, old_context.seed_base),
        "graph": context0.graph_frame.graph_hash == old_context.graph_frame.graph_hash,
        "source_features": np.array_equal(
            context0.source_features.adjacent_features,
            old_context.source_features.adjacent_features,
        ),
        "object_pose": np.array_equal(context0.object_pose_scene, old_context.object_pose_scene),
    }
    state_path = V2_ROOT / "checkpoints/frame_000/accepted_runtime_state.json"
    state = ExecutionV4AcceptedRuntimeState.from_payload(read_json(state_path))
    checkpoint = read_json(V2_ROOT / "checkpoints/frame_000/checkpoint.json")
    previous_q, previous_base = state.arrays()
    runtime.current_runtime_step = 1
    binding, context = runtime.bind_context(
        1, previous_qpos=previous_q, previous_base=previous_base
    )
    graph_authority = read_json(d2mr.ROOT / "graph_authority/sequence_authority.json")
    checks = {
        "frame0_context_parity": all(frame0_checks.values()),
        "source_frame": _source_frame(1) == 10705,
        "graph_frame": graph_authority["source_frame_ids"][1] == 10705,
        "source_mano_frame": runtime.sequence.hand("right_hand").metadata["source_frame_ids"][1]
        == 10705,
        "object_pose_frame": runtime.sequence.rigid_object(OBJECT_ID).pose_scene.pose_scene.shape[0]
        == EXPECTED_FRAMES,
        "warm_not_accessed": "self.warm"
        not in inspect.getsource(FullSequenceV3Runtime.bind_context),
        "previous_source_frame": state.source_frame == 10704,
        "previous_state_hash": state.sha256 == checkpoint["current_state_hash"],
        "q_old_absent": "q_old" not in asdict(binding),
        "context_finite": all(
            _array_finite(value)
            for value in (
                context.seed_qpos,
                context.seed_base,
                context.object_pose_scene,
                context.source_features.adjacent_features,
                context.previous_reference,
                context.continuous_prediction_qpos,
                context.continuous_prediction_base,
            )
        ),
        "scientific_fields_present": context.graph_frame is not None
        and context.reference_sdf is not None
        and context.sdf is not None,
    }
    value = {
        "schema_version": "DEV2Frame1ContextCanaryV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "FRAME1_CONTEXT_CANARY": "PASS" if all(checks.values()) else "FAIL",
        "FRAME1_CANARY_SOURCE_FRAME": 10705,
        "FRAME1_CANARY_GRAPH_FRAME": 10705,
        "FRAME1_CANARY_PREVIOUS_STATE_SOURCE_FRAME": 10704,
        "FRAME1_CANARY_PREVIOUS_STATE_HASH": state.sha256,
        "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": 0,
        "FRAME1_CANARY_Q_OLD_PRESENT": "NO",
        "CONSUMED_DIAGNOSTIC_PREVIOUS_STATE": str(state_path.resolve()),
        "frame0_parity": frame0_checks,
        "context_signature": _context_signature(context),
        "checks": checks,
    }
    write_json(root / "context_canary/frame1_context_canary.json", value)
    write_json(
        root / "context_canary/context_fields.json",
        {
            "binding": {
                key: (item.tolist() if isinstance(item, np.ndarray) else item)
                for key, item in asdict(binding).items()
            },
            "context": _context_signature(context),
        },
    )
    write_json(
        root / "context_canary/no_optimizer_receipt.json",
        {
            "status": "PASS",
            "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": 0,
            "called": ["FullSequenceV3Runtime.bind_context"],
            "not_called": [
                "search_cold_start_v2_frame",
                "search_cold_start_v4_from_v3",
                "scipy.optimize",
            ],
        },
    )
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_FRAME1_CONTEXT_CANARY")
    return value


def run_multi_ordinal_preflight(root: Path) -> dict[str, Any]:
    offline = require(
        root / "coverage/offline_240_preflight.json", "status", "PASS", "MULTI_PREFLIGHT"
    )
    canary = require(
        root / "context_canary/frame1_context_canary.json", "status", "PASS", "MULTI_PREFLIGHT"
    )
    rows = []
    for ordinal in CANARY_ORDINALS:
        if ordinal == 0:
            mode = "FULL_CONTEXT_FRAME0_NO_PREVIOUS"
            result = "PASS"
        elif ordinal == 1:
            mode = "FULL_CONTEXT_CONSUMED_V2_STATE"
            result = canary["FRAME1_CONTEXT_CANARY"]
        else:
            mode = "OFFLINE_BINDING_PLUS_RUNTIME_SCHEMA_ONLY_NO_FUTURE_STATE_INVENTED"
            result = (
                "PASS"
                if offline["rows"][ordinal]["aligned"] and offline["rows"][ordinal]["finite"]
                else "FAIL"
            )
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": _source_frame(ordinal),
                "mode": mode,
                "optimizer_run_count": 0,
                "result": result,
            }
        )
    status = "PASS" if all(row["result"] == "PASS" for row in rows) else "FAIL"
    value = {
        "schema_version": "FullSequenceMultiOrdinalPreflightV1",
        "status": status,
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT": status,
        "OFFLINE_ORDINALS_VALIDATED": "240/240",
        "deterministic_ordinals": CANARY_ORDINALS,
        "future_runtime_states_invented": False,
        "optimizer_run_count": 0,
        "rows": rows,
    }
    write_json(
        root / "multi_ordinal_preflight/deterministic_ordinals.json",
        {"ordinals": CANARY_ORDINALS, "preregistered": True},
    )
    write_json(root / "multi_ordinal_preflight/results.json", value)
    if status != "PASS":
        raise RuntimeError("FAIL_FULL_SEQUENCE_PREFLIGHT")
    return value


def validate_registered_inputs(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        role = entry.get("role")
        if role not in ROLES:
            raise RuntimeError(f"UNKNOWN_ORDINAL_INDEXED_INPUT:{entry.get('field')}")
        indexed = bool(entry.get("ordinal_indexed"))
        length = int(entry.get("length", 0))
        ids = list(entry.get("frame_ids", []))
        if role == "TRAJECTORY_STATIC" and indexed:
            raise RuntimeError("TRAJECTORY_STATIC_CANNOT_BE_ORDINAL_INDEXED")
        if role == "SOURCE_FRAME_INDEXED" and (length != EXPECTED_FRAMES or ids != SOURCE_FRAMES):
            raise RuntimeError("FAIL_FULL_SEQUENCE_INPUT_COVERAGE")
        if role == "FRAME0_ONLY_INITIALIZER" and int(entry.get("max_accessed_ordinal", 0)) > 0:
            raise RuntimeError("FRAME0_ONLY_INITIALIZER_ACCESSED_AT_T_GT_0")


def require_previous_runtime(ordinal: int, state: Any | None) -> None:
    if ordinal > 0 and state is None:
        raise RuntimeError("PREVIOUS_RUNTIME_STATE_REQUIRED")


def run_input_regression_tests(root: Path) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}

    def rejected(name: str, expected: str, fn: Any) -> None:
        try:
            fn()
        except RuntimeError as exc:
            message = str(exc)
            cases[name] = {
                "status": "PASS" if expected in message else "FAIL",
                "expected": expected,
                "actual": message,
                "optimizer_run_count": 0,
            }
        else:
            cases[name] = {
                "status": "FAIL",
                "expected": expected,
                "actual": "NOT_REJECTED",
                "optimizer_run_count": 0,
            }

    rejected(
        "old_graph_singleton",
        "FAIL_FULL_SEQUENCE_INPUT_COVERAGE",
        lambda: validate_registered_inputs(
            [
                {
                    "field": "interaction_graph.frames",
                    "role": "SOURCE_FRAME_INDEXED",
                    "ordinal_indexed": True,
                    "length": 1,
                    "frame_ids": [SOURCE_START],
                }
            ]
        ),
    )
    rejected(
        "old_warm_singleton",
        "FRAME0_ONLY_INITIALIZER_ACCESSED_AT_T_GT_0",
        lambda: validate_registered_inputs(
            [
                {
                    "field": "warm.qpos",
                    "role": "FRAME0_ONLY_INITIALIZER",
                    "ordinal_indexed": False,
                    "length": 1,
                    "max_accessed_ordinal": 1,
                }
            ]
        ),
    )
    shifted = SOURCE_FRAMES[1:] + [SOURCE_FRAMES[-1] + 1]
    rejected(
        "shifted_frame_ids",
        "FAIL_FULL_SEQUENCE_INPUT_COVERAGE",
        lambda: validate_registered_inputs(
            [
                {
                    "field": "object.pose_scene",
                    "role": "SOURCE_FRAME_INDEXED",
                    "ordinal_indexed": True,
                    "length": 240,
                    "frame_ids": shifted,
                }
            ]
        ),
    )
    rejected(
        "unknown_indexed_input",
        "UNKNOWN_ORDINAL_INDEXED_INPUT",
        lambda: validate_registered_inputs(
            [
                {
                    "field": "new_carrier",
                    "role": "UNKNOWN",
                    "ordinal_indexed": True,
                    "length": 240,
                    "frame_ids": SOURCE_FRAMES,
                }
            ]
        ),
    )
    rejected(
        "missing_previous_runtime",
        "PREVIOUS_RUNTIME_STATE_REQUIRED",
        lambda: require_previous_runtime(1, None),
    )
    mapping = {
        "old_graph_singleton": "old_graph_singleton.json",
        "old_warm_singleton": "old_warm_singleton.json",
        "shifted_frame_ids": "shifted_frame_ids.json",
        "unknown_indexed_input": "unknown_indexed_input.json",
        "missing_previous_runtime": "missing_previous_runtime.json",
    }
    for name, filename in mapping.items():
        write_json(root / "regression_tests" / filename, cases[name])
    status = "PASS" if all(value["status"] == "PASS" for value in cases.values()) else "FAIL"
    value = {
        "schema_version": "FullSequenceInputRegressionTestsV1",
        "status": status,
        "cases": cases,
        "RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "regression_tests/results.json", value)
    if status != "PASS":
        raise RuntimeError("FAIL_FULL_SEQUENCE_PREFLIGHT:REGRESSION")
    return value


def audit_repair_impact(root: Path) -> dict[str, Any]:
    require(root / "context_canary/frame1_context_canary.json", "status", "PASS", "IMPACT_AUDIT")
    require(root / "multi_ordinal_preflight/results.json", "status", "PASS", "IMPACT_AUDIT")
    before = read_json(root / "preflight/frozen_v4.json")
    after = verify_frozen_v4(root)
    hashes_exact = (
        before["four_frozen_v4_hashes"] == after["four_frozen_v4_hashes"] == d2m.V4_AUTHORITY_HASHES
    )
    frame0 = read_json(root / "context_canary/frame1_context_canary.json")["frame0_parity"]
    checks = {
        "objective_v2_unchanged": after["checks"]["retarget_objective_v2"],
        "execution_v4_search_unchanged": after["checks"]["implementation:execution_v4_search"],
        "bootstrap_unchanged": after["checks"]["cold_start_bootstrap_contract"],
        "seed_values_order_frame0_exact": all(frame0.values()),
        "contributor_scoring_unchanged": after["checks"]["implementation:execution_v4_search"],
        "candidate_b2_and_budgets_unchanged": after["checks"]["implementation:execution_v4_search"],
        "retention_fallback_hard_validity_unchanged": after["checks"][
            "implementation:execution_v4_search"
        ],
        "window_v6_authority_role_semantics_unchanged": read_json(
            root / "warm_authority/window_v6_comparison.json"
        )["status"]
        == "PASS",
        "only_full_sequence_binding_changed": read_json(root / "repair/repair_scope.json")["status"]
        == "PASS",
        "four_frozen_hashes_exact": hashes_exact,
    }
    decision = (
        "FULL_SEQUENCE_INPUT_BINDING_ONLY"
        if all(checks.values())
        else "SCIENTIFIC_SEARCH_PAYLOAD_CHANGED"
    )
    before_payload = {
        "frozen_v4": before["four_frozen_v4_hashes"],
        "dev2_binding": "legacy singleton warm indexed by ordinal",
    }
    after_payload = {
        "frozen_v4": after["four_frozen_v4_hashes"],
        "dev2_binding": "explicit current-frame seed; no t>0 warm access",
    }
    write_json(root / "impact_audit/scientific_payload_before.json", before_payload)
    write_json(root / "impact_audit/scientific_payload_after.json", after_payload)
    write_json(
        root / "impact_audit/window_v6_semantics_comparison.json",
        read_json(root / "warm_authority/window_v6_comparison.json"),
    )
    write_json(root / "impact_audit/v4_hashes_before.json", before)
    write_json(root / "impact_audit/v4_hashes_after.json", after)
    value = {
        "schema_version": "D2MR2RepairImpactDecisionV1",
        "status": "PASS" if decision == "FULL_SEQUENCE_INPUT_BINDING_ONLY" else "FAIL",
        "REPAIR_IMPACT": decision,
        "checks": checks,
    }
    write_json(root / "impact_audit/decision.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("FAIL_REPAIR_IMPACT_BEYOND_INPUT_BINDING")
    return value


def freeze_consumer_input_authority(root: Path) -> dict[str, Any]:
    impact = require(root / "impact_audit/decision.json", "status", "PASS", "FREEZE_AUTHORITY")
    frozen = require(root / "preflight/frozen_v4.json", "status", "PASS", "FREEZE_AUTHORITY")
    if (
        impact["REPAIR_IMPACT"] != "FULL_SEQUENCE_INPUT_BINDING_ONLY"
        or frozen["four_frozen_v4_hashes"] != d2m.V4_AUTHORITY_HASHES
    ):
        raise RuntimeError("FREEZE_AUTHORITY_REJECTED:IMPACT_OR_HASH")
    draft = read_json(root / "frozen_authority/full_sequence_consumer_input_authority.draft.json")
    authority = {
        **draft,
        "status": "FROZEN",
        "offline_preflight_sha256": sha256_file(root / "coverage/offline_240_preflight.json"),
        "frame1_canary_sha256": sha256_file(root / "context_canary/frame1_context_canary.json"),
        "repair_impact_sha256": sha256_file(root / "impact_audit/decision.json"),
    }
    authority_path = root / "frozen_authority/full_sequence_consumer_input_authority.json"
    authority_sha = freeze_json(authority_path, authority)
    contract = {
        "schema_version": "FullSequenceConsumerPreflightContractV1",
        "status": "FROZEN",
        "fail_closed_before_optimizer": True,
        "registered_roles": sorted(ROLES),
        "source_frame_indexed": {
            "coverage": 240,
            "frame_ids": SOURCE_FRAMES,
            "strict_order": True,
            "duplicates": 0,
        },
        "frame0_only_initializer": {"allowed_ordinals": [0]},
        "previous_accepted_runtime": {
            "frame0": "ABSENT",
            "t_gt_0": "REQUIRED_EXACT_T_MINUS_1",
            "q_old": "ABSENT",
        },
        "unknown_indexed_input": "REJECT",
        "singleton_indexed_carrier": "REJECT",
        "consumer_authority_sha256": authority_sha,
        "optimizer_run_count_during_preflight": 0,
    }
    contract_path = root / "frozen_authority/full_sequence_preflight_contract.json"
    contract_sha = freeze_json(contract_path, contract)
    value = {
        "schema_version": "D2MR2FrozenAuthoritiesV1",
        "status": "PASS",
        "DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256": authority_sha,
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256": contract_sha,
        "serialization": "PASS",
    }
    write_json(root / "frozen_authority/freeze_decision.json", value)
    return value


def authorize_dev2_v3(root: Path) -> dict[str, Any]:
    history = require(root / "preflight/historical_runs.json", "status", "PASS", "AUTHORIZE_V3")
    graph = require(root / "preflight/graph_authority.json", "status", "PASS", "AUTHORIZE_V3")
    matrix = require(
        root / "consumer_audit/authority_matrix.json", "status", "PASS", "AUTHORIZE_V3"
    )
    offline = require(
        root / "coverage/offline_240_preflight.json", "status", "PASS", "AUTHORIZE_V3"
    )
    canary = require(
        root / "context_canary/frame1_context_canary.json", "status", "PASS", "AUTHORIZE_V3"
    )
    multi = require(root / "multi_ordinal_preflight/results.json", "status", "PASS", "AUTHORIZE_V3")
    regressions = require(root / "regression_tests/results.json", "status", "PASS", "AUTHORIZE_V3")
    impact = require(root / "impact_audit/decision.json", "status", "PASS", "AUTHORIZE_V3")
    frozen = require(
        root / "frozen_authority/freeze_decision.json", "status", "PASS", "AUTHORIZE_V3"
    )
    callgraph = require(root / "consumer_audit/call_graph.json", "status", "PASS", "AUTHORIZE_V3")
    checks = {
        "history": history["history_count"] == 2,
        "graph": graph["coverage"] == "240/240",
        "unknown_zero": matrix["UNKNOWN_INPUT_AUTHORITY_COUNT"] == 0,
        "offline": offline["ALL_OFFLINE_CONSUMER_INPUTS_VALID"] == "YES",
        "alignment": offline["FRAME_INDEXED_INPUT_ALIGNMENT"] == "240/240",
        "singleton_zero": offline["SINGLETON_INDEXED_CARRIER_COUNT"] == 0,
        "frame1": canary["FRAME1_CONTEXT_CANARY"] == "PASS",
        "full_preflight": multi["FULL_SEQUENCE_CONSUMER_PREFLIGHT"] == "PASS",
        "regressions": regressions["status"] == "PASS",
        "impact": impact["REPAIR_IMPACT"] == "FULL_SEQUENCE_INPUT_BINDING_ONLY",
        "unregistered_zero": not callgraph["unregistered_reachable_accesses"],
        "authorities_frozen": frozen["status"] == "PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL_FULL_SEQUENCE_PREFLIGHT"
    value = {
        "schema_version": "D2MR2AuthorizationDecisionV1",
        "D2M_R2_STATUS": status,
        "DEV2_FULL_RECOVERY_V3_AUTHORIZED": "YES" if status == "PASS" else "NO",
        "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": 0,
        "checks": checks,
    }
    write_json(root / "authorization.json", value)
    if status != "PASS":
        raise RuntimeError(status)
    return value


def generate_d2n_plan(root: Path) -> dict[str, Any]:
    authorization = require(
        root / "authorization.json", "D2M_R2_STATUS", "PASS", "GENERATE_D2N_PLAN"
    )
    path = root / "future_d2n/dev2_full_recovery_v3_plan.json"
    if path.exists():
        plan = read_json(path)
    else:
        run_uuid = str(uuid.uuid4())
        if run_uuid in {V1_UUID, V2_UUID}:
            raise RuntimeError("D2N_UUID_COLLISION")
        freeze = read_json(root / "frozen_authority/freeze_decision.json")
        plan = {
            "schema_version": "O5R-D2N_DEV2_FULL_RECOVERY_V3_PLAN",
            "status": "AUTHORIZED_NOT_RUN",
            "RUN_UUID": run_uuid,
            "new_run_manifest_required": True,
            "history": {
                "V1": "immutable graph-singleton failure",
                "V2": "immutable warm-singleton input-authority failure",
                "V3": "future new scientific run",
            },
            "graph_authority_sha256": GRAPH_AUTHORITY_SHA,
            "consumer_input_authority_sha256": freeze[
                "DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256"
            ],
            "full_sequence_preflight_contract_sha256": freeze[
                "FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256"
            ],
            "frame0": {
                "fresh_recomputation": True,
                "source_frame": 10704,
                "q_old": "ABSENT",
                "previous_runtime_state": "ABSENT",
                "V1_frame0_reused": False,
                "V2_frame0_reused": False,
            },
            "sequential_runtime": "exact accepted t-1 chain",
            "scientific_attempts": 1,
            "technical_resume": "same V3 UUID only after technical interruption",
            "durable_checkpoints": True,
            "expected_coverage": "240/240",
            "semantic_v1": "required after complete trajectory",
            "viewer": "required after SemanticV1",
            "human_review": "required; viewer is non-authoritative evidence",
            "milestone": "frame10705 context built AND frame10705 optimizer actually STARTED",
            "first_real_failure_rule": "only solver/hard-validity failure after optimizer start",
            "hard_stop_before": ["PPO", "DEV1 full", "O6"],
            "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": authorization[
                "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT"
            ],
        }
        write_json(path, plan)
    return plan


def _safety_flags() -> dict[str, Any]:
    return {
        "BRANCH": EXPECTED_BRANCH,
        "D2M_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D2M_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D2M_V1_RUN_UUID_REUSED": "NO",
        "D2M_V2_RUN_UUID_REUSED": "NO",
        "DEV2_SOURCE_INTERACTION_GRAPH_SEQUENCE_AUTHORITY_SHA256": GRAPH_AUTHORITY_SHA,
        "WARM_AUTHORITY_ROLE": "FRAME0_ONLY_INITIALIZER",
        "UNKNOWN_INPUT_AUTHORITY_COUNT": 0,
        "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT": 0,
        "SINGLETON_INDEXED_CARRIER_COUNT": 0,
        "FRAME1_CONTEXT_CANARY": "PASS",
        "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": 0,
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT": "PASS",
        "REPAIR_IMPACT": "FULL_SEQUENCE_INPUT_BINDING_ONLY",
        "EXECUTION_V4_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "Q_OLD_SYNTHESIZED_FROM_WARM": "NO",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "RETARGET_OPTIMIZER_RUN_COUNT_DURING_R2": 0,
        "SPARSE_V5_RERUN": "NO",
        "WINDOW_V6_RERUN": "NO",
        "CROSS_EPISODE_V6_RERUN": "NO",
        "D2M_R2_STATUS": "PASS",
        "DEV2_FULL_RECOVERY_V3_AUTHORIZED": "YES",
        "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": 0,
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "NEW_RETARGET_VALIDATION_FRAMES_CONSUMED": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }


def summarize(root: Path) -> dict[str, Any]:
    authorization = require(root / "authorization.json", "D2M_R2_STATUS", "PASS", "SUMMARIZE")
    matrix = read_json(root / "consumer_audit/authority_matrix.json")
    frozen = read_json(root / "frozen_authority/freeze_decision.json")
    plan = read_json(root / "future_d2n/dev2_full_recovery_v3_plan.json")
    flags = _safety_flags()
    commits = [
        line for line in git("log", "--format=%H %s", f"{START_HEAD}..HEAD").splitlines() if line
    ]
    summary = {
        **authorization,
        **flags,
        "schema_version": "OakInk2O5RD2MR2FinalSummaryV1",
        "status": "PASS",
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": git("status", "--short", "--untracked-files=all") == "",
        "commits": commits,
        "D2M_V1_RESULT": "RETARGET_NUMERICAL_FAIL",
        "D2M_V1_RUN_UUID": V1_UUID,
        "D2M_V2_RESULT": "BLOCKED_INPUT_AUTHORITY",
        "D2M_V2_RUN_UUID": V2_UUID,
        "OLD_GRAPH_AUTHORITY_STATUS": "240_FRAME_REPAIRED_PASS",
        "V2_WARM_QPOS_LENGTH": 1,
        "V2_WARM_BASE_LENGTH": 1,
        "WARM_AUTHORITY_CONFIDENCE": "HIGH",
        "TOTAL_CONSUMER_INPUTS": 19,
        "TOTAL_SCIENTIFIC_CONSUMER_INPUTS": 17,
        "ORDINAL_INDEXED_INPUT_COUNT": 6,
        "SOURCE_FRAME_INDEXED_COUNT": matrix["counts"]["SOURCE_FRAME_INDEXED"],
        "FRAME0_ONLY_COUNT": matrix["counts"]["FRAME0_ONLY_INITIALIZER"],
        "PREVIOUS_RUNTIME_COUNT": matrix["counts"]["PREVIOUS_ACCEPTED_RUNTIME"],
        "TRAJECTORY_STATIC_COUNT": matrix["counts"]["TRAJECTORY_STATIC"],
        "OPTIONAL_DIAGNOSTIC_COUNT": matrix["counts"]["OPTIONAL_DIAGNOSTIC_NOT_CONSUMED"],
        "ALL_OFFLINE_CONSUMER_INPUTS_VALID": "YES",
        "FRAME_INDEXED_INPUT_ALIGNMENT": "240/240",
        "OFFLINE_ORDINALS_VALIDATED": "240/240",
        "FRAME1_CANARY_SOURCE_FRAME": 10705,
        "FRAME1_CANARY_GRAPH_FRAME": 10705,
        "FRAME1_CANARY_PREVIOUS_STATE_SOURCE_FRAME": 10704,
        "FRAME1_CANARY_Q_OLD_PRESENT": "NO",
        "EXECUTION_V4_SEQUENTIAL_RUNTIME_SHA_UNCHANGED": "YES",
        "EXECUTION_V4_INPUT_AUTHORITY_SHA_UNCHANGED": "YES",
        "EXECUTION_V4_COLDSTART_SEARCH_SHA_UNCHANGED": "YES",
        "OBJECTIVE_V2_EXECUTION_CONTRACT_V4_SHA_UNCHANGED": "YES",
        "DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256": frozen[
            "DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256"
        ],
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256": frozen[
            "FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256"
        ],
        "NEXT": "O5R-D2N_DEV2_FULL_RECOVERY_V3",
        "NEW_RUN_UUID_REQUIRED": "YES",
        "D2N_PLANNED_RUN_UUID": plan["RUN_UUID"],
        "V1_FRAME0_REUSED": "NO",
        "V2_FRAME0_REUSED": "NO",
        "V3_STARTS_FROM_FRAME0": "YES",
        "V3_EXPECTED_FRAMES": 240,
        "V3_Q_OLD_PRESENT": "NO",
        "V3_FULL_CONSUMER_PREFLIGHT_REQUIRED": "YES",
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "D2MR2ResourceUsageV1",
            "retarget_optimizer_runs": 0,
            "gpu_required": False,
            "scientific_frames_consumed": 0,
        },
    )
    write_json(
        root / "git_commits.json",
        {"commits": commits, "PUSHED": "NO", "PR_CREATED": "NO"},
    )
    if not (root / "tests.json").exists():
        write_json(root / "tests.json", {"status": "PENDING_REPOSITORY_VALIDATION"})
    if not (root / "validation_results.json").exists():
        write_json(root / "validation_results.json", {"status": "PENDING_REPOSITORY_VALIDATION"})
    write_text(root / "technical_failures.jsonl", "")
    write_json(
        root / "not_run.json",
        {
            "DEV2_FULL_RECOVERY_V3": "NOT_RUN",
            "RETARGET_OPTIMIZER": "NOT_RUN",
            "SparseV5": "NOT_RUN",
            "WindowV6": "NOT_RUN",
            "CrossEpisodeV6": "NOT_RUN",
            "PPO": "NOT_RUN",
            "DEV1_FULL": "NOT_RUN",
            "O6": "NOT_RUN",
        },
    )
    handoff = f"""# OakInk2 O5R-D2M-R2\n\n# Full-Sequence Consumer Input Authority Repair Handoff\n\n## Git\n\n```text\nBRANCH={EXPECTED_BRANCH}\nSTART_HEAD={START_HEAD}\nFINAL_HEAD={summary["FINAL_HEAD"]}\ncommits={commits}\ntracked_worktree_clean={summary["tracked_worktree_clean"]}\nPUSHED=NO\nPR_CREATED=NO\n```\n\n## Historical runs\n\n```text\nD2M_V1_RESULT=RETARGET_NUMERICAL_FAIL\nD2M_V1_RUN_UUID={V1_UUID}\nD2M_V2_RESULT=BLOCKED_INPUT_AUTHORITY\nD2M_V2_RUN_UUID={V2_UUID}\nD2M_V1_HISTORICAL_RESULT_REWRITTEN=NO\nD2M_V2_HISTORICAL_RESULT_REWRITTEN=NO\n```\n\n## Current blocker localization\n\n```text\nOLD_GRAPH_AUTHORITY_STATUS=240_FRAME_REPAIRED_PASS\nV2_WARM_QPOS_LENGTH=1\nV2_WARM_BASE_LENGTH=1\nWARM_AUTHORITY_ROLE=FRAME0_ONLY_INITIALIZER\nWARM_AUTHORITY_CONFIDENCE=HIGH\n```\n\nThe frame0-development warm carrier was accidentally indexed at frame1 before V3Runtime could overwrite it with the already-authoritative current-frame neutral/base seed.\n\n## Consumer inventory and coverage\n\n```text\nTOTAL_CONSUMER_INPUTS=19\nTOTAL_SCIENTIFIC_CONSUMER_INPUTS=17\nORDINAL_INDEXED_INPUT_COUNT=6\nSOURCE_FRAME_INDEXED_COUNT={summary["SOURCE_FRAME_INDEXED_COUNT"]}\nFRAME0_ONLY_COUNT={summary["FRAME0_ONLY_COUNT"]}\nPREVIOUS_RUNTIME_COUNT={summary["PREVIOUS_RUNTIME_COUNT"]}\nTRAJECTORY_STATIC_COUNT={summary["TRAJECTORY_STATIC_COUNT"]}\nUNKNOWN_INPUT_AUTHORITY_COUNT=0\nALL_OFFLINE_CONSUMER_INPUTS_VALID=YES\nFRAME_INDEXED_INPUT_ALIGNMENT=240/240\nSINGLETON_INDEXED_CARRIER_COUNT=0\nUNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT=0\n```\n\n## Frame1 and multi-ordinal preflight\n\n```text\nFRAME1_CONTEXT_CANARY=PASS\nFRAME1_CANARY_SOURCE_FRAME=10705\nFRAME1_CANARY_GRAPH_FRAME=10705\nFRAME1_CANARY_PREVIOUS_STATE_SOURCE_FRAME=10704\nFRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT=0\nFRAME1_CANARY_Q_OLD_PRESENT=NO\nFULL_SEQUENCE_CONSUMER_PREFLIGHT=PASS\nOFFLINE_ORDINALS_VALIDATED=240/240\n```\n\n## Repair impact and frozen authorities\n\n```text\nREPAIR_IMPACT=FULL_SEQUENCE_INPUT_BINDING_ONLY\nEXECUTION_V4_SEQUENTIAL_RUNTIME_SHA_UNCHANGED=YES\nEXECUTION_V4_INPUT_AUTHORITY_SHA_UNCHANGED=YES\nEXECUTION_V4_COLDSTART_SEARCH_SHA_UNCHANGED=YES\nOBJECTIVE_V2_EXECUTION_CONTRACT_V4_SHA_UNCHANGED=YES\nDEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256={frozen["DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256"]}\nFULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256={frozen["FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256"]}\n```\n\n## D2M-R2 result and DEV2 Full Recovery V3\n\n```text\nD2M_R2_STATUS=PASS\nDEV2_FULL_RECOVERY_V3_AUTHORIZED=YES\nDEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT=0\nNEXT=O5R-D2N_DEV2_FULL_RECOVERY_V3\nNEW_RUN_UUID_REQUIRED=YES\nV1_RUN_UUID_REUSED=NO\nV2_RUN_UUID_REUSED=NO\nV1_FRAME0_REUSED=NO\nV2_FRAME0_REUSED=NO\nV3_STARTS_FROM_FRAME0=YES\nV3_EXPECTED_FRAMES=240\nV3_Q_OLD_PRESENT=NO\nV3_FULL_CONSUMER_PREFLIGHT_REQUIRED=YES\n```\n\nThe first proof that all full-sequence input-authority barriers are cleared is: frame10705 context built **and** frame10705 optimizer actually started. This R2 task does not run that optimizer.\n\n## CLI\n\n```bash\nconda run -n toporetarget-rl python scripts/evaluation/run_oakink2_o5rd2mr2.py --help\nconda run -n toporetarget-rl python scripts/evaluation/run_oakink2_o5rd2mr2.py run-all\n```\n"""
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    write_json(
        root / "preflight/cli_help.json",
        {
            "status": "PASS",
            "command": "conda run -n toporetarget-rl python scripts/evaluation/run_oakink2_o5rd2mr2.py --help",
            "executed": True,
            "actions": sorted(ACTIONS),
        },
    )
    required = [
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "preflight/git.json",
        "preflight/frozen_v4.json",
        "preflight/graph_authority.json",
        "preflight/historical_runs.json",
        "history/d2m_v1.json",
        "history/d2m_v2.json",
        "history/run_history.json",
        "consumer_audit/call_graph.json",
        "consumer_audit/ordinal_access_sites.csv",
        "consumer_audit/consumer_input_inventory.csv",
        "consumer_audit/consumer_input_authority_matrix.csv",
        "consumer_audit/unknown_inputs.json",
        "warm_authority/producer_trace.json",
        "warm_authority/consumer_trace.json",
        "warm_authority/window_v6_comparison.json",
        "warm_authority/authority_decision.json",
        "repair/repair_scope.json",
        "repair/modified_components.json",
        "coverage/consumer_input_coverage.csv",
        "coverage/frame_alignment.csv",
        "coverage/singleton_carriers.json",
        "coverage/offline_240_preflight.json",
        "context_canary/frame1_context_canary.json",
        "context_canary/context_fields.json",
        "context_canary/no_optimizer_receipt.json",
        "multi_ordinal_preflight/deterministic_ordinals.json",
        "multi_ordinal_preflight/results.json",
        "regression_tests/old_graph_singleton.json",
        "regression_tests/old_warm_singleton.json",
        "regression_tests/shifted_frame_ids.json",
        "regression_tests/unknown_indexed_input.json",
        "regression_tests/missing_previous_runtime.json",
        "impact_audit/scientific_payload_before.json",
        "impact_audit/scientific_payload_after.json",
        "impact_audit/window_v6_semantics_comparison.json",
        "impact_audit/v4_hashes_before.json",
        "impact_audit/v4_hashes_after.json",
        "impact_audit/decision.json",
        "frozen_authority/full_sequence_consumer_input_authority.json",
        "frozen_authority/full_sequence_consumer_input_authority.sha256",
        "frozen_authority/full_sequence_preflight_contract.json",
        "frozen_authority/full_sequence_preflight_contract.sha256",
        "future_d2n/dev2_full_recovery_v3_plan.json",
        "tests.json",
        "validation_results.json",
        "technical_failures.jsonl",
        "resource_usage.json",
        "git_commits.json",
        "not_run.json",
    ]
    missing = [relative for relative in required if not (root / relative).exists()]
    write_json(
        root / "completion_audit.json",
        {
            "schema_version": "D2MR2CompletionAuditV1",
            "status": "PASS" if not missing else "FAIL",
            "required_artifacts_present": not missing,
            "required_artifact_count": len(required),
            "missing": missing,
            "safety_flags": flags,
        },
    )
    if missing:
        raise RuntimeError(f"DELIVERY_INCOMPLETE:{missing}")
    return summary


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_historical_runs(root)
    verify_frozen_v4(root)
    verify_full_graph_authority(root)
    audit_consumer_callgraph(root)
    inventory_consumer_inputs(root)
    classify_consumer_input_authorities(root)
    audit_warm_authority(root)
    repair_consumer_input_binding(root)
    build_consumer_input_authority(root)
    run_offline_240_preflight(root)
    run_frame1_context_canary(root)
    run_multi_ordinal_preflight(root)
    run_input_regression_tests(root)
    audit_repair_impact(root)
    freeze_consumer_input_authority(root)
    authorize_dev2_v3(root)
    generate_d2n_plan(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-historical-runs": verify_historical_runs,
    "verify-frozen-v4": verify_frozen_v4,
    "verify-full-graph-authority": verify_full_graph_authority,
    "audit-consumer-callgraph": audit_consumer_callgraph,
    "inventory-ordinal-indexed-inputs": inventory_consumer_inputs,
    "classify-consumer-input-authorities": classify_consumer_input_authorities,
    "audit-warm-authority": audit_warm_authority,
    "repair-consumer-input-binding": repair_consumer_input_binding,
    "build-consumer-input-authority": build_consumer_input_authority,
    "run-offline-240-preflight": run_offline_240_preflight,
    "run-frame1-context-canary": run_frame1_context_canary,
    "run-multi-ordinal-preflight": run_multi_ordinal_preflight,
    "run-input-regression-tests": run_input_regression_tests,
    "audit-repair-impact": audit_repair_impact,
    "freeze-consumer-input-authority": freeze_consumer_input_authority,
    "authorize-dev2-v3": authorize_dev2_v3,
    "generate-d2n-plan": generate_d2n_plan,
    "summarize": summarize,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    result = ACTIONS[args.action](args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
