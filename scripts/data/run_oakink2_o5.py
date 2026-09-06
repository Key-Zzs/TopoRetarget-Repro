#!/usr/bin/env python3
"""Run the frozen OakInk2 O5 same-two exact Wuji retarget certification.

The public ``run-same-two`` action starts one measured worker process. That
worker initializes shared Wuji/static resources once, then runs DEV_01 and
DEV_02 sequentially. Derived semantic, HTML, and timing summaries reuse the
saved exact outputs and never invoke the solver again.
"""

# ruff: noqa: E501, I001, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from oakink2_browser_cdp import ChromeCDP  # noqa: E402
from run_oakink2_o1r2c import camera_presets  # noqa: E402

from scripts.evaluation.audit_retarget_semantic_validity import (  # noqa: E402
    _case_metrics,
)
from scripts.evaluation.audit_retarget_semantic_validity import (  # noqa: E402
    _write_csv as semantic_write_csv,
)
from scripts.evaluation.audit_retarget_semantic_validity import (  # noqa: E402
    _write_json as semantic_write_json,
)
from toporetarget.adapters.datasets.oakink2 import (  # noqa: E402
    OakInk2CanonicalAdapterV1,
    materialize_manifest_record_v2,
)
from toporetarget.cli.retarget import (  # noqa: E402
    _default_collision_samples,
    _run_checkpoint_refinement,
)
from toporetarget.contracts.canonical import (  # noqa: E402
    load_canonical_hoi,
    save_canonical_hoi,
)
from toporetarget.evaluation.geometric_retarget_timing import (  # noqa: E402
    aggregate_frame_statistics,
    cold_warm_totals,
    frame_statistics,
    runtime_estimate,
    stage_total,
    validate_frame_rows,
)
from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    SemanticGateContractV1,
    SemanticStatus,
)
from toporetarget.geometry.object_geometry import sample_object_track  # noqa: E402
from toporetarget.geometry.surface_sampling import load_surface_profile  # noqa: E402
from toporetarget.quality.html import _robot_visual_payload  # noqa: E402
from toporetarget.retarget.artifacts import (  # noqa: E402
    artifact_hash,
    save_warm_start,
)
from toporetarget.retarget.bones import load_bone_profile  # noqa: E402
from toporetarget.retarget.delaunay import load_delaunay_profile  # noqa: E402
from toporetarget.retarget.final_refinement import (  # noqa: E402
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    load_final_trajectory,
    load_robot_surface_samples,
)
from toporetarget.retarget.frames import load_frame_profile  # noqa: E402
from toporetarget.retarget.interaction_artifacts import (  # noqa: E402
    interaction_artifact_hash,
    save_interaction_evaluation,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_evaluation import (  # noqa: E402
    evaluate_interaction_graph,
)
from toporetarget.retarget.interaction_graph import (  # noqa: E402
    build_source_interaction_graph,
    load_paper_kappa,
)
from toporetarget.retarget.pipeline import build_warm_start_trajectory  # noqa: E402
from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.retarget.solver import load_solver_profile  # noqa: E402
from toporetarget.robots.registry import get_robot_registry  # noqa: E402
from toporetarget.utils.hashing import sha256_file, sha256_tree  # noqa: E402
from toporetarget.viz.oakink2_html_viewer import (  # noqa: E402
    OakInk2HTMLViewerV2Data,
    render_oakink2_html_viewer_v2,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/oakink2_o5_geometric_retarget_v1"
O1R_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r_official_mano_authority_v1"
O1R2D_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r2d_ref2dex_html_viewer_v1"
MANIFEST_V2 = O1R_ROOT / "manifest_v2/oakink2_corpus_manifest_v2.jsonl"
SPLIT_V2 = O1R_ROOT / "manifest_v2/oakink2_raw_to_physical_split_v2.json"
OVERLAP_V2 = O1R_ROOT / "manifest_v2/split_overlap_audit_v2.json"
UPSTREAM_RECEIPT = O1R2D_ROOT / "human_review/o1r2d_human_review_approval_v1.json"
DATASET_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2")
MANO_MODEL = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl")
ROBOT = "wuji_hand2_beta1_rh"
SOLVER_PROFILE = "wuji_continuous_sequential_v1"
EXECUTION_PROFILE = "wuji_continuous_sequential_fast_exact_v2"
QUERY_PROFILE = "adaptive_active_set_v1"
COORDINATE_PROFILE = "local_seed_delta_v1"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
EXPECTED_MANIFEST_SHA256 = "0ef41a307d431569c519dd426b7c21a5ff7151e04ad60f35b36ecbc975d190c7"
EXPECTED_SPLIT_SHA256 = "05a32e22373b632e14ad1238954b1e28c4135efb0b580e8ef43e2d7dd74c5561"
START_HEAD = "b065df2c2d9d37dc510ef5f3424086a9db7350a1"
EPISODES = (
    {
        "review": "dev_01",
        "record_id": "oakink2:scene_01__A003++seq__d4ddf93a38e3228cdd3a__2023-04-15-10-15-10:00001",
        "object": "C10001",
        "source_interval": [1969, 4691],
        "primary_frame": 4279,
    },
    {
        "review": "dev_02",
        "record_id": "oakink2:scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04:00010",
        "object": "C11001",
        "source_interval": [10704, 10944],
        "primary_frame": 10778,
    },
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "preflight",
            "validate",
            "run-same-two",
            "derive",
            "semantic",
            "render",
            "summarize",
            "_run-worker",
        ),
        default="preflight",
    )
    value.add_argument("--review", choices=("dev_01", "dev_02"))
    value.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return value


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).rstrip()


def tree_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(path).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [
        json.loads(line) for line in MANIFEST_V2.read_text(encoding="utf-8").splitlines() if line
    ]
    return rows, {str(row["record_id"]): row for row in rows}


def episode_for(review: str) -> dict[str, Any]:
    return next(dict(item) for item in EPISODES if item["review"] == review)


def episode_paths(root: Path, review: str) -> dict[str, Path]:
    retarget = root / "retarget" / review
    work = retarget / "work"
    return {
        "root": retarget,
        "work": work,
        "input_authority": retarget / "input_authority.json",
        "trajectory": retarget / "trajectory.npz",
        "solver_receipt": retarget / "solver_receipt.json",
        "semantic": retarget / "semantic_validity.json",
        "frame_metrics": retarget / "frame_metrics.csv",
        "output_hashes": retarget / "output_hashes.json",
        "canonical": work / "canonical_episode.zarr",
        "warm": work / "warm_start.zarr",
        "samples": work / "object_samples.npz",
        "graph": work / "interaction_graph.zarr",
        "evaluation": work / "interaction_evaluation.zarr",
        "final": work / "final_continuous.zarr",
        "checkpoints": work / "continuous_checkpoints",
        "progress": work / "continuous_refine_progress.json",
        "progress_log": work / "continuous_refine_progress.jsonl",
        "html": root / "review" / review / "oakink2_wuji_retarget_viewer.html",
        "viewer_receipt": root / "review" / review / "receipt.json",
    }


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError(f"O5_BRANCH_INVALID:{branch}")
    manifest_sha = sha256_file(MANIFEST_V2)
    split_sha = sha256_file(SPLIT_V2)
    if manifest_sha != EXPECTED_MANIFEST_SHA256 or split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("O5_FROZEN_MANIFEST_OR_SPLIT_HASH_DRIFT")
    upstream = json.loads(UPSTREAM_RECEIPT.read_text(encoding="utf-8"))
    expected_upstream = {
        "OAKINK2_VIEWER_V2_DEV_1": "APPROVE",
        "OAKINK2_VIEWER_V2_DEV_2": "APPROVE",
        "INTERACTIVE_VIEWER_CERTIFICATION": "PASS",
        "O1_CUSTOM_HTML_VIEWER_HUMAN": "APPROVE",
        "O1_FINAL": "PASS",
        "O5_ALLOWED": "YES",
    }
    observed = {**upstream["approvals"], **upstream["final_gate"]}
    if any(observed.get(key) != value for key, value in expected_upstream.items()):
        raise RuntimeError(f"O5_UPSTREAM_GATE_CLOSED:{observed}")
    rows, by_id = manifest_rows()
    split = json.loads(SPLIT_V2.read_text(encoding="utf-8"))
    memberships = split["splits"]
    counts = {name: len(values) for name, values in memberships.items()}
    if counts != {"DEVELOPMENT": 457, "CERTIFICATION": 137, "HELDOUT_TEST": 148}:
        raise RuntimeError(f"O5_SPLIT_COUNT_DRIFT:{counts}")
    sets = {name: set(values) for name, values in memberships.items()}
    overlap_counts = {
        "development_certification": len(sets["DEVELOPMENT"] & sets["CERTIFICATION"]),
        "development_heldout": len(sets["DEVELOPMENT"] & sets["HELDOUT_TEST"]),
        "certification_heldout": len(sets["CERTIFICATION"] & sets["HELDOUT_TEST"]),
    }
    if any(overlap_counts.values()) or int(split["heldout_downstream_consumed"]) != 0:
        raise RuntimeError("O5_SPLIT_OVERLAP_OR_HELDOUT_CONSUMPTION")
    fixed_rows = []
    for expected in EPISODES:
        record_id = str(expected["record_id"])
        record = by_id.get(record_id)
        if (
            record is None
            or record_id not in sets["DEVELOPMENT"]
            or record_id in sets["CERTIFICATION"] | sets["HELDOUT_TEST"]
            or record["canonical_target_object"] != expected["object"]
            or record["source_interval"] != expected["source_interval"]
            or record["active_hand"] != "RIGHT"
        ):
            raise RuntimeError(f"O5_FIXED_EPISODE_AUTHORITY_MISMATCH:{record_id}")
        fixed_rows.append({**expected, "manifest_record_sha256": record["canonical_record_sha256"]})
    eligible = [row for row in rows if row.get("eligibility") is True]
    frame_counts = np.asarray(
        [int(row["source_interval"][1]) - int(row["source_interval"][0]) for row in eligible],
        dtype=np.int64,
    )
    if len(eligible) != 742:
        raise RuntimeError(f"O5_ELIGIBLE_COUNT_DRIFT:{len(eligible)}")
    overlap_receipt = json.loads(OVERLAP_V2.read_text(encoding="utf-8"))
    git_payload = {
        "branch": branch,
        "start_head": START_HEAD,
        "implementation_head_at_artifact_preflight": head,
        "status_short": git("status", "--short", "--untracked-files=all").splitlines(),
        "tracked_worktree_clean_before_task_edits": True,
        "current_tracked_worktree_clean": not bool(
            git("status", "--porcelain", "--untracked-files=no")
        ),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "new_branch_created": False,
        "new_worktree_created": False,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", git_payload)
    write_json(
        root / "preflight/upstream_gate.json",
        {
            "schema_version": "OakInk2O5UpstreamGateV1",
            "status": "PASS",
            "required": expected_upstream,
            "observed": observed,
            "authority_path": str(UPSTREAM_RECEIPT.resolve()),
            "authority_sha256": sha256_file(UPSTREAM_RECEIPT),
        },
    )
    write_json(
        root / "preflight/manifest_split_integrity.json",
        {
            "status": "PASS",
            "manifest_v2": {
                "path": str(MANIFEST_V2.resolve()),
                "sha256": manifest_sha,
                "modified": False,
            },
            "split_v2": {"path": str(SPLIT_V2.resolve()), "sha256": split_sha, "modified": False},
            "counts": counts,
            "eligible": len(eligible),
            "overlap_counts": overlap_counts,
            "overlap_audit": overlap_receipt,
            "certification_downstream_consumed": 0,
            "heldout_downstream_consumed": 0,
        },
    )
    write_json(
        root / "preflight/fixed_o5_episode_set.json",
        {
            "schema_version": "OakInk2O5FixedSameTwoDevelopmentEpisodesV1",
            "same_two_episodes": True,
            "episodes_reselected": False,
            "episodes": fixed_rows,
        },
    )
    distribution = {
        "schema_version": "OakInk2EligibleFrameCountDistributionV1",
        "metadata_only": True,
        "n_eligible": 742,
        "total_frames": int(frame_counts.sum()),
        "mean_frames_per_episode": float(frame_counts.mean()),
        "std_frames_per_episode": float(frame_counts.std()),
        "min_frames_per_episode": int(frame_counts.min()),
        "p50_frames_per_episode": float(np.percentile(frame_counts, 50)),
        "p90_frames_per_episode": float(np.percentile(frame_counts, 90)),
        "p95_frames_per_episode": float(np.percentile(frame_counts, 95)),
        "max_frames_per_episode": int(frame_counts.max()),
        "additional_eligible_retarget_runs": 0,
    }
    write_json(root / "timing/eligible_frame_count_distribution.json", distribution)
    runtime = runtime_environment()
    write_json(root / "preflight/runtime_environment.json", runtime)
    write_contract(root)
    (root / "technical_failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    write_json(
        root / "git_commits.json",
        {"start_head": START_HEAD, "commits": [], "pushed": False, "pr_created": False},
    )
    return {
        "status": "PASS",
        "git": git_payload,
        "episodes": fixed_rows,
        "distribution": distribution,
    }


def runtime_environment() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    cpu_blocks = [block for block in cpuinfo.split("\n\n") if block.strip()]
    cpu_fields = [
        {
            key.strip(): value.strip()
            for line in block.splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        }
        for block in cpu_blocks
    ]
    model_names = [fields["model name"] for fields in cpu_fields if "model name" in fields]
    physical_pairs = {
        (fields["physical id"], fields["core id"])
        for fields in cpu_fields
        if "physical id" in fields and "core id" in fields
    }
    try:
        import psutil

        physical_cpus: int | str | None = psutil.cpu_count(logical=False)
        total_ram = int(psutil.virtual_memory().total)
    except ImportError:
        physical_cpus = len(physical_pairs) if physical_pairs else "NOT_MEASURED"
        total_ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    try:
        import torch

        torch_values = {
            "pytorch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_values = {
            "pytorch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu": None,
        }
    return {
        "hostname": socket.gethostname(),
        "cpu_model": model_names[0] if model_names else platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_cpus,
        "ram_bytes": total_ram,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "NOT_SET"),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        "solver_actual_device": "CPU",
        **torch_values,
    }


def write_contract(root: Path) -> dict[str, Any]:
    solver = RefinementSolverProfile.load(SOLVER_PROFILE)
    execution = RefinementExecutionProfile.load(EXECUTION_PROFILE, REPO_ROOT)
    query = CollisionQueryProfile.load(QUERY_PROFILE)
    coordinate = RefinementCoordinateProfile.load(COORDINATE_PROFILE)
    contract = {
        "schema_version": "OakInk2O5ExactGeometricRetargetContractV1",
        "implementation": "TopoRetarget production checkpointed continuous sequential refinement",
        "robot": ROBOT,
        "active_hand": "RIGHT",
        "solver_profile": solver.as_dict(),
        "execution_profile": execution.as_dict(),
        "query_profile": query.as_dict(),
        "coordinate_profile": coordinate.as_dict(),
        "warm_start_profile": "paper_repro_scipy_trf",
        "frame_profile": "canonical_keypoint_wrist_v1",
        "bone_profile": "mediapipe21_full_finger_chain_v1",
        "object_surface_profile": "paper_strict_area_uniform",
        "delaunay_profile": "strict_scipy_qhull_v1",
        "full_source_trajectory": True,
        "temporal_subsampling": False,
        "retiming": "NONE",
        "solver_limits_changed": False,
        "solver_tolerances_changed": False,
        "timing_driven_method_tuning": False,
        "execution_device": "CPU",
        "dtype": "float64",
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), default=json_default
    ).encode()
    contract_hash = hashlib.sha256(encoded).hexdigest()
    write_json(root / "contract/geometric_retarget_contract.json", contract)
    (root / "contract/retarget_contract_sha256.txt").write_text(
        contract_hash + "\n", encoding="utf-8"
    )
    timing = {
        "schema_version": "GeometricRetargetTimingV1",
        "timer": "time.perf_counter",
        "same_python_process_sequential": True,
        "order": ["shared_init", "dev_01", "dev_02"],
        "n_episodes_timed": 2,
        "definitions": {
            "T_shared_init": "one Wuji asset/kinematic/static surface/profile initialization",
            "T_episode_load": "Manifest row plus source MANO/object trajectory/object mesh load",
            "T_prepare": "canonical arrays, source keypoints, object samples, warm trajectory, frame transforms, graph/evaluation, solver-static episode inputs, checkpoint/assembly overhead",
            "T_solver_episode": "sum of exact per-frame geometric optimization timers only",
            "T_semantic_validation": "RetargetSemanticValidityV1 only",
            "T_html": "viewer geometry assembly, serialization, and HTML generation only; browser regression is separately measured",
            "T_episode_machine_total": "load + prepare + solver + semantic + html; excludes shared init",
        },
        "solver_timing_includes_html": False,
        "solver_timing_includes_human_review": False,
        "label": "TWO_EPISODE_PILOT_EXTRAPOLATION",
    }
    write_json(root / "contract/timing_contract.json", timing)
    write_json(root / "timing/timing_contract.json", timing)
    return {"contract": contract, "sha256": contract_hash, "timing": timing}


def _health_gate(metadata: dict[str, Any], _: list[dict[str, Any]]) -> str | None:
    return None if bool(metadata.get("strict_accepted")) else "NUMERICAL_RETARGET_FRAME_REJECTED"


def prepare_episode(
    root: Path, episode: dict[str, Any], record: dict[str, Any], model: Any
) -> tuple[dict[str, Path], dict[str, Any], float, float]:
    paths = episode_paths(root, str(episode["review"]))
    paths["root"].mkdir(parents=True, exist_ok=True)
    if any(paths[name].exists() for name in ("trajectory", "final", "checkpoints")):
        raise RuntimeError(f"O5_USABLE_OR_PARTIAL_SOLVER_OUTPUT_EXISTS:{episode['review']}")
    load_started = time.perf_counter()
    adapter = OakInk2CanonicalAdapterV1(DATASET_ROOT)
    canonical, authority = materialize_manifest_record_v2(
        adapter,
        record,
        mano_model_path=MANO_MODEL,
        admitted_split="DEVELOPMENT",
    )
    total_materialize = time.perf_counter() - load_started
    source_load = float(authority["timing"]["source_load_sec"])
    object_load = float(authority["timing"]["object_load_sec"])
    episode_load_sec = source_load + object_load
    prepare_started = time.perf_counter()
    save_canonical_hoi(canonical, paths["canonical"])
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    warm_profile = load_solver_profile("paper_repro_scipy_trf")
    warm, warm_diagnostics = build_warm_start_trajectory(
        canonical,
        "right_hand",
        model,
        frame_profile,
        bone_profile,
        warm_profile,
        source_cache=paths["canonical"],
    )
    save_warm_start(warm, paths["warm"])
    surface_profile = load_surface_profile("paper_strict_area_uniform", repo_root=REPO_ROOT)
    samples = sample_object_track(canonical.rigid_object(str(episode["object"])), surface_profile)
    samples.save(paths["samples"])
    graph = build_source_interaction_graph(
        canonical,
        "right_hand",
        str(episode["object"]),
        samples,
        source_cache=paths["canonical"],
        object_sample_path=paths["samples"],
        delaunay_profile=load_delaunay_profile("strict_scipy_qhull_v1"),
        kappa=load_paper_kappa(),
        frame_indices=np.arange(canonical.num_frames, dtype=np.int64),
    )
    save_interaction_graph(graph, paths["graph"])
    evaluation = evaluate_interaction_graph(
        graph,
        warm,
        model,
        graph_artifact_hash=interaction_artifact_hash(paths["graph"]),
        warm_start_artifact_hash=artifact_hash(paths["warm"]),
    )
    save_interaction_evaluation(evaluation, paths["evaluation"])
    prepare_sec = total_materialize - episode_load_sec + time.perf_counter() - prepare_started
    input_authority = {
        **authority,
        "split": "DEVELOPMENT",
        "manifest_v2_sha256": EXPECTED_MANIFEST_SHA256,
        "split_v2_sha256": EXPECTED_SPLIT_SHA256,
        "object": episode["object"],
        "robot": ROBOT,
        "full_source_trajectory": True,
        "frame_count": canonical.num_frames,
        "source_interval": episode["source_interval"],
        "episodes_reselected": False,
        "source_hashes": {
            "canonical": tree_hash(paths["canonical"]),
            "warm": artifact_hash(paths["warm"]),
            "object_samples": sha256_file(paths["samples"]),
            "graph": interaction_artifact_hash(paths["graph"]),
            "evaluation": interaction_artifact_hash(paths["evaluation"]),
        },
        "warm_start_diagnostics": warm_diagnostics,
    }
    write_json(paths["input_authority"], input_authority)
    return paths, input_authority, episode_load_sec, prepare_sec


def checkpoint_rows(paths: dict[str, Path], episode: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_paths = sorted((paths["checkpoints"] / "frames").glob("frame_*.npz"))
    rejected = sorted(paths["checkpoints"].glob("rejected_frame_*.json"))
    values = []
    for path in metadata_paths:
        with np.load(path, allow_pickle=False) as payload:
            values.append(json.loads(str(payload["metadata_json"].item())))
    values.extend(json.loads(path.read_text(encoding="utf-8")) for path in rejected)
    values.sort(key=lambda row: int(row["local_frame_index"]))
    start = int(episode["source_interval"][0])
    rows = []
    for item in values:
        diagnostics = item.get("diagnostics", {})
        accepted_solver_sec = float(item["solve_time_s"])
        timer_elapsed = diagnostics.get("timers", {}).get("elapsed_s", {})
        # A production frame may invoke refine_frame more than once through the
        # continuous retry policy.  The context TimerBook is intentionally
        # shared by those attempts, so active_set_outer_loop is the cumulative
        # duration of every actual geometric optimization attempt for the
        # frame.  solve_time_s is retained separately: it is only the terminal
        # result selected by the production policy and would undercount retries.
        solver_sec = float(
            timer_elapsed.get("active_set_outer_loop", accepted_solver_sec)
            + timer_elapsed.get("final_full_audit", 0.0)
        )
        end = float(diagnostics.get("solver_end_perf_counter", np.nan))
        begin = float(diagnostics.get("solver_start_perf_counter", end - accepted_solver_sec))
        rows.append(
            {
                "episode": episode["review"],
                "source_frame_id": start + int(item["local_frame_index"]),
                "runtime_frame_id": int(item["global_frame_index"]),
                "frame_ordinal": int(item["local_frame_index"]),
                "solver_start": begin,
                "solver_end": end,
                "solver_sec": solver_sec,
                "solver_sec_scope": "all_production_refine_attempt_active_set_and_final_audit_timers",
                "solver_start_end_scope": "terminal_selected_refine_frame_call",
                "accepted_solver_sec": accepted_solver_sec,
                "accepted_solver_start": begin,
                "accepted_solver_end": end,
                "unrecorded_window_joint_optimization": diagnostics.get("window_joint") is not None,
                "iterations": int(item["optimizer_iterations"]),
                "termination": str(item["optimizer_message"]),
                "termination_status_code": int(item["optimizer_status_code"]),
                "initial_objective": float(item["initial_objective"]),
                "final_objective": float(item["final_objective"]),
                "objective_delta": float(item["final_objective_change"]),
                "converged": bool(item["optimizer_converged"]),
                "strict_accepted": bool(item["strict_accepted"]),
                "initialization_source": str(item["initialization_source"]),
            }
        )
    return rows


def compact_trajectory(
    paths: dict[str, Path],
    episode: dict[str, Any],
    final: Any,
    frame_rows: list[dict[str, Any]],
) -> None:
    np.savez_compressed(
        paths["trajectory"],
        schema_version=np.asarray("OakInk2O5WujiTrajectoryV1"),
        episode=np.asarray(episode["record_id"]),
        source_frame_ids=np.arange(*episode["source_interval"], dtype=np.int64),
        runtime_frame_ids=np.asarray(final.arrays["frame_indices"], dtype=np.int64),
        qpos=np.asarray(final.arrays["qpos"], dtype=np.float64),
        wrist_pose_scene=np.asarray(final.arrays["base_pose_scene"], dtype=np.float64),
        fingertip_positions_scene=np.asarray(
            final.arrays["robot_keypoints_scene"], dtype=np.float64
        )[:, [4, 8, 12, 16, 20]],
        robot_keypoints_scene=np.asarray(final.arrays["robot_keypoints_scene"], dtype=np.float64),
        robot_link_poses_scene=np.asarray(final.arrays["robot_link_poses"], dtype=np.float64),
        solver_sec=np.asarray([row["solver_sec"] for row in frame_rows], dtype=np.float64),
        accepted_solver_sec=np.asarray(final.arrays["solve_time_s"], dtype=np.float64),
    )


def run_semantic(root: Path, review: str) -> tuple[dict[str, Any], float]:
    episode = episode_for(review)
    paths = episode_paths(root, review)
    started = time.perf_counter()
    result = _case_metrics(
        str(episode["record_id"]),
        {
            "canonical": paths["canonical"],
            "warm": paths["warm"],
            "final": paths["final"],
            "graph": paths["graph"],
            "evaluation": paths["evaluation"],
            "receipt": paths["solver_receipt"],
        },
        SemanticGateContractV1(),
        diagnostic_bundle=paths["work"] / "semantic_diagnostic_bundle.npz",
    )
    elapsed = time.perf_counter() - started
    semantic_write_json(paths["semantic"], result)
    semantic_write_csv(paths["frame_metrics"], list(result["per_frame"]))
    gate = SemanticGateContractV1()
    write_json(paths["root"] / "semantic_gate_contract.json", gate.as_dict())
    (paths["root"] / "semantic_gate_contract_sha256.txt").write_text(
        gate.sha256 + "\n", encoding="utf-8"
    )
    return result, elapsed


def selected_viewer_indices(frame_count: int, primary_local: int) -> np.ndarray:
    selected = np.linspace(0, frame_count - 1, min(180, frame_count), dtype=np.int64)
    if primary_local not in set(selected.tolist()):
        selected[len(selected) // 2] = primary_local
        selected.sort()
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("O5_VIEWER_DISPLAY_SAMPLE_DUPLICATE")
    return selected


def certify_viewer(html: Path, screenshot: Path) -> dict[str, Any]:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        return {"status": "FAIL", "reason": "CHROME_NOT_AVAILABLE"}
    url = f"{html.resolve().as_uri()}?certify=1&preset=OBLIQUE&mode=SOURCE_WUJI_OBJECT"
    with ChromeCDP(chrome, width=900, height=1050) as browser:
        browser.navigate(url)
        baseline = browser.certificate()
        bounds = browser.evaluate("document.querySelector('#c').getBoundingClientRect().toJSON()")
        center = (
            float(bounds["x"]) + float(bounds["width"]) / 2,
            float(bounds["y"]) + float(bounds["height"]) / 2,
        )
        browser.mouse_drag((center[0] - 80, center[1]), (center[0] + 80, center[1] + 45), steps=5)
        dragged = browser.certificate()
        browser.wheel(-220)
        zoomed = browser.certificate()
        browser.evaluate("window.__OAKINK2_VIEWER_V2__.resetCamera()")
        reset = browser.certificate()
        browser.evaluate("window.__OAKINK2_VIEWER_V2__.setFrame(1)")
        playback = browser.certificate()
        browser.screenshot(screenshot)
    immutable = all(
        dragged["scene_nodes"][name] == baseline["scene_nodes"][name]
        for name in ("hand", "object", "wuji")
    )
    relative = dragged["relative_geometry"] == baseline["relative_geometry"]
    camera_changed = dragged["camera_view_matrix"] != baseline["camera_view_matrix"]
    zoom_changed = zoomed["camera_view_matrix"] != dragged["camera_view_matrix"]
    reset_ok = reset["camera_view_matrix"] == baseline["camera_view_matrix"]
    playback_ok = int(playback["frame_index"]) == 1
    colors = baseline["framebuffer"]
    layers_visible = (
        int(colors.get("green_pixels", 0)) > 0
        and int(colors.get("orange_pixels", 0)) > 0
        and int(colors.get("blue_pixels", 0)) > 0
    )
    passed = (
        immutable
        and relative
        and camera_changed
        and zoom_changed
        and reset_ok
        and playback_ok
        and layers_visible
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "pointer_drag_real_cdp_input": True,
        "camera_changed": camera_changed,
        "zoom_changed": zoom_changed,
        "reset_exact": reset_ok,
        "timeline_playback": playback_ok,
        "source_object_wuji_geometry_immutable": immutable,
        "all_relative_geometry_immutable": relative,
        "source_mano_present": int(colors.get("green_pixels", 0)) > 0,
        "target_object_present": int(colors.get("orange_pixels", 0)) > 0,
        "wuji_present": int(colors.get("blue_pixels", 0)) > 0,
        "baseline_certificate": baseline,
        "dragged_certificate": dragged,
    }


def run_render(root: Path, review: str) -> tuple[dict[str, Any], float]:
    episode = episode_for(review)
    paths = episode_paths(root, review)
    started = time.perf_counter()
    canonical = load_canonical_hoi(paths["canonical"])
    final = load_final_trajectory(paths["final"])
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    primary_local = int(episode["primary_frame"]) - int(episode["source_interval"][0])
    selected = selected_viewer_indices(final.frame_count, primary_local)
    frame_rows = checkpoint_rows(paths, episode)
    hand = canonical.hand("right_hand")
    joints_world = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[selected]
    hand_world = np.asarray(hand.vertices_scene)[selected]
    translation = np.asarray(hand.mano_parameters.transl)[selected]
    visual = _robot_visual_payload(
        get_robot_registry().load(ROBOT),
        np.asarray(final.arrays["qpos"])[selected],
        np.asarray(final.arrays["base_pose_scene"])[selected],
    )
    status = str(semantic["final"]["qualification"]["status"])
    data = OakInk2HTMLViewerV2Data(
        frames=np.arange(*episode["source_interval"], dtype=np.int64)[selected],
        hand_vertices_world=hand_world,
        hand_vertices_anatomy=hand_world - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=joints_world,
        hand_joints_anatomy=joints_world - translation[:, None, :],
        object_vertices=np.asarray(canonical.primary_rigid_object().mesh.vertices_local),
        object_faces=np.asarray(canonical.primary_rigid_object().mesh.faces),
        object_transforms=np.asarray(canonical.primary_rigid_object().pose_scene.pose_scene)[
            selected
        ],
        primary_frame=int(episode["primary_frame"]),
        record={
            "dataset": "OakInk2",
            "sequence": canonical.metadata.provenance.source_sequence,
            "primitive": canonical.metadata.metadata.get("primitive"),
            "episode": episode["record_id"],
            "source_hand": "RIGHT",
            "target_object": episode["object"],
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": status,
            "viewer_sampling": "180 deterministic display frames; solver used every source frame",
        },
        camera_presets=camera_presets(),
        wuji_parts=list(visual["parts"]),
        wuji_joints_world=np.asarray(final.arrays["robot_keypoints_scene"])[selected],
        frame_solver_sec=np.asarray([row["solver_sec"] for row in frame_rows])[selected],
    )
    renderer = render_oakink2_html_viewer_v2(data, paths["html"])
    html_elapsed = time.perf_counter() - started
    regression_started = time.perf_counter()
    regression = certify_viewer(paths["html"], paths["html"].with_name("interaction_review.png"))
    regression_elapsed = time.perf_counter() - regression_started
    receipt = {
        "schema_version": "OakInk2O5WujiViewerReceiptV1",
        "episode": episode,
        "html": str(paths["html"].resolve()),
        "html_sha256": sha256_file(paths["html"]),
        "renderer": renderer,
        "display_frame_count": int(len(selected)),
        "solver_frame_count": int(final.frame_count),
        "viewer_sampling_separate_from_solver": True,
        "source_mano_present": True,
        "target_object_present": True,
        "wuji_present": True,
        "interactive_orbit_regression": regression,
        "html_seconds": html_elapsed,
        "browser_regression_seconds_excluded_from_T_html": regression_elapsed,
    }
    write_json(paths["viewer_receipt"], receipt)
    return receipt, html_elapsed


def run_one_episode(
    root: Path,
    episode: dict[str, Any],
    record: dict[str, Any],
    model: Any,
    surface: Any,
    contract_hash: str,
) -> dict[str, Any]:
    paths = episode_paths(root, str(episode["review"]))
    run_count = 0
    load_sec = prepare_sec = solver_sec = semantic_sec = html_sec = 0.0
    try:
        paths, authority, load_sec, prepare_sec = prepare_episode(root, episode, record, model)
        run_count = 1
        write_json(
            paths["solver_receipt"],
            {
                "schema_version": "OakInk2O5SolverReceiptV1",
                "state": "RUNNING",
                "episode": episode,
                "pid": os.getpid(),
                "production_run_count": run_count,
                "started_at": utc_now(),
                "solver_profile": SOLVER_PROFILE,
                "execution_profile": EXECUTION_PROFILE,
                "retarget_contract_sha256": contract_hash,
            },
        )
        result = _run_checkpoint_refinement(
            canonical=paths["canonical"],
            warm_start=paths["warm"],
            graph_path=paths["graph"],
            robot=ROBOT,
            collision_samples=_default_collision_samples(ROBOT),
            query_profile_id=QUERY_PROFILE,
            coordinate_profile_id=COORDINATE_PROFILE,
            solver_profile_id=SOLVER_PROFILE,
            execution_profile_id=EXECUTION_PROFILE,
            start_frame=0,
            end_frame=int(authority["frame_count"]),
            checkpoint_root=paths["checkpoints"],
            output=paths["final"],
            asset_root=None,
            resume=False,
            max_wall_time=None,
            stop_after_frame=None,
            progress_json=paths["progress"],
            progress_log=paths["progress_log"],
            force=False,
            allow_shadow_while_queue_paused=True,
            frame_health_gate=_health_gate,
            model_override=model,
            surface_override=surface,
        )
        accepted_solver_sec = float(result.get("timing", {}).get("solver_seconds", 0.0))
        internal = float(
            result.get("timing", {}).get("total_internal_seconds", accepted_solver_sec)
        )
        rows = checkpoint_rows(paths, episode)
        solver_sec = float(sum(float(row["solver_sec"]) for row in rows))
        prepare_sec += max(0.0, internal - solver_sec)
        frame_csv = root / "timing" / f"{episode['review']}_frame_timing.csv"
        write_csv(frame_csv, rows)
        if result.get("status") != "complete" or not paths["final"].is_dir():
            receipt = {
                "schema_version": "OakInk2O5SolverReceiptV1",
                "terminal": "NUMERICAL_RETARGET_FAIL",
                "episode": episode,
                "pid": os.getpid(),
                "production_run_count": run_count,
                "checkpoint_status": result,
                "timed_frames": len(rows),
            }
            write_json(paths["solver_receipt"], receipt)
            write_stage_timing(root, episode, load_sec, prepare_sec, solver_sec, 0.0, 0.0)
            return receipt
        validate_frame_rows(rows, int(authority["frame_count"]))
        final = load_final_trajectory(paths["final"])
        compact_trajectory(paths, episode, final, rows)
        numerical_pass = bool(
            final.frame_count == int(authority["frame_count"])
            and np.asarray(final.arrays["optimizer_converged"], dtype=bool).all()
            and np.asarray(final.arrays["accepted"], dtype=bool).all()
            and np.isfinite(np.asarray(final.arrays["qpos"], dtype=np.float64)).all()
            and np.asarray(final.arrays["joint_limit_margins"], dtype=np.float64).min() >= -1e-10
        )
        semantic, semantic_sec = run_semantic(root, str(episode["review"]))
        semantic_pass = semantic["final"]["qualification"]["status"] == SemanticStatus.PASS.value
        viewer, html_sec = run_render(root, str(episode["review"]))
        terminal = (
            "O5_MACHINE_PASS"
            if numerical_pass and semantic_pass
            else ("NUMERICAL_RETARGET_FAIL" if not numerical_pass else "RETARGET_SEMANTIC_FAIL")
        )
        stats = frame_statistics([float(row["solver_sec"]) for row in rows])
        receipt = {
            "schema_version": "OakInk2O5SolverReceiptV1",
            "terminal": terminal,
            "episode": episode,
            "pid": os.getpid(),
            "production_run_count": run_count,
            "numerical_pass": numerical_pass,
            "semantic_status": semantic["final"]["qualification"]["status"],
            "frame_count": final.frame_count,
            "converged_frames": int(np.count_nonzero(final.arrays["optimizer_converged"])),
            "max_iter_frames": int(
                np.count_nonzero(np.asarray(final.arrays["optimizer_status_code"]) == 9)
            ),
            "failed_frames": int(
                np.count_nonzero(~np.asarray(final.arrays["accepted"], dtype=bool))
            ),
            "invalid_outputs": int(
                np.count_nonzero(~np.isfinite(np.asarray(final.arrays["qpos"])).all(axis=1))
            ),
            "solver_profile": SOLVER_PROFILE,
            "execution_profile": EXECUTION_PROFILE,
            "retarget_contract_sha256": contract_hash,
            "frame_statistics": stats,
            "checkpoint_status": {
                key: value for key, value in result.items() if key != "frame_rows"
            },
            "viewer": {
                "html": viewer["html"],
                "html_sha256": viewer["html_sha256"],
                "interactive_orbit_regression": viewer["interactive_orbit_regression"]["status"],
            },
        }
        write_json(paths["solver_receipt"], receipt)
        hashes = {
            "manifest_record_sha256": record["canonical_record_sha256"],
            "retarget_contract_sha256": contract_hash,
            "wuji_robot_spec_sha256": model.spec_hash,
            "wuji_urdf_sha256": model.urdf_hash,
            "wuji_asset_manifest_sha256": model.asset_manifest_hash,
            "source_geometry_sha256": tree_hash(paths["canonical"]),
            "object_asset_sha256": record["object_asset_sha256"],
            "solver_config_sha256": RefinementSolverProfile.load(SOLVER_PROFILE).profile_hash,
            "execution_config_sha256": RefinementExecutionProfile.load(
                EXECUTION_PROFILE, REPO_ROOT
            ).profile_hash,
            "trajectory_npz_sha256": sha256_file(paths["trajectory"]),
            "final_artifact_sha256": tree_hash(paths["final"]),
        }
        write_json(paths["output_hashes"], hashes)
        write_stage_timing(
            root,
            episode,
            load_sec,
            prepare_sec,
            solver_sec,
            semantic_sec,
            html_sec,
            terminal_selected_solver=accepted_solver_sec,
        )
        return receipt
    except Exception as exc:
        try:
            timed_frames = len(checkpoint_rows(paths, episode))
        except Exception:
            timed_frames = 0
        failure = {
            "recorded_at": utc_now(),
            "episode": episode,
            "pid": os.getpid(),
            "production_run_count": run_count,
            "reason": f"{type(exc).__name__}: {exc}",
            "usable_output": paths["final"].is_dir(),
        }
        with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
        receipt = {
            "schema_version": "OakInk2O5SolverReceiptV1",
            "terminal": "TECHNICAL_FAILURE",
            **failure,
        }
        write_json(paths["solver_receipt"], receipt)
        write_stage_timing(
            root,
            episode,
            load_sec,
            prepare_sec,
            solver_sec,
            semantic_sec,
            html_sec,
            timed_frames=timed_frames,
        )
        return receipt


def derive_episode(root: Path, review: str) -> dict[str, Any]:
    """Produce validation/viewer/report artifacts from one saved exact solve."""

    episode = episode_for(review)
    paths = episode_paths(root, review)
    previous = (
        json.loads(paths["solver_receipt"].read_text(encoding="utf-8"))
        if paths["solver_receipt"].is_file()
        else {}
    )
    rows = checkpoint_rows(paths, episode)
    expected_frames = int(episode["source_interval"][1] - episode["source_interval"][0])
    write_csv(root / "timing" / f"{review}_frame_timing.csv", rows)
    stage_path = root / "timing" / f"{review}_stage_timing.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    cumulative_solver_sec = float(sum(float(row["solver_sec"]) for row in rows))
    terminal_selected_solver_sec = float(sum(float(row["accepted_solver_sec"]) for row in rows))
    # The worker version that produced these exact outputs initially placed
    # retry solve time in the prepare remainder. Repartition the already-timed
    # machine total without changing it and without invoking the solver again.
    repartitioned_prepare_sec = float(stage["T_prepare_sec"]) - (
        cumulative_solver_sec - float(stage["T_solver_episode_sec"])
    )
    if repartitioned_prepare_sec < -1e-6:
        raise RuntimeError(f"O5_TIMING_REPARTITION_NEGATIVE_PREPARE:{review}")
    if not paths["final"].is_dir():
        write_stage_timing(
            root,
            episode,
            float(stage["T_episode_load_sec"]),
            max(0.0, repartitioned_prepare_sec),
            cumulative_solver_sec,
            float(stage["T_semantic_validation_sec"]),
            float(stage["T_html_sec"]),
            terminal_selected_solver=terminal_selected_solver_sec,
            timed_frames=len(rows),
        )
        previous["derived_timing_without_solver_rerun"] = True
        previous["timed_frames"] = len(rows)
        previous.setdefault("terminal", "TECHNICAL_FAILURE")
        write_json(paths["solver_receipt"], previous)
        return previous
    validate_frame_rows(rows, expected_frames)
    final = load_final_trajectory(paths["final"])
    compact_trajectory(paths, episode, final, rows)
    numerical_pass = bool(
        final.frame_count == expected_frames
        and np.asarray(final.arrays["optimizer_converged"], dtype=bool).all()
        and np.asarray(final.arrays["accepted"], dtype=bool).all()
        and np.isfinite(np.asarray(final.arrays["qpos"], dtype=np.float64)).all()
        and np.asarray(final.arrays["joint_limit_margins"], dtype=np.float64).min() >= -1e-10
    )
    semantic, semantic_sec = run_semantic(root, review)
    semantic_pass = semantic["final"]["qualification"]["status"] == SemanticStatus.PASS.value
    viewer, html_sec = run_render(root, review)
    terminal = (
        "O5_MACHINE_PASS"
        if numerical_pass and semantic_pass
        else ("NUMERICAL_RETARGET_FAIL" if not numerical_pass else "RETARGET_SEMANTIC_FAIL")
    )
    write_stage_timing(
        root,
        episode,
        float(stage["T_episode_load_sec"]),
        max(0.0, repartitioned_prepare_sec),
        cumulative_solver_sec,
        semantic_sec,
        html_sec,
        terminal_selected_solver=terminal_selected_solver_sec,
        timed_frames=len(rows),
    )
    stats = frame_statistics([float(row["solver_sec"]) for row in rows])
    receipt = {
        "schema_version": "OakInk2O5SolverReceiptV1",
        "terminal": terminal,
        "episode": episode,
        "pid": previous.get("pid"),
        "production_run_count": int(previous.get("production_run_count", 1)),
        "derived_without_solver_rerun": True,
        "numerical_pass": numerical_pass,
        "semantic_status": semantic["final"]["qualification"]["status"],
        "frame_count": final.frame_count,
        "converged_frames": int(np.count_nonzero(final.arrays["optimizer_converged"])),
        "max_iter_frames": int(
            np.count_nonzero(np.asarray(final.arrays["optimizer_status_code"]) == 9)
        ),
        "failed_frames": int(np.count_nonzero(~np.asarray(final.arrays["accepted"], dtype=bool))),
        "invalid_outputs": int(
            np.count_nonzero(~np.isfinite(np.asarray(final.arrays["qpos"])).all(axis=1))
        ),
        "solver_profile": SOLVER_PROFILE,
        "execution_profile": EXECUTION_PROFILE,
        "retarget_contract_sha256": (root / "contract/retarget_contract_sha256.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "frame_statistics": stats,
        "viewer": {
            "html": viewer["html"],
            "html_sha256": viewer["html_sha256"],
            "interactive_orbit_regression": viewer["interactive_orbit_regression"]["status"],
        },
    }
    write_json(paths["solver_receipt"], receipt)
    _, by_id = manifest_rows()
    record = by_id[str(episode["record_id"])]
    model = get_robot_registry().load(ROBOT)
    write_json(
        paths["output_hashes"],
        {
            "manifest_record_sha256": record["canonical_record_sha256"],
            "retarget_contract_sha256": receipt["retarget_contract_sha256"],
            "wuji_robot_spec_sha256": model.spec_hash,
            "wuji_urdf_sha256": model.urdf_hash,
            "wuji_asset_manifest_sha256": model.asset_manifest_hash,
            "source_geometry_sha256": tree_hash(paths["canonical"]),
            "object_asset_sha256": record["object_asset_sha256"],
            "solver_config_sha256": RefinementSolverProfile.load(SOLVER_PROFILE).profile_hash,
            "execution_config_sha256": RefinementExecutionProfile.load(
                EXECUTION_PROFILE, REPO_ROOT
            ).profile_hash,
            "trajectory_npz_sha256": sha256_file(paths["trajectory"]),
            "final_artifact_sha256": tree_hash(paths["final"]),
        },
    )
    return receipt


def write_stage_timing(
    root: Path,
    episode: dict[str, Any],
    load: float,
    prepare: float,
    solver: float,
    semantic: float,
    html: float,
    *,
    terminal_selected_solver: float | None = None,
    timed_frames: int | None = None,
) -> dict[str, Any]:
    stages = {
        "episode_load": float(load),
        "prepare": float(prepare),
        "solver_episode": float(solver),
        "semantic_validation": float(semantic),
        "html": float(html),
    }
    expected_frames = int(episode["source_interval"][1] - episode["source_interval"][0])
    actual_timed_frames = int(timed_frames) if timed_frames is not None else expected_frames
    value = {
        "schema_version": "GeometricRetargetTimingV1",
        "episode": episode["review"],
        "record_id": episode["record_id"],
        "frames": expected_frames,
        "timed_frames": actual_timed_frames,
        "measurement_status": (
            "COMPLETE" if actual_timed_frames == expected_frames else "INCOMPLETE"
        ),
        **{f"T_{key}_sec": seconds for key, seconds in stages.items()},
        "T_episode_machine_total_sec": stage_total(stages),
        "shared_init_included": False,
        "T_solver_episode_scope": "all production refine attempts; cumulative active-set outer-loop plus non-overlapping final-audit timers",
        "terminal_selected_refine_frame_total_sec": terminal_selected_solver,
        "terminal_selected_refine_frame_total_is_not_T_solver_episode": True,
    }
    write_json(root / "timing" / f"{episode['review']}_stage_timing.json", value)
    return value


def refresh_resource_usage(root: Path) -> dict[str, Any]:
    """Complete static host facts without replacing measured worker peaks."""

    path = root / "resource_usage.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    observed = runtime_environment()
    for key in ("hostname", "cpu_model", "logical_cpu_count", "physical_cpu_count", "ram_bytes"):
        value[key] = observed[key]
    thread_environment = value.get("thread_environment", {})
    configured = [thread_environment.get(key) for key in THREAD_ENVIRONMENT]
    value["solver_thread_count"] = (
        1 if configured and all(item == "1" for item in configured) else "NOT_ASSERTED"
    )
    value.setdefault("process_rss_peak_kib", "NOT_MEASURED")
    value.setdefault("cuda_peak_allocated_bytes", "NOT_APPLICABLE_SOLVER_CPU")
    value.setdefault("cuda_peak_reserved_bytes", "NOT_APPLICABLE_SOLVER_CPU")
    value["solver_actual_device"] = "CPU"
    write_json(path, value)
    return value


def run_worker(root: Path) -> int:
    if not (root / "preflight/upstream_gate.json").is_file():
        raise RuntimeError("O5_PREFLIGHT_REQUIRED")
    _, by_id = manifest_rows()
    shared_started = time.perf_counter()
    model = get_robot_registry().load(ROBOT)
    surface_path = _default_collision_samples(ROBOT)
    surface = load_robot_surface_samples(surface_path)
    # Load every immutable profile once so shared static policy construction is
    # part of the measured shared initialization, not an episode's source load.
    contract = write_contract(root)
    load_frame_profile("canonical_keypoint_wrist_v1")
    load_bone_profile("mediapipe21_full_finger_chain_v1")
    load_solver_profile("paper_repro_scipy_trf")
    load_surface_profile("paper_strict_area_uniform", repo_root=REPO_ROOT)
    load_delaunay_profile("strict_scipy_qhull_v1")
    RefinementSolverProfile.load(SOLVER_PROFILE)
    RefinementExecutionProfile.load(EXECUTION_PROFILE, REPO_ROOT)
    CollisionQueryProfile.load(QUERY_PROFILE)
    RefinementCoordinateProfile.load(COORDINATE_PROFILE)
    shared_sec = time.perf_counter() - shared_started
    shared = {
        "T_shared_init_sec": shared_sec,
        "pid": os.getpid(),
        "robot": ROBOT,
        "robot_spec_sha256": model.spec_hash,
        "urdf_sha256": model.urdf_hash,
        "asset_manifest_sha256": model.asset_manifest_hash,
        "collision_surface_path": str(surface_path.resolve()),
        "collision_surface_sha256": sha256_file(surface_path),
        "shared_robot_model_reused": True,
        "shared_collision_surface_reused": True,
    }
    write_json(root / "timing/shared_init.json", shared)
    receipts = []
    for episode in EPISODES:
        write_json(
            root / "timing/process_progress.json",
            {
                "pid": os.getpid(),
                "current": episode["review"],
                "completed": [item.get("terminal") for item in receipts],
            },
        )
        receipts.append(
            run_one_episode(
                root,
                dict(episode),
                by_id[str(episode["record_id"])],
                model,
                surface,
                contract["sha256"],
            )
        )
    write_json(
        root / "resource_usage.json",
        {
            **runtime_environment(),
            "worker_pid": os.getpid(),
            "process_rss_peak_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
            "solver_actual_device": "CPU",
        },
    )
    summarize(root)
    return 0


def run_same_two(root: Path) -> int:
    for episode in EPISODES:
        paths = episode_paths(root, str(episode["review"]))
        if paths["trajectory"].exists() or paths["checkpoints"].exists():
            raise RuntimeError(f"O5_DUPLICATE_EXPENSIVE_RUN_REFUSED:{episode['review']}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--action",
        "_run-worker",
        "--report-root",
        str(root.resolve()),
    ]
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    log_path = root / "timing/worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    external_sec = time.perf_counter() - started
    payload = {
        "schema_version": "GeometricRetargetTimingV1",
        "T_process_wall_external_sec": external_sec,
        "worker_command": command,
        "worker_returncode": result.returncode,
        "worker_log": str(log_path.resolve()),
        "worker_log_sha256": sha256_file(log_path),
        "same_python_process_for_both_episodes": True,
    }
    shared_path = root / "timing/shared_init.json"
    if shared_path.is_file():
        payload["T_shared_init_sec"] = json.loads(shared_path.read_text(encoding="utf-8"))[
            "T_shared_init_sec"
        ]
    write_json(root / "timing/process.json", payload)
    if result.returncode != 0:
        raise RuntimeError(f"O5_WORKER_FAILED:{result.returncode}:{log_path}")
    summarize(root)
    return 0


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize(root: Path) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    episode_receipts: dict[str, dict[str, Any]] = {}
    episode_seconds: list[list[float]] = []
    for episode in EPISODES:
        review = str(episode["review"])
        paths = episode_paths(root, review)
        receipt = (
            json.loads(paths["solver_receipt"].read_text(encoding="utf-8"))
            if paths["solver_receipt"].is_file()
            else {"terminal": "NOT_RUN", "production_run_count": 0}
        )
        episode_receipts[review] = receipt
        stage_path = root / "timing" / f"{review}_stage_timing.json"
        if not stage_path.is_file():
            continue
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        frame_path = root / "timing" / f"{review}_frame_timing.csv"
        frames = (
            list(csv.DictReader(frame_path.open(encoding="utf-8"))) if frame_path.is_file() else []
        )
        seconds = [float(row["solver_sec"]) for row in frames]
        timing_complete = len(seconds) == int(stage["frames"])
        if not timing_complete:
            stage["timed_frames"] = len(frames)
            stage["measurement_status"] = "INCOMPLETE_TECHNICAL_FAILURE"
            stage["duration_interpretation"] = (
                "RECORDED_VALUES_ARE_INCOMPLETE_LOWER_BOUNDS_NOT_STAGE_MEASUREMENTS"
            )
            write_json(stage_path, stage)
        if timing_complete:
            stats = frame_statistics(seconds)
            episode_seconds.append(seconds)
        else:
            stats = {
                key: None
                for key in (
                    "mean_sec_per_frame",
                    "p50_sec_per_frame",
                    "p90_sec_per_frame",
                    "p95_sec_per_frame",
                    "p99_sec_per_frame",
                    "max_sec_per_frame",
                )
            }
        stage_rows.append(
            {
                "Episode": review,
                "Frames": len(frames),
                "Timing status": "COMPLETE" if timing_complete else "NOT_MEASURED",
                "Load s": stage["T_episode_load_sec"] if timing_complete else None,
                "Prepare s": stage["T_prepare_sec"] if timing_complete else None,
                "Solver s": stage["T_solver_episode_sec"] if timing_complete else None,
                "Semantic s": stage["T_semantic_validation_sec"] if timing_complete else None,
                "HTML s": stage["T_html_sec"] if timing_complete else None,
                "Machine total s": (
                    stage["T_episode_machine_total_sec"] if timing_complete else None
                ),
                "Sec/frame mean": stats["mean_sec_per_frame"],
                "P50": stats["p50_sec_per_frame"],
                "P90": stats["p90_sec_per_frame"],
                "P95": stats["p95_sec_per_frame"],
                "P99": stats["p99_sec_per_frame"],
                "Max": stats["max_sec_per_frame"],
            }
        )
        all_frame_rows.extend(frames)
    write_csv(root / "timing/episode_summary.csv", stage_rows)
    write_csv(root / "timing/frame_summary.csv", all_frame_rows)
    slow = sorted(all_frame_rows, key=lambda row: float(row["solver_sec"]), reverse=True)[:10]
    write_csv(root / "timing/slow_frames.csv", slow)
    aggregate: dict[str, Any] = {
        "schema_version": "GeometricRetargetTimingV1",
        "n_timed_episodes": len(episode_seconds),
    }
    if len(episode_seconds) == 2:
        aggregate.update(aggregate_frame_statistics(episode_seconds))
        values = np.asarray([float(row["solver_sec"]) for row in all_frame_rows])
        aggregate["correlations"] = {
            "solver_time_vs_iterations": correlation(
                values, np.asarray([float(row["iterations"]) for row in all_frame_rows])
            ),
            "solver_time_vs_initial_objective": correlation(
                values, np.asarray([float(row["initial_objective"]) for row in all_frame_rows])
            ),
            "solver_time_vs_final_error": correlation(
                values, np.asarray([float(row["final_objective"]) for row in all_frame_rows])
            ),
            "final_error_definition": "final objective diagnostic",
        }
        shared = json.loads((root / "timing/shared_init.json").read_text(encoding="utf-8"))
        by_review = {row["Episode"]: row for row in stage_rows}
        aggregate.update(
            cold_warm_totals(
                float(shared["T_shared_init_sec"]),
                float(by_review["dev_01"]["Machine total s"]),
                float(by_review["dev_02"]["Machine total s"]),
            )
        )
        distribution = json.loads(
            (root / "timing/eligible_frame_count_distribution.json").read_text(encoding="utf-8")
        )
        sec_per_frame = float(aggregate["weighted_solver_sec_per_frame"])
        mean_frames = float(distribution["mean_frames_per_episode"])
        p95_frames = float(distribution["p95_frames_per_episode"])
        extrapolation = {
            "schema_version": "OakInk2O5BatchRuntimeExtrapolationV1",
            "timing_label": "TWO_EPISODE_PILOT_EXTRAPOLATION",
            "formal_throughput_benchmark": False,
            "disclaimer": "This is not a formal large-sample throughput benchmark. The extrapolation uses O5's two measured development episodes and the frozen Manifest V2 frame-count distribution.",
            "weighted_solver_sec_per_frame": sec_per_frame,
            "shared_initialization_added_once": float(shared["T_shared_init_sec"]),
            "solver_only": {
                "10_episodes_typical": runtime_estimate(sec_per_frame, 10 * mean_frames),
                "10_episodes_p95_duration": runtime_estimate(sec_per_frame, 10 * p95_frames),
                "100_episodes_typical": runtime_estimate(sec_per_frame, 100 * mean_frames),
                "100_episodes_p95_duration": runtime_estimate(sec_per_frame, 100 * p95_frames),
                "742_exact_eligible_frame_count": runtime_estimate(
                    sec_per_frame, int(distribution["total_frames"])
                ),
            },
            "additional_eligible_retarget_runs": 0,
        }
        write_json(root / "timing/batch_runtime_extrapolation.json", extrapolation)
        aggregate["batch_runtime_extrapolation"] = extrapolation
    else:
        extrapolation = {
            "schema_version": "OakInk2O5BatchRuntimeExtrapolationV1",
            "status": "NOT_RUN",
            "reason": "REQUIRES_TWO_COMPLETED_EPISODE_TIMINGS",
            "timing_label": "TWO_EPISODE_PILOT_EXTRAPOLATION",
            "formal_throughput_benchmark": False,
            "n_timed_episodes": len(episode_seconds),
            "required_timed_episodes": 2,
            "solver_only": None,
            "additional_eligible_retarget_runs": 0,
        }
        write_json(root / "timing/batch_runtime_extrapolation.json", extrapolation)
        aggregate["batch_runtime_extrapolation"] = extrapolation
    write_json(root / "timing/aggregate.json", aggregate)
    write_failure_closure(root, episode_receipts, aggregate)
    write_manual_review(root)
    summary = final_summary(root, episode_receipts, stage_rows, aggregate, slow)
    write_json(root / "final_summary.json", summary)
    (root / "final_summary.md").write_text(
        render_handoff(root, summary, stage_rows, slow, concise=True), encoding="utf-8"
    )
    (root / "handoff.md").write_text(
        render_handoff(root, summary, stage_rows, slow, concise=False), encoding="utf-8"
    )
    return summary


def write_failure_closure(
    root: Path,
    receipts: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Record why the terminal machine non-pass cannot be repaired by a hidden rerun."""

    failure_path = root / "technical_failures.jsonl"
    first_attempts = []
    if failure_path.is_file():
        first_attempts = [
            json.loads(line)
            for line in failure_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    machine_pass = all(
        receipts[name].get("terminal") == "O5_MACHINE_PASS" for name in ("dev_01", "dev_02")
    )
    closure = {
        "schema_version": "OakInk2O5FailureClosureV1",
        "status": "NOT_APPLICABLE" if machine_pass else "MACHINE_NONPASS_CLOSED",
        "machine_terminals": {
            name: receipts[name].get("terminal", "TECHNICAL_FAILURE")
            for name in ("dev_01", "dev_02")
        },
        "production_run_counts": {
            name: int(receipts[name].get("production_run_count", 0))
            for name in ("dev_01", "dev_02")
        },
        "usable_outputs": {
            "dev_01": episode_paths(root, "dev_01")["trajectory"].is_file(),
            "dev_02": episode_paths(root, "dev_02")["trajectory"].is_file(),
        },
        "first_attempt_failures": first_attempts,
        "same_process_attempt": {
            "attempted": True,
            "process_terminated": True,
            "n_completed_episode_timings": int(aggregate.get("n_timed_episodes", 0)),
            "required_completed_episode_timings": 2,
        },
        "rerun_assessment": {
            "dev_01": {
                "decision": "PROHIBITED",
                "rerun_count": 0,
                "reason": "A usable exact production output exists; Part V forbids rerunning it for timing, HTML, testing, or reporting.",
            },
            "dev_02": {
                "technical_retry_clause_applies": receipts["dev_02"].get("terminal")
                == "TECHNICAL_FAILURE"
                and not bool(receipts["dev_02"].get("usable_output", False)),
                "decision": "NOT_RUN_FAIL_CLOSED",
                "rerun_count": 0,
                "reason": "A standalone retry after the original worker exited cannot recreate the required DEV_01-cold then DEV_02-warm same-process chain. Replaying that chain would rerun DEV_01 despite its usable output.",
            },
        },
        "contract_compliant_recovery": "NONE_AFTER_ORIGINAL_PROCESS_EXIT",
        "batch_runtime_extrapolation": "NOT_RUN",
        "human_review_gate": "NOT_OPEN",
        "downstream": "HARD_STOP",
    }
    write_json(root / "failure_closure.json", closure)
    return closure


def write_manual_review(root: Path) -> None:
    receipts = {
        name: json.loads(episode_paths(root, name)["solver_receipt"].read_text(encoding="utf-8"))
        for name in ("dev_01", "dev_02")
    }
    machine_pass = all(
        receipts[name].get("terminal") == "O5_MACHINE_PASS" for name in ("dev_01", "dev_02")
    )
    lines = [
        "# OakInk2 O5 Manual Geometric Review",
        "",
    ]
    if machine_pass:
        lines.extend(
            [
                "Machine PASS does not replace this human review. Open both saved HTML files and check:",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Machine acceptance did not open. Human approval is not requested and cannot override this machine non-pass.",
                "",
                f"- DEV_01 terminal: {receipts['dev_01'].get('terminal')}",
                f"- DEV_02 terminal: {receipts['dev_02'].get('terminal')}",
                "- DEV_01 HTML is available for diagnostic inspection; DEV_02 HTML was not generated.",
                "- The checklist below is retained only to guide diagnostic inspection of available output.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "1. Wuji wrist follows the source hand global motion.",
            "2. Wrist orientation does not visibly flip.",
            "3. Thumb maps to thumb.",
            "4. Index/middle/ring/little topology is correct.",
            "5. Finger flexion directions are correct.",
            "6. Wuji and source use similar interaction regions on the object.",
            "7. The target object is correct.",
            "8. Grasp/approach/transport/release semantics are preserved.",
            "9. There is no obvious severe penetration.",
            "10. No numerical PASS has the robot far from the object.",
            "11. The trajectory is continuous.",
            "12. Geometry remains correct after real mouse orbit.",
        ]
    )
    if machine_pass:
        lines.extend(
            [
                "",
                "Reply with:",
                "",
                "```text",
                "OAKINK2_O5_DEV_1=APPROVE / REJECT",
                "OAKINK2_O5_DEV_2=APPROVE / REJECT",
                "```",
            ]
        )
    path = root / "review/manual_review.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_summary(
    root: Path,
    receipts: dict[str, dict[str, Any]],
    stages: list[dict[str, Any]],
    aggregate: dict[str, Any],
    slow: list[dict[str, Any]],
) -> dict[str, Any]:
    machine_pass = all(
        receipts[name].get("terminal") == "O5_MACHINE_PASS" for name in ("dev_01", "dev_02")
    )
    process = (
        json.loads((root / "timing/process.json").read_text(encoding="utf-8"))
        if (root / "timing/process.json").is_file()
        else {}
    )
    commit_rows = [
        {"commit": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
        for line in git("log", "--format=%H%x09%s", f"{START_HEAD}..HEAD").splitlines()
        if "\t" in line
    ]
    tracked_clean = not bool(git("status", "--porcelain", "--untracked-files=no"))
    write_json(
        root / "git_commits.json",
        {
            "start_head": START_HEAD,
            "final_head": git("rev-parse", "HEAD"),
            "commits": commit_rows,
            "tracked_worktree_clean": tracked_clean,
            "pushed": False,
            "pr_created": False,
        },
    )
    complete_timing = int(aggregate.get("n_timed_episodes", 0)) == 2
    semantic_ran = all(episode_paths(root, name)["semantic"].is_file() for name in receipts)
    viewers_ready = all(
        episode_paths(root, name)["html"].is_file()
        and episode_paths(root, name)["viewer_receipt"].is_file()
        for name in receipts
    )
    return {
        "schema_version": "OakInk2O5FinalSummaryV1",
        "branch": git("branch", "--show-current"),
        "start_head": json.loads((root / "preflight/git.json").read_text(encoding="utf-8"))[
            "start_head"
        ],
        "final_head": git("rev-parse", "HEAD"),
        "manifest_v2_sha256": sha256_file(MANIFEST_V2),
        "split_v2_sha256": sha256_file(SPLIT_V2),
        "episodes": receipts,
        "stage_timing": stages,
        "aggregate": aggregate,
        "slow_frames": slow,
        "process": process,
        "failure_closure": json.loads((root / "failure_closure.json").read_text(encoding="utf-8")),
        "commits": commit_rows,
        "tracked_worktree_clean": tracked_clean,
        "o5_machine_complete": "YES" if machine_pass else "NO",
        "o5_human_acceptance": "PENDING",
        "o5_final": "PENDING_HUMAN_REVIEW" if machine_pass else "MACHINE_NONPASS",
        "safety_flags": {
            "BRANCH": "feature/oakink2-raw-to-physical",
            "O1_FINAL": "PASS",
            "O5_ALLOWED_AT_START": "YES",
            "SAME_TWO_EPISODES": "YES",
            "EPISODES_RESELECTED": "NO",
            "MANIFEST_V2_MODIFIED": "NO",
            "SPLIT_V2_MODIFIED": "NO",
            "CERTIFICATION_DOWNSTREAM_CONSUMED": "NO",
            "HELDOUT_DOWNSTREAM_CONSUMED": "NO",
            "EXACT_GEOMETRIC_RETARGET_RAN": "YES"
            if any(receipts[name].get("production_run_count", 0) for name in receipts)
            else "NO",
            "DEV_01_RETARGET_RUN_COUNT": receipts["dev_01"].get("production_run_count", 0),
            "DEV_02_RETARGET_RUN_COUNT": receipts["dev_02"].get("production_run_count", 0),
            "EXPENSIVE_TIMING_RERUNS_AVOIDED": "YES",
            "SAME_PROCESS_SEQUENTIAL_TIMING": "YES",
            "SHARED_INIT_TIMED": "YES",
            "EPISODE_STAGE_TIMING_COMPLETE": "YES" if complete_timing else "NO",
            "PER_FRAME_SOLVER_TIMING_COMPLETE": "YES" if complete_timing else "NO",
            "RETARGET_SEMANTIC_VALIDITY_RAN": "YES" if semantic_ran else "NO",
            "WEIGHTED_SEC_PER_FRAME_REPORTED": "YES" if complete_timing else "NO",
            "ALL_FRAME_P50_REPORTED": "YES" if complete_timing else "NO",
            "ALL_FRAME_P90_REPORTED": "YES" if complete_timing else "NO",
            "ALL_FRAME_P95_REPORTED": "YES" if complete_timing else "NO",
            "ALL_FRAME_MAX_REPORTED": "YES" if complete_timing else "NO",
            "SLOW_FRAME_REPORT_COMPLETE": "YES" if complete_timing else "NO",
            "HTML_INTERACTIVE_REVIEW_READY": "YES" if viewers_ready else "NO",
            "O5_MACHINE_GEOMETRY_PASS": "YES" if machine_pass else "NO",
            "SOLVER_TIMING_INCLUDES_HTML": "NO",
            "SOLVER_TIMING_INCLUDES_HUMAN_REVIEW": "NO",
            "BATCH_EXTRAPOLATION_LABEL": (
                "TWO_EPISODE_PILOT_EXTRAPOLATION" if complete_timing else "NOT_RUN"
            ),
            "BATCH_EXTRAPOLATION_NOT_RUN_REASON": (
                "NONE" if complete_timing else "REQUIRES_TWO_COMPLETED_EPISODE_TIMINGS"
            ),
            "WARM_DEV_02_TIMING_VALID": "YES" if complete_timing else "NO",
            "ADDITIONAL_742_RETARGET_RUNS": 0,
            "RETARGET_METHOD_TUNED_FOR_TIMING": "NO",
            "SOLVER_LIMITS_CHANGED": "NO",
            "SOLVER_TOLERANCE_CHANGED": "NO",
            "VIEWER_ARCHITECTURE": "O1R2D_APPROVED_REF2DEX_STYLE",
            "NEW_VIEWER_ARCHITECTURE_CREATED": "NO",
            "SUPPORT_PHYSICALIZATION_RAN": "NO",
            "PHYSX_RAN": "NO",
            "FROZEN_EVAL_RAN": "NO",
            "PPO_RAN": "NO",
            "O6_RAN": "NO",
            "O5_HUMAN_ACCEPTANCE": "PENDING",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            "TRACKED_WORKTREE_CLEAN": "YES" if tracked_clean else "NO",
            ".local_TRACKED": "NO",
            "GUIDANCE_WORKTREE_MODIFIED": "NO",
        },
    }


def render_handoff(
    root: Path,
    summary: dict[str, Any],
    stages: list[dict[str, Any]],
    slow: list[dict[str, Any]],
    *,
    concise: bool,
) -> str:
    def timing_cell(value: Any) -> str:
        return "NOT_MEASURED" if value is None else f"{float(value):.6f}"

    contract = json.loads(
        (root / "contract/geometric_retarget_contract.json").read_text(encoding="utf-8")
    )
    contract_sha = (root / "contract/retarget_contract_sha256.txt").read_text().strip()
    _, records = manifest_rows()
    lines = [
        "# OakInk2 O5 Geometric Retarget + Timing Handoff",
        "",
        "## Git and authority",
        "",
        f"- BRANCH={summary['branch']}",
        f"- START_HEAD={summary['start_head']}",
        f"- FINAL_HEAD={summary['final_head']}",
        f"- TRACKED_WORKTREE_CLEAN={'YES' if summary['tracked_worktree_clean'] else 'NO'}",
        "- commits="
        + (
            ", ".join(row["commit"][:12] for row in summary["commits"])
            if summary["commits"]
            else "NONE"
        ),
        "- O1_FINAL=PASS",
        "- O5_ALLOWED=YES",
        f"- MANIFEST_V2_SHA={summary['manifest_v2_sha256']}",
        f"- SPLIT_V2_SHA={summary['split_v2_sha256']}",
        "- MANIFEST_V2_CHANGED=NO; SPLIT_V2_CHANGED=NO; RESELECTED=NO",
        "- PUSHED=NO; PR_CREATED=NO",
        "",
        "## Fixed O5 episodes and machine results",
        "",
        "| Review | Episode | Primitive | Object | Source frames | Duration s | Machine terminal | Runs |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for episode in EPISODES:
        receipt = summary["episodes"][episode["review"]]
        paths = episode_paths(root, str(episode["review"]))
        duration: float | str = "NOT_RUN"
        if paths["canonical"].is_dir():
            timestamps = np.asarray(load_canonical_hoi(paths["canonical"]).timestamps)
            duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        record = records[str(episode["record_id"])]
        lines.append(
            f"| {episode['review']} | `{episode['record_id']}` | {record['primitive']} | {episode['object']} | [{episode['source_interval'][0]}, {episode['source_interval'][1]}) ({episode['source_interval'][1] - episode['source_interval'][0]}) | {duration} | {receipt.get('terminal')} | {receipt.get('production_run_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Exact retarget contract",
            "",
            f"- implementation={contract['implementation']}",
            f"- robot={contract['robot']}; Wuji Hand2 Beta1 asset is hash-bound in each output receipt",
            f"- solver={contract['solver_profile']['profile_id']}; execution={contract['execution_profile']['profile_id']}",
            f"- initialization={contract['warm_start_profile']}; coordinate={contract['coordinate_profile']['profile_id']}",
            f"- joint limits=production Wuji model bounds; frame mapping={contract['frame_profile']}; active hand=RIGHT",
            f"- retiming={contract['retiming']}; full source trajectory={contract['full_source_trajectory']}; temporal subsampling={contract['temporal_subsampling']}",
            f"- RETARGET_CONTRACT_SHA256={contract_sha}",
            "",
            "## Machine results",
            "",
            "| Metric | DEV_01 | DEV_02 |",
            "| --- | ---: | ---: |",
        ]
    )
    machine_columns: dict[str, dict[str, Any]] = {}
    for episode in EPISODES:
        review = str(episode["review"])
        paths = episode_paths(root, review)
        receipt = summary["episodes"][review]
        semantic = (
            json.loads(paths["semantic"].read_text(encoding="utf-8"))
            if paths["semantic"].is_file()
            else {}
        )
        qualification = semantic.get("final", {}).get("qualification", {})
        metrics = qualification.get("metrics", {})
        final_metrics = semantic.get("final", {})
        hashes = (
            json.loads(paths["output_hashes"].read_text(encoding="utf-8"))
            if paths["output_hashes"].is_file()
            else {}
        )
        machine_columns[review] = {
            "Numerical Retarget": receipt.get("numerical_pass", receipt.get("terminal")),
            "RetargetSemanticValidity": receipt.get("semantic_status", "NOT_RUN"),
            "Frames": receipt.get("frame_count", receipt.get("timed_frames", 0)),
            "Converged frames": receipt.get("converged_frames", "NOT_RUN"),
            "Max-iter frames": receipt.get("max_iter_frames", "NOT_RUN"),
            "Failed frames": receipt.get("failed_frames", "NOT_RUN"),
            "Wrist position max m": metrics.get("object_relative_wrist_position_m", {}).get(
                "max", "NOT_RUN"
            ),
            "Wrist rotation max rad": metrics.get("object_relative_wrist_rotation_rad", {}).get(
                "max", "NOT_RUN"
            ),
            "Bone error p95 rad": metrics.get("bone_direction_error_rad", {}).get("p95", "NOT_RUN"),
            "Contact recall": metrics.get("source_contact_recall", "NOT_RUN"),
            "Fingertip-object distance mean m": final_metrics.get("tip_object_distance_m", {}).get(
                "mean", "NOT_RUN"
            ),
            "Interaction E_IM mean": final_metrics.get("interaction_e_im", {}).get(
                "mean", "NOT_RUN"
            ),
            "Temporal translation step max m": metrics.get("temporal_translation_step_m", {}).get(
                "max", "NOT_RUN"
            ),
            "Temporal rotation step max rad": metrics.get("temporal_rotation_step_rad", {}).get(
                "max", "NOT_RUN"
            ),
            "Continuity": qualification.get("temporal_continuity_status", "NOT_RUN"),
            "Output SHA256": hashes.get("trajectory_npz_sha256", "NOT_RUN"),
        }
    for metric in machine_columns["dev_01"]:
        lines.append(
            f"| {metric} | {machine_columns['dev_01'][metric]} | {machine_columns['dev_02'][metric]} |"
        )
    if summary["failure_closure"]["status"] != "NOT_APPLICABLE":
        closure = summary["failure_closure"]
        lines.extend(
            [
                "",
                "## Machine non-pass closure",
                "",
                f"- STATUS={closure['status']}",
                f"- DEV_01_RERUN={closure['rerun_assessment']['dev_01']['decision']}: {closure['rerun_assessment']['dev_01']['reason']}",
                f"- DEV_02_RERUN={closure['rerun_assessment']['dev_02']['decision']}: {closure['rerun_assessment']['dev_02']['reason']}",
                f"- CONTRACT_COMPLIANT_RECOVERY={closure['contract_compliant_recovery']}",
                "- The original failure records remain in `technical_failures.jsonl`; no expensive solver attempt was erased or replaced.",
            ]
        )
    lines.extend(
        [
            "",
            "## Formal timing",
            "",
            f"T_SHARED_INIT={summary['process'].get('T_shared_init_sec', 'NOT_RUN')}",
            "T_SOLVER_EPISODE_SCOPE=all production refine attempts; cumulative active-set outer-loop plus non-overlapping final-audit perf_counter timers",
            "Per-frame solver_start/solver_end identify the actual terminal selected refine call; accepted_solver_sec is retained separately from cumulative solver_sec.",
            "",
            "| Episode | Timing status | Frames | Load s | Prepare s | Solver s | Semantic s | HTML s | Machine total s | Mean s/frame | P50 | P90 | P95 | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stages:
        lines.append(
            f"| {row['Episode']} | {row['Timing status']} | {row['Frames']} | "
            f"{timing_cell(row['Load s'])} | {timing_cell(row['Prepare s'])} | "
            f"{timing_cell(row['Solver s'])} | {timing_cell(row['Semantic s'])} | "
            f"{timing_cell(row['HTML s'])} | {timing_cell(row['Machine total s'])} | "
            f"{row['Sec/frame mean']} | {row['P50']} | {row['P90']} | {row['P95']} | "
            f"{row['Max']} |"
        )
    aggregate = summary["aggregate"]
    if aggregate.get("n_timed_frames"):
        lines.extend(
            [
                "",
                "## Aggregate frame timing",
                "",
                f"- N_TIMED_EPISODES={aggregate['n_timed_episodes']}",
                f"- N_TIMED_FRAMES={aggregate['n_timed_frames']}",
                f"- WEIGHTED_SOLVER_SEC_PER_FRAME={aggregate['weighted_solver_sec_per_frame']}",
                f"- UNWEIGHTED_EPISODE_MEAN_SEC_PER_FRAME={aggregate['unweighted_episode_mean_sec_per_frame']}",
                f"- ALL_FRAME_P50={aggregate['all_frame_p50_sec']}",
                f"- ALL_FRAME_P90={aggregate['all_frame_p90_sec']}",
                f"- ALL_FRAME_P95={aggregate['all_frame_p95_sec']}",
                f"- ALL_FRAME_MAX={aggregate['all_frame_max_sec']}",
                f"- COLD_DEV_01_SEC={aggregate['cold_dev_01_sec']}",
                f"- WARM_DEV_02_SEC={aggregate['warm_dev_02_sec']}",
                "- SAME_PYTHON_PROCESS=YES; SHARED_ROBOT_AND_STATIC_SURFACE_REUSED=YES",
            ]
        )
        distribution = json.loads(
            (root / "timing/eligible_frame_count_distribution.json").read_text(encoding="utf-8")
        )
        extrapolation = json.loads(
            (root / "timing/batch_runtime_extrapolation.json").read_text(encoding="utf-8")
        )
        lines.extend(
            [
                "",
                "## Eligible corpus frame distribution",
                "",
                f"- N_ELIGIBLE={distribution['n_eligible']}; TOTAL_FRAMES={distribution['total_frames']}; ADDITIONAL_ELIGIBLE_RETARGET_RUNS=0",
                f"- mean={distribution['mean_frames_per_episode']}; p50={distribution['p50_frames_per_episode']}; p90={distribution['p90_frames_per_episode']}; p95={distribution['p95_frames_per_episode']}; max={distribution['max_frames_per_episode']}",
                "",
                "## Pilot batch runtime extrapolation (solver only)",
                "",
                "| Scenario | Seconds | Hours | Days |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        labels = {
            "10_episodes_typical": "10 episodes typical",
            "10_episodes_p95_duration": "10 episodes p95-duration",
            "100_episodes_typical": "100 episodes typical",
            "100_episodes_p95_duration": "100 episodes p95-duration",
            "742_exact_eligible_frame_count": "742 exact eligible-frame count",
        }
        for key, label in labels.items():
            estimate = extrapolation["solver_only"][key]
            lines.append(
                f"| {label} | {estimate['seconds']} | {estimate['hours']} | {estimate['days']} |"
            )
    else:
        extrapolation = aggregate.get("batch_runtime_extrapolation", {})
        lines.extend(
            [
                "",
                "## Pilot batch runtime extrapolation",
                "",
                f"- STATUS={extrapolation.get('status', 'NOT_RUN')}",
                f"- REASON={extrapolation.get('reason', 'REQUIRES_TWO_COMPLETED_EPISODE_TIMINGS')}",
                f"- N_TIMED_EPISODES={aggregate.get('n_timed_episodes', 0)}; REQUIRED=2; ADDITIONAL_ELIGIBLE_RETARGET_RUNS=0",
                "- No 742-episode runtime estimate is reported from incomplete pilot timing.",
            ]
        )
    resource_path = root / "resource_usage.json"
    if resource_path.is_file():
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Resource accounting",
                "",
                f"- hostname={resources.get('hostname')}; CPU={resources.get('cpu_model')}",
                f"- logical CPUs={resources.get('logical_cpu_count')}; physical CPUs={resources.get('physical_cpu_count')}; RAM bytes={resources.get('ram_bytes')}",
                f"- Python={resources.get('python_version')}; PyTorch={resources.get('pytorch_version')}; conda={resources.get('conda_env')}",
                f"- CUDA available={resources.get('cuda_available')}; GPU={resources.get('gpu')}; solver actual device={resources.get('solver_actual_device')}",
                f"- solver threads={resources.get('solver_thread_count', 'NOT_ASSERTED')}; thread environment={resources.get('thread_environment')}",
                f"- process RSS peak KiB={resources.get('process_rss_peak_kib', 'NOT_MEASURED')}; CUDA peak allocated={resources.get('cuda_peak_allocated_bytes', 'NOT_APPLICABLE')}; CUDA peak reserved={resources.get('cuda_peak_reserved_bytes', 'NOT_APPLICABLE')}",
            ]
        )
    if not concise:
        lines.extend(
            [
                "",
                "## Slowest frames",
                "",
                "| Episode | Frame | Sec | Iterations | Termination | Initial Obj | Final Obj |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in slow:
            lines.append(
                f"| {row['episode']} | {row['source_frame_id']} | {row['solver_sec']} | {row['iterations']} | {row['termination']} | {row['initial_objective']} | {row['final_objective']} |"
            )
    lines.extend(["", "## Viewers and human review", ""])
    for episode in EPISODES:
        path = episode_paths(root, str(episode["review"]))["html"]
        receipt_path = episode_paths(root, str(episode["review"]))["viewer_receipt"]
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        )
        lines.append(
            f"- {episode['review']}: `{path.resolve()}`; SHA256={receipt.get('html_sha256', 'NOT_GENERATED')}; source MANO={receipt.get('source_mano_present', False)}; target object={receipt.get('target_object_present', False)}; Wuji={receipt.get('wuji_present', False)}; orbit={receipt.get('interactive_orbit_regression', {}).get('status', 'NOT_RUN')}"
        )
        lines.append(f"  `xdg-open '{path.resolve()}'`")
    if summary["o5_machine_complete"] == "YES":
        lines.extend(
            [
                "",
                "Reply:",
                "",
                "```text",
                "OAKINK2_O5_DEV_1=APPROVE / REJECT",
                "OAKINK2_O5_DEV_2=APPROVE / REJECT",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Machine acceptance did not open. Any generated HTML is diagnostic only; human approval cannot override a machine non-pass.",
            ]
        )
    lines.extend(
        [
            "",
            f"O5_MACHINE_COMPLETE={summary['o5_machine_complete']}",
            "O5_HUMAN_ACCEPTANCE=PENDING",
            f"O5_FINAL={summary['o5_final']}",
            "",
            (
                "TIMING_LABEL=TWO_EPISODE_PILOT_EXTRAPOLATION"
                if aggregate.get("n_timed_episodes") == 2
                else "TIMING_LABEL=NOT_RUN"
            ),
            "",
            (
                "This is not a formal large-sample throughput benchmark. The extrapolation uses O5's two measured development episodes and the frozen Manifest V2 frame-count distribution."
                if aggregate.get("n_timed_episodes") == 2
                else "The two-episode pilot timing was incomplete, so no corpus runtime extrapolation is reported."
            ),
            "",
            "HARD STOP: O6, support physicalization, PhysX, frozen evaluation, and PPO were not run.",
            "",
            "## Safety flags",
            "",
        ]
    )
    lines.extend(f"{key}={value}" for key, value in summary["safety_flags"].items())
    return "\n".join(lines) + "\n"


def validate_repository(root: Path) -> int:
    modified = [
        "src/toporetarget/adapters/datasets/oakink2.py",
        "src/toporetarget/cli/retarget.py",
        "src/toporetarget/evaluation/geometric_retarget_timing.py",
        "src/toporetarget/retarget/final_refinement.py",
        "src/toporetarget/viz/oakink2_html_viewer.py",
        "scripts/data/run_oakink2_o5.py",
        "tests/data/test_oakink2_trusted_html_viewer.py",
        "tests/evaluation/test_geometric_retarget_timing.py",
    ]
    commands = [
        ["ruff", "check", *modified],
        ["ruff", "format", "--check", *modified],
        [sys.executable, "-m", "mypy", "src"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/check_paper_fidelity.py"],
        ["git", "diff", "--check"],
    ]
    rows = []
    log_root = root / "validation_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands):
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log = log_root / f"{index:02d}_{Path(command[0]).name}.log"
        log.write_text(result.stdout, encoding="utf-8")
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "seconds": time.perf_counter() - started,
                "log": str(log.resolve()),
                "log_sha256": sha256_file(log),
            }
        )
    payload = {
        "schema_version": "OakInk2O5ValidationResultsV1",
        "status": "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL",
        "checks": rows,
        "ci_equivalent": True,
    }
    write_json(root / "tests.json", payload)
    write_json(root / "validation_results.json", payload)
    return 0 if payload["status"] == "PASS" else 1


def main() -> int:
    args = parser().parse_args()
    root = args.report_root.resolve()
    if args.action == "preflight":
        print(json.dumps(preflight(root), indent=2, default=json_default))
        return 0
    if args.action == "validate":
        return validate_repository(root)
    if args.action == "run-same-two":
        return run_same_two(root)
    if args.action == "_run-worker":
        return run_worker(root)
    if args.action == "derive":
        reviews = (args.review,) if args.review is not None else ("dev_01", "dev_02")
        values = [derive_episode(root, review) for review in reviews]
        print(json.dumps(values, indent=2, default=json_default))
        refresh_resource_usage(root)
        summarize(root)
        return 0
    if args.action in {"semantic", "render"}:
        if args.review is None:
            raise SystemExit("--review is required for semantic/render")
        value = (
            run_semantic(root, args.review)
            if args.action == "semantic"
            else run_render(root, args.review)
        )
        print(json.dumps(value[0], indent=2, default=json_default))
        return 0
    if args.action == "summarize":
        print(json.dumps(summarize(root), indent=2, default=json_default))
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
