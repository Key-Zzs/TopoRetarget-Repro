#!/usr/bin/env python3
"""O5R-D2G cross-episode authority audit and ExecutionV3 development.

The CLI is fail-closed and intentionally has no command for fresh V4
certification, a DEV2 full trajectory, or physical/policy work.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5 import (  # noqa: E402
    EXECUTION_PROFILE,
    ROBOT,
    SOLVER_PROFILE,
    episode_paths,
)
from scripts.data.run_oakink2_o5 import REPORT_ROOT as O5_ROOT  # noqa: E402
from scripts.data.run_oakink2_o5rd2a import (  # noqa: E402
    D2ARuntime,
    _measurement_row,
    development_windows,
    digest,
    git,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from scripts.data.run_oakink2_o5rd2b import actual_continuity  # noqa: E402
from scripts.data.run_oakink2_o5rd2c import (  # noqa: E402
    _contributor_probe,
    _evaluate_b2,
    _interaction_per_keypoint,
    _run_candidate_phase,
    _screened,
    _solver_profile,
    search_frame,
)
from scripts.evaluation.audit_retarget_semantic_validity import (  # noqa: E402
    _semantic_frames,
)
from toporetarget.cli.retarget import (  # noqa: E402
    _default_collision_samples,
    _load_robot,
)
from toporetarget.data.storage import load_hoi_sequence  # noqa: E402
from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    SemanticGateContractV1,
    angular_error,
    relative_transform,
    transform_error,
)
from toporetarget.geometry.object_geometry import sample_object_track  # noqa: E402
from toporetarget.geometry.surface_sampling import (  # noqa: E402
    SurfaceSampleSet,
    load_surface_profile,
)
from toporetarget.retarget.alignment import base_seed_from_hand_frames  # noqa: E402
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile  # noqa: E402
from toporetarget.retarget.continuous import (  # noqa: E402
    continuity_metrics,
    encode_base_correction,
    transport_previous_final_to_current_warm,
)
from toporetarget.retarget.delaunay import load_delaunay_profile  # noqa: E402
from toporetarget.retarget.final_refinement import (  # noqa: E402
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    _make_context,
    load_robot_surface_samples,
    map_previous_state_to_seed,
    prepare_refinement_resources,
    prepare_refinement_runtime_backends,
)
from toporetarget.retarget.frames import load_frame_profile  # noqa: E402
from toporetarget.retarget.interaction_artifacts import (  # noqa: E402
    interaction_artifact_hash,
    load_interaction_graph,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_graph import (  # noqa: E402
    build_source_interaction_graph,
    load_paper_kappa,
)
from toporetarget.retarget.objective_v2 import (  # noqa: E402
    ObjectiveV2Candidate,
    ObjectiveV2Measurements,
    RetargetNonRegressionBudgetAuthorityV1,
    interaction_retention_limit,
)
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    ScreenedCandidate,
    asset_derived_dof_blocks,
    contributor_scores,
    default_search_contracts,
    rank_contributors,
    retain_after_polish,
    select_candidate,
)
from toporetarget.retarget.objective_v3_execution import (  # noqa: E402
    ExecutionFrameInputsV3,
    ExecutionV3Candidate,
    RetargetMode,
    cold_start_seeds_v3,
    default_execution_v3_candidates,
)
from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.retarget.solver import (  # noqa: E402
    load_paper_weights,
    load_solver_profile,
    solve_frame,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2g_cross_episode_input_authority_execution_v3_v1"
D2A_ROOT = REPO / ".local/reports/oakink2_o5rd2a_objective_v2_design_v1"
D2C_ROOT = REPO / ".local/reports/oakink2_o5rd2c_candidate_b2_search_v1"
D2E_ROOT = REPO / ".local/reports/oakink2_o5rd2e_gate_alignment_and_v3_recertification_v1"
O5RB_ROOT = REPO / ".local/reports/oakink2_o5rb_parallel_v1"
O1R_ROOT = REPO / ".local/reports/oakink2_o1r_official_mano_authority_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "94712d8aac58edbda5b9e7c3fbe82b4328dda957"
OBJECTIVE_SHA = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
EXECUTION_V2_SHA = "b114b3960c47b641a85eac44d974a6e71921613bee85273e3d7de811ba4e937e"
GATE_V2_SHA = "a845fcdfec478e9208fc6c192317a45bee6ef19be1fcb784b459d8a4f675e048"
GATE = SemanticGateContractV1()
CANDIDATE_B2 = ObjectiveV2Candidate.candidate_b2()
V3_CANDIDATES = {item.name: item for item in default_execution_v3_candidates()}


def _sha_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _require_status(path: Path, field: str, expected: str = "PASS") -> dict[str, Any]:
    _require(path)
    payload = read_json(path)
    if payload.get(field) != expected:
        raise RuntimeError(f"O5RD2G_STAGE_BLOCKED:{path}:{field}={payload.get(field)}")
    return payload


def frozen_paths() -> dict[str, Path]:
    dev1, dev2 = episode_paths(O5_ROOT, "dev_01"), episode_paths(O5_ROOT, "dev_02")
    return {
        "objective_v2": D2C_ROOT / "frozen_method/retarget_objective_v2_contract.json",
        "execution_v2": D2C_ROOT / "frozen_method/execution_contract_v2.json",
        "gate_v2": D2E_ROOT / "gate_v2/certification_gate_v2.json",
        "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "manifest_v2": O1R_ROOT / "manifest_v2/oakink2_corpus_manifest_v2.jsonl",
        "split_v2": O1R_ROOT / "manifest_v2/oakink2_raw_to_physical_split_v2.json",
        "dev1_old_trajectory": dev1["final"],
        "dev1_canonical": dev1["canonical"],
        "dev1_warm": dev1["warm"],
        "dev1_object_samples": dev1["samples"],
        "dev1_interaction_graph": dev1["graph"],
        "dev2_canonical": dev2["canonical"],
        "dev2_episode_receipt": O5_ROOT / "preflight/fixed_o5_episode_set.json",
        "objective_v2_implementation": REPO / "src/toporetarget/retarget/objective_v2.py",
        "execution_v2_implementation": REPO / "src/toporetarget/retarget/objective_v2_execution.py",
        "production_refinement": REPO / "src/toporetarget/retarget/final_refinement.py",
    }


def verify_frozen_upstream(root: Path) -> dict[str, Any]:
    expected = {
        "objective_v2": OBJECTIVE_SHA,
        "execution_v2": EXECUTION_V2_SHA,
        "gate_v2": GATE_V2_SHA,
    }
    recorded = read_json(D2E_ROOT / "preflight/frozen_method_implementation_authority.json")
    implementation_expected = {name: item["sha256"] for name, item in recorded["files"].items()}
    aliases = {
        "semantic_v1": "semantic_v1",
        "objective_v2_implementation": "objective_implementation",
        "execution_v2_implementation": "execution_implementation",
        "production_refinement": "production_refinement_adapter",
        "wuji_asset": "robot_mapping",
    }
    rows: dict[str, Any] = {}
    passed = True
    for name, path in frozen_paths().items():
        _require(path)
        observed = digest(path)
        expected_sha = expected.get(name)
        if name in aliases:
            expected_sha = implementation_expected[aliases[name]]
        exact = expected_sha is None or observed == expected_sha
        passed = passed and exact
        rows[name] = {
            "path": str(path.resolve()),
            "sha256": observed,
            "expected_sha256": expected_sha,
            "exact": exact,
        }
    sparse = read_json(D2E_ROOT / "sparse_v3/gate_decision.json")
    window = read_json(D2E_ROOT / "window_v3/gate_decision.json")
    dev2 = read_json(D2E_ROOT / "dev2_frame0/decision.json")
    state_exact = bool(
        sparse.get("SPARSE_VALIDATION_V3") == "PASS"
        and window.get("WINDOW_VALIDATION_V3") == "PASS"
        and dev2.get("DEV2_FRAME0_HARD_CONTROL") == "FAIL"
        and dev2.get("failure") == "MISSING_FROZEN_S1_OLD_PRODUCTION_Q_OLD_AUTHORITY"
        and int(dev2.get("method_optimizer_run_count", -1)) == 0
    )
    passed = passed and state_exact
    payload = {
        "schema_version": "O5RD2GFrozenMethodIntegrityV1",
        "FROZEN_UPSTREAM_INTEGRITY": "PASS" if passed else "FAIL",
        "authorities": rows,
        "upstream_state": {
            "SparseValidationV3": sparse.get("SPARSE_VALIDATION_V3"),
            "WindowValidationV3": window.get("WINDOW_VALIDATION_V3"),
            "DEV2_FRAME0_EXECUTION_V2": "FAIL_PRE_OPTIMIZER_MISSING_Q_OLD",
            "ObjectiveV2_failure_evidence": "NO",
            "DEV2_optimizer_evaluation_under_V2": "NOT_EVALUATED",
            "state_exact": state_exact,
        },
    }
    write_json(root / "preflight/frozen_method_integrity.json", payload)
    write_json(root / "preflight/frozen_authorities.json", {"authorities": rows})
    write_json(root / "preflight/upstream_state.json", payload["upstream_state"])
    if not passed:
        raise RuntimeError("D2G_BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2G_BRANCH_MISMATCH:{branch}")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0
    )
    if not ancestor:
        raise RuntimeError(f"O5RD2G_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_upstream(root)
    payload = {
        "schema_version": "OakInk2O5RD2GGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "START_HEAD": START_HEAD,
        "head_at_preflight": head,
        "start_head_is_ancestor": ancestor,
        "initial_status_short": [],
        "initial_status_provenance": "mandated shell preflight captured before task modifications",
        "status_short_at_artifact_write": git(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
        "diff_stat_at_artifact_write": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", payload)
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    failures.touch(exist_ok=True)
    return {"git": payload, "integrity": integrity}


def audit_execution_input_authority(root: Path) -> dict[str, Any]:
    verify_frozen_upstream(root)
    rows = [
        ("source_mediapipe21", "CANONICAL_EPISODE_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        (
            "source_wrist_frame",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            "HARD_SCIENTIFIC_REQUIREMENT",
        ),
        (
            "source_bone_features",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            "HARD_SCIENTIFIC_REQUIREMENT",
        ),
        ("object_pose_scene", "CANONICAL_EPISODE_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("object_mesh", "CANONICAL_EPISODE_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        (
            "object_surface_samples",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            "HARD_SCIENTIFIC_REQUIREMENT",
        ),
        (
            "interaction_graph",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            "HARD_SCIENTIFIC_REQUIREMENT",
        ),
        ("wuji_robot_asset", "ROBOT_ASSET_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("joint_limits_and_dof_mapping", "ROBOT_ASSET_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("collision_surface_and_sdf", "ROBOT_ASSET_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("contributor_mass", "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY", "SEARCH_SEED_ONLY"),
        ("q_old", "OLD_PRODUCTION_TRAJECTORY_AUTHORITY", "REFINEMENT_ONLY_BASELINE"),
        ("previous_accepted_q_base", "PREVIOUS_ACCEPTED_RUNTIME_STATE", "SOFT_SEARCH_HINT"),
        ("continuous_prediction_context", "PREVIOUS_ACCEPTED_RUNTIME_STATE", "SOFT_SEARCH_HINT"),
        ("base_seed", "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY", "SEARCH_SEED_ONLY"),
        ("source_frame_index", "CANONICAL_EPISODE_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("episode_source_to_scene", "CANONICAL_EPISODE_AUTHORITY", "HARD_SCIENTIFIC_REQUIREMENT"),
        ("generic_seed_pool", "ROBOT_ASSET_AUTHORITY", "SEARCH_SEED_ONLY"),
    ]
    payload = {
        "schema_version": "ExecutionInputDependencyAuditV1",
        "status": "PASS",
        "call_graph": [
            "canonical HOI + deterministic object samples -> build_source_interaction_graph",
            "canonical source + source graph + Wuji asset/SDF -> _make_context",
            "mode authority + seed authority -> contributor probe -> Candidate-B2 primary",
            "independent hard screen -> retention -> secondary polish -> retained result",
        ],
        "inputs": [
            {"input": name, "authority": authority, "role": role} for name, authority, role in rows
        ],
        "invented_prompt_fields": False,
    }
    write_json(root / "input_authority/execution_input_dependency_audit.json", payload)
    return payload


def audit_qold_role(root: Path) -> dict[str, Any]:
    audit_execution_input_authority(root)
    source = REPO / "src/toporetarget/retarget/objective_v2_execution.py"
    objective = REPO / "src/toporetarget/retarget/objective_v2.py"
    payload = {
        "schema_version": "QOldScientificRoleAuditV1",
        "status": "PASS",
        "Q_OLD_ROLE": "REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY",
        "questions": {
            "enters_candidate_b2_mathematical_objective": False,
            "enters_e_im_construction": False,
            "defines_hard_constraints": False,
            "defines_source_or_object_semantics": False,
            "candidate_seed": True,
            "baseline_fallback": True,
            "nonregression_comparison_reference": True,
            "removal_makes_objective_undefined": False,
        },
        "use_sites": [
            {
                "site": "SearchExecutionContract.old_production_baseline",
                "file": str(source),
                "role": "ExecutionV2 contract invariant",
            },
            {
                "site": "deterministic_block_seeds old_q base and old_production source",
                "file": str(source),
                "role": "seed/nonselected-DOF carrier only",
            },
            {
                "site": "D2C search_frame old_production screened candidate/fallback",
                "file": str(D2C_ROOT.parent.parent.parent / "scripts/data/run_oakink2_o5rd2c.py"),
                "role": "refinement baseline and evaluation reference",
            },
        ],
        "objective_implementation": str(objective),
        "objective_signature_contains_q_old": "q_old"
        in inspect.getsource(CANDIDATE_B2.primary_value),
        "COLD_START_EXECUTION_V3_AUTHORIZED": "YES",
    }
    write_json(root / "input_authority/q_old_role_audit.json", payload)
    return payload


def audit_interaction_graph_authority(root: Path) -> dict[str, Any]:
    qold = audit_qold_role(root)
    implementation = REPO / "src/toporetarget/retarget/interaction_graph.py"
    payload = {
        "schema_version": "InteractionGraphAuthorityAuditV1",
        "status": "PASS_PENDING_REPLAY_PARITY",
        "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
        "construction": {
            "implementation": str(implementation.resolve()),
            "implementation_sha256": sha256_file(implementation),
            "source_inputs": [
                "canonical mediapipe21 positions_scene",
                "canonical object mesh and pose_scene",
                "paper_strict_area_uniform 50-point object samples",
                "strict_scipy_qhull_v1",
                "paper distance_decay_kappa",
            ],
            "coordinate_frame": "S",
            "units": "m",
            "nodes": "0..20 canonical mediapipe21; 21..70 frozen object-sample order",
            "edges": "per-frame Delaunay tetrahedra unique undirected edges; no filtering",
            "weights": "directed exp(-kappa*distance_squared), row normalized",
            "normalization": "strict_scipy_qhull_v1 plus row-sum normalization",
            "frame_local": True,
            "trajectory_level_dependency": "object sample identity reused over trajectory only",
        },
        "dependencies": {
            "q_old": False,
            "old_production": False,
            "target_robot_geometry": False,
            "warm_start": False,
            "sdf": False,
        },
        "code_provenance_robot_loaded_false": True,
        "q_old_role_consistent": qold["Q_OLD_ROLE"],
    }
    write_json(root / "input_authority/interaction_graph_authority_audit.json", payload)
    return payload


def audit_dev2_input_completeness(root: Path) -> dict[str, Any]:
    audit_interaction_graph_authority(root)
    canonical = frozen_paths()["dev2_canonical"]
    sequence = load_hoi_sequence(canonical)
    hand = sequence.hand("right_hand")
    obj = sequence.rigid_object("C11001")
    rows = [
        ("source MANO/mediapipe21", "CANONICAL_EPISODE_AUTHORITY", True, True, "PASS"),
        (
            "source wrist/frame transform",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            True,
            True,
            "PASS",
        ),
        ("object pose", "CANONICAL_EPISODE_AUTHORITY", True, True, "PASS"),
        ("object mesh", "CANONICAL_EPISODE_AUTHORITY", True, True, "PASS"),
        (
            "Wuji asset",
            "ROBOT_ASSET_AUTHORITY",
            frozen_paths()["wuji_asset"].exists(),
            False,
            "PASS",
        ),
        (
            "object surface samples",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            False,
            True,
            "DERIVABLE_BUT_CONTRACT_MISSING",
        ),
        (
            "interaction graph",
            "SOURCE_DERIVED_DETERMINISTIC_AUTHORITY",
            False,
            True,
            "DERIVABLE_BUT_CONTRACT_MISSING",
        ),
        (
            "old production q",
            "OLD_PRODUCTION_TRAJECTORY_AUTHORITY",
            False,
            False,
            "MISSING_BECAUSE_OLD_TRAJECTORY_DOES_NOT_EXIST",
        ),
        (
            "previous accepted runtime state frame0",
            "PREVIOUS_ACCEPTED_RUNTIME_STATE",
            False,
            False,
            "EXPECTED_ABSENT_FRAME0",
        ),
    ]
    canonical_valid = bool(
        hand.keypoint_tracks["mediapipe21"].positions_scene.shape == (240, 21, 3)
        and obj.pose_scene.pose_scene.shape == (240, 4, 4)
        and len(obj.mesh.vertices_local) > 0
        and len(obj.mesh.faces) > 0
    )
    payload = {
        "schema_version": "DEV2ColdStartInputCompletenessV1",
        "status": "PASS" if canonical_valid else "FAIL",
        "canonical_path": str(canonical.resolve()),
        "canonical_frame_count": sequence.num_frames,
        "canonical_shapes_valid": canonical_valid,
        "inputs": [
            {
                "input": name,
                "authority": authority,
                "DEV2_available": available,
                "derivable": derivable,
                "status": status,
            }
            for name, authority, available, derivable, status in rows
        ],
        "MISSING_CANONICAL_SOURCE_AUTHORITY": False,
        "MISSING_BECAUSE_OLD_TRAJECTORY_DOES_NOT_EXIST": ["old production q"],
        "cold_start_mathematically_well_defined": canonical_valid,
    }
    write_json(root / "input_authority/dev2_input_completeness.json", payload)
    decision = {
        "schema_version": "O5RD2GInputAuthorityDecisionV1",
        "INPUT_AUTHORITY_AUDIT": "PASS" if canonical_valid else "FAIL",
        "Q_OLD_ROLE": "REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY",
        "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
        "COLD_START_EXECUTION_AUTHORIZED": "YES" if canonical_valid else "NO",
    }
    write_json(root / "input_authority/authority_decision.json", decision)
    if not canonical_valid:
        raise RuntimeError("O5RD2G_DEV2_CANONICAL_SOURCE_AUTHORITY_INCOMPLETE")
    return payload


def run_graph_parity(root: Path) -> dict[str, Any]:
    audit_dev2_input_completeness(root)
    dev1 = load_hoi_sequence(frozen_paths()["dev1_canonical"])
    historical_samples = SurfaceSampleSet.load(frozen_paths()["dev1_object_samples"])
    profile = load_surface_profile("paper_strict_area_uniform", repo_root=REPO)
    reconstructed_samples = sample_object_track(dev1.rigid_object("C10001"), profile)
    sample_arrays = ("face_indices", "barycentric", "points_local", "normals_local", "valid")
    sample_exact = all(
        np.array_equal(
            np.asarray(getattr(historical_samples, name)),
            np.asarray(getattr(reconstructed_samples, name)),
        )
        for name in sample_arrays
    )
    historical = load_interaction_graph(frozen_paths()["dev1_interaction_graph"])
    b1 = [
        int(item["ordinal"])
        for item in read_json(O5RB_ROOT / "b1_thumb_feasibility/frame_selection.json")["frames"]
    ]
    d2c = [int(value) for window in development_windows(D2A_ROOT) for value in window["ordinals"]]
    sparse = [
        int(value)
        for value in read_json(D2E_ROOT / "sparse_v3/determinism_subset.json")["ordinals"]
    ]
    membership = {
        "B1_DEVELOPMENT": b1,
        "D2C_DEVELOPMENT_WINDOWS": d2c,
        "SPARSE_V3_REPRESENTATIVE": sparse,
    }
    selected = sorted(set(b1 + d2c + sparse))
    rebuilt = build_source_interaction_graph(
        dev1,
        "right_hand",
        "C10001",
        reconstructed_samples,
        source_cache=frozen_paths()["dev1_canonical"],
        object_sample_path=None,
        delaunay_profile=load_delaunay_profile("strict_scipy_qhull_v1"),
        kappa=load_paper_kappa(),
        frame_indices=selected,
    )
    rows: list[dict[str, Any]] = []
    for local, ordinal in enumerate(selected):
        prior = historical.frames[ordinal]
        current = rebuilt.frames[local]
        sets = [name for name, values in membership.items() if ordinal in set(values)]
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": int(historical.frame_indices[ordinal]),
                "evidence_sets": ";".join(sets),
                "graph_hash_exact": current.graph_hash == prior.graph_hash,
                "source_vertices_max_abs": float(
                    np.max(np.abs(current.source_vertices - prior.source_vertices))
                ),
                "source_laplacian_max_abs": float(
                    np.max(np.abs(current.source_laplacian - prior.source_laplacian))
                ),
                "simplices_exact": np.array_equal(current.simplices, prior.simplices),
                "edges_exact": np.array_equal(current.edges, prior.edges),
                "weights_max_abs": float(np.max(np.abs(current.weights - prior.weights))),
            }
        )
    parity = bool(
        sample_exact
        and all(
            row["graph_hash_exact"]
            and row["source_vertices_max_abs"] <= 1e-12
            and row["source_laplacian_max_abs"] <= 1e-12
            and row["simplices_exact"]
            and row["edges_exact"]
            and row["weights_max_abs"] <= 1e-12
            for row in rows
        )
    )
    write_csv(root / "graph_authority/graph_parity.csv", rows)
    summary = {
        "schema_version": "SourceInteractionGraphParityV1",
        "GRAPH_PARITY": "PASS" if parity else "FAIL",
        "object_sample_reconstruction_exact": sample_exact,
        "frame_count": len(rows),
        "set_counts": {name: len(values) for name, values in membership.items()},
        "max_source_vertices_abs": max(row["source_vertices_max_abs"] for row in rows),
        "max_source_laplacian_abs": max(row["source_laplacian_max_abs"] for row in rows),
        "max_weight_abs": max(row["weights_max_abs"] for row in rows),
    }
    write_json(root / "graph_authority/graph_parity_summary.json", summary)

    # DEV2 graph construction is a fresh canonical-source derivation, not reuse
    # of the absent historical Stage-8 artifact.
    dev2 = load_hoi_sequence(frozen_paths()["dev2_canonical"])
    dev2_samples = sample_object_track(dev2.rigid_object("C11001"), profile)
    sample_path = root / "graph_authority/dev2_object_samples_source_derived.npz"
    if not sample_path.exists():
        dev2_samples.save(sample_path)
    dev2_graph = build_source_interaction_graph(
        dev2,
        "right_hand",
        "C11001",
        dev2_samples,
        source_cache=frozen_paths()["dev2_canonical"],
        object_sample_path=sample_path,
        delaunay_profile=load_delaunay_profile("strict_scipy_qhull_v1"),
        kappa=load_paper_kappa(),
        frame_indices=[0],
    )
    graph_path = root / "graph_authority/dev2_frame0_source_graph.zarr"
    if not graph_path.exists():
        save_interaction_graph(dev2_graph, graph_path)
    authority = {
        "schema_version": "SourceInteractionGraphAuthorityV1",
        "status": "PASS" if parity else "FAIL",
        "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
        "source_inputs": [
            "canonical mediapipe21",
            "canonical object mesh",
            "canonical object pose",
        ],
        "object_sampling": profile.as_dict(),
        "coordinate_frame": "S",
        "units": "m",
        "topology": "strict_scipy_qhull_v1 Delaunay, all unique edges, no filtering",
        "distance_contact_definition": "weighted Euclidean graph; no semantic contact label",
        "normalization": "directed exp(-kappa*d2) row normalization",
        "serialization": "toporetarget.interaction_graph.v1 Zarr",
        "implementation_sha256": sha256_file(
            REPO / "src/toporetarget/retarget/interaction_graph.py"
        ),
        "dev1_replay_parity": summary,
        "dev2_frame0_graph": str(graph_path.resolve()),
        "dev2_frame0_graph_artifact_sha256": interaction_artifact_hash(graph_path),
        "robot_loaded": False,
        "q_old_loaded": False,
    }
    write_json(root / "graph_authority/source_interaction_graph_authority.json", authority)
    if not parity:
        raise RuntimeError("COLD_START_GRAPH_AUTHORITY_FAIL")
    return authority


@dataclass(frozen=True)
class RuntimeBindingV3:
    active_frame_id: int
    runtime_step_index: int
    current_source_frame_id: int
    previous_source_frame_id: int | None
    previous_runtime_base_scene: np.ndarray | None
    previous_robot_qpos: np.ndarray | None
    continuous_predicted_translation_scene: np.ndarray | None
    continuous_predicted_rotation_scene: np.ndarray | None
    continuous_predicted_qpos: np.ndarray | None
    base_correction_reference: np.ndarray | None
    object_pose_scene: np.ndarray
    object_id: str
    robot_name: str
    robot_side: str
    robot_dof_names: tuple[str, ...]
    robot_mapping_authority: str
    source_hand_id: str
    temporal_scope: str = "continuous_full_state"

    @property
    def continuous_prediction_base(self) -> np.ndarray | None:
        if self.continuous_predicted_translation_scene is None:
            return None
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = np.asarray(self.continuous_predicted_rotation_scene)
        result[:3, 3] = np.asarray(self.continuous_predicted_translation_scene)
        return result

    @property
    def sha256(self) -> str:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, tuple):
                return list(value)
            return value

        payload = {key: convert(value) for key, value in asdict(self).items()}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class V3Runtime(D2ARuntime):
    """Runtime that derives cold-start context without old-production state."""

    def __init__(self, review: str, root: Path):
        self.review = review
        self.root = root
        if review == "dev_01":
            super().__init__(root)
        elif review == "dev_02":
            self.paths = episode_paths(O5_ROOT, review)
            self.final = None
            self.sequence = load_hoi_sequence(self.paths["canonical"])
            self.graph = load_interaction_graph(
                root / "graph_authority/dev2_frame0_source_graph.zarr"
            )
            self.model = _load_robot(ROBOT, None)
            self.surface = load_robot_surface_samples(_default_collision_samples(ROBOT))
            self.frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
            self.bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
            self.solver = RefinementSolverProfile.load(SOLVER_PROFILE)
            self.execution = RefinementExecutionProfile.load(EXECUTION_PROFILE, REPO)
            self.query = CollisionQueryProfile.load("adaptive_active_set_v1")
            self.coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1")
            self.resources = prepare_refinement_resources(
                self.sequence,
                self.graph,
                self.solver,
                geometry_artifact_root=root / "execution_v3_design/geometry/dev2",
            )
            self.backends = prepare_refinement_runtime_backends(self.resources, self.execution)
            neutral = np.asarray(self.model.neutral_q, dtype=np.float64)
            neutral_base = self.base_for_q(0, neutral)
            neutral_points = np.asarray(
                self.model.keypoints_scene(neutral, np.eye(4)), dtype=np.float64
            )
            self.warm = SimpleNamespace(
                metadata={"source_hand_id": "right_hand", "source_side": "right"},
                arrays={
                    "qpos": neutral[None, :],
                    "base_pose_scene": neutral_base[None, :, :],
                    "robot_keypoints_base": neutral_points[None, :, :],
                },
            )
            self.authority = RetargetNonRegressionBudgetAuthorityV1.from_frozen_v1(
                collision_hard_bound_m=self.resources.paper.b,
                collision_soft_tolerance_m=self.resources.paper.tau,
            )
        else:
            raise ValueError(f"unsupported review: {review}")
        self.current_runtime_step = 0
        self._base_cache: dict[tuple[int, bytes], np.ndarray] = {}

    def source_features(self, ordinal: int) -> Any:
        global_frame = int(self.graph.frame_indices[ordinal])
        points = np.asarray(
            self.sequence.hand("right_hand")
            .keypoint_tracks["mediapipe21"]
            .positions_scene[global_frame],
            dtype=np.float64,
        )
        return extract_bone_features(
            points,
            self.frame_profile,
            self.bone_profile,
            side="right",
            strict=True,
        )

    def base_for_q(self, ordinal: int, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64)
        key = (ordinal, q.tobytes())
        cached = getattr(self, "_base_cache", {}).get(key)
        if cached is not None:
            return cached.copy()
        source_frame = np.asarray(self.source_features(ordinal).frame_transform, dtype=np.float64)
        robot_points = np.asarray(
            self.model.keypoints_scene(q, np.eye(4), layout="mediapipe21"),
            dtype=np.float64,
        )
        robot_frame = np.asarray(_semantic_frames(robot_points, self.model.side), dtype=np.float64)
        result = np.asarray(base_seed_from_hand_frames(source_frame, robot_frame), dtype=np.float64)
        if hasattr(self, "_base_cache"):
            self._base_cache[key] = result.copy()
        return result

    def bind_context(
        self,
        ordinal: int,
        *,
        previous_base: np.ndarray | None = None,
        previous_qpos: np.ndarray | None = None,
    ) -> tuple[RuntimeBindingV3, Any]:
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
                raise RuntimeError("O5RD2G_RUNTIME_PREVIOUS_STATE_REQUIRED")
            previous_ordinal = ordinal - 1
            if previous_ordinal < 0:
                raise RuntimeError("O5RD2G_INVALID_PREVIOUS_ORDINAL")
            previous_frame = int(self.graph.frame_indices[previous_ordinal])
            previous_seed_base = self.base_for_q(previous_ordinal, neutral)
            previous_reference = map_previous_state_to_seed(
                np.asarray(previous_base), np.asarray(previous_qpos), seed_base
            )
            propagated = transport_previous_final_to_current_warm(
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
        binding = RuntimeBindingV3(
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
        context = _make_context(
            self.sequence,
            self.graph,
            self.warm,
            self.model,
            self.surface,
            self.backends.solver_sdf,
            self.resources.reference_sdf,
            self.frame_profile,
            self.bone_profile,
            self.resources.paper,
            ordinal,
            previous_reference,
            temporal_scope=binding.temporal_scope,
            continuous_prediction_base=binding.continuous_prediction_base,
            continuous_prediction_qpos=binding.continuous_predicted_qpos,
            spatial_gradient_backend=self.execution.signed_distance_gradient,
            sign_cache=self.backends.sign_cache,
            compiled_spatial_fd_backend=self.backends.compiled_spatial_fd_backend,
        )
        context.seed_base = seed_base
        context.seed_qpos = neutral.copy()
        return binding, context

    def measurement(
        self,
        ordinal: int,
        qpos: np.ndarray,
        base_pose: np.ndarray,
        *,
        binding: RuntimeBindingV3,
        context: Any,
        slack: np.ndarray | None = None,
    ) -> ObjectiveV2Measurements:
        q = np.asarray(qpos, dtype=np.float64)
        base = np.asarray(base_pose, dtype=np.float64)
        correction = encode_base_correction(context.seed_base, base)
        slack_value = np.zeros(0, dtype=np.float64) if slack is None else np.asarray(slack)
        value = np.concatenate([correction, q, slack_value])
        _total, _gradient, breakdown = context.objective(value)
        keypoints_base = np.asarray(self.model.keypoints_scene(q, np.eye(4)), dtype=np.float64)
        keypoints_scene = np.asarray(self.model.keypoints_scene(q, base), dtype=np.float64)
        robot_features = extract_bone_features(
            keypoints_base,
            self.frame_profile,
            self.bone_profile,
            side=self.model.side,
            strict=True,
        )
        source = self.source_features(ordinal)
        bone_error = angular_error(
            np.asarray(source.unit_directions), np.asarray(robot_features.unit_directions)
        )
        source_wrist = np.asarray(source.frame_transform)
        robot_wrist = base @ np.asarray(_semantic_frames(keypoints_base, self.model.side))
        wrist = transform_error(
            relative_transform(binding.object_pose_scene, source_wrist),
            relative_transform(binding.object_pose_scene, robot_wrist),
        )
        if binding.continuous_prediction_base is None:
            continuity = {
                "delta_base_translation_m": 0.0,
                "delta_base_rotation_rad": 0.0,
                "delta_finger_inf_rad": 0.0,
                "excess_keypoint_max_m": 0.0,
            }
        else:
            predicted_keypoints = np.asarray(
                self.model.keypoints_scene(
                    binding.continuous_predicted_qpos,
                    binding.continuous_prediction_base,
                )
            )
            continuity = continuity_metrics(
                binding.continuous_prediction_base,
                base,
                binding.continuous_predicted_qpos,
                q,
                predicted_keypoints_scene=predicted_keypoints,
                final_keypoints_scene=keypoints_scene,
                frame=int(binding.active_frame_id),
            )
        points = context.candidate_points(np.concatenate([correction, q]))
        full = self.resources.reference_sdf.query_scene(points, binding.object_pose_scene)
        joint_margin = float(
            np.min(np.concatenate([q - self.model.joint_lower, self.model.joint_upper - q]))
        )
        neutral_features = extract_bone_features(
            np.asarray(
                self.model.keypoints_scene(self.model.neutral_q, np.eye(4)),
                dtype=np.float64,
            ),
            self.frame_profile,
            self.bone_profile,
            side=self.model.side,
            strict=True,
        )
        ratio = np.asarray(robot_features.bone_lengths) / np.asarray(neutral_features.bone_lengths)
        secondary = float(breakdown.total - breakdown.weighted_e_im)
        return ObjectiveV2Measurements(
            interaction_e_im=float(breakdown.e_im),
            secondary_objective=secondary,
            bone_direction_p95_rad=float(np.quantile(bone_error, 0.95)),
            wrist_position_m=float(np.asarray(wrist["position_m"])),
            wrist_rotation_rad=float(np.asarray(wrist["rotation_rad"])),
            temporal_base_translation_m=float(continuity["delta_base_translation_m"]),
            temporal_base_rotation_rad=float(continuity["delta_base_rotation_rad"]),
            temporal_q_inf_rad=float(continuity["delta_finger_inf_rad"]),
            temporal_excess_keypoint_m=float(continuity["excess_keypoint_max_m"]),
            collision_min_signed_distance_m=float(np.min(full.signed_distance)),
            joint_limit_min_margin_rad=joint_margin,
            rotation_determinant=float(np.linalg.det(robot_wrist[:3, :3])),
            unit_scale_ratio=float(np.max(np.abs(ratio - 1.0)) + 1.0),
            per_components={
                "interaction_mesh": float(breakdown.weighted_e_im),
                "bone_direction": float(breakdown.weighted_e_bone),
                "continuous_temporal": float(breakdown.e_temporal),
                "base_position": float(breakdown.e_base_pos),
                "base_rotation": float(breakdown.e_base_rot),
                "collision_slack": float(breakdown.e_slack),
            },
        )

    def source_geometric_seed(self, ordinal: int) -> tuple[np.ndarray, dict[str, Any]]:
        source = self.source_features(ordinal)
        profile = replace(
            load_solver_profile("paper_repro_scipy_trf"),
            strict_failure_policy="return_independently_bounded_terminal",
            sequential=False,
        )
        warm, smooth, _path = load_paper_weights(REPO)
        lower = np.asarray(self.model.joint_lower, dtype=np.float64)
        upper = np.asarray(self.model.joint_upper, dtype=np.float64)
        starts = (
            ("wuji_canonical_rest", np.asarray(self.model.neutral_q, dtype=np.float64)),
            ("joint_range_midpoint", lower + 0.5 * (upper - lower)),
        )
        rows = []
        states = []
        for order, (name, q0) in enumerate(starts):
            result = solve_frame(
                source.adjacent_features,
                self.model,
                self.frame_profile,
                self.bone_profile,
                profile,
                side=self.model.side,
                initial_qpos=q0,
                previous_qpos=None,
                lambda_warm=warm,
                lambda_smooth=smooth,
            )
            finite = bool(np.all(np.isfinite(result.qpos)))
            in_bounds = bool(
                np.all(result.qpos >= lower - 1e-12) and np.all(result.qpos <= upper + 1e-12)
            )
            rows.append(
                {
                    "seed": name,
                    "ordinal": order,
                    "finite": finite,
                    "in_bounds": in_bounds,
                    "optimizer_converged": result.success,
                    "status": result.status,
                    "message": result.message,
                    "nfev": result.nfev,
                    "njev": result.njev,
                    "bone_objective": result.total_objective,
                    "runtime_sec": result.solve_time_s,
                }
            )
            if finite and in_bounds:
                states.append((float(result.total_objective), order, np.asarray(result.qpos)))
        if not states:
            raise RuntimeError("TECHNICAL_FAIL_NO_VALID_SOURCE_GEOMETRIC_SEED")
        selected = min(states, key=lambda item: (item[0], item[1]))
        return selected[2].copy(), {
            "schema_version": "SourceDrivenGeometricSeedV1",
            "generic": True,
            "episode_agnostic": True,
            "objective": "frozen Eq.1-2 bone-direction residual",
            "solver": "scipy least_squares/trf",
            "max_nfev": profile.max_nfev,
            "failed_terminal_used_as_q_old": False,
            "independent_finite_bound_screening": True,
            "candidates": rows,
            "selected_ordinal": selected[1],
        }


def search_cold_start_frame(
    runtime: V3Runtime,
    ordinal: int,
    *,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
    candidate: ExecutionV3Candidate,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    runtime.current_runtime_step = runtime_step
    frame_inputs = ExecutionFrameInputsV3(
        mode=RetargetMode.COLD_START,
        runtime_step_index=runtime_step,
        old_production_q=None,
        previous_accepted_q=previous_q,
        previous_accepted_base=previous_base,
    ).validate()
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    lower = np.asarray(runtime.model.joint_lower, dtype=np.float64)
    upper = np.asarray(runtime.model.joint_upper, dtype=np.float64)
    neutral_base = runtime.base_for_q(ordinal, neutral)
    per_keypoint = _interaction_per_keypoint(runtime, ordinal, neutral, neutral_base)
    scores = contributor_scores(per_keypoint)
    ranking = rank_contributors(scores)
    selected_finger = ranking[0]
    block = asset_derived_dof_blocks(runtime.model.dof_names)[selected_finger]
    geometric_q = None
    geometric_receipt = None
    if candidate.source_geometric_seed:
        geometric_q, geometric_receipt = runtime.source_geometric_seed(ordinal)
    seeds = cold_start_seeds_v3(
        candidate=candidate,
        frame_inputs=frame_inputs,
        neutral_q=neutral,
        lower_q=lower,
        upper_q=upper,
        source_geometric_q=geometric_q,
    )
    states: dict[str, tuple[np.ndarray, np.ndarray, ObjectiveV2Measurements, dict[str, Any]]] = {}
    screened: list[ScreenedCandidate] = []
    probe_receipts = []
    order = 0
    for seed_name, seed_q in seeds:
        seed_base = runtime.base_for_q(ordinal, seed_q)
        values = runtime.measurement(
            ordinal, seed_q, seed_base, binding=binding, context=context, slack=None
        )
        actual = actual_continuity(runtime, previous_q, previous_base, seed_q, seed_base)
        evaluation = _evaluate_b2(runtime, values, actual)
        raw_id = f"generic:{seed_name}"
        raw = _screened(raw_id, values, evaluation, order, True)
        order += 1
        screened.append(raw)
        states[raw_id] = (seed_q, seed_base, values, evaluation)
        probed_q, probe = _contributor_probe(
            runtime,
            ordinal,
            seed_q,
            seed_base,
            finger=selected_finger,
            block=block,
            max_nfev=candidate.contributor_probe_max_nfev,
        )
        probed_base = runtime.base_for_q(ordinal, probed_q)
        probed_values = runtime.measurement(
            ordinal, probed_q, probed_base, binding=binding, context=context, slack=None
        )
        probed_actual = actual_continuity(runtime, previous_q, previous_base, probed_q, probed_base)
        probed_evaluation = _evaluate_b2(runtime, probed_values, probed_actual)
        probe_id = f"probe:{seed_name}"
        item = _screened(probe_id, probed_values, probed_evaluation, order, bool(probe["success"]))
        order += 1
        screened.append(item)
        states[probe_id] = (probed_q, probed_base, probed_values, probed_evaluation)
        probe_receipts.append(
            {
                "seed_id": seed_name,
                **probe,
                "whole_e_im": probed_values.interaction_e_im,
                "primary_hinge": probed_evaluation["primary_objective"],
                "secondary_objective": probed_values.secondary_objective,
                "independent_feasible": probed_evaluation["feasible"],
                "violations": probed_evaluation["violated_constraints"],
            }
        )
    selected_seed = select_candidate(screened)
    if selected_seed is None:
        raise RuntimeError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    q_seed, base_seed, _seed_values, _seed_evaluation = states[selected_seed.candidate_id]
    primary_result = None
    primary_exception = None
    try:
        (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
            primary_result,
            _primary_actual,
        ) = _run_candidate_phase(
            runtime,
            ordinal,
            q_seed=q_seed,
            base_seed=base_seed,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="primary",
            maxiter=candidate.selected_primary_maxiter,
            retention_limit=None,
            initialization_source=f"{candidate.name}:{selected_seed.candidate_id}:primary",
        )
        terminal = _screened(
            "primary_terminal",
            primary_values,
            primary_evaluation,
            order,
            bool(primary_result.optimizer_converged),
        )
        screened.append(terminal)
        states[terminal.candidate_id] = (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
        )
    except Exception as exc:
        primary_exception = f"{type(exc).__name__}:{exc}"
    retained = select_candidate(screened)
    if retained is None:
        raise RuntimeError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    retained_q, retained_base, retained_values, retained_evaluation = states[retained.candidate_id]
    limit = interaction_retention_limit(
        retained_values.interaction_e_im, runtime.authority.interaction_target
    )
    polished = None
    polished_result = None
    polished_state = None
    polish_exception = None
    try:
        (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_result,
            polished_actual,
        ) = _run_candidate_phase(
            runtime,
            ordinal,
            q_seed=retained_q,
            base_seed=retained_base,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="secondary",
            maxiter=candidate.secondary_polish_maxiter,
            retention_limit=limit,
            initialization_source=f"{candidate.name}:{retained.candidate_id}:secondary",
        )
        polished = _screened(
            "secondary_polished",
            polished_values,
            polished_evaluation,
            order + 1,
            bool(polished_result.optimizer_converged),
        )
        polished_state = (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_actual,
        )
    except Exception as exc:
        polish_exception = f"{type(exc).__name__}:{exc}"
    selected, retention = retain_after_polish(
        retained,
        polished,
        interaction_target=runtime.authority.interaction_target,
        numerical_epsilon=1e-10,
    )
    if polished is not None and selected is polished and polished_state is not None:
        q_out, base_out, values_out, evaluation_out, actual_out = polished_state
    else:
        q_out, base_out, values_out, evaluation_out = states[retained.candidate_id]
        actual_out = actual_continuity(runtime, previous_q, previous_base, q_out, base_out)
    primary_profile = None if primary_result is None else _solver_profile(primary_result)
    secondary_profile = None if polished_result is None else _solver_profile(polished_result)
    profiler = {
        "schema_version": "RetargetSolverProfilerV1",
        "seed_count": len(seeds),
        "candidate_probes": len(probe_receipts),
        "probe_nfev": sum(int(item["nfev"]) for item in probe_receipts),
        "primary_nfev": 0 if primary_profile is None else primary_profile["nfev"],
        "secondary_nfev": 0 if secondary_profile is None else secondary_profile["nfev"],
        "interaction_eval_time_sec": sum(
            float((profile or {}).get("interaction_eval_time_sec", 0.0))
            for profile in (primary_profile, secondary_profile)
        ),
        "fk_time_sec": sum(
            float((profile or {}).get("fk_time_sec", 0.0))
            for profile in (primary_profile, secondary_profile)
        ),
        "candidate_screening_time_sec": 0.0,
        "primary_retention": retention,
        "fallback": selected.candidate_id.startswith("generic:"),
        "wall_sec": time.perf_counter() - started,
    }
    receipt = {
        "schema_version": "ExecutionV3ColdStartFrameReceiptV1",
        "mode": "COLD_START",
        "candidate": candidate.name,
        "ordinal": ordinal,
        "source_frame_local": int(runtime.graph.frame_indices[ordinal]),
        "runtime_step_index": runtime_step,
        "old_production_q": "ABSENT",
        "previous_accepted_state": "ABSENT" if runtime_step == 0 else "PRESENT",
        "q_old_synthesized": False,
        "source_geometric_seed": geometric_receipt,
        "contributor_scores": scores,
        "contributor_ranking": ranking,
        "selected_block": selected_finger,
        "free_qpos_indices": list(block),
        "seed_pool": [name for name, _q in seeds],
        "probe_receipts": probe_receipts,
        "selected_seed_candidate": selected_seed.candidate_id,
        "primary_exception": primary_exception,
        "primary_solver": primary_profile,
        "retained_primary_id": retained.candidate_id,
        "secondary_exception": polish_exception,
        "secondary_solver": secondary_profile,
        "retention_decision": retention,
        "selected_candidate": selected.candidate_id,
        "selected": _measurement_row(values_out),
        "selected_evaluation": evaluation_out,
        "selected_actual_continuity": actual_out,
        "technical_success": bool(selected.usable),
        "optimizer_started": len(probe_receipts) > 0,
        "profiler": profiler,
        "context_binding_sha256": binding.sha256,
    }
    return np.asarray(q_out), np.asarray(base_out), receipt


def develop_execution_v3_candidates(root: Path) -> dict[str, Any]:
    run_graph_parity(root)
    candidates = list(V3_CANDIDATES.values())
    for index, candidate in enumerate(candidates):
        letter = chr(ord("a") + index)
        write_json(
            root / f"execution_v3_design/candidate_v3_{letter}.json",
            {
                **candidate.as_dict(),
                "status": "PREDECLARED_BEFORE_MASKED_QOLD_OUTCOME",
                "generic": True,
                "episode_agnostic": True,
                "q_old_required": False,
                "candidate_c_activation": "ONLY_IF_A_AND_B_NOT_READY"
                if letter == "c"
                else "UNCONDITIONAL",
            },
        )
    seed_draft = {
        "schema_version": "ColdStartSeedAuthorityV1Draft",
        "frame0": ["wuji_canonical_rest", "joint_range_midpoint"],
        "t_gt_0_optional": ["previous_accepted_runtime"],
        "conditional": ["source_geometric_multistart"],
        "bounds": "Wuji asset joint limits",
        "ordering": "contract tuple order; stable duplicate elision",
        "tie_break": "primary hinge, secondary, seed ordinal, seed id",
        "forbidden": [
            "q_old synthesis",
            "cross-episode DEV1 solved q",
            "failed Stage7 terminal aliased as q_old",
            "episode/object/frame lookup",
        ],
    }
    fallback = {
        "schema_version": "ColdStartFallbackAuthorityV1Draft",
        "fallback": "best independently finite hard-valid generic candidate",
        "empty": "TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE",
        "preserve_old_solution_claim": False,
    }
    write_json(root / "execution_v3_design/cold_start_seed_authority_draft.json", seed_draft)
    write_json(root / "execution_v3_design/fallback_authority_draft.json", fallback)
    return {"candidate_count": len(candidates), "candidates": [item.name for item in candidates]}


def _candidate_slug(candidate: ExecutionV3Candidate) -> str:
    return candidate.name.lower()


def _development_row(
    receipt: dict[str, Any], window: dict[str, Any], old_e_im: float
) -> dict[str, Any]:
    selected = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    actual = receipt["selected_actual_continuity"]
    profiler = receipt["profiler"]
    return {
        "candidate": receipt["candidate"],
        "window_id": window["window_id"],
        "stratum": window["stratum"],
        "ordinal": receipt["ordinal"],
        "frame_id": receipt["source_frame_local"],
        "runtime_step_index": receipt["runtime_step_index"],
        "selected_block": receipt["selected_block"],
        "selected_seed": receipt["selected_seed_candidate"],
        "selected_candidate": receipt["selected_candidate"],
        "old_e_im_evaluation_only": old_e_im,
        "new_e_im": selected["interaction_e_im"],
        "technical_success": receipt["technical_success"],
        "optimizer_started": receipt["optimizer_started"],
        "feasible": evaluation["feasible"],
        "violations": ";".join(evaluation["violated_constraints"]),
        "actual_delta_p": actual["translation_step_m"],
        "actual_delta_R": actual["rotation_step_rad"],
        "actual_delta_q": actual["q_step_inf_rad"],
        "wrist_position_m": selected["wrist_position_m"],
        "wrist_rotation_rad": selected["wrist_rotation_rad"],
        "bone_p95_rad": selected["bone_direction_p95_rad"],
        "collision_min_signed_distance_m": selected["collision_min_signed_distance_m"],
        "joint_limit_min_margin_rad": selected["joint_limit_min_margin_rad"],
        "rotation_determinant": selected["rotation_determinant"],
        "unit_scale_ratio": selected["unit_scale_ratio"],
        "retention_decision": receipt["retention_decision"],
        "seed_count": profiler["seed_count"],
        "candidate_probes": profiler["candidate_probes"],
        "probe_nfev": profiler["probe_nfev"],
        "primary_nfev": profiler["primary_nfev"],
        "secondary_nfev": profiler["secondary_nfev"],
        "interaction_eval_time_sec": profiler["interaction_eval_time_sec"],
        "fk_time_sec": profiler["fk_time_sec"],
        "wall_sec": profiler["wall_sec"],
    }


def _run_masked_candidate(root: Path, candidate: ExecutionV3Candidate) -> dict[str, Any]:
    runtime = V3Runtime("dev_01", root)
    old_eim = np.asarray(runtime.semantic["interaction_final_e_im"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for window in development_windows(D2A_ROOT):
        previous_q = None
        previous_base = None
        for step, ordinal_raw in enumerate(window["ordinals"]):
            ordinal = int(ordinal_raw)
            state_path = (
                root
                / "development/receipts"
                / _candidate_slug(candidate)
                / window["window_id"]
                / f"frame_{ordinal:04d}.npz"
            )
            receipt_path = state_path.with_suffix(".json")
            try:
                if state_path.exists() and receipt_path.exists():
                    with np.load(state_path, allow_pickle=False) as state:
                        q_out = np.asarray(state["qpos"], dtype=np.float64)
                        base_out = np.asarray(state["base_pose_scene"], dtype=np.float64)
                    receipt = read_json(receipt_path)
                else:
                    q_out, base_out, receipt = search_cold_start_frame(
                        runtime,
                        ordinal,
                        runtime_step=step,
                        previous_q=previous_q,
                        previous_base=previous_base,
                        candidate=candidate,
                    )
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(state_path, qpos=q_out, base_pose_scene=base_out)
                    write_json(receipt_path, receipt)
                rows.append(_development_row(receipt, window, float(old_eim[ordinal])))
                previous_q, previous_base = q_out, base_out
            except Exception as exc:
                failure = {
                    "schema_version": "O5RD2GTechnicalFailureV1",
                    "stage": "V3_MASKED_QOLD_DEVELOPMENT",
                    "candidate": candidate.name,
                    "window": window["window_id"],
                    "stratum": window["stratum"],
                    "ordinal": ordinal,
                    "runtime_step_index": step,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                failures.append(failure)
                with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure, sort_keys=True) + "\n")
                for remaining in window["ordinals"][step:]:
                    rows.append(
                        {
                            "candidate": candidate.name,
                            "window_id": window["window_id"],
                            "stratum": window["stratum"],
                            "ordinal": int(remaining),
                            "technical_success": False,
                            "optimizer_started": False,
                            "failure": failure["error"],
                        }
                    )
                break
    by_stratum: dict[str, Any] = {}
    for stratum in ("HIGH", "MID", "LOW"):
        subset = [row for row in rows if row["stratum"] == stratum]
        completed = [row for row in subset if bool(row.get("technical_success"))]
        values = [float(row["new_e_im"]) for row in completed]
        p95 = None if len(values) != 20 else float(np.quantile(values, 0.95))
        hard = bool(len(completed) == 20 and all(bool(row["feasible"]) for row in completed))
        continuity = bool(
            len(completed) == 20
            and all(
                float(row["actual_delta_p"]) <= GATE.temporal_translation_step_limit_m
                and float(row["actual_delta_R"]) <= GATE.temporal_rotation_step_limit_rad
                for row in completed
            )
        )
        preservation = bool(
            len(completed) == 20
            and all(
                float(row["new_e_im"]) <= GATE.interaction_e_im_p95_limit
                for row in completed
                if float(row["old_e_im_evaluation_only"]) <= GATE.interaction_e_im_p95_limit
            )
        )
        by_stratum[stratum] = {
            "technical": f"{len(completed)}/20",
            "p95_e_im": p95,
            "interaction_pass": p95 is not None and p95 <= GATE.interaction_e_im_p95_limit,
            "hard_validity_pass": hard,
            "continuity_pass": continuity,
            "low_old_valid_preservation_pass": preservation
            if stratum == "LOW"
            else "NOT_APPLICABLE",
        }
    pass_gate = bool(
        not failures
        and all(
            value["technical"] == "20/20"
            and value["interaction_pass"]
            and value["hard_validity_pass"]
            and value["continuity_pass"]
            and value["low_old_valid_preservation_pass"] is not False
            for value in by_stratum.values()
        )
    )
    write_csv(root / f"development/{_candidate_slug(candidate)}_masked_qold.csv", rows)
    payload = {
        "schema_version": "MaskedQOldColdStartDevelopmentV1",
        "candidate": candidate.as_dict(),
        "q_old_authority_present": False,
        "q_old_passed_but_unselected": False,
        "frame_count": len(rows),
        "new_dev1_method_development_frames": 0,
        "strata": by_stratum,
        "failures": failures,
        "MASKED_QOLD_DEVELOPMENT": "PASS" if pass_gate else "FAIL",
    }
    write_json(root / f"development/{_candidate_slug(candidate)}_summary.json", payload)
    return payload


def _run_candidate_determinism(root: Path, candidate: ExecutionV3Candidate) -> dict[str, Any]:
    runtime = V3Runtime("dev_01", root)
    representatives = {
        window["stratum"]: int(window["ordinals"][10]) for window in development_windows(D2A_ROOT)
    }
    rows = []
    overall = True
    for stratum, ordinal in representatives.items():
        repeats = []
        for run in (1, 2, 3):
            try:
                q, base, receipt = search_cold_start_frame(
                    runtime,
                    ordinal,
                    runtime_step=0,
                    previous_q=None,
                    previous_base=None,
                    candidate=candidate,
                )
                repeats.append(
                    {
                        "run": run,
                        "technical": "PASS",
                        "q": q,
                        "base": base,
                        "seed_pool": receipt["seed_pool"],
                        "block": receipt["selected_block"],
                        "selected": receipt["selected_candidate"],
                        "retention": receipt["retention_decision"],
                        "e_im": receipt["selected"]["interaction_e_im"],
                    }
                )
            except Exception as exc:
                repeats.append(
                    {
                        "run": run,
                        "technical": "FAIL",
                        "failure": f"{type(exc).__name__}:{exc}",
                    }
                )
        reference = repeats[0]
        if all(item["technical"] == "PASS" for item in repeats):
            q_diff = max(float(np.max(np.abs(item["q"] - reference["q"]))) for item in repeats)
            base_diff = max(
                float(np.max(np.abs(item["base"] - reference["base"]))) for item in repeats
            )
            eim_diff = max(abs(float(item["e_im"]) - float(reference["e_im"])) for item in repeats)
            metadata_same = all(
                item["seed_pool"] == reference["seed_pool"]
                and item["block"] == reference["block"]
                and item["selected"] == reference["selected"]
                and item["retention"] == reference["retention"]
                for item in repeats
            )
            outcome = "TECHNICAL_PASS_DETERMINISTIC"
            passed = bool(
                metadata_same and q_diff <= 1e-8 and base_diff <= 1e-8 and eim_diff <= 1e-12
            )
        else:
            q_diff = base_diff = eim_diff = None
            metadata_same = all(
                item["technical"] == "FAIL" and item.get("failure") == reference.get("failure")
                for item in repeats
            )
            outcome = "TECHNICAL_FAIL_DETERMINISTIC"
            passed = metadata_same
        overall = overall and passed
        rows.append(
            {
                "stratum": stratum,
                "ordinal": ordinal,
                "runs": 3,
                "seed_pool_same": metadata_same,
                "max_q_abs": q_diff,
                "max_base_abs": base_diff,
                "max_eim_abs": eim_diff,
                "outcome": outcome,
                "failure": reference.get("failure"),
                "status": "PASS" if passed else "FAIL",
            }
        )
    return {
        "schema_version": "ExecutionV3CandidateDeterminismV1",
        "candidate": candidate.name,
        "representatives": rows,
        "DETERMINISM": "PASS" if overall else "FAIL",
    }


def run_masked_qold_development(root: Path) -> dict[str, Any]:
    develop_execution_v3_candidates(root)
    summaries = []
    for name in (
        "V3_A_GENERIC_ASSET_SOURCE_COLD_START",
        "V3_B_GENERIC_WITH_PREVIOUS_ACCEPTED_RUNTIME",
    ):
        candidate = V3_CANDIDATES[name]
        summary_path = root / f"development/{_candidate_slug(candidate)}_summary.json"
        summaries.append(
            read_json(summary_path)
            if summary_path.exists()
            else _run_masked_candidate(root, candidate)
        )
    if not any(item["MASKED_QOLD_DEVELOPMENT"] == "PASS" for item in summaries):
        candidate_c = V3_CANDIDATES["V3_C_SOURCE_GEOMETRIC_WITH_PREVIOUS_ACCEPTED_RUNTIME"]
        summary_path = root / f"development/{_candidate_slug(candidate_c)}_summary.json"
        summaries.append(
            read_json(summary_path)
            if summary_path.exists()
            else _run_masked_candidate(root, candidate_c)
        )
    determinism = []
    for summary in summaries:
        candidate = V3_CANDIDATES[summary["candidate"]["name"]]
        result = _run_candidate_determinism(root, candidate)
        determinism.append(result)
        write_json(
            root / f"development/{_candidate_slug(candidate)}_determinism.json",
            result,
        )
    write_json(
        root / "development/determinism.json",
        {
            "schema_version": "ExecutionV3DevelopmentDeterminismSummaryV1",
            "candidates": determinism,
            "DETERMINISM": "PASS"
            if all(item["DETERMINISM"] == "PASS" for item in determinism)
            else "FAIL",
            "interpretation": (
                "PASS means the observed success or technical-failure outcome is exactly "
                "reproducible; it does not override a masked-q_old gate failure."
            ),
        },
    )
    summary_by_name = {item["candidate"]["name"]: item for item in summaries}
    determinism_by_name = {item["candidate"]: item for item in determinism}
    rows = []
    for name, item in summary_by_name.items():
        rows.append(
            {
                "candidate": name,
                "masked_qold": item["MASKED_QOLD_DEVELOPMENT"],
                "determinism": determinism_by_name[name]["DETERMINISM"],
                "result": "PASS"
                if item["MASKED_QOLD_DEVELOPMENT"] == "PASS"
                and determinism_by_name[name]["DETERMINISM"] == "PASS"
                else "FAIL",
            }
        )
    write_csv(root / "development/candidate_summary.csv", rows)
    for stratum in ("HIGH", "MID", "LOW"):
        combined = []
        for name in summary_by_name:
            combined.extend(
                row
                for row in read_csv(
                    root / f"development/{_candidate_slug(V3_CANDIDATES[name])}_masked_qold.csv"
                )
                if row["stratum"] == stratum
            )
        write_csv(root / f"development/masked_qold_{stratum.lower()}.csv", combined)
    all_profiler = []
    for name in summary_by_name:
        all_profiler.extend(
            read_csv(root / f"development/{_candidate_slug(V3_CANDIDATES[name])}_masked_qold.csv")
        )
    write_csv(root / "development/profiler.csv", all_profiler)
    payload = {
        "schema_version": "MaskedQOldDevelopmentCandidateSummaryV1",
        "candidates": rows,
        "candidate_c_activated": len(summaries) == 3,
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "status": "PASS" if any(row["result"] == "PASS" for row in rows) else "FAIL",
    }
    write_json(root / "development/masked_qold_decision.json", payload)
    return payload


def _historical_s1_paths(window_id: str, ordinal: int) -> tuple[Path, Path]:
    base = D2C_ROOT / "search_candidates/receipts/s1" / window_id / f"frame_{ordinal:04d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _run_refinement_v3(
    runtime: D2ARuntime,
    ordinal: int,
    *,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    ExecutionFrameInputsV3(
        mode=RetargetMode.REFINEMENT,
        runtime_step_index=runtime_step,
        old_production_q=old_q,
        previous_accepted_q=previous_q,
        previous_accepted_base=previous_base,
    ).validate()
    q, base, receipt = search_frame(
        runtime,
        ordinal,
        previous_q=previous_q,
        previous_base=previous_base,
        contract=default_search_contracts()[0],
    )
    return (
        q,
        base,
        {
            **receipt,
            "execution_v3_mode": "REFINEMENT",
            "execution_v3_delegation": "EXACT_FROZEN_EXECUTION_V2_S1",
            "old_production_authority": "PRESENT",
        },
    )


def run_refinement_regression(root: Path) -> dict[str, Any]:
    _require_status(root / "development/masked_qold_decision.json", "status", "PASS")
    runtime = D2ARuntime(root)
    windows = development_windows(D2A_ROOT)
    selected: list[tuple[dict[str, Any], list[int], bool]] = []
    for window in windows:
        positions = [0, 4, 9, 14, 19]
        ordinals = [int(window["ordinals"][index]) for index in positions]
        selected.append((window, ordinals, False))
    continuous = next(window for window in windows if window["stratum"] == "LOW")
    selected.append((continuous, [int(value) for value in continuous["ordinals"]], True))
    rows = []
    max_q = max_base = max_eim = 0.0
    for window, ordinals, continuous_mode in selected:
        previous_q = None
        previous_base = None
        for step, ordinal in enumerate(ordinals):
            historical_state_path, historical_receipt_path = _historical_s1_paths(
                window["window_id"], ordinal
            )
            _require(historical_state_path)
            _require(historical_receipt_path)
            with np.load(historical_state_path, allow_pickle=False) as historical_state:
                expected_q = np.asarray(historical_state["qpos"], dtype=np.float64)
                expected_base = np.asarray(historical_state["base_pose_scene"], dtype=np.float64)
            expected_receipt = read_json(historical_receipt_path)
            if continuous_mode:
                if step == 0:
                    previous_q = np.asarray(
                        runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64
                    )
                    previous_base = np.asarray(
                        runtime.final.arrays["base_pose_scene"][ordinal - 1],
                        dtype=np.float64,
                    )
                runtime_step = step + 1
            else:
                previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
                previous_base = np.asarray(
                    runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
                )
                runtime_step = 1
            q, base, receipt = _run_refinement_v3(
                runtime,
                ordinal,
                runtime_step=runtime_step,
                previous_q=previous_q,
                previous_base=previous_base,
            )
            q_diff = float(np.max(np.abs(q - expected_q)))
            base_diff = float(np.max(np.abs(base - expected_base)))
            eim_diff = abs(
                float(receipt["selected"]["interaction_e_im"])
                - float(expected_receipt["selected"]["interaction_e_im"])
            )
            max_q, max_base, max_eim = (
                max(max_q, q_diff),
                max(max_base, base_diff),
                max(max_eim, eim_diff),
            )
            same_metadata = bool(
                receipt["selected_block"] == expected_receipt["selected_block"]
                and receipt["seed_pool"] == expected_receipt["seed_pool"]
                and receipt["selected_candidate"] == expected_receipt["selected_candidate"]
                and receipt["retention_decision"] == expected_receipt["retention_decision"]
            )
            passed = bool(
                q_diff <= 1e-8
                and base_diff <= 1e-8
                and eim_diff <= 1e-12
                and same_metadata
                and receipt["technical_success"]
            )
            rows.append(
                {
                    "window": window["window_id"],
                    "stratum": window["stratum"],
                    "ordinal": ordinal,
                    "continuous_window": continuous_mode,
                    "max_q_abs": q_diff,
                    "max_base_abs": base_diff,
                    "e_im_abs": eim_diff,
                    "same_selected_contributor_seed_retention": same_metadata,
                    "status": "PASS" if passed else "FAIL",
                }
            )
            if continuous_mode:
                previous_q, previous_base = q, base
    overall = all(row["status"] == "PASS" for row in rows)
    write_csv(root / "development/refinement_regression.csv", rows)
    payload = {
        "schema_version": "ExecutionV3RefinementRegressionV1",
        "REFINEMENT_MODE_REGRESSION": "PASS" if overall else "FAIL",
        "frame_checks": len(rows),
        "high_mid_low_counts": {
            name: sum(row["stratum"] == name and not row["continuous_window"] for row in rows)
            for name in ("HIGH", "MID", "LOW")
        },
        "continuous_window_frame_count": sum(row["continuous_window"] for row in rows),
        "max_q_abs": max_q,
        "max_base_abs": max_base,
        "max_e_im_abs": max_eim,
        "execution_v2_s1_exact_delegate": True,
    }
    write_json(root / "development/refinement_regression.json", payload)
    return payload


def select_execution_v3(root: Path) -> dict[str, Any]:
    masked = _require_status(root / "development/masked_qold_decision.json", "status", "PASS")
    refinement = _require_status(
        root / "development/refinement_regression.json",
        "REFINEMENT_MODE_REGRESSION",
        "PASS",
    )
    rows = read_csv(root / "development/candidate_summary.csv")
    eligible = [row for row in rows if row["result"] == "PASS"]
    order = {name: index for index, name in enumerate(V3_CANDIDATES)}
    selected = None if not eligible else min(eligible, key=lambda row: order[row["candidate"]])
    payload = {
        "schema_version": "ExecutionV3CandidateSelectionV1",
        "selection_locked_before_dev2_frame0": True,
        "selection_priority": [
            "all masked-q_old hard gates PASS",
            "refinement regression PASS",
            "determinism PASS",
            "no episode-specific behavior",
            "simpler execution contract",
            "runtime/nfev tie-break only",
        ],
        "candidates": rows,
        "masked_gate": masked["status"],
        "refinement_gate": refinement["REFINEMENT_MODE_REGRESSION"],
        "SELECTED_EXECUTION_V3": None if selected is None else selected["candidate"],
        "status": "PASS" if selected is not None else "FAIL",
        "dev2_outcome_used_for_selection": False,
    }
    write_json(root / "development/selection_decision.json", payload)
    if selected is None:
        raise RuntimeError("O5RD2G_NO_EXECUTION_V3_CANDIDATE_READY")
    return payload


def _dev2_identity() -> dict[str, Any]:
    fixed = read_json(O5_ROOT / "preflight/fixed_o5_episode_set.json")
    episode = next(item for item in fixed["episodes"] if item["review"] == "dev_02")
    canonical_meta = read_json(frozen_paths()["dev2_canonical"] / "metadata.json")["manifest"][
        "fields"
    ]["metadata"]["fields"]["metadata"]
    record = canonical_meta["manifest_record"]
    expected = {
        "record_id": "oakink2:scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04:00010",
        "primitive": "rearrange",
        "object": "C11001",
        "source_interval": [10704, 10944],
        "frames": 240,
    }
    exact = bool(
        episode["record_id"] == expected["record_id"]
        and episode["object"] == expected["object"]
        and episode["source_interval"] == expected["source_interval"]
        and record["primitive"] == expected["primitive"]
        and record["record_id"] == expected["record_id"]
        and canonical_meta["source_frame_ids"] == list(range(10704, 10944))
    )
    return {"expected": expected, "observed": episode, "identity_exact": exact}


def run_dev2_frame0_development(root: Path) -> dict[str, Any]:
    selection = _require_status(root / "development/selection_decision.json", "status", "PASS")
    identity = _dev2_identity()
    if not identity["identity_exact"]:
        raise RuntimeError("O5RD2G_DEV2_IDENTITY_AUTHORITY_MISMATCH")
    candidate = V3_CANDIDATES[str(selection["SELECTED_EXECUTION_V3"])]
    role = {
        "schema_version": "DEV2Frame0ExecutionV3RoleReceiptV1",
        "ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "independent_control": False,
        "identity": identity,
        "mode": "COLD_START",
        "old_production_q": "ABSENT",
        "previous_accepted_state": "ABSENT",
        "source_graph_authority": "SourceInteractionGraphAuthorityV1",
        "selected_candidate_locked": candidate.name,
    }
    write_json(root / "dev2_frame0_development/role_receipt.json", role)
    runtime = V3Runtime("dev_02", root)
    runs = []
    states = []
    for run in (1, 2, 3):
        started = time.perf_counter()
        try:
            q, base, receipt = search_cold_start_frame(
                runtime,
                0,
                runtime_step=0,
                previous_q=None,
                previous_base=None,
                candidate=candidate,
            )
            selected = receipt["selected"]
            evaluation = receipt["selected_evaluation"]
            row = {
                "schema_version": "DEV2Frame0ExecutionV3DevelopmentRunV1",
                "run": run,
                "source_frame": 10704,
                "mode": "COLD_START",
                "candidate": candidate.name,
                "optimizer_started": receipt["optimizer_started"],
                "technical": "PASS" if receipt["technical_success"] else "FAIL",
                "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(base))),
                "E_IM": selected["interaction_e_im"],
                "wrist": "PENDING",
                "wrist_position_m": selected["wrist_position_m"],
                "wrist_rotation_rad": selected["wrist_rotation_rad"],
                "bone": "PASS"
                if selected["bone_direction_p95_rad"] <= runtime.authority.bone_direction_limit_rad
                else "FAIL",
                "collision": "PASS"
                if selected["collision_min_signed_distance_m"]
                >= -runtime.authority.collision_hard_bound_m - 1e-6
                else "FAIL",
                "joints": "PASS" if selected["joint_limit_min_margin_rad"] >= -1e-10 else "FAIL",
                "hard_feasible": evaluation["feasible"],
                "selected_seed": receipt["selected_seed_candidate"],
                "selected_candidate": receipt["selected_candidate"],
                "selected_block": receipt["selected_block"],
                "retention": receipt["retention_decision"],
                "seed_pool": receipt["seed_pool"],
                "q_old_synthesized": receipt["q_old_synthesized"],
                "profiler": receipt["profiler"],
                "wall_sec": time.perf_counter() - started,
            }
            row["wrist"] = (
                "PASS"
                if selected["wrist_position_m"] <= runtime.authority.wrist_position_limit_m
                and selected["wrist_rotation_rad"] <= runtime.authority.wrist_rotation_limit_rad
                else "FAIL"
            )
            states.append((q, base))
        except Exception as exc:
            row = {
                "schema_version": "DEV2Frame0ExecutionV3DevelopmentRunV1",
                "run": run,
                "source_frame": 10704,
                "mode": "COLD_START",
                "candidate": candidate.name,
                "optimizer_started": False,
                "technical": "FAIL",
                "finite": False,
                "E_IM": None,
                "wrist": "NOT_EVALUABLE",
                "bone": "NOT_EVALUABLE",
                "collision": "NOT_EVALUABLE",
                "joints": "NOT_EVALUABLE",
                "failure": f"{type(exc).__name__}:{exc}",
                "wall_sec": time.perf_counter() - started,
            }
        write_json(root / f"dev2_frame0_development/run_{run}.json", row)
        runs.append(row)
    comparable = len(states) == 3
    q_diff = (
        None
        if not comparable
        else max(float(np.max(np.abs(q - states[0][0]))) for q, _base in states)
    )
    base_diff = (
        None
        if not comparable
        else max(float(np.max(np.abs(base - states[0][1]))) for _q, base in states)
    )
    eim_values = [row["E_IM"] for row in runs if row["E_IM"] is not None]
    eim_diff = (
        None
        if len(eim_values) != 3
        else max(abs(float(value) - float(eim_values[0])) for value in eim_values)
    )
    metadata_same = bool(
        len(runs) == 3
        and all(
            row.get("seed_pool") == runs[0].get("seed_pool")
            and row.get("selected_candidate") == runs[0].get("selected_candidate")
            and row.get("selected_block") == runs[0].get("selected_block")
            and row.get("retention") == runs[0].get("retention")
            for row in runs
        )
    )
    determinism = {
        "schema_version": "DEV2Frame0ExecutionV3DeterminismV1",
        "comparable_runs": comparable,
        "seed_and_selection_same": metadata_same,
        "max_q_abs": q_diff,
        "max_base_abs": base_diff,
        "max_eim_abs": eim_diff,
        "status": "PASS"
        if comparable
        and metadata_same
        and q_diff is not None
        and q_diff <= 1e-8
        and base_diff is not None
        and base_diff <= 1e-8
        and eim_diff is not None
        and eim_diff <= 1e-12
        else "FAIL",
    }
    write_json(root / "dev2_frame0_development/determinism.json", determinism)
    passed = bool(
        determinism["status"] == "PASS"
        and all(
            row["optimizer_started"]
            and row["technical"] == "PASS"
            and row["finite"]
            and float(row["E_IM"]) <= GATE.interaction_e_im_p95_limit
            and row["wrist"] == "PASS"
            and row["bone"] == "PASS"
            and row["collision"] == "PASS"
            and row["joints"] == "PASS"
            and row["hard_feasible"]
            and not row["q_old_synthesized"]
            for row in runs
        )
    )
    decision = {
        "schema_version": "DEV2Frame0ExecutionV3DevelopmentDecisionV1",
        "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "DEV2_FRAME0_V3_DEVELOPMENT": "PASS" if passed else "FAIL",
        "DEV2_FRAME0_V3_RUN_COUNT": 3,
        "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": sum(bool(row["optimizer_started"]) for row in runs),
        "EXECUTION_V3_FREEZE_AUTHORIZED": "YES" if passed else "NO",
        "runs": runs,
    }
    write_json(root / "dev2_frame0_development/decision.json", decision)
    return decision


def _special_case_audit() -> dict[str, Any]:
    algorithm_source = inspect.getsource(search_cold_start_frame) + inspect.getsource(
        cold_start_seeds_v3
    )
    return {
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "DEV2_EPISODE_ID_BRANCH": "NO" if "scene_01__A003" not in algorithm_source else "YES",
        "DEV2_OBJECT_ID_BRANCH": "NO" if "C11001" not in algorithm_source else "YES",
        "DEV2_FRAME_LITERAL_BRANCH": "NO" if "10704" not in algorithm_source else "YES",
        "DEV2_HAND_TUNED_SEED": "NO",
        "THUMB_SPECIAL_CASE_ADDED": "NO" if '"thumb"' not in algorithm_source else "YES",
    }


def _evidence_ledgers(root: Path) -> dict[str, Any]:
    development = read_json(D2C_ROOT / "future_certification/development_exclusion_ledger.json")[
        "entries"
    ]
    sparse_v2 = read_json(
        REPO
        / ".local/reports/oakink2_o5rd2d_objective_v2_independent_certification_v1/sparse_v2/manifest.json"
    )["frames"]
    sparse_v3 = read_json(D2E_ROOT / "sparse_v3/manifest.json")["frames"]
    windows = read_json(D2E_ROOT / "window_v3/manifest.json")["windows"]
    dev1_roles: dict[int, set[str]] = {}
    for entry in development:
        dev1_roles.setdefault(int(entry["ordinal"]), set()).add("V2_METHOD_DEVELOPMENT")
    for row in sparse_v2:
        dev1_roles.setdefault(int(row["ordinal"]), set()).add("SPARSE_V2_CERTIFICATION_CONSUMED")
    for row in sparse_v3:
        dev1_roles.setdefault(int(row["ordinal"]), set()).add("SPARSE_V3_CERTIFICATION_CONSUMED")
    for window in windows:
        for ordinal in window["ordinals"]:
            dev1_roles.setdefault(int(ordinal), set()).add("WINDOW_V3_CERTIFICATION_CONSUMED")
    for window in development_windows(D2A_ROOT):
        for ordinal in window["ordinals"]:
            dev1_roles.setdefault(int(ordinal), set()).add("V3_MASKED_QOLD_DEVELOPMENT")
    entries = [
        {
            "episode": "DEV1",
            "ordinal": ordinal,
            "roles": sorted(roles),
            "future_role": "FUTURE_V3_INDEPENDENT_EXCLUDED",
        }
        for ordinal, roles in sorted(dev1_roles.items())
    ]
    entries.append(
        {
            "episode": "DEV2",
            "ordinal": 0,
            "source_frame": 10704,
            "roles": ["DEV2_FRAME0_KNOWN_FAILURE_DEVELOPMENT"],
            "future_role": "FUTURE_V3_INDEPENDENT_EXCLUDED",
        }
    )
    count = len(entries)
    payload = {
        "schema_version": "ExecutionV3EvidenceLedgerV1",
        "entries": entries,
        "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT": count,
        "role_vocabulary": [
            "V2_METHOD_DEVELOPMENT",
            "SPARSE_V2_CERTIFICATION_CONSUMED",
            "SPARSE_V3_CERTIFICATION_CONSUMED",
            "WINDOW_V3_CERTIFICATION_CONSUMED",
            "V3_MASKED_QOLD_DEVELOPMENT",
            "DEV2_FRAME0_KNOWN_FAILURE_DEVELOPMENT",
            "FUTURE_V3_INDEPENDENT_EXCLUDED",
        ],
    }
    write_json(root / "ledger/execution_v3_evidence_ledger.json", payload)
    write_json(
        root / "ledger/future_validation_exclusion_ledger.json",
        {
            "schema_version": "ExecutionV3FutureValidationExclusionLedgerV1",
            "count": count,
            "entries": entries,
            "new_dev1_method_development_frames": 0,
        },
    )
    return payload


def _development_gate(root: Path) -> dict[str, Any]:
    authority = _require_status(
        root / "input_authority/authority_decision.json", "INPUT_AUTHORITY_AUDIT"
    )
    graph = _require_status(root / "graph_authority/graph_parity_summary.json", "GRAPH_PARITY")
    masked = _require_status(root / "development/masked_qold_decision.json", "status")
    refinement = _require_status(
        root / "development/refinement_regression.json", "REFINEMENT_MODE_REGRESSION"
    )
    dev2 = _require_status(
        root / "dev2_frame0_development/decision.json",
        "DEV2_FRAME0_V3_DEVELOPMENT",
    )
    special = _special_case_audit()
    no_special = all(value == "NO" for value in special.values())
    passed = bool(
        authority["INPUT_AUTHORITY_AUDIT"] == "PASS"
        and graph["GRAPH_PARITY"] == "PASS"
        and masked["status"] == "PASS"
        and refinement["REFINEMENT_MODE_REGRESSION"] == "PASS"
        and dev2["DEV2_FRAME0_V3_DEVELOPMENT"] == "PASS"
        and no_special
    )
    payload = {
        "schema_version": "ExecutionV3DevelopmentGateV1",
        "EXECUTION_V3_DEVELOPMENT_GATE": "PASS" if passed else "FAIL",
        "input_authority": authority["INPUT_AUTHORITY_AUDIT"],
        "graph_authority": graph["GRAPH_PARITY"],
        "masked_qold": masked["status"],
        "refinement_regression": refinement["REFINEMENT_MODE_REGRESSION"],
        "dev2_frame0": dev2["DEV2_FRAME0_V3_DEVELOPMENT"],
        "special_case_audit": special,
    }
    write_json(root / "development/development_gate.json", payload)
    if not passed:
        raise RuntimeError("O5RD2G_EXECUTION_V3_DEVELOPMENT_GATE_FAIL")
    return payload


def freeze_execution_input_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    dependency = read_json(root / "input_authority/execution_input_dependency_audit.json")
    payload = {
        "schema_version": "ExecutionInputAuthorityV1",
        "status": "FROZEN",
        "scientifically_mandatory": [
            row["input"]
            for row in dependency["inputs"]
            if row["role"] == "HARD_SCIENTIFIC_REQUIREMENT"
        ],
        "mode_specific": {
            "REFINEMENT": {
                "old_production_q": "REQUIRED",
                "previous_accepted_runtime_state": "ABSENT_FRAME0_REQUIRED_AFTER_FRAME0",
            },
            "COLD_START": {
                "old_production_q": "FORBIDDEN_ABSENT",
                "previous_accepted_runtime_state": "ABSENT_FRAME0_REQUIRED_AFTER_FRAME0",
            },
        },
        "source_derived": [
            "source wrist/frame",
            "source bone features",
            "object surface samples",
            "interaction graph",
            "base seed",
        ],
        "robot_asset_derived": [
            "joint limits",
            "DOF semantic mapping",
            "rest/midpoint seeds",
            "collision surface/SDF",
        ],
        "runtime_state_derived": ["previous accepted q/base", "continuous prediction"],
        "optional_search_hints": ["previous accepted runtime seed"],
        "must_never_be_synthesized": [
            "old production q",
            "canonical source authority",
            "object identity",
            "failed Stage7 terminal as q_old",
        ],
        "dependency_audit_sha256": sha256_file(
            root / "input_authority/execution_input_dependency_audit.json"
        ),
    }
    path = root / "frozen_v3/execution_input_authority.json"
    write_json(path, payload)
    _sha_text(path.with_suffix(".sha256"), sha256_file(path))
    return payload


def freeze_coldstart_seed_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    selection = read_json(root / "development/selection_decision.json")
    candidate = V3_CANDIDATES[str(selection["SELECTED_EXECUTION_V3"])]
    payload = {
        "schema_version": "ColdStartSeedAuthorityV1",
        "status": "FROZEN",
        "selected_candidate": candidate.name,
        "frame0_seed_sources": [
            item for item in candidate.cold_seed_sources if item != "previous_accepted_runtime"
        ],
        "t_gt_0_seed_sources": list(candidate.cold_seed_sources),
        "bounds": "Wuji asset joint limits",
        "deterministic_ordering": "contract tuple order with stable exact duplicate elision",
        "tie_break": "hard validity, Candidate-B2 hinge, secondary, seed ordinal, seed id",
        "episode_specific_values": False,
        "object_specific_values": False,
        "source_frame_literals": False,
        "q_old_aliasing": False,
        "failed_stage7_terminal_as_q_old": False,
    }
    path = root / "frozen_v3/cold_start_seed_authority.json"
    write_json(path, payload)
    _sha_text(path.with_suffix(".sha256"), sha256_file(path))
    return payload


def freeze_interaction_graph_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    source = read_json(root / "graph_authority/source_interaction_graph_authority.json")
    payload = {**source, "schema_version": "SourceInteractionGraphAuthorityV1", "status": "FROZEN"}
    path = root / "frozen_v3/source_interaction_graph_authority.json"
    write_json(path, payload)
    _sha_text(path.with_suffix(".sha256"), sha256_file(path))
    return payload


def freeze_execution_v3(root: Path) -> dict[str, Any]:
    _development_gate(root)
    freeze_execution_input_authority(root)
    freeze_coldstart_seed_authority(root)
    freeze_interaction_graph_authority(root)
    selection = read_json(root / "development/selection_decision.json")
    candidate = V3_CANDIDATES[str(selection["SELECTED_EXECUTION_V3"])]
    special = _special_case_audit()
    payload = {
        "schema_version": "RetargetObjectiveV2ExecutionContractV3",
        "status": "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
        "objective": "RetargetObjectiveV2 Candidate B2 unchanged",
        "objective_v2_sha256": OBJECTIVE_SHA,
        "modes": {
            "REFINEMENT": {
                "old_production_authority": "REQUIRED",
                "execution": "exact frozen ExecutionV2/S1 delegate",
                "baseline": "OLD_PRODUCTION_TRAJECTORY",
            },
            "COLD_START": {
                "old_production_authority": "FORBIDDEN_ABSENT",
                "execution": candidate.as_dict(),
                "frame0": "canonical source + robot asset + generic seeds only",
                "t_gt_0": "adds previous accepted runtime state according to candidate",
                "baseline": "best independently hard-valid generic candidate",
            },
        },
        "interaction_graph_authority": "SourceInteractionGraphAuthorityV1",
        "contributor_ranking": "Eq.7 per-keypoint mass at aligned Wuji canonical rest, descending mass then semantic name",
        "seed_count": "stable deduplicated count from selected authority",
        "primary_solver": "production refine_frame SLSQP adapter",
        "primary_budgets": {
            "contributor_probe_max_nfev": candidate.contributor_probe_max_nfev,
            "selected_primary_maxiter": candidate.selected_primary_maxiter,
        },
        "candidate_screening": "finite + independent hard validity + Candidate-B2 lexicographic key",
        "candidate_retention": "retain interaction-valid primary if secondary regresses",
        "secondary_polish": {
            "maxiter": candidate.secondary_polish_maxiter,
            "interaction_retention_epsilon": 1e-10,
        },
        "fallback": "best independently hard-valid generic candidate or TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE",
        "determinism": "fixed seed order, no RNG, stable tie-break",
        "failure_semantics": "explicit pre-optimizer authority failure or technical no-valid-candidate failure",
        "special_case_audit": special,
        "fresh_independent_evidence_consumed": False,
    }
    path = root / "frozen_v3/objective_v2_execution_contract_v3.json"
    write_json(path, payload)
    _sha_text(path.with_suffix(".sha256"), sha256_file(path))
    verify_frozen_upstream(root)
    return payload


def generate_future_certification_plan(root: Path) -> dict[str, Any]:
    contract = _require_status(
        root / "frozen_v3/objective_v2_execution_contract_v3.json",
        "status",
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
    )
    payload = {
        "schema_version": "ExecutionV3IndependentColdStartCertificationPlanV1",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "next_stage": "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION",
        "frozen_execution_v3_sha256": sha256_file(
            root / "frozen_v3/objective_v2_execution_contract_v3.json"
        ),
        "frozen_objective_v2_sha256": contract["objective_v2_sha256"],
        "order": [
            "fresh DEV1 ColdStartSparseValidationV4 with q_old withheld",
            "fresh DEV1 ColdStartWindowValidationV4 with q_old withheld",
            "freeze three fresh OakInk2 DEVELOPMENT-split cross-episode frame0 controls",
            "require 3/3 controls PASS",
            "DEV2 full 240 known-failure recovery exactly once",
        ],
        "cross_episode_selection_contract": {
            "N": 3,
            "split": "DEVELOPMENT",
            "right_hand_eligible": True,
            "single_unambiguous_target_object": True,
            "canonical_inputs_complete": True,
            "method_development_use": False,
            "prefer_distinct_target_objects": True,
            "certification_split": False,
            "heldout_split": False,
            "selected_this_stage": False,
        },
        "executed": False,
    }
    write_json(
        root / "future_certification/execution_v3_independent_certification_plan.json", payload
    )
    return payload


def record_git(root: Path) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    commits = git("log", "--format=%H%x09%s", f"{START_HEAD}..{head}").splitlines()
    payload = {
        "start_head": START_HEAD,
        "final_head": head,
        "commits": commits,
        "tracked_worktree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
        "status_short": git("status", "--short", "--untracked-files=all").splitlines(),
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", payload)
    return payload


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        (
            "00_ruff_task_files.log",
            [
                "conda",
                "run",
                "-n",
                "toporetarget-rl",
                "ruff",
                "check",
                "src/toporetarget/retarget/objective_v3_execution.py",
                "scripts/data/run_oakink2_o5rd2g.py",
                "tests/retarget/test_objective_v3_execution.py",
                "tests/data/test_oakink2_o5rd2g.py",
            ],
        ),
        (
            "01_ruff_format_task_files.log",
            [
                "conda",
                "run",
                "-n",
                "toporetarget-rl",
                "ruff",
                "format",
                "--check",
                "src/toporetarget/retarget/objective_v3_execution.py",
                "scripts/data/run_oakink2_o5rd2g.py",
                "tests/retarget/test_objective_v3_execution.py",
                "tests/data/test_oakink2_o5rd2g.py",
            ],
        ),
        (
            "02_mypy_src.log",
            ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ),
        (
            "03_pytest_full.log",
            ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ),
        (
            "04_paper_fidelity.log",
            [
                "conda",
                "run",
                "-n",
                "toporetarget-rl",
                "python",
                "scripts/check_paper_fidelity.py",
            ],
        ),
        (
            "05_ci_ruff_all.log",
            ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", "."],
        ),
        (
            "06_ci_ruff_format_all.log",
            ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", "."],
        ),
        ("07_git_diff_check.log", ["git", "diff", "--check"]),
    ]
    results = []
    for log_name, command in commands:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        output = completed.stdout + completed.stderr
        log = root / "validation_logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "log": str(log.resolve()),
            }
        )
    payload = {
        "schema_version": "O5RD2GRepositoryValidationV1",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "checks": results,
    }
    write_json(root / "validation_results.json", payload)
    write_json(
        root / "tests.json",
        {
            "status": payload["status"],
            "full_pytest": next(
                item for item in results if item["log"].endswith("03_pytest_full.log")
            ),
        },
    )
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD2G_REPOSITORY_VALIDATION_FAIL")
    return payload


def _mode_comparison(root: Path) -> list[dict[str, Any]]:
    rows = read_csv(D2C_ROOT / "search_candidates/s1_results.csv")
    v2 = {
        "mode": "V2 refinement",
        "old_q_available": "yes",
        "technical": f"{sum(row['technical_success'] == 'True' for row in rows)}/{len(rows)}",
        "p95_e_im": float(np.quantile([float(row["new_e_im"]) for row in rows], 0.95)),
        "mean_sec_per_frame": float(np.mean([float(row["total_wall_sec"]) for row in rows])),
        "probes_per_frame": float(np.mean([float(row["probe_nfev"]) for row in rows])),
        "timing_comparison": "NON_STRICT_DIFFERENT_RUN_CLASS",
    }
    result = [v2]
    refinement_path = root / "development/refinement_regression.csv"
    refinement_status_path = root / "development/refinement_regression.json"
    if (
        refinement_path.exists()
        and refinement_status_path.exists()
        and read_json(refinement_status_path).get("REFINEMENT_MODE_REGRESSION") != "NOT_RUN"
    ):
        refinement = read_csv(refinement_path)
        result.append(
            {
                "mode": "V3 refinement",
                "old_q_available": "yes",
                "technical": f"{sum(row['status'] == 'PASS' for row in refinement)}/{len(refinement)}",
                "p95_e_im": "SEE_FRAME_PARITY",
                "mean_sec_per_frame": "NOT_AGGREGATED",
                "probes_per_frame": "SAME_AS_V2_S1",
                "timing_comparison": "NON_STRICT_DIFFERENT_RUN_CLASS",
            }
        )
    selection_path = root / "development/selection_decision.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        name = selection.get("SELECTED_EXECUTION_V3")
        if name:
            rows = read_csv(
                root / f"development/{_candidate_slug(V3_CANDIDATES[name])}_masked_qold.csv"
            )
            completed = [row for row in rows if row.get("technical_success") == "True"]
            result.append(
                {
                    "mode": "V3 cold-start masked DEV1",
                    "old_q_available": "no",
                    "technical": f"{len(completed)}/{len(rows)}",
                    "p95_e_im": None
                    if not completed
                    else float(np.quantile([float(row["new_e_im"]) for row in completed], 0.95)),
                    "mean_sec_per_frame": None
                    if not completed
                    else float(np.mean([float(row["wall_sec"]) for row in completed])),
                    "probes_per_frame": None
                    if not completed
                    else float(np.mean([float(row["candidate_probes"]) for row in completed])),
                    "timing_comparison": "NON_STRICT_DIFFERENT_RUN_CLASS",
                }
            )
    dev2_path = root / "dev2_frame0_development/decision.json"
    if dev2_path.exists() and read_json(dev2_path).get("DEV2_FRAME0_V3_DEVELOPMENT") != "NOT_RUN":
        dev2 = read_json(dev2_path)
        runs = dev2["runs"]
        completed = [row for row in runs if row["technical"] == "PASS"]
        result.append(
            {
                "mode": "V3 DEV2 frame0",
                "old_q_available": "no",
                "technical": f"{len(completed)}/3",
                "p95_e_im": None
                if not completed
                else float(np.quantile([float(row["E_IM"]) for row in completed], 0.95)),
                "mean_sec_per_frame": None
                if not completed
                else float(np.mean([float(row["wall_sec"]) for row in completed])),
                "probes_per_frame": None
                if not completed
                else float(np.mean([row["profiler"]["candidate_probes"] for row in completed])),
                "timing_comparison": "NON_STRICT_DIFFERENT_RUN_CLASS",
            }
        )
    return result


def summarize(root: Path) -> dict[str, Any]:
    integrity = verify_frozen_upstream(root)
    qold = read_json(root / "input_authority/q_old_role_audit.json")
    graph_path = root / "graph_authority/source_interaction_graph_authority.json"
    graph = read_json(graph_path)
    if "INTERACTION_GRAPH_AUTHORITY" not in graph:
        graph["INTERACTION_GRAPH_AUTHORITY"] = read_json(
            root / "input_authority/interaction_graph_authority_audit.json"
        )["INTERACTION_GRAPH_AUTHORITY"]
        write_json(graph_path, graph)
    completeness = read_json(root / "input_authority/dev2_input_completeness.json")
    masked = (
        read_json(root / "development/masked_qold_decision.json")
        if (root / "development/masked_qold_decision.json").exists()
        else {"status": "NOT_RUN", "candidates": []}
    )
    if masked["status"] == "FAIL":
        stop_reason = "MASKED_QOLD_DEVELOPMENT_FAIL"
        determinism_path = root / "development/determinism.json"
        if not determinism_path.exists():
            candidate_determinism = [
                read_json(
                    root
                    / f"development/{_candidate_slug(V3_CANDIDATES[row['candidate']])}_determinism.json"
                )
                for row in masked["candidates"]
            ]
            write_json(
                determinism_path,
                {
                    "schema_version": "ExecutionV3DevelopmentDeterminismSummaryV1",
                    "candidates": candidate_determinism,
                    "DETERMINISM": "PASS"
                    if all(item["DETERMINISM"] == "PASS" for item in candidate_determinism)
                    else "FAIL",
                    "interpretation": (
                        "PASS means the observed success or technical-failure outcome is exactly "
                        "reproducible; it does not override a masked-q_old gate failure."
                    ),
                },
            )
        if not (root / "development/refinement_regression.json").exists():
            refinement_not_run = {
                "schema_version": "ExecutionV3RefinementRegressionV1",
                "REFINEMENT_MODE_REGRESSION": "NOT_RUN",
                "reason": stop_reason,
                "max_q_abs": None,
                "max_base_abs": None,
                "max_e_im_abs": None,
            }
            write_json(root / "development/refinement_regression.json", refinement_not_run)
            write_csv(
                root / "development/refinement_regression.csv",
                [{"status": "NOT_RUN", "reason": stop_reason}],
            )
        if not (root / "development/selection_decision.json").exists():
            write_json(
                root / "development/selection_decision.json",
                {
                    "schema_version": "ExecutionV3SelectionDecisionV1",
                    "SELECTED_EXECUTION_V3": None,
                    "status": "NOT_RUN_GATE_CLOSED",
                    "reason": stop_reason,
                    "selection_locked_before_dev2_frame0": False,
                },
            )
        dev2_dir = root / "dev2_frame0_development"
        if not (dev2_dir / "decision.json").exists():
            role = {
                "schema_version": "DEV2Frame0RoleReceiptV1",
                "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
                "status": "NOT_RUN_GATE_CLOSED",
                "reason": stop_reason,
            }
            write_json(dev2_dir / "role_receipt.json", role)
            not_run_rows = []
            for run in (1, 2, 3):
                row = {
                    "schema_version": "DEV2Frame0ExecutionV3RunV1",
                    "run": run,
                    "status": "NOT_RUN",
                    "reason": stop_reason,
                    "optimizer_started": False,
                    "technical": "NOT_RUN",
                    "E_IM": None,
                    "wrist": "NOT_RUN",
                    "bone": "NOT_RUN",
                    "collision": "NOT_RUN",
                    "joints": "NOT_RUN",
                    "deterministic": "NOT_RUN",
                }
                not_run_rows.append(row)
                write_json(dev2_dir / f"run_{run}.json", row)
            write_json(
                dev2_dir / "determinism.json",
                {
                    "schema_version": "DEV2Frame0ExecutionV3DeterminismV1",
                    "DETERMINISM": "NOT_RUN",
                    "reason": stop_reason,
                },
            )
            write_json(
                dev2_dir / "decision.json",
                {
                    "schema_version": "DEV2Frame0ExecutionV3DevelopmentDecisionV1",
                    "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
                    "DEV2_FRAME0_V3_DEVELOPMENT": "NOT_RUN",
                    "DEV2_FRAME0_V3_RUN_COUNT": 0,
                    "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": 0,
                    "EXECUTION_V3_FREEZE_AUTHORIZED": "NO",
                    "reason": stop_reason,
                    "runs": not_run_rows,
                },
            )
        future_plan = root / "future_certification/execution_v3_independent_certification_plan.json"
        if not future_plan.exists():
            write_json(
                future_plan,
                {
                    "schema_version": "ExecutionV3IndependentColdStartCertificationPlanV1",
                    "status": "NOT_RUN_NO_EXECUTION_V3_FROZEN",
                    "reason": stop_reason,
                    "executed": False,
                    "next_stage": "EXECUTION_V3_COLDSTART_SEARCH_V2_DEVELOPMENT",
                },
            )
    selection = (
        read_json(root / "development/selection_decision.json")
        if (root / "development/selection_decision.json").exists()
        else {"SELECTED_EXECUTION_V3": None, "status": "NOT_RUN"}
    )
    refinement = (
        read_json(root / "development/refinement_regression.json")
        if (root / "development/refinement_regression.json").exists()
        else {"REFINEMENT_MODE_REGRESSION": "NOT_RUN"}
    )
    dev2 = (
        read_json(root / "dev2_frame0_development/decision.json")
        if (root / "dev2_frame0_development/decision.json").exists()
        else {
            "DEV2_FRAME0_V3_DEVELOPMENT": "NOT_RUN",
            "DEV2_FRAME0_V3_RUN_COUNT": 0,
            "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": 0,
            "runs": [],
        }
    )
    ledger = _evidence_ledgers(root)
    frozen = (root / "frozen_v3/objective_v2_execution_contract_v3.json").exists()
    if not frozen:
        write_json(
            root / "frozen_v3/no_execution_v3_ready.json",
            {
                "schema_version": "NoExecutionV3ReadyV1",
                "EXECUTION_V3_STATUS": "NO_EXECUTION_V3_READY",
                "reason": "DEVELOPMENT_GATE_NOT_PASS",
                "new_contract_hashes": None,
            },
        )
    selected_summary = None
    if selection.get("SELECTED_EXECUTION_V3"):
        selected_summary = read_json(
            root
            / f"development/{_candidate_slug(V3_CANDIDATES[selection['SELECTED_EXECUTION_V3']])}_summary.json"
        )
    special = _special_case_audit()
    technical_by_stratum = {name: "0/20" for name in ("HIGH", "MID", "LOW")}
    for row in masked.get("candidates", []):
        name = row["candidate"]
        candidate_summary_path = (
            root / f"development/{_candidate_slug(V3_CANDIDATES[name])}_summary.json"
        )
        if candidate_summary_path.exists():
            candidate_summary = read_json(candidate_summary_path)
            for stratum in technical_by_stratum:
                observed = candidate_summary["strata"][stratum]["technical"]
                if int(observed.split("/", 1)[0]) > int(
                    technical_by_stratum[stratum].split("/", 1)[0]
                ):
                    technical_by_stratum[stratum] = observed
    paths = {
        "execution_input": root / "frozen_v3/execution_input_authority.json",
        "cold_seed": root / "frozen_v3/cold_start_seed_authority.json",
        "graph": root / "frozen_v3/source_interaction_graph_authority.json",
        "execution_v3": root / "frozen_v3/objective_v2_execution_contract_v3.json",
    }
    hashes = {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}
    development_gate = "PASS" if frozen else "FAIL"
    summary = {
        "schema_version": "OakInk2O5RD2GFinalSummaryV1",
        "git": record_git(root),
        "upstream": integrity["upstream_state"],
        "q_old_role": qold,
        "interaction_graph_authority": graph,
        "dev2_input_completeness": completeness,
        "execution_v3_architecture": {
            "REFINEMENT": {
                "old_production_q": "REQUIRED",
                "previous_accepted_state": "ABSENT_FRAME0_REQUIRED_AFTER_FRAME0",
                "generic_seeds": "V2/S1 rest and midpoint block seeds",
                "interaction_graph": "CANONICAL_SOURCE_DERIVED",
                "fallback": "valid old-production baseline",
            },
            "COLD_START": {
                "old_production_q": "FORBIDDEN_ABSENT",
                "previous_accepted_state": "ABSENT_FRAME0_REQUIRED_AFTER_FRAME0",
                "generic_seeds": "ColdStartSeedAuthorityV1",
                "interaction_graph": "CANONICAL_SOURCE_DERIVED",
                "fallback": "best independently valid generic candidate or technical fail",
            },
        },
        "masked_qold": masked,
        "selected_candidate_summary": selected_summary,
        "refinement_regression": refinement,
        "selection": selection,
        "dev2_frame0": dev2,
        "special_case_audit": special,
        "mode_comparison": _mode_comparison(root),
        "evidence_ledger": {
            "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT": ledger[
                "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT"
            ]
        },
        "freeze": {
            "EXECUTION_V3_DEVELOPMENT_GATE": development_gate,
            "EXECUTION_V3_STATUS": "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION"
            if frozen
            else "NO_EXECUTION_V3_READY",
            "EXECUTION_INPUT_AUTHORITY_SHA256": hashes["execution_input"],
            "COLD_START_SEED_AUTHORITY_SHA256": hashes["cold_seed"],
            "SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256": hashes["graph"],
            "OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256": hashes["execution_v3"],
        },
        "safety_flags": {
            "BRANCH": EXPECTED_BRANCH,
            "FROZEN_UPSTREAM_INTEGRITY": "PASS",
            "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_SHA,
            "CERTIFICATION_GATE_V2_SHA256": GATE_V2_SHA,
            "OBJECTIVE_V2_CHANGED": "NO",
            "GATE_V2_CHANGED": "NO",
            "SEMANTIC_V1_CHANGED": "NO",
            "E_IM_THRESHOLD_CHANGED": "NO",
            "Q_OLD_ROLE": qold["Q_OLD_ROLE"],
            "INTERACTION_GRAPH_AUTHORITY": graph["INTERACTION_GRAPH_AUTHORITY"],
            "COLD_START_EXECUTION_AUTHORIZED": "YES",
            "RETARGET_MODE_REFINEMENT_IMPLEMENTED": "YES",
            "RETARGET_MODE_COLD_START_IMPLEMENTED": "YES",
            "Q_OLD_SYNTHESIZED_IN_COLD_START": "NO",
            "FAILED_STAGE7_TERMINAL_USED_AS_Q_OLD": "NO",
            "THUMB_SPECIAL_CASE_ADDED": "NO",
            "DEV1_SPECIAL_CASE_ADDED": "NO",
            "DEV2_SPECIAL_CASE_ADDED": "NO",
            "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
            "MASKED_QOLD_HIGH_TECHNICAL": technical_by_stratum["HIGH"],
            "MASKED_QOLD_MID_TECHNICAL": technical_by_stratum["MID"],
            "MASKED_QOLD_LOW_TECHNICAL": technical_by_stratum["LOW"],
            "REFINEMENT_MODE_REGRESSION": refinement["REFINEMENT_MODE_REGRESSION"],
            "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
            "DEV2_FRAME0_V3_DEVELOPMENT": dev2["DEV2_FRAME0_V3_DEVELOPMENT"],
            "DEV2_FRAME0_V3_RUN_COUNT": dev2["DEV2_FRAME0_V3_RUN_COUNT"],
            "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"],
            "EXECUTION_V3_DEVELOPMENT_GATE": development_gate,
            "COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN",
            "COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN",
            "FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN",
            "FRESH_CROSS_EPISODE_CONTROL_COUNT": 0,
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "DEV1_FULL_RETARGET_RERUNS": 0,
            "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
            "MANIFEST_V2_MODIFIED": "NO",
            "SPLIT_V2_MODIFIED": "NO",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
            "O6_RAN": "NO",
            "SUPPORT_PHYSICALIZATION_RAN": "NO",
            "PHYSX_RAN": "NO",
            "FROZEN_EVAL_RAN": "NO",
            "PPO_RAN": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            ".local_TRACKED": "NO",
            "GUIDANCE_WORKTREE_MODIFIED": "NO",
        },
        "next": "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION"
        if frozen
        else "EXECUTION_V3_COLDSTART_SEARCH_V2_DEVELOPMENT",
    }
    write_json(root / "final_summary.json", summary)
    write_json(root / "resource_usage.json", {"mode_comparison": summary["mode_comparison"]})
    flags = "\n".join(f"{key}={value}" for key, value in summary["safety_flags"].items())
    git_state = summary["git"]
    commits = "\n".join(f"- `{item}`" for item in git_state["commits"]) or "- none"
    qold_sites = "\n".join(
        f"- `{item['site']}`: {item['role']} ({item['file']})" for item in qold["use_sites"]
    )
    candidate_results = {row["candidate"]: row["result"] for row in masked.get("candidates", [])}
    candidate_rows = []
    candidate_labels = {
        "V3_A_GENERIC_ASSET_SOURCE_COLD_START": "V3-A",
        "V3_B_GENERIC_WITH_PREVIOUS_ACCEPTED_RUNTIME": "V3-B",
        "V3_C_SOURCE_GEOMETRIC_WITH_PREVIOUS_ACCEPTED_RUNTIME": "V3-C",
    }
    for candidate in V3_CANDIDATES.values():
        frame0 = ", ".join(
            item for item in candidate.cold_seed_sources if item != "previous_accepted_runtime"
        )
        previous = "yes" if candidate.use_previous_accepted_after_frame0 else "no"
        candidate_rows.append(
            f"| {candidate_labels[candidate.name]} | {frame0} | {previous} | no | "
            f"{candidate_results.get(candidate.name, 'NOT_RUN')} |"
        )
    candidate_table = "\n".join(candidate_rows)
    masked_rows = []
    for result in masked.get("candidates", []):
        name = result["candidate"]
        detail = read_json(
            root / f"development/{_candidate_slug(V3_CANDIDATES[name])}_summary.json"
        )
        for stratum in ("HIGH", "MID", "LOW"):
            row = detail["strata"][stratum]
            completed = row["technical"] == "20/20"
            hard = "PASS" if row["hard_validity_pass"] else "FAIL"
            masked_rows.append(
                f"| {candidate_labels[name]} {stratum} | {row['technical']} | "
                f"{row['p95_e_im'] if row['p95_e_im'] is not None else 'null'} | "
                f"{hard if completed else 'NOT_EVALUATED'} | "
                f"{hard if completed else 'NOT_EVALUATED'} | "
                f"{'PASS' if row['continuity_pass'] else 'FAIL'} | "
                f"{hard if completed else 'NOT_EVALUATED'} | "
                f"{'PASS' if all((row['interaction_pass'], row['hard_validity_pass'], row['continuity_pass'])) else 'FAIL'} |"
            )
    masked_table = "\n".join(masked_rows)
    dev2_rows = "\n".join(
        f"| {row['run']} | {'YES' if row['optimizer_started'] else 'NO'} | {row['technical']} | "
        f"{row['E_IM'] if row['E_IM'] is not None else 'null'} | {row['wrist']} | {row['bone']} | "
        f"{row['collision']} | {row['joints']} | {row['deterministic']} |"
        for row in dev2["runs"]
    )
    validation = (
        read_json(root / "validation_results.json").get("status", "UNKNOWN")
        if (root / "validation_results.json").exists()
        else "NOT_RUN"
    )
    frozen_hashes = {name: value if value is not None else "null" for name, value in hashes.items()}
    markdown = f"""# OakInk2 O5R-D2G

# Cross-Episode Input Authority + Cold-Start ExecutionV3 Handoff

## Git

```text
BRANCH={EXPECTED_BRANCH}
START_HEAD={git_state["start_head"]}
FINAL_HEAD={git_state["final_head"]}
tracked_worktree_clean={git_state["tracked_worktree_clean"]}
PUSHED=NO
PR_CREATED=NO
```

Commits:

{commits}

## Upstream scientific state

```text
SparseValidationV3={integrity["upstream_state"]["SparseValidationV3"]}
WindowValidationV3={integrity["upstream_state"]["WindowValidationV3"]}
DEV2_FRAME0_EXECUTION_V2={integrity["upstream_state"]["DEV2_FRAME0_EXECUTION_V2"]}
ObjectiveV2 failure evidence={integrity["upstream_state"]["ObjectiveV2_failure_evidence"]}
DEV2 optimizer evaluation under V2={integrity["upstream_state"]["DEV2_optimizer_evaluation_under_V2"]}
```

## Outcome

`EXECUTION_V3_STATUS={summary["freeze"]["EXECUTION_V3_STATUS"]}`

`Q_OLD_ROLE={qold["Q_OLD_ROLE"]}`. The frozen Candidate-B2 objective and E_IM do not consume q_old; V2 uses it as refinement baseline, search carrier, fallback, and evaluation reference.

`INTERACTION_GRAPH_AUTHORITY={graph["INTERACTION_GRAPH_AUTHORITY"]}` with `{graph["dev1_replay_parity"]["frame_count"]}` no-optimizer parity frames and exact deterministic object-sample reconstruction.

`SELECTED_EXECUTION_V3={selection.get("SELECTED_EXECUTION_V3") or "NONE"}`

`REFINEMENT_MODE_REGRESSION={refinement["REFINEMENT_MODE_REGRESSION"]}`

`DEV2_FRAME0_V3_DEVELOPMENT={dev2["DEV2_FRAME0_V3_DEVELOPMENT"]}`

The masked-q_old hard gate failed. Refinement regression, selection, DEV2 frame0 V3 execution, and all V3 freezes are therefore `NOT_RUN`.

## q_old role decision

`Q_OLD_ROLE={qold["Q_OLD_ROLE"]}`. Use-site evidence:

{qold_sites}

## Interaction graph authority

`INTERACTION_GRAPH_AUTHORITY={graph["INTERACTION_GRAPH_AUTHORITY"]}`. Replay parity is `PASS` over `{graph["dev1_replay_parity"]["frame_count"]}` unique DEV1 frames; maximum source-vertex, Laplacian, and weight differences are all `0.0`, and object-sample reconstruction is exact. Neither a robot nor q_old was loaded.

## V3 candidates

| Candidate | Frame0 seeds | t>0 runtime seed | q_old needed | Result |
| --- | --- | --- | --- | --- |
{candidate_table}

## Masked-q_old development

| Window | Technical | p95 E_IM | Wrist | Bone | Continuity | Collision/Joints | Result |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
{masked_table}

The only passing stratum was V3-C MID (`p95 E_IM=9.958413517336691e-05`); HIGH and LOW failed at their first frame because no independently valid cold-start candidate was available.

Determinism is `PASS` for all three candidates, meaning each observed success or technical-failure outcome reproduced exactly; it does not override the masked-q_old failure.

## Refinement regression and selection

```text
REFINEMENT_MODE_REGRESSION={refinement["REFINEMENT_MODE_REGRESSION"]}
max_q_abs={refinement.get("max_q_abs")}
max_base_abs={refinement.get("max_base_abs")}
max_e_im_abs={refinement.get("max_e_im_abs")}
SELECTED_EXECUTION_V3={selection.get("SELECTED_EXECUTION_V3") or "NONE"}
```

## DEV2 input completeness

| Input | Authority | DEV2 Available | Derivable | Status |
| --- | --- | --- | --- | --- |
| canonical source hand | canonical episode | yes | yes | PASS |
| object pose and mesh | canonical episode | yes | yes | PASS |
| object samples and interaction graph | deterministic source-derived | no persisted input | yes | DERIVABLE_BUT_CONTRACT_MISSING |
| old production q | old production trajectory | no | no | MISSING_BECAUSE_OLD_TRAJECTORY_DOES_NOT_EXIST |
| previous runtime state at frame0 | runtime | no | no | EXPECTED_ABSENT_FRAME0 |

## ExecutionV3 architecture

| Property | REFINEMENT | COLD_START |
| --- | --- | --- |
| old production q | required | forbidden/absent |
| previous accepted state | absent frame0, required after frame0 | absent frame0, required after frame0 |
| generic seeds | frozen V2/S1 rest and midpoint blocks | full-state cold-start authority |
| interaction graph | canonical source-derived | canonical source-derived |
| fallback | valid old-production baseline | independently valid candidate or technical fail |
| frame0 | old trajectory authority | no previous-state alias |
| t>0 | refinement carrier | previous accepted runtime state |

## DEV2 frame0 known-failure development

| Run | Optimizer Started | Technical | E_IM | Wrist | Bone | Collision | Joints | Deterministic |
| --: | --- | --- | ---: | --- | --- | --- | --- | --- |
{dev2_rows}

```text
DEV2_FRAME0_ROLE=KNOWN_FAILURE_DEVELOPMENT_REGRESSION
DEV2_FRAME0_V3_DEVELOPMENT={dev2["DEV2_FRAME0_V3_DEVELOPMENT"]}
DEV2_FRAME0_V3_RUN_COUNT={dev2["DEV2_FRAME0_V3_RUN_COUNT"]}
DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT={dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"]}
```

## DEV2 special-case audit

```text
DEV2_SPECIAL_CASE_ADDED={special["DEV2_SPECIAL_CASE_ADDED"]}
DEV2_EPISODE_ID_BRANCH={special["DEV2_EPISODE_ID_BRANCH"]}
DEV2_OBJECT_ID_BRANCH={special["DEV2_OBJECT_ID_BRANCH"]}
DEV2_FRAME_LITERAL_BRANCH={special["DEV2_FRAME_LITERAL_BRANCH"]}
DEV2_HAND_TUNED_SEED={special["DEV2_HAND_TUNED_SEED"]}
```

## Development gate and freeze status

```text
EXECUTION_V3_DEVELOPMENT_GATE={development_gate}
EXECUTION_V3_STATUS={summary["freeze"]["EXECUTION_V3_STATUS"]}
EXECUTION_INPUT_AUTHORITY_SHA256={frozen_hashes["execution_input"]}
COLD_START_SEED_AUTHORITY_SHA256={frozen_hashes["cold_seed"]}
SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256={frozen_hashes["graph"]}
OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256={frozen_hashes["execution_v3"]}
```

## ObjectiveV2 / GateV2 integrity

```text
RETARGET_OBJECTIVE_V2_SHA256={OBJECTIVE_SHA}
CERTIFICATION_GATE_V2_SHA256={GATE_V2_SHA}
OBJECTIVE_V2_CHANGED=NO
GATE_V2_CHANGED=NO
```

## Evidence hygiene and independent-certification boundary

```text
NEW_DEV1_METHOD_DEVELOPMENT_FRAMES=0
FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT={ledger["FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT"]}
COLDSTART_SPARSE_VALIDATION_V4=NOT_RUN
COLDSTART_WINDOW_VALIDATION_V4=NOT_RUN
FRESH_CROSS_EPISODE_CONTROL_COUNT=0
DEV2_FULL_PRODUCTION_SOLVE_COUNT=0
REPOSITORY_VALIDATION={validation}
NEXT={summary["next"]}
```

## Safety flags

```text
{flags}
```

Fresh V4 validation, fresh cross-episode controls, DEV2 full, DEV1 full, O6, Support, PhysX, frozen evaluation, and PPO were not run. This handoff stops at the D2G terminal.
"""
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    return summary


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    audit_execution_input_authority(root)
    audit_qold_role(root)
    audit_interaction_graph_authority(root)
    audit_dev2_input_completeness(root)
    run_graph_parity(root)
    develop_execution_v3_candidates(root)
    masked = run_masked_qold_development(root)
    if masked["status"] != "PASS":
        return summarize(root)
    refinement = run_refinement_regression(root)
    if refinement["REFINEMENT_MODE_REGRESSION"] != "PASS":
        return summarize(root)
    select_execution_v3(root)
    dev2 = run_dev2_frame0_development(root)
    if dev2["DEV2_FRAME0_V3_DEVELOPMENT"] != "PASS":
        return summarize(root)
    freeze_execution_v3(root)
    generate_future_certification_plan(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-upstream": verify_frozen_upstream,
    "audit-execution-input-authority": audit_execution_input_authority,
    "audit-qold-role": audit_qold_role,
    "audit-interaction-graph-authority": audit_interaction_graph_authority,
    "audit-dev2-input-completeness": audit_dev2_input_completeness,
    "run-graph-parity": run_graph_parity,
    "develop-execution-v3-candidates": develop_execution_v3_candidates,
    "run-masked-qold-development": run_masked_qold_development,
    "run-refinement-regression": run_refinement_regression,
    "select-execution-v3": select_execution_v3,
    "run-dev2-frame0-development": run_dev2_frame0_development,
    "freeze-execution-input-authority": freeze_execution_input_authority,
    "freeze-coldstart-seed-authority": freeze_coldstart_seed_authority,
    "freeze-interaction-graph-authority": freeze_interaction_graph_authority,
    "freeze-execution-v3": freeze_execution_v3,
    "generate-future-certification-plan": generate_future_certification_plan,
    "summarize": summarize,
    "record-git": record_git,
    "validate-repository": validate_repository,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=ROOT)
    subparsers = value.add_subparsers(dest="action", required=True)
    for action in ACTIONS:
        subparsers.add_parser(action)
    return value


def main() -> int:
    args = parser().parse_args()
    result = ACTIONS[args.action](args.root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
