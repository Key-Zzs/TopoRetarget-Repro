"""Pure planning helpers for the Stage 10 DAG."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .cache import path_hash, signature_payload
from .registry import get_node_specs, validate_dag
from .schema import NodeState, WorkflowPlan, WorkflowRequest, run_id_for, stable_hash, write_json

PROFILE_PATHS = {
    "paper_retarget": "configs/paper/retarget.yaml",
    "object_surface": "configs/geometry/object_surface_sampling.yaml",
    "signed_distance": "configs/geometry/signed_distance.yaml",
    "bone_profile": "configs/retarget/bones/mediapipe21_full_finger_chain_v1.yaml",
    "frame_profile": "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
    "delaunay_profile": "configs/retarget/interaction/strict_scipy_qhull_v1.yaml",
    "query_profile": "configs/retarget/collision_queries/adaptive_active_set_v1.yaml",
    "coordinate_profile": "configs/retarget/refinement/local_seed_delta_v1.yaml",
    "solver_profile": "configs/retarget/refinement_solvers/scipy_slsqp_active_set_v1.yaml",
    "robot_surface": "configs/geometry/robot_collision_sampling.yaml",
    "contact_mapping": "configs/datasets/grab_contact_parts.yaml",
}

STAGE9_DEPENDENT_NODES = {
    "final_refinement",
    "validate_final_refinement",
    "full_surface_penetration_audit",
    "semantic_sanity_validation",
    "generate_review_bundle",
    "write_run_manifest",
}


def git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return commit or "unknown", dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def profile_hashes(
    repo_root: Path, solver_profile_id: str = "scipy_slsqp_active_set_v1"
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in PROFILE_PATHS.items():
        if name == "solver_profile":
            relative = f"configs/retarget/refinement_solvers/{solver_profile_id}.yaml"
        path = repo_root / relative
        if path.is_file():
            result[name] = path_hash(path)
    if "solver_profile" not in result:
        raise ValueError(f"unregistered refinement solver profile: {solver_profile_id}")
    return result


def node_profile_hashes(node_id: str, configs: dict[str, str]) -> dict[str, str]:
    """Scope solver-profile invalidation to Stage 9 and downstream nodes."""

    if node_id in STAGE9_DEPENDENT_NODES:
        return dict(configs)
    return {key: value for key, value in configs.items() if key != "solver_profile"}


def run_paths(request: WorkflowRequest, *, run_id: str | None = None) -> dict[str, str]:
    identifier = run_id or run_id_for(request)
    root = (request.run_root / identifier).resolve()
    artifacts = root / "artifacts"
    reports = root / "reports"
    review = root / "review"
    exports = root / "exports"
    solver_hash = profile_hashes(request.repo_root.resolve(), request.refinement_solver_profile)[
        "solver_profile"
    ]
    final_name = "final.zarr"
    if request.refinement_solver_profile != "scipy_slsqp_active_set_v1":
        final_name = f"final__{request.refinement_solver_profile}__{solver_hash[:16]}.zarr"
    return {
        "run_root": str(root),
        "manifest": str(root / "manifest.json"),
        "plan": str(root / "plan.json"),
        "status": str(root / "status.json"),
        "logs": str(root / "logs"),
        "reports": str(reports),
        "review": str(review),
        "exports": str(exports),
        "canonical": str(artifacts / "canonical.zarr"),
        "object_samples": str(artifacts / "object_samples.npz"),
        "warm_start": str(artifacts / "warm_start.zarr"),
        "graph": str(artifacts / "interaction_graph.zarr"),
        "evaluation": str(artifacts / "interaction_evaluation.zarr"),
        "final": str(artifacts / final_name),
        "reference_zarr": str(exports / "robot_reference.zarr"),
        "reference_npz": str(exports / "robot_reference.npz"),
    }


def _node_outputs(node_id: str, paths: dict[str, str]) -> dict[str, str]:
    reports = Path(paths["reports"])
    values: dict[str, str] = {
        "resolve_source": str(reports / "source.json"),
        "canonicalize_grab": paths["canonical"],
        "validate_canonical": str(reports / "canonical_validation.json"),
        "enrich_mediapipe21": str(reports / "mediapipe21.json"),
        "validate_keypoints": str(reports / "keypoint_validation.json"),
        "audit_object_mesh": str(reports / "object_mesh_audit.json"),
        "sample_object_surface": paths["object_samples"],
        "validate_object_samples": str(reports / "object_samples_validation.json"),
        "generate_warm_start": paths["warm_start"],
        "validate_warm_start": str(reports / "warm_start_validation.json"),
        "build_interaction_graph": paths["graph"],
        "validate_interaction_graph": str(reports / "interaction_graph_validation.json"),
        "evaluate_warm_start_interaction": paths["evaluation"],
        "final_refinement": paths["final"],
        "validate_final_refinement": str(reports / "final_validation.json"),
        "full_surface_penetration_audit": str(reports / "penetration_audit.json"),
        "semantic_sanity_validation": str(reports / "semantic_sanity.json"),
        "generate_review_bundle": paths["review"],
        "write_run_manifest": paths["manifest"],
    }
    return {"artifact": values[node_id]}


def build_plan(
    request: WorkflowRequest,
    *,
    selected_window: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> WorkflowPlan:
    """Create a plan without importing MANO, robot, mesh, or solver modules."""

    request.validate()
    order = validate_dag()
    identifier = run_id or run_id_for(request)
    paths = run_paths(request, run_id=identifier)
    repo_root = request.repo_root.resolve()
    configs = profile_hashes(repo_root, request.refinement_solver_profile)
    request_value = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in request.__dict__.items()
    }
    request_value["run_root"] = str(request.run_root)
    request_value["repo_root"] = str(request.repo_root)
    nodes: list[NodeState] = []
    previous_output = None
    for node_id in order:
        spec = next(item for item in get_node_specs() if item.node_id == node_id)
        scoped_request = dict(request_value)
        if node_id not in STAGE9_DEPENDENT_NODES:
            scoped_request.pop("refinement_solver_profile", None)
        inputs = {"source": stable_hash(scoped_request)}
        if previous_output is not None:
            inputs["upstream_plan"] = stable_hash(previous_output)
        payload = signature_payload(
            node_id=node_id,
            implementation_version=spec.implementation_version,
            inputs=inputs,
            configs=node_profile_hashes(node_id, configs),
            parameters={"selected_window": selected_window, "paths": _node_outputs(node_id, paths)},
        )
        output = _node_outputs(node_id, paths)
        node = NodeState(
            node_id=node_id,
            implementation_version=spec.implementation_version,
            dependencies=list(spec.dependencies),
            input_hashes=inputs,
            config_hashes=node_profile_hashes(node_id, configs),
            output_paths=output,
            expected_signature=str(payload["signature"]),
        )
        nodes.append(node)
        previous_output = payload
    assumptions = [
        "A_WORKFLOW_CONTACT_WINDOW_THRESHOLD_001",
        "A_WORKFLOW_SOURCE_CONTACT_SANITY_001",
        "A_WORKFLOW_FINAL_CONTACT_SANITY_001",
        "A_WORKFLOW_ARTIFACT_REUSE_001",
        "A_WORKFLOW_INVALIDATION_001",
        "A_WORKFLOW_MANUAL_ACCEPTANCE_001",
    ]
    return WorkflowPlan(
        run_id=identifier,
        run_root=paths["run_root"],
        request=request_value,
        nodes=nodes,
        selected_window=selected_window,
        assumptions=assumptions,
    )


def write_plan(plan: WorkflowPlan, path: str | Path) -> Path:
    return write_json(plan, path)


__all__ = [
    "PROFILE_PATHS",
    "build_plan",
    "git_state",
    "node_profile_hashes",
    "profile_hashes",
    "run_paths",
    "write_plan",
]
