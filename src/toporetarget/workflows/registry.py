"""The explicit Stage 10 DAG registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowNodeSpec:
    node_id: str
    dependencies: tuple[str, ...]
    implementation_version: str = "stage10-v1"


NODE_SPECS: tuple[WorkflowNodeSpec, ...] = (
    WorkflowNodeSpec("resolve_source", ()),
    WorkflowNodeSpec("canonicalize_grab", ("resolve_source",)),
    WorkflowNodeSpec("validate_canonical", ("canonicalize_grab",)),
    WorkflowNodeSpec("enrich_mediapipe21", ("validate_canonical",)),
    WorkflowNodeSpec("validate_keypoints", ("enrich_mediapipe21",)),
    WorkflowNodeSpec("audit_object_mesh", ("validate_keypoints",)),
    WorkflowNodeSpec("sample_object_surface", ("audit_object_mesh",)),
    WorkflowNodeSpec("validate_object_samples", ("sample_object_surface",)),
    WorkflowNodeSpec("generate_warm_start", ("validate_object_samples",)),
    WorkflowNodeSpec("validate_warm_start", ("generate_warm_start",)),
    WorkflowNodeSpec("build_interaction_graph", ("validate_warm_start",)),
    WorkflowNodeSpec("validate_interaction_graph", ("build_interaction_graph",)),
    WorkflowNodeSpec("evaluate_warm_start_interaction", ("validate_interaction_graph",)),
    WorkflowNodeSpec("final_refinement", ("evaluate_warm_start_interaction",)),
    WorkflowNodeSpec("validate_final_refinement", ("final_refinement",)),
    WorkflowNodeSpec("full_surface_penetration_audit", ("validate_final_refinement",)),
    WorkflowNodeSpec("semantic_sanity_validation", ("full_surface_penetration_audit",)),
    WorkflowNodeSpec("generate_review_bundle", ("semantic_sanity_validation",)),
    WorkflowNodeSpec("write_run_manifest", ("generate_review_bundle",)),
)


def get_node_specs() -> tuple[WorkflowNodeSpec, ...]:
    return NODE_SPECS


def validate_dag(specs: tuple[WorkflowNodeSpec, ...] = NODE_SPECS) -> list[str]:
    by_id = {item.node_id: item for item in specs}
    if len(by_id) != len(specs):
        raise ValueError("workflow DAG contains duplicate node IDs")
    for item in specs:
        missing = set(item.dependencies) - set(by_id)
        if missing:
            raise ValueError(f"{item.node_id} has missing dependencies: {sorted(missing)}")
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"workflow DAG cycle detected at {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in by_id[node_id].dependencies:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(node_id)

    for item in specs:
        visit(item.node_id)
    return order


__all__ = ["NODE_SPECS", "WorkflowNodeSpec", "get_node_specs", "validate_dag"]
