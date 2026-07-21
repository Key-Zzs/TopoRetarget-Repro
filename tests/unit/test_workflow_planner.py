from pathlib import Path

from toporetarget.workflows.planning import build_plan, node_profile_hashes
from toporetarget.workflows.registry import validate_dag
from toporetarget.workflows.schema import WorkflowRequest


def test_stage10_plan_has_declared_dag_and_stable_artifact_layout() -> None:
    request = WorkflowRequest(
        sequence="s1/airplane_lift",
        index=Path(".local/index/grab"),
        hand="right",
        robot="artimano_rh",
        start_frame=240,
        end_frame=300,
        window_length=60,
        repo_root=Path(__file__).resolve().parents[2],
        run_root=Path("/runs"),
    )
    plan = build_plan(request, selected_window={"start_frame": 240, "end_frame": 300})
    assert len(plan.nodes) == 19
    assert [node.node_id for node in plan.nodes] == validate_dag()
    assert plan.nodes[0].dependencies == []
    assert plan.nodes[-1].dependencies == ["generate_review_bundle"]
    assert plan.nodes[1].output_paths["artifact"].endswith("artifacts/canonical.zarr")
    assert "A_WORKFLOW_INVALIDATION_001" in plan.assumptions


def test_solver_profile_change_invalidates_stage9_only_and_downstream() -> None:
    values = dict(
        sequence="s1/airplane_lift",
        index=Path(".local/index/grab"),
        hand="right",
        robot="artimano_rh",
        start_frame=240,
        end_frame=300,
        window_length=60,
        repo_root=Path(__file__).resolve().parents[2],
        run_root=Path("/runs"),
    )
    v1 = build_plan(
        WorkflowRequest(**values), selected_window={"start_frame": 240, "end_frame": 300}
    )
    v2 = build_plan(
        WorkflowRequest(
            **values, refinement_solver_profile="scipy_slsqp_active_set_contact_rich_v2"
        ),
        selected_window={"start_frame": 240, "end_frame": 300},
    )
    v1_nodes = {node.node_id: node for node in v1.nodes}
    v2_nodes = {node.node_id: node for node in v2.nodes}
    assert (
        v1_nodes["generate_warm_start"].expected_signature
        == v2_nodes["generate_warm_start"].expected_signature
    )
    assert (
        v1_nodes["final_refinement"].expected_signature
        != v2_nodes["final_refinement"].expected_signature
    )
    assert node_profile_hashes("generate_warm_start", {"solver_profile": "v2", "x": "1"}) == {
        "x": "1"
    }
    assert node_profile_hashes("final_refinement", {"solver_profile": "v2", "x": "1"}) == {
        "solver_profile": "v2",
        "x": "1",
    }
