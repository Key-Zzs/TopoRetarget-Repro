from pathlib import Path

import pytest

from toporetarget.workflows.schema import WorkflowRequest, run_id_for, stable_hash


def _request(**overrides: object) -> WorkflowRequest:
    values: dict[str, object] = {
        "sequence": "s1/airplane_lift",
        "index": Path(".local/index/grab"),
        "hand": "right",
        "robot": "artimano_rh",
        "start_frame": 240,
        "end_frame": 300,
        "window_length": 60,
        "repo_root": Path("/repo"),
    }
    values.update(overrides)
    return WorkflowRequest(**values)  # type: ignore[arg-type]


def test_workflow_request_requires_explicit_bounded_window() -> None:
    _request().validate()
    assert run_id_for(_request()) == "s1__airplane_lift__right__artimano_rh__f000240_f000300"
    with pytest.raises(ValueError, match="explicit frame range"):
        _request(start_frame=None, end_frame=None, auto_contact_window=False).validate()
    with pytest.raises(ValueError, match="does not match"):
        _request(robot="artimano_lh").validate()


def test_stable_hash_is_order_independent_and_path_aware() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert stable_hash({"path": Path("/tmp/input")}) == stable_hash({"path": "/tmp/input"})


def test_v2_solver_selection_is_explicit_without_changing_upstream_run_identity() -> None:
    v1 = _request()
    v2 = _request(refinement_solver_profile="scipy_slsqp_active_set_contact_rich_v2")
    assert run_id_for(v2) == run_id_for(v1)
