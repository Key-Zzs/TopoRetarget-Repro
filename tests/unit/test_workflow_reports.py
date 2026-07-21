from toporetarget.workflows.reports import (
    build_execution_reports,
    build_input_audit,
    stage9_window_geometry_audit,
    write_execution_reports,
)
from toporetarget.workflows.schema import WorkflowRequest


def test_input_audit_hashes_bounded_files_without_scanning_external_roots(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    index = repo / "index"
    index.mkdir()
    (index / "index.jsonl").write_text("{}\n", encoding="utf-8")
    (index / "manifest.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    manual = tmp_path / "manual.json"
    manual.write_text("{}\n", encoding="utf-8")
    request = WorkflowRequest(
        sequence="s1/example",
        index=index,
        hand="right",
        robot="artimano_rh",
        start_frame=0,
        end_frame=60,
        window_length=60,
        mano_model_root=tmp_path / "external_mano",
        asset_root=tmp_path / "external_assets",
        repo_root=repo,
        run_root=tmp_path / "runs",
    )
    audit = build_input_audit(
        request=request,
        selection={"sequence": request.sequence, "source_path": str(source)},
        manual_acceptance=manual,
        profile_hashes={},
    )
    assert audit["status"] == "pass"
    assert audit["source"]["sha256"]
    assert audit["external_roots"]["mano_model_root"]["exists"] is False
    assert audit["read_only_contract"]["native_time_preserved"] is True


def test_execution_reports_keep_failed_runs_honest(tmp_path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    manifest = {
        "run_id": "synthetic",
        "run_status": "failed",
        "run_root": str(tmp_path),
        "source_path": str(source),
        "source_hash": __import__("hashlib").sha256(b"source").hexdigest(),
        "nodes": [
            {
                "node_id": "final_refinement",
                "status": "failed",
                "reused": False,
                "skipped": False,
                "expected_signature": "sig",
                "actual_signature": None,
                "validation_status": "fail",
                "invalidation_reason": None,
                "duration_s": 1.5,
                "output_hashes": {},
            }
        ],
        "contact_window_selection": {"selection_hash": "selection"},
        "validations": {},
        "manual_acceptance": {},
    }
    reports = build_execution_reports(manifest, run_root=tmp_path, elapsed_s=2.0)
    assert reports["stage10_summary"]["completion_status"] == "STAGE10_BLOCKED"
    assert reports["source_integrity"]["source_integrity_check"] == "pass"
    assert reports["end_to_end_validation"]["status"] == "blocked"
    assert reports["semantic_sanity"]["status"] == "blocked"


def test_write_execution_reports_maintains_global_stage10_index(tmp_path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    run_root = tmp_path / "runs" / "run-a"
    manifest = {
        "repo_root": str(tmp_path),
        "run_root": str(run_root),
        "run_id": "run-a",
        "run_status": "failed",
        "source_path": str(source),
        "source_hash": __import__("hashlib").sha256(b"source").hexdigest(),
        "nodes": [],
        "contact_window_selection": {},
        "validations": {},
        "manual_acceptance": {},
    }
    paths = write_execution_reports(manifest, run_root=run_root, elapsed_s=0.0)
    assert paths["end_to_end_validation_csv"].is_file()
    assert (tmp_path / ".local" / "reports" / "stage10" / "resume_validation.json").is_file()


def test_stage9_window_geometry_classification_is_bound_to_stage10_manifest(tmp_path) -> None:
    report = tmp_path / ".local/reports/stage9_solver_closeout/window_geometry_audit.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        '{"diagnostic_config": {"distance_units": "m"}, "windows": ['
        '{"window_id": "contact", "window_identity": {"sequence": "s1/x", '
        '"hand": "right", "global_frame_range": [240, 300]}, '
        '"window_class": "contact_rich", "classification_matches_expected": true, '
        '"classification_metrics": {"semantic_contact_frame_ratio": 1.0, '
        '"source_mano_object_median_m": 0.03, "source_contact_geometry_median_m": 0.003}, '
        '"aggregate_distances": {}}]}',
        encoding="utf-8",
    )
    request = WorkflowRequest(
        sequence="s1/x",
        index=tmp_path / "index",
        hand="right",
        robot="artimano_rh",
        start_frame=240,
        end_frame=300,
        window_length=60,
        repo_root=tmp_path,
        run_root=tmp_path / "runs",
    )
    audit = stage9_window_geometry_audit(
        repo_root=tmp_path,
        request=request,
        selected={"start_frame": 240, "end_frame": 300},
    )
    assert audit["status"] == "pass"
    assert audit["window_class"] == "contact_rich"
    assert audit["semantic_contact_frame_ratio"] == 1.0
