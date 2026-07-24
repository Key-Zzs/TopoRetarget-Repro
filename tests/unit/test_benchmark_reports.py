import json
from pathlib import Path

from toporetarget.benchmark.dashboard import build_dashboard
from toporetarget.benchmark.evaluate import write_selection_blocked_reports


def test_blocked_selection_report_preserves_rejection_counts(tmp_path: Path) -> None:
    grab_candidates = [
        {"native_sample_id": "s1/a", "rejection_reasons": []},
        {"native_sample_id": "s1/b", "rejection_reasons": ["contact_frame_ratio_below_0.70"]},
    ]
    contactpose_candidates = [
        {
            "native_sample_id": "cp/a",
            "rejection_reasons": ["official_contact_annotation_unavailable_or_unrecognized"],
        },
        {
            "native_sample_id": "cp/b",
            "rejection_reasons": [
                "official_contact_annotation_unavailable_or_unrecognized",
                "excluded_deep_concave_diagnostic_set",
            ],
        },
    ]
    (tmp_path / "grab_candidates.json").write_text(json.dumps(grab_candidates), encoding="utf-8")
    (tmp_path / "contactpose_candidates.json").write_text(
        json.dumps(contactpose_candidates), encoding="utf-8"
    )
    selection = {
        "grab": {
            "status": "pass",
            "candidate_pool_count": 10,
            "evaluated_candidate_count": 2,
            "scan_truncated": True,
            "valid_additional_count": 1,
            "selected": [{"native_sample_id": "s1/a"}],
        },
        "contactpose": {
            "status": "blocked",
            "candidate_count": 2,
            "selected": [],
        },
    }
    result = write_selection_blocked_reports(benchmark_root=tmp_path, selection_result=selection)
    assert result["status"] == "Q1_CONTACTPOSE_SELECTION_BLOCKED"
    stats = json.loads((tmp_path / "selection_rejection_stats.json").read_text())
    assert stats["contactpose"]["reasons"][0] == {
        "count": 2,
        "reason": "official_contact_annotation_unavailable_or_unrecognized",
    }
    dashboard = build_dashboard(tmp_path)
    assert dashboard.is_file()
    assert "Q1_CONTACTPOSE_SELECTION_BLOCKED" in dashboard.read_text(encoding="utf-8")
