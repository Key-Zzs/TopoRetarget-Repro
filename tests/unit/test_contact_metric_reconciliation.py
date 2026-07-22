from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toporetarget.workflows.contact_metric_reconciliation import (
    _classify_offset_rows,
    _definition_matrix,
    _penetration_metrics,
    _stats,
)
from toporetarget.workflows.contact_shadow_ablation import (
    MANDATORY_PROFILES,
    MAX_SHADOW_FRAMES,
    run_contact_shadow_ablation,
)


def test_raw_penetration_is_not_tau_adjusted_or_slack_adjusted() -> None:
    metrics = _penetration_metrics(np.asarray([-0.002, 0.004]), tau=0.001, bound=0.03)

    assert metrics == {
        "raw_max_penetration_m": 0.002,
        "tau_adjusted_max_penetration_m": 0.001,
        "hard_violation_max_m": 0.0,
    }


def test_stats_reports_signed_difference_distribution() -> None:
    stats = _stats(np.asarray([-0.002, 0.001, 0.003]))

    assert stats["count"] == 3
    assert stats["min_m"] == -0.002
    assert stats["max_m"] == 0.003
    assert stats["max_abs_m"] == 0.003


def test_definition_matrix_names_legacy_backend_as_diagnostic_only() -> None:
    final = SimpleNamespace(metadata={"paper_weights": {"tau_m": 0.001, "b_m": 0.03}})
    reference = SimpleNamespace(describe=lambda: {"backend_id": "reference"})
    legacy = SimpleNamespace(describe=lambda: {"backend_id": "legacy"})

    entries = _definition_matrix(final, reference, legacy)
    by_field = {entry["field"]: entry for entry in entries}

    assert by_field["max_penetration"]["subtract_tau"] is False
    assert by_field["max_penetration"]["slack"] is False
    assert by_field["stage9_3_legacy_full512_min"]["acceptance_role"] == (
        "not valid for formal acceptance"
    )


def _offset_row(
    *, reliable: bool, signed_median: float, outward: float, inward: float
) -> dict[str, object]:
    return {
        "normal_reliable": reliable,
        "signed_offset_median_m": signed_median,
        "outward_ratio": outward,
        "inward_ratio": inward,
        "classification": "reliable_directional"
        if reliable
        else "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE",
    }


def test_unsigned_offset_does_not_prove_inflated_and_direction_taxonomy_is_explicit() -> None:
    assert _classify_offset_rows(
        [_offset_row(reliable=False, signed_median=0.01, outward=1.0, inward=0.0)]
    ) == ("COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE")
    assert _classify_offset_rows(
        [_offset_row(reliable=True, signed_median=0.01, outward=0.9, inward=0.0)]
    ) == ("COLLISION_GEOMETRY_OUTWARD_INFLATED")
    assert _classify_offset_rows(
        [_offset_row(reliable=True, signed_median=-0.01, outward=0.0, inward=0.9)]
    ) == ("COLLISION_GEOMETRY_INSET")
    assert (
        _classify_offset_rows(
            [
                _offset_row(reliable=True, signed_median=0.01, outward=0.9, inward=0.0),
                _offset_row(reliable=True, signed_median=-0.01, outward=0.0, inward=0.9),
            ]
        )
        == "COLLISION_GEOMETRY_MIXED_OFFSET"
    )


def test_shadow_ablation_fails_closed_without_solver_invocation(tmp_path: Path) -> None:
    reconciliation = tmp_path / "reconciliation"
    output = tmp_path / "shadow"
    reconciliation.mkdir()
    (reconciliation / "metric_reconciliation_summary.json").write_text(
        '{"reconciliation_gate_pass": false, "gate": {"reconciliation_gate_pass": false}, '
        '"stage9_4_readiness": "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX"}\n',
        encoding="utf-8",
    )
    (reconciliation / "shadow_frame_selection.json").write_text(
        '{"frames": [{"local_frame": 4}, {"local_frame": 19}, {"local_frame": 55}]}\n',
        encoding="utf-8",
    )

    manifest = run_contact_shadow_ablation(reconciliation, output)

    assert manifest["ran"] is False
    assert manifest["solver_invocation_count"] == 0
    assert manifest["frames"] == [4, 19, 55]
    assert set(manifest["profiles"]) == set(MANDATORY_PROFILES)
    profile_payload = __import__("json").loads(
        (output / "shadow_profiles.json").read_text(encoding="utf-8")
    )
    assert all(
        item["diagnostic_only"]
        and not item["paper_method"]
        and not item["accepted_reference"]
        and item["isolation"]["formal_artifact_path"] == "never_write"
        for item in profile_payload["profiles"]
    )
    assert (output / "stage9_4_readiness.json").read_text(encoding="utf-8").find(
        "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX"
    ) >= 0


def test_shadow_ablation_rejects_more_than_three_frames(tmp_path: Path) -> None:
    reconciliation = tmp_path / "reconciliation"
    reconciliation.mkdir()
    (reconciliation / "metric_reconciliation_summary.json").write_text(
        '{"reconciliation_gate_pass": false, "gate": {"reconciliation_gate_pass": false}}\n',
        encoding="utf-8",
    )
    (reconciliation / "shadow_frame_selection.json").write_text(
        '{"frames": []}\n', encoding="utf-8"
    )

    try:
        run_contact_shadow_ablation(
            reconciliation,
            tmp_path / "shadow",
            frames=tuple(range(MAX_SHADOW_FRAMES + 1)),
        )
    except ValueError as exc:
        assert "at most 3" in str(exc)
    else:
        raise AssertionError("more than three shadow frames must be rejected")
