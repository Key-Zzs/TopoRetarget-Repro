from __future__ import annotations

import json
from pathlib import Path

from scripts.report_h3_protocol_cleanup import _stable_hash, build


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_protocol_cleanup_retains_history_and_replaces_binary_gate(tmp_path: Path) -> None:
    protocol_value = {
        "schema_version": "H3PhysicalizationProtocolV1",
        "freeze": {"H3_EXECUTION_HEAD": "a" * 40},
    }
    protocol = tmp_path / "protocol.json"
    _write(protocol, protocol_value)
    protocol_hash = tmp_path / "protocol_hash.txt"
    protocol_hash.write_text(_stable_hash(protocol_value), encoding="utf-8")
    h3a = tmp_path / "h3a.json"
    _write(
        h3a,
        {
            "H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT": (
                "SourceControllerAutoV2_ZERO_RESIDUAL_THEN_BOUNDED_L0_V1"
            )
        },
    )
    h3b = tmp_path / "h3b.json"
    _write(h3b, {"H3B_SELECTED_RETARGET_EXECUTION_CONTRACT": "fast_exact_v2"})
    supersession = tmp_path / "supersession.json"
    _write(
        supersession,
        {
            "status": "SUPERSEDED_FOR_UNSEEN_OBJECT_SPLIT",
            "historical_manifest_path": ".local/old_p6.json",
        },
    )
    result = build(protocol, protocol_hash, h3a, h3b, supersession, tmp_path / "out")
    assert result["source_admission"] == "SourceControllerExecutableV2"
    assert result["unbounded_l0_production"] is False
    assert result["historical_reports_deleted"] is False
    assert (tmp_path / "out/retained_history.csv").is_file()
