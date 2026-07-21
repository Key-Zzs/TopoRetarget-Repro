import json

import pytest

from toporetarget.workflows.validation import validate_manual_acceptance


def test_manual_acceptance_requires_human_frames_and_non_invalid_interpretation(tmp_path) -> None:
    path = tmp_path / "manual.json"
    payload = {
        "schema_version": "toporetarget.manual_acceptance.v1",
        "status": "pass",
        "reviewer": "human",
        "reviewed_frames": [0, 29, 59],
        "current_window_interpretation": "pre_contact",
        "contact_rich_clip_validated": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_manual_acceptance(path)["reviewer"] == "human"
    payload["current_window_interpretation"] = "invalid"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_manual_acceptance(path)
