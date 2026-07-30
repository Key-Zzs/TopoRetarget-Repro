from __future__ import annotations

import pytest

from toporetarget.workflows import s1_3_jobs


def test_s1_3_pause_resume_and_scope_isolation(tmp_path) -> None:
    initial = s1_3_jobs.initialize(tmp_path)
    assert initial["paused"] is False
    paused = s1_3_jobs.pause(tmp_path)
    assert paused["paused"] is True
    assert paused["scheduler"]["state"] == "pause_requested"
    resumed = s1_3_jobs.resume(tmp_path)
    assert resumed["paused"] is False
    assert resumed["scheduler"]["state"] == "ready"
    with pytest.raises(ValueError, match="unsupported jobs scope"):
        s1_3_jobs.pause(tmp_path, scope="main")
