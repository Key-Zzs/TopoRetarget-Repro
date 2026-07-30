from __future__ import annotations

import pytest

from toporetarget.retarget.final_jobs import (
    FinalJobPaused,
    FinalRefinementCPUConfig,
    assert_final_jobs_allowed,
    pause_final_jobs,
    paused,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile


def test_pause_sentinel_fails_closed_without_touching_other_job_roots(tmp_path) -> None:
    root = tmp_path / "repo"
    pause_final_jobs(root, reason="unit test")
    assert paused(root)
    with pytest.raises(FinalJobPaused, match="PAUSED_BY_OPERATOR_CONTROL"):
        assert_final_jobs_allowed(root)
    assert (root / ".local" / "control" / "final_jobs" / "pause_manifest.json").is_file()


def test_fast_exact_execution_profile_stays_nondefault_until_parity() -> None:
    profile = RefinementExecutionProfile.load("wuji_continuous_sequential_fast_exact_v1")
    assert profile.role == "performance_candidate"
    assert profile.math_equivalent
    assert profile.final_full_surface_audit
    assert not profile.recommended
    assert not profile.stage12_default


def test_worker_cap_is_tied_to_physical_cpu_budget() -> None:
    config = FinalRefinementCPUConfig.load()
    assert config.max_workers == 1
    assert config.blas_threads == config.torch_threads == config.torch_interop_threads == 1
