"""Opt-in real Stage 9.1 artifact checks.

The test never discovers or mutates licensed data.  A user supplies the exact
new v2 artifact and, optionally, its deterministic repeat through environment
variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from toporetarget.retarget.final_refinement import (
    FINAL_REFINEMENT_SCHEMA_VERSION_V2,
    load_final_trajectory,
)


@pytest.mark.licensed_data
def test_stage9_v2_real_60_frame_acceptance_opt_in() -> None:
    artifact_name = os.environ.get("TOPORETARGET_STAGE9_V2_ARTIFACT")
    if not artifact_name:
        pytest.skip("set TOPORETARGET_STAGE9_V2_ARTIFACT for the real 60-frame check")
    artifact = load_final_trajectory(Path(artifact_name))
    assert artifact.schema_version == FINAL_REFINEMENT_SCHEMA_VERSION_V2
    assert artifact.frame_count == 60
    assert artifact.metadata["solver_profile_id"] == "scipy_slsqp_active_set_contact_rich_v2"
    assert bool(np.all(artifact.arrays["optimizer_converged"]))
    assert bool(np.all(artifact.arrays["accepted"]))
    assert bool(np.all(artifact.arrays["active_set_converged"]))
    assert bool(np.all(artifact.arrays["qpos_bounds_pass"]))
    assert bool(np.all(artifact.arrays["slack_bounds_pass"]))
    assert bool(np.all(artifact.arrays["active_constraints_feasible"]))
    assert bool(np.all(artifact.arrays["full_surface_hard_audit_pass"]))
    assert bool(np.all(artifact.arrays["full_surface_soft_audit_pass"]))
    assert bool(np.all(artifact.arrays["unqueried_soft_violation_count"] == 0))
    repeat_name = os.environ.get("TOPORETARGET_STAGE9_V2_REPEAT_ARTIFACT")
    if repeat_name:
        repeat = load_final_trajectory(Path(repeat_name))
        assert repeat.frame_count == artifact.frame_count
        assert artifact.arrays.keys() == repeat.arrays.keys()
        for name in artifact.arrays:
            assert np.array_equal(artifact.arrays[name], repeat.arrays[name], equal_nan=True), name
