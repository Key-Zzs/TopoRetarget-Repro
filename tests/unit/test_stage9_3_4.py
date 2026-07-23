from __future__ import annotations

import numpy as np

from toporetarget.retarget.final_refinement import CollisionQueryProfile
from toporetarget.workflows.stage9_3_4 import _kabsch, _rotation_matrix_from_vector


def test_kabsch_is_se3_without_reflection_or_scale() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation = _rotation_matrix_from_vector(np.asarray([0.1, -0.2, 0.3]))
    target = (rotation @ source.T).T + np.asarray([0.2, -0.1, 0.05])
    transform = _kabsch(source, target)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-10)
    assert np.allclose((transform[:3, :3] @ source.T).T + transform[:3, 3], target, atol=1e-10)


def test_diagnostic_zero_margin_profile_is_nonnegative_and_versioned() -> None:
    profile = CollisionQueryProfile.load("zero_active_margin_diagnostic_v1")
    assert profile.paper_status == "diagnostic_only"
    assert profile.active_margin_m == 0.0
    assert profile.validate() is profile
