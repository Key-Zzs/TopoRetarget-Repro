from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.geometry_audit.exact_evaluator import evaluate_runtime_proxy_state


def test_exact_evaluator_fails_before_backend_on_state_shape() -> None:
    with pytest.raises(ValueError, match="object pose"):
        evaluate_runtime_proxy_state(
            manifest_path="not-read-before-shape-check.json",
            clip="hocap_170105",
            object_pose=np.zeros((10, 7)),
            hand_collision_body_pose=np.zeros((10, 1, 21, 7)),
            hand_collision_body_names=(),
        )
