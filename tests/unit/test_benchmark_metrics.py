import numpy as np
import pytest

from toporetarget.metrics.common import trajectory_metrics
from toporetarget.metrics.contactpose import (
    contact_alignment_eq11,
    contact_precision_eq10,
    penetration_eq12,
)
from toporetarget.metrics.registry import metric_definitions


def test_contactpose_appendix_metrics_use_declared_units() -> None:
    source_hand = np.array([[0.0, 0.0, 0.0]])
    robot_hand = np.array([[0.001, 0.0, 0.0]])
    source_object = np.zeros((1, 3))
    robot_object = np.zeros((1, 3))
    assert contact_precision_eq10(
        robot_hand, source_hand, robot_object, source_object
    ) == pytest.approx(1.0)
    assert contact_alignment_eq11(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        source_object,
        source_object,
    ) == pytest.approx(90.0)
    assert penetration_eq12(np.array([[0.0, -0.001, 0.003]])) == {
        "max_penetration_mm": pytest.approx(1.0),
        "penetration_rate_2mm": pytest.approx(0.0),
        "min_signed_distance_mm": pytest.approx(-1.0),
        "threshold_mm": pytest.approx(2.0),
    }


def test_contact_alignment_rejects_zero_length_segment() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        contact_alignment_eq11(
            np.zeros((1, 3)),
            np.array([[1.0, 0.0, 0.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )


def test_static_trajectory_temporal_metrics_are_not_applicable() -> None:
    result = trajectory_metrics(dynamic=False, qpos=np.zeros((1, 2)), timestamps=np.zeros(1))
    assert result["q_velocity"] == "NOT_APPLICABLE"
    assert result["temporal_lag_diagnostic"] == "NOT_APPLICABLE"


def test_metric_registry_keeps_paper_exact_and_proxy_semantics_separate() -> None:
    definitions = {item.metric_id: item for item in metric_definitions()}
    assert definitions["contact_precision_eq10"].semantics == "PAPER_EXACT"
    assert definitions["grab_contact_precision_proxy"].semantics == "DATASET_PROXY"
    assert definitions["solve_time_ms_per_unit"].semantics == "ENGINEERING_DIAGNOSTIC"
