"""Explicit metric semantics for the Q1--Q3 benchmark."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from toporetarget.benchmark.schema import METRIC_REGISTRY_VERSION


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    display_name: str
    mathematical_definition: str
    unit: str
    direction: str
    applicable_datasets: tuple[str, ...]
    semantics: str
    required_inputs: tuple[str, ...]
    missing_data_behavior: str
    aggregation_rule: str
    implementation_version: str = "1.0.0"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _m(
    metric_id: str,
    display_name: str,
    definition: str,
    unit: str,
    datasets: tuple[str, ...],
    semantics: str,
    inputs: tuple[str, ...],
    direction: str = "lower_is_better",
    missing: str = "N/A with reason; never zero",
    aggregation: str = "per-unit then equal-weight macro mean/median",
) -> MetricDefinition:
    return MetricDefinition(
        metric_id,
        display_name,
        definition,
        unit,
        direction,
        datasets,
        semantics,
        inputs,
        missing,
        aggregation,
    )


def metric_definitions() -> list[MetricDefinition]:
    result = [
        _m(
            "contact_precision_eq10",
            "Contact precision",
            "mean_c ||(h_r-o_r)-(h_s-o_s)||_2",
            "mm",
            ("contactpose",),
            "PAPER_EXACT",
            (
                "source_contact_points",
                "robot_contact_points",
                "source_object_points",
                "robot_object_points",
            ),
        ),
        _m(
            "contact_alignment_eq11",
            "Contact alignment",
            "mean_c acos(clip(u_r dot u_s,-1,1))",
            "degree",
            ("contactpose",),
            "PAPER_EXACT",
            ("source_segment_vectors", "robot_segment_vectors"),
        ),
        _m(
            "max_penetration_eq12",
            "Maximum penetration",
            "max_t,x [-d_Mt(x)]_+",
            "mm",
            ("contactpose", "grab"),
            "PAPER_EXACT",
            ("robot_surface_signed_distance",),
        ),
        _m(
            "penetration_rate_2mm_eq12",
            "Penetration rate > 2 mm",
            "N^-1 sum_t 1[max_x[-d_Mt(x)]_+ > 0.002]",
            "unit fraction",
            ("contactpose", "grab"),
            "PAPER_EXACT",
            ("robot_surface_signed_distance",),
            aggregation="dynamic: frame fraction; static: per-unit sample result, no pseudo frames",
        ),
        _m(
            "solve_time_ms_per_unit",
            "Solve time",
            "sum(frame solve seconds)*1000 / unit",
            "ms/unit",
            ("contactpose", "grab"),
            "ENGINEERING_DIAGNOSTIC",
            ("solve_time_s",),
        ),
        _m(
            "grab_contact_precision_proxy",
            "GRAB contact precision proxy",
            "mean object-relative source/robot proxy contact displacement",
            "mm",
            ("grab",),
            "DATASET_PROXY",
            ("source_proxy_points", "robot_proxy_points"),
        ),
        _m(
            "grab_contact_alignment_proxy",
            "GRAB contact alignment proxy",
            "mean angle of source/robot proxy contact segments",
            "degree",
            ("grab",),
            "DATASET_PROXY",
            ("source_proxy_vectors", "robot_proxy_vectors"),
        ),
    ]
    for threshold in (2, 3, 5, 8, 10):
        for measure in ("precision", "recall", "f1"):
            result.append(
                _m(
                    f"grab_contact_retention_{measure}_at_{threshold}mm",
                    f"GRAB contact retention {measure} at {threshold} mm",
                    f"thresholded source/robot proxy {measure} at {threshold} mm",
                    "unit fraction",
                    ("grab",),
                    "DATASET_PROXY",
                    ("source_proxy_distances", "robot_proxy_distances"),
                )
            )
    result.extend(
        [
            _m(
                "grab_per_finger_contact_retention_proxy",
                "GRAB per-finger contact retention",
                "per-finger thresholded proxy retention",
                "unit fraction",
                ("grab",),
                "DATASET_PROXY",
                ("source_semantic_contacts", "robot_proxy_contacts"),
            ),
            _m(
                "grab_contact_region_drift_proxy",
                "GRAB contact-region drift",
                "source proxy region to robot proxy region drift",
                "mm",
                ("grab",),
                "DATASET_PROXY",
                ("source_proxy_regions", "robot_proxy_regions"),
            ),
        ]
    )
    common = [
        ("optimizer_status", "optimizer status", "native optimizer termination code", "code"),
        ("solver_success", "solver success", "native solver success flag", "boolean"),
        ("strict_accepted", "strict accepted", "optimizer success plus strict audits", "boolean"),
        ("qpos_bounds", "qpos bounds", "all qpos within limits", "boolean"),
        ("slack_bounds", "slack bounds", "all slack within bounds", "boolean"),
        (
            "full_surface_hard_audit",
            "full 512 hard audit",
            "all full-surface hard constraints pass",
            "boolean",
        ),
        ("raw_max_penetration", "Raw maximum penetration", "max[-signed distance]_+", "mm"),
        (
            "raw_penetration_rate_2mm",
            "Raw penetration rate >2 mm",
            "fraction of applicable frames above 2 mm",
            "unit fraction",
        ),
        ("min_signed_distance", "Minimum signed distance", "min full signed distance", "mm"),
        ("active_set_rounds", "Active-set rounds", "outer continuation rounds", "rounds"),
        ("query_set_count", "QuerySet count", "number of persisted collision queries", "queries"),
        ("raw_keypoint_rmse", "Raw keypoint RMSE", "RMSE source/robot semantic keypoints", "mm"),
        (
            "morphology_normalized_keypoint_rmse",
            "Morphology-normalized keypoint RMSE",
            "RMSE after declared morphology normalization",
            "unitless",
        ),
        ("e_bone", "Bone-direction error", "E_bone", "unitless"),
        ("e_im", "Interaction-mesh error", "E_IM", "m^2"),
        ("laplacian_residual_mean", "Laplacian residual mean", "mean shared-graph residual", "m"),
        ("laplacian_residual_max", "Laplacian residual max", "max shared-graph residual", "m"),
        (
            "joint_limit_margin",
            "Joint-limit margin",
            "minimum normalized joint-limit margin",
            "unitless",
        ),
        ("q_velocity", "q velocity", "finite difference q velocity", "rad/s"),
        ("q_acceleration", "q acceleration", "finite difference q acceleration", "rad/s^2"),
        ("q_jerk", "q jerk", "finite difference q jerk", "rad/s^3"),
        (
            "base_translation_velocity",
            "Base translation velocity",
            "finite difference base translation",
            "m/s",
        ),
        (
            "base_translation_acceleration",
            "Base translation acceleration",
            "finite difference base translation",
            "m/s^2",
        ),
        (
            "base_rotation_velocity",
            "Base rotation velocity",
            "finite difference base rotation",
            "rad/s",
        ),
        ("max_interframe_q_step", "Maximum inter-frame q step", "max |q_t-q_{t-1}|", "rad"),
        ("max_base_step", "Maximum base step", "max base pose step", "mixed"),
        ("temporal_lag_diagnostic", "Temporal lag diagnostic", "bounded lag proxy", "frames"),
    ]
    for metric_id, display, definition, unit in common:
        result.append(
            _m(
                metric_id,
                display,
                definition,
                unit,
                ("grab", "contactpose"),
                "GENERIC_GEOMETRIC"
                if metric_id
                not in {
                    "optimizer_status",
                    "solver_success",
                    "strict_accepted",
                    "qpos_bounds",
                    "slack_bounds",
                    "full_surface_hard_audit",
                }
                else "ENGINEERING_DIAGNOSTIC",
                (metric_id,),
                missing="N/A for unavailable input; static temporal metrics are NOT_APPLICABLE",
            )
        )
    return result


def registry_payload() -> dict[str, Any]:
    definitions = metric_definitions()
    return {
        "schema_version": METRIC_REGISTRY_VERSION,
        "implementation_version": "toporetarget.benchmark.metric_registry.v1",
        "metrics": [item.as_dict() for item in definitions],
    }
