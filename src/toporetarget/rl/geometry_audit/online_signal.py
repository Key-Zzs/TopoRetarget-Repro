"""Qualification contract for a fast, non-authoritative online geometry signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _recall(mask: np.ndarray, detected: np.ndarray) -> float:
    positives = int(mask.sum())
    return 1.0 if positives == 0 else float(np.mean(detected[mask]))


@dataclass(frozen=True)
class OnlineGeometrySignalQualificationV1:
    catastrophic_recall: float
    over_3mm_recall: float
    penetration_sign_agreement: float
    no_contact_false_positive_rate: float
    holdout_p95_absolute_depth_error_m: float
    calibration_sample_count: int
    holdout_sample_count: int
    exact_top_k: int
    elite_count: int
    exact_for_all_elites: bool
    exact_for_all_final_candidates: bool
    exact_for_all_formal_replicas: bool

    @property
    def fast_signal_pass(self) -> bool:
        return (
            self.catastrophic_recall == 1.0
            and self.over_3mm_recall >= 0.99
            and self.penetration_sign_agreement >= 0.99
            and self.no_contact_false_positive_rate <= 0.01
            and self.holdout_p95_absolute_depth_error_m <= 0.0005
        )

    @property
    def exact_topk_fallback_pass(self) -> bool:
        return (
            self.exact_top_k >= 2 * self.elite_count
            and self.exact_for_all_elites
            and self.exact_for_all_final_candidates
            and self.exact_for_all_formal_replicas
        )

    def as_dict(self) -> dict[str, Any]:
        if self.calibration_sample_count < 1 or self.holdout_sample_count < 1:
            raise ValueError("online signal qualification requires frozen calibration and holdout")
        if self.fast_signal_pass:
            status = "STAGE16D_ONLINE_GEOMETRY_SIGNAL_VALIDATED"
            role = "candidate pre-ranking, PPO dense penalty, and online monitoring"
        elif self.exact_topk_fallback_pass:
            status = "STAGE16D_EXACT_TOPK_GEOMETRY_FALLBACK_VALIDATED"
            role = "broad prefilter only; exact python-fcl selects elites and final candidates"
        else:
            status = "STAGE16D_GEOMETRY_AWARE_OPTIMIZATION_RUNTIME_BLOCKED"
            role = "not authorized"
        return {
            "schema_version": "OnlineRuntimeProxyGeometrySignalQualificationV1",
            **asdict(self),
            "fast_signal_pass": self.fast_signal_pass,
            "exact_topk_fallback_pass": self.exact_topk_fallback_pass,
            "status": status,
            "authorized_role": role,
            "formal_gate_authority": False,
            "formal_gate": "exact python-fcl RuntimeCollisionProxy metric",
        }


def qualify_online_geometry_signal(
    *,
    estimated_penetration_m: np.ndarray,
    exact_penetration_m: np.ndarray,
    contact_active: np.ndarray,
    split: np.ndarray,
    exact_top_k: int,
    elite_count: int,
    exact_for_all_elites: bool,
    exact_for_all_final_candidates: bool,
    exact_for_all_formal_replicas: bool,
    numerical_epsilon_m: float = 5.0e-7,
) -> OnlineGeometrySignalQualificationV1:
    estimated = np.asarray(estimated_penetration_m, dtype=np.float64)
    exact = np.asarray(exact_penetration_m, dtype=np.float64)
    active = np.asarray(contact_active, dtype=bool)
    labels = np.asarray(split, dtype=str)
    if estimated.shape != exact.shape or active.shape != exact.shape or labels.shape != exact.shape:
        raise ValueError("online geometry qualification arrays must have identical shape")
    if exact.ndim != 1 or not np.isfinite(estimated).all() or not np.isfinite(exact).all():
        raise ValueError("online geometry qualification requires finite 1D arrays")
    if np.any(estimated < 0.0) or np.any(exact < 0.0):
        raise ValueError("penetration depths must be nonnegative")
    calibration = labels == "calibration"
    holdout = labels == "holdout"
    if not calibration.any() or not holdout.any() or np.any(~(calibration | holdout)):
        raise ValueError("split must contain only nonempty calibration and holdout sets")
    detected = estimated > numerical_epsilon_m
    exact_sign = exact > numerical_epsilon_m
    no_contact = (~active) & (~exact_sign)
    false_positive = float(np.mean(detected[no_contact])) if no_contact.any() else 0.0
    holdout_error = np.abs(estimated[holdout] - exact[holdout])
    return OnlineGeometrySignalQualificationV1(
        catastrophic_recall=_recall(exact >= 0.010, estimated >= 0.010),
        over_3mm_recall=_recall(exact > 0.003, estimated > 0.003),
        penetration_sign_agreement=float(np.mean(detected == exact_sign)),
        no_contact_false_positive_rate=false_positive,
        holdout_p95_absolute_depth_error_m=float(np.quantile(holdout_error, 0.95)),
        calibration_sample_count=int(calibration.sum()),
        holdout_sample_count=int(holdout.sum()),
        exact_top_k=int(exact_top_k),
        elite_count=int(elite_count),
        exact_for_all_elites=exact_for_all_elites,
        exact_for_all_final_candidates=exact_for_all_final_candidates,
        exact_for_all_formal_replicas=exact_for_all_formal_replicas,
    )


__all__ = ["OnlineGeometrySignalQualificationV1", "qualify_online_geometry_signal"]
