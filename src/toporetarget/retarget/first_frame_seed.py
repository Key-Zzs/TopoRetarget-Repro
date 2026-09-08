"""Versioned, episode-agnostic first-frame seed authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class FirstFrameSeedAuthorityV2:
    """Declarative candidate set; it never changes Stage-7 numerical math."""

    profile_id: str
    version: str
    candidate_ids: tuple[str, ...]
    probe_max_nfev: int
    selection_rule: tuple[str, ...]
    profile_hash: str
    source_path: Path

    def candidates(self, neutral_q: Any, lower: Any, upper: Any) -> list[tuple[str, np.ndarray]]:
        neutral = np.asarray(neutral_q, dtype=np.float64)
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        if neutral.ndim != 1 or neutral.shape != lo.shape or lo.shape != hi.shape:
            raise ValueError("FIRST_FRAME_SEED_SHAPE_MISMATCH")
        if (
            not np.all(np.isfinite(neutral))
            or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
        ):
            raise ValueError("FIRST_FRAME_SEED_NONFINITE_AUTHORITY")
        if np.any(lo > hi):
            raise ValueError("FIRST_FRAME_SEED_INVALID_BOUNDS")
        if np.any(neutral < lo) or np.any(neutral > hi):
            raise ValueError("FIRST_FRAME_SEED_NEUTRAL_OUT_OF_BOUNDS")
        catalog = {
            "v1_neutral": neutral,
            "joint_range_midpoint": lo + 0.5 * (hi - lo),
            "joint_range_lower_quartile": lo + 0.25 * (hi - lo),
            "joint_range_upper_quartile": lo + 0.75 * (hi - lo),
        }
        unknown = sorted(set(self.candidate_ids) - set(catalog))
        if unknown:
            raise ValueError(f"FIRST_FRAME_SEED_UNKNOWN_CANDIDATE:{unknown}")
        return [(name, catalog[name].copy()) for name in self.candidate_ids]

    def select(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Rank valid probes by objective, then frozen candidate order."""

        order = {name: index for index, name in enumerate(self.candidate_ids)}
        eligible = [
            row
            for row in records
            if bool(row.get("finite"))
            and bool(row.get("within_bounds"))
            and bool(row.get("probe_progressed"))
            and np.isfinite(float(row.get("probe_final_objective", np.inf)))
        ]
        if not eligible:
            raise ValueError("NO_VALID_FIRST_FRAME_SEED")
        return min(
            eligible,
            key=lambda row: (
                not bool(row.get("probe_success")),
                float(row["probe_final_objective"]),
                order[str(row["seed_id"])],
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "FirstFrameSeedAuthorityV2",
            "profile_id": self.profile_id,
            "version": self.version,
            "candidate_ids": list(self.candidate_ids),
            "probe_max_nfev": self.probe_max_nfev,
            "selection_rule": list(self.selection_rule),
            "profile_hash": self.profile_hash,
            "source_path": str(self.source_path),
            "scientific_math_changed": False,
            "episode_specific_logic": False,
        }


def load_first_frame_seed_authority_v2(path: str | Path | None = None) -> FirstFrameSeedAuthorityV2:
    """Load the tracked V2 authority and bind raw bytes into the receipt."""

    source = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3]
        / "configs/retarget/warm_start/first_frame_seed_authority_v2.yaml"
    )
    raw = source.read_bytes()
    values = yaml.safe_load(raw) or {}
    if not isinstance(values, dict):
        raise ValueError("FIRST_FRAME_SEED_AUTHORITY_NOT_MAPPING")
    candidate_ids = tuple(str(item) for item in values.get("candidate_ids", ()))
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("FIRST_FRAME_SEED_CANDIDATES_INVALID")
    probe_max_nfev = int(values.get("probe_max_nfev", 0))
    if probe_max_nfev <= 0:
        raise ValueError("FIRST_FRAME_SEED_PROBE_BUDGET_INVALID")
    return FirstFrameSeedAuthorityV2(
        profile_id=str(values["profile_id"]),
        version=str(values["version"]),
        candidate_ids=candidate_ids,
        probe_max_nfev=probe_max_nfev,
        selection_rule=tuple(str(item) for item in values.get("selection_rule", ())),
        profile_hash=hashlib.sha256(raw).hexdigest(),
        source_path=source.resolve(),
    )


__all__ = ["FirstFrameSeedAuthorityV2", "load_first_frame_seed_authority_v2"]
