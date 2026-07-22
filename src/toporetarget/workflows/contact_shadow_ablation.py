"""Bounded causal shadow-ablation boundary for Stage 9.3.1.

The shadow workflow is deliberately fail-closed.  It can only be extended to
call the validated Stage 9 solver after the reconciliation manifest says that
the full-512 identity, reference SDF, transform chain, and acceptance replay
all pass.  A failed gate writes diagnostic metadata and performs zero solver
invocations.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MANDATORY_PROFILES = (
    "official_baseline_reproduction",
    "half_active_margin",
    "zero_active_margin",
    "full_512_query_reference",
    "minimal_soft_safe_projection_from_warm",
    "official_slack_projection_from_warm",
)
MAX_SHADOW_FRAMES = 3


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_empty_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["profile", "frame", "status", "accepted", "reason"])


def run_contact_shadow_ablation(
    reconciliation_root: str | Path,
    output_root: str | Path,
    *,
    profiles: tuple[str, ...] = MANDATORY_PROFILES,
    frames: tuple[int, ...] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Create a fail-closed shadow bundle; never mutate formal artifacts."""

    recon_root = Path(reconciliation_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"shadow output exists; pass --force: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = recon_root / "metric_reconciliation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary.get("gate", {})
    requested = tuple(profiles)
    if len(frames) > MAX_SHADOW_FRAMES:
        raise ValueError(f"at most {MAX_SHADOW_FRAMES} shadow frames are allowed")
    unknown = sorted(set(requested) - set(MANDATORY_PROFILES))
    if unknown:
        raise ValueError(f"unknown mandatory shadow profile(s): {', '.join(unknown)}")
    gate_pass = bool(
        summary.get("reconciliation_gate_pass") and gate.get("reconciliation_gate_pass")
    )
    reason = (
        "reconciliation gate passed; solver implementation is intentionally not implicit"
        if gate_pass
        else "SHADOW_NOT_RUN_RECONCILIATION_GATE_FAILED"
    )
    manifest = {
        "schema_version": "toporetarget.contact_shadow_ablation.v1",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "solver_invocation_count": 0,
        "ran": False,
        "gate_pass": gate_pass,
        "reason": reason,
        "profiles": list(requested),
        "frames": list(frames),
        "formal_artifact_mutation": False,
    }
    frame_selection = json.loads(
        (recon_root / "shadow_frame_selection.json").read_text(encoding="utf-8")
    )
    selected_frames = tuple(frames) or tuple(
        int(record["local_frame"]) for record in frame_selection.get("frames", [])
    )
    manifest["frames"] = list(selected_frames)
    profiles_payload = {
        "schema_version": "toporetarget.shadow_profiles.v1",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "profiles": [
            {
                "profile_id": profile,
                "status": "not_run",
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "isolation": {
                    "active_margin_changed": profile
                    in {"half_active_margin", "zero_active_margin"},
                    "query_mode_changed": profile == "full_512_query_reference",
                    "objective_changed": profile
                    in {
                        "minimal_soft_safe_projection_from_warm",
                        "official_slack_projection_from_warm",
                    },
                    "paper_weights_changed": False,
                    "formal_artifact_path": "never_write",
                },
                "reason": reason,
            }
            for profile in requested
        ],
    }
    _write_json(destination / "shadow_manifest.json", manifest)
    _write_json(destination / "shadow_frame_selection.json", frame_selection)
    _write_json(destination / "shadow_profiles.json", profiles_payload)
    _write_empty_csv(destination / "shadow_results_per_frame.csv")
    _write_json(
        destination / "shadow_results_per_profile.json",
        {"status": "not_run", "profiles": list(requested), "reason": reason},
    )
    _write_json(
        destination / "shadow_causal_analysis.json",
        {
            "schema_version": "toporetarget.shadow_causal_analysis.v1",
            "status": "not_run",
            "reason": reason,
            "causes": [],
        },
    )
    (destination / "shadow_causal_analysis.md").write_text(
        f"# Shadow causal analysis\n\nStatus: `not_run`\n\nReason: `{reason}`\n", encoding="utf-8"
    )
    _write_json(
        destination / "official_vs_projection.json", {"status": "not_run", "reason": reason}
    )
    _write_json(
        destination / "official_vs_margin_ablation.json", {"status": "not_run", "reason": reason}
    )
    readiness_status = summary.get("stage9_4_readiness", "STAGE9_4_NOT_YET_JUSTIFIED")
    _write_json(
        destination / "stage9_4_readiness.json",
        {"status": readiness_status, "enter_stage9_4": False, "reason": reason},
    )
    return manifest


__all__ = ["MANDATORY_PROFILES", "MAX_SHADOW_FRAMES", "run_contact_shadow_ablation"]
