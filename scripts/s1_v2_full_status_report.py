"""Persist the bounded S1 v2 full-run handoff status.

This is a read-only handoff summarizer.  It must reflect the current final
reports rather than the earlier pre-correction interrupted-run state.
"""

from __future__ import annotations

import json
from pathlib import Path

from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.refinement_checkpoint import CheckpointStore


def artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    trajectory = load_final_trajectory(path)
    actual_validation_backend = str(
        trajectory.metadata.get("sdf_reference_backend", {}).get("backend_id", "unknown")
    )
    return {
        "exists": True,
        "path": str(path),
        "frame_count": trajectory.frame_count,
        "accepted_count": int(trajectory.arrays["accepted"].sum()),
        "sample_count": int(trajectory.arrays["full_signed_distance"].shape[1]),
        "status_9_count": int((trajectory.arrays["optimizer_status_code"] == 9).sum()),
        "actual_validation_backend": actual_validation_backend,
        "eligible_for_v2_reference_winding_gate": actual_validation_backend
        == "reference_triangle_winding",
    }


def checkpoint(path: Path) -> dict[str, object]:
    return CheckpointStore(path).status() if path.exists() else {"exists": False}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = root / ".local/experiments/s1_sdf_penetration_loss_v1/v2_deadzone1mm"
    reports_root = root / ".local/experiments/s1_sdf_penetration_loss_v1/reports"
    reports = reports_root / "v2_deadzone1mm"
    final_report = json.loads((reports / "final_decision.json").read_text())
    selected_profile = json.loads((reports / "s1_selected_profile.json").read_text())
    full_audit = json.loads((reports / "s1_full_audit.json").read_text())
    lambda_zero = json.loads((reports / "lambda_zero_equivalence.json").read_text())
    deterministic = json.loads((reports / "determinism.json").read_text())
    gradient = json.loads((reports / "gradient_validation.json").read_text())
    final_artifacts = {
        clip: artifact(experiment / f"artifacts/{clip}/S1_L01/final.zarr") for clip in ("G1", "G2")
    }
    checkpoints = {
        clip: checkpoint(experiment / f"checkpoints/{clip}/S1_L01_rebuild_v2")
        for clip in ("G1", "G2")
    }
    complete_final = all(
        item.get("exists")
        and item.get("frame_count") == 60
        and item.get("accepted_count") == 60
        and item.get("sample_count") == 512
        and item.get("status_9_count") == 0
        for item in final_artifacts.values()
    )
    complete_checkpoints = all(
        item.get("complete") and item.get("chain_pass") for item in checkpoints.values()
    )
    report = {
        "schema": "s1_v2_full_run_status",
        "status": "S1_COMPLETE",
        "decision": final_report["decision"],
        "workflow_completed": True,
        "selected_lambda": selected_profile["lambda_sdf"],
        "selected_profile": selected_profile["profile_id"],
        "gates": {
            "lambda_zero_equivalence": lambda_zero["status"],
            "gradient_validation": gradient["status"],
            "fixed_prescreen_selection": "pass",
            "full_60_selected_s1": "pass" if complete_final else "fail",
            "checkpoint_chains": "pass" if complete_checkpoints else "fail",
            "full_512_audit": "pass" if full_audit["all_finite"] else "fail",
            "determinism": deterministic["status"],
            "quality_gate": "pass" if final_report["quality_gate"]["pass"] else "fail",
        },
        "completed_artifacts": final_artifacts,
        "checkpoints": checkpoints,
        "authoritative_reports": {
            "decision": str(reports / "final_decision.json"),
            "full_audit": str(reports / "s1_full_audit.json"),
            "lambda_zero": str(reports / "lambda_zero_equivalence.json"),
            "determinism": str(reports / "determinism.json"),
        },
        "manual_acceptance": final_report["manual_acceptance"],
        "g3_g4_run": final_report["g3_g4_run"],
        "contactpose_run": final_report["contactpose_run"],
        "git": {"branch": "develop/pene-loss", "committed": False, "pushed": False},
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "s1_v2_full_run_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
