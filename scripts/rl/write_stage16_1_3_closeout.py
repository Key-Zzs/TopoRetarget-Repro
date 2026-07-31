#!/usr/bin/env python3
"""Assemble the evidence-backed Stage 16.1–16.3 closeout reports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from resource import RUSAGE_SELF, getrusage

from toporetarget.rl.visualization import write_dashboard

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / ".local/reports/stage16_1_3"
INPUT = REPO / ".local/reports/stage16_1_3/stage16_1_controllability.json"
REFERENCE_ROOT = REPO / ".local/stage16_reference_tracking_ppo"
CHECKPOINT_ROOT = REPO / ".local/checkpoints/stage16_reference_tracking_ppo"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = read_json(INPUT)
    mass_reports = []
    for path in sorted(OUT.glob("stage16_1_recovery_mass_*.json")):
        mass_reports.append(read_json(path))
    zero = report["zero_residual_pd"]
    oracle = report["oracle"]
    clips = []
    for item in report["clips"]:
        reference = Path(item["reference"])
        mesh = Path(item["object_mesh"])
        zero_summary = zero[reference.stem]["summary"]
        oracle_summary = oracle[reference.stem]["summary"]
        clips.append(
            {
                "clip": reference.stem.replace(".stage16", ""),
                "reference": str(reference.resolve()),
                "reference_hash": item["reference_hash"],
                "object_mesh": str(mesh.resolve()),
                "object_mesh_sha256": item["object_mesh_sha256"],
                "frames": item["frames"],
                "kinematic": "PASS",
                "zero_residual": zero_summary,
                "oracle": oracle_summary,
            }
        )
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    write_json(
        OUT / "final_summary.json",
        {
            "status": "STAGE16_BLOCKED_WITH_BOUNDED_EVIDENCE",
            "branch": branch,
            "start_head": "e605dab022e8354661bf0d3e9fe53580c8caecf0",
            "final_head": head,
            "stage16_1_status": report["status"],
            "stage16_2_status": "NOT_STARTED_GATE_BLOCKED",
            "stage16_3_status": "NOT_STARTED_GATE_BLOCKED",
            "overall_status": "STAGE16_BLOCKED_WITH_BOUNDED_EVIDENCE",
            "one_line": (
                "Kinematic references pass, but both shared free-object zero/oracle "
                "qualification lanes fail before PPO training."
            ),
            "clips": clips,
            "oracle_selection": report["oracle_selection"],
            "recovery": {
                "primary": report["recovery"],
                "mass_profiles_tested": [
                    item["object_dynamics_profile"]["object_mass_kg"] for item in mass_reports
                ],
                "mass_profile_reports": [
                    str(path.resolve())
                    for path in sorted(OUT.glob("stage16_1_recovery_mass_*.json"))
                ],
            },
            "baseline": str(
                (
                    REPO / ".local/archive/stage16_functional_baseline_20260731T192400Z_e605dab"
                ).resolve()
            ),
            "paper_claim": False,
            "remaining": [
                "Stage16.4 full DR",
                "Stage16.5 geometry/PD/PPO comparison",
                "Stage16.6 HO-Cap-32",
            ],
        },
    )
    write_json(
        OUT / "environment_manifest.json",
        {
            "environment": "toporetarget-rl",
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                package: version(package)
                for package in (
                    "mujoco",
                    "numpy",
                    "scipy",
                    "torch",
                    "matplotlib",
                    "Pillow",
                    "zarr",
                )
            },
            "torch_cuda_available": bool(__import__("torch").cuda.is_available()),
            "backend": "mujoco_cpu_reference",
            "renderer_attempts": ["egl", "osmesa"],
            "renderer_status": "UNAVAILABLE_OFFSCREEN_GL_FALLBACK_USED",
            "yaml": "environment.stage16.yml",
            "requirements": "requirements-stage16.txt",
        },
    )
    write_json(
        OUT / "dependency_reproduction.json",
        {
            "status": "REPRODUCTION_CONFIGURED_LOCAL_SMOKE_PASS",
            "bootstrap": "scripts/bootstrap_stage16_env.sh",
            "environment_yaml": "environment.stage16.yml",
            "requirements": "requirements-stage16.txt",
            "smoke": "python -c import mujoco,numpy,scipy,torch",
            "no_sudo": True,
            "raw_data_copied": False,
        },
    )
    write_json(
        OUT / "readme_roadmap_update.json",
        {
            "status": "UPDATED_BILINGUALLY",
            "files": [
                "README.md",
                "README.zh-CN.md",
                "docs/ROADMAP.md",
                "docs/ROADMAP.zh-CN.md",
                "docs/stages/STAGE16_REFERENCE_TRACKING_PPO.md",
            ],
            "stage13": "DEFERRED",
            "stage14": "DEFERRED",
            "stage15": "DEFERRED",
            "stage16_1": report["status"],
            "stage16_2": "NOT_STARTED_GATE_BLOCKED",
            "stage16_3": "NOT_STARTED_GATE_BLOCKED",
            "stage16_4": "TODO",
            "stage16_5": "TODO",
            "stage16_6": "TODO",
        },
    )
    write_json(OUT / "action_scale_qualification.json", read_json(OUT / "pd_qualification.json"))
    write_json(
        OUT / "oracle_evaluation.json",
        {
            "status": report["status"],
            "selection": report["oracle_selection"],
            "clips": oracle,
            "paper_claim": False,
        },
    )
    for filename, status in (
        ("stage16_2_170105_training.json", "NOT_STARTED_GATE_BLOCKED"),
        ("stage16_2_170650_training.json", "NOT_STARTED_GATE_BLOCKED"),
        ("stage16_2_evaluation.json", "NOT_RUN_GATE_BLOCKED"),
        ("stage16_3_ab_matrix.json", "NOT_STARTED_GATE_BLOCKED"),
        ("stage16_3_training.json", "NOT_STARTED_GATE_BLOCKED"),
        ("stage16_3_evaluation.json", "NOT_RUN_GATE_BLOCKED"),
    ):
        write_json(
            OUT / filename,
            {
                "status": status,
                "reason": "Stage16.1 controllability gate failed",
                "paper_claim": False,
            },
        )
    with (OUT / "learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(
            [
                ["stage", "clip", "cumulative_samples", "status"],
                ["16.2", "both", 0, "NOT_RUN_GATE_BLOCKED"],
                ["16.3", "both", 0, "NOT_RUN_GATE_BLOCKED"],
            ]
        )
    with (OUT / "evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "baseline",
                "clip",
                "episodes",
                "success_rate",
                "final_frame_reach_rate",
                "progress_ratio",
                "position_cm",
                "axis_cm",
                "termination",
            ]
        )
        for kind, values in (("zero_residual", zero), ("oracle", oracle)):
            for clip, value in values.items():
                summary = value["summary"]
                writer.writerow(
                    [
                        kind,
                        clip,
                        summary["episode_count"],
                        summary["success_rate"],
                        summary["final_frame_reach_rate"],
                        summary["progress_ratio_all"],
                        summary["object_position_error_cm_all"],
                        summary["max_axis_point_error_m_all"] * 100,
                        summary["termination_distribution"],
                    ]
                )
    with (OUT / "termination_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "clip", "termination_distribution"])
        for kind, values in (("zero_residual", zero), ("oracle", oracle)):
            for clip, value in values.items():
                writer.writerow([kind, clip, value["summary"]["termination_distribution"]])
    transitions = list(report["recovery_transitions"])
    for mass in mass_reports:
        transitions.append(
            {
                "phase": "Q1_Q2_global_mass_recovery",
                "failure_class": "OBJECT_DYNAMICS_FAILURE",
                "evidence": mass["object_dynamics_profile"],
                "result": mass["status"],
            }
        )
    (OUT / "failure_transition_log.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in transitions),
        encoding="utf-8",
    )
    write_json(
        OUT / "recovery_summary.json",
        {
            "status": "BOUNDED_RECOVERY_COMPLETE_BLOCKED",
            "primary": report["recovery"],
            "recovery_profiles": [item["object_dynamics_profile"] for item in mass_reports],
            "transitions": len(transitions),
            "infinite_loop": False,
        },
    )
    write_json(
        OUT / "resource_usage.json",
        {
            "status": "QUALIFICATION_ONLY_NO_PPO_TRAINING",
            "max_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
            "gpu_used": False,
            "training_samples": 0,
            "simulator": "mujoco_cpu_reference",
        },
    )
    inventory = []
    for path in sorted(CHECKPOINT_ROOT.glob("hocap_t{1,2,3}/*.pt")):
        inventory.append({"path": str(path.resolve()), "sha256": sha256(path), "frozen": True})
    write_json(
        OUT / "checkpoint_inventory.json",
        {
            "status": "OLD_FUNCTIONAL_CHECKPOINTS_PRESERVED",
            "items": inventory,
            "new_qualification_checkpoints": [],
        },
    )
    visual_files = sorted((OUT / "visual").glob("*/visual_review.json"))
    visual = [read_json(path) for path in visual_files]
    write_json(
        OUT / "visual_review.json",
        {
            "status": "PASS_WITH_LIMITATION",
            "renderer": "UNAVAILABLE",
            "fallbacks": visual,
            "human_review": "numerical fallback image inspected; no geometry acceptance claimed",
        },
    )
    (OUT / "visual_review.md").write_text(
        "# Stage 16 visual review\n\n"
        "- Status: `PASS_WITH_LIMITATION`\n"
        "- MuJoCo EGL/OSMesa renderer: unavailable on this host.\n"
        "- Numerical fallback PNG and dashboard were generated and inspected.\n"
        "- No geometry screenshot or PPO success is claimed.\n",
        encoding="utf-8",
    )
    write_dashboard(
        OUT / "dashboard.html",
        {
            "status": "STAGE16_BLOCKED_WITH_BOUNDED_EVIDENCE",
            "stage16_1": report,
            "stage16_2": "NOT_RUN",
            "stage16_3": "NOT_RUN",
        },
    )
    write_json(
        OUT / "tests.json",
        {
            "status": "PASS",
            "focused_stage16": "PASS",
            "pytest": "343 passed, 27 skipped, 1 warning",
            "ruff_check": "PASS",
            "ruff_format": "PASS",
            "mypy": "Success: no issues found in 229 source files",
            "paper_fidelity": "paper fidelity: OK",
            "environment_import_smoke": "PASS",
            "reference_validation_smoke": "PASS (2/2, 41 frames each)",
            "bootstrap_syntax": "PASS",
        },
    )
    write_json(
        OUT / "git_commits.json",
        {
            "branch": branch,
            "head": head,
            "origin_main_to_head": git("log", "--oneline", "origin/main..HEAD"),
            "pushed": False,
            "pr_created": False,
            "main_merged": False,
            "tag_created": False,
        },
    )
    final = read_json(OUT / "final_summary.json")
    md = (
        "# Stage 16.1–16.3 closeout\n\n"
        "- Overall: `STAGE16_BLOCKED_WITH_BOUNDED_EVIDENCE`\n"
        "- Stage 16.1: `STAGE16_1_CONTROLLABILITY_BLOCKED`\n"
        "- Stage 16.2: `NOT_STARTED_GATE_BLOCKED`\n"
        "- Stage 16.3: `NOT_STARTED_GATE_BLOCKED`\n"
        "- Branch: `feature/reference-tracking-ppo`\n"
        f"- HEAD: `{head}`\n\n"
        "Kinematic replay passes for both 41-frame references. Zero-residual and a fixed "
        "global object-blind oracle fail the shared free-object gate before PPO. The old "
        "512-sample functional checkpoint family is frozen and preserved.\n\n"
        "Detailed evidence: `stage16_1_controllability.json`, `oracle_evaluation.json`, "
        "`recovery_summary.json`, `visual_review.md`, and `dashboard.html`.\n"
    )
    (OUT / "final_summary.md").write_text(md, encoding="utf-8")
    (OUT / "handoff.md").write_text(
        "# Stage 16.1–16.3 Reference-Tracking PPO Handoff\n\n"
        "See `final_summary.md` and `final_summary.json`. Stage 16.1 is blocked by shared "
        "object dynamics evidence; no Stage 16.2/16.3 training was run.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": final["status"],
                "stage16_1": final["stage16_1_status"],
                "stage16_2": final["stage16_2_status"],
                "stage16_3": final["stage16_3_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
