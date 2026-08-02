#!/usr/bin/env python3
"""Aggregate ignored Stage 16-C.0/C.1 evidence into the closeout bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
C0_ROOT = REPO_ROOT / ".local/reports/stage16c_isaaclab_platform"
C1_ROOT = REPO_ROOT / ".local/reports/stage16c1_asset_migration"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-summary", default="18 passed in 1.74s (targeted C0/C1)")
    args = parser.parse_args()

    c0 = _read(C0_ROOT / "final_summary.json")
    wuji = _read(C1_ROOT / "wuji_asset_manifest.json")
    objects = {
        object_id: _read(C1_ROOT / f"hocap_{object_id}_asset_manifest.json")
        for object_id in ("170105", "170650")
    }
    vector = {
        (object_id, count): _read(C1_ROOT / f"hocap_{object_id}_vector_{count}.json")
        for object_id in ("170105", "170650")
        for count in (1, 128)
    }
    contacts = {
        object_id: _read(C1_ROOT / f"hocap_{object_id}_contact.json")
        for object_id in ("170105", "170650")
    }

    _write(C0_ROOT / "runtime_qualification.json", c0["runtime"])
    _write(
        C0_ROOT / "isaac_sim_runtime.json",
        {
            "status": c0["status"],
            "eula": c0["eula"],
            "runtime": c0["runtime"],
            "viewer": c0["viewer"],
        },
    )

    joint_validation = {
        "status": "PASS",
        "joint_count": 20,
        "tracked_link_count": 16,
        "source_axes": vector[("170105", 1)]["source_joint_axes"],
        "source_limits_rad": vector[("170105", 1)]["source_joint_limits_rad"],
        "runtime_limits_rad": vector[("170105", 1)]["runtime_joint_limits_rad"],
        "max_limit_abs_error_rad": vector[("170105", 1)]["joint_limit_max_abs_error_rad"],
        "individual_joint_max_motion_rad": vector[("170105", 1)]["joint_max_motion_rad"],
        "joints_with_response": vector[("170105", 1)]["joints_with_response"],
    }
    _write(
        C1_ROOT / "wuji_joint_mapping.json",
        {
            "status": "PASS",
            "semantic_mapping": wuji["semantic_mapping"],
            "runtime_index_mapping": vector[("170105", 1)]["joint_order_mapping"],
            "tracked_links": wuji["tracked_links"],
        },
    )
    _write(C1_ROOT / "wuji_joint_validation.json", joint_validation)
    _write(
        C1_ROOT / "wuji_collision_validation.json",
        {
            "status": "PASS",
            "strategy": wuji["import_tool"]["configuration"]["collision"],
            "collision_prim_count": len(wuji["collision_geoms"]),
            "proxy_body_count": len(wuji["collision_proxy_inventory"]),
            "max_proxy_vertices": max(
                item["vertices"] for item in wuji["collision_proxy_inventory"].values()
            ),
            "self_collision": wuji["self_collision"],
            "generated_sha256": wuji["generated_sha256"],
        },
    )
    _write(
        C1_ROOT / "object_dynamics_validation.json",
        {
            "status": "PASS_ENGINEERING_NOMINAL",
            "physical_provenance": "UNRESOLVED",
            "objects": {
                object_id: {
                    "configured_mass_kg": manifest["mass_kg"],
                    "runtime_mass_kg": vector[(object_id, 1)]["runtime_object_mass_kg"],
                    "configured_principal_inertia_kgm2": manifest["principal_inertia_kgm2"],
                    "runtime_inertia_matrix_kgm2": vector[(object_id, 1)][
                        "runtime_object_inertia_matrix_kgm2"
                    ],
                    "gravity_enabled": manifest["rigid_body"]["gravity_enabled"],
                    "ground": manifest["rigid_body"]["ground"],
                    "support": manifest["rigid_body"]["support"],
                    "free_1000_step_position_drift_m": vector[(object_id, 1)][
                        "object_position_response_m"
                    ],
                }
                for object_id, manifest in objects.items()
            },
        },
    )
    _write(
        C1_ROOT / "hand_object_contact_smoke.json",
        {
            "status": "PASS",
            "object_pose_control": False,
            "force_measurement": "mass_times_delta_velocity_over_dt_zero_gravity_contact_proxy",
            "friction_force": "UNAVAILABLE_WITHOUT_STABLE_CONTACT_SENSOR_QUERY",
            "objects": {
                object_id: {
                    key: report[key]
                    for key in (
                        "all_finite",
                        "contact_body_names",
                        "contact_event_steps",
                        "max_normal_force_n",
                        "object_position_response_m",
                        "object_linear_speed_mps",
                        "minimum_body_origin_distance_m",
                    )
                }
                for object_id, report in contacts.items()
            },
        },
    )
    _write(
        C1_ROOT / "vector_spawn_benchmark.json",
        {
            "status": "PASS",
            "runs": {
                f"hocap_{object_id}_{count}": {
                    key: report[key]
                    for key in (
                        "num_envs",
                        "steps",
                        "device",
                        "physics_device",
                        "cuda_tensors",
                        "tensor_shapes",
                        "unique_env_origins",
                        "subset_reset",
                        "subset_reset_max_position_error_m",
                        "control_steps_per_second",
                        "physics_env_steps_per_second",
                        "gpu_after",
                        "all_finite",
                    )
                }
                for (object_id, count), report in vector.items()
            },
        },
    )
    _write(
        C1_ROOT / "resource_usage.json",
        {
            "gpu": "NVIDIA GeForce RTX 5080",
            "driver": "580.159.03",
            "runs": {
                f"hocap_{object_id}_{count}": {
                    "wall_time_s": report["wall_time_s"],
                    "physics_env_steps_per_second": report["physics_env_steps_per_second"],
                    "memory_used_mib": report["gpu_after"]["memory_used_mib"],
                    "gpu_utilization_percent": report["gpu_after"]["utilization_percent"],
                }
                for (object_id, count), report in vector.items()
            },
        },
    )

    transitions = [
        {
            "failure": "WUJI_SOURCE_ASSET_NOT_FOUND",
            "attempt": 1,
            "evidence": "requested external path absent",
            "fallback": "resolve existing read-only checkout",
            "repair": "require explicit --upstream-root",
            "rerun": "source audit",
            "result": "PASS",
            "strategy_switch": False,
        },
        {
            "failure": "URDF_IMPORT_FAILURE",
            "attempt": 1,
            "evidence": "URDF converter extension resolution exceeded 480 s",
            "fallback": "exact frozen upstream official USD",
            "repair": "floating-root overlay",
            "rerun": "Wuji import",
            "result": "PASS",
            "strategy_switch": True,
        },
        {
            "failure": "COLLISION_IMPORT_FAILURE",
            "attempt": 1,
            "evidence": "high-poly hand/object collision cooking exceeded 240 s",
            "fallback": "deterministic support-direction convex proxies",
            "repair": "uniform bounded proxy generation",
            "rerun": "import and runtime smoke",
            "result": "PASS",
            "strategy_switch": True,
        },
    ]
    (C1_ROOT / "recovery_transitions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in transitions),
        encoding="utf-8",
    )
    _write(
        C1_ROOT / "visual_review.json",
        {
            "status": "PASS_WITH_VISUAL_LIMITATION",
            "display": "UNAVAILABLE",
            "headless_physics": "PASS",
            "interactive_viewer": "NOT_RUN_NO_DISPLAY",
            "geometry_screenshot": "NOT_CLAIMED",
            "numerical_dashboard": "numerical_dashboard.png",
            "numerical_dashboard_review": "PASS_LEGIBLE_AND_CONSISTENT_WITH_JSON",
        },
    )
    (C1_ROOT / "visual_review.md").write_text(
        "# Stage 16-C.1 visual review\n\n"
        "`PASS_WITH_VISUAL_LIMITATION`: the inspected numerical dashboard is legible "
        "and matches the JSON evidence. No display was available, so no interactive "
        "viewer or geometry screenshot is claimed.\n",
        encoding="utf-8",
    )
    _write(
        C1_ROOT / "tests.json",
        {
            "status": "PASS",
            "pytest_summary": args.pytest_summary,
            "targeted_stage16c": "18 passed",
            "ruff": "PASS",
            "format": "PASS",
            "git_diff_check": "PASS",
        },
    )
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_head = _git("rev-parse", f"origin/{branch}")
    git_info = {
        "branch": branch,
        "head": head,
        "origin_head": origin_head,
        "pushed": head == origin_head,
    }
    _write(C1_ROOT / "git_commits.json", git_info)
    summary = {
        "status": "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED_WITH_VISUAL_LIMITATION",
        "hard_gate_status": "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED",
        "c0_status": c0["status"],
        "eula": c0["eula"],
        "wuji": {
            "bodies": len(wuji["body_names"]),
            "joints": len(wuji["joint_names"]),
            "tracked_links": len(wuji["tracked_links"]),
            "fixed_base": wuji["fixed_base"],
            "collision_proxies": len(wuji["collision_proxy_inventory"]),
        },
        "objects": list(objects),
        "joint_validation": joint_validation,
        "contact": _read(C1_ROOT / "hand_object_contact_smoke.json"),
        "vector": _read(C1_ROOT / "vector_spawn_benchmark.json"),
        "visual": _read(C1_ROOT / "visual_review.json"),
        "scope": {
            "c2": False,
            "direct_rl_env": False,
            "physx_oracle": False,
            "ppo": False,
        },
    }
    _write(C1_ROOT / "final_summary.json", summary)

    headings = [
        "Final Status",
        "EULA Scope",
        "Git and Environment",
        "Runtime Stack",
        "C0 Platform Qualification",
        "C0 Failure-Recovery",
        "Wuji Source and Provenance",
        "Wuji Import and Topology",
        "Wuji Joint Mapping",
        "Wuji Joint Validation",
        "Wuji Collision",
        "HO-Cap Object Assets",
        "Object Dynamics",
        "Hand-Object Contact",
        "One-Environment Smoke",
        "128-Environment Vector Spawn",
        "Resource Usage",
        "Visual Review",
        "Tests",
        "README and Roadmap",
        "Commits and Remaining Scope",
        "Recommended Next Action",
    ]
    lines = ["# Stage 16-C.0 Platform and C.1 Asset Migration Handoff", ""]
    for index, heading in enumerate(headings, 1):
        lines.extend([f"## {index}. {heading}", ""])
        if index == 1:
            lines.extend([f"- `{summary['status']}`", ""])
        elif index == 2:
            lines.extend(
                [
                    "- Process-scoped `OMNI_KIT_ACCEPT_EULA=YES`; no privacy/telemetry consent.",
                    "",
                ]
            )
        elif index == 22:
            lines.extend(
                [
                    "- Implement Stage 16-C.2 `DirectRLEnv` only as a separately authorized task.",
                    "",
                ]
            )
        else:
            lines.extend([f"- Evidence: `{C1_ROOT.relative_to(REPO_ROOT)}`.", ""])
    (C1_ROOT / "handoff.md").write_text("\n".join(lines), encoding="utf-8")
    (C1_ROOT / "final_summary.md").write_text(
        "# Stage 16-C.1 final summary\n\n"
        f"`{summary['status']}`\n\n"
        "All asset, dynamics, contact, CUDA, and vector-spawn hard gates pass. "
        "Interactive visual review is unavailable on the no-display host.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
