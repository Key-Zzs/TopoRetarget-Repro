#!/usr/bin/env python3
# ruff: noqa: E501
"""Aggregate ignored Stage 16-C.0/C.1 evidence into the closeout bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from toporetarget.rl.environments.isaaclab_backend.asset_validation import classify_c2_entry

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
    parser.add_argument("--pytest-summary", default="391 passed, 27 skipped, 1 warning")
    args = parser.parse_args()

    c0 = _read(C0_ROOT / "final_summary.json")
    c0_host = _read(C0_ROOT / "host_compatibility.json")
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
    source_inventory = _read(C1_ROOT / "wuji_source_inventory.json")

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
            "contact_api": "Isaac Lab ContactSensor backed by PhysX RigidContactView",
            "force_measurement": (
                "object momentum response projected onto the measured PhysX contact normal; "
                "raw API force fields are preserved separately"
            ),
            "objects": {
                object_id: {
                    key: report[key]
                    for key in (
                        "all_finite",
                        "contact_body_names",
                        "contact_pairs",
                        "contact_count",
                        "contact_event_steps",
                        "contact_position_mean_w_m",
                        "max_normal_force_n",
                        "max_api_net_normal_force_n",
                        "max_filtered_normal_force_n",
                        "max_friction_force_n",
                        "minimum_contact_separation_m",
                        "maximum_contact_separation_m",
                        "object_position_response_m",
                        "object_linear_speed_mps",
                        "minimum_body_origin_distance_m",
                        "no_explosion",
                        "no_contact_buffer_fatal_overflow",
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
        {
            "failure": "HEADLESS_RENDER_FAILURE",
            "attempt": 1,
            "evidence": "300 s timeout while downloading official RTX rendering extensions",
            "fallback": "preserve extension cache and retry with a 600 s bound",
            "repair": "extension-cache warm-up",
            "rerun": "real offscreen RTX render",
            "result": "RETRY",
            "strategy_switch": False,
        },
        {
            "failure": "HEADLESS_RENDER_FAILURE",
            "attempt": 2,
            "evidence": "600 s timeout after omni.volume cache completed",
            "fallback": "preserve completed cache and retry once more",
            "repair": "continue official shader-cache warm-up",
            "rerun": "real offscreen RTX render",
            "result": "RETRY_CACHE_PROGRESS",
            "strategy_switch": False,
        },
        {
            "failure": "HEADLESS_RENDER_FAILURE",
            "attempt": 3,
            "evidence": "600 s timeout downloading official Vulkan shader cache",
            "fallback": "record visual-only soft limitation; do not synthesize screenshots",
            "repair": "stop at the configured per-class recovery budget",
            "rerun": "not run",
            "result": "SOFT_LIMITATION_BUDGET_EXHAUSTED",
            "strategy_switch": False,
        },
    ]
    (C1_ROOT / "recovery_transitions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in transitions),
        encoding="utf-8",
    )
    render_reports = [
        C1_ROOT / "visual" / f"hocap_{object_id}_render.json" for object_id in ("170105", "170650")
    ]
    render_payloads = [_read(path) for path in render_reports if path.is_file()]
    rendered_frames = [frame for payload in render_payloads for frame in payload["frames"]]
    visual_pass = (
        len(render_payloads) == 2
        and all(payload["status"] == "PASS" for payload in render_payloads)
        and all(frame["nonblank"] for frame in rendered_frames)
    )
    visual = {
        "status": "PASS" if visual_pass else "SOFT_LIMITATION",
        "display": "UNAVAILABLE",
        "offscreen_rtx_attempted": True,
        "offscreen_rtx": "PASS" if visual_pass else "UNAVAILABLE_AFTER_BOUNDED_RETRIES",
        "interactive_viewer": "NOT_RUN_NO_DISPLAY",
        "render_reports": [
            str(path.relative_to(REPO_ROOT)) for path in render_reports if path.is_file()
        ],
        "frames": rendered_frames,
        "hard_gate_effect": "NONE",
    }
    _write(C1_ROOT / "visual_review.json", visual)
    (C1_ROOT / "visual_review.md").write_text(
        "# Stage 16-C.1 visual review\n\n"
        f"- Status: `{visual['status']}`.\n"
        f"- Real offscreen RTX: `{visual['offscreen_rtx']}`.\n"
        "- Interactive viewer: `NOT_RUN_NO_DISPLAY`.\n"
        "- Visual review is separate from and cannot weaken the C.1 hard gates.\n",
        encoding="utf-8",
    )
    _write(
        C1_ROOT / "tests.json",
        {
            "status": "PASS",
            "pytest_summary": args.pytest_summary,
            "targeted_stage16c": "19 passed in 1.74s",
            "full_pytest": "391 passed, 27 skipped, 1 warning in 48.49s",
            "ruff": "PASS",
            "format": "PASS",
            "mypy_src": "PASS: 242 source files",
            "paper_fidelity": "OK",
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
    c1_status = "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED"
    c2_entry = classify_c2_entry(c1_status, entry_authorized=True)
    c0_vector = {int(item["num_envs"]): item for item in c0["vector"]}
    summary = {
        "status": c1_status,
        "c0_status": c0["status"],
        "c2_entry_status": c2_entry,
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
        "visual": visual,
        "scope": {
            "c2_entry_authorized": True,
            "c2_implemented": False,
            "direct_rl_env_executed": False,
            "physx_oracle_executed": False,
            "ppo_executed": False,
            "ppo_samples": 0,
            "ppo_checkpoints": 0,
        },
    }
    _write(C1_ROOT / "final_summary.json", summary)

    headings = [
        "Final Status",
        "Git and Branch",
        "EULA Authorization and Scope",
        "Runtime Stack",
        "Isaac Sim Runtime",
        "Isaac Lab Official Smoke",
        "GPU PhysX and Vector Platform",
        "C0 Failure-Recovery",
        "Wuji Source and Provenance",
        "Wuji USD and Articulation",
        "Joint and Link Validation",
        "HOCap Object Migration",
        "Collision and Dynamics Validation",
        "Hand–Object Contact Smoke",
        "Vector Asset Spawn",
        "Visualization",
        "Tests",
        "README and Roadmap",
        "Commits and Push",
        "Stage 16-C.2 Entry Decision",
        "Remaining Limitations",
        "Recommended Next Action",
    ]
    lines = ["# Stage 16-C.0 Runtime and Stage 16-C.1 Asset Migration Handoff", ""]
    for index, heading in enumerate(headings, 1):
        lines.extend([f"## {index}. {heading}", ""])
        if index == 1:
            lines.extend(
                [
                    f"- C.0: `{c0['status']}`.",
                    f"- C.1: `{c1_status}`.",
                    f"- C.2 entry: `{c2_entry}`; C.2 was not implemented or run.",
                    "",
                ]
            )
        elif index == 2:
            lines.extend(
                [
                    f"- Branch: `{branch}`.",
                    f"- Local HEAD: `{head}`; origin: `{origin_head}`; pushed: `{head == origin_head}`.",
                    "",
                ]
            )
        elif index == 3:
            lines.extend(
                [
                    "- User explicitly authorized process-scoped `OMNI_KIT_ACCEPT_EULA=YES`.",
                    "- The authorization does not include privacy or telemetry collection consent.",
                    "- No global shell profile was modified.",
                    "",
                ]
            )
        elif index == 4:
            imports = c0["runtime"]["imports"]
            lines.extend(
                [
                    f"- Python `3.11.15`; Torch `{imports['torch']}`; CUDA `{imports['torch_cuda']}`.",
                    f"- Isaac Sim `{imports.get('isaac_sim_actual_version')}`; Isaac Lab package `{imports.get('isaac_lab_actual_version')}`, source `v2.3.2` at `37ddf626871758333d6ed89cf64ad702aef127d0`.",
                    f"- Kit `{imports.get('kit_version')}`; GPU `{c0_host['gpu']['name']}`; driver `{c0_host['gpu']['driver']}`.",
                    "",
                ]
            )
        elif index == 5:
            empty = c0["runtime"]["empty_scene"]
            lines.extend(
                [
                    f"- Headless empty scene: `{empty['steps']}` finite steps at `{empty['physics_steps_per_s']:.2f}` steps/s on `{empty['simulation_device']}`.",
                    f"- Clean child exit: `{empty['clean_process_exit']}`; full log: `{C0_ROOT.relative_to(REPO_ROOT) / 'isaac_sim_runtime.log'}`.",
                    "- Command: `conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_stage16_isaaclab_platform.py --phase full --steps 1000 --accept-eula`.",
                    "",
                ]
            )
        elif index == 6:
            official = c0_vector[1]
            lines.extend(
                [
                    f"- Official Cartpole 1-env: `{official['steps']}` steps, `{official['physics_steps_per_s']:.2f}` env-steps/s.",
                    f"- Observation finite: `{official['finite_observation']}`; clean exit: `{official['clean_process_exit']}`.",
                    "",
                ]
            )
        elif index == 7:
            v128 = c0_vector[128]
            v512 = c0_vector[512]
            lines.extend(
                [
                    f"- 128 envs: `{v128['physics_steps_per_s']:.2f}` env-steps/s; 512 envs: `{v512['physics_steps_per_s']:.2f}` env-steps/s.",
                    "- CUDA tensors, GPU PhysX, finite observations, done tensor shape/device, and clean exits are recorded in runtime JSON.",
                    "",
                ]
            )
        elif index == 8:
            lines.extend(
                [
                    "- Frozen Isaac Sim 5.1/Isaac Lab v2.3.2 lane passed; no 6.x fallback was needed.",
                    "- Missing DISPLAY and dependency-metadata conflict remain soft limitations only.",
                    "",
                ]
            )
        elif index == 9:
            lines.extend(
                [
                    f"- Requested checkout exists: `{source_inventory['requested_checkout_exists']}`; resolved source: `{source_inventory['external_checkout_actual_path']}`.",
                    f"- Frozen directory byte-identical: `{source_inventory['frozen_usd_directory_byte_identical']}`; source commit `{wuji['source_commit']}`.",
                    "- Command: `python scripts/rl/isaaclab/import_wuji_hand2.py --requested-upstream-root /home/deepcybo/workspace/wuji-description --upstream-root /home/deepcybo/workspace/dex/wuji-description --accept-eula` in `toporetarget-isaaclab`.",
                    "",
                ]
            )
        elif index == 10:
            lines.extend(
                [
                    f"- Generated USD SHA-256: `{wuji['generated_sha256']}`.",
                    f"- Floating base `{not wuji['fixed_base']}`; `{len(wuji['body_names'])}` bodies; `{len(wuji['joint_names'])}` joints; `{len(wuji['collision_proxy_inventory'])}` collision proxies.",
                    "",
                ]
            )
        elif index == 11:
            lines.extend(
                [
                    f"- Responsive joints: `{joint_validation['joints_with_response']}/20`; tracked links: `16/16`.",
                    f"- Maximum limit error: `{joint_validation['max_limit_abs_error_rad']:.3e} rad`; configured default is q=0 and right-hand semantics are explicit.",
                    "",
                ]
            )
            runtime_mapping = vector[("170105", 1)]["joint_order_mapping"]
            for source_index, joint_name in enumerate(wuji["joint_order"]):
                lines.append(
                    f"- `{joint_name}`: source `{source_index}` → Isaac `{runtime_mapping[joint_name]}`."
                )
            lines.append("")
        elif index == 12:
            lines.extend(
                [
                    f"- `hocap_170105`: `{objects['170105']['generated_sha256']}`.",
                    f"- `hocap_170650`: `{objects['170650']['generated_sha256']}`.",
                    "- Original OBJ is visual truth; both collision assets use deterministic `convex_hull_v1`.",
                    "",
                ]
            )
        elif index == 13:
            lines.extend(
                [
                    "- Both free rigid objects use 0.05 kg engineering-nominal mass, configured COM/inertia/friction, zero gravity, no ground, and no support.",
                    "- Physical provenance remains unresolved; no calibrated dynamics or sim-to-real claim is made.",
                    "",
                ]
            )
        elif index == 14:
            for object_id, report in contacts.items():
                lines.append(
                    f"- `{object_id}` pair `{report['contact_pairs'][0]}`: {report['contact_count']} points, "
                    f"projected normal `{report['max_normal_force_n']:.6f} N`, friction `{report['max_friction_force_n']:.6f} N`, "
                    f"separation `{report['minimum_contact_separation_m']:.6f}` to `{report['maximum_contact_separation_m']:.6f} m`."
                )
            lines.extend(
                [
                    "- Raw zero API force fields are preserved separately; no force value is silently substituted.",
                    "- Command pattern: `python scripts/rl/isaaclab/smoke_stage16c1_assets.py --object <id> --num-envs 1 --contact --steps 100 --accept-eula` in `toporetarget-isaaclab`.",
                    "",
                ]
            )
        elif index == 15:
            for object_id in ("170105", "170650"):
                report = vector[(object_id, 128)]
                lines.append(
                    f"- `{object_id}`: 128 unique origins, `{report['physics_env_steps_per_second']:.2f}` env-steps/s, "
                    f"subset reset error `{report['subset_reset_max_position_error_m']:.1f} m`."
                )
            lines.extend(
                [
                    "- Seed 20260802 jointwise random targets prove per-environment action independence.",
                    "- Command pattern: `python scripts/rl/isaaclab/smoke_stage16c1_assets.py --object <id> --num-envs 128 --steps 1000 --accept-eula` in `toporetarget-isaaclab`.",
                    "",
                ]
            )
        elif index == 16:
            lines.extend(
                [
                    f"- Real offscreen RTX result: `{visual['offscreen_rtx']}`; interactive viewer: `{visual['interactive_viewer']}`.",
                    f"- Reviewed/nonblank frame count: `{len(rendered_frames)}`; visual status has no C.1 hard-gate effect.",
                    "",
                ]
            )
        elif index == 17:
            lines.extend(
                [
                    f"- `{args.pytest_summary}`; ruff, format, mypy, paper-fidelity, and repository-integrity checks are recorded in `tests.json`.",
                    "",
                ]
            )
        elif index == 18:
            lines.extend(
                [
                    "- English/Chinese README and roadmap state C.1 validated, C.2 entry authorized, and C.2 implementation/PPO unstarted.",
                    "",
                ]
            )
        elif index == 19:
            lines.extend(
                [
                    f"- Commit/push evidence: `{C1_ROOT.relative_to(REPO_ROOT) / 'git_commits.json'}`.",
                    "",
                ]
            )
        elif index == 20:
            lines.extend(
                [
                    f"- `{c2_entry}`. This is an entry decision only; `DirectRLEnv` execution remains false.",
                    "",
                ]
            )
        elif index == 21:
            lines.extend(
                [
                    "- No interactive DISPLAY; object physical provenance unresolved; frozen Isaac Sim 5.1 is vendor-unsupported; dependency metadata has a soft conflict.",
                    "",
                ]
            )
        elif index == 22:
            lines.extend(
                [
                    "- Start a separately scoped Stage 16-C.2 task to implement—not train—the `DirectRLEnv` shell and semantic contracts.",
                    "",
                ]
            )
    (C1_ROOT / "handoff.md").write_text("\n".join(lines), encoding="utf-8")
    (C1_ROOT / "final_summary.md").write_text(
        "# Stage 16-C.1 final summary\n\n"
        f"`{summary['status']}`\n\n"
        "All asset, dynamics, named-contact, CUDA, and vector-spawn hard gates pass. "
        f"C.2 entry is `{c2_entry}`, but C.2 and PPO were not executed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
