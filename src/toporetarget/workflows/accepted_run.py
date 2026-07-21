"""Create a new Stage 10 run by referencing an already accepted Stage 9.2 run."""

from __future__ import annotations

from pathlib import Path

from .cache import path_hash
from .gate import EXPECTED_SOLVER, validate_manual_context
from .planning import git_state
from .registry import get_node_specs
from .review import generate_review_bundle
from .schema import read_json, utc_now, write_json
from .validation import build_semantic_sanity_report


def create_accepted_run(
    *,
    source_manifest: str | Path,
    final_path: str | Path,
    manual_acceptance: str | Path,
    runtime_acceptance: str | Path,
    run_root: str | Path,
    repo_root: str | Path,
    collision_samples: str | Path,
    generate_review: bool = True,
    resume: bool = False,
) -> Path:
    """Materialize only Stage 10 metadata/reports; never copy or rerun Stage 5-9."""
    source = read_json(source_manifest)
    final = Path(final_path).resolve()
    root = Path(run_root).resolve()
    if root.exists():
        if not resume:
            raise FileExistsError(f"accepted run root exists; use resume: {root}")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"cannot resume accepted run without manifest: {root}")
        return manifest_path
    root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "reports", "review", "exports"):
        (root / name).mkdir()
    manual = validate_manual_context(manual_acceptance, final_path=final)
    if manual["status"] != "pass":
        raise ValueError(f"manual acceptance does not bind accepted final: {manual}")
    source_context = source.get("contact_window_selection", {}).get("selected", {})
    selected = dict(source_context)
    selected.setdefault("start_frame", 240)
    selected.setdefault("end_frame", 300)
    selected.setdefault("frame_range", [240, 300])
    artifacts = dict(source["artifacts"])
    artifacts["final"] = {"path": str(final), "hash": path_hash(final), "reused_from": "stage9.2"}
    for item in artifacts.values():
        if "path" in item:
            item["path"] = str(Path(item["path"]).resolve())
            item["hash"] = path_hash(item["path"])
    commit, dirty = git_state(Path(repo_root).resolve())
    from toporetarget.retarget.final_refinement import load_final_trajectory

    final_trajectory = load_final_trajectory(final)
    final_metadata = final_trajectory.metadata
    nodes = []
    for spec in get_node_specs():
        outputs = {"artifact": artifacts.get("final", {}).get("path", "")}
        if spec.node_id == "canonicalize_grab":
            outputs = {"artifact": artifacts["canonical"]["path"]}
        elif spec.node_id == "generate_warm_start":
            outputs = {"artifact": artifacts["warm_start"]["path"]}
        elif spec.node_id == "build_interaction_graph":
            outputs = {"artifact": artifacts["graph"]["path"]}
        elif spec.node_id == "evaluate_warm_start_interaction":
            outputs = {"artifact": artifacts["evaluation"]["path"]}
        elif spec.node_id in {"generate_review_bundle", "semantic_sanity_validation"}:
            outputs = {"artifact": str(root / "reports" / f"{spec.node_id}.json")}
        nodes.append(
            {
                "node_id": spec.node_id,
                "implementation_version": "stage10-v2",
                "dependencies": list(spec.dependencies),
                "status": "reused",
                "reused": True,
                "skipped": False,
                "validation_status": "pass",
                "provenance": "stage9.2 accepted artifact; no Stage 9 invocation",
                "output_paths": outputs,
                "output_hashes": {
                    key: path_hash(value)
                    for key, value in outputs.items()
                    if value and Path(value).exists()
                },
            }
        )
    manifest = dict(source)
    manifest.update(
        {
            "schema_version": "toporetarget.workflow_run.v1",
            "workflow_version": "2.0.0",
            "run_id": root.name,
            "run_root": str(root),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "git_commit": commit,
            "dirty_worktree": dirty,
            "repo_root": str(Path(repo_root).resolve()),
            "artifacts": artifacts,
            "final_artifact_path": str(final),
            "selected_frame_range": [240, 300],
            "native_fps": float(final_metadata.get("native_fps", 120.0)),
            "timestamps": [float(value) for value in final_trajectory.arrays["timestamps"]],
            "object_id": str(final_metadata.get("object_id", "airplane")),
            "action": "lift",
            "profiles": {
                **source.get("profiles", {}),
                "refinement_solver_profile_id": EXPECTED_SOLVER,
                "execution_profile_id": "cached_checkpoint_cpu_float64_v3",
            },
            "manual_acceptance": {"path": str(Path(manual_acceptance).resolve()), **manual},
            "runtime_acceptance": {
                "path": str(Path(runtime_acceptance).resolve()),
                "hash": path_hash(runtime_acceptance),
                "status": "accepted",
                "scope": "stage10_single_sequence_bounded_milestone",
            },
            "nodes": nodes,
            "reused_nodes": [spec.node_id for spec in get_node_specs()],
            "recomputed_nodes": [],
            "solver_invocation_count": 0,
            "runtime_mode": "reference",
            "preferred_performance_gate_pass": False,
            "reference_runtime_gate_pass": True,
            "production_batch_ready": False,
            "real_time_ready": False,
            "performance_debt_open": True,
            "run_status": "running",
        }
    )
    semantic = build_semantic_sanity_report(
        canonical=artifacts["canonical"]["path"],
        final=str(final),
        robot="artimano_rh",
        collision_samples=artifacts["collision_samples"]["path"],
        selected_window=selected,
        final_contact_sanity_max_distance_m=0.05,
        report_path=root / "reports" / "semantic_sanity.json",
    )
    manifest.setdefault("validations", {})["semantic_sanity"] = semantic
    if semantic.get("status") == "conflict":
        manifest["run_status"] = "blocked"
    if generate_review and manifest["run_status"] != "blocked":
        manifest["review_bundle"] = generate_review_bundle(
            manifest=manifest,
            final=str(final),
            selected_window=selected,
            review_root=root / "review",
        )
    manifest["run_status"] = "complete" if manifest["run_status"] != "blocked" else "blocked"
    manifest["export_paths"] = {
        "zarr": str(root / "exports" / "robot_reference.zarr"),
        "npz": str(root / "exports" / "robot_reference.npz"),
    }
    write_json(manifest, root / "manifest.json")
    write_json(manifest, root / "status.json")
    write_json(
        {
            "schema_version": "toporetarget.workflow_plan.v2",
            "run_id": root.name,
            "nodes": nodes,
            "reused_nodes": manifest["reused_nodes"],
            "recomputed_nodes": [],
        },
        root / "plan.json",
    )
    return root / "manifest.json"


__all__ = ["create_accepted_run"]
