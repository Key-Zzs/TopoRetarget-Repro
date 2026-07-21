"""Bounded Stage 10 executor composed from the existing Stage 5-9 CLIs."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .cache import artifact_hashes, cache_record, can_reuse, path_hash, signature_payload
from .contact_window import select_contact_windows
from .planning import (
    STAGE9_DEPENDENT_NODES,
    build_plan,
    git_state,
    node_profile_hashes,
    profile_hashes,
    run_paths,
)
from .registry import get_node_specs
from .reports import (
    build_input_audit,
    stage9_window_geometry_audit,
    write_execution_reports,
)
from .review import generate_review_bundle
from .schema import (
    WorkflowRequest,
    WorkflowRunManifest,
    stable_hash,
    utc_now,
    write_json,
)
from .validation import (
    build_semantic_sanity_report,
    cross_stage_identity_report,
    environment_snapshot,
    validate_manual_acceptance,
)


class WorkflowExecutionError(RuntimeError):
    """Raised when one workflow node fails or is blocked."""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_cli(command: list[str], *, repo_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    _write_text(
        log_path, "$ " + _command_string(command) + "\n\n" + result.stdout + "\n" + result.stderr
    )
    if result.returncode != 0:
        raise WorkflowExecutionError(
            "command failed "
            f"({result.returncode}): {_command_string(command)}\n{result.stderr[-2000:]}"
        )


def _node_record_path(run_root: Path, node_id: str) -> Path:
    return run_root / "cache" / f"{node_id}.json"


def _manual_acceptance_path(repo_root: Path) -> Path:
    return repo_root / ".local" / "reports" / "stage9" / "manual_acceptance.json"


def _selection_paths(repo_root: Path) -> tuple[Path, Path]:
    root = repo_root / ".local" / "reports" / "stage10"
    return root / "contact_window_candidates.json", root / "contact_window_selection.json"


def _source_integrity_snapshot(manifest: WorkflowRunManifest) -> dict[str, Any]:
    """Record raw-source integrity even when a downstream node fails."""

    if not manifest.source_path or manifest.source_hash is None:
        return {}
    try:
        source_hash_after = path_hash(manifest.source_path)
    except (OSError, ValueError) as exc:
        return {
            "source_path": manifest.source_path,
            "source_hash_before": manifest.source_hash,
            "source_hash_after": None,
            "source_integrity_check": "unknown",
            "raw_source_not_written_by_workflow": None,
            "error": str(exc),
        }
    source_unchanged = source_hash_after == manifest.source_hash
    return {
        "source_path": manifest.source_path,
        "source_hash_before": manifest.source_hash,
        "source_hash_after": source_hash_after,
        "source_integrity_check": "pass" if source_unchanged else "fail",
        "raw_source_not_written_by_workflow": source_unchanged,
    }


def _request_payload(request: WorkflowRequest, selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": request.sequence,
        "index": str(request.index),
        "hand": request.hand,
        "robot": request.robot,
        "refinement_solver_profile": request.refinement_solver_profile,
        "selected_window": selected,
        "window_length": request.window_length,
        "thresholds": {
            "minimum_contact_frame_ratio": request.minimum_contact_frame_ratio,
            "maximum_source_contact_median_distance_m": (
                request.maximum_source_contact_median_distance_m
            ),
            "final_contact_sanity_max_distance_m": request.final_contact_sanity_max_distance_m,
        },
        "mano_model_root": None
        if request.mano_model_root is None
        else str(request.mano_model_root),
        "asset_root": None if request.asset_root is None else str(request.asset_root),
    }


def _selected_request(request: WorkflowRequest, selected: dict[str, Any]) -> WorkflowRequest:
    return replace(
        request,
        start_frame=int(selected["start_frame"]),
        end_frame=int(selected["end_frame"]),
        auto_contact_window=False,
    )


def _outputs_for_node(node_id: str, paths: dict[str, str]) -> dict[str, str]:
    reports = Path(paths["reports"])
    values = {
        "resolve_source": {"report": str(reports / "source.json")},
        "canonicalize_grab": {"canonical": paths["canonical"]},
        "validate_canonical": {"report": str(reports / "canonical_validation.json")},
        "enrich_mediapipe21": {"report": str(reports / "mediapipe21.json")},
        "validate_keypoints": {"report": str(reports / "keypoint_validation.json")},
        "audit_object_mesh": {"report": str(reports / "object_mesh_audit.json")},
        "sample_object_surface": {"samples": paths["object_samples"]},
        "validate_object_samples": {"report": str(reports / "object_samples_validation.json")},
        "generate_warm_start": {"warm_start": paths["warm_start"]},
        "validate_warm_start": {"report": str(reports / "warm_start_validation.json")},
        "build_interaction_graph": {"graph": paths["graph"]},
        "validate_interaction_graph": {
            "report": str(reports / "interaction_graph_validation.json")
        },
        "evaluate_warm_start_interaction": {"evaluation": paths["evaluation"]},
        "final_refinement": {"final": paths["final"]},
        "validate_final_refinement": {"report": str(reports / "final_validation.json")},
        "full_surface_penetration_audit": {"report": str(reports / "penetration_audit.json")},
        "semantic_sanity_validation": {"report": str(reports / "semantic_sanity.json")},
        "generate_review_bundle": {"review": paths["review"]},
        "write_run_manifest": {"manifest": paths["manifest"]},
    }
    return values[node_id]


def _validation_path(node_id: str, outputs: dict[str, str]) -> str | None:
    if "report" in outputs:
        return outputs["report"]
    return None


def _node_signature(
    node_id: str,
    request: WorkflowRequest,
    selected: dict[str, Any],
    dependency_hashes: dict[str, str],
    configs: dict[str, str],
) -> str:
    spec = next(item for item in get_node_specs() if item.node_id == node_id)
    request_payload = _request_payload(request, selected)
    if node_id not in STAGE9_DEPENDENT_NODES:
        request_payload.pop("refinement_solver_profile", None)
    payload = signature_payload(
        node_id=node_id,
        implementation_version=spec.implementation_version,
        inputs={"request": stable_hash(request_payload), **dependency_hashes},
        configs=node_profile_hashes(node_id, configs),
        parameters={"selected_window": selected},
    )
    return str(payload["signature"])


def _status_payload(manifest: WorkflowRunManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "status": manifest.run_status,
        "nodes": [node.as_dict() for node in manifest.nodes],
        "pending_human_acceptance": manifest.run_status == "pending_human_acceptance",
        "updated_at": manifest.updated_at,
    }


def _write_status(manifest: WorkflowRunManifest, paths: dict[str, str]) -> None:
    manifest.updated_at = utc_now()
    write_json(_status_payload(manifest), paths["status"])


def _run_node_command(
    node_id: str,
    *,
    request: WorkflowRequest,
    selected: dict[str, Any],
    paths: dict[str, str],
    log_path: Path,
    force_output: bool,
) -> None:
    py = sys.executable
    repo = request.repo_root.resolve()
    canonical = paths["canonical"]
    mano = request.mano_model_root
    asset = request.asset_root
    start = int(selected["start_frame"])
    end = int(selected["end_frame"])
    side = request.hand
    commands: dict[str, list[str]] = {
        "canonicalize_grab": [
            py,
            "-m",
            "toporetarget",
            "data",
            "convert",
            "--dataset",
            "grab",
            "--sequence",
            request.sequence,
            "--index",
            str(request.index),
            "--hands",
            side,
            "--contact-mode",
            "semantic",
            "--include-mediapipe21",
            "--start-frame",
            str(start),
            "--end-frame",
            str(end),
            "--output",
            canonical,
        ],
        "validate_canonical": [
            py,
            "-m",
            "toporetarget",
            "data",
            "validate",
            "--dataset",
            "grab",
            "--sequence",
            request.sequence,
            "--index",
            str(request.index),
            "--hands",
            side,
            "--contact-mode",
            "semantic",
            "--start-frame",
            str(start),
            "--end-frame",
            str(end),
            "--canonical",
            canonical,
            "--report",
            str(Path(paths["reports"]) / "canonical_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "canonical_validation.csv"),
        ],
        "validate_keypoints": [
            py,
            "-m",
            "toporetarget",
            "keypoints",
            "validate",
            "--input",
            canonical,
            "--hand",
            side,
            "--layout",
            "mediapipe21",
            "--report",
            str(Path(paths["reports"]) / "keypoint_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "keypoint_validation.csv"),
        ],
        "audit_object_mesh": [
            py,
            "-m",
            "toporetarget",
            "geometry",
            "inspect-mesh",
            "--canonical",
            canonical,
            "--object-id",
            "primary",
            "--json",
            str(Path(paths["reports"]) / "object_mesh_audit.json"),
        ],
        "sample_object_surface": [
            py,
            "-m",
            "toporetarget",
            "geometry",
            "sample-object",
            "--canonical",
            canonical,
            "--object-id",
            "primary",
            "--profile",
            "paper_strict_area_uniform",
            "--output",
            paths["object_samples"],
            "--report",
            str(Path(paths["reports"]) / "object_samples.json"),
        ],
        "validate_object_samples": [
            py,
            "-m",
            "toporetarget",
            "geometry",
            "validate-samples",
            "--samples",
            paths["object_samples"],
            "--canonical",
            canonical,
            "--object-id",
            "primary",
            "--report",
            str(Path(paths["reports"]) / "object_samples_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "object_samples_validation.csv"),
        ],
        "generate_warm_start": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "warm-start",
            "--canonical",
            canonical,
            "--hand",
            side,
            "--robot",
            request.robot,
            "--start-frame",
            "0",
            "--end-frame",
            str(end - start),
            "--frame-profile",
            "canonical_keypoint_wrist_v1",
            "--bone-profile",
            "mediapipe21_full_finger_chain_v1",
            "--solver-profile",
            "paper_repro_scipy_trf",
            "--output",
            paths["warm_start"],
        ],
        "validate_warm_start": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "validate-warm-start",
            "--canonical",
            canonical,
            "--warm-start",
            paths["warm_start"],
            "--report",
            str(Path(paths["reports"]) / "warm_start_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "warm_start_validation.csv"),
        ],
        "build_interaction_graph": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "build-interaction-graph",
            "--canonical",
            canonical,
            "--hand",
            side,
            "--object-samples",
            paths["object_samples"],
            "--delaunay-profile",
            "strict_scipy_qhull_v1",
            "--start-frame",
            "0",
            "--end-frame",
            str(end - start),
            "--output",
            paths["graph"],
            "--report",
            str(Path(paths["reports"]) / "interaction_graph_build.json"),
        ],
        "validate_interaction_graph": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "validate-interaction-graph",
            "--canonical",
            canonical,
            "--object-samples",
            paths["object_samples"],
            "--graph",
            paths["graph"],
            "--report",
            str(Path(paths["reports"]) / "interaction_graph_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "interaction_graph_validation.csv"),
        ],
        "evaluate_warm_start_interaction": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "evaluate-interaction",
            "--graph",
            paths["graph"],
            "--warm-start",
            paths["warm_start"],
            "--robot",
            request.robot,
            "--output",
            paths["evaluation"],
        ],
        "final_refinement": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "refine",
            "--canonical",
            canonical,
            "--warm-start",
            paths["warm_start"],
            "--graph",
            paths["graph"],
            "--robot",
            request.robot,
            "--collision-samples",
            str(
                repo
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{request.robot}_neutral.npz"
            ),
            "--query-profile",
            "adaptive_active_set_v1",
            "--coordinate-profile",
            "local_seed_delta_v1",
            "--solver-profile",
            request.refinement_solver_profile,
            "--start-frame",
            "0",
            "--end-frame",
            str(end - start),
            "--output",
            paths["final"],
        ],
        "validate_final_refinement": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "validate-refinement",
            "--canonical",
            canonical,
            "--warm-start",
            paths["warm_start"],
            "--graph",
            paths["graph"],
            "--final",
            paths["final"],
            "--robot",
            request.robot,
            "--collision-samples",
            str(
                repo
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{request.robot}_neutral.npz"
            ),
            "--report",
            str(Path(paths["reports"]) / "final_validation.json"),
            "--csv",
            str(Path(paths["reports"]) / "final_validation.csv"),
        ],
        "full_surface_penetration_audit": [
            py,
            "-m",
            "toporetarget",
            "retarget",
            "audit-penetration",
            "--canonical",
            canonical,
            "--warm-start",
            paths["warm_start"],
            "--final",
            paths["final"],
            "--robot",
            request.robot,
            "--collision-samples",
            str(
                repo
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{request.robot}_neutral.npz"
            ),
            "--report",
            str(Path(paths["reports"]) / "penetration_audit.json"),
            "--csv",
            str(Path(paths["reports"]) / "penetration_audit.csv"),
        ],
    }
    command = commands.get(node_id)
    if command is None:
        raise WorkflowExecutionError(f"node has no CLI command: {node_id}")
    if mano is not None and "--mano-model-root" in command:
        command += ["--mano-model-root", str(mano)]
    elif mano is not None and node_id in {"canonicalize_grab", "validate_canonical"}:
        command += ["--mano-model-root", str(mano)]
    if asset is not None and node_id in {
        "generate_warm_start",
        "validate_warm_start",
        "evaluate_warm_start_interaction",
        "final_refinement",
        "validate_final_refinement",
        "full_surface_penetration_audit",
    }:
        command += ["--asset-root", str(asset)]
    if force_output and node_id in {
        "canonicalize_grab",
        "sample_object_surface",
        "generate_warm_start",
        "build_interaction_graph",
        "evaluate_warm_start_interaction",
        "final_refinement",
    }:
        command += ["--force"]
    _run_cli(command, repo_root=repo, log_path=log_path)


def _write_stage10_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(payload, path)


def _report_status(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        import json

        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "invalid"
    if not isinstance(payload, dict):
        return "invalid"
    status = payload.get("status")
    if status is None and "pass" in payload:
        status = payload["pass"]
    if status is None and "all_frames_valid" in payload:
        status = payload["all_frames_valid"]
    if status is True:
        return "pass"
    if status is False:
        return "fail"
    return None if status is None else str(status)


def _assert_node_report_pass(node_id: str, outputs: dict[str, str]) -> None:
    report = outputs.get("report")
    if report is None:
        return
    status = _report_status(report)
    if status in {"fail", "False", "invalid"}:
        raise WorkflowExecutionError(f"{node_id} validation report failed: {report}")


def _execute_special_node(
    node_id: str,
    *,
    request: WorkflowRequest,
    selected: dict[str, Any],
    paths: dict[str, str],
    manifest: WorkflowRunManifest,
) -> None:
    repo = request.repo_root.resolve()
    reports = Path(paths["reports"])
    if node_id == "resolve_source":
        from .contact_window import resolve_index_sequence

        entry, source_path = resolve_index_sequence(request.index, request.sequence)
        payload = {
            "status": "pass",
            "sequence": request.sequence,
            "source_path": str(source_path),
            "source_hash": path_hash(source_path),
            "index_entry": entry,
            "selected_frame_range": [selected["start_frame"], selected["end_frame"]],
            "no_external_write": True,
        }
        _write_stage10_report(reports / "source.json", payload)
        manifest.source_path = str(source_path)
        manifest.source_hash = str(payload["source_hash"])
        manifest.source_sequence = request.sequence
        manifest.subject = entry.get("subject_id")
        manifest.object_id = entry.get("object_token")
        manifest.action = entry.get("action_token")
    elif node_id == "enrich_mediapipe21":
        payload = {
            "status": "pass",
            "stage": "Stage 3 MediaPipe-style 21 enrichment",
            "canonical": paths["canonical"],
            "artifact_hash": path_hash(paths["canonical"]),
            "mapping_profile": "mano_v1_2_smplx_to_mediapipe21",
            "native_timestamps_preserved": True,
            "note": (
                "Stage 5 conversion requested include_mediapipe21; no new mapping is computed here."
            ),
        }
        _write_stage10_report(reports / "mediapipe21.json", payload)
    elif node_id == "semantic_sanity_validation":
        final_report = build_semantic_sanity_report(
            canonical=paths["canonical"],
            final=paths["final"],
            robot=request.robot,
            collision_samples=str(
                repo
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{request.robot}_neutral.npz"
            ),
            selected_window=selected,
            final_contact_sanity_max_distance_m=request.final_contact_sanity_max_distance_m,
            report_path=reports / "semantic_sanity.json",
        )
        manifest.validations["semantic_sanity"] = final_report
    elif node_id == "generate_review_bundle":
        review_manifest = manifest.as_dict()
        review = generate_review_bundle(
            manifest=review_manifest,
            final=paths["final"],
            selected_window=selected,
            review_root=paths["review"],
        )
        manifest.review_bundle = review
    elif node_id == "write_run_manifest":
        manifest.final_artifact_path = paths["final"]
        manifest.artifacts.setdefault("final", {})["hash"] = path_hash(paths["final"])
        write_json(manifest, paths["manifest"])
    else:
        raise WorkflowExecutionError(f"unknown special node: {node_id}")


def run_workflow(
    request: WorkflowRequest,
    *,
    resume: bool = False,
    validate: bool = True,
    generate_review: bool = True,
    force_stage: str | None = None,
    force_output: bool = False,
    manual_acceptance: str | Path | None = None,
) -> Path:
    """Execute one selected bounded clip and return its manifest path."""

    request = replace(
        request, repo_root=request.repo_root.resolve(), run_root=request.run_root.resolve()
    )
    request.validate()
    manual = (
        Path(manual_acceptance)
        if manual_acceptance is not None
        else _manual_acceptance_path(request.repo_root)
    )
    manual_payload = validate_manual_acceptance(manual)
    if manual_payload["current_window_interpretation"] == "invalid":
        raise WorkflowExecutionError("Stage 9 manual acceptance is invalid")
    selection = select_contact_windows(request, mano_model_root=request.mano_model_root)
    candidates_path, selection_path = _selection_paths(request.repo_root)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "candidates": selection.get("candidates", []),
            "selection_hash": selection.get("selection_hash"),
        },
        candidates_path,
    )
    write_json(selection, selection_path)
    if selection.get("status") != "pass" or selection.get("selected") is None:
        raise WorkflowExecutionError(
            "no contact-rich candidate passed the deterministic selection gates"
        )
    selected = dict(selection["selected"])
    selected["contact_frames"] = [int(value) for value in selected.get("contact_frames", [])]
    selected["contact_counts"] = {
        str(key): int(value) for key, value in selected.get("contact_counts", {}).items()
    }
    window_geometry_audit = stage9_window_geometry_audit(
        repo_root=request.repo_root, request=request, selected=selected
    )
    if window_geometry_audit.get("status") == "reject":
        raise WorkflowExecutionError(
            "selected Stage 10 window is not classified contact_rich by the Stage 9 geometry audit"
        )
    selected_request = _selected_request(request, selected)
    selected_profile_hashes = profile_hashes(
        selected_request.repo_root, selected_request.refinement_solver_profile
    )
    identifier = request.run_root / (
        request.sequence.replace("/", "__")
        + (
            f"__{request.hand}__{request.robot}__f{selected_request.start_frame:06d}"
            f"_f{selected_request.end_frame:06d}"
        )
    )
    selected_request = replace(selected_request, run_root=request.run_root)
    plan = build_plan(selected_request, selected_window=selected, run_id=identifier.name)
    paths = run_paths(selected_request, run_id=identifier.name)
    root = Path(paths["run_root"])
    for name in ("logs", "reports", "review", "exports", "artifacts", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    write_json(plan, paths["plan"])
    write_json(
        plan,
        selected_request.repo_root / ".local" / "reports" / "stage10" / "workflow_plan.json",
    )
    write_json(
        build_input_audit(
            request=selected_request,
            selection=selection,
            manual_acceptance=manual,
            profile_hashes=selected_profile_hashes,
        ),
        selected_request.repo_root / ".local" / "reports" / "stage10" / "input_audit.json",
    )
    commit, dirty = git_state(selected_request.repo_root)
    manifest = WorkflowRunManifest(
        run_id=plan.run_id,
        run_root=paths["run_root"],
        git_commit=commit,
        dirty_worktree=dirty,
        repo_root=str(selected_request.repo_root),
        environment=environment_snapshot(selected_request.repo_root),
        index_path=str(selected_request.index.resolve()),
        index_hash=path_hash(selected_request.index / "index.jsonl"),
        source_sequence=selected_request.sequence,
        mano_model_root=None
        if selected_request.mano_model_root is None
        else str(selected_request.mano_model_root.resolve()),
        asset_root=None
        if selected_request.asset_root is None
        else str(selected_request.asset_root.resolve()),
        hand=selected_request.hand,
        robot=selected_request.robot,
        selected_frame_range=[selected["start_frame"], selected["end_frame"]],
        contact_window_selection=selection,
        stage9_window_geometry_audit=window_geometry_audit,
        semantic_contact_statistics=selected,
        source_contact_geometry_sanity={
            "status": selected.get("source_geometry_status"),
            "median_distance_m": selected.get("source_contact_median_distance_m"),
            "min_distance_m": selected.get("source_contact_min_distance_m"),
        },
        profiles=selected_profile_hashes
        | {"refinement_solver_profile_id": selected_request.refinement_solver_profile},
        config_hashes=selected_profile_hashes,
        manual_acceptance={"path": str(manual), **manual_payload},
        assumptions=plan.assumptions,
        run_status="running",
    )
    collision_path = str(
        selected_request.repo_root
        / ".local"
        / "cache"
        / "geometry"
        / "robot_surface"
        / f"{selected_request.robot}_neutral.npz"
    )
    manifest.artifacts = {
        "canonical": {"path": paths["canonical"]},
        "object_samples": {"path": paths["object_samples"]},
        "warm_start": {"path": paths["warm_start"]},
        "graph": {"path": paths["graph"]},
        "evaluation": {"path": paths["evaluation"]},
        "final": {"path": paths["final"]},
        "collision_samples": {"path": collision_path},
    }
    manifest.nodes = plan.nodes
    manifest_path = Path(paths["manifest"])
    _write_status(manifest, paths)
    node_ids = [item.node_id for item in get_node_specs()]
    force_triggered = False
    output_hashes: dict[str, str] = {}
    configs = selected_profile_hashes
    skipped_nodes: set[str] = set()
    execution_started = time.perf_counter()
    if force_stage is not None and force_stage not in node_ids:
        raise WorkflowExecutionError(f"unknown workflow node for --force-stage: {force_stage}")
    if not validate:
        skipped_nodes.update(
            {
                "validate_canonical",
                "validate_keypoints",
                "validate_object_samples",
                "validate_warm_start",
                "validate_interaction_graph",
                "validate_final_refinement",
                "full_surface_penetration_audit",
                "semantic_sanity_validation",
            }
        )
    if not generate_review:
        skipped_nodes.add("generate_review_bundle")
    for node_id in node_ids:
        if force_stage == node_id:
            force_triggered = True
        node = next(item for item in manifest.nodes if item.node_id == node_id)
        node_outputs = _outputs_for_node(node_id, paths)
        dependency_hashes = {
            f"dependency:{dependency}": output_hashes.get(dependency, "")
            for dependency in node.dependencies
        }
        expected = _node_signature(node_id, selected_request, selected, dependency_hashes, configs)
        node.expected_signature = expected
        request_payload = _request_payload(selected_request, selected)
        if node_id not in STAGE9_DEPENDENT_NODES:
            request_payload.pop("refinement_solver_profile", None)
        node.input_hashes = {
            "request": stable_hash(request_payload),
            **dependency_hashes,
        }
        node.config_hashes = node_profile_hashes(node_id, configs)
        if node_id in skipped_nodes:
            node.status = "planned"
            node.skipped = True
            node.validation_status = "skipped"
            node.output_paths = node_outputs
            node.actual_signature = expected
            _write_status(manifest, paths)
            continue
        cache_path = _node_record_path(root, node_id)
        should_try_reuse = resume and not force_triggered and force_stage != node_id
        if should_try_reuse:
            reusable, reason = can_reuse(cache_path, expected_signature=expected)
            if reusable:
                node.status = "reused"
                node.reused = True
                node.validation_status = "pass"
                node.actual_signature = expected
                node.output_hashes = artifact_hashes(node_outputs)
                output_hashes[node_id] = stable_hash(node.output_hashes)
                _write_status(manifest, paths)
                continue
            node.invalidation_reason = reason
        node.status = "running"
        node.started_at = utc_now()
        _write_status(manifest, paths)
        started = time.perf_counter()
        try:
            if node_id in {
                "resolve_source",
                "enrich_mediapipe21",
                "semantic_sanity_validation",
                "generate_review_bundle",
                "write_run_manifest",
            }:
                _execute_special_node(
                    node_id,
                    request=selected_request,
                    selected=selected,
                    paths=paths,
                    manifest=manifest,
                )
            elif node_id in {
                "canonicalize_grab",
                "validate_canonical",
                "validate_keypoints",
                "audit_object_mesh",
                "sample_object_surface",
                "validate_object_samples",
                "generate_warm_start",
                "validate_warm_start",
                "build_interaction_graph",
                "validate_interaction_graph",
                "evaluate_warm_start_interaction",
                "final_refinement",
                "validate_final_refinement",
                "full_surface_penetration_audit",
            }:
                _run_node_command(
                    node_id,
                    request=selected_request,
                    selected=selected,
                    paths=paths,
                    log_path=root / "logs" / f"{node_id}.log",
                    force_output=force_output or force_triggered,
                )
            else:
                raise WorkflowExecutionError(f"unimplemented workflow node: {node_id}")
            _assert_node_report_pass(node_id, node_outputs)
            node.output_paths = node_outputs
            node.output_hashes = artifact_hashes(node_outputs)
            node.actual_signature = expected
            node.validation_status = "pass"
            node.status = "passed"
            output_hashes[node_id] = stable_hash(node.output_hashes)
            _write_stage10_report(
                cache_path,
                cache_record(
                    node_id=node_id,
                    expected_signature=expected,
                    output_paths=node_outputs,
                    validation_path=_validation_path(node_id, node_outputs),
                ),
            )
        except Exception as exc:
            node.status = "failed"
            node.error = str(exc)
            node.validation_status = "fail"
            node.ended_at = utc_now()
            node.duration_s = time.perf_counter() - started
            manifest.run_status = "failed"
            manifest.source_integrity = _source_integrity_snapshot(manifest)
            write_execution_reports(
                manifest,
                run_root=paths["run_root"],
                elapsed_s=time.perf_counter() - execution_started,
            )
            _write_status(manifest, paths)
            write_json(manifest, manifest_path)
            raise
        node.ended_at = utc_now()
        node.duration_s = time.perf_counter() - started
        _write_status(manifest, paths)
    from toporetarget.data.storage import load_hoi_sequence

    canonical_sequence = load_hoi_sequence(paths["canonical"])
    if canonical_sequence.metadata.native_fps is None:
        raise WorkflowExecutionError("canonical artifact has no native FPS")
    manifest.native_fps = float(canonical_sequence.metadata.native_fps)
    manifest.timestamps = [float(value) for value in canonical_sequence.metadata.timestamps]
    source_hash_after = path_hash(manifest.source_path)
    source_unchanged = source_hash_after == manifest.source_hash
    if not source_unchanged:
        manifest.run_status = "failed"
        manifest.source_integrity = {
            "source_path": manifest.source_path,
            "source_hash_before": manifest.source_hash,
            "source_hash_after": source_hash_after,
            "source_integrity_check": "fail",
            "raw_source_not_written_by_workflow": False,
        }
        write_json(manifest, manifest_path)
        _write_status(manifest, paths)
        raise WorkflowExecutionError("raw GRAB source changed during workflow execution")
    manifest.run_status = "pending_human_acceptance"
    manifest.source_integrity = {
        "source_path": manifest.source_path,
        "source_hash_before": manifest.source_hash,
        "source_hash_after": source_hash_after,
        "source_integrity_check": "pass",
        "raw_source_not_written_by_workflow": True,
    }
    manifest.final_artifact_path = paths["final"]
    manifest.final_visualization_command = str(Path(paths["review"]) / "visualize_command.txt")
    manifest.artifacts = {
        "canonical": {"path": paths["canonical"], "hash": path_hash(paths["canonical"])},
        "object_samples": {
            "path": paths["object_samples"],
            "hash": path_hash(paths["object_samples"]),
        },
        "warm_start": {"path": paths["warm_start"], "hash": path_hash(paths["warm_start"])},
        "graph": {"path": paths["graph"], "hash": path_hash(paths["graph"])},
        "evaluation": {"path": paths["evaluation"], "hash": path_hash(paths["evaluation"])},
        "final": {"path": paths["final"], "hash": path_hash(paths["final"])},
        "collision_samples": {
            "path": str(
                selected_request.repo_root
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{selected_request.robot}_neutral.npz"
            ),
            "hash": path_hash(
                selected_request.repo_root
                / ".local"
                / "cache"
                / "geometry"
                / "robot_surface"
                / f"{selected_request.robot}_neutral.npz"
            ),
        },
    }
    manifest.validations["cross_stage_identity"] = cross_stage_identity_report(
        canonical=paths["canonical"],
        warm_start=paths["warm_start"],
        graph=paths["graph"],
        final=paths["final"],
        object_samples=paths["object_samples"],
        robot=selected_request.robot,
    )
    write_execution_reports(
        manifest, run_root=paths["run_root"], elapsed_s=time.perf_counter() - execution_started
    )
    write_json(manifest, manifest_path)
    _write_status(manifest, paths)
    return manifest_path


__all__ = ["WorkflowExecutionError", "run_workflow"]
