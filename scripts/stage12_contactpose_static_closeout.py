#!/usr/bin/env python3
"""Build auditable Stage-12 static ContactPose closeout evidence.

This utility never launches refinement.  It records the two new ContactPose
results and builds an aggregate, symlink-only view of the six immutable v4
results plus the static results for HTML and provenance attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from toporetarget.contracts.canonical import load_canonical_hoi
from toporetarget.quality.html import smoke_html
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "stage12_selection.yaml"
STATIC_ACCEPTED_STATUSES = {
    "STATIC_FRAME_ACCEPTED",
    "STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING",
}


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value).strip("_")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _runtime_policy_ast_matches(executed_commit: str, current_commit: str) -> bool:
    """Allow a post-run formatting-only amend, but never a semantic rewrite."""

    import ast

    paths = (
        "scripts/stage12_dataset_validation.py",
        "src/toporetarget/retarget/static_runtime_policy.py",
    )
    for path in paths:
        executed = subprocess.run(
            ["git", "show", f"{executed_commit}:{path}"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        current = subprocess.run(
            ["git", "show", f"{current_commit}:{path}"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        if ast.dump(ast.parse(executed), include_attributes=False) != ast.dump(
            ast.parse(current), include_attributes=False
        ):
            return False
    return True


def _link(destination: Path, source: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"refuse to replace aggregate evidence link: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def _finite_arrays(
    arrays: dict[str, np.ndarray], *, allowed_not_applicable: frozenset[str] = frozenset()
) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(value))))
        for name, value in arrays.items()
        if name not in allowed_not_applicable and np.issubdtype(np.asarray(value).dtype, np.number)
    )


def _mesh_audit(canonical_path: Path) -> tuple[dict[str, Any], Any]:
    canonical = load_canonical_hoi(canonical_path)
    object_track = canonical.rigid_objects[0]
    mesh_definition = object_track.mesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_definition.vertices_local, dtype=np.float64),
        faces=np.asarray(mesh_definition.faces, dtype=np.int64),
        process=False,
    )
    faces_sorted = np.sort(mesh.faces, axis=1)
    duplicate_faces = int(len(faces_sorted) - len(np.unique(faces_sorted, axis=0)))
    edges, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    del edges
    degenerate_faces = int(np.count_nonzero(mesh.area_faces <= 0.0))
    audit = {
        "object_id": object_track.object_id,
        "raw_object_mesh_path": object_track.metadata.get("raw_contact_map_path"),
        "original_mesh_hash": mesh_definition.mesh_hash,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "finite_vertices": bool(np.all(np.isfinite(mesh.vertices))),
        "finite_faces": bool(np.all(np.isfinite(mesh.faces))),
        "object_watertight": bool(mesh.is_watertight),
        "edge_manifold": bool(np.all(edge_counts == 2)),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "connected_components": int(mesh.body_count),
        "euler_number": int(mesh.euler_number),
        "degenerate_faces": degenerate_faces,
        "duplicate_faces": duplicate_faces,
    }
    return audit, canonical


def _annotate_static_html(
    html_path: Path, audit: dict[str, Any], execution_sign_backend: Any
) -> None:
    """Add non-solver static provenance to an already-rendered HTML artifact."""

    marker = 'id="stage12-static-context"'
    document = html_path.read_text(encoding="utf-8")
    if marker in document:
        return
    context = (
        '<section id="stage12-static-context"><h2>Static sample context</h2>'
        "<p><strong>sample mode:</strong> static single frame; "
        "<strong>temporal metrics:</strong> NOT_APPLICABLE</p>"
        "<p><strong>object visual mesh:</strong> original object mesh; "
        "<strong>sign method:</strong> generalized_winding_on_original_mesh; "
        f"<strong>execution sign backend:</strong> {html.escape(str(execution_sign_backend))}</p>"
        f"<p><strong>object watertight audit:</strong> {audit['object_watertight']}; "
        "<strong>sign proxy overlay:</strong> not used; "
        "<strong>repair patches:</strong> none</p></section>"
    )
    if "</body>" not in document:
        raise ValueError(f"HTML has no closing body tag: {html_path}")
    html_path.write_text(document.replace("</body>", context + "</body>"), encoding="utf-8")


def _static_selection_evidence(
    *,
    root: Path,
    static_manifest: dict[str, Any],
) -> dict[str, Any]:
    report_path = root / "metrics" / "retarget_report.json"
    report = _read_json(report_path)
    checkpoint = dict(report.get("final_diagnostics", {}).get("checkpoint", {}))
    rows = list(checkpoint.get("frame_rows", []))
    if len(rows) != 1:
        raise ValueError(f"static selection must have exactly one frame row: {root}")
    row = dict(rows[0])
    canonical_path = root / "canonical" / "canonical_hoi_v2.zarr"
    warm_path = root / "warm" / "warm_start.zarr"
    final_path = Path(str(report["paths"]["final"]))
    audit, canonical = _mesh_audit(canonical_path)
    warm = load_warm_start(warm_path)
    final = load_final_trajectory(final_path)
    _annotate_static_html(
        root / "html" / "source_warm_final_wuji.html",
        audit,
        row.get("execution_profile", {}).get("sign_backend"),
    )
    source_positions = canonical.hands[0].keypoint_tracks["mediapipe21"].positions_scene
    static_not_applicable_arrays = frozenset(
        {
            "continuity_base_rotation_rad",
            "continuity_base_translation_m",
            "continuity_excess_keypoint_m",
            "continuity_finger_inf_rad",
            "stationarity_residual",
        }
    )
    final_finite = _finite_arrays(final.arrays, allowed_not_applicable=static_not_applicable_arrays)
    strict_gates = {
        "report_status_accepted": report.get("status") in STATIC_ACCEPTED_STATUSES,
        "strict_accepted": bool(row.get("strict_accepted", False)),
        "full_audit_exactly_one": int(row.get("diagnostics", {}).get("full_audit_call_count", 0))
        == 1,
        "sign_mismatch_count_zero": int(row.get("diagnostics", {}).get("sign_mismatch_count", 0))
        == 0,
        "false_certified_reuse_count_zero": int(
            row.get("diagnostics", {}).get("sign_cache", {}).get("false_certified_reuse_count", 0)
        )
        == 0,
        "all_values_finite": bool(row.get("all_values_finite", False)),
        "unqueried_violation_count_zero": int(row.get("unqueried_soft_violation_count", 0)) == 0,
        "source_finite": bool(np.all(np.isfinite(source_positions))),
        "warm_finite": _finite_arrays(warm.arrays),
        "final_finite": final_finite,
        "checkpoint_reload_pass": final.frame_count == 1,
        "object_visible_html_smoke": smoke_html(
            root / "html" / "source_warm_final_wuji.html", expected_frames=1, profiles=2
        ).get("status")
        == "pass",
        "no_repeated_manufactured_frames": canonical.metadata.metadata.get(
            "repeated_pose_manufacturing"
        )
        is False,
    }
    runtime = dict(report.get("runtime_policy", {}))
    if runtime.get("rolling_p95_gate") != "NOT_APPLICABLE":
        strict_gates["static_rolling_p95_not_applicable"] = False
    else:
        strict_gates["static_rolling_p95_not_applicable"] = True
    if runtime.get("consecutive_slow_frame_gate") != "NOT_APPLICABLE":
        strict_gates["static_consecutive_slow_not_applicable"] = False
    else:
        strict_gates["static_consecutive_slow_not_applicable"] = True
    sign_manifest = {
        "schema_version": "toporetarget.stage12.contactpose_original_sign.v1",
        **audit,
        "sign_method": "generalized_winding_on_original_mesh",
        "execution_sign_backend": row.get("execution_profile", {}).get("sign_backend"),
        "sign_proxy_used": False,
        "sign_proxy_classification": None,
        "physical_inside_outside_exactness": (
            "TOPOLOGICAL_WATERTIGHTNESS_AUDITED_NOT_OFFICIAL_CONTACT_GROUND_TRUTH"
        ),
        "non_watertight_source_policy": "UNRESOLVED_FOR_NON_WATERTIGHT_SOURCE",
        "metric_classification": "ENGINEERING_DIAGNOSTIC",
        "paper_exact_claim": False,
        "official_contact_ground_truth_claim": False,
    }
    collision_report = {
        "schema_version": "toporetarget.stage12.contactpose_collision_report.v1",
        "metric_classification": "ENGINEERING_DIAGNOSTIC",
        "strict_gates": strict_gates,
        "active_set_rounds": row.get("active_set_rounds"),
        "query_count": row.get("query_count"),
        "full_surface_backend_id": row.get("diagnostics", {}).get("full_surface_backend_id"),
        "sign_cache": row.get("diagnostics", {}).get("sign_cache", {}),
    }
    artifacts = {
        "canonical": canonical_path,
        "warm": warm_path,
        "graph": root / "exports" / "interaction_graph.zarr",
        "final": final_path,
        "html": root / "html" / "source_warm_final_wuji.html",
    }
    hashes = {name: _tree_hash(path) for name, path in artifacts.items()}
    artifact_manifest = {
        "schema_version": "toporetarget.stage12.contactpose_static_artifact_manifest.v1",
        "status": "COMPLETE_ACCEPTED" if all(strict_gates.values()) else "INCOMPLETE_OR_INVALID",
        "code_commit": static_manifest["code_commit"],
        "source_contract": static_manifest["source_contract"],
        "source_contract_hash": static_manifest["source_contract_hash"],
        "execution_profile": static_manifest["retarget_profile"],
        "execution_profile_hash": static_manifest["profile_hash"],
        "artifacts": {
            name: {"path": str(path), "sha256_tree": hashes[name]}
            for name, path in artifacts.items()
        },
    }
    provenance = {
        "schema_version": "toporetarget.stage12.contactpose_static_provenance.v1",
        "formal_static_closeout_manifest": static_manifest,
        "source_artifact_hash": hashes["canonical"],
        "warm_hash": hashes["warm"],
        "graph_hash": hashes["graph"],
        "final_hash": hashes["final"],
        "upstream_artifacts": "read_only_symlink",
        "old_source_v1_lineage_allowed": False,
    }
    runtime_profile = {
        "schema_version": "toporetarget.stage12.contactpose_static_runtime_profile.v1",
        "runtime_policy": runtime,
        "execution_profile": row.get("execution_profile", {}),
        "cpu_runtime": report.get("cpu_runtime", {}),
        "static_frame_count": 1,
        "temporal_metrics": "NOT_APPLICABLE",
        "not_applicable_final_arrays": sorted(static_not_applicable_arrays),
    }
    for name, payload in {
        "artifact_manifest.json": artifact_manifest,
        "provenance.json": provenance,
        "source_contract.json": {
            "source_contract": static_manifest["source_contract"],
            "source_contract_hash": static_manifest["source_contract_hash"],
            "source_manifest": _read_json(root / "source" / "source_contract_manifest.json"),
        },
        "sign_geometry_manifest.json": sign_manifest,
        "runtime_profile.json": runtime_profile,
        "collision_report.json": collision_report,
    }.items():
        _write_json(root / name, payload)
    return {
        "dataset": report["dataset"],
        "sequence": report["sequence"],
        "root": str(root),
        "status": report["status"],
        "wall_time_s": float(row["solve_time_s"]),
        "strict_gates": strict_gates,
        "sign_geometry_manifest": str(root / "sign_geometry_manifest.json"),
        "html": str(artifacts["html"]),
    }


def _prepare_aggregate(
    *,
    aggregate_root: Path,
    upstream_root: Path,
    static_root: Path,
    static_manifest: dict[str, Any],
) -> None:
    selections = (yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8")) or {})["selections"]
    for selection in selections:
        dataset = str(selection["dataset"])
        unit = _safe(str(selection["sequence"]))
        source_root = (
            static_root / dataset / unit
            if dataset == "contactpose"
            else upstream_root / dataset / unit
        )
        destination = aggregate_root / dataset / unit
        for name in ("canonical", "warm", "exports", "final", "html", "source"):
            _link(destination / name, source_root / name)
        for name in (
            "retarget_report.json",
            "retarget_report.md",
            "source_qualification.json",
            "source_qualification.md",
        ):
            source = source_root / "metrics" / name
            if source.exists():
                _link(destination / "metrics" / name, source)
        _write_aggregate_derivatives(
            root=destination,
            selection=selection,
            static_manifest=static_manifest,
        )
    aggregate_manifest = {
        "schema_version": "toporetarget.stage12.source_v2_v4_static_aggregate.v1",
        "code_commit": static_manifest["code_commit"],
        "source_contract": static_manifest["source_contract"],
        "source_contract_hash": static_manifest["source_contract_hash"],
        "selection_contract_hash": static_manifest["selection_contract_hash"],
        "retarget_profile": static_manifest["retarget_profile"],
        "static_closeout_root": str(static_root),
        "six_dynamic_artifacts": "read_only_symlink_from_parent_formal_run",
    }
    _write_json(aggregate_root / "formal_run_manifest.json", aggregate_manifest)


def _write_aggregate_derivatives(
    *, root: Path, selection: dict[str, Any], static_manifest: dict[str, Any]
) -> None:
    """Write derived manifests next to symlinked immutable solver artifacts."""

    report_path = root / "metrics" / "retarget_report.json"
    report = _read_json(report_path)
    checkpoint = dict(report.get("final_diagnostics", {}).get("checkpoint", {}))
    frame_rows = list(checkpoint.get("frame_rows", []))
    final_path = Path(str(report["paths"]["final"]))
    source_manifest_path = root / "source" / "source_contract_manifest.json"
    execution = dict(next(iter(frame_rows), {}).get("execution_profile", {}))
    artifact_paths = {
        "canonical": root / "canonical" / "canonical_hoi_v2.zarr",
        "warm": root / "warm" / "warm_start.zarr",
        "graph": root / "exports" / "interaction_graph.zarr",
        "final": final_path,
        "html": root / "html" / "source_warm_final_wuji.html",
    }
    hashes = {name: _tree_hash(path) for name, path in artifact_paths.items()}
    expected_frames = int(report["input"]["frame_count"])
    accepted = checkpoint.get("accepted_frames") == list(range(expected_frames))
    full_audits = [
        int(dict(item.get("diagnostics", {})).get("full_audit_call_count", 0))
        for item in frame_rows
    ]
    strict = {
        "accepted_frames": accepted,
        "full_audit_exactly_one_per_frame": len(full_audits) == expected_frames
        and all(value == 1 for value in full_audits),
        "strict_acceptance": all(bool(item.get("strict_accepted", False)) for item in frame_rows),
        "finite": all(bool(item.get("all_values_finite", False)) for item in frame_rows),
        "unqueried_violation_count_zero": all(
            int(item.get("unqueried_soft_violation_count", 0)) == 0 for item in frame_rows
        ),
    }
    sign_cache = dict(next(iter(frame_rows), {}).get("diagnostics", {}).get("sign_cache", {}))
    source_manifest = _read_json(source_manifest_path)
    static = expected_frames == 1 and bool(
        str(selection.get("sample_mode", selection.get("sample_type", ""))).startswith("static")
    )
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "toporetarget.stage12.aggregate_artifact_manifest.v1",
            "status": "COMPLETE_ACCEPTED" if all(strict.values()) else "INCOMPLETE_OR_INVALID",
            "source_contract": static_manifest["source_contract"],
            "source_contract_hash": static_manifest["source_contract_hash"],
            "execution_profile": execution.get("profile_id"),
            "execution_profile_hash": execution.get("profile_hash"),
            "artifacts": {
                name: {"path": str(path), "sha256_tree": hashes[name]}
                for name, path in artifact_paths.items()
            },
        },
    )
    _write_json(
        root / "provenance.json",
        {
            "schema_version": "toporetarget.stage12.aggregate_provenance.v1",
            "source_contract": static_manifest["source_contract"],
            "source_contract_hash": static_manifest["source_contract_hash"],
            "selection_contract_hash": static_manifest["selection_contract_hash"],
            "canonical_hash": hashes["canonical"],
            "warm_hash": hashes["warm"],
            "graph_hash": hashes["graph"],
            "final_hash": hashes["final"],
            "source_artifacts": "read_only_symlink",
            "old_source_v1_lineage_allowed": False,
        },
    )
    _write_json(
        root / "source_contract.json",
        {
            "source_contract": static_manifest["source_contract"],
            "source_contract_hash": static_manifest["source_contract_hash"],
            "source_manifest": source_manifest,
        },
    )
    _write_json(
        root / "runtime_profile.json",
        {
            "schema_version": "toporetarget.stage12.aggregate_runtime_profile.v1",
            "sample_kind": "static_single_frame" if static else "dynamic_trajectory",
            "temporal_metrics": "NOT_APPLICABLE" if static else "APPLICABLE",
            "runtime_policy": report.get("runtime_policy", {}),
            "execution_profile": execution,
            "cpu_runtime": report.get("cpu_runtime", {}),
        },
    )
    _write_json(
        root / "sign_geometry_manifest.json",
        {
            "schema_version": "toporetarget.stage12.aggregate_sign_geometry.v1",
            "sign_method": "generalized_winding_on_original_mesh",
            "original_mesh_hash": sign_cache.get("mesh_hash"),
            "execution_sign_backend": execution.get("sign_backend"),
            "sign_proxy_used": False,
            "metric_classification": "ENGINEERING_DIAGNOSTIC" if static else "PROFILE_DEFINED",
            "paper_exact_claim": False,
            "official_contact_ground_truth_claim": False,
        },
    )
    _write_json(
        root / "collision_report.json",
        {
            "schema_version": "toporetarget.stage12.aggregate_collision_report.v1",
            "metric_classification": "ENGINEERING_DIAGNOSTIC" if static else "PROFILE_DEFINED",
            "strict_gates": strict,
            "sign_mismatch_count": sum(
                int(dict(item.get("diagnostics", {})).get("sign_mismatch_count", 0))
                for item in frame_rows
            ),
            "false_certified_reuse_count": sum(
                int(
                    dict(dict(item.get("diagnostics", {})).get("sign_cache", {})).get(
                        "false_certified_reuse_count", 0
                    )
                )
                for item in frame_rows
            ),
        },
    )


def _html_smoke_matrix(aggregate_root: Path) -> dict[str, Any]:
    selections = (yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8")) or {})["selections"]
    rows = []
    for selection in selections:
        unit = _safe(str(selection["sequence"]))
        root = aggregate_root / str(selection["dataset"]) / unit
        report = _read_json(root / "metrics" / "retarget_report.json")
        expected_frames = int(report["input"]["frame_count"])
        html_path = root / "html" / "source_warm_final_wuji.html"
        smoke = smoke_html(html_path, expected_frames=expected_frames, profiles=2)
        static_context = True
        if expected_frames == 1:
            static_context = 'id="stage12-static-context"' in html_path.read_text(encoding="utf-8")
        rows.append(
            {
                "dataset": selection["dataset"],
                "sequence": selection["sequence"],
                "frame_count": expected_frames,
                "html": str(html_path),
                "smoke": smoke,
                "static_context_present": static_context,
                "pass": smoke.get("status") == "pass" and static_context,
            }
        )
    return {
        "schema_version": "toporetarget.stage12.html_smoke_matrix.v1",
        "rows": rows,
        "all_pass": all(row["pass"] for row in rows),
    }


def _eight_selection_matrix(aggregate_root: Path) -> dict[str, Any]:
    selections = (yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8")) or {})["selections"]
    rows = []
    for selection in selections:
        unit = _safe(str(selection["sequence"]))
        root = aggregate_root / str(selection["dataset"]) / unit
        report = _read_json(root / "metrics" / "retarget_report.json")
        checkpoint = dict(report.get("final_diagnostics", {}).get("checkpoint", {}))
        frame_rows = list(checkpoint.get("frame_rows", []))
        frame_count = int(report["input"]["frame_count"])
        acceptable = report.get("status") in {"pass", *STATIC_ACCEPTED_STATUSES}
        requirements = {
            "source": (root / "canonical/canonical_hoi_v2.zarr").is_dir(),
            "warm": (root / "warm/warm_start.zarr").is_dir(),
            "graph": (root / "exports/interaction_graph.zarr").is_dir(),
            "final": Path(str(report["paths"]["final"])).is_dir(),
            "full_audit": len(frame_rows) == frame_count
            and all(
                int(row.get("diagnostics", {}).get("full_audit_call_count", 0)) == 1
                for row in frame_rows
            ),
            "artifact_manifest": (root / "artifact_manifest.json").is_file(),
            "provenance": (root / "provenance.json").is_file(),
            "source_contract": (root / "source_contract.json").is_file(),
            "sign_geometry_manifest": (root / "sign_geometry_manifest.json").is_file(),
            "runtime_profile": (root / "runtime_profile.json").is_file(),
            "collision_report": (root / "collision_report.json").is_file(),
            "html": (root / "html/source_warm_final_wuji.html").is_file(),
            "report": (root / "metrics/retarget_report.json").is_file(),
            "accepted": acceptable
            and checkpoint.get("accepted_frames") == list(range(frame_count))
            and checkpoint.get("complete") is True,
        }
        rows.append(
            {
                "dataset": selection["dataset"],
                "sequence": selection["sequence"],
                "frame_count": frame_count,
                "status": report.get("status"),
                "requirements": requirements,
                "complete": all(requirements.values()),
            }
        )
    return {
        "schema_version": "toporetarget.stage12.eight_selection_completion_matrix.v1",
        "rows": rows,
        "count": len(rows),
        "complete_count": sum(row["complete"] for row in rows),
        "all_complete": all(row["complete"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    args = parser.parse_args()
    static_root = args.static_root.expanduser().resolve()
    upstream_root = args.upstream_root.expanduser().resolve()
    aggregate_root = args.aggregate_root.expanduser().resolve()
    reports_root = args.reports_root.expanduser().resolve()
    static_manifest = _read_json(static_root / "formal_static_closeout_manifest.json")
    current_commit = _git_head()
    executed_commit = str(static_manifest["code_commit"])
    if executed_commit != current_commit and not _runtime_policy_ast_matches(
        executed_commit, current_commit
    ):
        raise ValueError("static lineage runtime-policy code changed after execution")
    results = []
    for sequence in ("contactpose_full1_use_mug", "contactpose_full31_use_banana"):
        results.append(
            _static_selection_evidence(
                root=static_root / "contactpose" / sequence,
                static_manifest=static_manifest,
            )
        )
    _prepare_aggregate(
        aggregate_root=aggregate_root,
        upstream_root=upstream_root,
        static_root=static_root,
        static_manifest=static_manifest,
    )
    html_matrix = _html_smoke_matrix(aggregate_root)
    _write_json(reports_root / "html_smoke_matrix.json", html_matrix)
    matrix = _eight_selection_matrix(aggregate_root)
    _write_json(reports_root / "eight_selection_matrix.json", matrix)
    closeout_pass = (
        all(all(item["strict_gates"].values()) for item in results)
        and bool(html_matrix["all_pass"])
        and bool(matrix["all_complete"])
    )
    payload = {
        "schema_version": "toporetarget.stage12.contactpose_static_closeout.v1",
        "status": "CONTACTPOSE_EXACT_ORIGINAL_PATH_COMPLETE"
        if closeout_pass
        else "CONTACTPOSE_STATIC_CLOSEOUT_BLOCKED",
        "stage12_status": "STAGE12_COMPLETED_WITH_STATIC_CONTACTPOSE_LIMITATION",
        "execution_code_commit": executed_commit,
        "current_remote_bound_commit": current_commit,
        "runtime_policy_ast_equivalent_to_current": _runtime_policy_ast_matches(
            executed_commit, current_commit
        ),
        "proxy_experiment_triggered": False,
        "static_results": results,
        "html_smoke_all_pass": html_matrix["all_pass"],
        "eight_selection_matrix": {
            "complete_count": matrix["complete_count"],
            "all_complete": matrix["all_complete"],
        },
        "aggregate_root": str(aggregate_root),
        "upstream_root": str(upstream_root),
    }
    _write_json(reports_root / "contactpose_static_closeout.json", payload)
    print(
        json.dumps(
            {"status": payload["status"], "aggregate_root": str(aggregate_root)}, sort_keys=True
        )
    )
    return 0 if payload["status"] == "CONTACTPOSE_EXACT_ORIGINAL_PATH_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
