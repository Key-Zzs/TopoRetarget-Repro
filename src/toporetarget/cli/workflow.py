"""Stage 10 GRAB-to-Arti-MANO workflow commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import typer

from toporetarget.workflows.accepted_run import create_accepted_run
from toporetarget.workflows.contact_audit import run_contact_audit
from toporetarget.workflows.contact_window import select_contact_windows
from toporetarget.workflows.executor import WorkflowExecutionError, run_workflow
from toporetarget.workflows.export import export_reference
from toporetarget.workflows.gate import build_runtime_acceptance, evaluate_gate
from toporetarget.workflows.mesh_visualization import render_mesh_html
from toporetarget.workflows.planning import build_plan, write_plan
from toporetarget.workflows.schema import WorkflowRequest, read_json, write_json
from toporetarget.workflows.validation import (
    build_semantic_sanity_report,
    cross_stage_identity_report,
)
from toporetarget.workflows.visualization import run_visualization, write_visualization_report

app = typer.Typer(help="Stage 10 bounded, resumable GRAB-to-Arti-MANO workflows.")


@app.command("audit-contact-retention")
def audit_contact_retention_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest; all inputs resolve from it."),
    output_dir: Path = typer.Option(Path(".local/runs/stage9_3_contact_audit"), "--output-dir"),
    html: bool = typer.Option(False, "--html/--no-html"),
    interactive: bool = typer.Option(False, "--interactive"),
    surface_samples: int = typer.Option(8192, "--surface-samples", min=8192),
    thresholds_mm: str = typer.Option("1,2,3,5,8,10", "--thresholds-mm"),
    frame_start: int | None = typer.Option(None, "--frame-start", min=0),
    frame_end: int | None = typer.Option(None, "--frame-end", min=1),
    links: str | None = typer.Option(None, "--links"),
    force: bool = typer.Option(False, "--force"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    run_shadow_ablation: bool = typer.Option(False, "--run-shadow-ablation"),
    shadow_frames: str = typer.Option("auto", "--shadow-frames"),
    headless_smoke_test: bool = typer.Option(False, "--headless-smoke-test"),
) -> None:
    """Audit source/warm/final contact retention without invoking Stage 9."""
    try:
        parsed_thresholds = [
            float(value.strip()) for value in thresholds_mm.split(",") if value.strip()
        ]
        selected_links = (
            [value.strip() for value in links.split(",") if value.strip()] if links else None
        )
        payload = run_contact_audit(
            run,
            output_dir,
            thresholds_mm=parsed_thresholds,
            surface_samples=surface_samples,
            frame_start=frame_start,
            frame_end=frame_end,
            links=selected_links,
            html=html,
            interactive=interactive,
            force=force,
            no_cache=no_cache,
            run_shadow_ablation=run_shadow_ablation,
            shadow_frames=shadow_frames,
            headless_smoke_test=headless_smoke_test,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"contact retention audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("gate-status")
def gate_status_command(
    manual_acceptance: Path = typer.Option(..., "--manual-acceptance"),
    runtime_acceptance: Path = typer.Option(..., "--runtime-acceptance"),
    final: Path = typer.Option(
        Path(".local/cache/retarget/final/stage9_2_contact_rich_60f_v3.zarr"), "--final"
    ),
    status: Path = typer.Option(
        Path(".local/reports/stage9_performance/stage9_2_status.json"), "--stage9-status"
    ),
    performance: Path = typer.Option(
        Path(".local/reports/stage9_performance/performance_gate.json"), "--performance"
    ),
    checkpoint: Path = typer.Option(
        Path(".local/cache/retarget/final_checkpoints/stage9_2_contact_rich_60f_v3/manifest.json"),
        "--checkpoint",
    ),
    report: Path = typer.Option(Path(".local/reports/stage10/stage10_gate.json"), "--report"),
) -> None:
    """Derive, rather than accept, the Stage 10 unblocked decision."""
    try:
        payload = evaluate_gate(
            final_path=final,
            manual_path=manual_acceptance,
            runtime_path=runtime_acceptance,
            status_path=status,
            performance_path=performance,
            checkpoint_path=checkpoint,
            repo_root=_repo_root(),
            output=report,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        if not payload["stage10_unblocked"]:
            raise typer.Exit(code=1)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"workflow gate failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("accept-reference-runtime")
def accept_reference_runtime_command(
    final: Path = typer.Option(
        Path(".local/cache/retarget/final/stage9_2_contact_rich_60f_v3.zarr"), "--final"
    ),
    manual_acceptance: Path = typer.Option(..., "--manual-acceptance"),
    status: Path = typer.Option(
        Path(".local/reports/stage9_performance/stage9_2_status.json"), "--stage9-status"
    ),
    performance: Path = typer.Option(
        Path(".local/reports/stage9_performance/performance_gate.json"), "--performance"
    ),
    checkpoint: Path = typer.Option(
        Path(".local/cache/retarget/final_checkpoints/stage9_2_contact_rich_60f_v3/manifest.json"),
        "--checkpoint",
    ),
    output: Path = typer.Option(
        Path(".local/reports/stage9/reference_runtime_acceptance.json"), "--output"
    ),
) -> None:
    """Record the explicit user decision for this bounded reference-runtime milestone."""
    try:
        payload = build_runtime_acceptance(
            final_path=final,
            manual_path=manual_acceptance,
            status_path=status,
            performance_path=performance,
            checkpoint_path=checkpoint,
            output=output,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"reference-runtime acceptance failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-accepted")
def run_accepted_command(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
    final: Path = typer.Option(..., "--final"),
    manual_acceptance: Path = typer.Option(..., "--manual-acceptance"),
    runtime_acceptance: Path = typer.Option(..., "--runtime-acceptance"),
    run_root: Path = typer.Option(..., "--run-root"),
    collision_samples: Path = typer.Option(..., "--collision-samples"),
    no_review: bool = typer.Option(False, "--no-review"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Create a fresh Stage 10 run from accepted artifacts without Stage 9 execution."""
    try:
        manifest = create_accepted_run(
            source_manifest=source_manifest,
            final_path=final,
            manual_acceptance=manual_acceptance,
            runtime_acceptance=runtime_acceptance,
            run_root=run_root,
            repo_root=_repo_root(),
            collision_samples=collision_samples,
            generate_review=not no_review,
            resume=resume,
        )
        typer.echo(str(manifest))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"accepted workflow failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _request(
    *,
    sequence: str,
    index: Path,
    hand: str,
    robot: str,
    start_frame: int | None,
    end_frame: int | None,
    auto_contact_window: bool,
    window_length: int,
    mano_model_root: Path | None,
    asset_root: Path | None,
    run_root: Path,
    refinement_solver_profile: str = "scipy_slsqp_active_set_v1",
) -> WorkflowRequest:
    request = WorkflowRequest(
        sequence=sequence,
        index=index,
        hand=hand,
        robot=robot,
        refinement_solver_profile=refinement_solver_profile,
        start_frame=start_frame,
        end_frame=end_frame,
        auto_contact_window=auto_contact_window,
        window_length=window_length,
        mano_model_root=mano_model_root,
        asset_root=asset_root,
        run_root=run_root,
        repo_root=_repo_root(),
    )
    request.validate()
    return request


@app.command("select-contact-window")
def select_contact_window_command(
    index: Path = typer.Option(..., "--index"),
    sequence: str | None = typer.Option(None, "--sequence"),
    subject: str | None = typer.Option(None, "--subject"),
    object_name: str | None = typer.Option(None, "--object", "--object-name"),
    hand: str = typer.Option("right", "--hand"),
    window_length: int = typer.Option(60, "--window-length", min=1),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    report: Path = typer.Option(
        Path(".local/reports/stage10/contact_window_selection.json"), "--report"
    ),
    candidates_report: Path | None = typer.Option(None, "--candidates-report"),
) -> None:
    """Select one deterministic contact-rich window from an explicit sequence/query."""

    try:
        if (start_frame is None) != (end_frame is None):
            raise typer.BadParameter("--start-frame and --end-frame must be supplied together")
        if start_frame is not None:
            assert end_frame is not None
            if end_frame - start_frame != window_length:
                raise typer.BadParameter("explicit frame range must equal --window-length")
        if sequence is None:
            if subject is None and object_name is None:
                raise typer.BadParameter("provide --sequence or a finite --subject/--object query")
            from toporetarget.data.indexes.grab import load_grab_index

            sequences = [
                str(item["sequence_id"])
                for item in load_grab_index(index)
                if (subject is None or item.get("subject_id") == subject)
                and (object_name is None or item.get("object_token") == object_name)
            ]
        else:
            sequences = [sequence]
        if not sequences:
            raise typer.BadParameter("finite query matched no indexed sequences")
        reports: list[dict[str, Any]] = []
        for item in sorted(sequences):
            request = _request(
                sequence=item,
                index=index,
                hand=hand,
                robot="artimano_rh" if hand == "right" else "artimano_lh",
                start_frame=start_frame,
                end_frame=end_frame,
                auto_contact_window=start_frame is None,
                window_length=window_length,
                mano_model_root=mano_model_root,
                asset_root=None,
                run_root=Path(".local/runs/grab"),
            )
            reports.append(select_contact_windows(request, mano_model_root=mano_model_root))
        candidates = [item for report_item in reports for item in report_item.get("candidates", [])]
        accepted = [item for item in candidates if not item.get("rejection_reasons")]
        accepted.sort(
            key=lambda item: (
                -float(item["contact_frame_ratio"]),
                -int(item["total_hand_contact_vertices"]),
                float(item["source_contact_median_distance_m"] or 1e9),
                int(item["start_frame"]),
                str(item["sequence"]),
            )
        )
        selected = accepted[0] if accepted else None
        payload = {
            "schema_version": "toporetarget.contact_window_selection.v1",
            "status": "pass" if selected is not None else "fail",
            "query": {"sequence": sequence, "subject": subject, "object": object_name},
            "hand": hand,
            "window_length": window_length,
            "candidates": candidates,
            "selected": selected,
            "source_selection_reports": reports,
        }
        write_json(payload, report)
        if candidates_report is not None:
            write_json({"candidates": candidates}, candidates_report)
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        if selected is None:
            raise typer.Exit(code=1)
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"contact-window selection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("plan-grab")
def plan_grab_command(
    sequence: str = typer.Option(..., "--sequence"),
    index: Path = typer.Option(..., "--index"),
    hand: str = typer.Option("right", "--hand"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    auto_contact_window: bool = typer.Option(False, "--auto-contact-window"),
    window_length: int = typer.Option(60, "--window-length", min=1),
    refinement_solver_profile: str = typer.Option(
        "scipy_slsqp_active_set_v1", "--refinement-solver-profile"
    ),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    run_root: Path = typer.Option(Path(".local/runs/grab"), "--run-root"),
    output: Path | None = typer.Option(None, "--output", "--plan-output"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Create a DAG plan; dry-run never loads MANO or runs a Stage 5-9 command."""

    del dry_run  # Planning is always dry; execution is a separate command.
    try:
        request = _request(
            sequence=sequence,
            index=index,
            hand=hand,
            robot=robot,
            start_frame=start_frame,
            end_frame=end_frame,
            auto_contact_window=auto_contact_window,
            window_length=window_length,
            refinement_solver_profile=refinement_solver_profile,
            mano_model_root=mano_model_root,
            asset_root=None,
            run_root=run_root,
        )
        plan = build_plan(request)
        destination = output or Path(plan.run_root) / "plan.json"
        write_plan(plan, destination)
        write_plan(
            plan,
            request.repo_root / ".local" / "reports" / "stage10" / "workflow_plan.json",
        )
        typer.echo(json.dumps(plan.as_dict(), indent=2, sort_keys=True, default=str))
    except (ValueError, OSError) as exc:
        typer.echo(f"workflow planning failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-grab")
def run_grab_command(
    sequence: str = typer.Option(..., "--sequence"),
    index: Path = typer.Option(..., "--index"),
    hand: str = typer.Option("right", "--hand"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    auto_contact_window: bool = typer.Option(False, "--auto-contact-window"),
    window_length: int = typer.Option(60, "--window-length", min=1),
    refinement_solver_profile: str = typer.Option(
        "scipy_slsqp_active_set_v1", "--refinement-solver-profile"
    ),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    run_root: Path = typer.Option(Path(".local/runs/grab"), "--run-root"),
    resume: bool = typer.Option(False, "--resume"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    generate_review: bool = typer.Option(True, "--generate-review/--no-generate-review"),
    force_stage: str | None = typer.Option(None, "--force-stage"),
    force_output: bool = typer.Option(False, "--force-output"),
    manual_acceptance: Path | None = typer.Option(None, "--manual-acceptance"),
) -> None:
    """Run exactly one bounded sequence/window."""

    try:
        request = _request(
            sequence=sequence,
            index=index,
            hand=hand,
            robot=robot,
            start_frame=start_frame,
            end_frame=end_frame,
            auto_contact_window=auto_contact_window,
            window_length=window_length,
            refinement_solver_profile=refinement_solver_profile,
            mano_model_root=mano_model_root,
            asset_root=asset_root,
            run_root=run_root,
        )
        manifest = run_workflow(
            request,
            resume=resume,
            validate=validate,
            generate_review=generate_review,
            force_stage=force_stage,
            force_output=force_output,
            manual_acceptance=manual_acceptance,
        )
        typer.echo(str(manifest))
    except (ValueError, OSError, RuntimeError, WorkflowExecutionError) as exc:
        typer.echo(f"workflow run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("status")
def status_command(run: Path = typer.Option(..., "--run")) -> None:
    """Show node state without loading robot or solver modules."""

    try:
        payload = read_json(run)
        typer.echo(
            json.dumps(
                {
                    "run_id": payload.get("run_id"),
                    "run_status": payload.get("run_status"),
                    "nodes": [
                        {
                            "node_id": item.get("node_id"),
                            "status": item.get("status"),
                            "reused": item.get("reused"),
                            "output_paths": item.get("output_paths"),
                            "output_hashes": item.get("output_hashes"),
                            "input_hashes": item.get("input_hashes"),
                            "validation_status": item.get("validation_status"),
                            "duration_s": item.get("duration_s"),
                            "invalidation_reason": item.get("invalidation_reason"),
                            "error": item.get("error"),
                        }
                        for item in payload.get("nodes", [])
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"workflow status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate")
def validate_command(
    run: Path = typer.Option(..., "--run"),
    report: Path | None = typer.Option(None, "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Re-run semantic and cross-stage validation from manifest artifact paths."""

    try:
        manifest = read_json(run)
        artifact = manifest["artifacts"]
        selected = manifest["contact_window_selection"]["selected"]
        semantic = build_semantic_sanity_report(
            canonical=artifact["canonical"]["path"],
            final=artifact["final"]["path"],
            robot=manifest["robot"],
            collision_samples=artifact["collision_samples"]["path"],
            selected_window=selected,
            final_contact_sanity_max_distance_m=0.05,
        )
        identity = cross_stage_identity_report(
            canonical=artifact["canonical"]["path"],
            warm_start=artifact["warm_start"]["path"],
            graph=artifact["graph"]["path"],
            final=artifact["final"]["path"],
            object_samples=artifact["object_samples"]["path"],
            robot=manifest["robot"],
        )
        payload = {
            "status": semantic["status"],
            "semantic_sanity": semantic,
            "cross_stage_identity": identity,
        }
        destination = (
            report or Path(manifest["run_root"]) / "reports" / "end_to_end_validation.json"
        )
        write_json(payload, destination)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["section", "key", "value"])
                for section, values in payload.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            writer.writerow([section, key, json.dumps(value, default=str)])
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"workflow validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("review-template")
def review_template_command(
    run: Path = typer.Option(..., "--run"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    manifest = read_json(run)
    start, end = manifest["selected_frame_range"]
    length = int(end - start)
    destination = (
        output or Path(manifest["run_root"]) / "review" / "manual_acceptance.template.json"
    )
    write_json(
        {
            "schema_version": "toporetarget.manual_acceptance.v1",
            "status": "pending_human_review",
            "reviewer": "",
            "reviewed_frames": [0, length // 2, max(0, length - 1)],
            "current_window_interpretation": None,
            "source_object_alignment": None,
            "warm_start_object_alignment": None,
            "final_object_alignment": None,
            "sdf_visual_consistency": None,
            "right_left_semantics": None,
            "no_visible_discontinuity": None,
            "no_visible_mirroring": None,
            "no_unexplained_scale_error": None,
            "contact_rich_clip_validated": False,
            "notes": ["Human reviewer must fill this file; Codex does not write pass."],
        },
        destination,
    )
    typer.echo(str(destination))


@app.command("visualize")
def visualize_command(
    run: Path = typer.Option(..., "--run"),
    interactive: bool = typer.Option(False, "--interactive"),
    view: str = typer.Option("scene", "--view"),
    frame: int | None = typer.Option(None, "--frame", min=0),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    display_stride: int = typer.Option(1, "--display-stride", min=1),
    output: Path | None = typer.Option(None, "--output"),
    report: Path | None = typer.Option(None, "--report"),
    show_source_hand: bool = typer.Option(True, "--show-source-hand/--hide-source-hand"),
    show_warm_start: bool = typer.Option(True, "--show-warm-start/--hide-warm-start"),
    show_final: bool = typer.Option(True, "--show-final/--hide-final"),
    show_object: bool = typer.Option(True, "--show-object/--hide-object"),
    show_interaction_edges: bool = typer.Option(
        True, "--show-interaction-edges/--hide-interaction-edges"
    ),
    show_collision_samples: bool = typer.Option(
        True, "--show-collision-samples/--hide-collision-samples"
    ),
    show_query_set: bool = typer.Option(True, "--show-query-set/--hide-query-set"),
    show_penetrations: bool = typer.Option(True, "--show-penetrations/--hide-penetrations"),
    show_slack: bool = typer.Option(True, "--show-slack/--hide-slack"),
) -> None:
    try:
        result = run_visualization(
            run,
            frame=frame,
            view=view,
            start_frame=start_frame,
            end_frame=end_frame,
            display_stride=display_stride,
            output=output,
            interactive=interactive,
            flags={
                "show_source_hand": show_source_hand,
                "show_warm_start": show_warm_start,
                "show_final": show_final,
                "show_object": show_object,
                "show_interaction_edges": show_interaction_edges,
                "show_collision_samples": show_collision_samples,
                "show_query_set": show_query_set,
                "show_penetrations": show_penetrations,
                "show_slack": show_slack,
            },
        )
        if report is not None:
            write_visualization_report(result, report)
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"workflow visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-mesh")
def visualize_mesh_command(
    run: Path = typer.Option(..., "--run"),
    mode: str = typer.Option(
        "mesh",
        "--mode",
        help="mesh, full-graph, figure4-style, laplacian-diagnostic, or combined",
    ),
    output: Path | None = typer.Option(None, "--output"),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    max_object_points: int = typer.Option(1200, "--max-object-points", min=1),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Open the generated HTML in the default browser; the HTML is interactive by default.",
    ),
    open_browser: bool = typer.Option(False, "--open-browser"),
) -> None:
    """Write a self-contained HTML viewer with source/warm/final hand meshes."""

    try:
        result = render_mesh_html(
            run,
            output=output,
            mode=mode,
            start_frame=start_frame,
            end_frame=end_frame,
            max_object_points=max_object_points,
            asset_root=asset_root,
            open_browser=open_browser or interactive,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"workflow mesh visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("export-reference")
def export_reference_command(
    run: Path = typer.Option(..., "--run"),
    format: str = typer.Option("zarr", "--format"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    try:
        manifest = read_json(run)
        destination = output or Path(manifest["run_root"]) / "exports" / f"robot_reference.{format}"
        result = export_reference(
            manifest,
            output=destination,
            format=format,
            metadata_path=destination.with_suffix(destination.suffix + ".json"),
        )
        manifest.setdefault("export_paths", {})[format] = str(destination.resolve())
        metadata = result.get("metadata", {})
        for key in ("native_fps", "object_id", "source_sequence", "subject", "action"):
            if metadata.get(key) is not None:
                manifest[key if key != "source_sequence" else "source_sequence"] = metadata[key]
        if metadata.get("side"):
            manifest["hand"] = metadata["side"]
        if metadata.get("robot"):
            manifest["robot"] = metadata["robot"]
        manifest["updated_at"] = __import__(
            "toporetarget.workflows.schema", fromlist=["utc_now"]
        ).utc_now()
        write_json(manifest, run)
        typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"reference export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
