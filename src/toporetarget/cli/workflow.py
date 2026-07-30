"""Stage 10 GRAB-to-Arti-MANO workflow commands."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import typer

from toporetarget.workflows.accepted_run import create_accepted_run
from toporetarget.workflows.contact_audit import run_contact_audit
from toporetarget.workflows.contact_canonical_reaudit import (
    SHADOW_PROFILES,
    Stage932PreconditionError,
    run_canonical_reaudit,
    run_canonical_shadow_ablation,
)
from toporetarget.workflows.contact_metric_reconciliation import (
    run_contact_metric_reconciliation,
)
from toporetarget.workflows.contact_shadow_ablation import (
    MANDATORY_PROFILES,
    run_contact_shadow_ablation,
)
from toporetarget.workflows.contact_window import select_contact_windows
from toporetarget.workflows.executor import WorkflowExecutionError, run_workflow
from toporetarget.workflows.export import export_reference
from toporetarget.workflows.faithful_finalization import finalize_faithful_reproduction
from toporetarget.workflows.four_state_review import render_four_state_review_html
from toporetarget.workflows.gate import build_runtime_acceptance, evaluate_gate
from toporetarget.workflows.grab_suite import SuiteRunError, run_suite
from toporetarget.workflows.mesh_visualization import render_mesh_html
from toporetarget.workflows.planning import build_plan, write_plan
from toporetarget.workflows.s1_2a_stress import DEFAULT_CONFIG as S1_2A_DEFAULT_CONFIG
from toporetarget.workflows.s1_2a_stress import run_s1_2a
from toporetarget.workflows.s1_2a_stress import status as s1_2a_status
from toporetarget.workflows.s1_penetration import run_s1, s1_status
from toporetarget.workflows.s1_signal_rich import (
    DEFAULT_CONFIG as S1_SIGNAL_RICH_DEFAULT_CONFIG,
)
from toporetarget.workflows.s1_signal_rich import (
    audit_backends as audit_signal_rich_backends,
)
from toporetarget.workflows.s1_signal_rich import (
    diagnose_g1 as diagnose_signal_rich_g1,
)
from toporetarget.workflows.s1_signal_rich import (
    freeze_stress_set as freeze_signal_rich_stress_set,
)
from toporetarget.workflows.s1_signal_rich import (
    run_signal_rich,
)
from toporetarget.workflows.s1_signal_rich import (
    scan_source_candidates as scan_signal_rich_candidates,
)
from toporetarget.workflows.s1_signal_rich import (
    status as signal_rich_status,
)
from toporetarget.workflows.schema import WorkflowRequest, read_json, write_json
from toporetarget.workflows.shadow_equivalence import (
    PROFILES as SHADOW_EQUIVALENCE_PROFILES,
)
from toporetarget.workflows.shadow_equivalence import (
    calibrate_shadow_equivalence,
    run_stage9_shadow_ablation,
)
from toporetarget.workflows.stage9_3_4 import (
    Stage934Error,
    audit_solver_lineage,
    run_base_seed_ablation,
    run_current_baseline_repeats,
    run_current_causal_baseline,
    run_historical_replay,
    run_refinement_multistart,
    run_same_lineage_ablation,
    run_stage934,
    stage9_causal_status,
)
from toporetarget.workflows.stage9_3_5 import (
    PROFILES as PROJECTION_PROFILES,
)
from toporetarget.workflows.stage9_3_5 import (
    Stage935Error,
    run_attribution,
    run_branch,
    run_constraints,
    run_counterfactuals,
    run_projection,
    run_scan,
    run_status,
)
from toporetarget.workflows.stage9_4 import run_one_shot
from toporetarget.workflows.validation import (
    build_semantic_sanity_report,
    cross_stage_identity_report,
)
from toporetarget.workflows.visualization import run_visualization, write_visualization_report
from toporetarget.workflows.warm_start_audit import run_warm_start_audit

app = typer.Typer(help="Stage 10 bounded, resumable GRAB-to-Arti-MANO workflows.")


def _parse_frames(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


@app.command("run-s1-2a-stress-discovery")
def run_s1_2a_stress_discovery_command(
    config: Path = typer.Option(S1_2A_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_2a_e0_penetration_stress_v1"), "--experiment-root"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    generate_html: bool = typer.Option(True, "--generate-html/--no-generate-html"),
) -> None:
    """Run the complete S1.2A E0 stress discovery and E0/S1 comparison."""
    try:
        value = run_s1_2a(
            Path.cwd(),
            config_path=config,
            experiment_root=experiment_root,
            resume=resume,
            generate=generate_html,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        typer.echo(f"S1.2A stress discovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("s1-2a-stress-status")
def s1_2a_stress_status_command(
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_2a_e0_penetration_stress_v1"), "--experiment-root"
    ),
) -> None:
    """Show S1.2A source, warm, E0 probe, backend, and final state."""
    try:
        typer.echo(json.dumps(s1_2a_status(experiment_root), indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"S1.2A stress status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-s1-penetration-loss")
def run_s1_penetration_loss_command(
    run_root: Path = typer.Option(
        Path(".local/experiments/s1_sdf_penetration_loss_v1"), "--run-root", "--experiment-root"
    ),
    config: Path = typer.Option(
        Path("configs/experiments/s1_sdf_penetration_loss_v1.yaml"), "--config"
    ),
    max_wall_time: float = typer.Option(1800.0, "--max-wall-time", min=1.0),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    generate_html: bool = typer.Option(True, "--generate-html/--no-generate-html"),
) -> None:
    """Run the frozen two-clip G1/G2 S1 comparison without manual acceptance."""
    try:
        value = run_s1(
            Path.cwd(),
            config_path=config,
            run_root=run_root,
            max_wall_time=max_wall_time,
            resume=resume,
            dry_run=dry_run,
        )
        value["generate_html"] = generate_html
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        typer.echo(f"S1 penetration-loss workflow failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-sdf-loss-comparison")
def run_sdf_loss_comparison_command(
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_sdf_penetration_loss_v1"), "--experiment-root"
    ),
    config: Path = typer.Option(
        Path("configs/experiments/s1_sdf_penetration_loss_v1.yaml"), "--config"
    ),
    max_wall_time: float = typer.Option(1800.0, "--max-wall-time", min=1.0),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Compatibility alias for the bounded S1 comparison entry point."""
    value = run_s1(
        Path.cwd(),
        config_path=config,
        run_root=experiment_root,
        max_wall_time=max_wall_time,
        resume=resume,
    )
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command("visualize-sdf-loss-comparison")
def visualize_sdf_loss_comparison_command(
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_sdf_penetration_loss_v1"), "--experiment-root"
    ),
) -> None:
    """Report the self-contained S1 mesh-comparison HTML entry points."""
    html_root = experiment_root / "html"
    paths = sorted(str(path) for path in html_root.glob("*.html"))
    if not paths:
        typer.echo("S1 HTML is not available; complete run-s1-penetration-loss first.", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps({"status": "pass", "html": paths}, indent=2))


@app.command("s1-penetration-loss-status")
def s1_penetration_loss_status_command(
    run_root: Path = typer.Option(
        Path(".local/experiments/s1_sdf_penetration_loss_v1"), "--run-root"
    ),
) -> None:
    """Show S1 selection, decision, and checkpoint progress."""
    try:
        typer.echo(json.dumps(s1_status(run_root), indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"S1 penetration-loss status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("diagnose-source-penetration")
def diagnose_source_penetration_command(
    config: Path = typer.Option(S1_SIGNAL_RICH_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
) -> None:
    """Run the independent G1 source-MANO and collision-coverage diagnosis."""
    try:
        typer.echo(
            json.dumps(
                diagnose_signal_rich_g1(Path.cwd(), config, experiment_root),
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"source penetration diagnosis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("scan-penetration-signal")
def scan_penetration_signal_command(
    config: Path = typer.Option(S1_SIGNAL_RICH_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
) -> None:
    """Enumerate and audit the source-only GRAB candidate pool."""
    try:
        typer.echo(
            json.dumps(
                scan_signal_rich_candidates(Path.cwd(), config, experiment_root),
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"penetration signal scan failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("freeze-penetration-stress-set")
def freeze_penetration_stress_set_command(
    config: Path = typer.Option(S1_SIGNAL_RICH_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
) -> None:
    """Freeze exactly three source/E0-selected stress clips when the gate permits."""
    try:
        typer.echo(
            json.dumps(
                freeze_signal_rich_stress_set(Path.cwd(), config, experiment_root),
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"stress-set freeze failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("audit-sdf-backends")
def audit_sdf_backends_command(
    config: Path = typer.Option(S1_SIGNAL_RICH_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
) -> None:
    """Audit solver-fast versus reference triangle-winding SDFs on frozen clips."""
    try:
        typer.echo(
            json.dumps(
                audit_signal_rich_backends(Path.cwd(), config, experiment_root),
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"SDF backend audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-s1-signal-rich-evaluation")
def run_s1_signal_rich_evaluation_command(
    config: Path = typer.Option(S1_SIGNAL_RICH_DEFAULT_CONFIG, "--config"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_wall_time: float = typer.Option(1800.0, "--max-wall-time", min=1.0),
    generate_html: bool = typer.Option(True, "--generate-html/--no-generate-html"),
) -> None:
    """Run the complete resumable S1.1 dependency chain without manual choice."""
    del max_wall_time  # Per-probe/per-clip limits are immutable config values.
    try:
        value = run_signal_rich(
            Path.cwd(),
            config_path=config,
            experiment_root=experiment_root,
            resume=resume,
            generate=generate_html,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        typer.echo(f"S1.1 signal-rich workflow failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("s1-signal-rich-status")
def s1_signal_rich_status_command(
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_1_signal_rich_grab_v1"), "--experiment-root"
    ),
) -> None:
    """Show S1.1 scan, probe, freeze, backend, and full-run state."""
    try:
        typer.echo(
            json.dumps(signal_rich_status(experiment_root), indent=2, sort_keys=True, default=str)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"S1.1 signal-rich status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("sdf-loss-status")
def sdf_loss_status_command(
    experiment_root: Path = typer.Option(
        Path(".local/experiments/s1_sdf_penetration_loss_v1"), "--experiment-root"
    ),
) -> None:
    """Compatibility alias for S1 checkpoint/report status."""
    typer.echo(json.dumps(s1_status(experiment_root), indent=2, sort_keys=True, default=str))


@app.command("audit-solver-lineage")
def audit_solver_lineage_command(
    run: Path = typer.Option(..., "--run", help="Formal Stage 10 manifest."),
    output_root: Path = typer.Option(Path(".local/runs/stage9_3_4_provenance"), "--output-root"),
) -> None:
    """Build the versioned solver-effective provenance closure."""
    try:
        value = audit_solver_lineage(run, output_root)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"solver lineage audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-historical-replay")
def run_historical_replay_command(
    provenance_root: Path = typer.Option(..., "--provenance-root"),
    output_root: Path = typer.Option(
        Path(".local/runs/stage9_3_4_historical_lane"), "--output-root"
    ),
    frames: str = typer.Option("auto", "--frames"),
) -> None:
    """Run the historical lane without substituting the current environment."""
    try:
        value = run_historical_replay(provenance_root, output_root, frames=frames)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"historical replay failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-current-causal-baseline")
def run_current_causal_baseline_command(
    run: Path = typer.Option(..., "--run", help="Formal Stage 10 manifest."),
    output_root: Path = typer.Option(Path(".local/runs/stage9_3_4_current_lane"), "--output-root"),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=1.0),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Create or resume the independent current-lineage 60-frame baseline."""
    try:
        value = run_current_causal_baseline(
            run, output_root, max_wall_time=max_wall_time, resume=resume
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"current causal baseline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-current-baseline-repeats")
def run_current_baseline_repeats_command(
    run: Path = typer.Option(..., "--run", help="Formal Stage 10 manifest."),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(Path(".local/runs/stage9_3_4_provenance"), "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    repeat_count: int = typer.Option(3, "--repeat-count", min=3, max=5),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=1.0),
) -> None:
    """Run three independent bounded replays of the current-lineage baseline."""
    try:
        selected = () if frames == "auto" else _parse_frames(frames)
        value = run_current_baseline_repeats(
            run,
            current_baseline,
            output_root,
            frames=selected,
            repeat_count=repeat_count,
            max_wall_time=max_wall_time,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"current baseline repeats failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stage9-causal-status")
def stage9_causal_status_command(
    provenance_root: Path = typer.Option(..., "--provenance-root"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(Path(".local/reports/stage9_3_4"), "--output-root"),
) -> None:
    """Assemble Stage 9.3.4 causal reports, readiness, and HTML."""
    try:
        value = stage9_causal_status(provenance_root, current_baseline, output_root)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"Stage 9 causal status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-refinement-multistart")
def run_refinement_multistart_command(
    run: Path = typer.Option(..., "--run"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(Path(".local/runs/stage9_3_4_multistart"), "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    query_mode: str = typer.Option("frozen-first,native-query", "--query-mode"),
) -> None:
    """Run bounded same-lineage initialization diagnostics."""
    del query_mode
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_refinement_multistart(run, current_baseline, output_root, frames=selected)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"refinement multistart failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-base-seed-ablation")
def run_base_seed_ablation_command(
    run: Path = typer.Option(..., "--run"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(
        Path(".local/runs/stage9_3_4_base_seed_ablation"), "--output-root"
    ),
    frames: str = typer.Option("auto", "--frames"),
    protocols: str = typer.Option("initialization-only,seed-and-prior", "--protocols"),
) -> None:
    """Run SE(3)-guarded base-seed diagnostics."""
    del protocols
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_base_seed_ablation(run, current_baseline, output_root, frames=selected)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"base seed ablation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-same-lineage-ablation")
def run_same_lineage_ablation_command(
    run: Path = typer.Option(..., "--run"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(
        Path(".local/runs/stage9_3_4_mandatory_ablation"), "--output-root"
    ),
    frames: str = typer.Option("auto", "--frames"),
    profiles: str = typer.Option("all", "--profiles"),
) -> None:
    """Run margin, full-QuerySet, and projection diagnostics."""
    del profiles
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_same_lineage_ablation(run, current_baseline, output_root, frames=selected)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"same-lineage ablation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stage9-3-4")
def stage9_3_4_command(
    run: Path = typer.Option(..., "--run"),
    output_root: Path = typer.Option(Path(".local/runs/stage9_3_4_current_lane"), "--output-root"),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=1.0),
) -> None:
    """Run all bounded Stage 9.3.4 lanes in their declared order."""
    try:
        value = run_stage934(run, output_root=output_root, max_wall_time=max_wall_time)
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage934Error) as exc:
        typer.echo(f"Stage 9.3.4 failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("scan-warm-final-feasibility")
def scan_warm_final_feasibility_command(
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    output_root: Path = typer.Option(..., "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    samples: int = typer.Option(1001, "--samples", min=1001),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    """Scan the warm-to-final path without invoking a solver."""
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_scan(
            current_lineage_manifest,
            current_baseline,
            output_root,
            frames=selected,
            samples=samples,
            resume=resume,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"warm-final feasibility scan failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-feasibility-projection")
def run_feasibility_projection_command(
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    path_scan_root: Path = typer.Option(..., "--path-scan-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    profiles: str = typer.Option(",".join(PROJECTION_PROFILES), "--profiles"),
    full_512: bool = typer.Option(True, "--full-512/--no-full-512"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=1.0),
    solver_attempts: int = typer.Option(3, "--solver-attempts", min=1, max=8),
) -> None:
    """Run deterministic diagnostic minimal/official-slack projections."""
    if not full_512:
        typer.echo("Stage 9.3.5 projection requires full-512 constraints", err=True)
        raise typer.Exit(code=2)
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        selected_profiles = tuple(item.strip() for item in profiles.split(",") if item.strip())
        value = run_projection(
            current_lineage_manifest,
            current_baseline,
            path_scan_root,
            output_root,
            frames=selected,
            profiles=selected_profiles,
            resume=resume,
            max_wall_time=max_wall_time,
            solver_attempts=solver_attempts,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"feasibility projection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-state-counterfactuals")
def run_state_counterfactuals_command(
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    projection_root: Path | None = typer.Option(None, "--projection-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
) -> None:
    """Evaluate state counterfactuals without running a solver."""
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_counterfactuals(
            current_lineage_manifest,
            current_baseline,
            output_root,
            projection_root=projection_root,
            frames=selected,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"state counterfactuals failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("attribute-objective-constraints")
def attribute_objective_constraints_command(
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    projection_root: Path | None = typer.Option(None, "--projection-root"),
    counterfactual_root: Path = typer.Option(..., "--counterfactual-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    constraint_output_root: Path | None = typer.Option(None, "--constraint-output-root"),
    frames: str = typer.Option("auto", "--frames"),
) -> None:
    """Attribute objective terms and full-512 collision pressure."""
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        value = run_attribution(
            current_lineage_manifest,
            current_baseline,
            counterfactual_root,
            output_root,
            projection_root=projection_root,
            frames=selected,
        )
        constraint_root = constraint_output_root or (
            output_root.parent.parent / "stage9_3_5_constraint_attribution"
        )
        constraints = run_constraints(
            current_lineage_manifest,
            current_baseline,
            constraint_root,
            projection_root=projection_root,
            frames=selected,
        )
        value["constraints"] = {"row_count": len(constraints.get("rows", []))}
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"objective/constraint attribution failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-projection-branch")
def run_projection_branch_command(
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    projection_root: Path = typer.Option(..., "--projection-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    candidate: str = typer.Option("auto", "--candidate"),
    max_frames: int = typer.Option(10, "--max-frames", min=1, max=10),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    """Run the bounded branch gate and only roll out approved candidates."""
    try:
        value = run_branch(
            current_lineage_manifest,
            current_baseline,
            projection_root,
            output_root,
            candidate=candidate,
            max_frames=max_frames,
            resume=resume,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"projection branch failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stage9-causal-closure-status")
def stage9_causal_closure_status_command(
    current_lineage_manifest: Path = typer.Option(..., "--current-lineage-manifest"),
    current_baseline: Path = typer.Option(..., "--current-baseline"),
    projection_root: Path = typer.Option(..., "--projection-root"),
    counterfactual_root: Path = typer.Option(..., "--counterfactual-root"),
    objective_root: Path = typer.Option(..., "--objective-root"),
    constraint_root: Path = typer.Option(..., "--constraint-root"),
    branch_root: Path = typer.Option(..., "--branch-root"),
    output_root: Path = typer.Option(Path(".local/reports/stage9_3_5"), "--output-root"),
) -> None:
    """Assemble the Stage 9.3.5 causal closure, HTML, and Stage 9.4 gate."""
    try:
        value = run_status(
            current_lineage_manifest,
            current_baseline,
            projection_root=projection_root,
            counterfactual_root=counterfactual_root,
            objective_root=objective_root,
            constraint_root=constraint_root,
            branch_root=branch_root,
            output_root=output_root,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage935Error) as exc:
        typer.echo(f"Stage 9.3.5 causal closure status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stage9-one-shot")
def stage9_one_shot_command() -> None:
    """Run the bounded Stage 9.3.6--9.4 causal closure and repair bundle."""
    try:
        value = run_one_shot()
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Stage 9 one-shot closure failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("finalize-faithful-reproduction")
def finalize_faithful_reproduction_command(
    manual_acceptance: Path | None = typer.Option(
        None,
        "--manual-acceptance",
        help="Human-completed acceptance JSON. Omit to prepare a pending-signoff bundle.",
    ),
    output_root: Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional new Stage 10 run root; existing historical Stage 10 is never overwritten.",
    ),
) -> None:
    """Prepare or human-finalize the faithful v3 fixed Stage 10 export."""
    try:
        value = finalize_faithful_reproduction(
            output_root=output_root,
            manual_acceptance=manual_acceptance,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Faithful reproduction finalization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("audit-warm-start")
def audit_warm_start_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest; inputs resolve from it."),
    canonical_contact_audit: Path = typer.Option(..., "--canonical-contact-audit"),
    output_root: Path = typer.Option(..., "--output-root"),
    html: bool = typer.Option(False, "--html/--no-html"),
    run_reachability_diagnostics: bool = typer.Option(
        False, "--run-reachability-diagnostics/--no-reachability-diagnostics"
    ),
    diagnostic_frames: str = typer.Option("auto", "--diagnostic-frames"),
) -> None:
    """Audit Stage 7 warm-start fidelity without mutating official artifacts."""
    try:
        payload = run_warm_start_audit(
            run,
            canonical_contact_audit,
            output_root,
            html_output=html,
            run_reachability_diagnostics=run_reachability_diagnostics,
            diagnostic_frames=diagnostic_frames,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"warm-start audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-contact-shadow-ablation")
def run_contact_shadow_ablation_v2_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest."),
    canonical_audit_root: Path = typer.Option(..., "--canonical-audit-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    profiles: str = typer.Option(",".join(SHADOW_PROFILES), "--profiles"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the gate-approved Stage 9.3.2 diagnostic shadow solver profiles."""
    try:
        selected_frames = (
            ()
            if frames.strip().lower() == "auto"
            else tuple(int(value.strip()) for value in frames.split(",") if value.strip())
        )
        selected_profiles = tuple(value.strip() for value in profiles.split(",") if value.strip())
        payload = run_canonical_shadow_ablation(
            run,
            canonical_audit_root,
            output_root,
            frames=selected_frames,
            profiles=selected_profiles,
            force=force,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage932PreconditionError) as exc:
        typer.echo(f"canonical contact shadow ablation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("calibrate-shadow-equivalence")
def calibrate_shadow_equivalence_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest."),
    stage7_audit: Path = typer.Option(..., "--stage7-audit"),
    canonical_audit: Path = typer.Option(..., "--canonical-audit"),
    frames: str = typer.Option("auto", "--frames"),
    baseline_repeats: int = typer.Option(3, "--baseline-repeats", min=3, max=5),
    output_root: Path = typer.Option(..., "--output-root"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    """Calibrate the versioned Stage 9.3.3 official replay contract."""
    try:
        selected = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        payload = calibrate_shadow_equivalence(
            run,
            stage7_audit,
            canonical_audit,
            output_root,
            frames=selected,
            baseline_repeats=baseline_repeats,
            resume=resume,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"shadow equivalence calibration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-stage9-shadow-ablation")
def run_stage9_shadow_ablation_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest."),
    equivalence_root: Path = typer.Option(..., "--equivalence-root"),
    canonical_audit: Path = typer.Option(..., "--canonical-audit"),
    profiles: str = typer.Option(",".join(SHADOW_EQUIVALENCE_PROFILES), "--profiles"),
    frames: str = typer.Option("auto", "--frames"),
    output_root: Path = typer.Option(..., "--output-root"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=1.0),
    html: bool = typer.Option(True, "--html/--no-html"),
) -> None:
    """Run bounded, isolated Stage 9.3.3 shadow profiles after calibration."""
    try:
        selected_frames = () if frames.strip().lower() == "auto" else _parse_frames(frames)
        selected_profiles = tuple(item.strip() for item in profiles.split(",") if item.strip())
        payload = run_stage9_shadow_ablation(
            run,
            equivalence_root,
            canonical_audit,
            output_root,
            frames=selected_frames,
            profiles=selected_profiles,
            resume=resume,
            max_wall_time=max_wall_time,
            html_output=html,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Stage 9.3.3 shadow ablation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stage9-shadow-status")
def stage9_shadow_status_command(
    equivalence_root: Path = typer.Option(..., "--equivalence-root"),
    shadow_root: Path | None = typer.Option(None, "--shadow-root"),
) -> None:
    """Report Stage 9.3.3 calibration, checkpoint, and readiness status."""
    payload: dict[str, Any] = {}
    equivalence = equivalence_root.expanduser().resolve()
    for name in (
        "audit_manifest.json",
        "shadow_equivalence_contract.json",
        "numerical_noise_envelope.json",
        "official_baseline_equivalence.json",
    ):
        path = equivalence / name
        if path.exists():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    if shadow_root is not None:
        shadow = shadow_root.expanduser().resolve()
        for name in ("shadow_manifest.json", "stage9_4_readiness.json", "stage9_3_3_summary.json"):
            path = shadow / name
            if path.exists():
                payload[name] = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command("contact-audit-status")
def contact_audit_status_command(
    canonical_audit_root: Path = typer.Option(..., "--canonical-audit-root"),
    shadow_root: Path | None = typer.Option(None, "--shadow-root"),
) -> None:
    """Report canonical audit gate, shadow status, and Stage 9.4 readiness."""
    root = canonical_audit_root.expanduser().resolve()
    payload: dict[str, Any] = {}
    for name in ("audit_manifest.json", "stage9_3_2_summary.json", "stage9_4_readiness.json"):
        path = root / name
        if path.exists():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    if shadow_root is not None:
        path = shadow_root.expanduser().resolve() / "shadow_manifest.json"
        if path.exists():
            payload["shadow_manifest.json"] = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command("reaudit-contact-canonical")
def reaudit_contact_canonical_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest."),
    legacy_audit_root: Path = typer.Option(..., "--legacy-audit-root"),
    reconciliation_root: Path = typer.Option(..., "--reconciliation-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    surface_samples: int = typer.Option(8192, "--surface-samples", min=8192),
    precomputed_audit_root: Path | None = typer.Option(None, "--precomputed-audit-root"),
    html: bool = typer.Option(False, "--html/--no-html"),
    headless_smoke_test: bool = typer.Option(False, "--headless-smoke-test"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run Stage 9.3.2 formal contact audit on the reference winding SDF."""
    try:
        payload = run_canonical_reaudit(
            run,
            legacy_audit_root,
            reconciliation_root,
            output_root,
            surface_samples=surface_samples,
            html=html,
            headless_smoke_test=headless_smoke_test,
            force=force,
            precomputed_audit_root=precomputed_audit_root,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, Stage932PreconditionError) as exc:
        typer.echo(f"canonical contact re-audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
    evaluation_backend: str = typer.Option("configured", "--evaluation-backend"),
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
            evaluation_backend=evaluation_backend,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"contact retention audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("reconcile-contact-metrics")
def reconcile_contact_metrics_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest; all inputs resolve from it."),
    contact_audit_root: Path = typer.Option(..., "--contact-audit-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Reconcile Stage 9.2 and Stage 9.3 signed-distance definitions."""
    try:
        payload = run_contact_metric_reconciliation(
            run,
            contact_audit_root,
            output_root,
            force=force,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"contact metric reconciliation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run-contact-shadow-ablation-legacy")
def run_contact_shadow_ablation_command(
    run: Path = typer.Option(..., "--run", help="Stage 10 manifest; retained for provenance."),
    reconciliation_root: Path = typer.Option(..., "--reconciliation-root"),
    output_root: Path = typer.Option(..., "--output-root"),
    frames: str = typer.Option("auto", "--frames"),
    profiles: str = typer.Option(",".join(MANDATORY_PROFILES), "--profiles"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run only gate-approved diagnostic shadow profiles in an isolated root."""
    del run
    try:
        selected_frames = (
            ()
            if frames.strip().lower() == "auto"
            else tuple(int(value.strip()) for value in frames.split(",") if value.strip())
        )
        selected_profiles = tuple(value.strip() for value in profiles.split(",") if value.strip())
        payload = run_contact_shadow_ablation(
            reconciliation_root,
            output_root,
            profiles=selected_profiles,
            frames=selected_frames,
            force=force,
        )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"contact shadow ablation failed: {exc}", err=True)
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


@app.command("run-grab-suite")
def run_grab_suite_command(
    suite: Path = typer.Option(..., "--suite", help="Frozen YAML suite definition."),
    grab_root: Path = typer.Option(..., "--grab-root"),
    index: Path = typer.Option(..., "--index"),
    mano_model_root: Path = typer.Option(..., "--mano-model-root"),
    robot: str | None = typer.Option(None, "--robot"),
    solver_profile: str | None = typer.Option(None, "--solver-profile"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/grab_suite"), "--experiment-root"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_wall_time: float = typer.Option(1800.0, "--max-wall-time", min=1.0),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate"),
    export_reference: bool = typer.Option(True, "--export-reference/--no-export-reference"),
    generate_html: bool = typer.Option(True, "--generate-html/--no-generate-html"),
    unit: str | None = typer.Option(None, "--unit"),
) -> None:
    """Run a frozen multi-clip GRAB suite with generic Stage 5-9 components."""
    try:
        value = run_suite(
            suite=suite,
            grab_root=grab_root,
            index=index,
            mano_model_root=mano_model_root,
            robot=robot,
            solver_profile=solver_profile,
            experiment_root=experiment_root,
            resume=resume,
            max_wall_time=max_wall_time,
            evaluate=evaluate,
            export_reference_bundles=export_reference,
            generate_html=generate_html,
            unit=unit,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, SuiteRunError) as exc:
        typer.echo(f"GRAB suite run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("evaluate-retargeting-suite")
def evaluate_retargeting_suite_command(
    suite: Path = typer.Option(..., "--suite"),
    grab_root: Path = typer.Option(..., "--grab-root"),
    index: Path = typer.Option(..., "--index"),
    mano_model_root: Path = typer.Option(..., "--mano-model-root"),
    robot: str | None = typer.Option(None, "--robot"),
    solver_profile: str | None = typer.Option(None, "--solver-profile"),
    experiment_root: Path = typer.Option(
        Path(".local/experiments/grab_suite"), "--experiment-root"
    ),
    unit: str | None = typer.Option(None, "--unit"),
) -> None:
    """Re-run independent validation, export, and HTML evaluation for fixed artifacts."""
    try:
        value = run_suite(
            suite=suite,
            grab_root=grab_root,
            index=index,
            mano_model_root=mano_model_root,
            robot=robot,
            solver_profile=solver_profile,
            experiment_root=experiment_root,
            resume=True,
            max_wall_time=1.0,
            evaluate=True,
            export_reference_bundles=True,
            generate_html=True,
            unit=unit,
        )
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    except (OSError, ValueError, RuntimeError, SuiteRunError) as exc:
        typer.echo(f"suite evaluation failed: {exc}", err=True)
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
    old_final: Path | None = typer.Option(
        None,
        "--old-final",
        help="Current-lineage final for a four-state Stage 9 review.",
    ),
    comparison_final: Path | None = typer.Option(
        None,
        "--comparison-final",
        help="Candidate final for a four-state Stage 9 review.",
    ),
    review_report_root: Path | None = typer.Option(
        None,
        "--review-report-root",
        help="Stage 9 report root containing comparison and decision JSON files.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Open the generated HTML in the default browser; the HTML is interactive by default.",
    ),
    open_browser: bool = typer.Option(False, "--open-browser"),
) -> None:
    """Write a self-contained HTML viewer with source/warm/final hand meshes."""

    try:
        four_state_requested = any(
            value is not None for value in (old_final, comparison_final, review_report_root)
        )
        if four_state_requested:
            if old_final is None or comparison_final is None or review_report_root is None:
                raise ValueError(
                    "--old-final, --comparison-final, and --review-report-root "
                    "must be supplied together"
                )
            result = render_four_state_review_html(
                run,
                old_final=old_final,
                comparison_final=comparison_final,
                review_report_root=review_report_root,
                output=output,
                start_frame=start_frame,
                end_frame=end_frame,
                max_object_points=max_object_points,
                asset_root=asset_root,
                open_browser=open_browser or interactive,
            )
        else:
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
