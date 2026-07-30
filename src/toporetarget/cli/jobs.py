"""Operator-facing final-refinement queue controls."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.retarget.final_jobs import (
    PAUSE_STATE,
    control_root,
    pause_final_jobs,
    paused,
    runtime_root,
)

app = typer.Typer(help="Pause and inspect Stage-12 final-refinement jobs.")


def _root(repo_root: Path | None) -> Path:
    return (repo_root or Path(__file__).resolve().parents[3]).resolve()


def _state(root: Path) -> str:
    manifest = control_root(root) / "pause_manifest.json"
    if manifest.is_file():
        return str(json.loads(manifest.read_text()).get("state", PAUSE_STATE))
    return PAUSE_STATE if paused(root) else "READY"


@app.command("pause-final")
def pause_final_command(
    reason: str = typer.Option("operator requested final-job quiescence", "--reason"),
    repo_root: Path | None = typer.Option(None, "--repo-root"),
) -> None:
    """Fail closed before any new final job consumes a queue item."""

    typer.echo(
        json.dumps(pause_final_jobs(_root(repo_root), reason=reason), indent=2, sort_keys=True)
    )


@app.command("status-final")
def status_final_command(repo_root: Path | None = typer.Option(None, "--repo-root")) -> None:
    """Read pause/scheduler state and bounded heartbeat locations without launching work."""

    root = _root(repo_root)
    control = control_root(root)
    scheduler = control / "scheduler_state.json"
    active = control / "active_jobs.json"
    scheduler_state = json.loads(scheduler.read_text()) if scheduler.exists() else None
    payload = {
        "state": _state(root),
        "new_final_tasks_allowed": bool(
            scheduler_state.get("new_final_tasks_allowed", not paused(root))
            if scheduler_state is not None
            else not paused(root)
        ),
        "control_root": str(control),
        "runtime_root": str(runtime_root(root)),
        "scheduler_state": scheduler_state,
        "active_jobs": json.loads(active.read_text()) if active.exists() else [],
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("drain-final")
def drain_final_command(repo_root: Path | None = typer.Option(None, "--repo-root")) -> None:
    """Report drain state only; running legacy workers are never killed implicitly."""

    root = _root(repo_root)
    typer.echo(
        json.dumps(
            {
                "state": _state(root),
                "action": "report_only",
                "legacy_workers_terminated": False,
                "message": "Use a validated checkpoint/graceful-stop path for a running worker.",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("resume-final")
def resume_final_command() -> None:
    """Deliberately refuses implicit resume; resumption needs explicit review authority."""

    raise typer.Exit(code=2)
