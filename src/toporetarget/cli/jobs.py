"""Job-control CLI with explicit scope isolation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.workflows import s1_3_jobs

app = typer.Typer(help="Scoped, checkpoint-safe workflow job controls.")


def _root() -> Path:
    return Path.cwd()


@app.command("pause")
def pause_command(scope: str = typer.Option(..., "--scope")) -> None:
    typer.echo(json.dumps(s1_3_jobs.pause(_root(), scope=scope), indent=2, sort_keys=True))


@app.command("resume")
def resume_command(scope: str = typer.Option(..., "--scope")) -> None:
    typer.echo(json.dumps(s1_3_jobs.resume(_root(), scope=scope), indent=2, sort_keys=True))


@app.command("status")
def status_command(scope: str = typer.Option(..., "--scope")) -> None:
    typer.echo(json.dumps(s1_3_jobs.status(_root(), scope=scope), indent=2, sort_keys=True))


@app.command("drain")
def drain_command(scope: str = typer.Option(..., "--scope")) -> None:
    typer.echo(json.dumps(s1_3_jobs.drain(_root(), scope=scope), indent=2, sort_keys=True))
