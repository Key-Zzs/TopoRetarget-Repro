"""GRAB Arti-MANO A--E quality experiment commands."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from toporetarget.quality.orchestrator import freeze_selection, run_a_to_e
from toporetarget.quality.schema import EXPERIMENT_ID, read_json
from toporetarget.quality.surfaces import build_artimano_surface_profile

app = typer.Typer(help="Frozen four-clip GRAB Arti-MANO quality experiment.")


def _echo(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _root(value: Path) -> Path:
    return value.expanduser()


@app.command("freeze")
def freeze(
    grab_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB"), "--grab-root"
    ),
    mano_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano"), "--mano-model-root"
    ),
    asset_root: Path = typer.Option(Path("third_party/robot_hands/artimano"), "--asset-root"),
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
) -> None:
    """Freeze the four prescribed native-frame units and write the lock."""
    _echo(
        freeze_selection(
            grab_root=grab_root,
            mano_root=mano_root,
            asset_root=asset_root,
            experiment_root=experiment_root,
        )
    )


@app.command("build-contact-surfaces")
def build_contact_surfaces(
    asset_root: Path = typer.Option(Path("third_party/robot_hands/artimano"), "--asset-root"),
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
) -> None:
    profile = build_artimano_surface_profile(
        experiment_root / "surface_contact", asset_root=asset_root
    )
    _echo(profile.as_dict())


@app.command("run-a-to-e")
def run_a_to_e_command(
    config: Path = typer.Option(Path(f"configs/experiments/{EXPERIMENT_ID}.yaml"), "--config"),
    grab_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB"), "--grab-root"
    ),
    mano_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano"), "--mano-model-root"
    ),
    asset_root: Path = typer.Option(Path("third_party/robot_hands/artimano"), "--asset-root"),
    index_path: Path = typer.Option(Path(".local/index/grab"), "--index"),
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_wall_time: int = typer.Option(1800, "--max-wall-time", min=1),
    generate_html: bool = typer.Option(True, "--generate-html/--no-generate-html"),
    skip_unit: list[str] = typer.Option([], "--skip-unit"),
) -> None:
    """Run A--E with frozen profiles, checkpoints, reports, and HTML."""
    del config
    _echo(
        run_a_to_e(
            grab_root=grab_root,
            mano_root=mano_root,
            asset_root=asset_root,
            index_path=index_path,
            experiment_root=experiment_root,
            resume=resume,
            max_wall_time=max_wall_time,
            generate_html=generate_html,
            skip_units=frozenset(skip_unit),
        )
    )


@app.command("run-baselines")
def run_baselines() -> None:
    """Compatibility entry point; the orchestrator enforces A--E dependencies."""
    run_a_to_e_command()


@app.command("run-morphology-warm")
def run_morphology_warm() -> None:
    run_a_to_e_command()


@app.command("run-contact-final")
def run_contact_final() -> None:
    run_a_to_e_command()


@app.command("run-matrix")
def run_matrix() -> None:
    run_a_to_e_command()


@app.command("evaluate")
def evaluate(
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
) -> None:
    _echo(read_json(experiment_root / "reports" / "experiment_summary.json"))


@app.command("visualize-mesh")
def visualize_mesh(
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
) -> None:
    _echo(
        {
            "html_root": str((experiment_root / "html").resolve()),
            "status": "generated_by_run_a_to_e",
        }
    )


@app.command("status")
def status(
    experiment_root: Path = typer.Option(
        Path(f".local/experiments/{EXPERIMENT_ID}"), "--experiment-root"
    ),
) -> None:
    reports = experiment_root / "reports"
    for name in ("experiment_summary.json", "experiment_status.json", "failure_report.json"):
        candidate = reports / name
        if candidate.is_file():
            payload = read_json(candidate)
            payload.setdefault("status_artifact", str(candidate.resolve()))
            _echo(payload)
            return
    _echo({"status": "not_started", "experiment_root": str(experiment_root.resolve())})


__all__ = ["app"]
