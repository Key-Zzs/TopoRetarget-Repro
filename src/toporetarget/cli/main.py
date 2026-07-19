from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.cli.data import app as data_app
from toporetarget.cli.doctor import app as doctor_app
from toporetarget.cli.geometry import app as geometry_app
from toporetarget.cli.keypoints import app as keypoints_app
from toporetarget.cli.robots import app as robots_app
from toporetarget.config.loader import load_path_config
from toporetarget.paths.assets import AssetImportError, import_artimano

app = typer.Typer(
    name="toporetarget",
    help="Paper-traceable TopoRetarget reproduction tooling.",
    no_args_is_help=True,
)
app.add_typer(doctor_app, name="doctor")
app.add_typer(data_app, name="data")
app.add_typer(keypoints_app, name="keypoints")
app.add_typer(robots_app, name="robots")
app.add_typer(geometry_app, name="geometry")
assets_app = typer.Typer(help="Manage local robot assets.")
app.add_typer(assets_app, name="assets")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@assets_app.command("import-artimano")
def import_artimano_command(
    source_root: Path | None = typer.Option(None, "--source-root", help="ManipTrans checkout."),
    destination: Path | None = typer.Option(None, "--destination", help="Local asset destination."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and report without copying."),
    force: bool = typer.Option(False, "--force", help="Replace an existing destination."),
) -> None:
    repo_root = _repo_root()
    config = load_path_config(
        repo_root,
        overrides={"maniptrans_root": source_root, "artimano_asset_root": destination},
    )
    try:
        result = import_artimano(
            config.maniptrans_root,
            config.artimano_asset_root,
            dry_run=dry_run,
            force=force,
        )
    except (AssetImportError, OSError) as exc:
        typer.echo(f"Arti-MANO import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))
