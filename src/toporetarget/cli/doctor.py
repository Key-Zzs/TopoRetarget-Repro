from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.config.loader import load_path_config
from toporetarget.paper.fidelity import validate_paper_fidelity
from toporetarget.paths.assets import check_artimano_assets
from toporetarget.paths.datasets import DatasetPathResolver, discovery_report

app = typer.Typer(help="Check local datasets, assets, and paper traceability.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _print_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@app.command("datasets")
def datasets(
    root: Path | None = typer.Option(None, "--root", help="Override Ref2Dex storage root."),
    max_depth: int = typer.Option(
        4, min=0, help="Maximum directory depth below each data directory."
    ),
    json_path: Path | None = typer.Option(
        None, "--json", help="Write the JSON report to this path."
    ),
) -> None:
    _run_datasets(root=root, max_depth=max_depth, json_path=json_path)


def _run_datasets(*, root: Path | None, max_depth: int, json_path: Path | None) -> None:
    repo_root = _repo_root()
    config = load_path_config(repo_root, overrides={"storage_root": root} if root else None)
    resolver = DatasetPathResolver(
        config.storage_root,
        repo_root / "configs" / "datasets" / "registry.yaml",
        max_depth=max_depth,
    )
    results = resolver.discover()
    report = discovery_report(results, storage_root=config.storage_root, max_depth=max_depth)
    output_path = json_path or repo_root / ".local" / "reports" / "dataset_discovery.json"
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        typer.echo(
            f"{result.canonical_dataset_name}: {result.status} "
            f"(candidates={len(result.candidate_directories)}, readable={result.readable})"
        )
    typer.echo(f"report: {output_path}")


@app.command("assets")
def assets(
    destination: Path | None = typer.Option(
        None, "--destination", help="Arti-MANO asset destination."
    ),
) -> None:
    _run_assets(destination=destination)


def _run_assets(*, destination: Path | None) -> None:
    repo_root = _repo_root()
    config = load_path_config(
        repo_root, overrides={"artimano_asset_root": destination} if destination else None
    )
    result = check_artimano_assets(config.artimano_asset_root)
    _print_json(result.as_dict())
    if result.status != "ok":
        raise typer.Exit(code=1)


@app.command("paper")
def paper() -> None:
    _run_paper()


def _run_paper() -> None:
    errors = validate_paper_fidelity(_repo_root())
    if errors:
        typer.echo("paper fidelity: FAILED", err=True)
        for error in errors:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo("paper fidelity: OK")


@app.command("all")
def all_checks() -> None:
    _run_datasets(root=None, max_depth=4, json_path=None)
    _run_assets(destination=None)
    _run_paper()
