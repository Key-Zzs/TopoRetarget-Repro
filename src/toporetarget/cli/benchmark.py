"""Q1--Q3 benchmark CLI: selection, freeze, execution, evaluation, dashboard."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from toporetarget.benchmark.contactpose import ContactPoseDatasetAdapter
from toporetarget.benchmark.dashboard import build_dashboard
from toporetarget.benchmark.evaluate import evaluate_benchmark, write_selection_blocked_reports
from toporetarget.benchmark.runner import PROFILES, run_benchmark, verify_frozen_manifest
from toporetarget.benchmark.schema import read_json, write_json
from toporetarget.benchmark.selection import freeze_selection, select_contactpose, select_grab

app = typer.Typer(help="Frozen multi-dataset interaction benchmark and automatic evaluation.")


def _echo(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command("inspect-datasets")
def inspect_datasets(
    grab_root: Path = typer.Option(..., "--grab-root"),
    contactpose_root: Path = typer.Option(..., "--contactpose-root"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Inspect roots and schemas without changing raw data."""

    adapter = ContactPoseDatasetAdapter(contactpose_root)
    payload = {
        "schema_version": "toporetarget.benchmark.dataset_audit.v1",
        "grab": {
            "root": str(grab_root.resolve()),
            "exists": grab_root.is_dir(),
            "expected_layout": {
                "grab": (grab_root / "grab").is_dir(),
                "object_meshes": (grab_root / "tools" / "object_meshes").is_dir(),
            },
        },
        "contactpose": adapter.inspect(),
    }
    write_json(payload, output)
    _echo(payload)


@app.command("select")
def select(
    config: Path = typer.Option(..., "--config"),
    grab_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB"), "--grab-root"
    ),
    contactpose_root: Path = typer.Option(
        Path("/mnt/nas/storage/Ref2Dex_storage/ContactPose/data"), "--contactpose-root"
    ),
    grab_additional_target: int = typer.Option(3, "--grab-additional-target", min=1),
    grab_additional_minimum: int = typer.Option(2, "--grab-additional-minimum", min=1),
    contactpose_target: int = typer.Option(4, "--contactpose-target", min=1),
    contactpose_minimum: int = typer.Option(3, "--contactpose-minimum", min=1),
    output_root: Path = typer.Option(Path(".local/benchmarks/hoi_benchmark_v1"), "--output-root"),
    grab_index: Path = typer.Option(Path(".local/index/grab"), "--grab-index"),
    grab_scan_limit: int = typer.Option(16, "--grab-scan-limit", min=3),
) -> None:
    """Deterministically select candidates before any baseline run."""

    output_root.mkdir(parents=True, exist_ok=True)
    grab_result = select_grab(
        grab_root=grab_root,
        output_root=output_root,
        additional_target=grab_additional_target,
        additional_minimum=grab_additional_minimum,
        index_path=grab_index,
        scan_limit=grab_scan_limit,
    )
    contactpose_result = select_contactpose(
        root=contactpose_root,
        output_root=output_root,
        target=contactpose_target,
        minimum=contactpose_minimum,
    )
    payload = {"config": str(config), "grab": grab_result, "contactpose": contactpose_result}
    write_json(payload, output_root / "selection_result.json")
    write_json(payload, output_root / "benchmark_config_resolved.json")
    if grab_result["status"] != "pass" or contactpose_result["status"] != "pass":
        write_selection_blocked_reports(
            benchmark_root=output_root,
            selection_result=payload,
        )
        build_dashboard(output_root)
    _echo(payload)
    if grab_result["status"] != "pass" or contactpose_result["status"] != "pass":
        raise typer.Exit(code=2)


@app.command("freeze")
def freeze(
    benchmark_root: Path = typer.Option(
        Path(".local/benchmarks/hoi_benchmark_v1"), "--benchmark-root"
    ),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    """Freeze selection and write benchmark units, manifest, and lock."""

    payload = read_json(benchmark_root / "selection_result.json")
    result = freeze_selection(
        grab_result=payload["grab"],
        contactpose_result=payload["contactpose"],
        config=payload.get("config", {}),
        output_root=benchmark_root,
        repo_root=repo_root,
    )
    _echo(
        {
            "manifest_hash": result["manifest_hash"],
            "unit_count": len(result["selected_units"]),
            "lock": str(benchmark_root / "benchmark_selection.lock"),
        }
    )


@app.command("run")
def run(
    benchmark_root: Path = typer.Option(
        Path(".local/benchmarks/hoi_benchmark_v1"), "--benchmark-root"
    ),
    run_root: Path = typer.Option(Path(".local/runs/benchmark_q1_q3"), "--run-root"),
    profiles: str = typer.Option(",".join(PROFILES), "--profiles"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_wall_time: int = typer.Option(1800, "--max-wall-time", min=1),
) -> None:
    """Run only manifest-bound frozen profiles and preserve failures."""

    result = run_benchmark(
        benchmark_root=benchmark_root,
        run_root=run_root,
        profiles=[item for item in profiles.split(",") if item],
        resume=resume,
        max_wall_time=max_wall_time,
        repo_root=Path("."),
    )
    _echo(
        {
            "run_count": len(result["runs"]),
            "complete_count": sum(item["status"] == "complete" for item in result["runs"]),
            "manifest_hash": result["selection_manifest_hash"],
        }
    )


@app.command("evaluate")
def evaluate(
    benchmark_root: Path = typer.Option(
        Path(".local/benchmarks/hoi_benchmark_v1"), "--benchmark-root"
    ),
    run_root: Path = typer.Option(Path(".local/runs/benchmark_q1_q3"), "--run-root"),
    metric_registry: Path = typer.Option(
        Path("configs/metrics/hoi_metrics_v1.yaml"), "--metric-registry"
    ),
    html: bool = typer.Option(False, "--html"),
) -> None:
    """Evaluate metrics, gates, aggregation, and an optional dashboard."""

    del metric_registry
    result = evaluate_benchmark(benchmark_root=benchmark_root, run_root=run_root)
    if html:
        result["dashboard"] = str(build_dashboard(benchmark_root))
    _echo({"summary": result["benchmark_summary"], "dashboard": result.get("dashboard")})


@app.command("status")
def status(
    benchmark_root: Path = typer.Option(
        Path(".local/benchmarks/hoi_benchmark_v1"), "--benchmark-root"
    ),
) -> None:
    """Report frozen manifest, run, and evaluation state."""

    if (benchmark_root / "benchmark_selection_manifest.json").is_file():
        manifest = verify_frozen_manifest(benchmark_root)
        payload = {
            "manifest_hash": manifest["manifest_hash"],
            "selected_units": len(manifest.get("selected_units", [])),
            "selection_frozen": True,
        }
    else:
        payload = {
            "manifest_hash": None,
            "selected_units": 0,
            "selection_frozen": False,
        }
    for name in ("benchmark_status.json", "benchmark_summary.json", "failure_report.json"):
        path = benchmark_root / name
        if path.is_file():
            payload[name] = read_json(path)
    _echo(payload)


@app.command("dashboard")
def dashboard(
    benchmark_root: Path = typer.Option(
        Path(".local/benchmarks/hoi_benchmark_v1"), "--benchmark-root"
    ),
) -> None:
    """Build the self-contained HTML benchmark dashboard."""

    _echo({"dashboard": str(build_dashboard(benchmark_root))})


__all__ = ["app"]
