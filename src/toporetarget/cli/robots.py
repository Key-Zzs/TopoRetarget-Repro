"""CLI commands for generic robot-hand models and Arti-MANO inspection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from toporetarget.robots.registry import get_robot_registry
from toporetarget.robots.reports import jacobian_check, write_json
from toporetarget.robots.visualization import render_robot_hand

app = typer.Typer(help="Inspect generic URDF robot hands and Arti-MANO target adapters.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _output_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else _repo_root() / path


def _load_model(robot: str, asset_root: Path | None):
    return get_robot_registry(repo_root=_repo_root()).load(robot, asset_root=asset_root)


def _json_print(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _read_qpos(path: Path, model: Any) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        value = loaded.get("qpos", loaded) if isinstance(loaded, dict) else loaded
    if isinstance(value, dict):
        return np.asarray([float(value[name]) for name in model.dof_names], dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (model.num_dofs,):
        raise typer.BadParameter(
            f"qpos file must contain shape [{model.num_dofs}], got {result.shape}"
        )
    return result


def _pose_q(model: Any, pose: str, seed: int, qpos_file: Path | None) -> np.ndarray:
    if qpos_file is not None:
        return _read_qpos(qpos_file, model)
    if pose == "neutral":
        return model.neutral_q
    if pose == "random":
        rng = np.random.default_rng(seed)
        lower, upper = model.joint_lower, model.joint_upper
        lower = np.where(np.isfinite(lower), lower, -np.pi)
        upper = np.where(np.isfinite(upper), upper, np.pi)
        return lower + rng.uniform(0.1, 0.9, size=model.num_dofs) * (upper - lower)
    raise typer.BadParameter("pose must be neutral or random")


@app.command("list")
def list_robots(
    asset_root: Path | None = typer.Option(
        None, "--asset-root", help="Override local asset root; no URDF is parsed."
    ),
) -> None:
    """List YAML-registered robots without loading URDFs or meshes."""

    _json_print(get_robot_registry(repo_root=_repo_root()).list(asset_root=asset_root))


@app.command("inspect")
def inspect_robot(
    robot: str = typer.Option(..., "--robot"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    json_path: Path | None = typer.Option(None, "--json"),
) -> None:
    try:
        model = _load_model(robot, asset_root)
        result = model.describe()
        if json_path is not None:
            target = _output_path(json_path)
            assert target is not None
            write_json(result, target)
        _json_print(result)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        typer.echo(f"robot inspect failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate")
def validate_robot(
    robot: str = typer.Option(..., "--robot"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    report: Path | None = typer.Option(None, "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    seed: int = typer.Option(4, "--seed"),
    dtype: str = typer.Option("float64", "--dtype"),
) -> None:
    try:
        model = _load_model(robot, asset_root)
        result = model.validate(seed=seed, dtype=dtype)
        if report is not None:
            target = _output_path(report)
            assert target is not None
            result.write_json(target)
        if csv_report is not None:
            target = _output_path(csv_report)
            assert target is not None
            result.write_csv(target)
        _json_print(result.as_dict())
        if result.status != "pass":
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        typer.echo(f"robot validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("fk")
def fk(
    robot: str = typer.Option(..., "--robot"),
    pose: str = typer.Option("neutral", "--pose"),
    seed: int = typer.Option(4, "--seed"),
    qpos_file: Path | None = typer.Option(None, "--qpos-file"),
    dtype: str = typer.Option("float64", "--dtype"),
    device: str = typer.Option("cpu", "--device"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    try:
        import torch

        model = _load_model(robot, None)
        q = _pose_q(model, pose, seed, qpos_file)
        torch_dtype = getattr(torch, dtype)
        q_tensor = torch.tensor(q, dtype=torch_dtype, device=device)
        transforms = model.forward_kinematics_base(q_tensor)
        result = {
            "robot": robot,
            "pose": pose,
            "seed": seed,
            "dtype": dtype,
            "device": device,
            "qpos": model.qpos_to_named_dict(q_tensor.cpu()),
            "link_transforms_base": {
                name: value.detach().cpu().tolist() for name, value in transforms.items()
            },
            "anchors_base": model.keypoints_base(q_tensor).detach().cpu().tolist(),
            "anchor_metadata": model.keypoint_metadata(),
            "profile_hash": model.anchor_profile.sha256,
        }
        if output is not None:
            target = _output_path(output)
            assert target is not None
            write_json(result, target)
        _json_print(result)
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"robot FK failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("anchors")
def anchors(
    robot: str = typer.Option(..., "--robot"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    json_path: Path | None = typer.Option(None, "--json"),
    csv_path: Path | None = typer.Option(None, "--csv"),
) -> None:
    try:
        model = _load_model(robot, asset_root)
        points = model.keypoints_base(model.neutral_q).detach().cpu().numpy()
        rows = []
        for index, (anchor, point) in enumerate(
            zip(model.anchor_profile.anchors, points, strict=True)
        ):
            rows.append(
                {
                    "index": index,
                    "semantic": anchor.semantic_name,
                    "anchor_type": anchor.anchor_type,
                    "joint": anchor.joint_name,
                    "link": anchor.link_name,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "assumptions": list(anchor.assumptions),
                }
            )
        if json_path is not None:
            target = _output_path(json_path)
            assert target is not None
            write_json(
                {
                    "robot": robot,
                    "profile": model.anchor_profile.as_dict(),
                    "profile_hash": model.anchor_profile.sha256,
                    "anchors": rows,
                },
                target,
            )
        if csv_path is not None:
            target = _output_path(csv_path)
            assert target is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "index",
                        "semantic",
                        "anchor_type",
                        "joint",
                        "link",
                        "x",
                        "y",
                        "z",
                        "assumptions",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    row = dict(row)
                    row["assumptions"] = ";".join(row["assumptions"])
                    writer.writerow(row)
        _json_print({"robot": robot, "profile_hash": model.anchor_profile.sha256, "anchors": rows})
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        typer.echo(f"robot anchors failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("jacobian-check")
def jacobian_check_command(
    robot: str = typer.Option(..., "--robot"),
    pose: str = typer.Option("random", "--pose"),
    seed: int = typer.Option(4, "--seed"),
    qpos_file: Path | None = typer.Option(None, "--qpos-file"),
    dtype: str = typer.Option("float64", "--dtype"),
    epsilon: float = typer.Option(1e-6, "--epsilon"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    try:
        model = _load_model(robot, None)
        q = _pose_q(model, pose, seed, qpos_file)
        result = jacobian_check(model, q, epsilon=epsilon, dtype=dtype)
        result["pose"] = pose
        result["seed"] = seed
        if report is not None:
            target = _output_path(report)
            assert target is not None
            write_json(result, target)
        _json_print(result)
        if not result["passed"]:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        typer.echo(f"robot Jacobian check failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize")
def visualize(
    robot: str = typer.Option(..., "--robot"),
    pose: str = typer.Option("neutral", "--pose"),
    seed: int = typer.Option(4, "--seed"),
    qpos_file: Path | None = typer.Option(None, "--qpos-file"),
    geometry: str = typer.Option("visual", "--geometry"),
    output: Path | None = typer.Option(None, "--output"),
    show: bool = typer.Option(False, "--show"),
    show_keypoints: bool = typer.Option(False, "--show-keypoints"),
    show_skeleton: bool = typer.Option(False, "--show-skeleton"),
    show_labels: bool = typer.Option(False, "--show-labels"),
    show_base_frame: bool = typer.Option(False, "--show-base-frame"),
    show_link_frames: bool = typer.Option(False, "--show-link-frames"),
    show_joint_axes: bool = typer.Option(False, "--show-joint-axes"),
) -> None:
    try:
        model = _load_model(robot, None)
        q = _pose_q(model, pose, seed, qpos_file)
        if output is None and not show:
            raise typer.BadParameter("provide --output or pass --show")
        result = render_robot_hand(
            model,
            q,
            geometry=geometry,
            output=_output_path(output),
            show=show,
            show_keypoints=show_keypoints,
            show_skeleton=show_skeleton,
            show_labels=show_labels,
            show_base_frame=show_base_frame,
            show_link_frames=show_link_frames,
            show_joint_axes=show_joint_axes,
        )
        _json_print(
            {
                "robot": robot,
                "pose": pose,
                "seed": seed,
                "geometry": geometry,
                "output": None if result is None else str(result),
            }
        )
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"robot visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
