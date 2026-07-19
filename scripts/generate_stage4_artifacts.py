"""Generate the opt-in local Arti-MANO Stage 4 reports and PNG artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from toporetarget.paths.assets import check_artimano_assets
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.reports import jacobian_check, write_json
from toporetarget.robots.visualization import render_robot_hand, render_robot_pair
from toporetarget.utils.hashing import sha256_file


def _mesh_aggregate(manifest: dict[str, object]) -> str:
    records = manifest.get("imported_files", [])
    values = []
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and str(record.get("path", "")).lower().endswith(".obj"):
                values.append((str(record.get("path")), str(record.get("sha256"))))
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _random_q(model: object, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.where(np.isfinite(model.joint_lower), model.joint_lower, -np.pi)
    upper = np.where(np.isfinite(model.joint_upper), model.joint_upper, np.pi)
    return lower + rng.uniform(0.1, 0.9, size=model.num_dofs) * (upper - lower)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    output = repo / ".local" / "reports" / "stage4"
    output.mkdir(parents=True, exist_ok=True)
    models = {side: load_artimano_model(side) for side in ("right", "left")}
    for side, model in models.items():
        prefix = "artimano_rh" if side == "right" else "artimano_lh"
        random_q = _random_q(model)
        write_json(model.describe(), output / f"{prefix}_inspect.json")
        validation = model.validate(seed=4, dtype="float64")
        validation.write_json(output / f"{prefix}_validation.json")
        validation.write_csv(output / f"{prefix}_validation.csv")
        rows = []
        points = model.keypoints_base(model.neutral_q).detach().cpu().numpy()
        for index, (anchor, point) in enumerate(
            zip(model.anchor_profile.anchors, points, strict=True)
        ):
            rows.append(
                {
                    "index": index,
                    "semantic": anchor.semantic_name,
                    "anchor_type": anchor.anchor_type,
                    "joint": anchor.joint_name or "",
                    "link": anchor.link_name or "",
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "assumptions": ";".join(anchor.assumptions),
                }
            )
        import csv

        with (output / f"{prefix}_anchors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            jacobian_check(model, random_q, dtype="float64"), output / f"{prefix}_jacobian.json"
        )
        render_robot_hand(
            model,
            model.neutral_q,
            geometry="visual",
            output=output / f"{prefix}_neutral_visual.png",
            show_keypoints=True,
            show_skeleton=True,
            show_labels=True,
            show_base_frame=True,
        )
        render_robot_hand(
            model,
            model.neutral_q,
            geometry="collision",
            output=output / f"{prefix}_neutral_collision.png",
            show_keypoints=True,
            show_skeleton=True,
        )
        render_robot_hand(
            model,
            random_q,
            geometry="both",
            output=output / f"{prefix}_random_overlay.png",
            show_keypoints=True,
            show_skeleton=True,
            show_labels=True,
            show_joint_axes=True,
            title_suffix=" | pose=random seed=4",
        )
    render_robot_pair(models["left"], models["right"], output=output / "artimano_both_neutral.png")

    asset_root = models["right"].asset_root
    assert asset_root is not None
    manifest_path = asset_root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check = check_artimano_assets(asset_root)
    write_json(
        {
            "status": check.status,
            "asset_root": str(asset_root),
            "upstream_commit": manifest.get("upstream_commit"),
            "asset_manifest_hash": sha256_file(manifest_path),
            "rh_urdf_hash": sha256_file(asset_root / "rh_mano.urdf"),
            "lh_urdf_hash": sha256_file(asset_root / "lh_mano.urdf"),
            "imported_file_count": len(manifest.get("imported_files", [])),
            "mesh_file_count": sum(
                1
                for record in manifest.get("imported_files", [])
                if str(record.get("path", "")).lower().endswith(".obj")
            ),
            "mesh_aggregate_hash": _mesh_aggregate(manifest),
            "unresolved_mesh_references": check.missing_mesh_references,
            "unchanged": check.status == "ok" and manifest.get("modified") is False,
            "changed_files": check.changed_files,
        },
        output / "asset_integrity.json",
    )


if __name__ == "__main__":
    main()
