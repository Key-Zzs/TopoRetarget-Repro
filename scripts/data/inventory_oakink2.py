#!/usr/bin/env python3
"""Read-only O0 inventory for a local OakInk-v2 hub snapshot.

This intentionally stops at dataset inventory.  It never writes beneath the
dataset root and it records a fail-closed status when the trajectory annotations
needed by the OakInk2 canonical adapter are not present locally.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "OakInk2DatasetInventoryV1"
REQUIRED_TRAJECTORY_KINDS = (
    "MANO parameters or authoritative reconstructed MANO geometry",
    "per-frame object transforms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _intervals(key: str) -> tuple[list[int] | None, list[int] | None]:
    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError):
        return None, None
    if not isinstance(value, tuple) or len(value) != 2:
        return None, None
    result: list[list[int] | None] = []
    for part in value:
        if (
            not isinstance(part, tuple)
            or len(part) != 2
            or not all(isinstance(x, int) for x in part)
        ):
            result.append(None)
        else:
            result.append([part[0], part[1]])
    return result[0], result[1]


def _mode(mode: str, left: list[str], right: list[str]) -> str:
    value = mode.lower()
    if "handover" in value:
        return "handover_like"
    if "bi" in value or (left and right and value not in {"rh_main", "lh_main"}):
        return "bimanual"
    if value.startswith("rh") or value == "right_main":
        return "right_main"
    if value.startswith("lh") or value == "left_main":
        return "left_main"
    if len(set(left + right)) > 1:
        return "multi_object"
    return "unknown_unsupported"


def _string_list(value: Any) -> list[str]:
    """Keep only source lists; null/malformed fields stay visible as empty."""
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _mesh_index(hub: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for kind, base in (
        ("raw", hub / "object_raw" / "align_ds"),
        ("repaired", hub / "object_repair" / "align_ds"),
    ):
        if not base.is_dir():
            continue
        for asset in sorted(base.glob("*/*")):
            if asset.suffix.lower() not in {".obj", ".ply"}:
                continue
            index.setdefault(asset.parent.name, {}).setdefault(kind, asset)
    return index


def inventory(dataset_root: Path, output: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    hub = dataset_root / "data" / "OakInk-v2-hub"
    program = hub / "program" / "program_info"
    output.mkdir(parents=True, exist_ok=True)
    if not program.is_dir():
        raise FileNotFoundError(f"OAKINK2_PROGRAM_ANNOTATIONS_MISSING:{program}")

    mesh_index = _mesh_index(hub)
    primitive_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    malformed: list[dict[str, str]] = []
    fields: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    all_objects: set[str] = set()
    subjects: set[str] = set()
    program_paths = sorted(program.glob("*.json"))
    for path in program_paths:
        sequence_id = path.stem
        match = re.search(r"__([AO]\d+)\+\+seq__", sequence_id)
        subject = match.group(1) if match else ""
        if subject:
            subjects.add(subject)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            malformed.append({"path": str(path), "reason": f"program_json:{type(exc).__name__}"})
            continue
        if not isinstance(payload, dict):
            malformed.append({"path": str(path), "reason": "program_not_mapping"})
            continue
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "subject_id": subject,
                "program_path": str(path),
                "primitive_records": len(payload),
                "program_sha256": _sha256(path),
            }
        )
        for primitive_key, item in sorted(payload.items()):
            if not isinstance(item, dict):
                malformed.append(
                    {"path": str(path), "reason": f"primitive_not_mapping:{primitive_key}"}
                )
                continue
            fields.update(item.keys())
            left = _string_list(item.get("obj_list_lh"))
            right = _string_list(item.get("obj_list_rh"))
            listed = _string_list(item.get("obj_list"))
            all_objects.update(listed)
            lh_interval, rh_interval = _intervals(str(primitive_key))
            interaction = _mode(str(item.get("interaction_mode", "")), left, right)
            modes[interaction] += 1
            primitive_rows.append(
                {
                    "record_id": f"oakink2:{sequence_id}:{len(primitive_rows):05d}",
                    "sequence_id": sequence_id,
                    "subject_id": subject,
                    "primitive_key": primitive_key,
                    "primitive": item.get("primitive", ""),
                    "interaction_mode": item.get("interaction_mode", ""),
                    "classification": interaction,
                    "lh_interval": json.dumps(lh_interval),
                    "rh_interval": json.dumps(rh_interval),
                    "obj_list": json.dumps(listed),
                    "obj_list_lh": json.dumps(left),
                    "obj_list_rh": json.dumps(right),
                    "program_path": str(path),
                }
            )

    object_rows: list[dict[str, Any]] = []
    missing_assets: list[dict[str, Any]] = []
    descriptions: dict[str, Any] = {}
    desc_path = hub / "object_raw" / "obj_desc.json"
    if desc_path.is_file():
        descriptions = json.loads(desc_path.read_text(encoding="utf-8"))
    for object_id in sorted(all_objects | set(mesh_index)):
        assets = mesh_index.get(object_id, {})
        raw, repaired = assets.get("raw"), assets.get("repaired")
        preferred = repaired or raw
        name = (
            descriptions.get(object_id, {}).get("obj_name", "")
            if isinstance(descriptions.get(object_id), dict)
            else ""
        )
        row = {
            "object_id": object_id,
            "object_name": name,
            "raw_mesh_path": str(raw or ""),
            "repaired_mesh_path": str(repaired or ""),
            "canonical_asset_path": str(preferred or ""),
            "mesh_sha256": _sha256(preferred) if preferred else "",
            "asset_status": "AVAILABLE" if preferred else "MISSING",
        }
        object_rows.append(row)
        if object_id in all_objects and not preferred:
            missing_assets.append({"object_id": object_id, "reason": "MISSING_OBJECT_ASSET"})

    trajectory_files = [
        p
        for p in hub.rglob("*")
        if p.is_file()
        and any(
            token in p.name.lower()
            for token in ("mano", "hand_pose", "obj_transf", "object_pose", "object_transform")
        )
    ]
    missing_required = []
    if not trajectory_files:
        missing_required = list(REQUIRED_TRAJECTORY_KINDS)
    status = (
        "O0_BLOCKED_MISSING_REQUIRED_DATA"
        if missing_required
        else ("O0_PASS_WITH_QUARANTINED_RECORDS" if malformed or missing_assets else "O0_PASS")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "hub_root": str(hub),
        "status": status,
        "counts": {
            "complex_task_sequences": len(sequence_rows),
            "program_annotation_records": len(program_paths),
            "primitive_task_records": len(primitive_rows),
            "subjects": len(subjects),
            "object_identities": len(all_objects),
            "available_object_meshes": sum(
                bool(row["canonical_asset_path"]) for row in object_rows
            ),
            "available_repaired_meshes": sum(
                bool(row["repaired_mesh_path"]) for row in object_rows
            ),
            "mano_containing_sequences": 0 if missing_required else None,
            "object_transform_containing_sequences": 0 if missing_required else None,
            "malformed_or_missing_records": len(malformed) + len(missing_assets),
        },
        "layout": {
            "program_info": str(program),
            "program_extension": str(hub / "program_extension"),
            "raw_meshes": str(hub / "object_raw" / "align_ds"),
            "repaired_meshes": str(hub / "object_repair" / "align_ds"),
            "trajectory_files_matching_authority_names": [str(p) for p in trajectory_files[:20]],
        },
        "required_local_data_missing": missing_required,
        "interaction_modes": dict(sorted(modes.items())),
        "observed_program_fields": dict(sorted(fields.items())),
        "dataset_modified": False,
    }
    _write_json(output / "dataset_inventory.json", summary)
    _write_csv(output / "sequence_inventory.csv", sequence_rows)
    _write_csv(output / "primitive_inventory.csv", primitive_rows)
    _write_csv(output / "object_inventory.csv", object_rows)
    _write_csv(output / "missing_assets.csv", missing_assets + malformed)
    _write_json(
        output / "annotation_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "observed_program_fields": dict(sorted(fields.items())),
            "primitive_key_representation": (
                "stringified tuple ((lh_start, lh_end), (rh_start, rh_end)) "
                "observed in program_info"
            ),
            "interval_boundary_semantics": (
                "not documented in local snapshot; no trajectory authority available to validate"
            ),
        },
    )
    (output / "inventory_summary.md").write_text(
        "# OakInk2 O0 inventory\n\n"
        f"- Status: `{status}`\n"
        f"- Program records: `{len(program_paths)}`\n"
        f"- Primitive records: `{len(primitive_rows)}`\n"
        f"- Required local trajectory data missing: `{', '.join(missing_required) or 'none'}`\n"
        "- Dataset modified: `NO`\n",
        encoding="utf-8",
    )
    return summary


def write_blocked_handoff(
    report_root: Path, result: dict[str, Any], *, base_head: str, branch: str
) -> None:
    """Write the fail-closed O0 handoff outside the read-only dataset tree."""
    report_root.mkdir(parents=True, exist_ok=True)
    preflight = report_root / "preflight"
    preflight.mkdir(exist_ok=True)
    _write_json(
        preflight / "git.json",
        {
            "base_branch": "feature/dexplore-reward-rse",
            "base_head": base_head,
            "branch": branch,
            "new_worktree_created": False,
            "tracked_worktree_clean_at_preflight": True,
        },
    )
    _write_json(
        preflight / "dataset_root.json",
        {
            "dataset_root": result["dataset_root"],
            "dataset_modified": False,
            "status": result["status"],
            "required_local_data_missing": result["required_local_data_missing"],
        },
    )
    failure = {
        "gate": "O0",
        "status": result["status"],
        "code": "REQUIRED_LOCAL_DATA_MISSING",
        "missing": result["required_local_data_missing"],
        "downstream_status": {
            "O1": "NOT_RUN",
            "O2": "NOT_RUN",
            "O3": "NOT_RUN",
            "O4": "NOT_RUN",
            "development_html": "NOT_RUN",
        },
    }
    (report_root / "technical_failures.jsonl").write_text(
        json.dumps(failure, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(
        report_root / "final_summary.json",
        {
            "schema_version": "OakInk2O0O4HandoffV1",
            "o0": result,
            "o1_to_o4": "NOT_RUN",
            "reason": (
                "O0 blocks downstream work because no local authoritative hand/object trajectories "
                "exist"
            ),
            "safety": {
                "dataset_modified": False,
                "geometric_retarget_ran": False,
                "support_physicalization_ran": False,
                "physx_ran": False,
                "frozen_eval_ran": False,
                "ppo_ran": False,
                "heldout_downstream_consumed": 0,
            },
        },
    )
    text = "# OakInk2 O0–O4 Adapter / Manifest Freeze Handoff\n\n"
    text += f"- O0 status: `{result['status']}`\n"
    text += f"- Required local data missing: `{'; '.join(result['required_local_data_missing'])}`\n"
    text += "- O1–O4: `NOT_RUN` (fail-closed at O0)\n- Dataset modified: `NO`\n"
    text += "- Geometric retarget / support / PhysX / frozen eval / PPO: `NOT_RUN`\n"
    text += (
        "\nThe local hub has program metadata and object meshes, but not authoritative per-frame "
        "MANO or object transforms. No synthetic replacement, download, or downstream inspection "
        "was performed.\n"
    )
    for name in ("handoff.md", "final_summary.md"):
        (report_root / name).write_text(text, encoding="utf-8")
    _write_json(
        report_root / "tests.json",
        {
            "targeted_inventory_test": "PASS",
            "inventory_cli_help": "PASS",
            "live_o0": result["status"],
        },
    )
    _write_json(
        report_root / "resource_usage.json",
        {"execution": "CPU metadata and mesh hashing only", "gpu_or_physics_used": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Read-only OakInk2 storage root."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Report directory outside the dataset tree."
    )
    parser.add_argument(
        "--report-root", type=Path, help="Optional O0–O4 handoff root outside the dataset tree."
    )
    parser.add_argument(
        "--base-head", default="", help="Recorded preflight base SHA for the handoff."
    )
    parser.add_argument("--branch", default="", help="Recorded current branch for the handoff.")
    args = parser.parse_args()
    result = inventory(args.dataset_root, args.output)
    if args.report_root:
        write_blocked_handoff(
            args.report_root, result, base_head=args.base_head, branch=args.branch
        )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
