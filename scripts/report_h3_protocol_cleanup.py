#!/usr/bin/env python3
"""Record superseded H3 protocol authorities while retaining historical evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-protocol", type=Path, required=True)
    parser.add_argument("--h3-protocol-hash", type=Path, required=True)
    parser.add_argument("--h3a-decision", type=Path, required=True)
    parser.add_argument("--h3b-decision", type=Path, required=True)
    parser.add_argument("--h3d-supersession", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H3_PROTOCOL_CLEANUP_JSON_OBJECT_REQUIRED:{path}")
    return value


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3_PROTOCOL_CLEANUP_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3_PROTOCOL_CLEANUP_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build(
    protocol_path: Path,
    protocol_hash_path: Path,
    h3a_path: Path,
    h3b_path: Path,
    supersession_path: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = _json(protocol_path.resolve())
    protocol_hash = protocol_hash_path.resolve().read_text(encoding="utf-8").strip()
    h3a = _json(h3a_path.resolve())
    h3b = _json(h3b_path.resolve())
    supersession = _json(supersession_path.resolve())
    if (
        protocol.get("schema_version") != "H3PhysicalizationProtocolV1"
        or _stable_hash(protocol) != protocol_hash
    ):
        raise ValueError("H3_PROTOCOL_CLEANUP_PROTOCOL_INVALID")
    if supersession.get("status") != "SUPERSEDED_FOR_UNSEEN_OBJECT_SPLIT":
        raise ValueError("H3_PROTOCOL_CLEANUP_P6_SUPERSESSION_INVALID")
    selected_source = str(h3a.get("H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT", ""))
    selected_retarget = str(h3b.get("H3B_SELECTED_RETARGET_EXECUTION_CONTRACT", ""))
    if not selected_source or not selected_retarget:
        raise ValueError("H3_PROTOCOL_CLEANUP_SELECTED_AUTHORITIES_MISSING")

    inventory = [
        {
            "protocol": "old SourceControllerMode.AUTO task-fidelity gate",
            "historical_role": "production source admission",
            "classification": "SUPERSEDED",
            "action": "retain reports; replace production authority",
        },
        {
            "protocol": "L0 0/10 task success implies source hard failure",
            "historical_role": "binary source gate",
            "classification": "SUPERSEDED",
            "action": "task fidelity becomes non-gating diagnostic",
        },
        {
            "protocol": "post-freeze unbounded L0",
            "historical_role": "diagnostic controller experiment",
            "classification": "HISTORICAL_DIAGNOSTIC",
            "action": "retain evidence; forbid production use",
        },
        {
            "protocol": "old P6 seen-object Frozen5",
            "historical_role": "metadata-only candidate set",
            "classification": "SUPERSEDED_FOR_UNSEEN_OBJECT_SPLIT",
            "action": "retain manifest and hash; freeze object-disjoint replacement",
        },
        {
            "protocol": "binary source-controller executability/fidelity semantics",
            "historical_role": "combined qualification",
            "classification": "SUPERSEDED",
            "action": "separate ExecutableV2 hard gate from FidelityV2 diagnostic",
        },
    ]
    superseded = [
        {
            "protocol": row["protocol"],
            "current_production_authority": authority,
            "disposition": row["action"],
        }
        for row, authority in zip(
            inventory,
            (
                selected_source,
                "SourceControllerExecutableV2",
                "bounded corrected L0 only after zero-residual non-executability",
                "H3UnseenObjectFrozen5ManifestV1",
                "SourceControllerExecutableV2+SourceControllerFidelityV2",
            ),
            strict=True,
        )
    ]
    retained = [
        {
            "path_or_class": ".local/reports/raw_to_physical_hardening_v2/",
            "reason_retained": "historical hardening and source-gate evidence",
            "current_production_authority": "NO",
        },
        {
            "path_or_class": str(supersession["historical_manifest_path"]),
            "reason_retained": "old P6 manifest and hash provenance",
            "current_production_authority": "NO",
        },
        {
            "path_or_class": "scripts/rl/isaaclab/replay_physical_hoi_trace.py",
            "reason_retained": "required physical trace replay utility",
            "current_production_authority": "YES_REPLAY_ONLY",
        },
        {
            "path_or_class": "scripts/rl/visualize_hocap_world_wrist_policy_mujoco.py",
            "reason_retained": "required source visualization utility",
            "current_production_authority": "YES_VISUALIZATION_ONLY",
        },
    ]
    current = {
        "schema_version": "H3ProtocolCleanupCurrentAuthoritiesV1",
        "H3_PROTOCOL_HASH": protocol_hash,
        "H3_EXECUTION_HEAD": protocol["freeze"]["H3_EXECUTION_HEAD"],
        "retarget_execution": selected_retarget,
        "source_controller": selected_source,
        "source_admission": "SourceControllerExecutableV2",
        "source_fidelity": "SourceControllerFidelityV2_DIAGNOSTIC_NON_GATING",
        "unbounded_l0_production": False,
        "held_out_split": "UNSEEN_OBJECT_INSTANCE_HELDOUT",
        "historical_reports_deleted": False,
        "visualization_scripts_deleted": False,
    }
    destination = output.resolve()
    _write_csv(destination / "inventory.csv", inventory)
    _write_csv(destination / "superseded.csv", superseded)
    _write_csv(destination / "retained_history.csv", retained)
    _write_json(destination / "current_authorities.json", current)
    return current


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            build(
                args.h3_protocol,
                args.h3_protocol_hash,
                args.h3a_decision,
                args.h3b_decision,
                args.h3d_supersession,
                args.output_root,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
