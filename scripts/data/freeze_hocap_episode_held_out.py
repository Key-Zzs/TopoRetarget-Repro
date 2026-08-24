#!/usr/bin/env python3
"""Freeze a metadata-only deterministic held-out EpisodeV1 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DEVELOPMENT_SEQUENCE_SUFFIXES = (
    "170105",  # PF/reward/controller development
    "170650",  # PF/support/manual-workflow development
    "165502",  # failed left-backend benchmark diagnostic and visual inspection
    "165807",  # EpisodeV1 left-hand inspection
    "201316",  # production-backend fast_exact_v2 benchmark case B/C and inspection
    "111118",  # superseded held-out pilot with observed outcomes
    "162842",  # superseded held-out pilot with observed outcomes
    "164242",  # superseded held-out pilot with observed outcomes
    "193506",  # superseded held-out pilot with observed outcomes
    "123725",  # superseded held-out pilot with observed outcomes
)
SELECTION_SEED = 20260824


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument(
        "--active-hand",
        choices=("right", "left", "any"),
        default="right",
        help=(
            "Physical-backend capability filter. The current Wuji/Isaac production "
            "runtime is right-hand; EpisodeV1 parsing and geometric retarget support both."
        ),
    )
    parser.add_argument(
        "--exclude-sequence-suffix",
        action="append",
        default=[],
        help="Additional raw-sequence timestamp suffix to exclude; repeatable.",
    )
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--old-manifest-sha256", required=True)
    return parser


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _selection_key(seed: int, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()


def _exclusion_reason(raw_sequence: str, suffixes: Iterable[str]) -> str:
    timestamp = raw_sequence.rsplit("_", 1)[-1]
    if timestamp in set(suffixes):
        return f"DEVELOPMENT_OR_OBSERVED_OUTCOME_SEQUENCE:{timestamp}"
    return ""


def select_episodes(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded_suffixes: Iterable[str],
    active_hand: str = "any",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suffixes = tuple(sorted(set(excluded_suffixes)))
    audited: list[dict[str, Any]] = []
    for source in rows:
        if source.get("physicalization_v1_eligible") is not True:
            continue
        if active_hand != "any" and source.get("active_hand") != active_hand:
            continue
        row = dict(source)
        row["selection_key"] = _selection_key(seed, str(row["episode_id"]))
        row["development_exclusion_reason"] = _exclusion_reason(str(row["raw_sequence"]), suffixes)
        audited.append(row)
    pool = [row for row in audited if not row["development_exclusion_reason"]]
    if len(pool) < count:
        raise ValueError(f"HELD_OUT_EPISODE_POOL_TOO_SMALL:{len(pool)}<{count}")

    chosen: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    used_objects: set[str] = set()
    used_subjects: set[str] = set()
    remaining = list(pool)
    for rank in range(1, count + 1):
        remaining.sort(
            key=lambda row: (
                str(row["raw_sequence"]) in used_sequences,
                str(row["target_object"]) in used_objects,
                str(row["subject"]) in used_subjects,
                str(row["selection_key"]),
                str(row["episode_id"]),
            )
        )
        selected = remaining.pop(0)
        selected = {**selected, "selection_rank": rank}
        chosen.append(selected)
        used_sequences.add(str(selected["raw_sequence"]))
        used_objects.add(str(selected["target_object"]))
        used_subjects.add(str(selected["subject"]))
    return chosen, audited


def _manifest_episode(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row["provenance"]
    return {
        "episode_id": row["episode_id"],
        "clip_id": row["episode_id"],
        "selection_rank": row["selection_rank"],
        "selection_key": row["selection_key"],
        "raw_sequence": row["raw_sequence"],
        "sequence": row["raw_sequence"],
        "subject": row["subject"],
        "active_hand": row["active_hand"],
        "target_object": row["target_object"],
        "object_id": row["target_object"],
        "primary_object_id": row["target_object"],
        "selected_frame_range": [row["start_frame"], row["end_frame"]],
        "duration_frames": row["duration_frames"],
        "physicalization_v1_eligible": True,
        "complete": row["complete"],
        "episode_type": row["episode_type"],
        "other_hand_same_target": row["other_hand_same_target"],
        "selection_outcome_fields_observed": [],
        "exclusion_audit": {"development_clip": False, "outcome_observed": False},
        "object_ids": [row["target_object"]],
        "provenance": provenance,
        "raw_path": str(Path(provenance["raw_mano"]["path"]).parent),
        "raw_hashes": {
            "poses_m.npy": provenance["raw_mano"]["sha256"],
            "poses_o.npy": provenance["raw_object"]["sha256"],
            "meta.yaml": provenance["meta"]["sha256"],
            f"mesh:{row['target_object']}": provenance["object_mesh"]["sha256"],
            "mano_calibration": provenance["mano_calibration"]["sha256"],
        },
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    rows = json.loads(args.episode_index.resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("HOCAP_EPISODE_INDEX_LIST_REQUIRED")
    suffixes = (*DEFAULT_DEVELOPMENT_SEQUENCE_SUFFIXES, *args.exclude_sequence_suffix)
    selected, audited = select_episodes(
        rows,
        count=args.count,
        seed=args.seed,
        excluded_suffixes=suffixes,
        active_hand=args.active_hand,
    )
    authority: dict[str, Any] = {
        "schema_version": "HOCapPrimaryObjectAuthorityV2",
        "status": "CURRENT_EPISODE_TARGET_AUTHORITY",
        "authority_kind": "HOCapSingleHandObjectEpisodeV1_target_object",
        "outcome_inputs_used": False,
        "mappings": [
            {
                "status": "RESOLVED",
                "sequence": row["raw_sequence"],
                "episode_id": row["episode_id"],
                "primary_object_id": row["target_object"],
                "available_object_ids": [row["target_object"]],
                "selected_frame_range": [row["start_frame"], row["end_frame"]],
                "frame_count": row["end_frame"],
                "authority_kind": "episode_target_object_exact_surface_lifecycle_v1",
                "outcome_inputs_used": False,
            }
            for row in selected
        ],
    }
    authority["authority_sha256"] = _canonical_hash(authority)
    manifest: dict[str, Any] = {
        "schema_version": "HOCapPhysicalizationHeldOutEpisodeManifestV1",
        "status": "FROZEN_NOT_EXECUTED",
        "HELD_OUT_SET_FROZEN": "YES",
        "dataset": "hocap",
        "selection_unit": "HOCapSingleHandObjectEpisodeV1",
        "physicalization_protocol": "HOCapPhysicalizationProtocolV1",
        "episode_contract_sha256": selected[0]["contract_sha256"],
        "held_out_count": args.count,
        "selection_basis": "metadata_only_no_retarget_pf_ppo_outcomes",
        "selection_seed": args.seed,
        "active_hand_backend_capability_filter": args.active_hand,
        "primary_object_authority_sha256": authority["authority_sha256"],
        "development_sequence_suffix_exclusions": sorted(set(suffixes)),
        "diversity_priority": ["raw_sequence", "target_object", "subject"],
        "episodes": [_manifest_episode(row) for row in selected],
        # Compatibility alias for fail-closed downstream manifest readers.
        "clips": [_manifest_episode(row) for row in selected],
        "geometric_retarget_run": False,
        "source_policy_l0_run": False,
        "physical_ppo_run": False,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    output = args.output_root.resolve()
    _write_text(
        output / "held_out_5_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        output / "held_out_5_manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
    )
    _write_text(
        output / "episode_object_authority.json",
        json.dumps(authority, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output / "manifest_sha256.txt", str(manifest["manifest_sha256"]) + "\n")
    fields = [
        "episode_id",
        "raw_sequence",
        "subject",
        "active_hand",
        "target_object",
        "start_frame",
        "end_frame",
        "duration_frames",
        "selection_key",
        "selection_rank",
        "development_exclusion_reason",
    ]
    selected_ids = {str(row["episode_id"]) for row in selected}
    candidate_rows = [
        {
            **row,
            "selection_rank": next(
                (
                    item["selection_rank"]
                    for item in selected
                    if item["episode_id"] == row["episode_id"]
                ),
                "",
            ),
        }
        for row in audited
        if not row["development_exclusion_reason"]
    ]
    exclusions = [row for row in audited if row["development_exclusion_reason"]]
    _write_csv(output / "candidate_episodes.csv", candidate_rows, fields)
    _write_csv(output / "development_exclusions.csv", exclusions, fields)
    old_path = args.old_manifest.resolve() if args.old_manifest else None
    receipt = {
        "schema_version": "HOCapHeldOutManifestSupersessionReceiptV1",
        "status": "SUPERSEDED_BY_SINGLE_HAND_OBJECT_EPISODE_V1",
        "old_manifest": str(old_path) if old_path else "NOT_PROVIDED",
        "old_manifest_sha256": args.old_manifest_sha256,
        "new_manifest_sha256": manifest["manifest_sha256"],
        "selected_episode_ids": sorted(selected_ids),
        "old_evidence_retained": True,
        "old_manifest_reuse_for_gpu_forbidden": True,
    }
    _write_text(
        output / "old_manifest_superseded_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
