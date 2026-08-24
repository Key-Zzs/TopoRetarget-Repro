#!/usr/bin/env python3
"""Parse all local HOCap sequences into SingleHandObjectEpisodeV1 rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap_episode import (  # noqa: E402
    HOCapSingleHandObjectEpisodeContractV1,
    aggregate_episode_rows,
    parse_sequence,
)
from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    DEFAULT_MANO_ROOT,
    DEFAULT_STORAGE_ROOT,
)

CSV_FIELDS = (
    "subject",
    "raw_sequence",
    "episode_id",
    "active_hand",
    "target_object",
    "episode_type",
    "start_frame",
    "approach_frame",
    "contact_frame",
    "pickup_frame",
    "transport_frame",
    "place_frame",
    "release_frame",
    "retreat_frame",
    "end_frame",
    "duration_frames",
    "duration_seconds",
    "other_hand_same_target",
    "overlapping_other_hand_other_object",
    "complete",
    "physicalization_v1_eligible",
    "exclusion_reason",
    "returned_near_initial_pose",
    "return_semantics",
    "semantic_contact_regions",
    "source_support_metadata",
    "provenance",
    "contract_sha256",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help="HOCap dataset root or storage root containing HOCap/data; no download.",
    )
    parser.add_argument(
        "--mano-model-root",
        type=Path,
        default=DEFAULT_MANO_ROOT,
        help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Episode index/report directory.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Optional subject_N/YYYYMMDD_HHMMSS selector; repeatable.",
    )
    parser.add_argument(
        "--hand",
        choices=("auto", "both", "left", "right"),
        default="auto",
        help="auto/both parses every official hand; left/right filters explicitly.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only contract- and source-hash-matched per-sequence receipts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this command's index files; raw HOCap inputs are never changed.",
    )
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _source_hash(sequence_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        sequence_dir / "meta.yaml",
        sequence_dir / "poses_m.npy",
        sequence_dir / "poses_o.npy",
    ):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "src/toporetarget/adapters/datasets/hocap.py",
        REPO_ROOT / "src/toporetarget/adapters/datasets/hocap_episode.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: (
                        json.dumps(row[name], sort_keys=True)
                        if isinstance(row.get(name), (dict, list))
                        else row.get(name)
                    )
                    for name in CSV_FIELDS
                }
            )
    os.replace(temporary, path)


def _selectors(sequence_root: Path, requested: list[str]) -> list[Path]:
    available = sorted(path.parent for path in sequence_root.glob("subject_*/*/meta.yaml"))
    if not requested:
        return available
    wanted = {value.removeprefix("hocap:").strip("/") for value in requested}
    selected = [path for path in available if str(path.relative_to(sequence_root)) in wanted]
    missing = sorted(wanted - {str(path.relative_to(sequence_root)) for path in selected})
    if missing:
        raise ValueError(f"HOCAP_EPISODE_SEQUENCE_NOT_AVAILABLE:{missing}")
    return selected


def _resume_row(
    receipt_path: Path,
    rows_path: Path,
    *,
    contract_hash: str,
    implementation_hash: str,
    source_hash: str,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    if not receipt_path.is_file() or not rows_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if (
        receipt.get("contract_sha256") != contract_hash
        or receipt.get("implementation_sha256") != implementation_hash
        or receipt.get("source_stat_sha256") != source_hash
        or not isinstance(rows, list)
    ):
        return None
    return rows, receipt


def _contract_payload(contract: HOCapSingleHandObjectEpisodeContractV1) -> dict[str, object]:
    return {
        "schema_version": contract.schema_version,
        "contract": contract.as_dict(),
        "production_unit": "one_active_hand_x_one_target_object_x_one_complete_lifecycle",
        "lifecycle": [
            "IDLE_PRE",
            "APPROACH",
            "CONTACT_ACQUISITION",
            "PICKUP",
            "TRANSPORT",
            "PLACE",
            "RELEASE",
            "RETREAT",
            "IDLE_POST",
        ],
        "segmentation_not_reward_phase_gate": True,
        "fixed_event_padding_authority": False,
        "both_hands_same_object": "BIMANUAL_SAME_OBJECT_INELIGIBLE",
        "left_to_both_to_right": "HANDOVER_INELIGIBLE",
        "different_object_overlapping_hands": "TWO_INDEPENDENT_EPISODES_SUPPORTED",
        "geometry_authority": {
            "hand": "MANO_whole_surface",
            "object": "exact_source_triangle_mesh",
            "keypoint_5cm_authority": False,
        },
        "angular_kinematics": "pose_derived_SO3_relative_rotation",
        "return_semantics": contract.return_semantics,
    }


def main() -> int:
    args = _parser().parse_args()
    started_utc = _utc()
    tick = time.perf_counter()
    requested_data_root = args.data_root.resolve()
    dataset_root = (
        requested_data_root
        if (requested_data_root / "data").is_dir()
        else requested_data_root / "HOCap"
    )
    sequence_root = dataset_root / "data"
    mano_root = args.mano_model_root.resolve()
    for path in (sequence_root, mano_root / "MANO_LEFT.pkl", mano_root / "MANO_RIGHT.pkl"):
        if not path.exists():
            raise FileNotFoundError(f"HOCAP_EPISODE_REQUIRED_INPUT_MISSING:{path}")
    output = args.output_root.resolve()
    index_paths = (
        output / "all_hocap_episodes.csv",
        output / "all_hocap_episodes.json",
        output / "aggregate.json",
        output / "segmentation_contract.json",
        output / "parse_receipt.json",
    )
    if any(path.exists() for path in index_paths) and not (args.resume or args.force):
        raise FileExistsError(f"HOCAP_EPISODE_INDEX_EXISTS:{output}")
    contract = HOCapSingleHandObjectEpisodeContractV1()
    contract_payload = _contract_payload(contract)
    contract_hash = _stable_hash(contract.as_dict())
    implementation_hash = _implementation_hash()
    selected_sides = None if args.hand in {"auto", "both"} else (args.hand,)
    all_rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    sequence_paths = _selectors(sequence_root, args.sequence)
    receipt_root = output / "sequence_receipts"
    row_root = output / "sequence_rows"
    for index, sequence_dir in enumerate(sequence_paths, start=1):
        relative = str(sequence_dir.relative_to(sequence_root))
        key = relative.replace("/", "__")
        receipt_path = receipt_root / f"{key}.json"
        rows_path = row_root / f"{key}.json"
        source_hash = _source_hash(sequence_dir)
        resumed = (
            _resume_row(
                receipt_path,
                rows_path,
                contract_hash=contract_hash,
                implementation_hash=implementation_hash,
                source_hash=source_hash,
            )
            if args.resume
            else None
        )
        if resumed is None:
            rows, receipt = parse_sequence(
                sequence_dir,
                dataset_root=dataset_root,
                mano_model_root=mano_root,
                selected_sides=selected_sides,
                contract=contract,
            )
            receipt.update(
                {
                    "contract_sha256": contract_hash,
                    "implementation_sha256": implementation_hash,
                    "source_stat_sha256": source_hash,
                    "status": "PASS",
                    "resumed": False,
                }
            )
            _atomic_json(rows_path, rows)
            _atomic_json(receipt_path, receipt)
        else:
            rows, receipt = resumed
            receipt = {**receipt, "resumed": True}
        all_rows.extend(rows)
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "sequence": relative,
                    "progress": f"{index}/{len(sequence_paths)}",
                    "candidate_episodes": len(rows),
                    "eligible": sum(bool(row.get("physicalization_v1_eligible")) for row in rows),
                    "resumed": receipt["resumed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    all_rows.sort(key=lambda row: str(row["episode_id"]))
    aggregate = aggregate_episode_rows(all_rows, receipts)
    aggregate["all_available_sequences_parsed"] = len(sequence_paths) == len(
        _selectors(sequence_root, [])
    )
    aggregate["contract_sha256"] = contract_hash
    support_explicit = [
        row["raw_sequence"]
        for row in receipts
        if row["source_support_metadata"]["source_explicit_support_present"]
    ]
    support_reconstructed = [
        row["raw_sequence"]
        for row in receipts
        if row["source_support_metadata"]["source_reconstructed_support_candidate_present"]
    ]
    aggregate["source_support_audit"] = {
        "result": "YES" if support_explicit else "PARTIALLY" if support_reconstructed else "NO",
        "explicit_sequences": support_explicit,
        "reconstruction_candidate_sequences": support_reconstructed,
        "checked_sequence_metadata": len(receipts),
    }
    _write_csv(output / "all_hocap_episodes.csv", all_rows)
    _atomic_json(output / "all_hocap_episodes.json", all_rows)
    _atomic_json(output / "aggregate.json", aggregate)
    _atomic_json(output / "segmentation_contract.json", contract_payload)
    receipt = {
        "schema_version": "HOCapEpisodeDatasetParseReceiptV1",
        "status": "PASS",
        "started_utc": started_utc,
        "ended_utc": _utc(),
        "wall_seconds": time.perf_counter() - tick,
        "data_root": str(requested_data_root),
        "dataset_root": str(dataset_root),
        "mano_model_root": str(mano_root),
        "hand_selection": args.hand,
        "network_download_performed": False,
        "raw_inputs_modified": False,
        "sequence_count": len(receipts),
        "contract_sha256": contract_hash,
        "implementation_sha256": implementation_hash,
        "aggregate": aggregate,
        "sequence_receipts": receipts,
    }
    _atomic_json(output / "parse_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
