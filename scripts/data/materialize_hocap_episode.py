#!/usr/bin/env python3
"""Materialize one eligible HOCap EpisodeV1 as a canonical HOI cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap import HOCapAdapterV1  # noqa: E402
from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    DEFAULT_MANO_ROOT,
    DEFAULT_STORAGE_ROOT,
)
from toporetarget.data.adapters.base import FrameRange  # noqa: E402
from toporetarget.data.storage import save_hoi_sequence  # noqa: E402
from toporetarget.utils.hashing import sha256_tree  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--mano-model-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Defaults to <output>.episode_contract.json.",
    )
    parser.add_argument(
        "--allow-ineligible-diagnostic",
        action="store_true",
        help="Diagnostic only; production physicalization requires eligibility.",
    )
    parser.add_argument(
        "--benchmark-first-frames",
        type=int,
        help=(
            "Materialize only the first N episode frames for a declared benchmark; "
            "never changes the frozen EpisodeV1 row."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_episode_row(index_path: Path, episode_id: str) -> dict[str, Any]:
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("HOCAP_EPISODE_INDEX_JSON_LIST_REQUIRED")
    rows = [row for row in value if isinstance(row, dict) and row.get("episode_id") == episode_id]
    if len(rows) != 1:
        raise ValueError(f"HOCAP_EPISODE_ID_CARDINALITY:{episode_id}:{len(rows)}")
    return rows[0]


def main() -> int:
    args = _parser().parse_args()
    index_path = args.episode_index.resolve()
    row = load_episode_row(index_path, args.episode_id)
    if row.get("physicalization_v1_eligible") is not True and not args.allow_ineligible_diagnostic:
        raise ValueError("HOCAP_EPISODE_NOT_PHYSICALIZATION_V1_ELIGIBLE")
    hand = str(row.get("active_hand", ""))
    if hand not in {"left", "right"}:
        raise ValueError("HOCAP_EPISODE_SINGLE_HAND_REQUIRED")
    start = row.get("start_frame")
    end = row.get("end_frame")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise ValueError("HOCAP_EPISODE_FRAME_RANGE_INVALID")
    source_episode_end = end
    if args.benchmark_first_frames is not None:
        if args.benchmark_first_frames < 2 or args.benchmark_first_frames > end - start:
            raise ValueError("HOCAP_EPISODE_BENCHMARK_FRAME_COUNT_INVALID")
        end = start + args.benchmark_first_frames
    output = args.output.resolve()
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None
        else output.with_suffix(".episode_contract.json")
    )
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"HOCAP_EPISODE_MATERIALIZATION_REFUSES_OVERWRITE:{output}")
    requested_data_root = args.data_root.resolve()
    storage_root = (
        requested_data_root.parent
        if (requested_data_root / "data").is_dir()
        else requested_data_root
    )
    adapter = HOCapAdapterV1(
        data_root=storage_root,
        mano_model_root=args.mano_model_root.resolve(),
    )
    sequence = adapter.load_sequence(
        str(row["raw_sequence"]),
        frame_range=FrameRange(start=start, end=end),
        hand=hand,
        primary_object_id=str(row["target_object"]),
    )
    save_hoi_sequence(sequence, output)
    tree_hashes = sha256_tree(output)
    source = row.get("provenance")
    if not isinstance(source, dict):
        raise ValueError("HOCAP_EPISODE_SOURCE_PROVENANCE_REQUIRED")
    receipt = {
        "schema_version": "CanonicalHOIEpisodeV1",
        "status": "PASS",
        "dataset": "hocap",
        "subject": row["subject"],
        "raw_sequence": row["raw_sequence"],
        "episode_id": row["episode_id"],
        "active_hand": hand,
        "target_object": row["target_object"],
        "source_frame_range": [start, end],
        "source_episode_frame_range": [start, source_episode_end],
        "frame_range_semantics": "start_inclusive_end_exclusive",
        "benchmark_truncation": (
            None
            if args.benchmark_first_frames is None
            else {
                "scope": "FIRST_N_EPISODE_FRAMES",
                "frames": args.benchmark_first_frames,
                "production_episode_mutated": False,
            }
        ),
        "timestamps_fps": sequence.metadata.native_fps,
        "event_frames": {
            name: row.get(name)
            for name in (
                "approach_frame",
                "contact_frame",
                "pickup_frame",
                "transport_frame",
                "place_frame",
                "release_frame",
                "retreat_frame",
            )
        },
        "raw_mano_provenance": source.get("raw_mano"),
        "object_pose_provenance": source.get("raw_object"),
        "object_mesh_provenance": source.get("object_mesh"),
        "source_support_metadata": row.get("source_support_metadata"),
        "other_hand_metadata": {
            "other_hand_same_target": row.get("other_hand_same_target"),
            "overlapping_other_hand_other_object": row.get("overlapping_other_hand_other_object"),
        },
        "episode_type": row["episode_type"],
        "eligibility": row["physicalization_v1_eligible"],
        "episode_contract_sha256": row["contract_sha256"],
        "episode_index": {"path": str(index_path), "sha256": _sha256(index_path)},
        "canonical_cache": {
            "path": str(output),
            "tree_hashes": tree_hashes,
            "tree_sha256": hashlib.sha256(
                json.dumps(tree_hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    }
    receipt["canonical_episode_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
