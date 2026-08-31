#!/usr/bin/env python3
"""Build the execution-time primary-object authority for frozen P7 clips.

The P7 refreeze intentionally contains no mutable execution authority.  This
script derives the exact target-object/frame mapping from that immutable
manifest without changing or re-freezing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_episode_manifest,
    atomic_write_json,
    stable_hash,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P7_EXECUTION_AUTHORITY_JSON_OBJECT_REQUIRED:{path}")
    return value


def _int_frame(value: object, *, label: str) -> int:
    try:
        frame = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"P7_EXECUTION_AUTHORITY_FRAME_INVALID:{label}:{value}") from exc
    if frame < 0:
        raise ValueError(f"P7_EXECUTION_AUTHORITY_FRAME_NEGATIVE:{label}:{frame}")
    return frame


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_episode_manifest(manifest)

    rows = manifest.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError("P7_EXECUTION_AUTHORITY_CLIPS_REQUIRED")

    mappings: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("P7_EXECUTION_AUTHORITY_CLIP_OBJECT_REQUIRED")
        episode_id = str(row.get("episode_id", row.get("clip_id", "")))
        sequence = str(row.get("sequence", row.get("raw_sequence", "")))
        primary_object_id = str(row.get("primary_object_id", row.get("object_id", "")))
        selected_range = row.get("selected_frame_range")
        if not episode_id or not sequence or not primary_object_id:
            raise ValueError("P7_EXECUTION_AUTHORITY_CLIP_FIELDS_MISSING")
        if episode_id in seen_episode_ids or sequence in seen_sequences:
            raise ValueError("P7_EXECUTION_AUTHORITY_MAPPING_NOT_UNIQUE")
        if not isinstance(selected_range, list) or len(selected_range) != 2:
            raise ValueError("P7_EXECUTION_AUTHORITY_FRAME_RANGE_REQUIRED")
        start = _int_frame(selected_range[0], label=f"{episode_id}:start")
        end = _int_frame(selected_range[1], label=f"{episode_id}:end")
        if end <= start:
            raise ValueError("P7_EXECUTION_AUTHORITY_FRAME_RANGE_NOT_INCREASING")

        seen_episode_ids.add(episode_id)
        seen_sequences.add(sequence)
        mappings.append(
            {
                "status": "RESOLVED",
                "sequence": sequence,
                "episode_id": episode_id,
                "primary_object_id": primary_object_id,
                "available_object_ids": [primary_object_id],
                "selected_frame_range": [start, end],
                "frame_count": end,
                "authority_kind": "episode_target_object_exact_surface_lifecycle_v1",
                "outcome_inputs_used": False,
            }
        )

    authority: dict[str, Any] = {
        "schema_version": "HOCapPrimaryObjectAuthorityV2",
        "status": "CURRENT_EPISODE_TARGET_AUTHORITY",
        "authority_kind": "HOCapSingleHandObjectEpisodeV1_target_object",
        "outcome_inputs_used": False,
        "mappings": mappings,
    }
    authority["authority_sha256"] = stable_hash(authority)

    output = args.output.resolve()
    if output.exists():
        existing = _json(output)
        if existing != authority:
            raise FileExistsError(f"P7_EXECUTION_AUTHORITY_REFUSES_OVERWRITE:{output}")
    else:
        atomic_write_json(output, authority)
    print(json.dumps({"status": "PASS", "output": str(output), **authority}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
