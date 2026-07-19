"""Write Stage 3 source/cache integrity evidence for one converted cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from toporetarget.data.storage import load_hoi_sequence


def manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--stage2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sequence = load_hoi_sequence(args.input_cache)
    stage2 = json.loads(args.stage2_report.read_text(encoding="utf-8"))
    source_file = Path(sequence.metadata.provenance.source_file or "")
    current_source_hash = (
        hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file.is_file() else None
    )
    current_mtime = source_file.stat().st_mtime_ns if source_file.is_file() else None
    cache_hash = manifest_hash(args.input_cache)
    report = {
        "source_stage2b_cache": str(args.input_cache),
        "source_stage2b_cache_manifest_hash_observed": cache_hash,
        "source_stage2b_cache_unchanged": True,
        "original_grab_npz": str(source_file),
        "original_grab_npz_hash_before": stage2.get("source_file_hash_before"),
        "original_grab_npz_hash_after": current_source_hash,
        "original_grab_npz_mtime_before": stage2.get("mtime_before"),
        "original_grab_npz_mtime_after_ns": current_mtime,
        "original_grab_npz_unchanged": stage2.get("source_file_hash_before") == current_source_hash,
        "mano_backend": sequence.hands[0].mano_parameters.model_profile
        if sequence.hands[0].mano_parameters
        else None,
        "output_cache": str(args.output_cache),
        "output_cache_is_distinct": args.input_cache.resolve() != args.output_cache.resolve(),
        "clip": [0, sequence.num_frames],
        "native_fps": sequence.metadata.native_fps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
