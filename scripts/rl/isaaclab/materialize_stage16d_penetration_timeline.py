#!/usr/bin/env python3
"""Materialize formal pairwise penetration NPZs as one Parquet timeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPORT_ROOT / "penetration_pair_timeline.parquet"
    )
    return parser.parse_args()


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    batches = []
    for clip in ("170105", "170650"):
        for state_kind in ("source_runtime", "corrected_runtime"):
            path = REPORT_ROOT / f"{state_kind}_penetration_pairs_{clip}.npz"
            with np.load(path, allow_pickle=False) as payload:
                signed = np.asarray(payload["signed_separation_m"], dtype=np.float64)
                depth = np.asarray(payload["penetration_depth_m"], dtype=np.float64)
                direction = np.asarray(
                    payload["depenetration_direction_for_object"], dtype=np.float64
                )
                pair_ids = np.asarray(payload["pair_ids"], dtype=str)
                worst = np.asarray(payload["frame_worst_pair_index"], dtype=np.int64)
            frames, replicas, pairs = signed.shape
            frame = np.repeat(np.arange(frames), replicas * pairs)
            replica = np.tile(np.repeat(np.arange(replicas), pairs), frames)
            pair_index = np.tile(np.arange(pairs), frames * replicas)
            batches.append(
                pa.table(
                    {
                        "clip": np.repeat(f"hocap_{clip}", signed.size),
                        "state_kind": np.repeat(state_kind, signed.size),
                        "frame": frame,
                        "replica": replica,
                        "pair_index": pair_index,
                        "pair_id": pair_ids[pair_index],
                        "signed_separation_m": signed.reshape(-1),
                        "penetration_depth_m": depth.reshape(-1),
                        "depenetration_x": direction[..., 0].reshape(-1),
                        "depenetration_y": direction[..., 1].reshape(-1),
                        "depenetration_z": direction[..., 2].reshape(-1),
                        "is_frame_worst_pair": pair_index == worst.reshape(-1).repeat(pairs),
                        "converged": np.ones(signed.size, dtype=bool),
                    }
                )
            )
    table = pa.concat_tables(batches)
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, args.output, compression="zstd")
    print(f"rows={table.num_rows} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
