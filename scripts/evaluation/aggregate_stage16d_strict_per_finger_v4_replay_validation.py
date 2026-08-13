#!/usr/bin/env python3
"""Validate and combine the two required Strict Per-Finger V4 replay receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CLIPS = ("hocap_170650", "hocap_170105")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_REPLAY_RECEIPT_OBJECT_REQUIRED:{path}")
    return value


def _trace_evidence(clip: str, trace_path: Path) -> dict[str, Any]:
    path = trace_path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "replica_tip_pair_presence",
            "replica_tip_pair_force_world",
            "replica_source_contact_mask",
            "replica_r_contact_v4",
            "reward_v4_samples",
            "trace_type",
            "requested_clip",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"STRICT_V4_REPLAY_TRACE_FIELDS_MISSING:{clip}:{missing}")
        presence = np.asarray(archive["replica_tip_pair_presence"])
        force = np.asarray(archive["replica_tip_pair_force_world"])
        source = np.asarray(archive["replica_source_contact_mask"])
        reward = np.asarray(archive["replica_r_contact_v4"])
        if (
            presence.shape != (321, 20, 5)
            or force.shape != (321, 20, 5, 3)
            or source.shape != (321, 20, 5)
            or reward.shape != (321, 20)
            or not np.isfinite(force).all()
            or not np.isfinite(reward).all()
        ):
            raise ValueError(f"STRICT_V4_REPLAY_TRACE_SHAPE_OR_FINITE_INVALID:{clip}")
        trace_type = str(archive["trace_type"].item())
        requested_clip = str(archive["requested_clip"].item())
        samples = int(archive["reward_v4_samples"].item())
    if trace_type != "stage16d_ppo26d" or requested_clip != clip or samples < 1_000_000:
        raise ValueError(f"STRICT_V4_REPLAY_TRACE_PROVENANCE_INVALID:{clip}")
    return {"path": str(path), "reward_v4_samples": samples}


def _headless_evidence(clip: str, receipt_path: Path, trace_path: Path) -> dict[str, Any]:
    receipt = _read(receipt_path.resolve())
    trace = _trace_evidence(clip, trace_path)
    if (
        receipt.get("status") != "STAGE16D_PPO26D_REPLAY_VALIDATED"
        or receipt.get("headless") is not True
        or receipt.get("object") != clip
        or Path(str(receipt.get("trace", ""))).resolve() != Path(trace["path"])
        or receipt.get("finite") is not True
        or int(receipt.get("frame_count", 0)) != 321
    ):
        raise ValueError(f"STRICT_V4_HEADLESS_REPLAY_RECEIPT_INVALID:{clip}")
    return {"receipt": str(receipt_path.resolve()), "trace": trace, "result": receipt}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        nargs=3,
        action="append",
        required=True,
        metavar=("CLIP", "RECEIPT", "TRACE"),
        help="One validated headless replay receipt and its source V4 Formal20 trace per clip.",
    )
    gui = parser.add_mutually_exclusive_group(required=True)
    gui.add_argument("--gui-receipt", type=Path)
    gui.add_argument("--gui-environment-unavailable", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {clip: (Path(receipt), Path(trace)) for clip, receipt, trace in args.headless}
    if set(inputs) != set(CLIPS) or len(inputs) != len(args.headless):
        raise ValueError(f"STRICT_V4_REPLAY_REQUIRES_EXACTLY_TWO_CLIPS:{sorted(inputs)}")
    headless = {
        clip: _headless_evidence(clip, receipt_path, trace_path)
        for clip, (receipt_path, trace_path) in sorted(inputs.items())
    }
    if args.gui_receipt is not None:
        gui_result = _read(args.gui_receipt.resolve())
        if gui_result.get("status") != "STAGE16D_PPO26D_REPLAY_VALIDATED":
            raise ValueError("STRICT_V4_GUI_REPLAY_RECEIPT_INVALID")
        gui = {"status": "STAGE16D_PPO26D_REPLAY_VALIDATED", "receipt": str(args.gui_receipt)}
    else:
        gui = {"status": "GUI_ENVIRONMENT_UNAVAILABLE"}
    result = {
        "schema_version": "Stage16DStrictPerFingerV4ReplayValidationV1",
        "status": "STAGE16D_STRICT_V4_REPLAY_VALIDATED",
        "headless": headless,
        "gui": gui,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
