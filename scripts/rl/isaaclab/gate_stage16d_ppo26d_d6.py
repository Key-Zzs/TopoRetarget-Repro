#!/usr/bin/env python3
"""Authorize or block Stage 16-D D6 from the two independent R7 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _validate_r7(path: Path, *, expected_clip: str) -> dict[str, Any]:
    payload = _read_json(path.resolve())
    if payload.get("schema_version") != "Stage16DPPO26DR7FormalQualificationV1":
        raise ValueError(f"D6 requires an R7 qualification artifact: {path}")
    if payload.get("clip") != expected_clip:
        raise ValueError(f"D6 clip identity mismatch: expected={expected_clip} path={path}")
    if not isinstance(payload.get("physics_qualified"), bool):
        raise ValueError(f"D6 R7 artifact has no boolean physics_qualified field: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-170650", type=Path, required=True)
    parser.add_argument("--qualification-170105", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    qualifications = {
        "hocap_170650": _validate_r7(args.qualification_170650, expected_clip="hocap_170650"),
        "hocap_170105": _validate_r7(args.qualification_170105, expected_clip="hocap_170105"),
    }
    eligible = {clip: bool(result["physics_qualified"]) for clip, result in qualifications.items()}
    authorized = all(eligible.values())
    status = (
        "STAGE16D_D6_MULTICLIP_AUTHORIZED"
        if authorized
        else "STAGE16D_D6_MULTICLIP_NOT_AUTHORIZED_SINGLE_CLIP_R7_FAILED"
    )
    payload = {
        "schema_version": "Stage16DPPO26DD6AuthorizationV1",
        "status": status,
        "multiclip_training_authorized": authorized,
        "single_clip_r7_physics_qualified": eligible,
        "r7_qualifications": {
            clip: str(path.resolve())
            for clip, path in {
                "hocap_170650": args.qualification_170650,
                "hocap_170105": args.qualification_170105,
            }.items()
        },
        "next_action": (
            "Run the fixed 50/50 multi-clip PPO-26D train/validation/formal protocol."
            if authorized
            else "Do not start multi-clip PPO and do not export D7 trajectories."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    with args.transitions.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "from": "D6_MULTICLIP",
                    "to": "D6_QUALIFICATION" if authorized else "D7_EXPORT",
                    "reason": status,
                    "authorization": str(args.output.resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"status": status, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
