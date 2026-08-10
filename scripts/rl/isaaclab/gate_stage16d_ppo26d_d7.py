#!/usr/bin/env python3
"""Record whether D7 export is allowed after the D6 authorization gate."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    authorization = _read_json(args.d6_authorization.resolve())
    if authorization.get("schema_version") != "Stage16DPPO26DD6AuthorizationV1":
        raise ValueError("D7 requires a Stage16DPPO26DD6AuthorizationV1 artifact")
    d6_authorized = authorization.get("multiclip_training_authorized")
    if not isinstance(d6_authorized, bool):
        raise ValueError("D7 requires an explicit D6 authorization boolean")
    status = (
        "STAGE16D_D7_EXPORT_AWAITING_D6_FORMAL_QUALIFICATION"
        if d6_authorized
        else "STAGE16D_D7_EXPORT_NOT_AUTHORIZED_D6_BLOCKED"
    )
    payload = {
        "schema_version": "Stage16DPPO26DD7ExportGateV1",
        "status": status,
        "d7_export_authorized": False,
        "d6_authorization": str(args.d6_authorization.resolve()),
        "reason": (
            "D6 is authorized but has not yet produced a formally qualified multi-clip policy."
            if d6_authorized
            else (
                "At least one single-clip R7 result failed physics qualification; "
                "D6 and D7 are blocked."
            )
        ),
        "preserved_artifacts_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    with args.transitions.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "from": "D7_EXPORT",
                    "to": "CLOSEOUT",
                    "reason": status,
                    "export_gate": str(args.output.resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"status": status, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
