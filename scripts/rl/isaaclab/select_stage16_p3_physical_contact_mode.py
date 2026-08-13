#!/usr/bin/env python3
"""Write the C2 development-only global physical contact-mode selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.physical_evaluation import CLIPS  # noqa: E402
from toporetarget.rl.physical_mode_selection import (  # noqa: E402
    CONTACT_MODES,
    select_global_physical_contact_mode,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SELECTION_REPORT_MUST_BE_OBJECT:{path}")
    return value


def _mode_directory(mode: str) -> str:
    return "v3" if mode == "aggregate_v3" else "v4"


def _reports(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, str]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    receipts: dict[str, dict[str, str]] = {}
    for mode in CONTACT_MODES:
        reports[mode] = {}
        receipts[mode] = {}
        for clip in CLIPS:
            path = (
                root / "physical_pilot" / _mode_directory(mode) / clip / "c2/dev/qualification.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"SELECTION_C2_DEVELOPMENT_REPORT_MISSING:{path}")
            reports[mode][clip] = _read(path)
            receipts[mode][clip] = str(path.resolve())
    return reports, receipts


def _table_rows(selection: dict[str, object]) -> list[dict[str, object]]:
    candidates = selection["candidates"]
    if not isinstance(candidates, dict):
        raise ValueError("SELECTION_CANDIDATES_INVALID")
    rows: list[dict[str, object]] = []
    for mode in CONTACT_MODES:
        candidate = candidates[mode]
        if not isinstance(candidate, dict):
            raise ValueError("SELECTION_CANDIDATE_INVALID")
        per_clip = candidate["per_clip_metrics"]
        if not isinstance(per_clip, dict):
            raise ValueError("SELECTION_PER_CLIP_INVALID")
        for clip in CLIPS:
            item = per_clip[clip]
            if not isinstance(item, dict):
                raise ValueError("SELECTION_PER_CLIP_ROW_INVALID")
            metrics = item["metrics"]
            if not isinstance(metrics, dict):
                raise ValueError("SELECTION_METRICS_INVALID")
            rows.append(
                {
                    "mode": mode,
                    "clip": clip,
                    "safety_pass": item["safety_pass"],
                    "safety_reasons": ";".join(item["safety_reasons"]),
                    **metrics,
                }
            )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tables(root: Path, selection: dict[str, object]) -> None:
    rows = _table_rows(selection)
    table_root = root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    csv_path = table_root / "physical_pilot_v3_v4_macro.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# C2 Physical Pilot: Global V3/V4 Selection Evidence",
        "",
        "Development-only metrics at C2 gravity 0.50 and friction 1.50. This table is not a "
        "formal P4 result.",
        "",
        "| Mode | Clip | Safe | SRq | SRp | Recall | Cross | No hand | dOmega | dV | P95 mm |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {clip} | {safety_pass} | {SRqualified:.3f} | {SRphysics:.3f} | "
            "{source_persistent_tip_recall:.3f} | {cross_finger_compensation:.3f} | "
            "{no_hand_object_contact_fraction:.3f} | {terminal_Delta_omega_radps:.3f} | "
            "{terminal_Delta_v_mps:.3f} | {hand_object_p95_penetration_m:.3f} |".format(
                **row,
                hand_object_p95_penetration_m=float(row["hand_object_p95_penetration_m"]) * 1000.0,
            )
        )
    lines.extend(
        [
            "",
            f"Selection status: `{selection['status']}`.",
            "",
            f"Reason: {selection['selection_reason']}",
            "",
        ]
    )
    (table_root / "physical_pilot_v3_v4_macro.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    output = root / "global_physical_contact_mode_selection.json"
    if output.exists():
        raise FileExistsError(f"SELECTION_OUTPUT_ALREADY_EXISTS:{output}")
    reports, paths = _reports(root)
    selection = select_global_physical_contact_mode(reports)
    selection["input_reports"] = paths
    _write_json(output, selection)
    _write_tables(root, selection)
    print(
        json.dumps({key: selection[key] for key in ("status", "selected_mode", "selection_reason")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
