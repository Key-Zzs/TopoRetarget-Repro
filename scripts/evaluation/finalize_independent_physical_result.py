#!/usr/bin/env python3
"""Freeze one terminal independent physical result and select its replay trace."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--frozen-evaluation-receipt", type=Path, required=True)
    parser.add_argument("--physical-refinement-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_FINAL_RESULT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_FINAL_RESULT_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verified(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if _artifact(path)["sha256"] != row.get("sha256"):
        raise RuntimeError("INDEPENDENT_FINAL_RESULT_UPSTREAM_HASH_DRIFT")
    return path


def _trace_from_summary(summary: dict[str, Any], *, accepted: bool) -> tuple[Path, dict[str, str]]:
    traces = summary.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("INDEPENDENT_FINAL_RESULT_TRACE_LIST_MISSING")
    trace_by_episode = {
        int(row["episode"]): Path(str(row["path"])).resolve()
        for row in traces
        if isinstance(row, dict) and "episode" in row and "path" in row
    }
    if len(trace_by_episode) != len(traces):
        raise ValueError("INDEPENDENT_FINAL_RESULT_TRACE_LIST_INVALID")
    per_episode = next(iter(trace_by_episode.values())).parent.parent / "per_episode.csv"
    with per_episode.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(traces):
        raise ValueError("INDEPENDENT_FINAL_RESULT_EPISODE_CARDINALITY_MISMATCH")

    def flag(row: dict[str, str], name: str) -> int:
        return int(row.get(name, "").lower() == "true")

    ranked = sorted(
        rows,
        key=lambda row: (
            flag(row, "PHYSICAL_HOI_ACCEPTED"),
            flag(row, "PF_V2"),
            flag(row, "DF_pose"),
            flag(row, "DF_linear"),
            flag(row, "DF_angular_v2"),
            -int(row["episode"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    if accepted and flag(selected, "PHYSICAL_HOI_ACCEPTED") != 1:
        raise RuntimeError("INDEPENDENT_FINAL_RESULT_ACCEPTED_TRACE_MISSING")
    trace = trace_by_episode[int(selected["episode"])]
    expected_hash = next(
        str(row["sha256"]) for row in traces if int(row["episode"]) == int(selected["episode"])
    )
    if not trace.is_file() or sha256_file(trace) != expected_hash:
        raise RuntimeError("INDEPENDENT_FINAL_RESULT_TRACE_HASH_DRIFT")
    return trace, selected


def main() -> int:
    args = _parser().parse_args()
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_FINAL_RESULT_CLIP_ID_INVALID")
    frozen_path = args.frozen_evaluation_receipt.resolve()
    frozen = _json(frozen_path)
    if (
        frozen.get("status") != "PASS"
        or frozen.get("clip_id") != args.clip_id
        or frozen.get("ppo_updates") != 0
    ):
        raise ValueError("INDEPENDENT_FINAL_RESULT_FROZEN_RECEIPT_INVALID")
    refinement: dict[str, Any] | None = None
    refinement_path: Path | None = None
    if frozen.get("accepted") is True:
        if args.physical_refinement_receipt is not None or frozen.get("ppo_required") is True:
            raise ValueError("INDEPENDENT_FINAL_RESULT_ACCEPTED_FROZEN_PPO_FORBIDDEN")
        decision = _json(_verified(frozen["decision"]))
        summary = decision.get("confirm20")
        accepted = True
        classification = "ACCEPTED_FROZEN"
        updates = 0
        samples = 0
    else:
        if args.physical_refinement_receipt is None or frozen.get("ppo_required") is not True:
            raise ValueError("INDEPENDENT_FINAL_RESULT_REFINEMENT_RECEIPT_REQUIRED")
        refinement_path = args.physical_refinement_receipt.resolve()
        refinement = _json(refinement_path)
        if (
            refinement.get("status") != "PASS"
            or refinement.get("clip_id") != args.clip_id
            or not 1 <= int(refinement.get("ppo_updates", -1)) <= 15
        ):
            raise ValueError("INDEPENDENT_FINAL_RESULT_REFINEMENT_RECEIPT_INVALID")
        accepted = bool(refinement.get("accepted") is True)
        classification = "ACCEPTED_AFTER_REFINEMENT" if accepted else "PPO_BUDGET_EXHAUSTED"
        updates = int(refinement["ppo_updates"])
        samples = int(refinement["ppo_samples"])
        best = refinement.get("best_checkpoint")
        if not isinstance(best, dict):
            raise ValueError("INDEPENDENT_FINAL_RESULT_BEST_CHECKPOINT_MISSING")
        summary = best.get("confirm20") if accepted else best.get("eval10")
    if not isinstance(summary, dict):
        raise ValueError("INDEPENDENT_FINAL_RESULT_SUMMARY_MISSING")
    trace, episode = _trace_from_summary(summary, accepted=accepted)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"INDEPENDENT_FINAL_RESULT_REFUSES_OVERWRITE:{output}")
    receipt = {
        "schema_version": "IndependentPhysicalFinalResultV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "classification": classification,
        "accepted": accepted,
        "ppo_updates": updates,
        "ppo_samples": samples,
        "frozen_evaluation_receipt": _artifact(frozen_path),
        "physical_refinement_receipt": (
            None if refinement_path is None else _artifact(refinement_path)
        ),
        "selected_trace": _artifact(trace),
        "selected_episode": episode,
        "selection_rule": "PHYSICAL_HOI_ACCEPTED_then_PF_V2_then_DF_pose_linear_angular",
        "summary_counts": summary.get("counts"),
        "failure_evidence_retained": not accepted,
    }
    atomic_write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
