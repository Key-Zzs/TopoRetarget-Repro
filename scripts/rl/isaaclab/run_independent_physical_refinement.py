#!/usr/bin/env python3
"""Run or fail-closed-finalize U15 PPO from a frozen evaluation receipt."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--frozen-evaluation-receipt", type=Path, required=True)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Verify an already-complete non-resumable U15 namespace and emit its receipt "
            "without launching another optimizer run."
        ),
    )
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_PHYSICAL_REFINEMENT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_PHYSICAL_REFINEMENT_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _flag(command: list[str], name: str) -> Path:
    indices = [index for index, value in enumerate(command) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"INDEPENDENT_PHYSICAL_REFINEMENT_COMMAND_FLAG_INVALID:{name}")
    return Path(command[indices[0] + 1]).resolve()


def _validate_complete(
    complete: dict[str, Any], progression: list[dict[str, str]], *, clip_id: str
) -> int:
    """Validate one immutable, fully completed U15 training lineage."""

    updates = int(complete.get("actual_new_updates", -1))
    if (
        complete.get("schema_version") != "PhysicalRefinementTrainingV1"
        or complete.get("clip") != clip_id
        or int(complete.get("max_new_updates", -1)) != 15
        or updates < 1
        or updates > 15
        or len(progression) != updates
        or any(int(row["new_update"]) != index for index, row in enumerate(progression, 1))
    ):
        raise RuntimeError("INDEPENDENT_PHYSICAL_REFINEMENT_RESULT_CONTRACT_INVALID")
    if not isinstance(complete.get("best_checkpoint"), dict):
        raise RuntimeError("INDEPENDENT_PHYSICAL_REFINEMENT_BEST_CHECKPOINT_MISSING")
    return updates


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_REQUIRES_EULA")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_CLIP_ID_INVALID")
    frozen_path = args.frozen_evaluation_receipt.resolve()
    frozen = _json(frozen_path)
    if (
        frozen.get("schema_version") != "IndependentFrozenPhysicalEvaluationReceiptV1"
        or frozen.get("status") != "PASS"
        or frozen.get("clip_id") != args.clip_id
        or frozen.get("accepted") is True
        or frozen.get("ppo_required") is not True
        or frozen.get("ppo_updates") != 0
    ):
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_FROZEN_DECISION_INVALID")
    command = frozen.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_FROZEN_COMMAND_INVALID")
    candidates = [index for index, value in enumerate(command) if value == "evaluate-first"]
    if len(candidates) != 1:
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_MODE_TOKEN_INVALID")
    command = list(command)
    command[candidates[0]] = "train"
    budget_indices = [index for index, value in enumerate(command) if value == "--max-new-updates"]
    if len(budget_indices) != 1 or command[budget_indices[0] + 1] != "15":
        raise ValueError("INDEPENDENT_PHYSICAL_REFINEMENT_U15_CONTRACT_DRIFT")
    report_root = _flag(command, "--report-root")
    run_root = _flag(command, "--run-root")
    final_receipt = report_root / "physical_refinement_receipt.json"
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_PHYSICAL_REFINEMENT_REFUSES_OVERWRITE:{final_receipt}")
    tick = time.perf_counter()
    log = report_root / "physical_refinement.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    complete_path = run_root / args.clip_id / "complete.json"
    progression_path = report_root / "training" / args.clip_id / "progression.csv"
    if args.finalize_existing:
        result_returncode = 0
        log.write_text(
            json.dumps(
                {
                    "operation": "finalize_existing",
                    "optimizer_subprocess_started": False,
                    "training_command": command,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        result_returncode = result.returncode
        log.write_text(result.stdout, encoding="utf-8")
    if result_returncode != 0 or not complete_path.is_file() or not progression_path.is_file():
        failure = {
            "schema_version": "IndependentPhysicalRefinementFailureV1",
            "status": "FAIL",
            "clip_id": args.clip_id,
            "returncode": result_returncode,
            "command": command,
            "log": _artifact(log),
        }
        atomic_write_json(report_root / "physical_refinement_failure.json", failure)
        raise RuntimeError("INDEPENDENT_PHYSICAL_REFINEMENT_EXECUTION_FAILED")
    complete = _json(complete_path)
    with progression_path.open(newline="", encoding="utf-8") as stream:
        progression = list(csv.DictReader(stream))
    updates = _validate_complete(complete, progression, clip_id=args.clip_id)
    receipt = {
        "schema_version": "IndependentPhysicalRefinementReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "accepted": bool(complete.get("accepted") is True),
        "ppo_updates": updates,
        "ppo_samples": int(complete["actual_new_samples"]),
        "eval10_by_update": progression,
        "best_checkpoint": complete["best_checkpoint"],
        "success_stop_triggered": bool(complete.get("success_stop_triggered") is True),
        "frozen_evaluation_receipt": _artifact(frozen_path),
        "complete": _artifact(complete_path),
        "progression": _artifact(progression_path),
        "command": command,
        "log": _artifact(log),
        "productive_run_seconds": time.perf_counter() - tick,
        "technical_retry_seconds": 0.0,
        "retry_count": 0,
        "cache_hit": False,
        "execution_mode": "finalize_existing" if args.finalize_existing else "train",
        "optimizer_subprocess_started": not args.finalize_existing,
    }
    atomic_write_json(final_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
