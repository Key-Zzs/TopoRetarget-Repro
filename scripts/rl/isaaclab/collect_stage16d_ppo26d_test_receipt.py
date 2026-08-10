#!/usr/bin/env python3
"""Run and record the required Stage 16-D.5 PPO-26D repository checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def command_results() -> list[dict[str, object]]:
    commands = (
        ("ruff_check", ("ruff", "check", ".")),
        ("ruff_format_check", ("ruff", "format", "--check", ".")),
        ("mypy", (sys.executable, "-m", "mypy", "src")),
        ("pytest", (sys.executable, "-m", "pytest", "-q")),
        ("paper_fidelity", (sys.executable, "scripts/check_paper_fidelity.py")),
        (
            "base_import_without_isaac",
            (sys.executable, "-c", "import toporetarget; print('PASS')"),
        ),
    )
    results = []
    for name, command in commands:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        output = (result.stdout + result.stderr).strip()
        results.append(
            {
                "name": name,
                "command": list(command),
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "output_tail": output[-4000:],
            }
        )
    return results


def main() -> int:
    args = parse_args()
    results = command_results()
    payload = {
        "schema_version": "Stage16DPPO26DTestReceiptV1",
        "status": "PASS" if all(result["passed"] for result in results) else "FAIL",
        "checks": results,
    }
    output = args.output_root.resolve() / "tests.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
