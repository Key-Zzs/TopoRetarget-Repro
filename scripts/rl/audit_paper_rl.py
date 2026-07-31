#!/usr/bin/env python3
"""Write a compact, page-addressable Appendix A.5 extraction from the local PDF audit."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(".local/reports/stage16_reference_tracking_ppo")
    root.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "item": "reference",
            "value": "finger joints, object pose and tracked links in robot base frame",
            "pdf_page": 13,
            "formula_or_table": "Appendix A.5",
            "exact_quote": "We use the base frame for reference quantities.",
            "implementation_mapping": "rl/contracts.py and rl/references.py",
            "confidence": "PAPER_EXACT",
        },
        {
            "item": "observation_and_action",
            "value": "residual action; q, qdot, previous action, axis points; lookahead 1/3/5",
            "pdf_page": 13,
            "formula_or_table": "Appendix A.5.1-A.5.2",
            "exact_quote": "The action at is a residual joint-position command.",
            "implementation_mapping": "rl/actuators.py and rl/observations.py",
            "confidence": "PAPER_EXACT",
        },
        {
            "item": "reward_termination",
            "value": "Table 4 weights, sigmas and termination thresholds",
            "pdf_page": 14,
            "formula_or_table": "Table 4",
            "exact_quote": "We define psi(e; sigma) = exp(-||e/sigma||^2).",
            "implementation_mapping": "rl/rewards.py and rl/termination.py",
            "confidence": "PAPER_EXACT",
        },
        {
            "item": "domain_randomization",
            "value": "Table 5 ranges and modes",
            "pdf_page": 15,
            "formula_or_table": "Table 5",
            "exact_quote": "The full randomization ranges are listed in Table 5.",
            "implementation_mapping": "rl/randomization.py",
            "confidence": "PAPER_EXACT",
        },
        {
            "item": "ppo",
            "value": (
                "4096 envs, 40 steps, 163840 samples; listed architecture and optimizer values"
            ),
            "pdf_page": 16,
            "formula_or_table": "Table 6",
            "exact_quote": "PPO epochs / minibatches 4 / 32.",
            "implementation_mapping": "rl/ppo",
            "confidence": "PAPER_EXACT",
        },
        {
            "item": "unpublished_details",
            "value": "simulator, PD gains, tracked links, axis offsets, PPO clip/value/grad fields",
            "pdf_page": 13,
            "formula_or_table": "Appendix A.5",
            "exact_quote": "not provided by the paper",
            "implementation_mapping": "docs/rl/PAPER_FIDELITY_LEDGER.yaml",
            "confidence": "UNRESOLVED",
        },
    ]
    locations = {
        "pdf": "docs/TopoRetarget.pdf",
        "pdf_sha256": "21c06a125430854dcff0d778283963b7fe107c8dfa79e3982639a80c21b206ab",
        "appendix_a5_pages": [13, 14, 15, 16],
        "existing_verified_manifest": "configs/paper/rl.yaml",
        "fidelity_ledger": "docs/rl/PAPER_FIDELITY_LEDGER.yaml",
    }
    (root / "paper_rl_extraction.json").write_text(
        json.dumps({"items": items}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "paper_source_locations.json").write_text(
        json.dumps(locations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Stage 16 paper RL extraction", ""]
    for item in items:
        lines.extend(
            [
                f"## {item['item']}",
                "",
                f"- Value: {item['value']}",
                f"- PDF page: {item['pdf_page']} ({item['formula_or_table']})",
                f"- Quote: {item['exact_quote']}",
                f"- Mapping: {item['implementation_mapping']}",
                f"- Confidence: {item['confidence']}",
                "",
            ]
        )
    (root / "paper_rl_extraction.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"items": len(items), "output_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
