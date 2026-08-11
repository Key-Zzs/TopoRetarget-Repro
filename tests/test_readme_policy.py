from __future__ import annotations

from pathlib import Path


def test_stable_readmes_exclude_run_log_receipts() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        ".local/reports/",
        "Start HEAD",
        "Final HEAD",
        "cumulative samples",
        "checkpoint SHA",
        "experiment runtime",
    )
    for name in ("README.md", "README.zh-CN.md"):
        content = (root / name).read_text(encoding="utf-8")
        assert not [needle for needle in forbidden if needle in content], name
