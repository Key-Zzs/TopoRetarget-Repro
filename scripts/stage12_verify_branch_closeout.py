#!/usr/bin/env python3
"""Verify the non-force archival and closure of the two compiled worktrees."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from time import time

BRANCHES = {
    "feature": {
        "branch": "feature/compiled-exact-sign",
        "worktree": "TopoRetarget-Repro-compiled-exact-sign",
        "archive_glob": "compiled-exact-sign_*_fe8b2d0bbb9d",
    },
    "develop": {
        "branch": "develop/compiled-kernel",
        "worktree": "TopoRetarget-Repro-compiled-kernel",
        "archive_glob": "compiled-kernel_*_712899dba373",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)


def _archive(root: Path, pattern: str) -> tuple[Path, dict[str, object]]:
    archive_parent = root.parent / "TopoRetarget-Repro-branch-archives"
    matches = sorted(archive_parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one archive matching {pattern!r}, found: {matches}")
    manifest_path = matches[0] / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("hash_verification_pass"):
        raise RuntimeError(f"archive was not internally verified: {matches[0]}")
    archived_local = matches[0] / ".local"
    files = list(manifest.get("files", []))
    hashes_match = all(
        isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and isinstance(row.get("sha256"), str)
        and _sha256(archived_local / row["path"]) == row["sha256"]
        for row in files
    )
    if not hashes_match:
        raise RuntimeError(f"archive hash re-verification failed: {matches[0]}")
    return matches[0], manifest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / ".local/reports/stage12_v4_completion/branch_closeout"
    output.mkdir(parents=True, exist_ok=True)
    old = output / "closeout_blocked.json"
    historical = output / "pre_restore_blocked.json"
    if old.is_file() and not historical.exists():
        shutil.copy2(old, historical)

    worktrees = _git(root, "worktree", "list", "--porcelain").stdout
    result: dict[str, object] = {
        "schema_version": "toporetarget.stage12.branch_closeout_gate.v1",
        "status": "BRANCH_WORKTREE_CLOSEOUT_COMPLETE",
        "generated_unix_s": time(),
        "integration_local_head": _git(
            root, "rev-parse", "integration/dataset-adapter-v1"
        ).stdout.strip(),
        "integration_remote_head": _git(
            root, "rev-parse", "origin/integration/dataset-adapter-v1"
        ).stdout.strip(),
        "branches": {},
    }
    all_verified = result["integration_local_head"] == result["integration_remote_head"]
    csv_rows: list[dict[str, object]] = []
    for name, spec in BRANCHES.items():
        archive, manifest = _archive(root, str(spec["archive_glob"]))
        worktree_path = root.parent / str(spec["worktree"])
        local = _git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{spec['branch']}")
        remote = _git(root, "ls-remote", "--exit-code", "--heads", "origin", str(spec["branch"]))
        payload = {
            "branch": spec["branch"],
            "source_commit": manifest["source_commit"],
            "archive": str(archive),
            "archived_file_count": manifest["archived_file_count"],
            "archived_bytes": manifest["archived_bytes"],
            "hash_verification_pass": True,
            "worktree_absent": not worktree_path.exists() and str(worktree_path) not in worktrees,
            "local_branch_absent": local.returncode != 0,
            "remote_branch_absent": remote.returncode == 2,
            "source_evidence_remaining_before_removal": manifest["source_evidence_remaining"],
        }
        result["branches"][name] = payload
        csv_rows.append({"name": name, **payload})
        all_verified = all_verified and all(
            bool(payload[key])
            for key in (
                "hash_verification_pass",
                "worktree_absent",
                "local_branch_absent",
                "remote_branch_absent",
            )
        )
    result["all_verified"] = all_verified
    if not all_verified:
        result["status"] = "BRANCH_WORKTREE_CLOSEOUT_INCOMPLETE"
        raise RuntimeError(json.dumps(result, sort_keys=True))
    (output / "closeout_complete.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "archive_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
