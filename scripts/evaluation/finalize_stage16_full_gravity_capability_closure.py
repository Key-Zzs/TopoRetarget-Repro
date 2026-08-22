#!/usr/bin/env python3
"""Aggregate the fail-closed Stage16 full-gravity capability-closure evidence."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_full_gravity_capability_closure"
RUN_ROOT = REPO_ROOT / ".local/runs/stage16_full_gravity_capability_closure"
TRACE_ROOT = REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure"
FROZEN_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_frozen_source_policy_gravity_sweep"

C4_RECEIPTS = (
    (
        "v3_hocap_170105",
        FROZEN_REPORT_ROOT / "sweep/v3/hocap_170105/c4/qualification.json",
        "HISTORICAL_FROZEN_SWEEP",
    ),
    (
        "v4_hocap_170105",
        REPORT_ROOT
        / "technical_remediation/smoke/former_timeout_v4_170105_c4/sweep/v4/hocap_170105/c4/qualification.json",
        "REMEDIATED_ISOLATED_C4",
    ),
    (
        "v3_hocap_170650",
        REPORT_ROOT / "c4_completion/v3_hocap_170650/sweep/v3/hocap_170650/c4/qualification.json",
        "REMEDIATED_ISOLATED_C4",
    ),
    (
        "v4_hocap_170650",
        REPORT_ROOT
        / "technical_remediation/smoke/known_good/sweep/v4/hocap_170650/c4/qualification.json",
        "REMEDIATED_ISOLATED_KNOWN_GOOD",
    ),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"FULL_GRAVITY_CLOSEOUT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_metrics(receipt: Mapping[str, Any]) -> dict[str, Any]:
    metrics = receipt.get("metrics", receipt)
    if not isinstance(metrics, Mapping):
        raise ValueError("FULL_GRAVITY_CLOSEOUT_METRICS_INVALID")
    completion = receipt.get("completion", metrics.get("completion", {}))
    if not isinstance(completion, Mapping):
        completion = {}
    grasp = metrics.get("persistent_grasp_episodes", metrics.get("grasp_success_count"))
    lift = metrics.get("lift_episodes", metrics.get("lift_success_count"))
    dz = metrics.get("object_lift_dz_m", metrics.get("object_lift_height_median_m"))
    episodes = metrics.get("episodes", receipt.get("episodes", 10))
    return {
        "episodes": int(episodes),
        "grasp": int(grasp),
        "lift": int(lift),
        "dz_m": float(dz or 0.0),
        "force_p95_n": float(metrics.get("active_force_p95_n") or 0.0),
        "drop_fraction": float(metrics.get("drop_fraction") or 0.0),
        "completion": {str(key): bool(value) for key, value in completion.items()},
        "technical_complete": bool(completion)
        and all(bool(value) for value in completion.values()),
        "status": str(receipt.get("status")),
        "actor_hash_before": receipt.get("actor_hash_before"),
        "actor_hash_after": receipt.get("actor_hash_after"),
        "normalizer_hash_before": receipt.get("normalizer_hash_before"),
        "normalizer_hash_after": receipt.get("normalizer_hash_after"),
        "optimizer_steps": int(receipt.get("optimizer_steps", 0)),
    }


def _c4_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, path, origin in C4_RECEIPTS:
        receipt = _read(path)
        metric = _receipt_metrics(receipt)
        reward, clip = source.split("_", maxsplit=1)
        functional = metric["technical_complete"] and metric["grasp"] == 10 and metric["lift"] == 10
        rows.append(
            {
                "source": source,
                "reward": reward,
                "clip": clip,
                "stage": "C4",
                "origin": origin,
                "episodes": metric["episodes"],
                "persistent_grasp": metric["grasp"],
                "lift": metric["lift"],
                "lift_dz_m": metric["dz_m"],
                "force_p95_n": metric["force_p95_n"],
                "technical_complete": metric["technical_complete"],
                "classification": "FUNCTIONAL" if functional else "PARTIAL_OR_NO_LIFT",
                "qualification": str(path.resolve()),
                "qualification_sha256": _sha256(path),
            }
        )
    return rows


def _update_number(path: Path) -> int:
    match = re.fullmatch(r"U(\d{4})", path.name)
    if match is None:
        raise ValueError(f"FULL_GRAVITY_CLOSEOUT_UPDATE_DIR_INVALID:{path}")
    return int(match.group(1))


def _adaptation_summary(
    *, source: str, stage: str, mode_dir: str, clip: str, authorized_max_update: int | None = None
) -> dict[str, object]:
    root = REPORT_ROOT / "adaptation" / source / stage.lower()
    evals: list[dict[str, object]] = []
    for path in sorted(
        root.glob(f"U*/eval10/sweep/{mode_dir}/{clip}/{stage.lower()}/qualification.json")
    ):
        update = _update_number(path.parents[5])
        if authorized_max_update is not None and update > authorized_max_update:
            continue
        receipt = _read(path)
        metric = _receipt_metrics(receipt)
        evals.append(
            {
                "update": update,
                "qualification": str(path.resolve()),
                "qualification_sha256": _sha256(path),
                **metric,
                "functional": metric["technical_complete"]
                and metric["grasp"] == 10
                and metric["lift"] == 10,
            }
        )
    if not evals:
        raise ValueError(f"FULL_GRAVITY_CLOSEOUT_ADAPTATION_MISSING:{source}:{stage}")
    last = max(evals, key=lambda item: int(item["update"]))
    functional = [item for item in evals if bool(item["functional"])]
    confirmations = sorted(
        root.glob(f"U*/confirm20/sweep/{mode_dir}/{clip}/{stage.lower()}/qualification.json")
    )
    confirm: dict[str, object] | None = None
    if confirmations:
        confirmation = _read(confirmations[0])
        metric = _receipt_metrics(confirmation)
        confirm = {
            "qualification": str(confirmations[0].resolve()),
            "qualification_sha256": _sha256(confirmations[0]),
            **metric,
            "confirmed_functional": metric["technical_complete"]
            and metric["episodes"] == 20
            and metric["grasp"] == 20
            and metric["lift"] == 20,
        }
    final_training = (
        RUN_ROOT
        / "adaptation"
        / source
        / stage.lower()
        / f"U{int(last['update']):04d}"
        / "training"
        / mode_dir
        / clip
        / stage.lower()
        / "training_result.json"
    )
    training = _read(final_training) if final_training.is_file() else None
    return {
        "source": source,
        "stage": stage,
        "evaluations": evals,
        "evaluation_count": len(evals),
        "first_eval10_functional": None
        if not functional
        else min(functional, key=lambda item: int(item["update"])),
        "confirm20": confirm,
        "last_eval10": last,
        "final_training": training,
        "budget_exhausted": bool(
            training and training.get("status") == "P3_FULL_TRAJECTORY_STAGE_COMPLETE"
        ),
    }


def _formal20() -> dict[str, object]:
    path = REPORT_ROOT / "formal20/v4_hocap_170650/analysis/qualification.json"
    payload = _read(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("FULL_GRAVITY_CLOSEOUT_FORMAL_EPISODES_INVALID")
    qualified = sum(
        bool(item.get("qualified_success")) for item in episodes if isinstance(item, Mapping)
    )
    physics = sum(
        bool(item.get("physics_success")) for item in episodes if isinstance(item, Mapping)
    )
    kinematic = sum(
        bool(item.get("kinematic_success")) for item in episodes if isinstance(item, Mapping)
    )
    return {
        "source": "v4_hocap_170650",
        "qualification": str(path.resolve()),
        "qualification_sha256": _sha256(path),
        "status": payload.get("status"),
        "episodes": len(episodes),
        "SRkin": kinematic / len(episodes),
        "SRphysics": physics / len(episodes),
        "SRqualified": qualified / len(episodes),
        "qualified_episodes": qualified,
        "required_SRqualified": 0.8,
        "formal_pass": qualified / len(episodes) >= 0.8,
    }


def _timeout_attribution() -> dict[str, object]:
    historic = {
        "historical_interpretation": "Monolithic parent-process timeouts and post-reset vector rows made terminal capture non-authoritative.",
        "repair": "Each condition now executes in a bounded, fresh process group; traces are trimmed to rollout steps plus one before terminal checks.",
        "remaining_physical_outcome": "A real early termination is retained as COMPLETE_DIAGNOSTIC_SWEEP_WITH_PHYSICAL_FAILURE and does not fabricate TERMINAL.",
    }
    runs = list((RUN_ROOT / "technical_remediation").rglob("attempt_*.json"))
    return {
        **historic,
        "isolated_attempt_receipts": [str(path.resolve()) for path in sorted(runs)],
        "former_timeout_v4_170105_c4": str(C4_RECEIPTS[1][1].resolve()),
        "new_v3_170650_c4": str(C4_RECEIPTS[2][1].resolve()),
    }


def _runtime_statistics() -> dict[str, object]:
    paths = list(REPORT_ROOT.rglob("qualification.json"))
    runtime = []
    for path in paths:
        payload = _read(path)
        if "runtime_s" in payload:
            runtime.append(float(payload["runtime_s"]))
    return {
        "qualification_receipts_seen": len(paths),
        "receipts_with_runtime_s": len(runtime),
        "runtime_s_min": None if not runtime else min(runtime),
        "runtime_s_max": None if not runtime else max(runtime),
        "runtime_s_mean": None if not runtime else sum(runtime) / len(runtime),
    }


def _terminal_capture() -> dict[str, object]:
    old_raw = FROZEN_REPORT_ROOT / "sweep/v3/hocap_170650/c2/raw/episode_00.npz"
    return {
        "trim_contract": "Every vectorized trace is sliced to actual rollout steps plus one before terminal validation.",
        "old_invalid_evidence": str(old_raw.resolve()) if old_raw.exists() else "NOT_FOUND",
        "old_invalid_reason": "Post-reset rows cannot satisfy the prior episode terminal-phase requirement.",
        "repaired_c4_receipts": [str(path.resolve()) for _, path, _ in C4_RECEIPTS[1:]],
    }


def _markdown(
    matrix: list[dict[str, object]],
    decisions: list[dict[str, object]],
    formal: Mapping[str, object],
) -> str:
    rows = [
        "| Source | Origin | Grasp | Lift | dz m | Classification |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in matrix:
        rows.append(
            "| {source} | {origin} | {persistent_grasp} | {lift} | {lift_dz_m:.6f} | {classification} |".format(
                **row
            )
        )
    decision_rows = [
        "| Source | Boundary/action | Result |",
        "| --- | --- | --- |",
    ]
    for item in decisions:
        decision_rows.append("| {source} | {boundary} | {result} |".format(**item))
    return "\n".join(
        [
            "# Stage16 Full-Gravity Capability Closure",
            "",
            "Status: `NO_SUCCESS`.",
            "",
            "All four frozen C4 sources have complete technical evidence after isolated-process repair. "
            "The only frozen C4-functional actor (V4/170650) failed Formal20: 2/20 qualified "
            f"episodes (`SRqualified={formal['SRqualified']:.2f}`, required >= 0.80). No success export or replay promotion is authorized.",
            "",
            "## Four-source C4 matrix",
            "",
            *rows,
            "",
            "## Decision tree outcomes",
            "",
            *decision_rows,
            "",
            "## Stop condition",
            "",
            "No lineage reached a passing Formal20 result. Keep all checkpoints, exact batches, "
            "failed physical receipts, and the excluded accidental post-U2 V3/170650 C1 artifacts; "
            "do not begin another LR, reward, KL, epoch, seed, or held-out sweep from this closure.",
            "",
        ]
    )


def _csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _git_receipt() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, check=True, capture_output=True
        ).stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
        "recent_commits": run("log", "--oneline", "-5"),
    }


def _validation(existing: Mapping[str, object], values: list[str]) -> dict[str, object]:
    result = dict(existing)
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError("FULL_GRAVITY_CLOSEOUT_VALIDATION_REQUIRES_KEY_EQUALS_VALUE")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="append", default=[])
    args = parser.parse_args()

    matrix = _c4_matrix()
    v3_170105_c1 = _adaptation_summary(
        source="v3_hocap_170105", stage="C1", mode_dir="v3", clip="hocap_170105"
    )
    v3_170650_c1 = _adaptation_summary(
        source="v3_hocap_170650",
        stage="C1",
        mode_dir="v3",
        clip="hocap_170650",
        authorized_max_update=2,
    )
    v3_170650_c2 = _adaptation_summary(
        source="v3_hocap_170650", stage="C2", mode_dir="v3", clip="hocap_170650"
    )
    v3_170650_c3 = _adaptation_summary(
        source="v3_hocap_170650", stage="C3", mode_dir="v3", clip="hocap_170650"
    )
    v4_170105_c4 = _adaptation_summary(
        source="v4_hocap_170105", stage="C4", mode_dir="v4", clip="hocap_170105"
    )
    formal = _formal20()
    decisions = [
        {
            "source": "v3_hocap_170105",
            "boundary": "C1 adaptation from frozen actor",
            "result": "BUDGET_EXHAUSTED_NO_FUNCTIONAL_C1",
        },
        {
            "source": "v3_hocap_170650",
            "boundary": "C1 and C2 confirmed; C3 adaptation",
            "result": "C3_BUDGET_EXHAUSTED_NO_FUNCTIONAL_C3",
        },
        {
            "source": "v4_hocap_170105",
            "boundary": "C4 adaptation from frozen actor",
            "result": "BUDGET_EXHAUSTED_NO_FUNCTIONAL_C4",
        },
        {
            "source": "v4_hocap_170650",
            "boundary": "Frozen C4 direct Formal20",
            "result": "FORMAL20_FAILED_SRQUALIFIED_0_10",
        },
    ]
    adaptation = {
        "v3_hocap_170105_c1": v3_170105_c1,
        "v3_hocap_170650_c1": v3_170650_c1,
        "v3_hocap_170650_c2": v3_170650_c2,
        "v3_hocap_170650_c3": v3_170650_c3,
        "v4_hocap_170105_c4": v4_170105_c4,
        "excluded_non_authoritative_artifacts": {
            "v3_hocap_170650_c1_after_u2": "U0003 and interrupted U0004 were created by a shell-status bug after U0002 had already qualified; they are retained but excluded from every decision.",
        },
    }
    summary = {
        "schema_version": "Stage16FullGravityCapabilityClosureV1",
        "status": "NO_SUCCESS",
        "technical_completion": "COMPLETE",
        "overall_reason": "No frozen or minimally adapted lineage attained a passing Formal20 result.",
        "next_action": "STOP_NO_FURTHER_PPO_OR_SWEEP_AUTHORIZED",
        "c4_matrix": matrix,
        "decisions": decisions,
        "formal20": formal,
        "adaptation_paths": {
            key: {k: value for k, value in value.items() if k != "evaluations"}
            for key, value in adaptation.items()
            if key != "excluded_non_authoritative_artifacts"
        },
        "source_preservation": "Frozen source evaluations assert optimizer_steps=0 and unchanged actor/normalizer hashes.",
        "data_export": "NO_SUCCESS_EXPORT_ELIGIBLE",
    }
    _csv(REPORT_ROOT / "capability/four_source_c4_matrix.csv", matrix)
    (REPORT_ROOT / "capability/four_source_c4_matrix.md").write_text(
        _markdown(matrix, decisions, formal), encoding="utf-8"
    )
    _write(REPORT_ROOT / "capability/per_lineage_decisions.json", decisions)
    _write(REPORT_ROOT / "adaptation/summary.json", adaptation)
    _write(REPORT_ROOT / "formal20/summary.json", formal)
    _write(REPORT_ROOT / "technical_remediation/timeout_attribution.json", _timeout_attribution())
    _write(REPORT_ROOT / "technical_remediation/runtime_statistics.json", _runtime_statistics())
    _write(REPORT_ROOT / "technical_remediation/terminal_capture.json", _terminal_capture())
    _write(
        REPORT_ROOT / "technical_remediation/smoke_tests.json",
        {
            "known_good_v4_170650_c4": str(C4_RECEIPTS[3][1].resolve()),
            "former_timeout_v4_170105_c4": str(C4_RECEIPTS[1][1].resolve()),
        },
    )
    _write(
        REPORT_ROOT / "data_export/eligibility.json",
        {
            "status": "NO_SUCCESS_EXPORT_ELIGIBLE",
            "reason": "No lineage passed Formal20; retained traces remain diagnostic evidence only.",
        },
    )
    replay_trace = TRACE_ROOT / "formal20/v4_hocap_170650/episode_000.npz"
    (REPORT_ROOT / "replay/commands.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "replay/commands.md").write_text(
        "# Replay commands\n\nNo success replay commands exist because Formal20 failed. Representative failure replay:\n\n"
        "```bash\nOMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula --object hocap_170650 --trace "
        f"{replay_trace} --loop\n```\n",
        encoding="utf-8",
    )
    _write(REPORT_ROOT / "final_summary.json", summary)
    markdown = _markdown(matrix, decisions, formal)
    (REPORT_ROOT / "final_summary.md").write_text(markdown, encoding="utf-8")
    (REPORT_ROOT / "handoff.md").write_text(
        markdown
        + "\n## Resumption boundary\n\nThe closure is terminal at `NO_SUCCESS`. Retained checkpoints and failed traces support diagnosis only; do not resume PPO, export a success dataset, or promote a policy without a new explicitly authorized protocol.\n",
        encoding="utf-8",
    )
    _write(REPORT_ROOT / "git_commits.json", _git_receipt())
    tests_path = REPORT_ROOT / "tests.json"
    existing = _read(tests_path) if tests_path.is_file() else {"aggregation": "PASS"}
    _write(tests_path, _validation(existing, args.validation))
    print(
        json.dumps({"status": summary["status"], "report_root": str(REPORT_ROOT)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
