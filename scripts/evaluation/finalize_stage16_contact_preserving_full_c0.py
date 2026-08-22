#!/usr/bin/env python3
"""Fail-closed closeout for Stage16 contact-preserving full-C0 validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / ".local/runs/stage16_contact_preserving_full_c0_validation"
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_contact_preserving_full_c0_validation"
HISTORICAL = (
    REPO_ROOT / ".local/reports/stage16_grasp_lift_skill_collapse/localization/grasp_vs_update.csv"
)
REPLAY_DRIVER = REPO_ROOT / "scripts/rl/isaaclab/replay_physical_hoi_trace.py"

PPO_FIELDS = (
    "actor_lr",
    "critic_lr",
    "actor_delta_previous",
    "actor_delta_source",
    "kl_previous_to_current",
    "actor_loss",
    "critic_loss",
    "entropy",
    "clip_fraction",
    "actor_grad_norm",
    "critic_grad_norm",
)
C0_TOTAL_SAMPLES = 1_048_576
C0_FROZEN_UPDATE_SAMPLES = 40_960


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-closeout", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CONTACT_PRESERVING_C0_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"CONTACT_PRESERVING_C0_EMPTY_CSV:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _int(value: object, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"CONTACT_PRESERVING_C0_INT_REQUIRED:{field}:{value!r}") from error


def _float(value: object, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"CONTACT_PRESERVING_C0_FLOAT_REQUIRED:{field}:{value!r}") from error


def _optional_float(value: object, *, field: str) -> float | None:
    """Keep a no-contact force percentile distinct from a numeric zero."""

    if value in {None, "", "None"}:
        return None
    return _float(value, field=field)


def _format_float(value: object, *, field: str) -> str:
    parsed = _optional_float(value, field=field)
    return "N/A" if parsed is None else f"{parsed:.6f}"


def _expected_c0_samples(update: int) -> int:
    if not 1 <= update <= 26:
        raise ValueError(f"CONTACT_PRESERVING_C0_UPDATE_OUT_OF_RANGE:{update}")
    return min(update * C0_FROZEN_UPDATE_SAMPLES, C0_TOTAL_SAMPLES)


def _summary(path: Path, *, expected_update: int, expected_episodes: int) -> dict[str, Any]:
    payload = _read_json(path / "summary.json")
    if (
        _int(payload.get("update"), field="update") != expected_update
        or _int(payload.get("episodes"), field="episodes") != expected_episodes
    ):
        raise ValueError(f"CONTACT_PRESERVING_C0_EVALUATION_SUMMARY_DRIFT:{path}")
    return payload


def _required_evidence() -> tuple[list[dict[str, str]], dict[int, dict[str, Any]], dict[str, Any]]:
    complete = _read_json(RUN_ROOT / "training_complete.json")
    if (
        complete.get("FULL_C0_RUN") is not True
        or _int(complete.get("updates"), field="updates") != 26
    ):
        raise ValueError("CONTACT_PRESERVING_C0_FULL_RUN_NOT_CONFIRMED")
    if _int(complete.get("stage_samples"), field="stage_samples") != C0_TOTAL_SAMPLES:
        raise ValueError("CONTACT_PRESERVING_C0_ENDPOINT_SAMPLE_DRIFT")
    if not (REPORT_ROOT / "source/evaluation.csv").is_file():
        raise FileNotFoundError("CONTACT_PRESERVING_C0_SOURCE_EVALUATION_CSV_MISSING")
    rows = _read_csv(REPORT_ROOT / "training/progression.csv")
    if len(rows) != 26 or [_int(row["update"], field="update") for row in rows] != list(
        range(1, 27)
    ):
        raise ValueError("CONTACT_PRESERVING_C0_PROGRESSION_INCOMPLETE")
    summaries: dict[int, dict[str, Any]] = {}
    for update, row in enumerate(rows, start=1):
        expected_samples = _expected_c0_samples(update)
        if _int(row["samples"], field="samples") != expected_samples:
            raise ValueError(f"CONTACT_PRESERVING_C0_SAMPLE_COUNTER_DRIFT:U{update:04d}")
        if any(row.get(field) in {None, "", "None"} for field in PPO_FIELDS):
            raise ValueError(f"CONTACT_PRESERVING_C0_PPO_HEALTH_MISSING:U{update:04d}")
        update_root = REPORT_ROOT / f"training/updates/U{update:04d}"
        for relative in (
            "checkpoint/checkpoint.pt",
            "exact_batch/exact_batch.pt",
            "train_receipt.json",
        ):
            if not (update_root / relative).is_file():
                raise FileNotFoundError(
                    f"CONTACT_PRESERVING_C0_UPDATE_EVIDENCE_MISSING:U{update:04d}:{relative}"
                )
        summaries[update] = _summary(
            REPORT_ROOT / f"frame0_eval/U{update:04d}", expected_update=update, expected_episodes=10
        )
        if _int(summaries[update]["samples"], field="summary_samples") != expected_samples:
            raise ValueError(f"CONTACT_PRESERVING_C0_EVALUATION_SAMPLE_DRIFT:U{update:04d}")
    endpoint = _summary(REPORT_ROOT / "endpoint/eval20", expected_update=26, expected_episodes=20)
    if _int(endpoint["samples"], field="endpoint_samples") != C0_TOTAL_SAMPLES:
        raise ValueError("CONTACT_PRESERVING_C0_ENDPOINT_EVALUATION_SAMPLE_DRIFT")
    if not (REPORT_ROOT / "replay/headless_validation.json").is_file():
        raise FileNotFoundError("CONTACT_PRESERVING_C0_HEADLESS_REPLAY_VALIDATION_MISSING")
    replay = _read_json(REPORT_ROOT / "replay/headless_validation.json")
    if (
        replay.get("status") != "STAGE16D_PPO26D_REPLAY_VALIDATED"
        or replay.get("finite") is not True
    ):
        raise ValueError("CONTACT_PRESERVING_C0_HEADLESS_REPLAY_NOT_PASSED")
    tests = _read_json(REPORT_ROOT / "tests.json")
    if tests.get("final_full_suite") != "PASS":
        raise ValueError("CONTACT_PRESERVING_C0_FINAL_TEST_EVIDENCE_MISSING")
    return rows, summaries, endpoint


def _source_summary() -> dict[str, Any]:
    source = _summary(REPORT_ROOT / "source", expected_update=0, expected_episodes=10)
    if (
        _int(source["persistent_grasp_episodes"], field="source_grasp") != 10
        or _int(source["lift_episodes"], field="source_lift") != 10
    ):
        raise ValueError("SOURCE_POLICY_REGRESSION")
    _write_csv(REPORT_ROOT / "source/evaluation.csv", [source])
    return source


def _historical_row(label: str, *, historical_path: Path = HISTORICAL) -> dict[str, str]:
    rows = [row for row in _read_csv(historical_path) if row.get("stage") == "C0"]
    # The original C0 evidence switches from ``U6`` to ``C0_U7`` after the
    # continuation boundary.  Both spellings name the same update under the
    # single frozen C0 lineage, so accept only that explicit prefix alias.
    accepted_labels = {label}
    if label.startswith("U") and label[1:].isdigit():
        accepted_labels.add(f"C0_{label}")
    matches = [row for row in rows if row.get("label") in accepted_labels]
    if len(matches) != 1:
        raise ValueError(f"CONTACT_PRESERVING_C0_HISTORICAL_POINT_MISSING:{label}")
    return matches[0]


def _nearest(rows: list[dict[str, str]], samples: int) -> dict[str, str]:
    return min(rows, key=lambda row: abs(_int(row["samples"], field="samples") - samples))


def _milestones(rows: list[dict[str, str]], source: dict[str, Any]) -> dict[str, object]:
    source_grasp = _int(source["persistent_grasp_episodes"], field="source_grasp")
    source_lift = _int(source["lift_episodes"], field="source_lift")

    def first(key: str, predicate: Any) -> int | None:
        for row in rows:
            if predicate(_int(row[key], field=key)):
                return _int(row["update"], field="update")
        return None

    persistent_zero_lift = None
    for prior, current in zip(rows, rows[1:], strict=True):
        if (
            _int(prior["lift_episodes"], field="lift") == 0
            and _int(current["lift_episodes"], field="lift") == 0
        ):
            persistent_zero_lift = _int(prior["update"], field="update")
            break
    return {
        "major_degradation_definition": "at_or_below_half_of_source_episode_count",
        "U_FIRST_GRASP_DEGRADATION": first(
            "persistent_grasp_episodes", lambda value: value < source_grasp
        ),
        "U_MAJOR_GRASP_DEGRADATION": first(
            "persistent_grasp_episodes", lambda value: value <= source_grasp / 2
        ),
        "U_ZERO_GRASP": first("persistent_grasp_episodes", lambda value: value == 0),
        "U_FIRST_LIFT_DEGRADATION": first("lift_episodes", lambda value: value < source_lift),
        "U_MAJOR_LIFT_DEGRADATION": first("lift_episodes", lambda value: value <= source_lift / 2),
        "U_ZERO_LIFT": first("lift_episodes", lambda value: value == 0),
        "U_PERSISTENT_ZERO_LIFT": persistent_zero_lift,
    }


def _classification(
    rows: list[dict[str, str]], endpoint: dict[str, Any], *, u26_equivalent: dict[str, str]
) -> tuple[str, dict[str, object]]:
    zero = [
        row
        for row in rows
        if _int(row["persistent_grasp_episodes"], field="grasp") == 0
        or _int(row["lift_episodes"], field="lift") == 0
    ]
    u26_samples = _int(u26_equivalent["samples"], field="u26_samples")
    avoided_original_u26 = (
        _int(u26_equivalent["persistent_grasp_episodes"], field="u26_grasp") > 0
        and _int(u26_equivalent["lift_episodes"], field="u26_lift") > 0
    )
    if zero:
        first_zero = zero[0]
        if _int(first_zero["samples"], field="zero_samples") < u26_samples:
            classification = "CANDIDATE_REGRESSION"
        elif avoided_original_u26:
            classification = "PRESERVATION_ONLY_DELAYS_COLLAPSE"
        else:
            classification = "CANDIDATE_REGRESSION"
    elif (
        all(_int(row["persistent_grasp_episodes"], field="grasp") == 10 for row in rows)
        and all(_int(row["lift_episodes"], field="lift") == 10 for row in rows)
        and _int(endpoint["persistent_grasp_episodes"], field="endpoint_grasp") == 20
        and _int(endpoint["lift_episodes"], field="endpoint_lift") == 20
    ):
        classification = "FULL_C0_PRESERVATION_VALIDATED"
    else:
        classification = "PARTIAL_PRESERVATION"
    return classification, {
        "DID_CANDIDATE_AVOID_ORIGINAL_U26_COLLAPSE": "YES" if avoided_original_u26 else "NO",
        "DID_COLLAPSE_REAPPEAR_LATER": "YES" if bool(zero) and avoided_original_u26 else "NO",
        "first_zero_update": None if not zero else _int(zero[0]["update"], field="zero_update"),
        "first_zero_samples": None if not zero else _int(zero[0]["samples"], field="zero_samples"),
    }


def _core_curve(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "update": _int(row["update"], field="update"),
            "samples": _int(row["samples"], field="samples"),
            "persistent_grasp_episodes": _int(row["persistent_grasp_episodes"], field="grasp"),
            "lift_episodes": _int(row["lift_episodes"], field="lift"),
            "contact_fraction": _float(row["contact_fraction"], field="contact_fraction"),
            "force_p95_n": _optional_float(row["force_p95_n"], field="force"),
            "tip_recall": _float(row["tip_recall"], field="tip_recall"),
            "lift_dz_m": _float(row["lift_dz_m"], field="lift_dz"),
        }
        for row in rows
    ]


def _candidate_episode_count(*, point: str, candidate_metrics: dict[str, Any]) -> int:
    """Per-update progression rows inherit the fixed Eval10 protocol, not an episodes field."""

    if point in {"Source", "Endpoint"}:
        return _int(candidate_metrics["episodes"], field="candidate_episodes")
    return 10


def _comparison(
    rows: list[dict[str, str]], source: dict[str, Any], endpoint: dict[str, Any]
) -> list[dict[str, object]]:
    points = (
        ("Source", "SOURCE"),
        ("U6-equivalent", "U6"),
        ("U25-equivalent", "U25"),
        ("U26-equivalent", "U26"),
        ("Endpoint", "U26"),
    )
    output: list[dict[str, object]] = []
    for point, historical_label in points:
        historical = _historical_row(historical_label)
        baseline_samples = _int(historical["samples"], field="baseline_samples")
        candidate = _nearest(rows, baseline_samples) if point != "Source" else None
        candidate_metrics: dict[str, Any]
        if point == "Source":
            candidate_metrics = source
        elif point == "Endpoint":
            candidate_metrics = endpoint
        else:
            assert candidate is not None
            candidate_metrics = candidate
        candidate_episodes = _candidate_episode_count(
            point=point, candidate_metrics=candidate_metrics
        )
        baseline_episodes = _int(historical["episodes"], field="baseline_episodes")
        baseline_grasp = round(
            _float(historical["grasp_episode_rate"], field="baseline_grasp") * baseline_episodes
        )
        baseline_lift = round(
            _float(historical["lift_episode_rate"], field="baseline_lift") * baseline_episodes
        )
        output.append(
            {
                "point": point,
                "baseline_label": historical["label"],
                "baseline_samples": baseline_samples,
                "baseline_1x_persistent_grasp_episodes": baseline_grasp,
                "baseline_1x_lift_episodes": baseline_lift,
                "candidate_update": (
                    0
                    if point == "Source"
                    else _int(candidate_metrics["update"], field="candidate_update")
                ),
                "candidate_samples": _int(candidate_metrics["samples"], field="candidate_samples"),
                "candidate_0p5_persistent_grasp_episodes": _int(
                    candidate_metrics["persistent_grasp_episodes"], field="candidate_grasp"
                ),
                "candidate_0p5_lift_episodes": _int(
                    candidate_metrics["lift_episodes"], field="candidate_lift"
                ),
                "candidate_episodes": candidate_episodes,
            }
        )
    return output


def _replay_commands(rows: list[dict[str, str]]) -> str:
    u25 = _nearest(rows, 1_024_000)
    u26 = _nearest(rows, 1_048_576)
    traces = {
        "Source": REPORT_ROOT / "source/contact_eval/SOURCE/episode_00.npz",
        "candidate U25-equivalent": REPORT_ROOT
        / f"frame0_eval/U{_int(u25['update'], field='u25_update'):04d}"
        / f"contact_eval/C0_U{_int(u25['update'], field='u25_update')}/episode_00.npz",
        "candidate U26-equivalent": REPORT_ROOT
        / f"frame0_eval/U{_int(u26['update'], field='u26_update'):04d}"
        / f"contact_eval/C0_U{_int(u26['update'], field='u26_update')}/episode_00.npz",
        "C0 endpoint": REPORT_ROOT / "endpoint/eval20/contact_eval/C0_ENDPOINT/episode_00.npz",
    }
    for label, trace in traces.items():
        if not trace.is_file():
            raise FileNotFoundError(f"CONTACT_PRESERVING_C0_REPLAY_TRACE_MISSING:{label}:{trace}")
    lines = ["# Stage16 Contact-Preserving C0 Replay Commands", ""]
    for label, trace in traces.items():
        lines.extend(
            (
                f"## {label}",
                "",
                "```bash",
                "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES "
                f"python {REPLAY_DRIVER} --accept-eula --object hocap_170105 --trace {trace}",
                "```",
                "",
            )
        )
    return "\n".join(lines)


def _markdown(
    *,
    source: dict[str, Any],
    endpoint: dict[str, Any],
    comparison: list[dict[str, object]],
    curve: list[dict[str, object]],
    classification: str,
    decision: dict[str, object],
    milestones: dict[str, object],
    best: dict[str, object],
) -> str:
    lines = [
        "# Stage16 Contact-Preserving Full C0 Longitudinal Validation",
        "",
        f"Classification: `{classification}`.",
        "",
        "## Source and endpoint",
        "",
        "| Point | Persistent grasp | Lift | Contact fraction | Force p95 (N) | Lift dz (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in (("Source Eval10", source), ("Endpoint Eval20", endpoint)):
        lines.append(
            f"| {name} | {metric['persistent_grasp_episodes']}/{metric['episodes']} | "
            f"{metric['lift_episodes']}/{metric['episodes']} | {metric['contact_fraction']:.6f} | "
            f"{_format_float(metric.get('active_contact_force_p95_n'), field='force_p95')} | "
            f"{metric['object_lift_dz_mean']:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Baseline versus candidate",
            "",
            "| Point | 1.0x baseline grasp | 1.0x baseline lift | "
            "0.5x candidate grasp | 0.5x candidate lift |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in comparison:
        lines.append(
            f"| {row['point']} | {row['baseline_1x_persistent_grasp_episodes']} | "
            f"{row['baseline_1x_lift_episodes']} | "
            f"{row['candidate_0p5_persistent_grasp_episodes']} | "
            f"{row['candidate_0p5_lift_episodes']} |"
        )
    lines.extend(
        (
            "",
            "## Full C0 longitudinal curve",
            "",
            "| Update | Samples | Persistent grasp | Lift | Contact frac | Force p95 | "
            "Tip recall | Lift dz |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for row in curve:
        lines.append(
            f"| {row['update']} | {row['samples']} | {row['persistent_grasp_episodes']}/10 | "
            f"{row['lift_episodes']}/10 | {row['contact_fraction']:.6f} | "
            f"{_format_float(row['force_p95_n'], field='force_p95')} | "
            f"{row['tip_recall']:.6f} | {row['lift_dz_m']:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            "- Original U26 collapse avoided: "
            f"`{decision['DID_CANDIDATE_AVOID_ORIGINAL_U26_COLLAPSE']}`",
            f"- Collapse reappeared later: `{decision['DID_COLLAPSE_REAPPEAR_LATER']}`",
            f"- Milestones: `{json.dumps(milestones, sort_keys=True)}`",
            f"- Best grasp/lift checkpoint: U{best['update']:04d}, {best['samples']} samples, "
            f"SHA256 `{best['checkpoint_sha256']}`.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_closeout:
        raise ValueError("--accept-closeout is required")
    source = _source_summary()
    rows, summaries, endpoint = _required_evidence()
    curve = _core_curve(rows)
    comparison = _comparison(rows, source, endpoint)
    u26_equivalent = _nearest(rows, _int(_historical_row("U26")["samples"], field="u26_samples"))
    classification, decision = _classification(rows, endpoint, u26_equivalent=u26_equivalent)
    milestones = _milestones(rows, source)
    best_row = max(
        rows,
        key=lambda row: (
            _int(row["lift_episodes"], field="lift"),
            _int(row["persistent_grasp_episodes"], field="grasp"),
            _optional_float(row["force_p95_n"], field="force") or float("-inf"),
        ),
    )
    checkpoint = (
        RUN_ROOT
        / f"training/updates/U{_int(best_row['update'], field='best_update'):04d}"
        / "checkpoint/checkpoint.pt"
    )
    best = {
        "update": _int(best_row["update"], field="best_update"),
        "samples": _int(best_row["samples"], field="best_samples"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "persistent_grasp_episodes": _int(
            best_row["persistent_grasp_episodes"], field="best_grasp"
        ),
        "lift_episodes": _int(best_row["lift_episodes"], field="best_lift"),
        "force_p95_n": _optional_float(best_row["force_p95_n"], field="best_force"),
    }
    _write_csv(REPORT_ROOT / "comparison/baseline_1x_vs_candidate_0p5.csv", comparison)
    comparison_markdown = _markdown(
        source=source,
        endpoint=endpoint,
        comparison=comparison,
        curve=curve,
        classification=classification,
        decision=decision,
        milestones=milestones,
        best=best,
    )
    (REPORT_ROOT / "comparison/baseline_1x_vs_candidate_0p5.md").write_text(
        comparison_markdown, encoding="utf-8"
    )
    (REPORT_ROOT / "replay/visualization_commands.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "replay/visualization_commands.md").write_text(
        _replay_commands(rows), encoding="utf-8"
    )
    preflight = _read_json(REPORT_ROOT / "git_preflight.json")
    final_head = _git("rev-parse", "HEAD")
    status = (
        "LONGITUDINALLY_VALIDATED_AT_C0"
        if classification == "FULL_C0_PRESERVATION_VALIDATED"
        else "SHADOW_ONLY_NOT_SUFFICIENT"
    )
    next_action = (
        "NEXT_CONTACT_PRESERVING_C1_VALIDATION"
        if classification == "FULL_C0_PRESERVATION_VALIDATED"
        else "NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION"
    )
    final = {
        "schema_version": "Stage16ContactPreservingFullC0ValidationV1",
        "classification": classification,
        "candidate_status": status,
        "next_action": next_action,
        "source_eval10": source,
        "endpoint_eval20": endpoint,
        "longitudinal_curve": curve,
        "historical_comparison": comparison,
        "milestones": milestones,
        "danger_window": decision,
        "best_grasp_lift_checkpoint": best,
        "CONTROLLER_REGRESSION": "NO_OBSERVED_RUNTIME_OR_FINITE_TRACKING_REGRESSION",
        "PRODUCTION_DEFAULT_SWITCHED": "NO",
        "C1_STARTED": "NO",
        "C2_STARTED": "NO",
        "C3_STARTED": "NO",
        "C4_STARTED": "NO",
        "V4_TRAINING_STARTED": "NO",
        "HOCAP_170650_TRAINING_STARTED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "git": {
            "branch": _git("branch", "--show-current"),
            "START_HEAD": preflight["START_HEAD"],
            "FINAL_HEAD": final_head,
            "worktree_status": _git("status", "--short", "--untracked-files=all"),
        },
    }
    _write_json(REPORT_ROOT / "final_summary.json", final)
    _write_json(
        REPORT_ROOT / "git_commits.json",
        {
            "START_HEAD": preflight["START_HEAD"],
            "FINAL_HEAD": final_head,
            "commits": _git("log", "--oneline", f"{preflight['START_HEAD']}..{final_head}"),
        },
    )
    (REPORT_ROOT / "final_summary.md").write_text(comparison_markdown, encoding="utf-8")
    (REPORT_ROOT / "handoff.md").write_text(
        "# Stage16 Contact-Preserving Full C0 Longitudinal Validation Handoff\n\n"
        f"Classification: `{classification}`. Candidate status: `{status}`. "
        f"Next action: `{next_action}`.\n\n" + comparison_markdown,
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "candidate_status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
