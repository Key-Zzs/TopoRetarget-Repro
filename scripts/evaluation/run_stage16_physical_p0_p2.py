#!/usr/bin/env python3
"""Materialize Stage 16 P0/P1/P2 evidence without starting any PPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.support_contract import (  # noqa: E402
    SourceSupportContractV1,
    discover_source_support_evidence,
)
from toporetarget.physics.support_feasibility import (  # noqa: E402
    build_support_timeline,
    decide_support_mode,
    support_diagnostic_summary,
)
from toporetarget.rl.physical_stage import (  # noqa: E402
    load_p1_rsi_acceptance_contract,
    load_p3_entry_gate,
    load_physical_bootstrap_contract,
)
from toporetarget.rl.rsi.contact_ready_v2 import (  # noqa: E402
    INITIAL_P3_BANKS,
    RSIStateSemanticClass,
    build_contact_ready_state_bank,
    build_safe_bank,
    save_safe_bank,
    save_state_bank,
    summarize_state_bank,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2"
SOURCE_CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
SOURCE_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data/subject_1")
CLIP_TO_SEQUENCE = {
    "hocap_170105": "20231025_170105",
    "hocap_170650": "20231025_170650",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PHYSICAL_STAGE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_path(name: str) -> Path:
    return REPO_ROOT / "configs/rl/stage16" / name


def _parent_provenance() -> dict[str, object]:
    inputs = {
        "zero_g_parent_contract": _config_path("stage16d_causal_zero_g_milestone.yaml"),
        "physical_bootstrap_contract": _config_path("stage16_physical_bootstrap.yaml"),
        "p3_entry_gate": _config_path("stage16_p3_entry_gate_v1.yaml"),
        "reference_kinematics_qualification": REPO_ROOT
        / ".local/reports/stage16d_reference_kinematics_v2/reference_kinematics_qualification.json",
        "evaluation_suite_v2": REPO_ROOT / "configs/evaluation/stage16_evaluation_suite_v2.yaml",
        "source_contact_contract": SOURCE_CONTACT_ROOT / "source_contact_contract.json",
        "contact_reward_configuration": _config_path("stage16d_ppo26d_reward.yaml"),
        "aggregate_v3_frozen_contract": REPO_ROOT
        / ".local/reports/stage16d_reward_v3_pairforce_unblock/contact_reward_contract.json",
        "strict_v4_frozen_contract": REPO_ROOT
        / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"PHYSICAL_BOOTSTRAP_PARENT_INPUT_MISSING:{missing}")
    source = _read_json(inputs["source_contact_contract"])
    if source.get("schema_version") != "SourcePerFingerContactEvidenceV1":
        raise ValueError("PHYSICAL_BOOTSTRAP_SOURCE_CONTACT_CONTRACT_INVALID")
    evaluation_text = inputs["evaluation_suite_v2"].read_text(encoding="utf-8")
    if "TopoRetargetEvaluationSuiteV2" not in evaluation_text:
        raise ValueError("PHYSICAL_BOOTSTRAP_EVALUATION_SUITE_INVALID")
    return {
        "schema_version": "Stage16PhysicalBootstrapParentProvenanceV1",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()
        },
        "parent_hashes_valid": True,
    }


def run_p0() -> None:
    bootstrap_path = _config_path("stage16_physical_bootstrap.yaml")
    gate_path = _config_path("stage16_p3_entry_gate_v1.yaml")
    bootstrap = load_physical_bootstrap_contract(bootstrap_path)
    entry_gate = load_p3_entry_gate(gate_path)
    acceptance = load_p1_rsi_acceptance_contract(gate_path)
    provenance = _parent_provenance()
    p0 = REPORT_ROOT / "p0"
    _write_json(p0 / "physical_bootstrap_contract.json", bootstrap.as_dict())
    _write_json(p0 / "p3_entry_gate_v1.json", entry_gate)
    _write_json(p0 / "parent_provenance.json", provenance)
    _write_json(
        p0 / "config_smoke.json",
        {
            "status": "PASS",
            "physical_bootstrap": bootstrap.as_dict(),
            "p1_acceptance": acceptance.as_dict(),
            "invalid_target_mode_policy": "fail_fast",
            "contact_modes": ["aggregate_v3", "strict_per_finger_v4"],
            "source_contact_contract_loaded": True,
            "evaluation_suite_loaded": True,
        },
    )


def _state_bank_path(clip: str) -> Path:
    return REPORT_ROOT / "p1" / f"rsi_state_bank_v2_{clip.removeprefix('hocap_')}.npz"


def _safe_bank_path(clip: str) -> Path:
    return REPORT_ROOT / "p1" / f"safe_bank_{clip.removeprefix('hocap_')}.npz"


def _diagnostic_path(clip: str) -> Path:
    return REPORT_ROOT / "p1" / f"gravity_short_diagnostics_{clip.removeprefix('hocap_')}.json"


def build_p1_banks() -> None:
    for clip in CLIP_TO_SEQUENCE:
        bank = build_contact_ready_state_bank(
            reference_path=REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz",
            source_contact_evidence_path=SOURCE_CONTACT_ROOT
            / clip
            / "source_contact_evidence_runtime.npz",
        )
        save_state_bank(_state_bank_path(clip), bank)
        _write_json(
            REPORT_ROOT / "p1" / f"rsi_state_summary_{clip.removeprefix('hocap_')}.json",
            summarize_state_bank(bank),
        )


def _p1_candidate_count(clip: str) -> int:
    with np.load(_state_bank_path(clip), allow_pickle=False) as archive:
        classes = np.asarray(archive["semantic_class"]).astype("U24")
    candidates = {
        "NEAR_CONTACT",
        "CONTACT_READY",
        "PERSISTENT_CONTACT",
        "MANIPULATION",
        "TERMINAL_HOLD",
    }
    return int(np.count_nonzero(np.isin(classes, tuple(candidates))))


def run_p1_diagnostics() -> None:
    """Launch one bounded fresh Isaac process per batch and merge its receipts."""

    gate = _config_path("stage16_p3_entry_gate_v1.yaml")
    worker = REPO_ROOT / "scripts/rl/isaaclab/diagnose_stage16_contact_ready_rsi_v2.py"
    batch_size = 16
    for clip in CLIP_TO_SEQUENCE:
        candidates = _p1_candidate_count(clip)
        batch_count = (candidates + batch_size - 1) // batch_size
        rows: list[dict[str, object]] = []
        writes = {"object_rollout_state_writes": 0, "wrist_root_state_writes_during_step": 0}
        for batch_index in range(batch_count):
            part = REPORT_ROOT / "p1/physx_batches" / f"{clip}_batch_{batch_index:02d}.json"
            command = [
                "conda",
                "run",
                "-n",
                "toporetarget-isaaclab",
                "python",
                str(worker),
                "--accept-eula",
                "--clip",
                clip,
                "--state-bank",
                str(_state_bank_path(clip)),
                "--reference-root",
                str(REFERENCE_ROOT),
                "--entry-gate",
                str(gate),
                "--output",
                str(part),
                "--batch-index",
                str(batch_index),
                "--state-groups-per-batch",
                str(batch_size),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            receipt = _read_json(part)
            if receipt.get("status") != "COMPLETE":
                raise RuntimeError(f"P1_RSI_V2_BATCH_INCOMPLETE:{clip}:{batch_index}")
            batch_rows = receipt.get("rows")
            if not isinstance(batch_rows, list) or not all(
                isinstance(row, dict) for row in batch_rows
            ):
                raise ValueError(f"P1_RSI_V2_BATCH_ROWS_INVALID:{clip}:{batch_index}")
            rows.extend(batch_rows)
            receipt_writes = receipt.get("reset_only_state_writes")
            if not isinstance(receipt_writes, dict):
                raise ValueError(f"P1_RSI_V2_BATCH_WRITES_INVALID:{clip}:{batch_index}")
            for key in writes:
                writes[key] += int(receipt_writes[key])
        if len(rows) != candidates * 4:
            raise RuntimeError(f"P1_RSI_V2_ROW_COUNT_INVALID:{clip}:{len(rows)}:{candidates * 4}")
        _write_json(
            _diagnostic_path(clip),
            {
                "schema_version": "Stage16ContactReadyRSIV2GravityDiagnosticV1",
                "status": "COMPLETE",
                "clip": clip,
                "target_gravity_world_mps2": [0.0, 0.0, -9.81],
                "friction": "current_stage16d_nominal",
                "candidate_state_count": candidates,
                "replicas_per_state": 4,
                "control_steps": 20,
                "controller": "zero_policy_residual_plus_reference_following",
                "ppo_training": False,
                "guidance": 0,
                "hidden_support": False,
                "reset_only_state_writes": writes,
                "batch_count": batch_count,
                "rows": rows,
            },
        )


def merge_p1_diagnostics() -> None:
    """Merge already-complete independent PhysX receipts without rerunning them."""

    batch_size = 16
    for clip in CLIP_TO_SEQUENCE:
        candidates = _p1_candidate_count(clip)
        batch_count = (candidates + batch_size - 1) // batch_size
        rows: list[dict[str, object]] = []
        writes = {"object_rollout_state_writes": 0, "wrist_root_state_writes_during_step": 0}
        for batch_index in range(batch_count):
            part = REPORT_ROOT / "p1/physx_batches" / f"{clip}_batch_{batch_index:02d}.json"
            if not part.is_file():
                raise FileNotFoundError(f"P1_RSI_V2_BATCH_RECEIPT_MISSING:{clip}:{batch_index}")
            receipt = _read_json(part)
            if receipt.get("status") != "COMPLETE":
                raise RuntimeError(f"P1_RSI_V2_BATCH_INCOMPLETE:{clip}:{batch_index}")
            if receipt.get("clip") != clip or receipt.get("batch_index") != batch_index:
                raise ValueError(f"P1_RSI_V2_BATCH_IDENTITY_INVALID:{clip}:{batch_index}")
            batch_rows = receipt.get("rows")
            if not isinstance(batch_rows, list) or not all(
                isinstance(row, dict) for row in batch_rows
            ):
                raise ValueError(f"P1_RSI_V2_BATCH_ROWS_INVALID:{clip}:{batch_index}")
            rows.extend(batch_rows)
            receipt_writes = receipt.get("reset_only_state_writes")
            if not isinstance(receipt_writes, dict):
                raise ValueError(f"P1_RSI_V2_BATCH_WRITES_INVALID:{clip}:{batch_index}")
            for key in writes:
                writes[key] += int(receipt_writes[key])
        if len(rows) != candidates * 4:
            raise RuntimeError(f"P1_RSI_V2_ROW_COUNT_INVALID:{clip}:{len(rows)}:{candidates * 4}")
        _write_json(
            _diagnostic_path(clip),
            {
                "schema_version": "Stage16ContactReadyRSIV2GravityDiagnosticV1",
                "status": "COMPLETE",
                "clip": clip,
                "target_gravity_world_mps2": [0.0, 0.0, -9.81],
                "friction": "current_stage16d_nominal",
                "candidate_state_count": candidates,
                "replicas_per_state": 4,
                "control_steps": 20,
                "controller": "zero_policy_residual_plus_reference_following",
                "ppo_training": False,
                "guidance": 0,
                "hidden_support": False,
                "reset_only_state_writes": writes,
                "batch_count": batch_count,
                "rows": rows,
            },
        )


def _save_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("P1_DIAGNOSTIC_PARQUET_REQUIRES_PYARROW") from exc
    if not rows:
        raise ValueError("P1_DIAGNOSTIC_PARQUET_ROWS_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _plot_p1_semantics(
    clip: str, state_bank: Mapping[str, np.ndarray], safe_bank: Mapping[str, np.ndarray]
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "NOT_AVAILABLE_MATPLOTLIB_MISSING"
    runtime = np.asarray(state_bank["runtime_index"], dtype=np.int64)
    semantic = np.asarray(state_bank["semantic_class"]).astype("U24")
    expected = np.asarray(state_bank["source_expected_contact"], dtype=bool)
    safe_label = dict(
        zip(
            np.asarray(safe_bank["all_runtime_index"], dtype=np.int64),
            np.asarray(safe_bank["all_gravity_label"]).astype("U16"),
            strict=True,
        )
    )
    class_index = {member.value: index for index, member in enumerate(RSIStateSemanticClass)}
    values = np.asarray([class_index[item] for item in semantic], dtype=np.int64)
    colors = np.asarray([safe_label[int(index)] == "GRAVITY_SAFE" for index in runtime], dtype=bool)
    path = REPORT_ROOT / "p1/screenshots" / f"{clip}_contact_ready_state_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].step(runtime, values, where="mid", color="black", linewidth=0.8)
    axes[0].scatter(runtime[colors], values[colors], c="tab:green", s=9, label="gravity safe")
    axes[0].set_yticks(list(class_index.values()), list(class_index))
    axes[0].set_ylabel("semantic class")
    axes[0].legend(loc="best")
    axes[1].step(runtime, expected.astype(int), where="mid", color="tab:blue")
    axes[1].set_ylabel("source contact")
    axes[1].set_xlabel("runtime reference index")
    axes[1].set_yticks((0, 1), ("no", "expected"))
    figure.suptitle(f"{clip}: source contact evidence and full-gravity P1 labels")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def finalize_p1() -> None:
    acceptance = load_p1_rsi_acceptance_contract(_config_path("stage16_p3_entry_gate_v1.yaml"))
    for clip in CLIP_TO_SEQUENCE:
        diagnostic = _read_json(_diagnostic_path(clip))
        if diagnostic.get("status") != "COMPLETE":
            raise RuntimeError(f"P1_RSI_V2_DIAGNOSTIC_INCOMPLETE:{clip}")
        rows = diagnostic.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"P1_RSI_V2_DIAGNOSTIC_ROWS_INVALID:{clip}")
        with np.load(_state_bank_path(clip), allow_pickle=False) as archive:
            state_bank = {name: np.asarray(archive[name]) for name in archive.files}
        safe_bank = build_safe_bank(
            state_bank=state_bank,
            diagnostic_rows=rows,
            acceptance=acceptance,
        )
        save_safe_bank(_safe_bank_path(clip), safe_bank)
        _save_parquet(
            REPORT_ROOT / "p1" / f"gravity_short_diagnostics_{clip.removeprefix('hocap_')}.parquet",
            rows,
        )
        counts = Counter(np.asarray(safe_bank["all_gravity_label"]).astype("U16"))
        safe_names = Counter(np.asarray(safe_bank["safe_bank"]).astype("U24"))
        initial_count = sum(safe_names[name] for name in INITIAL_P3_BANKS)
        status = "P1_RSI_V2_VALIDATED" if initial_count else "P1_RSI_V2_BLOCKED"
        visual = _plot_p1_semantics(clip, state_bank, safe_bank)
        _write_json(
            REPORT_ROOT / "p1" / f"rsi_v2_contract_{clip.removeprefix('hocap_')}.json",
            {
                "schema_version": "Stage16P1RSIAcceptanceResultV1",
                "clip": clip,
                "status": status,
                "gravity_label_counts": dict(sorted(counts.items())),
                "safe_bank_counts": dict(sorted(safe_names.items())),
                "initial_p3_safe_state_count": initial_count,
                "safe_bank_fraction": float(initial_count / len(state_bank["runtime_index"])),
                "automated_visual_audit": visual,
                "diagnostic_data_only": True,
            },
        )


def _plot_p2_support(clip: str, timeline: list[dict[str, object]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "NOT_AVAILABLE_MATPLOTLIB_MISSING"
    labels = sorted({str(row["classification"]) for row in timeline})
    indices = {label: index for index, label in enumerate(labels)}
    path = REPORT_ROOT / "p2/screenshots" / f"{clip}_support_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 2.8))
    for row in timeline:
        start = int(row["start_runtime_index"])
        stop = int(row["end_runtime_index_exclusive"])
        label = str(row["classification"])
        axis.broken_barh([(start, stop - start)], (indices[label] - 0.4, 0.8), label=label)
    axis.set_yticks(list(indices.values()), labels)
    axis.set_xlabel("runtime reference index")
    axis.set_title(f"{clip}: source-support evidence timeline (no invented support mesh)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def finalize_p2() -> None:
    all_decisions: dict[str, dict[str, object]] = {}
    contract = SourceSupportContractV1()
    _write_json(REPORT_ROOT / "p2/source_support_contract.json", contract.as_dict())
    for clip, sequence in CLIP_TO_SEQUENCE.items():
        discovery = discover_source_support_evidence(SOURCE_ROOT / sequence)
        _write_json(
            REPORT_ROOT / "p2" / f"support_evidence_{clip.removeprefix('hocap_')}.json",
            discovery,
        )
        with np.load(_state_bank_path(clip), allow_pickle=False) as archive:
            state_bank = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(_safe_bank_path(clip), allow_pickle=False) as archive:
            safe_bank = {name: np.asarray(archive[name]) for name in archive.files}
        label_map = dict(
            zip(
                np.asarray(safe_bank["all_runtime_index"], dtype=np.int64),
                np.asarray(safe_bank["all_gravity_label"]).astype("U16"),
                strict=True,
            )
        )
        has_source_asset = bool(discovery["source_scene_geometry_candidates"])
        timeline = build_support_timeline(
            runtime_index=np.asarray(state_bank["runtime_index"], dtype=np.int64),
            source_expected_contact=np.asarray(state_bank["source_expected_contact"], dtype=bool),
            gravity_label_by_state=label_map,
            source_support_available=has_source_asset,
        )
        _save_parquet(
            REPORT_ROOT / "p2" / f"support_timeline_{clip.removeprefix('hocap_')}.parquet", timeline
        )
        rows = _read_json(_diagnostic_path(clip))["rows"]
        if not isinstance(rows, list):
            raise ValueError("P2_SUPPORT_DIAGNOSTIC_ROWS_INVALID")
        support_diagnostic = support_diagnostic_summary(rows)
        _write_json(
            REPORT_ROOT / "p2/support_diagnostics" / f"{clip}_support_diagnostic.json",
            support_diagnostic,
        )
        decision = decide_support_mode(
            support_timeline=timeline,
            safe_bank_names=np.asarray(safe_bank["safe_bank"]).astype("U24").tolist(),
            hidden_support=False,
        )
        decision["clip"] = clip
        decision["source_scene_asset_found"] = has_source_asset
        decision["support_visual_audit"] = _plot_p2_support(clip, timeline)
        all_decisions[clip] = decision
    _write_json(
        REPORT_ROOT / "p2/support_decision.json",
        {"schema_version": "Stage16P2SupportDecisionV1", "clips": all_decisions},
    )


def _p3_decision() -> dict[str, object]:
    p1 = {
        clip: _read_json(REPORT_ROOT / "p1" / f"rsi_v2_contract_{clip.removeprefix('hocap_')}.json")
        for clip in CLIP_TO_SEQUENCE
    }
    p2 = _read_json(REPORT_ROOT / "p2/support_decision.json")["clips"]
    if not isinstance(p2, dict):
        raise ValueError("P3_SUPPORT_DECISION_INVALID")
    p0 = _read_json(REPORT_ROOT / "p0/config_smoke.json")
    provenance = _read_json(REPORT_ROOT / "p0/parent_provenance.json")
    p1_pass = all(item.get("status") == "P1_RSI_V2_VALIDATED" for item in p1.values())
    support_pass = all(
        item.get("p3_gate_classification") == "CONTACT_READY_ONLY_VALIDATED" for item in p2.values()
    )
    causality = True
    controller = True
    for clip in CLIP_TO_SEQUENCE:
        diagnostic = _read_json(_diagnostic_path(clip))
        rows = diagnostic.get("rows")
        if not isinstance(rows, list):
            raise ValueError("P3_DIAGNOSTIC_ROWS_INVALID")
        with np.load(_safe_bank_path(clip), allow_pickle=False) as archive:
            safe_indices = np.asarray(archive["runtime_index"], dtype=np.int64)
            safe_names = np.asarray(archive["safe_bank"]).astype("U24")
        initial_indices = set(
            safe_indices[np.isin(safe_names, INITIAL_P3_BANKS)].astype(np.int64).tolist()
        )
        allowed_rows = [row for row in rows if int(row["runtime_index"]) in initial_indices]
        if not allowed_rows:
            controller = False
        causality &= all(
            row.get("guidance") == 0
            and row.get("hidden_support") is False
            and row.get("rollout_object_state_writes") == 0
            and row.get("rollout_wrist_root_state_writes") == 0
            for row in rows
        )
        controller &= not any(
            bool(row.get("joint_limit_failure"))
            or bool(row.get("nonfinite"))
            or bool(row.get("catastrophic_failure"))
            for row in allowed_rows
        )
    geometry = False
    gates = {
        "G0_provenance": (
            p0.get("status") == "PASS" and provenance.get("parent_hashes_valid") is True
        ),
        "G1_rsi_v2": p1_pass,
        "G2_support": support_pass,
        "G3_geometry": geometry,
        "G4_controller_actuator": controller,
        "G5_causality": causality,
    }
    if all(gates.values()):
        status = "P3_READY_WITH_CONSTRAINTS"
        reason = (
            "No recoverable source support asset is locally available; only "
            "contact-ready safe resets are authorized."
        )
    else:
        status = "P3_BLOCKED_TECHNICAL"
        reason = (
            "One or more frozen P3 entry gates failed; in particular, the bounded P1 "
            "diagnostic does not replace the required current absolute geometry gate."
        )
    allowed = {
        clip: item.get("p3_allowed_reset_banks", [])
        for clip, item in p2.items()
        if isinstance(item, dict)
    }
    return {
        "schema_version": "Stage16P3EntryDecisionV1",
        "status": status,
        "reason": reason,
        "gates": gates,
        "gate_evidence": {
            "G3_geometry": "NOT_RUN: P1 is a 20-control-step contact-ready diagnostic; do not "
            "inherit zero-g geometry evidence as a full-gravity P3 pass."
        },
        "p3_allowed_reset_banks": allowed,
        "frame_zero_full_gravity_authorized": {clip: False for clip in CLIP_TO_SEQUENCE},
        "external_guidance": False,
        "support_mode": "none_for_contact_ready_training",
        "p3_training_started": False,
        "p4_human_decision_deferred": status == "P3_READY_WITH_CONSTRAINTS",
        "new_simulation_data": "DIAGNOSTIC_DATA_ONLY",
    }


def finalize() -> None:
    decision = _p3_decision()
    _write_json(REPORT_ROOT / "p3_entry_decision.json", decision)
    lines = [
        "# Stage 16 Physical P0–P2 Handoff",
        "",
        f"P3 entry decision: `{decision['status']}`.",
        "",
        "This run creates diagnostic evidence only. It does not start PPO, gravity curriculum, "
        "or friction curriculum.",
        "",
        "## Gate summary",
        "",
    ]
    gate_lines = [
        f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in decision["gates"].items()
    ]
    lines.extend(gate_lines)
    lines.extend(
        [
            "",
            "## Constraints",
            "",
            "- Frame-zero full-gravity reproduction is not authorized.",
            "- Initial P3 resets must be selected only from CONTACT_READY_SAFE, "
            "PERSISTENT_SAFE, or MANIPULATION_SAFE.",
            "- No external guidance, rollout object writes, rollout wrist-root writes, "
            "or hidden support are authorized.",
        ]
    )
    (REPORT_ROOT / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        REPORT_ROOT / "final_summary.json",
        {
            "schema_version": "Stage16PhysicalP0P2FinalSummaryV1",
            "p0": "COMPLETE",
            "p1": "VALIDATED" if decision["gates"]["G1_rsi_v2"] else "BLOCKED",
            "p2": "READY_WITH_CONSTRAINTS" if decision["gates"]["G2_support"] else "BLOCKED",
            "p3_entry_decision": decision["status"],
            "diagnostic_data_only": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "p0",
            "build-p1-banks",
            "run-p1-diagnostics",
            "merge-p1-diagnostics",
            "finalize-p1",
            "finalize-p2",
            "finalize",
        ),
    )
    args = parser.parse_args()
    if args.command == "p0":
        run_p0()
    elif args.command == "build-p1-banks":
        build_p1_banks()
    elif args.command == "run-p1-diagnostics":
        run_p1_diagnostics()
    elif args.command == "merge-p1-diagnostics":
        merge_p1_diagnostics()
    elif args.command == "finalize-p1":
        finalize_p1()
    elif args.command == "finalize-p2":
        finalize_p2()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
