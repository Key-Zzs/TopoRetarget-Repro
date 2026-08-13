#!/usr/bin/env python3
"""Verify Stage16-D zero-g closeout compatibility and write an ignored receipt.

This is deliberately a loader-level closeout: it neither trains PPO nor
rewrites historical checkpoints, traces, datasets, or qualification results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.geometry_audit.simulation_trace_replay import (  # noqa: E402
    load_stage16d_simulation_trace,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint  # noqa: E402
from toporetarget.rl.reference_tracking.contact_reward_mode import (  # noqa: E402
    ContactRewardMode,
    Stage16DContactRewardConfigV1,
    validate_frozen_contact_contract,
)

DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16d_causal_zero_g_closeout"
V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
MILESTONE_CONTRACT_PATH = REPO_ROOT / "configs/rl/stage16/stage16d_causal_zero_g_milestone.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"STAGE16D_CLOSEOUT_INPUT_MISSING:{path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_require_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"STAGE16D_CLOSEOUT_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _milestone_contract() -> dict[str, object]:
    """Load the durable, run-log-free milestone contract exactly as tracked."""

    loaded = yaml.safe_load(_require_file(MILESTONE_CONTRACT_PATH).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("STAGE16D_CLOSEOUT_MILESTONE_CONTRACT_OBJECT_REQUIRED")
    expected: dict[str, object] = {
        "schema_version": "Stage16DCausalZeroGravityMilestoneV1",
        "milestone": {"name": "stage16d_causal_zero_gravity", "version": 1},
        "physics": {
            "causal": True,
            "gravity_mode": "zero",
            "support": "absent",
            "external_guidance": False,
            "rollout_object_state_write": False,
            "rollout_wrist_root_write": False,
        },
        "reference": {"kinematics": "v2"},
        "action": {"ppo26d": True},
        "evaluation": {"suite": "v2"},
        "contact_reward": {
            "default": ContactRewardMode.AGGREGATE_V3.value,
            "available": [item.value for item in ContactRewardMode],
        },
        "method_status": {
            "aggregate_v3": "stable_baseline",
            "strict_per_finger_v4": "experimental_partial",
        },
    }
    if loaded != expected:
        raise ValueError("STAGE16D_CLOSEOUT_MILESTONE_CONTRACT_MISMATCH")
    return expected


def _config_audit() -> dict[str, object]:
    path = REPO_ROOT / "configs/rl/stage16/stage16d_ppo26d_reward.yaml"
    payload = yaml.safe_load(_require_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("STAGE16D_CLOSEOUT_CONTACT_CONFIG_OBJECT_REQUIRED")
    config = Stage16DContactRewardConfigV1.from_mapping(payload)
    reward = payload.get("reward")
    contact = reward.get("contact") if isinstance(reward, dict) else None
    if not isinstance(contact, dict) or contact.get("available") != [
        "aggregate_v3",
        "strict_per_finger_v4",
    ]:
        raise ValueError("STAGE16D_CLOSEOUT_CONTACT_CONFIG_AVAILABLE_MODES_MISSING")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "new_config_default": config.mode.value,
        "available_modes": [item.value for item in ContactRewardMode],
        "invalid_mode_policy": "fail_fast_no_hidden_fallback",
    }


def _contract(mode: ContactRewardMode, path: Path) -> dict[str, object]:
    payload = _load_json(path)
    parameters = validate_frozen_contact_contract(mode, payload)
    return {
        "mode": mode.value,
        "path": str(path),
        "sha256": _sha256(path),
        "status": payload["status"],
        "frozen_parameter_names": sorted(parameters),
    }


def _checkpoint(name: str, path: Path, expected_schema: str) -> dict[str, object]:
    payload = load_checkpoint(_require_file(path), map_location="cpu")
    schema = payload.get("schema_version")
    if schema != expected_schema:
        raise ValueError(
            f"STAGE16D_CLOSEOUT_CHECKPOINT_SCHEMA_MISMATCH:{name}:{schema}:{expected_schema}"
        )
    actor = payload.get("actor_critic")
    normalizer = payload.get("observation_normalization")
    if not isinstance(actor, dict) or not actor:
        raise ValueError(f"STAGE16D_CLOSEOUT_ACTOR_MISSING:{name}")
    if not isinstance(normalizer, dict) or not normalizer:
        raise ValueError(f"STAGE16D_CLOSEOUT_NORMALIZER_MISSING:{name}")
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": schema,
        "actor_key_count": len(actor),
        "normalizer_key_count": len(normalizer),
        "load": "PASS",
        "deterministic_inference_prerequisite": "actor_and_normalizer_loaded",
    }


def _trace(name: str, path: Path) -> dict[str, object]:
    loaded = load_stage16d_simulation_trace(
        _require_file(path), expected_body_names=HAND_COLLISION_BODY_NAMES
    )
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "frame_count": loaded.frame_count,
        "trace_kind": loaded.trace_kind,
        "qualification_status": loaded.qualification_status,
        "replay_loader": "PASS",
    }


def _manifest(
    name: str,
    path: Path,
    *,
    expected_mode: str,
    expected_schema: str,
    expected_status: str,
) -> dict[str, object]:
    """Validate manifest provenance and read one existing Zarr episode."""

    payload = _load_json(path)
    encoded = json.dumps(payload, sort_keys=True)
    if expected_mode not in encoded:
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_PROVENANCE_MISSING:{name}")
    if payload.get("schema_version") != expected_schema or payload.get("status") != expected_status:
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_SCHEMA_MISMATCH:{name}")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_FILES_MISSING:{name}")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_FILE_ENTRY_INVALID:{name}")
        if _sha256(_require_file(path.parent / relative)) != expected_hash:
            raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_FILE_HASH_MISMATCH:{name}:{relative}")
    episode_count = payload.get("episode_count")
    if not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_EPISODE_COUNT_INVALID:{name}")
    import zarr

    rollout_root = zarr.open_group(str(path.parent / "rollouts.zarr"), mode="r")
    episodes = rollout_root["episodes"]
    episode_names = sorted(episodes.group_keys())
    if len(episode_names) != episode_count:
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_EPISODE_GROUP_COUNT_MISMATCH:{name}")
    sample = episodes[episode_names[0]]
    pose = np.asarray(sample["object"]["object_pose"][0])
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_SAMPLE_INVALID:{name}")
    if (
        sample.attrs.get("causal_physics") is not True
        or sample.attrs.get("external_guidance") is not False
    ):
        raise ValueError(f"STAGE16D_CLOSEOUT_SIM_DATA_CAUSAL_SAMPLE_INVALID:{name}")
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "top_level_keys": sorted(payload),
        "schema_and_hash_validation": "PASS",
        "sample_episode": episode_names[0],
        "sample_object_pose_shape": list(pose.shape),
        "sample_episode_read": "PASS",
    }


def _smoke(name: str, path: Path, expected_status: str) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("status") != expected_status or payload.get("finite") is not True:
        raise ValueError(f"STAGE16D_CLOSEOUT_RUNTIME_SMOKE_FAILED:{name}")
    reward = payload.get("environment", {}).get("ppo26d", {}).get("reward", {})
    if not isinstance(reward, dict):
        raise ValueError(f"STAGE16D_CLOSEOUT_RUNTIME_REWARD_REPORT_MISSING:{name}")
    return {
        "name": name,
        "path": str(path),
        "status": payload["status"],
        "finite": True,
        "reward_identifier": reward.get("identifier"),
        "object_rollout_state_writes": payload["environment"].get("object_rollout_state_writes"),
        "wrist_root_state_writes_during_step": payload["environment"].get(
            "wrist_root_state_writes_during_step"
        ),
    }


def _local_tracking_audit() -> dict[str, object]:
    result = subprocess.run(
        ["git", "ls-files", ".local/**"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line]
    if tracked:
        raise ValueError(f"STAGE16D_CLOSEOUT_LOCAL_FILES_TRACKED:{tracked}")
    return {"tracked_local_files": 0, "status": "PASS"}


def _command_output(*command: str) -> str:
    """Run a provenance query and return its non-empty UTF-8 output."""

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _github_json(endpoint: str) -> dict[str, Any]:
    """Read an authenticated GitHub API response without embedding credentials."""

    payload = _command_output("gh", "api", endpoint)
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError(f"STAGE16D_CLOSEOUT_GITHUB_JSON_OBJECT_REQUIRED:{endpoint}")
    return decoded


def _github_repository() -> str:
    repository = _command_output(
        "gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"
    )
    if not repository or "/" not in repository:
        raise ValueError("STAGE16D_CLOSEOUT_GITHUB_REPOSITORY_UNRESOLVED")
    return repository


def _github_ci_contract() -> dict[str, object]:
    workflow = REPO_ROOT / ".github/workflows/ci.yml"
    text = _require_file(workflow).read_text(encoding="utf-8")
    required = [
        "push:",
        "pull_request:",
        'pip install -e ".[dev,viz,cache,robot,geometry,retarget]"',
        "ruff check .",
        "ruff format --check .",
        "mypy src",
        "pytest -q",
        "python scripts/check_paper_fidelity.py",
        "git diff --check",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"STAGE16D_CLOSEOUT_CI_CONTRACT_MISSING:{missing}")
    return {
        "workflow_name": "ci",
        "workflow_path": str(workflow),
        "workflow_sha256": _sha256(workflow),
        "events": ["push", "pull_request"],
        "required_commands": required[2:],
        "validation": "PASS",
    }


def _git_commits(start_head: str) -> dict[str, object]:
    current_head = _command_output("git", "rev-parse", "HEAD")
    commit_lines = _command_output(
        "git",
        "log",
        "--format=%H%x1f%s",
        f"{start_head}..{current_head}",
    )
    commits = []
    for line in commit_lines.splitlines() if commit_lines else []:
        commit, subject = line.split("\x1f", maxsplit=1)
        commits.append({"sha": commit, "subject": subject})
    if not commits:
        raise ValueError("STAGE16D_CLOSEOUT_NO_DELIVERY_COMMITS")
    return {
        "start_head": start_head,
        "head": current_head,
        "commits": commits,
        "commit_count": len(commits),
    }


def _ci_run_receipt(run_id: str, *, expected_event: str, expected_head: str) -> dict[str, object]:
    repository = _github_repository()
    run = _github_json(f"repos/{repository}/actions/runs/{run_id}")
    if run.get("event") != expected_event:
        raise ValueError(f"STAGE16D_CLOSEOUT_CI_EVENT_MISMATCH:{run.get('event')}:{expected_event}")
    if run.get("head_sha") != expected_head:
        raise ValueError("STAGE16D_CLOSEOUT_CI_HEAD_MISMATCH")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError(
            "STAGE16D_CLOSEOUT_CI_NOT_GREEN:"
            f"status={run.get('status')}:conclusion={run.get('conclusion')}"
        )
    return {
        "repository": repository,
        "workflow_name": run.get("name"),
        "run_id": run.get("id"),
        "event": run.get("event"),
        "head_sha": run.get("head_sha"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "url": run.get("html_url"),
        "validation": "PASS",
    }


def _pr_receipt(pr_number: int, *, expected_head: str) -> dict[str, object]:
    repository = _github_repository()
    pull = _github_json(f"repos/{repository}/pulls/{pr_number}")
    head = pull.get("head")
    if not isinstance(head, dict) or head.get("sha") != expected_head:
        raise ValueError("STAGE16D_CLOSEOUT_PR_HEAD_MISMATCH")
    if pull.get("state") != "open" or pull.get("base", {}).get("ref") != "main":
        raise ValueError("STAGE16D_CLOSEOUT_PR_NOT_OPEN_AGAINST_MAIN")
    return {
        "number": pull.get("number"),
        "url": pull.get("html_url"),
        "state": pull.get("state"),
        "base": pull.get("base", {}).get("ref"),
        "head": head.get("ref"),
        "head_sha": head.get("sha"),
        "mergeable": pull.get("mergeable"),
        "mergeable_state": pull.get("mergeable_state"),
        "validation": "PASS",
    }


def _find_pull_request_ci_run(*, expected_head: str) -> str:
    repository = _github_repository()
    listing = _github_json(
        f"repos/{repository}/actions/runs?event=pull_request&head_sha={expected_head}&per_page=100"
    )
    runs = listing.get("workflow_runs")
    if not isinstance(runs, list):
        raise ValueError("STAGE16D_CLOSEOUT_PR_CI_RUNS_MISSING")
    candidates = [run for run in runs if isinstance(run, dict) and run.get("name") == "ci"]
    if not candidates:
        raise ValueError("STAGE16D_CLOSEOUT_PR_CI_RUN_NOT_FOUND")
    run_id = candidates[0].get("id")
    if not isinstance(run_id, int):
        raise ValueError("STAGE16D_CLOSEOUT_PR_CI_RUN_ID_INVALID")
    return str(run_id)


def _open_pull_requests(branch: str) -> list[dict[str, object]]:
    """Return the currently open PRs for one exact branch name."""

    listed = _command_output(
        "gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number,url"
    )
    parsed = json.loads(listed)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError(f"STAGE16D_CLOSEOUT_OPEN_PR_QUERY_INVALID:{branch}")
    return [{"number": item.get("number"), "url": item.get("url")} for item in parsed]


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def _downstream_refs(branch: str) -> list[str]:
    """List named branch refs whose tips contain ``branch`` without mutating refs."""

    refs = _command_output(
        "git",
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads",
        "refs/remotes/origin",
    ).splitlines()
    excluded = {branch, f"origin/{branch}", "origin/HEAD"}
    return sorted(ref for ref in refs if ref not in excluded and _is_ancestor(branch, ref))


def _branch_cleanup_audit(*, delivery_branch: str, pr_number: int) -> dict[str, object]:
    """Audit only the two named branches; this routine never deletes either."""

    _command_output("git", "fetch", "origin", "--prune")
    worktrees = _command_output("git", "worktree", "list", "--porcelain")
    audit: dict[str, object] = {
        "policy": "audit_only_no_branch_deletion",
        "main": _command_output("git", "rev-parse", "origin/main"),
        "worktrees": worktrees.splitlines(),
        "branches": {},
    }
    for branch in ("feature/reference-tracking-ppo", delivery_branch):
        local_exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            == 0
        )
        remote_exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            == 0
        )
        tip = _command_output("git", "rev-parse", branch) if local_exists else None
        remote_tip = (
            _command_output("git", "rev-parse", f"origin/{branch}") if remote_exists else None
        )
        ancestor_of_delivery = _is_ancestor(branch, delivery_branch) if local_exists else False
        ancestor_of_origin_main = _is_ancestor(branch, "origin/main") if local_exists else False
        unique_commits_against_delivery = (
            int(_command_output("git", "rev-list", "--count", f"{delivery_branch}..{branch}"))
            if local_exists
            else None
        )
        unique_counts_vs_main = (
            _command_output("git", "rev-list", "--left-right", "--count", f"origin/main...{branch}")
            if local_exists
            else None
        )
        if unique_counts_vs_main is not None:
            main_unique, branch_unique = (int(value) for value in unique_counts_vs_main.split())
        else:
            main_unique = branch_unique = None
        # The delivery branch has an open PR by construction.  The older PPO
        # branch can be deleted only after this PR merges if it is contained in
        # the delivery branch and is not occupied by a worktree.
        occupied = f"branch refs/heads/{branch}" in worktrees
        open_prs = _open_pull_requests(branch)
        can_delete_after_merge = (
            ancestor_of_delivery
            and unique_commits_against_delivery == 0
            and not occupied
            and not open_prs
        )
        decision = (
            "DELETE_AFTER_PR_MERGE"
            if branch == delivery_branch or can_delete_after_merge
            else "RETAIN"
        )
        audit["branches"][branch] = {
            "local_exists": local_exists,
            "remote_exists": remote_exists,
            "tip": tip,
            "remote_tip": remote_tip,
            "ancestor_of_delivery": ancestor_of_delivery,
            "ancestor_of_origin_main": ancestor_of_origin_main,
            "unique_commits_against_delivery": unique_commits_against_delivery,
            "origin_main_unique_commits": main_unique,
            "unique_commits_vs_main": branch_unique,
            "occupied_by_worktree": occupied,
            "open_prs": open_prs,
            "downstream_refs": _downstream_refs(branch) if local_exists else [],
            "decision": decision,
            "reason": (
                "active_delivery_pr_must_merge_first"
                if branch == delivery_branch
                else "contained_in_delivery_and_safe_only_after_its_pr_merges"
                if can_delete_after_merge
                else "has_open_pr_active_worktree_or_unique_commits"
            ),
        }
    return audit


def _handoff_text() -> str:
    return """# Stage16-D causal zero-g handoff

Stage16-D is closed as a causal, simplified zero-gravity/no-support baseline.
Aggregate V3 is the stable default; Strict Per-Finger V4 remains an explicit
experimental partial objective. Neither outcome is a full-gravity or
real-world validation.

The next post-merge work must begin from `main` on a new
`feature/ppo-physical` branch and follow this fixed order:

1. Contact-ready RSI V2
2. Support Feasibility
3. Gravity + Friction Curriculum
4. Full-gravity / zero-guidance qualification
5. Multi-Clip

External guidance/data-H2R remains a separately labelled assisted fallback.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--record-full-local-gate",
        action="store_true",
        help=(
            "Record the immediately preceding full local gate after ruff, format, mypy, "
            "pytest, paper-fidelity, and diff checks have all passed."
        ),
    )
    parser.add_argument(
        "--start-head",
        help="Initial branch SHA used to make the delivery commit receipt exact.",
    )
    parser.add_argument(
        "--pre-pr-ci-run-id",
        help="Green push-event GitHub Actions run ID for the pushed delivery head.",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        help="Open PR number after its pull_request-event ci run is green.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    v3_contract = _contract(ContactRewardMode.AGGREGATE_V3, V3_ROOT / "reward_v3_contract.json")
    v4_contract = _contract(
        ContactRewardMode.STRICT_PER_FINGER_V4, V4_ROOT / "strict_v4_contract.json"
    )
    checkpoints = [
        _checkpoint(
            "v3",
            V3_ROOT / "ppo_v3/hocap_170650/runs/formal_v3_4m/checkpoints/"
            "stage16d_reward_v3_samples_2129920.pt",
            "Stage16DRewardV3CheckpointV1",
        ),
        _checkpoint(
            "v4",
            V4_ROOT / "ppo_v4/hocap_170650/checkpoints/stage16d_strict_v4_samples_1064960.pt",
            "Stage16DStrictPerFingerV4CheckpointV1",
        ),
    ]
    traces = [
        _trace(
            "historical_v1",
            V3_ROOT / "hocap_170650/v1_baseline_dev/hocap_170650/v1_dev_baseline_trace.npz",
        ),
        _trace(
            "v3",
            V3_ROOT / "hocap_170650/formal/hocap_170650/v3_formal_selected_2129920_trace.npz",
        ),
        _trace(
            "v4",
            V4_ROOT
            / "hocap_170650/formal/hocap_170650/ppo_v4_formal_selected_1064960_trace_replica0.npz",
        ),
    ]
    manifests = [
        _manifest(
            "v3",
            REPO_ROOT / ".local/sim_data/stage16d_reward_v3/hocap_170650/"
            "v3_formal_selected_2129920/manifest.json",
            expected_mode="v3",
            expected_schema="Stage16DRewardV3FormalSimulationManifestV1",
            expected_status="STAGE16D_REWARD_V3_FORMAL_SIM_DATA_EXPORTED",
        ),
        _manifest(
            "v4",
            REPO_ROOT / ".local/sim_data/stage16d_strict_per_finger_v4/hocap_170650/"
            "v4_formal_selected_1064960/manifest.json",
            expected_mode="v4",
            expected_schema="Stage16DStrictPerFingerV4FormalSimulationManifestV1",
            expected_status="STAGE16D_STRICT_V4_FORMAL_SIM_DATA_EXPORTED",
        ),
    ]
    smokes = [
        _smoke(
            "aggregate_v3",
            output / "runtime_smoke/v3/ppo_v3/hocap_170650/resume_smoke.json",
            "REWARD_V3_321_STEP_REWARD_SMOKE_PASS",
        ),
        _smoke(
            "strict_per_finger_v4",
            output / "runtime_smoke/v4/ppo_v4/hocap_170650/resume_smoke.json",
            "STRICT_V4_321_STEP_REWARD_SMOKE_PASS",
        ),
    ]
    milestone = _milestone_contract()
    _write_json(output / "milestone_contract.json", milestone)
    _write_json(output / "config_interface_audit.json", _config_audit())
    _write_json(
        output / "legacy_compatibility.json",
        {"contracts": [v3_contract, v4_contract], "checkpoints": checkpoints, "traces": traces},
    )
    _write_json(output / "smoke_tests.json", {"runtime_smokes": smokes, "sim_data": manifests})
    local_gate = {
        "local_tracking": _local_tracking_audit(),
        "full_local_gate_recorded": args.record_full_local_gate,
        "commands": (
            [
                "conda run -n toporetarget-rl ruff check .",
                "conda run -n toporetarget-rl ruff format --check .",
                "conda run -n toporetarget-rl python -m mypy src",
                "conda run -n toporetarget-rl python -m pytest -q",
                "conda run -n toporetarget-rl python scripts/check_paper_fidelity.py",
                "git diff --check",
            ]
            if args.record_full_local_gate
            else []
        ),
    }
    _write_json(output / "local_test_receipt.json", local_gate)
    if args.start_head is not None:
        _write_json(output / "git_commits.json", _git_commits(args.start_head))
    if args.pre_pr_ci_run_id is not None:
        if args.start_head is None:
            raise ValueError("STAGE16D_CLOSEOUT_PRE_PR_CI_REQUIRES_START_HEAD")
        expected_head = _command_output("git", "rev-parse", "HEAD")
        _write_json(output / "github_ci_contract.json", _github_ci_contract())
        _write_json(
            output / "pre_pr_ci_receipt.json",
            _ci_run_receipt(
                args.pre_pr_ci_run_id,
                expected_event="push",
                expected_head=expected_head,
            ),
        )
    if args.pr_number is not None:
        if args.start_head is None or args.pre_pr_ci_run_id is None:
            raise ValueError("STAGE16D_CLOSEOUT_PR_RECEIPT_REQUIRES_PRE_PR_RECEIPT")
        expected_head = _command_output("git", "rev-parse", "HEAD")
        _write_json(
            output / "pr_receipt.json",
            _pr_receipt(args.pr_number, expected_head=expected_head),
        )
        pr_run_id = _find_pull_request_ci_run(expected_head=expected_head)
        _write_json(
            output / "pr_ci_receipt.json",
            _ci_run_receipt(
                pr_run_id,
                expected_event="pull_request",
                expected_head=expected_head,
            ),
        )
        delivery_branch = _command_output("git", "branch", "--show-current")
        _write_json(
            output / "branch_cleanup_audit.json",
            _branch_cleanup_audit(delivery_branch=delivery_branch, pr_number=args.pr_number),
        )
        (output / "handoff.md").write_text(_handoff_text(), encoding="utf-8")
    summary = {
        "schema_version": "Stage16DCausalZeroGravityCloseoutReceiptV1",
        "status": "STAGE16D_CAUSAL_ZERO_G_CLOSEOUT_COMPATIBILITY_PASS",
        "milestone_contract": str(output / "milestone_contract.json"),
        "config_interface": "PASS",
        "legacy_checkpoints": "PASS",
        "replay_compatibility": "PASS",
        "simulation_data": "PASS",
        "runtime_smokes": "PASS",
        "local_full_gate": "PASS" if args.record_full_local_gate else "NOT_RECORDED",
        "local_tracked_files": 0,
        "delivery_receipts": ("PASS" if args.pr_number is not None else "NOT_RECORDED"),
    }
    _write_json(output / "final_summary.json", summary)
    (output / "final_summary.md").write_text(
        "# Stage16-D Causal Zero-G Closeout\n\n"
        "Compatibility checks passed for the frozen V3 default, opt-in V4, existing "
        "checkpoints/traces/simulation data, and one-step real Isaac runtime smokes.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
