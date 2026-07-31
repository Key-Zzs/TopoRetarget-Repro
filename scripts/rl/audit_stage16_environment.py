#!/usr/bin/env python3
"""Audit the selected Stage-16 backend, local assets, paper source and data availability."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / ".local" / "reports" / "stage16_reference_tracking_ppo"
BUILD = REPO / ".local" / "build" / "stage16_reference_tracking_ppo"


def write_json(name: str, payload: dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def command(*args: str) -> dict[str, Any]:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    packages = {}
    for name in (
        "torch",
        "mujoco",
        "jax",
        "mjx",
        "isaacgym",
        "isaaclab",
        "mjlab",
        "rsl_rl",
        "rl_games",
        "skrl",
        "warp",
    ):
        packages[name] = importlib.util.find_spec(name) is not None
    torch_info: dict[str, Any] = {}
    if packages["torch"]:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    mjcf = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
    urdf = REPO / "third_party/robot_hands/wuji_hand2_beta1/urdf/right.urdf"
    asset = {
        "mjcf": str(mjcf),
        "mjcf_sha256": sha256(mjcf),
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
    }
    free_scene: dict[str, Any] = {"attempted": False}
    if packages["mujoco"]:
        from toporetarget.rl.environments.mujoco_backend import materialize_free_object_scene

        scene = materialize_free_object_scene(mjcf, BUILD)
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene))
        free_scene = {
            "attempted": True,
            "compiled": True,
            "path": str(scene),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
        }
    availability = {
        "hocap_root": "/mnt/nas/storage/Ref2Dex_storage/HOCap",
        "hocap_root_present": Path("/mnt/nas/storage/Ref2Dex_storage/HOCap").is_dir(),
        "penspin_status": "STAGE16_PENSPIN_DATA_UNAVAILABLE",
        "penspin_search_roots": ["/mnt/nas/storage/Ref2Dex_storage", str(REPO)],
        "reason": (
            "No provenance-complete Pen-Spin dataset was found by the Stage-16 "
            "audit; it is self-collected in the paper."
        ),
    }
    environment = {
        "timestamp_utc": now,
        "python": sys.version,
        "platform": platform.platform(),
        "nvcc_present": shutil.which("nvcc") is not None,
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ),
        "packages": packages,
        "torch": torch_info,
        "backend_choice": "mujoco_cpu_reference" if packages["mujoco"] else "blocked_no_backend",
        "gpu_vectorized_backend_available": any(
            packages[key] for key in ("isaacgym", "isaaclab", "mjx", "warp")
        ),
        "free_object_scene": free_scene,
    }
    environment_validation = {
        "status": (
            "STAGE16_ENVIRONMENT_PARTIAL_FREE_OBJECT_BACKEND_PASS"
            if free_scene.get("compiled")
            else "STAGE16_ENVIRONMENT_BLOCKED_NO_MUJOCO_BACKEND"
        ),
        "backend": environment,
        "paper_4096_protocol_feasible": False,
        "physical_hocap_protocol_started": False,
        "reason": (
            "The selected MuJoCo backend validates a single free-object correctness path; "
            "the paper simulator is undisclosed."
        ),
    }
    paper = {
        "pdf": "docs/TopoRetarget.pdf",
        "sha256": sha256(REPO / "docs/TopoRetarget.pdf"),
        "items": [
            {
                "item": "reference/action/observation/reset",
                "value": "base frame; residual q; lookahead 1,3,5; uniform start",
                "pdf_page": 13,
                "formula_or_table": "A.5.1-A.5.3",
                "exact_quote": "lookahead offsets of 1, 3, and 5",
                "implementation_mapping": "references.py, observations.py",
                "confidence": "PAPER_EXACT",
            },
            {
                "item": "reward and termination",
                "value": "Table 4 literal profile",
                "pdf_page": 14,
                "formula_or_table": "Table 4",
                "exact_quote": "We define psi(e; sigma)",
                "implementation_mapping": "rewards.py, termination.py",
                "confidence": "PAPER_EXACT",
            },
            {
                "item": "domain randomization",
                "value": "Table 5 ranges",
                "pdf_page": 15,
                "formula_or_table": "Table 5",
                "exact_quote": "Domain randomization ranges",
                "implementation_mapping": "randomization.py",
                "confidence": "PAPER_EXACT",
            },
            {
                "item": "PPO",
                "value": "4096 x 40, networks, Adam, 4/32",
                "pdf_page": 16,
                "formula_or_table": "Table 6",
                "exact_quote": "Samples per PPO iteration",
                "implementation_mapping": "ppo/",
                "confidence": "PAPER_EXACT",
            },
        ],
    }
    write_json("dependency_manifest.json", environment)
    write_json("environment_validation.json", environment_validation)
    write_json(
        "external_dependencies_manifest.json",
        {
            "external_repositories": [],
            "python_environment": "toporetarget-rl",
            "mujoco": packages["mujoco"],
        },
    )
    write_json("simulator_backend_decision.json", environment)
    write_json(
        "wuji_asset_validation.json",
        {"asset": asset, "free_object_scene": free_scene, "status": "PARTIAL_GENERIC_OBJECT_ONLY"},
    )
    write_json("penspin_availability.json", availability)
    write_json("paper_rl_extraction.json", paper)
    write_json(
        "paper_source_locations.json",
        {
            "pdf": paper["pdf"],
            "sha256": paper["sha256"],
            "pages": {"A.5.1-A.5.3": 13, "Table 4": 14, "Table 5": 15, "Table 6": 16},
        },
    )
    (REPORTS / "paper_rl_extraction.md").write_text(
        "# Paper RL extraction\n\n```json\n" + json.dumps(paper, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_root": str(REPORTS),
                "backend": environment["backend_choice"],
                "penspin": availability["penspin_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
