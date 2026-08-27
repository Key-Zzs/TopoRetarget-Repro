from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from toporetarget.rl.ppo.checkpoint import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    path = REPO_ROOT / "scripts/rl/materialize_zero_residual_source.py"
    spec = importlib.util.spec_from_file_location("materialize_zero_residual_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialized_actor_is_identically_zero_and_bound_to_qualification(
    tmp_path: Path, monkeypatch
) -> None:
    script = _load_script()
    qualification = tmp_path / "qualification.json"
    checkpoint = tmp_path / "zero.pt"
    result = tmp_path / "result.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "SourceControllerQualificationV2",
                "mode": "ZERO_RESIDUAL_DETERMINISTIC",
                "source_controller_executability_v2": "PASS",
                "clip_id": "hocap_test",
                "episodes": 10,
                "runtime_contract": {
                    "source_controller_admission_v2": True,
                    "reference_bank": {
                        "frame_count": 321,
                        "hashes": {"hocap_test": "reference-hash"},
                    },
                    "joint_mapping": {"joint_position_target_limits_enforced": True},
                    "ppo26d": {
                        "fixed_clip": "hocap_test",
                        "active_clip_ids": ["hocap_test"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_zero_residual_source.py",
            "--qualification",
            str(qualification),
            "--checkpoint",
            str(checkpoint),
            "--result",
            str(result),
        ],
    )

    assert script.main() == 0
    payload = load_checkpoint(checkpoint, map_location="cpu")
    actor = {
        name: value for name, value in payload["actor_critic"].items() if name.startswith("actor.")
    }
    assert actor
    assert all(torch.count_nonzero(value) == 0 for value in actor.values())
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["optimizer_steps"] == 0
    assert report["training_samples"] == 0
    assert report["checkpoint_roundtrip_identical"] is True
