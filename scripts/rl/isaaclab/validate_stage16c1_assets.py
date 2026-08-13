#!/usr/bin/env python3
"""Validate the static Stage 16-C.1 source, mapping, and generated asset contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (  # noqa: E402
    load_asset_migration_config,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
    )
    args = parser.parse_args()
    cfg = load_asset_migration_config(args.config)
    joints = cfg.validate(REPO_ROOT)
    payload = {
        "status": "STAGE16C1_STATIC_ASSET_CONTRACT_VALIDATED",
        "wuji": {
            "root": cfg.wuji.root_link,
            "fixed_base": cfg.wuji.fixed_base,
            "actuated_joint_count": len(joints),
            "mapped_joint_count": len(cfg.wuji.semantic_mapping),
            "tracked_link_count": len(cfg.wuji.tracked_links),
            "joints": [joint.__dict__ for joint in joints],
        },
        "objects": [
            {**item.__dict__, "source_file": str(item.source_file)} for item in cfg.objects
        ],
        "scope": {"c2": False, "ppo": False, "direct_rl_env": False},
    }
    write_json(REPO_ROOT / cfg.report_root / "static_contract_validation.json", payload)
    print(json.dumps(payload, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
