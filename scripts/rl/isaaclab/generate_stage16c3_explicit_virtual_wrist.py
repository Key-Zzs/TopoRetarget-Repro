#!/usr/bin/env python3
"""Generate the ignored explicit virtual 3P+3R wrist wrapper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / ".local/generated_assets/isaaclab/wuji_hand2_beta1_explicit_virtual_wrist"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("conservative", "nominal", "high_authority_bounded"),
        default="nominal",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        from toporetarget.rl.environments.isaaclab_backend.virtual_wrist_asset import (
            write_explicit_virtual_wrist_wrapper,
        )

        output_dir = args.output_dir.resolve()
        base_asset = (
            REPO_ROOT
            / ".local/generated_assets/isaaclab/wuji_hand2_beta1"
            / "configuration/wujihand2_physics.usd"
        ).resolve()
        manifest = write_explicit_virtual_wrist_wrapper(
            base_asset=base_asset,
            output_usda=output_dir / "wujihand2_explicit_virtual_wrist.usda",
            profile_identifier=args.profile,
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
