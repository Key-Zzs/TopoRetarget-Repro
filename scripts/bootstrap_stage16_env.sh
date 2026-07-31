#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="toporetarget-rl"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; no sudo installation is attempted" >&2
  exit 2
fi

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  echo "environment already exists: $env_name"
else
  conda env create -f "$repo_root/environment.stage16.yml"
fi

conda run -n "$env_name" python -c \
  'import mujoco, numpy, scipy, torch; print("stage16 imports: mujoco", mujoco.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "torch", torch.__version__)'
conda run -n "$env_name" python "$repo_root/scripts/rl/validate_reference_clips.py" \
  "$repo_root/.local/stage16_reference_tracking_ppo/references/hocap_170105.stage16.npz"
echo "Stage-16 environment smoke complete: $env_name"
