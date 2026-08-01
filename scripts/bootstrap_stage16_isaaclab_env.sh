#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/configs/rl/stage16/isaaclab_platform.yaml"
environment_file="$repo_root/environment.stage16_isaaclab.yml"
env_name="toporetarget-isaaclab"
external_root="$repo_root/.local/external/IsaacLab"
report_root="$repo_root/.local/reports/stage16c_isaaclab_platform"
dry_run=0
verify_only=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_stage16_isaaclab_env.sh [options]

Options:
  --env-name NAME       Conda environment name (default: toporetarget-isaaclab)
  --external-root PATH  Ignored Isaac Lab source checkout
  --report-root PATH    Ignored qualification report directory
  --dry-run             Print the fixed commands without changing anything
  --verify-only         Do not install; run the platform verifier only
  -h, --help            Show this help

The script never uses sudo, deletes an environment, overwrites a non-empty
external directory, or accepts an NVIDIA EULA.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { echo "--env-name requires a value" >&2; exit 2; }
      env_name="$2"
      shift 2
      ;;
    --external-root)
      [[ $# -ge 2 ]] || { echo "--external-root requires a value" >&2; exit 2; }
      external_root="$2"
      shift 2
      ;;
    --report-root)
      [[ $# -ge 2 ]] || { echo "--report-root requires a value" >&2; exit 2; }
      report_root="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --verify-only)
      verify_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$env_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "PYTHON_VERSION_CONFLICT: unsafe environment name: $env_name" >&2
  exit 2
fi
if [[ -z "$external_root" || "$external_root" == "/" ]]; then
  echo "NETWORK_OR_ASSET_CACHE_FAILURE: external root must be a narrow path" >&2
  exit 2
fi
if [[ ! -f "$manifest" || ! -f "$environment_file" ]]; then
  echo "environment manifest is missing" >&2
  exit 2
fi
if [[ "$dry_run" -eq 0 ]] && ! command -v conda >/dev/null 2>&1; then
  echo "PYTHON_VERSION_CONFLICT: conda is required; sudo is not attempted" >&2
  exit 2
fi

yaml_scalar() {
  local key="$1"
  awk -v key="$key" '$1 == key ":" {print $2; exit}' "$manifest" | tr -d '"'
}

isaac_sim_version="$(yaml_scalar isaac_sim)"
isaac_lab_tag="$(yaml_scalar tag)"
isaac_lab_commit="$(yaml_scalar commit)"
torch_version="$(yaml_scalar torch)"
torchvision_version="$(yaml_scalar torchvision)"

if [[ -z "$isaac_sim_version" || -z "$isaac_lab_tag" || -z "$isaac_lab_commit" ]]; then
  echo "manifest version fields could not be parsed" >&2
  exit 2
fi

run() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$env_name"
}

verify_external_checkout() {
  if [[ ! -d "$external_root/.git" ]]; then
    echo "ISAAC_LAB_IMPORT_FAILURE: non-empty external root is not an Isaac Lab git checkout" >&2
    return 1
  fi
  local actual_url actual_head actual_status
  actual_url="$(git -C "$external_root" remote get-url origin)"
  actual_head="$(git -C "$external_root" rev-parse HEAD)"
  actual_status="$(git -C "$external_root" status --short)"
  if [[ "$actual_url" != "https://github.com/isaac-sim/IsaacLab.git" ]]; then
    echo "ISAAC_LAB_IMPORT_FAILURE: external checkout uses an unexpected remote" >&2
    return 1
  fi
  if [[ "$actual_head" != "$isaac_lab_commit" ]]; then
    echo "ISAAC_LAB_IMPORT_FAILURE: expected $isaac_lab_commit, found $actual_head" >&2
    return 1
  fi
  if [[ -n "$actual_status" ]]; then
    echo "ISAAC_LAB_IMPORT_FAILURE: external checkout is dirty; it is not modified automatically" >&2
    return 1
  fi
}

verify_command=(
  conda run -n "$env_name" python
  "$repo_root/scripts/verify_stage16_isaaclab_platform.py"
  --config "$manifest"
  --external-root "$external_root"
  --output-root "$report_root"
)

if [[ "$verify_only" -eq 1 ]]; then
  if [[ "$dry_run" -eq 1 ]]; then
    run "${verify_command[@]}"
    exit 0
  fi
  if ! env_exists; then
    echo "PYTHON_VERSION_CONFLICT: environment does not exist: $env_name" >&2
    exit 3
  fi
  run "${verify_command[@]}"
  exit 0
fi

if [[ "$dry_run" -eq 1 ]]; then
  run conda env create --name "$env_name" --file "$environment_file"
elif ! env_exists; then
  run conda env create --name "$env_name" --file "$environment_file"
else
  echo "environment already exists and will not be removed: $env_name"
fi

if [[ "$dry_run" -eq 0 ]]; then
  python_minor="$(conda run -n "$env_name" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$python_minor" != "3.11" ]]; then
    echo "PYTHON_VERSION_CONFLICT: expected 3.11, found $python_minor" >&2
    exit 3
  fi
fi

run conda run -n "$env_name" python -m pip install --upgrade \
  "torch==$torch_version" "torchvision==$torchvision_version" \
  --index-url https://download.pytorch.org/whl/cu128
run conda run -n "$env_name" python -m pip install \
  "isaacsim[all]==$isaac_sim_version" \
  --extra-index-url https://pypi.nvidia.com

if [[ -e "$external_root" ]]; then
  if [[ ! -d "$external_root" ]]; then
    echo "ISAAC_LAB_IMPORT_FAILURE: external root exists and is not a directory" >&2
    exit 4
  fi
  if [[ -n "$(find "$external_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ "$dry_run" -eq 0 ]]; then
      verify_external_checkout
    else
      echo "DRY_RUN: preserve and verify existing non-empty external root: $external_root"
    fi
  else
    run git clone --branch "$isaac_lab_tag" --depth 1 \
      https://github.com/isaac-sim/IsaacLab.git "$external_root"
  fi
else
  run git clone --branch "$isaac_lab_tag" --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "$external_root"
fi

if [[ "$dry_run" -eq 0 ]]; then
  verify_external_checkout
fi
run conda run -n "$env_name" python -m pip install --editable \
  "$external_root/source/isaaclab" \
  "$external_root/source/isaaclab_assets" \
  "$external_root/source/isaaclab_contrib" \
  "$external_root/source/isaaclab_tasks" \
  "$external_root/source/isaaclab_rl[none]" \
  "$external_root/source/isaaclab_mimic[none]"
run "${verify_command[@]}"
