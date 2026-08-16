#!/usr/bin/env bash
# Run the remaining G5 jobs strictly serially; every job must materialize its
# exact 4,194,304-sample receipt before the next one starts.
set -euo pipefail

repo=/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro-guidance
python_bin=/home/deepcybo/miniconda3/envs/toporetarget-isaaclab/bin/python
train="$repo/scripts/rl/isaaclab/train_stage16d_ppo26d_object_twist.py"
reference="$repo/.local/frozen_baselines/reference_kinematics_v2"
inputs="$repo/.local/frozen_baselines/guidance_calibration_v1/reward_inputs"
root="$repo/.local/reports/stage16_guidance_g0_g5/g5"

wait_for_running_v3_170650() {
  local receipt="$root/ppo_v3/hocap_170650/runs/formal_v3_guided_4m/training_result_4194304.json"
  while [[ ! -f "$receipt" ]]; do
    if ! pgrep -f 'formal_v3_guided_4m.*hocap_170650|hocap_170650.*formal_v3_guided_4m' >/dev/null; then
      echo "G5_SEQUENCE_CURRENT_V3_170650_MISSING_RECEIPT" >&2
      return 1
    fi
    sleep 60
  done
}

require_receipt() {
  local receipt=$1
  "$python_bin" - "$receipt" <<'PY'
import json
import sys
path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if payload.get("checkpoint") is None:
    raise SystemExit("G5_SEQUENCE_CHECKPOINT_MISSING")
print(path)
PY
}

run_v4() {
  local clip=$1
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$repo/src" "$python_bin" "$train" \
    --accept-eula --clip "$clip" --output-root "$root" --run-label formal_v4_guided_4m \
    --reference-root "$reference" --contact-mode strict_per_finger_v4 \
    --target-reward-v4-samples 4194304 \
    --strict-v4-contract "$inputs/strict_v4_contract.json" \
    --strict-v4-source-mask-root "$inputs" \
    --resume-checkpoint "$repo/.local/frozen_baselines/guidance_calibration_v1/v4/$clip/checkpoint.pt" \
    --num-envs 1024 --guidance-profile-id strong_bounded
  require_receipt "$root/ppo_v4/$clip/runs/formal_v4_guided_4m/training_result_4194304.json"
}

wait_for_running_v3_170650
require_receipt "$root/ppo_v3/hocap_170105/runs/formal_v3_guided_4m/training_result_4194304.json"
require_receipt "$root/ppo_v3/hocap_170650/runs/formal_v3_guided_4m/training_result_4194304.json"
run_v4 hocap_170105
run_v4 hocap_170650
echo G5_SEQUENCE_ALL_FOUR_TRAINING_COMPLETE
