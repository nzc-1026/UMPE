#!/usr/bin/env bash
# Generic staged-training entry: CONFIG and WORK come from the caller.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate umpe
cd /home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR_E4E5
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export PORT="${PORT:-28656}"
: "${CONFIG:?CONFIG is required}"
: "${WORK:?WORK is required}"
mkdir -p "$WORK"
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
if [ "${GPUS:-2}" = "1" ]; then
  python tools/train.py "$CONFIG" --work-dir "$WORK" --gpus 1 --launcher none --seed 0 --deterministic "${RESUME_ARGS[@]}"
else
  bash tools/dist_train.sh "$CONFIG" "${GPUS:-2}" --work-dir "$WORK" "${RESUME_ARGS[@]}"
fi
