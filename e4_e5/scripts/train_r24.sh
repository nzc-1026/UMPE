#!/usr/bin/env bash
# R24: raster-only 24-epoch baseline for the Z-as-BEV-query architecture.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate umpe
cd /home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR_E4E5
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export PORT="${PORT:-28643}"
WORK=/home/lbb/Documents/space/umpe_hdmap/results_20261029/R24/work_dirs
mkdir -p "$WORK"
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
bash tools/dist_train.sh \
  projects/configs/mapqr/r24_raster_24ep.py \
  "${GPUS:-4}" --work-dir "$WORK" "${RESUME_ARGS[@]}"
