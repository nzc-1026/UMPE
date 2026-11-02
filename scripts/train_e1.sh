#!/usr/bin/env bash
# E1: vector + raster + satellite, residual 0.1 learnable, 12+4 epochs.
# Requires an A0 12-epoch raster baseline checkpoint; edit load_from in the config.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-umpe}"
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export PORT="${PORT:-28541}"
mkdir -p work_dirs/e1
bash tools/dist_train.sh \
  configs/mapqr/umpe_nocam_vector_ft.py \
  "${GPUS:-5}" \
  --work-dir work_dirs/e1
