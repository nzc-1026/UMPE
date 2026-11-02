#!/usr/bin/env bash
# E2: vector + raster + satellite, residual 1.0 fixed, 24 epochs from scratch.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-umpe}"
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PORT="${PORT:-28542}"
mkdir -p work_dirs/e2
bash tools/dist_train.sh \
  configs/mapqr/umpe_nocam_vector_raster_24ep.py \
  "${GPUS:-4}" \
  --work-dir work_dirs/e2
