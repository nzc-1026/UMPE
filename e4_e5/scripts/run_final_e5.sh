#!/usr/bin/env bash
# Final E5: 24 epochs, single GPU, no NCCL.
set -u
ROOT=/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR_E4E5
OUT=/home/lbb/Documents/space/umpe_hdmap/results_20261031/E5_final
WORK=$OUT/work_dirs
mkdir -p "$WORK"

log() { echo "$(date '+%F %T') $*" >> "$OUT/status.log"; }

attempt=1
while [ "$attempt" -le 3 ]; do
  resume=""
  if [ "$attempt" -gt 1 ]; then
    resume=$(ls -t "$WORK"/*.pth 2>/dev/null | head -1)
    if [ -z "$resume" ]; then
      log "attempt=$attempt no checkpoint; abort"
      exit 1
    fi
  fi
  log "attempt=$attempt resume=${resume:-none}"
  if [ -n "$resume" ]; then
    CONFIG=projects/configs/mapqr/e5_final_24ep.py WORK="$WORK" \
      RESUME_FROM="$resume" GPUS=1 CUDA_VISIBLE_DEVICES=0 \
      bash "$ROOT/train_staged.sh" >> "$OUT/train.log" 2>&1
  else
    CONFIG=projects/configs/mapqr/e5_final_24ep.py WORK="$WORK" \
      GPUS=1 CUDA_VISIBLE_DEVICES=0 \
      bash "$ROOT/train_staged.sh" >> "$OUT/train.log" 2>&1
  fi
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "done rc=0"
    exit 0
  fi
  log "failed rc=$rc attempt=$attempt"
  attempt=$((attempt + 1))
  sleep 60
done
log "aborted"
exit 1
