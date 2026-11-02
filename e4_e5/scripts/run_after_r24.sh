#!/usr/bin/env bash
# Wait for R24, apply a quality gate, then run E5-S and E4-S in order.
set -u
ROOT=/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR_E4E5
R24=/home/lbb/Documents/space/umpe_hdmap/results_20261029/R24
OUT=/home/lbb/Documents/space/umpe_hdmap/results_20261029/E5_E4_S
STATUS=$OUT/queue_status.log
TRAINLOG=$OUT/queue_train.log
GPUS="0"
mkdir -p "$OUT"

log() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

free_enough() {
  for id in 0; do
    read -r used total < <(nvidia-smi -i "$id" --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    [ -z "${used:-}" ] && return 1
    free=$((total - used))
    if [ "$free" -lt 20000 ]; then
      log "GPU $id free=${free}MB < 20000MB"
      return 1
    fi
  done
  return 0
}

wait_for_r24() {
  while true; do
    if grep -q "R24 aborted" "$R24/queue_status.log" 2>/dev/null; then
      log "R24 aborted; stop follow-on queue"
      return 1
    fi
    # Wait for the whole R24 run, not merely for epoch_24.pth: the
    # checkpoint is written before the final validation finishes, and
    # starting E5-S at that moment overlaps the two NCCL jobs.
    if grep -q "R24 queue finished" "$R24/queue_status.log" 2>/dev/null; then
      while pgrep -f 'tools/train.py.*r24_raster_24ep' >/dev/null 2>&1; do
        log "R24 process still exiting; wait"
        sleep 30
      done
      sleep 30
      local map
      map=$(grep -o 'NuscMap_chamfer/mAP: [0-9.]*' "$R24/queue_train.log" 2>/dev/null \
            | tail -1 | awk '{print $2}')
      log "R24 finished; final mAP=${map:-unknown}"
      if [ -z "$map" ]; then
        log "R24 mAP missing; abort"
        return 1
      fi
      if ! awk -v m="$map" 'BEGIN{exit !(m>=0.20)}'; then
        log "R24 mAP $map < 0.20; abort follow-on queue"
        return 1
      fi
      return 0
    fi
    sleep 300
  done
}

run_stage() {
  local name=$1 config=$2 work=$3
  local attempt=1
  while [ "$attempt" -le 3 ]; do
    local resume=""
    if [ "$attempt" -gt 1 ]; then
      resume=$(ls -t "$work"/*.pth 2>/dev/null | head -1)
      if [ -z "$resume" ]; then
        log "$name attempt=$attempt no checkpoint; abort"
        return 1
      fi
    fi
    log "$name attempt=$attempt resume=${resume:-none}"
    if [ -n "$resume" ]; then
      CONFIG="$config" WORK="$work" RESUME_FROM="$resume" GPUS=1 \
        CUDA_VISIBLE_DEVICES="$GPUS" bash "$ROOT/train_staged.sh" >> "$TRAINLOG" 2>&1
    else
      CONFIG="$config" WORK="$work" GPUS=1 \
        CUDA_VISIBLE_DEVICES="$GPUS" bash "$ROOT/train_staged.sh" >> "$TRAINLOG" 2>&1
    fi
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "$name done rc=0"
      return 0
    fi
    log "$name failed rc=$rc attempt=$attempt"
    attempt=$((attempt + 1))
    sleep 60
  done
  return 1
}

log "follow-on queue start; waiting for R24"
if ! wait_for_r24; then
  log "follow-on queue stopped before E5-S"
  exit 1
fi

while ! free_enough; do
  log "waiting for enough free memory on $GPUS"
  sleep 300
done
# Give the previous NCCL/CUDA context time to disappear; starting a new
# 4-rank job immediately after another one has caused ncclInvalidUsage.
log "cooldown before first stage"
sleep 180

if ! run_stage E5-S-stage1 projects/configs/mapqr/e5_rq24_g_stage1_4ep.py \
      "$OUT/E5_S/stage1/work_dirs"; then
  log "E5-S stage1 aborted; stop queue"
  exit 1
fi
if ! run_stage E5-S-stage2 projects/configs/mapqr/e5_rq24_g_stage2_8ep.py \
      "$OUT/E5_S/stage2/work_dirs"; then
  log "E5-S stage2 aborted; stop queue"
  exit 1
fi
log "E5-S finished; starting E4-S"

if ! run_stage E4-S-stage1 projects/configs/mapqr/e4_rq24_stage1_4ep.py \
      "$OUT/E4_S/stage1/work_dirs"; then
  log "E4-S stage1 aborted; stop queue"
  exit 1
fi
if ! run_stage E4-S-stage2 projects/configs/mapqr/e4_rq24_stage2_8ep.py \
      "$OUT/E4_S/stage2/work_dirs"; then
  log "E4-S stage2 aborted; stop queue"
  exit 1
fi
log "follow-on queue finished"
