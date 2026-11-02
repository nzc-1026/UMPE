#!/usr/bin/env bash
# Queue for R24. User-approved shared GPUs: 0,1,3,4.
set -u
ROOT=/home/lbb/Documents/space/umpe_hdmap/UMPE-main/MapQR_E4E5
RES=/home/lbb/Documents/space/umpe_hdmap/results_20261029/R24
STATUS=$RES/queue_status.log
TRAINLOG=$RES/queue_train.log
GPUS="0,1,3,4"
mkdir -p "$RES"

log() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

free_enough() {
  for id in 0 1 3 4; do
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

run_stage() {
  local name=$1 script=$2 work=$3
  local attempt=1
  while [ "$attempt" -le 4 ]; do
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
      RESUME_FROM="$resume" CUDA_VISIBLE_DEVICES="$GPUS" GPUS=4 \
        bash "$ROOT/$script" >> "$TRAINLOG" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPUS" GPUS=4 \
        bash "$ROOT/$script" >> "$TRAINLOG" 2>&1
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

log "R24 queue start GPUs=$GPUS"
while ! free_enough; do
  log "waiting for enough free memory on $GPUS"
  sleep 300
done

if ! run_stage R24 train_r24.sh "$RES/work_dirs"; then
  log "R24 aborted after retries"
  exit 1
fi
log "R24 queue finished"
