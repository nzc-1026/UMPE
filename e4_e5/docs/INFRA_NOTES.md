# Infrastructure notes

## Large SD-map pickle

`inter_nuscenes_trainval_sdmap_new.pkl` is a 34,149-entry dictionary.  A single
`pickle.load` takes about 167 s in the project environment.  Starting four ranks at the
same time caused rank desynchronization during DDP initialization on the shared server.

## Working launch mode

The stable configuration is one GPU and no NCCL:

```bash
CONFIG=projects/configs/mapqr/e5_residual_24ep.py \
WORK=/path/to/work_dirs \
GPUS=1 CUDA_VISIBLE_DEVICES=0 \
bash train_staged.sh
```

`train_staged.sh` maps `GPUS=1` to:

```bash
python tools/train.py "$CONFIG" --work-dir "$WORK" \
  --gpus 1 --launcher none --seed 0 --deterministic
```

## Multi-rank barrier

For reference, the multi-rank variant inserted a Gloo CPU barrier before DDP:

```python
if dist.is_initialized():
    _sync_group = dist.new_group(backend='gloo')
    _sync_group.barrier()
model = MMDistributedDataParallel(
    model.cuda(),
    device_ids=[torch.cuda.current_device()],
    broadcast_buffers=False,
    find_unused_parameters=find_unused_parameters)
```

This synchronizes ranks after the slow dataset construction.  Four-rank runs still showed
intermittent NCCL failures on the shared machine, so the reported E5 runs use the single-GPU path.

## Checkpoints and logs

Checkpoints and raw logs are intentionally excluded from this repository.  They remain on the
GPU server under `results_20261029/` and `results_20261031/`.
