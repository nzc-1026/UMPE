# E4/E5: Z as the camera-BEV substitute

This folder contains the camera-free adaptation developed after E1–E3.  It is intended to be
reviewed together with the parent repository; the parent still holds the shared MapQR/MapTRv2
plugin and the E1–E3 configs.

## Motivation

The original MapQR uses a camera BEV `X` as the main BEV representation and query.  In this
project there is no camera, so the raster chain output `Z` takes over that role:

```text
satellite + rasterized SD
        -> StageFiLM raster encoder
        -> Z [B, 20000, 256]
        -> LayerNorm
        -> Z + bev_pos
        -> Z as the BEV query
        -> SD vector tokens as K/V
        -> gate or residual fusion
        -> MapTRv2 decoder
```

## Implemented variants

### R24: raster-only baseline

```text
Z + bev_pos -> decoder
```

- 24 epochs
- 4 GPUs x batch 5 = effective batch 20
- final mAP: **40.55%**
- best epoch 20: **41.71%**

### E5 gate

```text
Z_pos = Z + bev_pos
A_pos = CrossAttention(Z_pos, SD tokens) + bev_pos
gamma = sigmoid(MLP([Z_pos, A_pos]))      # initialized near 0.12
X = (1 - gamma) * Z_pos + gamma * A_pos
```

- Z can dominate if gamma stays small.
- 24 epochs, continued from the R24 checkpoint.
- best observed validation: **42.23%** at epoch 12.

### E5 residual

```text
Z_pos = Z + bev_pos
A = CrossAttention(Z_pos, SD tokens)
X = Z_pos + A
```

- The vector correction is always added at full scale.
- 24 epochs, continued from the R24 checkpoint.
- best observed validation: **41.90%** at epoch 16.
- later epochs overfit and the validation mAP drops, so the best checkpoint should be used.

## Current conclusion

- E3 (satellite + vector, learnable BEV) remains the best result: **50.24%**.
- Replacing the camera BEV with Z makes the raster prior the main representation; its ceiling is
  about 41–42% with the current StageFiLM encoder.
- Raster SD and vector SD are highly redundant; adding raster SD in E2 reduced mAP by 8.16 points
  relative to E3.

## Files

```text
modules/transformer.py             E4/E5 transformer: raster_query path, z_norm, bev_pos
modules/raster_query_fusion.py     RasterQueryFusion: gate, residual, attention output norm
modules/stagefilm_raster.py        E4/E5 copy with SourceDropout/device fixes
configs/r24_raster_24ep.py         R24 baseline
configs/e5_final_24ep.py           E5 gate, 24 epochs from R24
configs/e5_residual_24ep.py        E5 residual, 24 epochs from R24
configs/e5_rq24_g_24ep.py          gate config without staged loading
configs/e4_rq24_24ep.py            pure-query ablation config
configs/e5_rq24_g_stage*.py        earlier staged 4+8 design (kept for record)
configs/e4_rq24_stage*.py          earlier staged E4 design
configs/*frozen*.py                fixed-Z 24-epoch variants
scripts/*.sh                       launch scripts used on the GPU server
```

## Infrastructure note

The vector dataset loads a large SD-map pickle (about 167 s per process).  Four-rank DDP was
unstable on the shared server and intermittently raised `ncclInvalidUsage`.  The working setup uses
a single GPU with `--launcher none`, batch 10, and no NCCL.  The server-side `mmdet_train.py` also
contains a Gloo barrier before DDP for the multi-rank variant; see `docs/INFRA_NOTES.md`.

## Status (2026-11-02)

- R24 complete.
- E5 gate and E5 residual are being run to 24 epochs; validation is recorded in the server logs.
- No camera images and no HD-map prior are used.
