# UMPE / MapQR: camera-free, HD-prior-free map generation

This is a cleaned code snapshot for three experiments on **OSM + satellite → nuScenes HD map**
without camera images and without an HD-map prior.

The model is based on **MapQR / MapTRv2** and ports the vector-prior modules from **UMPE**:
`MapGraphTransformer`, `CrossAttentionBEV`, and `DualVecCrossFusion`.

> This repository contains the **code** used by the experiments, not the datasets or checkpoints.
> It is intended for architecture review and reproduction, not as a standalone pip package.

## Three experiments

| ID | Inputs | Residual | Training | mAP |
|---|---|---|---|---|
| **E1** | vector SD + raster SD + satellite | `residual_scale=0.1`, learnable | 12-epoch baseline + 4-epoch fine-tune (vector sees 4 epochs) | **38.74%** |
| **E2** | vector SD + raster SD + satellite | `residual_scale=1.0`, fixed | 24 epochs from scratch | **42.08%** |
| **E3** | vector SD + satellite | `residual_scale=1.0`, fixed | 24 epochs from scratch | **50.24%** |

E3 is the best result. E2 and E3 both train 24 epochs from scratch; the only input difference is
whether rasterized SD is present. E2 is 8.16 mAP lower, so in this camera-free setting the raster
branch did **not** complement the vector branch. E1 is not a 24-epoch comparison: its vector branch
is trained for only 4 epochs and its residual collapses to `0.00024`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for curves, per-class scores, and the exact caveats.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and tensor shapes.
See [CODE_MAP.md](CODE_MAP.md) for the provenance of every file.

## E4/E5: Z as the camera-BEV substitute

A second phase asks whether the raster-chain output `Z` can replace the camera BEV and serve as
the query/base for the vector branch.  The code and progress notes are in
[`e4_e5/`](e4_e5/README.md).

Current status:

- **R24** (raster-only Z): 24 epochs, **40.55%** mAP.
- **E5 gate** (`X=(1-gamma)Z+gamma*A`): best observed **42.23%** at epoch 12.
- **E5 residual** (`X=Z+A`): best observed **41.90%** at epoch 16.
- E3 remains the best result at **50.24%**.

The E4/E5 runs are being completed on a single GPU because the shared server's multi-rank
NCCL startup is unstable with the large SD-map pickle.  See `e4_e5/docs/INFRA_NOTES.md`.

## Architecture at a glance

```
satellite image (3x137x273) ──► StageFiLM raster encoder ──┐
rasterized SD    (3x100x200) ──► StageFiLM raster encoder ──┤
                                                          ├─► BEVFusion (residual alpha)
vector SD (8 classes, 11 pts) ──► MapGraphTransformer ─────┘
                                      │  SD tokens + confidence
                                      ▼
learnable BEV [200x100x256] ──► CrossAttentionBEV ──► DualVecCrossFusion
        + sinusoidal pos              (query = BEV)       (X_base + scale * Fuse(P))
                                                          │
                                                          ▼
                                              MapTRv2 decoder (900 queries)
                                                          │
                                                          ▼
                                        divider / ped_crossing / boundary
                                                  20 points per instance
```

In the paper, `X_base` is the camera BEV. In these experiments there is no camera:
`img_backbone=None`, `img_neck=None`, and `mlvl_feats=None`; `X_base` is the learnable
`bev_embedding` plus positional encoding. The satellite image is fused **after** the vector branch
through the raster encoder, not used as the vector query. This is the main adaptation and the main
limitation of this snapshot.

## Repository layout

```
configs/                     exact configs for E1/E2/E3 and their base configs
projects/mmdet3d_plugin/     trimmed MapQR/MapTRv2 plugin used by the experiments
  maptr/modules/vector_prior.py     [NEW] UMPE vector modules ported to MapQR
  maptr/modules/transformer.py      [MODIFIED] vector fusion + camera-free BEV
  maptr/modules/stagefilm_raster.py [OFFICIAL + small fix] raster encoder / BEVFusion
  maptr/detectors/maptrv2.py        [MODIFIED] map encoder + sdmap + no-camera path
  maptr/dense_heads/maptrv2_head.py [MODIFIED] vector_feats + no-camera batch handling
  datasets/...                      [MODIFIED] sdmap loading, PackSDMap, eval
tools/                       train.py, test.py, dist_train.sh, dist_test_map.sh
scripts/                     train_e1/e2/e3.sh, visualization, data preparation
docs/figures/                same-frame qualitative comparisons for E1/E2/E3
```

## What is included / excluded

**Included**

- The custom `projects/mmdet3d_plugin` code needed by the three experiments (including official
  MapQR/MapTRv2 modules used by them).
- The exact E1/E2/E3 configs and the base configs they inherit.
- Training/evaluation launch scripts, sdmap loading, map converter, and the visualization scripts
  used for the same-frame figures.
- Architecture notes and a file-level code map.

**Excluded**

- nuScenes raw data, UMPE-Prior SD pkl/PNG/JPG, generated GT, checkpoints, logs, work dirs.
- The full third-party `mmcv`, `mmdet`, `mmdet3d` source trees. Install the versions in
  [ENVIRONMENT.md](ENVIRONMENT.md).
- Unrelated AV2/VAD/BEVFormer detector variants and `*.bak_vec` files.

## Quick start (conceptual)

1. Install the environment in [ENVIRONMENT.md](ENVIRONMENT.md).
2. Put the plugin into a MapQR checkout, or run from this repo root with `PYTHONPATH=$PWD`.
3. Replace the absolute data paths in `configs/mapqr/*.py` (`/home/lbb/...`) with your paths.
4. For E1, edit `load_from` in `configs/mapqr/umpe_nocam_vector_ft.py` to point to the 12-epoch
   raster baseline checkpoint.
5. Run one of:
   ```bash
   bash scripts/train_e1.sh
   bash scripts/train_e2.sh
   bash scripts/train_e3.sh
   ```
6. Evaluate with `tools/dist_test_map.sh <config> <checkpoint> <gpus> --eval chamfer`.

The configs use `samples_per_gpu` 4 (E1/E3) or 5 (E2); effective batch is 20 in all three.

## Results and figures

- [EXPERIMENTS.md](EXPERIMENTS.md): three-experiment table, curves, per-class results, caveats.
- `docs/figures/three_exp_idx30_2x3.png`: clear satellite, crosswalk, all three classes.
- `docs/figures/three_exp_idx835_2x3.png`: complex intersection; E1/E2 over-predict crosswalks.
- `docs/figures/three_exp_idx985_2x3.png`: E1 misses a crosswalk; E2/E3 detect it.
- Additional frames: `idx25`, `idx465`, `idx555`, `idx735`, `idx795`, `idx1030` — see the full table in
  `EXPERIMENTS.md`; they cover success, over-prediction, E1 sparsity, and the crosswalk case where E3
  is clearly better.

The figures use the **same val frames** for all three models, `score > 0.50`, and the same
visualization layout. mAP is computed numerically by `nuscnes_eval.py` / `tpfp_chamfer.py`;
the figures are qualitative only.

## Provenance

- MapQR / MapTRv2: original open-source code (decoder, head, dataset, evaluation, raster modules).
- UMPE: original vector modules (`MapGraphTransformer`, `CrossAttentionBEV`, `DualVecCrossFusion`),
  ported into MapQR in `vector_prior.py`.
- The residual switch, camera-free learnable BEV, E1/E2/E3 configs, sdmap pipeline, and the
  same-frame visualization are adaptations made for this reproduction.

See [NOTICE.md](NOTICE.md). Keep the original licenses of MapQR/UMPE and third-party dependencies.
