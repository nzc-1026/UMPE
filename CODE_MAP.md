# Code map and provenance

Markers used at the top of files:

- `[OFFICIAL]`: original MapQR/MapTRv2/UMPE code, logic unchanged.
- `[OFFICIAL + small fix]`: original code with a small inference/path fix.
- `[MODIFIED]`: interface or camera-free adaptation.
- `[NEW]`: added for this reproduction.

## Core model

| File | Marker | Role | Used by |
|---|---|---|---|
| `maptr/modules/vector_prior.py` | NEW | `MapGraphTransformer`, `SineContinuousPositionalEncoding`, `CrossAttentionBEV`, `DualVecCrossFusion`; variable-length padding, confidence head, residual switch | E1/E2/E3 |
| `maptr/modules/transformer.py` | MODIFIED | Builds `vector_fusion_layer`; calls it before raster fusion; camera-free learnable-BEV path; `format_feats(None)` | E1/E2/E3 |
| `maptr/modules/stagefilm_raster.py` | OFFICIAL + fix | `StageFiLMRasterEncoder` (ResNet18 + FiLM) and `BEVFusion`/`SafeResidualFuse`; 3D input unsqueeze | E1/E2/E3 |
| `maptr/detectors/maptrv2.py` | MODIFIED | Builds `map_encoder`, extracts `vector_feats`, accepts `sdmap`, handles `img_feats=None`, flat `img_metas` | E1/E2/E3 |
| `maptr/dense_heads/maptrv2_head.py` | MODIFIED | Accepts `vector_feats`; no-camera batch/dtype/device from `img_metas`/`bev_embedding` | E1/E2/E3 |
| `maptr/modules/decoder.py` | OFFICIAL | `MapTRDecoder`, `CustomTransformerLayer`, `InstancePointAttention` | E1/E2/E3 |
| `maptr/modules/encoder.py` | OFFICIAL | BEVFormer/LSS encoders used by the original baseline | baseline |
| `maptr/modules/height_kernel_attention.py` | OFFICIAL | Height-aware attention dependency | E1/E2/E3 |
| `maptr/modules/geometry_kernel_attention.py` | OFFICIAL | Geometry attention dependency | E1/E2/E3 |
| `maptr/modules/builder.py` | OFFICIAL | Fuser builder | dependency |
| `maptr/assigners/maptr_assigner.py` | OFFICIAL | MapTR Hungarian assignment | training |
| `maptr/losses/*` | OFFICIAL | Chamfer / point / direction / segmentation losses | training |

## Data / eval

| File | Marker | Role | Used by |
|---|---|---|---|
| `datasets/nuscenes_offlinemap_dataset.py` | MODIFIED | Loads `sdmap_pkl`, converts polylines to numpy, keeps GT annotation, pops sdmap for previous frames | E1/E2/E3 |
| `datasets/pipelines/transform_3d.py` | MODIFIED | `PackSDMap` and `CustomCollect3D` sdmap handling | E1/E2/E3 |
| `datasets/pipelines/satellite_map.py` | OFFICIAL + path | Satellite loader; missing files return a black image | E1/E2/E3 |
| `datasets/pipelines/SD_map.py` | OFFICIAL + path | Rasterized-SD loader | E1/E2 |
| `datasets/nuscenes_dataset.py` | OFFICIAL | Base nuScenes dataset | dependency |
| `datasets/builder.py` | OFFICIAL | Custom dataset builder | dependency |
| `datasets/nuscnes_eval.py` | OFFICIAL | Chamfer mAP evaluation | all |
| `datasets/map_utils/tpfp_chamfer.py` | OFFICIAL | Chamfer TP/FP matching | all |
| `datasets/map_utils/mean_ap.py` | OFFICIAL | AP aggregation | all |
| `datasets/pipelines/loading.py`, `loading_utils.py`, `formating.py` | OFFICIAL | Data loading/formatting dependencies | dependency |
| `datasets/samplers/*` | OFFICIAL | Distributed/group samplers | dependency |

## Configs

| Config | Purpose |
|---|---|
| `configs/mapqr/mapqr_nusc_r50_24ep_new_Raster.py` | Base MapQR raster config; camera/HD/priors |
| `configs/mapqr/umpe_nocam_raster.py` | A0: raster SD + satellite, 12 epochs, no camera |
| `configs/mapqr/umpe_nocam_vector.py` | Base vector+raster config; vector modules, 12 epochs |
| `configs/mapqr/umpe_nocam_vector_ft.py` | **E1**: 12+4, residual 0.1 learnable |
| `configs/mapqr/umpe_nocam_vector_raster_24ep.py` | **E2**: 24 from scratch, residual 1.0 fixed |
| `configs/mapqr/umpe_nocam_vector_only_sat.py` | Base vector-only config; removes raster pipeline |
| `configs/mapqr/umpe_nocam_vector_only_sat_24ep.py` | **E3**: 24 from scratch, residual 1.0 fixed |

## Tools / scripts

| File | Role |
|---|---|
| `tools/train.py`, `tools/test.py` | MMDetection3D entry points |
| `tools/dist_train.sh`, `tools/dist_test_map.sh` | Distributed launch; test uses `--eval chamfer` |
| `tools/maptrv2/custom_nusc_map_converter.py` | nuScenes map API → 3-class vector GT |
| `scripts/train_e1.sh`, `train_e2.sh`, `train_e3.sh` | Launch commands for the three runs |
| `scripts/vis_three_experiments.py` | Same-frame E1/E2/E3 qualitative figures |
| `scripts/scan_ped_frames.py` | Select val frames with divider/ped/boundary GT |
| `scripts/data/parallel_map_converter.py`, `merge_partials.py`, `make_smoke_pkl.py` | Data preparation helpers |

## Key functions to inspect

1. `MapGraphTransformer.forward` — vector tokenization, padding mask, confidence.
2. `CrossAttentionBEV.forward` — BEV query, vector key/value, confidence bias.
3. `DualVecCrossFusion.forward` — HD/SD gate, source dropout, residual switch.
4. `MapTRPerceptionTransformer.get_bev_features` — camera-free learnable BEV branch.
5. `MapTRPerceptionTransformer.forward` — vector fusion before raster fusion.
6. `MapTRv2.extract_map_feat` — detector-level vector feature extraction.
7. `MapTRv2Head.forward` — 900 queries and decoder call.
8. `nuscnes_eval` / `tpfp_chamfer` — numerical mAP computation.

## Trimming note

This repository is a **cleaned subset** of the experiment checkout. `__init__.py` files were reduced
to the imports needed by these experiments; unrelated AV2/VAD/BEVFormer detectors, backbones, and
`*.bak_vec` files were removed. The model logic in the files above is unchanged from the experiment,
except for the markers/comments added for provenance.
