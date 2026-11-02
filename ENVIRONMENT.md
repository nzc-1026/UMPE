# Environment and data

## Software

| Component | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.10.0+cu111 |
| torchvision | 0.11.0+cu111 |
| CUDA | 11.1 |
| MMCV | 1.4.0 |
| MMDetection | 2.14.0 |
| MMDetection3D | 0.17.2 |
| OpenCV | 4.13.0 |

The original MapQR/UMPE environment is a conda env named `umpe`. This repo assumes that
`mmcv`, `mmdet`, and `mmdet3d` are installed; their source trees are not vendored here.

## Hardware used

- 5 × NVIDIA A800-SXM4-80GB (E1/E3 used 5; E2 used 4 free GPUs).
- E2: 4 GPUs × samples_per_gpu 5 = effective batch 20.
- E3: 5 GPUs × samples_per_gpu 4 = effective batch 20.
- E1: 5 GPUs × samples_per_gpu 4 = effective batch 20.
- fp16 off; `find_unused_parameters=True`; `queue_length=1`; `aux_seg` off.

## Data

| Data | Size / count | Use |
|---|---|---|
| nuScenes map frames | 34,149 (train 28,130 / val 6,019) | frame list, poses, map API |
| UMPE-Prior vector SD | 1.9 GB; 34,149 tokens; 8 OSM classes; 11 points/polyline | vector branch |
| UMPE-Prior raster SD | 154 MB; 34,083 PNG; 100×200 | raster branch (E1/E2) |
| UMPE-Prior satellite | 280 MB; 34,149 JPG; 137×273 | all experiments |
| Generated map GT | 91 MB train / 19 MB val; 3 classes; 20 points | supervision/eval |
| nuScenes raw | ~300 GB | poses/map API only; no camera input |

The configs use absolute `/home/lbb/...` paths from the experiment server. Replace
`_ROOT`/`_SD_PKL` and `load_from` with your own paths before running. Data, GT, checkpoints,
logs, and work dirs are intentionally not included in this repository.

## Training commands

```bash
bash scripts/train_e1.sh   # needs the A0 12-epoch raster baseline
bash scripts/train_e2.sh
bash scripts/train_e3.sh
```

Test:

```bash
bash tools/dist_test_map.sh \
  configs/mapqr/umpe_nocam_vector_only_sat_24ep.py \
  /path/to/mAP_epoch_24.pth 2 --eval chamfer
```

## Reproduction notes

- E1's `load_from` points to a 12-epoch raster baseline; edit it in the config.
- E2's config uses `_base_ = ['./umpe_nocam_vector.py']` and overrides residual/epochs.
- E3's config uses `_base_ = ['./umpe_nocam_vector_only_sat.py']`, which removes
  `LoadSDRasterFromFile` and the `SDRaster_img` key.
- The evaluation metric is Chamfer mAP @0.5/1.0/1.5 m; the training configs use
  `save_best='NuscMap_chamfer/mAP'`.
