# Three experiments

All experiments use the same data, the same MapTRv2 decoder, and the same Chamfer mAP evaluation.
The only differences are the input branches and the residual setting.

| ID | Inputs | Residual | Training | Vector epochs | mAP | divider | ped_crossing | boundary |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E1 | vector + raster + satellite | 0.1 learnable → 0.00024 | 12+4 | 4 | 38.74 | 41.24 | 37.75 | 37.23 |
| E2 | vector + raster + satellite | 1.0 fixed | 24 from scratch | 24 | 42.08 | 43.04 | 42.74 | 40.45 |
| E3 | vector + satellite | 1.0 fixed | 24 from scratch | 24 | **50.24** | **50.15** | **55.38** | **45.19** |

## Curves

| Epoch | E2 (vector+raster, 1.0) | E3 (pure vector, 1.0) |
|---:|---:|---:|
| 4 | 10.18 | 15.43 |
| 8 | 25.79 | 32.17 |
| 12 | 34.07 | 41.74 |
| 16 | 35.54 | 47.06 |
| 20 | 40.32 | 49.37 |
| 24 | 42.08 | 50.24 |

## What the numbers mean

- **E3 − E2 = +8.16** is the clean comparison: both are 24 epochs from scratch, only the raster SD
  branch is added/removed. Removing raster improves mAP by 8.16, including +12.64 on ped_crossing.
  In this camera-free setting the raster branch did not complement the vector branch.
- **E2 − E1 = +3.34** is **not** a clean residual comparison: E1's vector branch sees only 4 epochs,
  while E2's sees 24; E1 also has a learnable 0.1 residual that collapses to 0.00024.
- **E3 − E1 = +11.50** mixes input and training budget and should only be read as an overall gap.
- E3's 50.24% is independently verified with `tools/dist_test_map.sh` as 0.5023799.
  E2's 42.08% is the training-time evaluation; an independent rerun is still pending.

## Why E1's vector branch was suppressed

E1 starts from a 12-epoch raster baseline and fine-tunes for 4 epochs. The residual scalar is
learnable and initialized at 0.1; after training it is **0.00024**. The raster branch has a
pretrained ResNet18 and a fusion alpha around 0.54; raster SD and vector SD are two representations
of the same OSM. The optimizer therefore takes the raster shortcut and turns the vector branch off.
E1's 38.74 is close to the raster-only control (38.57), consistent with this diagnosis.

## Figures

The three figures use the **same val frames** for E1/E2/E3, selected to contain divider,
ped_crossing, and boundary GT. They use `score > 0.50`; prediction panels show faint dashed GT for
alignment. Red = divider, green = ped_crossing, blue = boundary.

| Figure | Frame | GT | E1 | E2 | E3 |
|---|---|---:|---:|---:|---:|
| `docs/figures/three_exp_idx30_2x3.png` | idx 30 | 4D/2P/5B = 11 | 3 | 3 | 6 |
| `docs/figures/three_exp_idx835_2x3.png` | idx 835 | 3D/2P/4B = 9 | 10 | 10 | 3 |
| `docs/figures/three_exp_idx985_2x3.png` | idx 985 | 4D/2P/3B = 9 | 4 | 7 | 8 |
| `docs/figures/three_exp_idx25_2x3.png` | idx 25 | 4D/2P/4B = 10 | 10 | 10 | 10 |
| `docs/figures/three_exp_idx465_2x3.png` | idx 465 | 2D/2P/3B = 7 | 10 | 10 | 10 |
| `docs/figures/three_exp_idx555_2x3.png` | idx 555 | 3D/3P/3B = 9 | 10 | 10 | 4 |
| `docs/figures/three_exp_idx735_2x3.png` | idx 735 | 5D/2P/4B = 11 | 4 | 10 | 8 |
| `docs/figures/three_exp_idx795_2x3.png` | idx 795 | 5D/2P/4B = 11 | 4 | 8 | 10 |
| `docs/figures/three_exp_idx1030_2x3.png` | idx 1030 | 5D/3P/4B = 12 | 8 | 8 | 10 |

- idx 30：卫星清楚，E1/E2 稀疏，E3 补出更多边界线；三类都可见。
- idx 835：复杂路口；E1/E2 各 10 条，绿色人行横道明显重复/过检；E3 只有 3 条，出现漏检。
- idx 985：E1 漏掉上方人行横道；E2/E3 都能检出。
- idx 25：三组都输出 10 条，和 GT 10 接近，是相对常见的成功案例。
- idx 465：GT 只有 7 条，三组都输出 10 条，全部偏多，说明“过检”不是 E1/E2 独有。
- idx 555：E3 只有 4 条、明显稀疏；E1/E2 各 10 条，反而更接近 GT 9。
- idx 735/795：E1 只有 4 条，E2/E3 分别 10/8 和 8/10，更接近 GT 11；说明 E1 的 4 轮矢量分支确实偏弱。
- idx 1030：E3 检出三处绿色人行横道；E1/E2 各 8 条，但绿色人行横道基本漏掉。这是 E3 在人行横道上优势最直观的一帧。

These figures are qualitative only; mAP is computed from the predicted/GT polylines by
`nuscnes_eval.py` and `tpfp_chamfer.py`, not from the images.

## Caveats

- E1 is a 12+4 short fine-tune, not a 24-epoch run. A fair "0.1 residual at 24 epochs" run
  (VR24-L) is not part of this snapshot.
- E2 and E3 use 4×5 and 5×4 GPUs respectively; effective batch is 20 in both.
- The camera-free `X_base` is a learnable BEV, not a satellite BEV. A more image-grounded variant
  (satellite BEV as X) is not included.
- The paper's 71.7 mAP uses camera + HD prior and is not directly comparable.
