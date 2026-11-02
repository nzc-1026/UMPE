# Architecture

## 1. Overall data flow

```
satellite  [B,137,273,3] ──► StageFiLMRasterEncoder ──┐
raster SD  [B,100,200,3] ──► StageFiLMRasterEncoder ──┤
                                                     ├─► BEVFusion (SE(2)-style gate + residual)
vector SD  (8 classes, 11 pts per polyline) ──► MapGraphTransformer
                                                     │
                                      SD tokens [B,N,256], confidence [B,N]
                                                     │
learnable BEV [B,20000,256] ──► CrossAttentionBEV ◄──┘
 + sinusoidal bev_pos            (query = BEV, key/value = SD tokens)
                                                     │
                                                     ▼
                                      DualVecCrossFusion
                                      X_out = X_base + scale * Fuse(P)
                                                     │
                                                     ▼
                                      MapTRv2 decoder (6 layers, 900 queries)
                                                     │
                                                     ▼
                          divider / ped_crossing / boundary, 20 pts per instance
```

The model is a MapTRv2/MapQR detector with the UMPE vector-prior branch inserted **before** the
raster fusion. The vector branch is the OSM vector path; the raster branch is the satellite +
rasterized-SD path.

## 2. Camera removal: what changes

In the paper, the vector branch queries a **camera BEV**:

- camera images → image backbone/neck → BEV encoder → `X_base` (camera BEV);
- `CrossAttentionBEV(query=X_base, key/value=vector tokens)` → vector correction;
- official `DualVecCrossFusion` returns the fused prior output (it replaces the BEV).

In this reproduction there is no camera:

- `img_backbone=None`, `img_neck=None`;
- `extract_feat(img=None)` returns `None`; `mlvl_feats=None`;
- `MapTRv2Head` has `self.bev_embedding = nn.Embedding(200*100, 256)`;
- `transformer.get_bev_features()` detects `mlvl_feats is None` and builds:
  ```python
  bev = bev_queries.unsqueeze(1).repeat(1, bs, 1)   # [20000, B, 256]
  bev = bev + bev_pos.flatten(2).permute(2, 0, 1)   # fixed sinusoidal position
  return dict(bev=bev.permute(1, 0, 2))              # [B, 20000, 256]
  ```
- `X_base` is therefore a **learnable BEV canvas**, not a camera BEV.

This is why the official replacement fusion is risky: it discards the only BEV base. `DualVecCrossFusion`
in `vector_prior.py` therefore supports:
```python
fused = g_hd * y_hd + g_sd * y_sd
if use_residual:
    fused = bev_tok + residual_scale * fused
```
- official behavior: `use_residual=False` → `X_out = fused` (replace);
- E1: `use_residual=True`, `residual_scale=0.1`, learnable → collapsed to `0.00024`;
- E2/E3: `use_residual=True`, `residual_scale=1.0`, `requires_grad=False`.

The satellite image is **not** used as the vector query in this snapshot. It goes through the
StageFiLM raster encoder and is fused after the vector branch. A more faithful image-grounded
variant would use the satellite BEV as `X_base` (not implemented in these three runs).

## 3. HD-prior removal: what changes

- HD vector branch: empty (`hd_tokens=None`); `DualVecCrossFusion` degenerates to the SD branch.
- Only OSM-derived SD vector (8 classes) and rasterized SD remain; HD lane-level priors are absent.
- The 11-class one-hot in the vector input still has slots for the 3 HD classes, but they are empty;
  the 8 SD classes are populated.

## 4. Vector branch (`vector_prior.py`)

### 4.1 `MapGraphTransformer`

- Input per polyline: 11 points × 32 sinusoidal PE + 11 class one-hot + 2 source embedding = **365**.
- `dmodel=256`, `hidden_dim=256`, `nheads=8`, `nlayers=6`.
- `SineContinuousPositionalEncoding`: `num_feats=16`, `temperature=1000`,
  `normalize_range=[30,60]`, `offset=[-15,-30]`.
- Output: per-token features `[B,N,256]` and confidence `[B,N]`.
- Variable-length polylines are padded; a padding mask is used in attention.

### 4.2 `CrossAttentionBEV`

- Query: BEV tokens `[B,20000,256]`.
- Key/value: vector tokens `[B,N,256]`.
- 8 heads, dropout 0.1, log-confidence bias on the attention logits.
- Output: `[B,20000,256]`.

### 4.3 `DualVecCrossFusion`

- Two cross-attention branches (HD and SD); HD is empty in these experiments.
- Gate: `softmax(gate_logits)` times source presence; source dropout 0.3 during training.
- Optional residual injection (see §2).
- Returns `(fused_bev, aux)`; `aux` contains gate values for logging.

## 5. Raster branch (`stagefilm_raster.py`)

- `StageFiLMRasterEncoder`: shared ResNet18 + FiLM conditioning, `embed_dims=256`, `cond_dim=32`.
- Satellite and raster SD are encoded separately; missing source (`None`) is handled.
- `BEVFusion`: SE(2)-style alignment/gating and `SafeResidualFuse`:
  ```python
  return bev_embed + alpha * fused_bev
  ```
  `alpha` is initialized at 0.0 and learned (official behavior).

## 6. Decoder / head

- `MapTRv2Head`: BEV 200×100, 3 classes, 20 points/instance.
- `num_query=900`: 100 one2one queries + 600 one2many queries (`k_one2many=6`).
- `MapTRDecoder`: 6 layers, 8 heads, FFN 512, dropout 0.1, `InstancePointAttention`.
- At inference only the 100 one2one queries are used.
- Loss: Focal (γ=2, α=0.25, weight 2.0), PtsL1 5.0, PtsDirCos 0.005; one2many weight 6.
- Auxiliary segmentation is disabled (`aux_seg` off) because there is no camera feature map.

## 7. Inputs and evaluation

- BEV: 200×100, `pc_range=[-15,-30,-2,15,30,2]`, ~0.3 m/pixel.
- Satellite: 137×273, ~0.22 m/pixel; raster SD: 100×200.
- GT: nuScenes map API, 3 classes, 20 points/instance; 28,130 train / 6,019 val frames.
- Metric: Chamfer mAP at 0.5/1.0/1.5 m, averaged over 3 classes.
  Implemented by `nuscnes_eval.py` and `map_utils/tpfp_chamfer.py`; the figures do not affect mAP.

## 8. Training protocol

| Run | Init | Total epochs | Vector epochs | Residual |
|---|---|---:|---:|---|
| E1 | A0 12-epoch raster baseline | 16 (12+4) | 4 | 0.1 learnable → 0.00024 |
| E2 | random | 24 | 24 | 1.0 fixed |
| E3 | random | 24 | 24 | 1.0 fixed |

E2 and E3 are the clean 24-epoch comparison. E1 is a short fine-tune and is **not** a clean
residual/epoch ablation.
