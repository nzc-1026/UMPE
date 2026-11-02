# [NEW] UMPE vector modules ported to MapQR (three-experiment repo).
"""Vector (OSM/SD polyline) prior encoder and fusion.

Ported from UMPE-main/Maptrv2:
  projects/mmdet3d_plugin/bevformer/modules/encoder.py
  projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py

Only the SD branch is kept (HD prior is intentionally disabled).  The
official implementation returns features for one sample at a time; here the
encoder accepts a list of per-sample dicts, pads the variable-length
polyline sets and returns batched tokens so that normal MapQR batches
(samples_per_gpu > 1) work.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    POSITIONAL_ENCODING,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule


@POSITIONAL_ENCODING.register_module()
class SineContinuousPositionalEncoding(BaseModule):
    """Continuous sinusoidal encoding for (x, y) polyline points."""

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        """x: [B, N, D] -> [B, N, D * num_feats]."""
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapGraphTransformer(BaseModule):
    """Encode OSM/SD polylines into tokens + confidence scores.

    The SD pickle stores, per frame token, a dict
    ``{osm_class: [tensor(11, 2), ...]}`` in the ego frame.  Each polyline
    becomes one token: continuous PE of its 11 points + class one-hot
    (11 dims, HD ids 0-2 unused) + source one-hot (2 dims, SD = [1, 0]).
    """

    TYPE_VOCAB = {
        'divider': 0,        # hd_divider
        'ped_crossing': 1,   # hd_ped_crossing
        'boundary': 2,       # hd_boundary
        'highway': 3,        # sd_highway
        'primary': 4,        # sd_primary
        'secondary': 5,      # sd_secondary
        'tertiary': 6,       # sd_tertiary
        'residential': 7,    # sd_residential
        'service': 8,        # sd_service
        'pedestrian': 9,     # sd_pedestrian
        'other': 10,         # sd_other
    }

    def __init__(self,
                 input_dim=365,      # 32 * 11 + 11 + 2
                 dmodel=256,
                 hidden_dim=256,
                 nheads=8,
                 nlayers=6,
                 batch_first=True,
                 pos_encoder=None,
                 max_polylines=256,
                 **kwargs):
        super(MapGraphTransformer, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.max_polylines = max_polylines
        self.map_embedding = nn.Linear(input_dim, dmodel)

        if pos_encoder is not None:
            self.use_positional_encoding = True
            self.pos_encoder = build_positional_encoding(pos_encoder)
        else:
            self.use_positional_encoding = False
            self.pos_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dmodel,
            nhead=nheads,
            dim_feedforward=hidden_dim,
            batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers)

        # input is dmodel (the official Maptrv2 code used hidden_dim here,
        # which only worked because their config set both to 256)
        self.confidence_head = nn.Sequential(
            nn.Linear(dmodel, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid())

        self.init_weights()

    def init_weights(self):
        xavier_init(self.map_embedding, distribution='uniform')
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                xavier_init(p, distribution='uniform')
        if self.pos_encoder is not None and hasattr(self.pos_encoder, 'init_weights'):
            self.pos_encoder.init_weights()

    def _to_line(self, line, device):
        if torch.is_tensor(line):
            line = line.detach().float().to(device)
        else:
            line = torch.as_tensor(np.asarray(line), dtype=torch.float32, device=device)
        line = line.reshape(-1, 2)
        if line.shape[0] != 11:
            idx = torch.linspace(
                0, line.shape[0] - 1, 11, device=line.device).round().long()
            line = line[idx]
        return line

    def process_map_data(self, prior_map):
        """dict{class_name: [polyline, ...]} -> [N,11,2], [N], [N]."""
        device = next(self.parameters()).device
        polylines, classes, valid = [], [], []
        for category, lines in prior_map.items():
            if lines is None or len(lines) == 0:
                continue
            cls_id = self.TYPE_VOCAB.get(category, self.TYPE_VOCAB['other'])
            for line in lines:
                polylines.append(self._to_line(line, device))
                classes.append(cls_id)
                valid.append(True)

        if len(polylines) == 0:
            graph = torch.zeros((1, 11, 2), dtype=torch.float32, device=device)
            cls = torch.zeros((1,), dtype=torch.long, device=device)
            valid = torch.zeros((1,), dtype=torch.bool, device=device)
        else:
            graph = torch.stack(polylines, dim=0)
            cls = torch.tensor(classes, dtype=torch.long, device=device)
            valid = torch.ones((graph.size(0),), dtype=torch.bool, device=device)
            if graph.size(0) > self.max_polylines:
                dist = graph.mean(dim=1).norm(dim=-1)
                keep = torch.argsort(dist)[:self.max_polylines]
                graph, cls, valid = graph[keep], cls[keep], valid[keep]
        return graph, cls, valid

    @staticmethod
    def _to_map_list(sdmap):
        """Flatten whatever nesting mmcv's collate produced into dicts."""
        if isinstance(sdmap, dict):
            return [sdmap]
        if hasattr(sdmap, 'data') and not torch.is_tensor(sdmap):
            return MapGraphTransformer._to_map_list(sdmap.data)
        if isinstance(sdmap, (list, tuple)):
            out = []
            for item in sdmap:
                out.extend(MapGraphTransformer._to_map_list(item))
            return out
        return []

    def forward(self, sdmap, map_type='sdmap', img_feats=None):
        """sdmap: list[dict] (batch) or a single dict.

        Returns
        -------
        tokens: [B, Nmax, dmodel]
        conf:   [B, Nmax] in (0, 1); padding positions are 0.
        """
        sdmap = self._to_map_list(sdmap)
        if len(sdmap) == 0:
            raise ValueError('sdmap batch is empty')

        feats, confs = [], []
        for prior_map in sdmap:
            graph, cls, valid = self.process_map_data(prior_map)
            num_real = int(valid.sum().item())

            if self.use_positional_encoding:
                graph = self.pos_encoder(graph)  # [N, 11, 32]
            n, npoints, pdim = graph.shape
            cls_onehot = F.one_hot(
                cls, num_classes=len(self.TYPE_VOCAB)).float()
            src_onehot = graph.new_zeros((n, 2))
            src_onehot[:, 0] = 1.0  # SD source

            feat = torch.cat(
                [graph.reshape(n, npoints * pdim), cls_onehot, src_onehot],
                dim=-1)
            feat = self.map_embedding(feat)  # [N, C]

            # Avoid an all-masked attention row when the frame has no SD line.
            enc_mask = ~valid
            if num_real == 0:
                enc_mask = torch.zeros_like(valid)
            feat = self.transformer_encoder(
                feat.unsqueeze(0), src_key_padding_mask=enc_mask.unsqueeze(0))
            conf = self.confidence_head(feat).squeeze(-1).squeeze(0)  # [N]
            if num_real == 0:
                conf = torch.zeros_like(conf)
            else:
                conf = conf * valid.float()

            feats.append(feat.squeeze(0))
            confs.append(conf)

        nmax = max(f.size(0) for f in feats)
        padded_feats, padded_confs = [], []
        for feat, conf in zip(feats, confs):
            pad = nmax - feat.size(0)
            if pad > 0:
                feat = F.pad(feat, (0, 0, 0, pad))
                conf = F.pad(conf, (0, pad))
            padded_feats.append(feat)
            padded_confs.append(conf)

        return torch.stack(padded_feats, dim=0), torch.stack(padded_confs, dim=0)


class CrossAttentionBEV(BaseModule):
    """BEV queries attend to prior-map tokens with a log-confidence bias."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 qkv_bias=True,
                 init_cfg=None,
                 epsilon=1e-6):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.epsilon = epsilon
        if self.head_dim * num_heads != self.embed_dims:
            raise ValueError(
                f'embed_dims ({embed_dims}) must be divisible by '
                f'num_heads ({num_heads})')

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bev_embed, prior_feats, confidence_token):
        # bev_embed: [B, H*W, C]; prior_feats: [B, N, C]; conf: [B, N]
        batch_size, seq_len_q, _ = bev_embed.shape
        _, seq_len_kv, _ = prior_feats.shape

        q = self.q_proj(bev_embed)
        k = self.k_proj(prior_feats)
        v = self.v_proj(prior_feats)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        confidence_bias = torch.log(confidence_token.clamp(min=self.epsilon))
        confidence_bias = confidence_bias.unsqueeze(1).unsqueeze(1)
        confidence_bias = confidence_bias.expand(-1, self.num_heads, seq_len_q, -1)
        attn_scores = attn_scores + confidence_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dims)
        return self.out_proj(attn_output)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DualVecCrossFusion(BaseModule):
    """Gated fusion of prior-map attention outputs.

    HD is never supplied in this reproduction, so the HD branch is dormant
    and the gate degenerates to the SD branch.  The class is kept identical
    to the official implementation so checkpoints/configs stay compatible.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 source_dropout_p=0.3,
                 bev_w=100,
                 bev_h=200,
                 use_residual=True,
                 residual_scale=0.1,
                 residual_scale_learnable=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.ca_hd = CrossAttentionBEV(embed_dims=embed_dims, num_heads=num_heads)
        self.ca_sd = CrossAttentionBEV(embed_dims=embed_dims, num_heads=num_heads)
        self.gate_logits = nn.Parameter(torch.tensor([0.0, 0.0]))  # [HD, SD]
        self.source_dropout_p = source_dropout_p
        # The official Maptrv2 layer replaces the BEV with the attention
        # output.  That assumes a strong camera BEV.  In the camera-free
        # setting the learnable BEV is the only base representation, so keep
        # it as a residual and let the prior branch learn a correction.
        self.use_residual = use_residual
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale)),
            requires_grad=residual_scale_learnable)

    def init_weights(self):
        super().init_weights()
        nn.init.constant_(self.gate_logits, 0.0)

    def _maybe_source_dropout(self, has_hd, has_sd):
        use_hd, use_sd = has_hd, has_sd
        if self.training and has_hd and has_sd and self.source_dropout_p > 0:
            if torch.rand(1).item() < self.source_dropout_p:
                if torch.rand(1).item() < 0.5:
                    use_hd = False
                else:
                    use_sd = False
        return use_hd, use_sd

    def forward(self,
                bev_tok,                     # [B, Nq, C]
                hd_tokens=None, hd_U=None,   # [B, Nh, C], [B, Nh]
                sd_tokens=None, sd_U=None):  # [B, Ns, C], [B, Ns]
        # In this repo `img_metas` collapses to one item per GPU, so the
        # learnable BEV enters with batch 1 while the SD tokens carry the
        # real per-sample batch.  Broadcast the BEV queries to match.
        ref_tokens = sd_tokens if sd_tokens is not None else hd_tokens
        if ref_tokens is not None and ref_tokens.size(0) > bev_tok.size(0):
            bev_tok = bev_tok.expand(ref_tokens.size(0), -1, -1)

        has_hd = (hd_tokens is not None) and (hd_tokens.size(1) > 0)
        has_sd = (sd_tokens is not None) and (sd_tokens.size(1) > 0)
        use_hd, use_sd = self._maybe_source_dropout(has_hd, has_sd)

        y_hd = torch.zeros_like(bev_tok)
        y_sd = torch.zeros_like(bev_tok)
        if use_hd:
            y_hd = self.ca_hd(bev_tok, hd_tokens, hd_U)
        if use_sd:
            y_sd = self.ca_sd(bev_tok, sd_tokens, sd_U)

        pres = torch.tensor(
            [float(use_hd), float(use_sd)], device=bev_tok.device)
        g = torch.softmax(self.gate_logits, dim=0) * pres
        g = g / g.sum().clamp_min(1e-6)
        fused = g[0] * y_hd + g[1] * y_sd
        if self.use_residual:
            fused = bev_tok + self.residual_scale * fused
        return fused, {'gate': g.detach(), 'use_hd': use_hd, 'use_sd': use_sd}
