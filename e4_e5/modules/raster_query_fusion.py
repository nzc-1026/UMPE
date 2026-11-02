# [NEW-E4E5] Raster-Z as BEV query; SD vector tokens as K/V.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner import BaseModule

from .vector_prior import CrossAttentionBEV


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RasterQueryFusion(BaseModule):
    """Use the fused raster prior Z as the BEV query for the SD vector branch.

    The incoming ``bev_tok`` is already LayerNorm-ed by the transformer.
    ``bev_pos`` is added to Z before attention and to the attention output.
    The gate is a convex combination of two positioned features, so the
    decoder receives exactly one copy of the positional encoding.

    RQ24  : X = CrossAttention(Q=Z+P, K/V=SD tokens) + P
    RQ24-G: X = (1-gamma) * (Z+P) + gamma * (CrossAttention(...)+P)
    Residual: X = (Z+P) + CrossAttention(Q=Z+P, K/V=SD tokens)
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 use_gate=False,
                 use_residual=False,
                 gate_init_bias=-2.0,
                 use_out_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.ca = CrossAttentionBEV(
            embed_dims=embed_dims, num_heads=num_heads, dropout=dropout)
        self.out_norm = (nn.LayerNorm(embed_dims)
                         if use_out_norm else nn.Identity())
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * embed_dims, embed_dims),
                nn.GELU(),
                nn.Linear(embed_dims, embed_dims),
            )
            # start as a constant gate near sigmoid(-2) = 0.12; a random
            # final layer would average 0.5 and immediately swamp Z.
            self.gate_init_bias = gate_init_bias
            self.reset_gate()

    def reset_gate(self):
        """Re-apply the gate initialization after the transformer's xavier pass."""
        if self.use_gate:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, self.gate_init_bias)

    def forward(self,
                bev_tok,                     # [B, 20000, C] = Z (normalized)
                hd_tokens=None, hd_U=None,
                sd_tokens=None, sd_U=None,
                bev_pos=None):               # [B, 20000, C] or None
        if sd_tokens is None:
            out = bev_tok
            if bev_pos is not None:
                out = out + bev_pos
            return out, {'gamma': None}
        # `img_metas` can collapse to one item per GPU while sdmap keeps the
        # real batch; broadcast the raster query to the vector batch.
        if sd_tokens.size(0) > bev_tok.size(0):
            bev_tok = bev_tok.expand(sd_tokens.size(0), -1, -1)
        z = bev_tok if bev_pos is None else bev_tok + bev_pos
        attn = self.ca(z, sd_tokens, sd_U)
        attn = self.out_norm(attn)
        if self.use_residual:
            # Z is the positioned base; the vector attention is an
            # unpositioned correction, so X = Z + A keeps one copy of P.
            return z + attn, {'gamma': None}
        if bev_pos is not None:
            attn = attn + bev_pos
        if self.use_gate:
            # Both branches carry P.  A convex combination therefore keeps
            # exactly one copy of the positional encoding, instead of only
            # gamma of it.
            gamma = torch.sigmoid(
                self.gate(torch.cat([z, attn], dim=-1)))
            out = (1.0 - gamma) * z + gamma * attn
            return out, {'gamma': gamma.detach()}
        return attn, {'gamma': None}
