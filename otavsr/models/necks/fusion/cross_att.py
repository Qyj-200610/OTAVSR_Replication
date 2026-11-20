# Copyright (c) OTAVSR. All rights reserved.
"""Multi-head cross-attention fusion, ``h = CrossAtt(z_a, z_v, W_att)``."""
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['CrossAttentionFusion']


@MODELS.register_module()
class CrossAttentionFusion(BaseModule):
    """Audio queries the visual stream (paper Eq.(12)).

    The paper only writes ``CrossAtt(z_a, z_v, W_att)`` and does not report the
    number of heads / blocks, so the defaults below are our choice: one
    pre-LN block with 8 heads (same head count as MMS-LLaMA's
    ``Multimodal_Attention``) plus a feed-forward sub-layer.

    Args:
        in_channels (int): feature width ``d`` of both modalities.
        out_channels (int, optional): width of ``Proj_av``.
        num_heads (int): attention heads.
        dropout (float): attention / FFN dropout.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = in_channels if out_channels is None else out_channels
        from otavsr.models.necks.projector import build_projector
        self.attn = nn.MultiheadAttention(
            in_channels, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(in_channels)
        self.norm_kv = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * in_channels, in_channels))
        self.proj = build_projector(in_channels, out_channels)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """Args: dict with ``a`` / ``v`` of shape ``[B, L, d]``.

        ``key_padding_mask`` (``[B, L]``, ``True`` for padded frames) is
        forwarded to the attention so that padded visual frames are ignored.
        """
        query = self.norm_q(x['a'])
        key_value = self.norm_kv(x['v'])
        attn_out, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False)
        hidden = x['a'] + attn_out
        hidden = hidden + self.ffn(self.norm_ffn(hidden))
        return self.proj(hidden)
