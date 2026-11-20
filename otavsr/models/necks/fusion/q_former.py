# Copyright (c) OTAVSR. All rights reserved.
"""Q-Former based fusion, ``h = QFormer([z_a ; z_v], W_q)`` (paper Eq.(12)).

The paper only states that the querying length ``L_q`` is proportional to the
clip duration.  We follow the official MMS-LLaMA implementation
(``src/conf/mms-llama.yaml``): ``queries_per_sec = 3``, 2 layers, dimension
1024, cross-attention in every layer.  In MMS-LLaMA the number of queries is
predicted by a separate speech-rate predictor; here it is computed
deterministically from the input length, which is enough to reproduce the
*shape* of the ablation (and is what you should report if you compare with the
paper's Q_former row).
"""
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['QFormerFusion', 'QFormerBlock']


class QFormerBlock(BaseModule):
    """Pre-LN self-attention + cross-attention + FFN on the query tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim))

    def forward(self, query, memory, memory_mask=None):
        q = self.norm_sa(query)
        attn, _ = self.self_attn(q, q, q, need_weights=False)
        query = query + attn

        q = self.norm_ca(query)
        attn, _ = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_mask,
            need_weights=False)
        query = query + attn
        query = query + self.ffn(self.norm_ffn(query))
        return query


@MODELS.register_module()
class QFormerFusion(BaseModule):
    """Pool ``L`` audio-visual frames into ``L_q ~ duration`` query tokens.

    Args:
        in_channels (int): width ``d`` of each modality.
        out_channels (int, optional): width of ``Proj_av``.
        queries_per_sec (float): queries per second of audio-visual frames.
        frame_rate (float): frame rate of the fused sequence (25 Hz).
        num_layers (int): transformer blocks (MMS-LLaMA uses 2).
        num_heads (int): attention heads.
        max_queries (int): size of the learnable query bank.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 queries_per_sec: float = 3.0,
                 frame_rate: float = 25.0,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 max_queries: int = 512,
                 dropout: float = 0.0,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = in_channels if out_channels is None else out_channels
        from otavsr.models.necks.projector import build_projector
        self.queries_per_sec = queries_per_sec
        self.frame_rate = frame_rate
        self.max_queries = max_queries
        self.query_tokens = nn.Parameter(
            torch.zeros(1, max_queries, in_channels))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.memory_proj = nn.Linear(2 * in_channels, in_channels)
        self.blocks = nn.ModuleList([
            QFormerBlock(in_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(in_channels)
        self.proj = build_projector(in_channels, out_channels)

    def num_queries(self, num_frames: int) -> int:
        """``L_q`` for a clip of ``num_frames`` frames at ``frame_rate``."""
        seconds = num_frames / self.frame_rate
        return int(
            max(1, min(self.max_queries, round(seconds * self.queries_per_sec))))

    def forward(self, x, memory_mask: Optional[torch.Tensor] = None):
        # paper writes QFormer([z_a ; z_v], W_q): channel concatenation
        memory = self.memory_proj(torch.cat([x['a'], x['v']], dim=-1))
        num_q = self.num_queries(memory.shape[1])
        query = self.query_tokens[:, :num_q].expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, memory, memory_mask)
        return self.proj(self.norm(query))