# Copyright (c) OTAVSR. All rights reserved.
"""Projection layers that map modality features into the LLM embedding space."""
from typing import Optional

import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['ModalityProjector', 'build_projector']


def build_projector(in_channels: int,
                    out_channels: int,
                    hidden_channels: Optional[int] = None,
                    layernorm: bool = False) -> nn.Sequential:
    """Linear -> GELU -> Linear(-> LayerNorm).

    This is the stack used by Llama-AVSR (``models/modeling_AVSRLLM.py``):
    ``Linear(audio_dim * ratio, intermediate) -> ReLU -> Linear(intermediate,
    hidden)``, with an optional ``LayerNorm`` at the end (their
    ``--layernorm-projector`` flag, from arXiv:2510.22603).
    """
    hidden_channels = hidden_channels or out_channels
    layers = [
        nn.Linear(in_channels, hidden_channels), nn.GELU(),
        nn.Linear(hidden_channels, out_channels)
    ]
    if layernorm:
        layers.append(nn.LayerNorm(out_channels))
    return nn.Sequential(*layers)


@MODELS.register_module()
class ModalityProjector(BaseModule):
    """``Proj_a`` / ``Proj_v`` of paper Eq.(1), optionally with downsampling.

    Args:
        in_channels (int): encoder width (1024 for Whisper-medium and
            AV-HuBERT-large).
        out_channels (int): LLM width (3072 for LLaMA-3.2-3B).
        hidden_channels (int, optional): inner width, defaults to ``out``.
        layernorm (bool): append the pre-fusion LayerNorm.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int] = None,
                 layernorm: bool = False,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.proj = build_projector(in_channels, out_channels,
                                    hidden_channels, layernorm)

    def forward(self, x):
        return self.proj(x)
