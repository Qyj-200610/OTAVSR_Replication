# Copyright (c) OTAVSR. All rights reserved.
"""Temporal concatenation, the ``T_Concat`` row of Table I / Table II.

This is the fusion used by Llama-AVSR (``[BOS][video][audio][prompt][target]``),
included here only as the baseline the paper argues against: it doubles the
token sequence (``2L``) and therefore the LLM cost.
"""
from typing import Optional

import torch
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['TemporalConcatFusion']


@MODELS.register_module()
class TemporalConcatFusion(BaseModule):
    """Stack the two modalities along the time axis, then project.

    Args:
        in_channels (int): width ``d`` of each modality.
        out_channels (int, optional): width of ``Proj_av``.
        modalities (tuple): order of the temporal concatenation. Llama-AVSR
            puts the video stream first.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 modalities: tuple = ('v', 'a'),
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = in_channels if out_channels is None else out_channels
        from otavsr.models.necks.projector import build_projector
        self.modalities = modalities
        self.proj = build_projector(in_channels, out_channels)

    def forward(self, x):
        hidden = torch.cat([x[key] for key in self.modalities], dim=1)
        return self.proj(hidden)
