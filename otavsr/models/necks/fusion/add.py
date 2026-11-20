# Copyright (c) OTAVSR. All rights reserved.
"""Element-wise addition, ``h = z_a + z_v`` (paper Eq.(12))."""
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['AddFusion']


@MODELS.register_module()
class AddFusion(BaseModule):
    """Requires both modalities to be projected to ``d`` first."""

    def __init__(self, in_channels: int, out_channels: int = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = in_channels if out_channels is None else out_channels
        from otavsr.models.necks.projector import build_projector
        self.proj = build_projector(in_channels, out_channels)

    def forward(self, x):
        return self.proj(x['a'] + x['v'])