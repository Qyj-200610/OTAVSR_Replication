# Copyright (c) OTAVSR. All rights reserved.
"""Channel-wise concatenation, ``h = [z_a ; z_v]`` (paper Eq.(12), adopted)."""
import torch
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['ChannelConcatFusion']


@MODELS.register_module()
class ChannelConcatFusion(BaseModule):
    """Concatenate the two modality features along the channel axis.

    The token length is unchanged (``L``), which is the point the paper makes
    against temporal concatenation.

    Args:
        in_channels (int): channel number of each modality (``d``).
        out_channels (int, optional): output channels of ``Proj_av``. Defaults
            to ``in_channels`` (i.e. ``d`` = LLM hidden size).
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_channels = in_channels if out_channels is None else out_channels
        from otavsr.models.necks.projector import build_projector
        self.proj = build_projector(2 * in_channels, out_channels)

    def forward(self, x):
        """Args: dict with ``a`` and ``v`` of shape ``[B, L, d]``."""
        return self.proj(torch.cat([x['a'], x['v']], dim=-1))
