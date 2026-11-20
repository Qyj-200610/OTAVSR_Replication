# Copyright (c) OTAVSR. All rights reserved.
"""Temporal downsampling of the acoustic stream (50 Hz -> 25 Hz)."""

import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['TemporalDownsample']


@MODELS.register_module()
class TemporalDownsample(BaseModule):
    """A single strided ``Conv1d`` that halves the frame rate.

    The paper says "another temporal convolution layer with a stride of 2" and
    reports neither the kernel size nor whether it is trainable; ``kernel_size
    = 3, padding = 1`` is the choice that keeps ``ceil(L / 2)`` exactly and is
    what we document in the guide.  Whisper already downsamples 100 Hz mel to
    50 Hz, so this layer takes 50 Hz -> 25 Hz and matches AV-HuBERT's rate.

    Args:
        channels (int): feature width (unchanged by the layer).
        kernel_size (int): convolution kernel.
        stride (int): stride; 2 for the paper's setting.
        trainable (bool): if ``False`` the weights are frozen.
    """

    def __init__(self,
                 channels: int = 1024,
                 kernel_size: int = 3,
                 stride: int = 2,
                 trainable: bool = True,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        if not trainable:
            for param in self.conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Args: ``[B, L, C]`` -> ``[B, ceil(L / stride), C]``."""
        hidden = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return hidden