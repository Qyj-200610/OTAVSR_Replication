# Copyright (c) OTAVSR. All rights reserved.
"""Base classifier, mirroring ``mmpretrain/models/classifiers/base.py``."""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseModel

__all__ = ['BaseClassifier']


class BaseClassifier(BaseModel, metaclass=ABCMeta):
    """Base class for the audio-visual recognisers of this project.

    The three modes follow mmpretrain: ``tensor`` returns the fused features,
    ``loss`` returns a dict of losses and ``predict`` returns predictions.
    """

    def __init__(self, init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        #: filled by :meth:`parse_losses`, read by ``OTMonitorHook``
        self._ot_stats: Dict[str, float] = {}

    def forward(self,
                inputs,
                data_samples: Optional[Sequence] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.extract_feat(inputs, data_samples)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        raise RuntimeError(f'Invalid mode "{mode}".')

    @abstractmethod
    def extract_feat(self, inputs, data_samples=None):
        """Return the intermediate features used by loss / predict."""

    def loss(self, inputs, data_samples=None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict(self, inputs, data_samples=None) -> List:
        raise NotImplementedError

    def parse_losses(self, losses: Dict[str, torch.Tensor]):
        """Split scalar diagnostics out of the loss dict.

        ``mmengine`` requires every value of the loss dict to be a tensor.
        The OT branch produces a handful of scalars that we want to log but
        not back-propagate; they are stashed in ``self._ot_stats`` and removed
        here, so ``OTMonitorHook`` can pick them up without touching mmengine.
        """
        self._ot_stats = {
            key: float(value)
            for key, value in losses.items() if not key.startswith('loss')
        }
        pure_losses = {
            key: value
            for key, value in losses.items() if key.startswith('loss')
        }
        return super().parse_losses(pure_losses)
