# Copyright (c) OTAVSR. All rights reserved.
"""Data sample of one audio-visual utterance."""
from typing import Optional

import torch
from mmengine.structures import BaseDataElement

__all__ = ['AVSRDataSample']


class AVSRDataSample(BaseDataElement):
    """One utterance of LRS3.

    Attributes set by the pipeline:
        num_frames (Tensor): number of valid frames of the 25 Hz fused
            sequence. The acoustic stream must be downsampled to exactly this
            length, otherwise the OT alignment is computed against shifted
            positions (see the guide §9.1).
        num_tokens (Tensor): number of target tokens (including EOS).
        target_ids (Tensor): ``[T]`` LLaMA token ids of the transcript.
        gt_text (str): the reference transcript, for WER.
        prompt (str): instruction text.
        sample_id (str): LRS3 clip name.
    """

    @property
    def num_frames(self) -> torch.Tensor:
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: torch.Tensor) -> None:
        self.set_field(value, '_num_frames', dtype=torch.Tensor)

    @property
    def num_tokens(self) -> torch.Tensor:
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value: torch.Tensor) -> None:
        self.set_field(value, '_num_tokens', dtype=torch.Tensor)

    @property
    def target_ids(self) -> torch.Tensor:
        return self._target_ids

    @target_ids.setter
    def target_ids(self, value: torch.Tensor) -> None:
        self.set_field(value, '_target_ids', dtype=torch.Tensor)

    @property
    def gt_text(self) -> Optional[str]:
        return self._gt_text

    @gt_text.setter
    def gt_text(self, value: str) -> None:
        self.set_field(value, '_gt_text', dtype=str)
