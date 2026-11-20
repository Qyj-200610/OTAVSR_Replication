# Copyright (c) OTAVSR. All rights reserved.
"""Data preprocessor that pads the variable-length AVSR batch."""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from otavsr.registry import MODELS

__all__ = ['AVSRDataPreprocessor']


def _pad_stack(seqs, padding_value: float = 0.0) -> torch.Tensor:
    """Stack tensors of different leading length (pad to the batch max)."""
    if isinstance(seqs, torch.Tensor):
        return seqs
    return torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(item) for item in seqs],
        batch_first=True,
        padding_value=padding_value)


def _pad_last(tensors, padding_value: float = 0.0) -> torch.Tensor:
    """Pad 2-D features ``[..., T]`` along the last axis."""
    if isinstance(tensors, torch.Tensor):
        return tensors
    max_len = max(item.shape[-1] for item in tensors)
    padded = [
        F.pad(item, (0, max_len - item.shape[-1]), value=padding_value)
        for item in tensors
    ]
    return torch.stack(padded)


@MODELS.register_module()
class AVSRDataPreprocessor(BaseDataPreprocessor):
    """Pad audio / video / target of a batch and pass the samples through.

    ``mmengine`` collates dicts with ``pseudo_collate``, so every field arrives
    as a *list* of tensors of different length. LRS3 clips range from a few
    frames to 400 (16 s at 25 fps), which makes this padding mandatory.

    Args:
        pad_value (float): value used for the audio / video padding.
    """

    def __init__(self, pad_value: float = 0.0) -> None:
        super().__init__()
        self.pad_value = pad_value

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Args: ``dict(inputs=..., data_samples=...)`` from the dataloader.

        Returns the same structure with padded tensors plus an
        ``audio_lengths`` field (in seconds) derived from ``num_frames``.
        """
        inputs = data['inputs']
        samples = data['data_samples']
        if isinstance(inputs, dict):
            audio = inputs['audio']
            video = inputs['video']
        else:  # a list of per-sample dicts
            audio = [item['audio'] for item in inputs]
            video = [item['video'] for item in inputs]

        batch = dict(
            audio=_pad_last(audio, self.pad_value),
            video=_pad_stack(video, self.pad_value),
        )

        lengths: Optional[torch.Tensor] = None
        if isinstance(inputs, dict) and inputs.get('audio_lengths') is not None:
            lengths = torch.as_tensor(inputs['audio_lengths'])
        elif samples is not None:
            num_frames = torch.stack([sample.num_frames for sample in samples])
            # 25 fps video -> 25 Hz encoder output, i.e. 2 audio frames each
            lengths = (num_frames / 25.0).float()
        if lengths is not None:
            batch['audio_lengths'] = lengths

        if samples is not None:
            batch['target_ids'] = _pad_stack(
                [sample.target_ids for sample in samples], padding_value=0)
        return dict(inputs=batch, data_samples=samples)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pad_value={self.pad_value})'
