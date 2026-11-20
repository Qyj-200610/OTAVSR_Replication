# Copyright (c) OTAVSR. All rights reserved.
"""Frozen Whisper encoder (paper §II-A).

The wrapper deliberately takes **log-mel features** rather than raw waveform:
Whisper's encoder expects ``[B, n_mels, 3000]`` and the mel extraction lives in
the dataset pipeline (``WhisperLogMel`` transform), which keeps the model side
free of audio IO.

Frozen means frozen: this module never returns gradients for its parameters,
and the paper freezes both encoders.
"""
from typing import Optional

import torch
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['WhisperEncoder']

#: frames per second of the Whisper-medium encoder output
WHISPER_FRAME_RATE = 50


@MODELS.register_module()
class WhisperEncoder(BaseModule):
    """Thin wrapper around ``transformers`` ``WhisperModel.encoder``.

    Args:
        model_name (str): HF hub id. Both official baselines use the
            English-only checkpoint ``openai/whisper-medium.en``.
        frozen (bool): freeze the encoder (paper: frozen).
        feature_rate (int): output frame rate, 50 Hz for Whisper-medium.
        lengths_scale (float): multiplier turning an audio duration in seconds
            into a valid frame count (``50`` for Whisper), used to cut the
            30 s padding that Whisper always produces.
    """

    def __init__(self,
                 model_name: str = 'openai/whisper-medium.en',
                 frozen: bool = True,
                 feature_rate: int = WHISPER_FRAME_RATE,
                 lengths_scale: Optional[float] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model_name = model_name
        self.frozen = frozen
        self.feature_rate = feature_rate
        self.lengths_scale = (float(feature_rate)
                              if lengths_scale is None else lengths_scale)
        self._encoder = None

    @property
    def out_channels(self) -> int:
        """1024 for Whisper-medium."""
        return 1024

    def _build(self):
        try:
            from transformers import WhisperModel
        except ImportError as error:  # pragma: no cover - env dependent
            raise ImportError(
                'WhisperEncoder needs `transformers` and a local copy of '
                f'{self.model_name!r}. Install it with `pip install '
                'transformers` and pre-download the checkpoint.') from error
        encoder = WhisperModel.from_pretrained(self.model_name).encoder
        encoder.requires_grad_(not self.frozen)
        encoder.eval() if self.frozen else encoder.train()
        return encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self._build()
        return self._encoder

    def forward(self,
                input_features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            input_features (Tensor): ``[B, n_mels, T_mel]`` log-mel features.
            lengths (Tensor, optional): audio durations in seconds, ``[B]``.
                Whisper pads every clip to 30 s, so without this the sequence
                length is always 1500 and the alignment means nothing.

        Returns:
            Tensor: ``[B, T, 1024]`` frame-level features at 50 Hz.
        """
        hidden = self.encoder(input_features).last_hidden_state
        if lengths is not None:
            valid = torch.ceil(
                lengths.to(hidden.device) * self.lengths_scale).long()
            steps = torch.arange(hidden.shape[1], device=hidden.device)
            hidden = hidden * (steps.unsqueeze(0) < valid.unsqueeze(1)).unsqueeze(
                -1).to(hidden.dtype)
        return hidden
