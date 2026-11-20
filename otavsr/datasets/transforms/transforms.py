# Copyright (c) OTAVSR. All rights reserved.
"""Visual augmentation and log-mel extraction."""
import numpy as np
import torch

from otavsr.registry import TRANSFORMS

__all__ = ['MouthCrop', 'RandomHorizontalFlip', 'WhisperLogMel']


@TRANSFORMS.register_module()
class MouthCrop:
    """Random (train) or centre (test) crop of the mouth ROI.

    auto-avsr delivers 96x96 crops; both official baselines feed AV-HuBERT
    88x88, so a crop of ``size=88`` is the paper's setting.

    Args:
        size (int): crop size, 88.
        random (bool): ``True`` for training.
    """

    def __init__(self, size: int = 88, random: bool = True) -> None:
        self.size = size
        self.random = random

    def transform(self, results: dict) -> dict:
        video = results['video']
        height, width = video.shape[1:3]
        if (height, width) == (self.size, self.size):
            return results
        if self.random:
            top = int(np.random.randint(0, height - self.size + 1))
            left = int(np.random.randint(0, width - self.size + 1))
        else:
            top = (height - self.size) // 2
            left = (width - self.size) // 2
        results['video'] = video[:, top:top + self.size,
                                 left:left + self.size]
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)


@TRANSFORMS.register_module()
class RandomHorizontalFlip:
    """Horizontal flip with probability ``prob`` (train only)."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.prob:
            results['video'] = torch.flip(results['video'], dims=[2])
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)


@TRANSFORMS.register_module()
class WhisperLogMel:
    """Log-mel spectrogram identical to ``WhisperFeatureExtractor``.

    Whisper-medium expects 80 mel bins, ``n_fft=400``, ``hop_length=160`` and a
    log10 magnitude with an 8-decade clamp. Extracting the mel here (and not in
    the model) keeps :class:`~otavsr.models.backbones.WhisperEncoder` free of
    audio IO and makes the pipeline cache-friendly.
    """

    def __init__(self,
                 n_mels: int = 80,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 hop_length: int = 160) -> None:
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def transform(self, results: dict) -> dict:
        try:
            import torchaudio
        except ImportError as error:  # pragma: no cover - env dependent
            raise ImportError('WhisperLogMel needs `torchaudio`.') from error
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels)(results['audio'].unsqueeze(0))
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        results['audio'] = (log_spec + 4.0) / 4.0
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
