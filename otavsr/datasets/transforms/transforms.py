# Copyright (c) OTAVSR. All rights reserved.
"""Visual augmentation and log-mel extraction.

All video tensors are channel-first ``[T, C, H, W]`` (see
:class:`~otavsr.datasets.transforms.loading.LoadAudioVideo`), i.e. spatial
dimensions are ``2:4`` and channels are dimension ``1``.
"""
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
        if video.dim() != 4:
            raise ValueError(
                'MouthCrop expects a channel-first [T, C, H, W] video tensor, '
                f'got shape {tuple(video.shape)}')
        height, width = video.shape[2:4]
        if (height, width) == (self.size, self.size):
            return results
        if self.random:
            top = int(np.random.randint(0, height - self.size + 1))
            left = int(np.random.randint(0, width - self.size + 1))
        else:
            top = (height - self.size) // 2
            left = (width - self.size) // 2
        results['video'] = video[:, :, top:top + self.size,
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
            results['video'] = torch.flip(results['video'], dims=[3])
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)


@TRANSFORMS.register_module()
class WhisperLogMel:
    """Log-mel spectrogram in the shape Whisper's encoder expects.

    Whisper-medium wants 80 mel bins, ``n_fft=400``, ``hop_length=160`` and a
    log10 magnitude with an 8-decade clamp. Extracting the mel here (and not in
    the model) keeps :class:`~otavsr.models.backbones.WhisperEncoder` free of
    audio IO and makes the pipeline cache-friendly.

    NOTE the mel filters here come from ``torchaudio`` (HTK-style, ``norm=None``)
    while the official Whisper checkpoints were trained with librosa's
    slaney-normalised filters. The two differ by a smooth frequency-dependent
    gain of a few dB. If you need bit-exact Whisper features, replace the
    transform with ``transformers.WhisperFeatureExtractor`` -- the paper/repos
    do not specify which one they used.
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
        self._mel = None

    def _get_mel(self):
        if self._mel is None:
            import torchaudio
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels)
        return self._mel

    def transform(self, results: dict) -> dict:
        audio = results['audio']
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        mel = self._get_mel()(audio.unsqueeze(0))
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        # [1, n_mels, T] -> [n_mels, T] : what WhisperEncoder expects
        results['audio'] = ((log_spec + 4.0) / 4.0).squeeze(0)
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
