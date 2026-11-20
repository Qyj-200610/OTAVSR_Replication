# Copyright (c) OTAVSR. All rights reserved.
"""Loading of the mouth-ROI video and the 16 kHz audio of one utterance."""
from typing import Optional

import numpy as np
import torch

from otavsr.registry import TRANSFORMS

__all__ = ['LoadAudioVideo']

#: frames per second of the LRS3 video, and therefore of the fused sequence
VIDEO_FPS = 25


def _read_video(path: str) -> np.ndarray:
    """Read ``[T, H, W, C]`` uint8 frames, trying the common backends."""
    try:
        import decord
        return decord.VideoReader(path).asnumpy()
    except ImportError:
        pass
    try:
        from torchvision.io import read_video
        return read_video(path, pts_unit='sec')[0].numpy()
    except Exception as error:  # pragma: no cover - env dependent
        raise RuntimeError(
            f'Cannot decode {path!r}: install `decord` or `torchvision` with '
            'video support.') from error


def _read_audio(path: str, sample_rate: int) -> torch.Tensor:
    """Read a mono waveform at ``sample_rate`` Hz."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
    except ImportError:
        try:
            import soundfile
            data, sr = soundfile.read(path)
            wav = torch.from_numpy(data).float().T
        except ImportError as error:  # pragma: no cover - env dependent
            raise RuntimeError(
                f'Cannot read {path!r}: install `torchaudio` or `soundfile`.'
            ) from error
    if wav.dim() > 1:
        wav = wav.mean(0)
    if sr != sample_rate:
        try:
            import torchaudio.functional as AF
            wav = AF.resample(wav, int(sr), sample_rate)
        except Exception:  # pragma: no cover - env dependent
            raise RuntimeError(f'Need torchaudio to resample {sr} -> {sample_rate}')
    return wav


@TRANSFORMS.register_module()
class LoadAudioVideo:
    """Fill ``results['audio']`` (waveform) and ``results['video']`` (frames).

    The video is returned **channel-first** as ``[T, C, H, W]`` with ``C = 1``,
    which is the layout AV-HuBERT/fairseq expects after the batch dimension
    (``[B, T, C, H, W]``). The audio is truncated to the *video* duration, not
    to ``max_frames``: otherwise a clip shorter than 16 s would carry more
    audio than video and the two streams would drift apart.

    Args:
        sample_rate (int): 16 kHz, as in the paper.
        max_frames (int, optional): clip the video to at most this many frames.
            The official pipelines use 16 s segments, i.e. 400 frames at 25 fps.
        video_key (str): key of the decoded frames in ``results``.
        audio_key (str): key of the waveform in ``results``.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 max_frames: Optional[int] = None,
                 video_key: str = 'video',
                 audio_key: str = 'audio') -> None:
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.video_key = video_key
        self.audio_key = audio_key

    def transform(self, results: dict) -> dict:
        frames = _read_video(results['video_path'])
        if self.max_frames is not None:
            frames = frames[:self.max_frames]
        num_frames = len(frames)
        # [T, H, W, C] -> [T, C, H, W]
        video = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
        results[self.video_key] = video

        wav = _read_audio(results['audio_path'], self.sample_rate)
        # keep the two streams temporally consistent: audio follows the video
        max_samples = int(round(num_frames / VIDEO_FPS * self.sample_rate))
        results[self.audio_key] = wav[:max_samples]
        results['num_frames'] = num_frames
        results['audio_lengths'] = num_frames / VIDEO_FPS
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
