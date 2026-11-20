# Copyright (c) OTAVSR. All rights reserved.
"""Length handling of the two frozen encoders.

``inputs['audio_lengths']`` is in **seconds** (see ``AVSRLLMClassifier``): each
backbone must convert it to its own frame rate, otherwise the padding mask of
the visual stream marks almost every frame as padded.
"""
from types import SimpleNamespace

import torch

from otavsr.models.backbones import AVHuBERTEncoder, WhisperEncoder


class _FakeAVH(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.last_mask = None

    def extract_features(self, source, padding_mask=None, mask=False):
        self.last_mask = padding_mask
        video = source['video']
        return video.flatten(2)  # [B, T, C*H*W] -- any feature width works


class _FakeWhisper(torch.nn.Module):

    def forward(self, input_features):
        batch = input_features.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.ones(batch, 1500, 1024))


def test_avhubert_lengths_are_seconds():
    encoder = AVHuBERTEncoder(checkpoint='unused.pt', frame_rate=25)
    fake = _FakeAVH()
    encoder._encoder = fake

    video = torch.zeros(2, 40, 1, 8, 8)
    encoder(video, lengths=torch.tensor([1.6, 0.4]))

    # 1.6 s * 25 Hz = 40 frames (no padding), 0.4 s * 25 Hz = 10 frames
    padded = fake.last_mask.sum(-1).tolist()
    assert padded == [0, 30]


def test_avhubert_explicit_padding_mask_wins():
    encoder = AVHuBERTEncoder(checkpoint='unused.pt')
    fake = _FakeAVH()
    encoder._encoder = fake

    video = torch.zeros(1, 20, 1, 8, 8)
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, 15:] = True
    encoder(video, padding_mask=mask, lengths=torch.tensor([1.0]))
    assert fake.last_mask is mask


def test_whisper_lengths_are_seconds_and_padding_is_zeroed():
    encoder = WhisperEncoder(feature_rate=50)
    encoder._encoder = _FakeWhisper()

    features = encoder(
        torch.randn(2, 80, 3000), lengths=torch.tensor([1.6, 0.4]))
    assert features.shape == (2, 1500, 1024)
    # 1.6 s * 50 Hz = 80 valid frames; everything beyond is zeroed
    assert features[0, :80].abs().sum() > 0
    assert features[0, 80:].abs().sum() == 0
    assert features[1, 20:].abs().sum() == 0


def test_whisper_without_lengths_keeps_everything():
    encoder = WhisperEncoder()
    encoder._encoder = _FakeWhisper()
    features = encoder(torch.randn(1, 80, 3000))
    assert features.shape == (1, 1500, 1024)
