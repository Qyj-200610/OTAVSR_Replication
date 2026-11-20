# Copyright (c) OTAVSR. All rights reserved.
"""Pipeline contract: tensor layouts, dtype of the metadata, batch padding."""
import torch

from otavsr.datasets.transforms import (MouthCrop, PackAVSRInputs,
                                        RandomHorizontalFlip)
from otavsr.models.data_preprocessors import AVSRDataPreprocessor
from otavsr.structures import AVSRDataSample


def _video(num_frames=40, size=96):
    # [T, C, H, W] -- what LoadAudioVideo produces
    return torch.rand(num_frames, 1, size, size)


class TestPackAVSRInputs:

    def test_int_num_frames(self):
        """``num_frames`` arrives as a python int from the CSV."""
        results = dict(
            audio=torch.randn(80, 16000),
            video=_video(),
            num_frames=40,
            gt_text='hello world',
            sample_id='lrs3/00001',
            target_ids=torch.tensor([2, 3, 4]))
        out = PackAVSRInputs()(results)

        sample = out['data_samples']
        assert isinstance(sample, AVSRDataSample)
        assert sample.num_frames.dtype == torch.long
        assert sample.num_frames.item() == 40
        assert sample.num_tokens.item() == 3
        assert sample.gt_text == 'hello world'
        assert sample.sample_id == 'lrs3/00001'
        assert out['inputs']['video'].shape == (40, 1, 96, 96)

    def test_lengths_are_forwarded(self):
        results = dict(
            audio=torch.randn(80, 16000),
            video=_video(),
            num_frames=40,
            gt_text='x',
            target_ids=torch.tensor([2]),
            audio_lengths=1.6)
        out = PackAVSRInputs()(results)
        assert out['inputs']['audio_lengths'] == 1.6


class TestVisualTransforms:

    def test_mouth_crop_keeps_channels(self):
        out = MouthCrop(size=88, random=True)(dict(video=_video(size=96)))
        assert out['video'].shape == (40, 1, 88, 88)

    def test_mouth_crop_is_noop_when_already_cropped(self):
        video = _video(size=88)
        out = MouthCrop(size=88, random=True)(dict(video=video))
        assert out['video'] is video

    def test_mouth_crop_rejects_wrong_layout(self):
        try:
            MouthCrop(size=88)(dict(video=torch.rand(4, 96, 96, 1)))
        except ValueError:
            return
        raise AssertionError('expected a ValueError for [T, H, W, C]')

    def test_flip_is_horizontal(self):
        video = torch.zeros(2, 1, 4, 6)
        video[:, :, :, 0] = 1.0  # a stripe on the left
        out = RandomHorizontalFlip(prob=1.0)(dict(video=video))
        flipped = out['video']
        assert flipped.shape == video.shape
        assert flipped[:, :, :, -1].max() == 1.0  # stripe moved to the right


class TestDataPreprocessor:

    def test_pads_variable_length_batch(self):
        samples = []
        for frames in (40, 10):
            sample = AVSRDataSample()
            sample.num_frames = torch.as_tensor(frames).long()
            sample.target_ids = torch.arange(1, 4)
            sample.num_tokens = torch.as_tensor(3).long()
            samples.append(sample)
        data = dict(
            inputs=dict(
                audio=[torch.randn(80, 3200), torch.randn(80, 800)],
                video=[_video(40), _video(10)],
            ),
            data_samples=samples)

        out = AVSRDataPreprocessor()(data, training=False)
        assert out['inputs']['audio'].shape == (2, 80, 3200)
        assert out['inputs']['video'].shape == (2, 40, 1, 96, 96)
        assert out['inputs']['target_ids'].shape == (2, 3)
        # seconds, not frames
        assert torch.allclose(
            out['inputs']['audio_lengths'], torch.tensor([1.6, 0.4]))

    def test_accepts_already_stacked_tensors(self):
        data = dict(
            inputs=dict(
                audio=torch.randn(2, 80, 3200),
                video=_video(40).unsqueeze(0).repeat(2, 1, 1, 1, 1),
                audio_lengths=torch.tensor([1.6, 1.6])),
            data_samples=None)
        out = AVSRDataPreprocessor()(data)
        assert out['inputs']['audio'].shape == (2, 80, 3200)
