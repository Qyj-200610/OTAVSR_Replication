# Copyright (c) OTAVSR. All rights reserved.
"""Tests of frame-based batching and of the babble-noise mixing."""
import torch

from otavsr.datasets.samplers import MaxFramesBatchSampler
from otavsr.datasets.transforms.noise import mix_at_snr


class _FakeDataset:

    def __init__(self, frames):
        self.data_list = [dict(num_frames=value) for value in frames]

    def __len__(self):
        return len(self.data_list)

    def get_data_info(self, index):
        return self.data_list[index]


class _SerializedFakeDataset:
    """Mimics ``BaseDataset(serialize_data=True)``: ``data_list`` is emptied
    and the samples are only reachable through ``get_data_info``."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.data_list = []

    def __len__(self):
        return len(self._frames)

    def get_data_info(self, index):
        return dict(num_frames=self._frames[index])


class _FakeSampler:

    def __init__(self, dataset, order=None):
        self.data_source = dataset
        self.order = order or list(range(len(dataset)))

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


class TestMaxFramesBatchSampler:

    def test_respects_budget(self):
        frames = [400, 400, 400, 200, 100, 50]
        dataset = _FakeDataset(frames)
        batch_sampler = MaxFramesBatchSampler(
            _FakeSampler(dataset), max_frames=800)
        batches = list(batch_sampler)
        assert sum(len(batch) for batch in batches) == len(frames)
        for batch in batches:
            assert sum(frames[index] for index in batch) <= 800

    def test_short_clips_are_grouped(self):
        dataset = _FakeDataset([100] * 10)
        batch_sampler = MaxFramesBatchSampler(
            _FakeSampler(dataset), max_frames=350)
        batches = list(batch_sampler)
        assert [len(batch) for batch in batches] == [3, 3, 3, 1]

    def test_registered_for_build_dataloader(self):
        from mmengine.registry import DATA_SAMPLERS
        assert 'MaxFramesBatchSampler' in DATA_SAMPLERS.module_dict

    def test_works_on_a_serialized_dataset(self):
        """``serialize_data=True`` empties ``data_list``; reading it directly
        used to yield zero batches (a silently empty epoch)."""
        frames = [400, 200, 100]
        dataset = _SerializedFakeDataset(frames)
        batch_sampler = MaxFramesBatchSampler(
            _FakeSampler(dataset), max_frames=600)
        batches = list(batch_sampler)
        assert sum(len(batch) for batch in batches) == 3
        assert len(batch_sampler) == len(batches)

    def test_length_is_consistent(self):
        dataset = _FakeDataset([400, 200, 100])
        batch_sampler = MaxFramesBatchSampler(
            _FakeSampler(dataset), max_frames=600)
        assert len(batch_sampler) == len(list(batch_sampler))


class TestBabbleNoise:

    def test_mix_hits_the_target_snr(self):
        torch.manual_seed(0)
        speech = torch.randn(16000)
        noise = torch.randn(16000)
        for snr in (-5.0, 0.0, 5.0):
            mixed = mix_at_snr(speech, noise, snr)
            noise_part = mixed - speech
            measured = 10 * torch.log10(
                speech.pow(2).mean() / noise_part.pow(2).mean())
            assert abs(measured.item() - snr) < 0.2

    def test_noise_transform_keeps_the_length(self):
        """The transform must not change the sample count of the waveform."""
        from otavsr.datasets.transforms import AddBabbleNoise
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'noise.wav')
            try:
                import torchaudio
                torchaudio.save(path, torch.randn(1, 16000), 16000)
            except Exception:  # torchaudio missing in this environment
                return
            transform = AddBabbleNoise(
                noise_path=path, eval_snr=0.0, seed=0)
            out = transform.transform(dict(audio=torch.randn(8000)))
            assert out['audio'].shape == (8000, )
            assert out['snr_db'] == 0.0
