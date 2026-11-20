# Copyright (c) OTAVSR. All rights reserved.
"""Packing of a transformed sample into ``inputs`` + ``data_samples``."""
import torch

from otavsr.registry import TRANSFORMS
from otavsr.structures import AVSRDataSample

__all__ = ['PackAVSRInputs']


@TRANSFORMS.register_module()
class PackAVSRInputs:
    """Build the ``AVSRDataSample`` and the model inputs.

    ``num_frames`` is the number of 25 Hz frames of the fused sequence: the
    acoustic stream is downsampled by 2 before fusion, so its valid length must
    be exactly ``2 * num_frames`` before the conv layer.
    """

    def __init__(self,
                 meta_keys=('sample_id', 'gt_text', 'num_frames'),
                 to_float32: bool = True) -> None:
        self.meta_keys = meta_keys
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        sample = AVSRDataSample()
        for key in self.meta_keys:
            if key in results:
                sample.set_field(results[key], key)
        sample.sample_id = results.get('sample_id', '')
        sample.gt_text = results.get('gt_text', '')
        sample.num_frames = torch.as_tensor(results['num_frames']).long()

        target_ids = results.get('target_ids')
        if target_ids is None:
            target_ids = torch.empty(0, dtype=torch.long)
        else:
            target_ids = torch.as_tensor(target_ids).long()
        sample.target_ids = target_ids
        sample.num_tokens = torch.as_tensor(target_ids.numel()).long()

        audio = results['audio']
        video = results['video']
        if self.to_float32:
            audio = audio.float()
            video = video.float()
        inputs = dict(audio=audio, video=video)
        if 'audio_lengths' in results:
            inputs['audio_lengths'] = results['audio_lengths']
        return dict(inputs=inputs, data_samples=sample)

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
