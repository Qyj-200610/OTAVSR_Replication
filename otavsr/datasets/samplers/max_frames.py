# Copyright (c) OTAVSR. All rights reserved.
"""Frame-based batching, the scheme both official baselines use.

``--max-frames-audiovisual 1000`` (Llama-AVSR) and ``max_tokens: 1000``
(MMS-LLaMA) both group *utterances by total frame count*, not by sample count.
With 16 s clips (400 frames at 25 Hz) that is ~2 utterances per GPU, while
short clips can be batched by 8-10. Batching by sample count instead either
wastes memory or triggers OOM at the tail of the length distribution.

``mmengine`` builds a ``batch_sampler`` dict with
``default_args=dict(sampler=..., batch_size=...)``, so this class wraps the
(already sharded and seeded) sampler instead of touching the dataset directly.
"""
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

# Registered in mmengine's DATA_SAMPLERS on purpose: ``build_dataloader``
# resolves ``batch_sampler`` through that registry. ``otavsr.SAMPLERS`` has it
# as parent, so name lookups through the project registry still work.
from mmengine.registry import DATA_SAMPLERS  # isort: skip

__all__ = ['MaxFramesBatchSampler']


@DATA_SAMPLERS.register_module()
class MaxFramesBatchSampler(Sampler):
    """Yield batches whose total number of frames stays below ``max_frames``.

    Args:
        sampler (Sampler): the wrapped sampler (``DefaultSampler``), which
            already applies shuffling and distributed sharding.
        batch_size (int): unused, accepted because mmengine passes it; the
            batch size is decided by ``max_frames``.
        max_frames (int): frame budget of one batch (1000 in the paper).
        drop_last (bool): drop the trailing batch.
        frames_key (str): field of the data list holding the frame count.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 1,
                 max_frames: int = 1000,
                 drop_last: bool = False,
                 frames_key: str = 'num_frames') -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_frames = int(max_frames)
        self.drop_last = drop_last
        dataset = getattr(sampler, 'data_source', None)
        if dataset is None:
            raise RuntimeError(
                'MaxFramesBatchSampler needs a sampler exposing `data_source` '
                '(mmengine DefaultSampler does).')
        data_list = getattr(dataset, 'data_list', None)
        if data_list is None:
            data_list = [
                dataset.get_data_info(index)
                for index in range(len(dataset))
            ]
        self.frames = np.asarray(
            [int(info[frames_key]) for info in data_list], dtype=np.int64)
        self._length: Optional[int] = None

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        total = 0
        for index in self.sampler:
            count = int(self.frames[index])
            if batch and total + count > self.max_frames:
                yield batch
                batch, total = [], 0
            batch.append(int(index))
            total += count
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # The batch count depends on the (shuffled) order, so it is measured
        # once per epoch. ``DefaultSampler`` is deterministic within an epoch,
        # hence the measurement matches the batches actually yielded.
        if self._length is None:
            self._length = sum(1 for _ in self.__iter__())
        return self._length
