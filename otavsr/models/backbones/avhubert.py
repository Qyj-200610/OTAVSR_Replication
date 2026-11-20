# Copyright (c) OTAVSR. All rights reserved.
"""Frozen AV-HuBERT encoder (paper §II-A).

``large_vox_iter5.pt`` is the *pretraining* checkpoint released by
``facebookresearch/av_hubert`` (LRS3 + VoxCeleb2, 24 layers, 1024 dim, 16
heads, 25 Hz).  It is loaded through fairseq, which is also why the whole
reproduction is pinned to ``python 3.9 + torch 2.1.x``.

Running this module requires the upstream fairseq build::

    cd third_party/av_hubert && pip install --editable ./fairseq

TODO(replication): verify the ``extract_features`` signature of your own
av_hubert checkout. Pretraining checkpoints expose
``AVHubertPretrainModel.extract_features(source, padding_mask, mask=False)``;
the finetuned ``AVHubertSeq2Seq`` uses the same name but a slightly different
signature. Only the *encoder* output is used here, never the CTC head.
"""
from typing import Optional

import torch
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['AVHuBERTEncoder']

#: default frame rate of AV-HuBERT features (its label rate)
AVHUBERT_FRAME_RATE = 25


@MODELS.register_module()
class AVHuBERTEncoder(BaseModule):
    """Wrapper around the fairseq AV-HuBERT model.

    Args:
        checkpoint (str): path to ``large_vox_iter5.pt``.
        frozen (bool): freeze the encoder (paper: frozen).
        frame_rate (int): 25 Hz.
        use_audio (bool): AV-HuBERT can consume audio too; the paper uses the
            visual stream only, so this stays ``False``.
    """

    def __init__(self,
                 checkpoint: str = 'pretrained_models/avhubert/large_vox_iter5.pt',
                 frozen: bool = True,
                 frame_rate: int = AVHUBERT_FRAME_RATE,
                 use_audio: bool = False,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.frame_rate = frame_rate
        self.use_audio = use_audio
        self._encoder = None

    @property
    def out_channels(self) -> int:
        """1024 for AV-HuBERT-large."""
        return 1024

    def _build(self):
        try:
            from fairseq import checkpoint_utils
        except ImportError as error:  # pragma: no cover - env dependent
            raise ImportError(
                'AVHuBERTEncoder needs fairseq. Build the upstream submodule '
                'first: `cd third_party/av_hubert && pip install --editable '
                './fairseq` (python 3.9 / torch 2.1 recommended).') from error
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [self.checkpoint])
        encoder = models[0]
        encoder.requires_grad_(not self.frozen)
        encoder.eval() if self.frozen else encoder.train()
        return encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self._build()
        return self._encoder

    def forward(self,
                video: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            video (Tensor): ``[B, T, 1, 88, 88]`` mouth crops (88 is the crop
                size the two official baselines use for AV-HuBERT).
            padding_mask (Tensor, optional): ``[B, T]``, ``True`` = padded.
            lengths (Tensor, optional): number of valid frames, ``[B]``;
                converted to ``padding_mask`` when given.

        Returns:
            Tensor: ``[B, T', 1024]`` features at 25 Hz.
        """
        if padding_mask is None and lengths is not None:
            steps = torch.arange(video.shape[1], device=video.device)
            padding_mask = steps.unsqueeze(0) >= lengths.unsqueeze(1)
        source = {'video': video, 'audio': None}
        features = self.encoder.extract_features(
            source=source, padding_mask=padding_mask, mask=False)
        if isinstance(features, tuple):  # older checkpoints return (x, ...)
            features = features[0]
        if isinstance(features, dict):
            features = features['x']
        return features
