# Copyright (c) OTAVSR. All rights reserved.
"""End-to-end LLM-AVSR model with the optional OT alignment branch.

Composition (paper Fig.1a)::

    x_a -> Whisper(frozen) --Conv1d(s2)--> Proj_a -> z_a \\
                                                         Fusion -> Proj_av -> h_av -> LLaMA(LoRA)
    x_v -> AV-HuBERT(frozen) -----------> Proj_v -> z_v /
                                                        \\
                                                z_t = FC(Embed(target)) --OT--> L_align

The OT branch is a *training-only* side path: it never touches ``h_av``.
"""
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseModule

from otavsr.registry import MODELS

from .base import BaseClassifier

__all__ = ['AVSRLLMClassifier']


@MODELS.register_module()
class AVSRLLMClassifier(BaseClassifier):
    """Audio-visual speech recogniser, mmpretrain ``ImageClassifier`` style.

    Args:
        audio_backbone (dict): config of the acoustic encoder.
        video_backbone (dict, optional): config of the visual encoder. Not
            needed when ``modality='audio'`` (useful for the M0 overfit test).
        head (dict): :class:`~otavsr.models.heads.LLMDecoder` config.
        audio_downsample (dict, optional): 50 Hz -> 25 Hz strided conv.
        audio_projector (dict, optional): ``Proj_a``.
        video_projector (dict, optional): ``Proj_v``.
        fusion (dict, optional): one of the Eq.(12) fusions (includes
            ``Proj_av``). Required when ``modality='audiovisual'``.
        ot_loss (dict, optional): :class:`~otavsr.models.losses.OTAlignmentLoss`
            config. ``None`` reproduces the paper's "no align" baseline.
        modality (str): ``'audiovisual'`` or ``'audio'``.
    """

    def __init__(self,
                 audio_backbone: dict,
                 head: dict,
                 video_backbone: Optional[dict] = None,
                 audio_downsample: Optional[dict] = None,
                 audio_projector: Optional[dict] = None,
                 video_projector: Optional[dict] = None,
                 fusion: Optional[dict] = None,
                 ot_loss: Optional[dict] = None,
                 modality: str = 'audiovisual',
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if modality not in ('audiovisual', 'audio'):
            raise ValueError(f"unsupported modality {modality!r}")
        self.modality = modality

        self.audio_backbone = MODELS.build(audio_backbone)
        self.video_backbone = (MODELS.build(video_backbone)
                               if video_backbone is not None else None)
        self.audio_downsample = (MODELS.build(audio_downsample)
                                 if audio_downsample is not None else None)
        self.audio_projector = (MODELS.build(audio_projector)
                                if audio_projector is not None else None)
        self.video_projector = (MODELS.build(video_projector)
                                if video_projector is not None else None)
        self.fusion = MODELS.build(fusion) if fusion is not None else None
        self.head = MODELS.build(head)
        self.ot_loss = MODELS.build(ot_loss) if ot_loss is not None else None

        if modality == 'audiovisual' and self.fusion is None:
            raise ValueError('`fusion` is required for audiovisual training.')

    # ------------------------------------------------------------------ #
    def _masks(self, data_samples: Sequence, num_frames: int,
               num_tokens: int):
        """Boolean masks for the 25 Hz fused sequence and the target tokens.

        The two masks have different lengths (``num_frames`` vs ``num_tokens``),
        so they need their own index ranges -- mixing them up silently shifts
        the OT reference tokens.
        """
        device = next(self.parameters()).device
        frames = torch.stack([sample.num_frames for sample in data_samples])
        tokens = torch.stack([sample.num_tokens for sample in data_samples])
        frame_steps = torch.arange(num_frames, device=device)
        token_steps = torch.arange(num_tokens, device=device)
        return (frame_steps.unsqueeze(0) < frames.unsqueeze(1),
                token_steps.unsqueeze(0) < tokens.unsqueeze(1))

    def extract_feat(self, inputs, data_samples=None) -> Dict[str, torch.Tensor]:
        """Encode both modalities and the text reference ``z_t``.

        Returns a dict with ``z_a``, ``z_v`` (``None`` in audio-only mode),
        ``z_t``, ``h_av`` and the two masks.
        """
        lengths = inputs.get('audio_lengths')
        hidden_a = self.audio_backbone(inputs['audio'], lengths=lengths)
        if self.audio_downsample is not None:
            hidden_a = self.audio_downsample(hidden_a)
            if lengths is not None:
                lengths = torch.ceil(lengths / 2.0).long()
        z_a = (self.audio_projector(hidden_a)
               if self.audio_projector is not None else hidden_a)

        z_v = None
        if self.video_backbone is not None:
            hidden_v = self.video_backbone(inputs['video'], lengths=lengths)
            z_v = (self.video_projector(hidden_v)
                   if self.video_projector is not None else hidden_v)

        target_ids = torch.stack(
            [sample.target_ids for sample in data_samples])
        z_t = self.head.embed_tokens(target_ids).float()

        src_mask, tgt_mask = self._masks(data_samples, z_a.shape[1],
                                         z_t.shape[1])
        if z_v is None:
            h_av = z_a
        else:
            h_av = self.fusion({'a': z_a, 'v': z_v})
        return dict(
            z_a=z_a,
            z_v=z_v,
            z_t=z_t,
            h_av=h_av,
            src_mask=src_mask,
            tgt_mask=tgt_mask)

    def loss(self, inputs, data_samples=None) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(inputs, data_samples)
        target_ids = torch.stack(
            [sample.target_ids for sample in data_samples])
        losses = self.head(feats['h_av'], target_ids)
        if self.ot_loss is not None:
            loss_a, stats_a = self.ot_loss.align_loss(
                feats['z_a'], feats['z_t'], feats['src_mask'], feats['tgt_mask'])
            if feats['z_v'] is None:
                align = loss_a
                stats = stats_a
            else:
                loss_v, stats_v = self.ot_loss.align_loss(
                    feats['z_v'], feats['z_t'], feats['src_mask'],
                    feats['tgt_mask'])
                align = 0.5 * (loss_a + loss_v)
                stats = {
                    key: 0.5 * (stats_a[key] + stats_v[key])
                    for key in stats_a
                }
            losses['loss_align'] = align * self.ot_loss.loss_weight
            losses.update({f'ot_{key}': value for key, value in stats.items()})
        return losses

    def predict(self, inputs, data_samples=None) -> List[str]:
        feats = self.extract_feat(inputs, data_samples)
        return self.head.generate(feats['h_av'])
