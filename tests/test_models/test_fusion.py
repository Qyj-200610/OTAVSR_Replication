# Copyright (c) OTAVSR. All rights reserved.
"""Shape tests of the fusion necks and of the full classifier wiring."""
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.models.classifiers import AVSRLLMClassifier
from otavsr.models.necks import (AddFusion, ChannelConcatFusion,
                                 CrossAttentionFusion, QFormerFusion,
                                 TemporalConcatFusion)
from otavsr.registry import MODELS
from otavsr.structures import AVSRDataSample

FEATURES = dict(a=torch.randn(2, 10, 8), v=torch.randn(2, 10, 8))


class TestFusion:

    def test_channel_concat(self):
        out = ChannelConcatFusion(in_channels=8, out_channels=4)(FEATURES)
        assert out.shape == (2, 10, 4)

    def test_add(self):
        out = AddFusion(in_channels=8, out_channels=4)(FEATURES)
        assert out.shape == (2, 10, 4)

    def test_cross_att(self):
        out = CrossAttentionFusion(in_channels=8, out_channels=4,
                                   num_heads=2)(FEATURES)
        assert out.shape == (2, 10, 4)

    def test_q_former_pools_to_queries(self):
        fusion = QFormerFusion(
            in_channels=8,
            out_channels=4,
            queries_per_sec=3.0,
            frame_rate=25.0,
            num_layers=1,
            num_heads=2,
            max_queries=32)
        out = fusion(FEATURES)
        # 10 frames at 25 Hz = 0.4 s -> 1.2 -> 1 query
        assert out.shape == (2, 1, 4)
        assert fusion.num_queries(250) == 30

    def test_temporal_concat(self):
        out = TemporalConcatFusion(in_channels=8, out_channels=4)(FEATURES)
        assert out.shape == (2, 20, 4)

    def test_temporal_downsample_halves_length(self):
        from otavsr.models.necks import TemporalDownsample
        out = TemporalDownsample(channels=8)(torch.randn(2, 11, 8))
        assert out.shape == (2, 6, 8)


class _MockBackbone(BaseModule):
    """Accepts the ``(x, lengths=...)`` contract of both real encoders."""

    def __init__(self):
        super().__init__()
        self.out_channels = 8

    def forward(self, x, lengths=None):
        batch, num_frames = x.shape[:2]
        return torch.randn(batch, num_frames, 8)


class _MockHead(BaseModule):
    """Minimal stand-in for LLMDecoder (no transformers needed)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.linear = nn.Linear(8, 16)

    def embed_tokens(self, ids):
        return self.embed(ids)

    def forward(self, av_embeds, target_ids):
        logits = self.linear(av_embeds[:, :target_ids.shape[1]])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 16), target_ids.reshape(-1))
        return {'loss_ar': loss}

    def generate(self, av_embeds):
        return [''] * av_embeds.shape[0]


MODELS.register_module(module=_MockBackbone, force=True)
MODELS.register_module(module=_MockHead, force=True)


def _samples(batch=2, num_frames=40, num_tokens=3):
    samples = []
    for _ in range(batch):
        sample = AVSRDataSample()
        sample.num_frames = torch.as_tensor(num_frames).long()
        sample.target_ids = torch.randint(1, 15, (num_tokens, ))
        sample.num_tokens = torch.as_tensor(num_tokens).long()
        sample.gt_text = 'hello world'
        samples.append(sample)
    return samples


class TestClassifierWiring:

    def setup_method(self):
        self.model = AVSRLLMClassifier(
            modality='audiovisual',
            audio_backbone=dict(type='_MockBackbone'),
            video_backbone=dict(type='_MockBackbone'),
            audio_downsample=dict(type='TemporalDownsample', channels=8),
            audio_projector=dict(
                type='ModalityProjector', in_channels=8, out_channels=8),
            video_projector=dict(
                type='ModalityProjector', in_channels=8, out_channels=8),
            fusion=dict(
                type='ChannelConcatFusion', in_channels=8, out_channels=8),
            head=dict(type='_MockHead'),
            ot_loss=dict(
                type='OTAlignmentLoss',
                lam=0.1,
                tau=0.07,
                s_tilde=None,
                rho=0.0,
                sinkhorn_iters=10,
                loss_weight=0.1))

    def test_loss_dict_and_parse(self):
        samples = _samples()
        inputs = dict(
            audio=torch.randn(2, 80, 120), video=torch.randn(2, 40, 1, 88, 88))
        losses = self.model.loss(inputs, samples)
        assert 'loss_ar' in losses
        assert 'loss_align' in losses
        assert torch.isfinite(losses['loss_ar'])
        # mmengine requires tensors only; diagnostics must be split out
        total, log_vars = self.model.parse_losses(losses)
        assert {'loss', 'loss_ar', 'loss_align'} <= set(log_vars)
        assert self.model._ot_stats  # diagnostics stashed for the hook
        total.backward()

    def test_streams_of_different_length_are_cropped(self):
        """The mel front-end can leave the acoustic branch one frame longer;
        fusing without cropping raises a cat/attention shape error."""
        samples = _samples(num_frames=40)
        # audio: 802 frames -> downsample -> 401; video: 400 frames
        inputs = dict(
            audio=torch.randn(2, 802, 8), video=torch.randn(2, 40, 1, 88, 88))
        feats = self.model.extract_feat(inputs, samples)
        assert feats['z_a'].shape[1] == feats['z_v'].shape[1] == 40
        losses = self.model.loss(inputs, samples)
        assert torch.isfinite(losses['loss_ar'])

    def test_audio_only_skips_fusion(self):
        model = AVSRLLMClassifier(
            modality='audio',
            audio_backbone=dict(type='_MockBackbone'),
            video_backbone=None,
            video_projector=None,
            fusion=None,
            audio_downsample=dict(type='TemporalDownsample', channels=8),
            audio_projector=dict(
                type='ModalityProjector', in_channels=8, out_channels=8),
            head=dict(type='_MockHead'),
            ot_loss=None)
        samples = _samples()
        inputs = dict(
            audio=torch.randn(2, 80, 120), video=torch.zeros(2, 40, 1, 88, 88))
        losses = model.loss(inputs, samples)
        assert set(losses) == {'loss_ar'}

    def test_predict_returns_metric_ready_dicts(self):
        """``WordErrorRate.process`` reads ``sample['pred_text']``; a bare list
        of strings raises ``'str' object has no attribute 'get'``."""
        from otavsr.evaluation.metrics import WordErrorRate
        samples = _samples()
        inputs = dict(
            audio=torch.randn(2, 80, 120), video=torch.randn(2, 40, 1, 88, 88))
        predictions = self.model.predict(inputs, samples)

        assert isinstance(predictions, list) and len(predictions) == 2
        assert set(predictions[0]) >= {'pred_text', 'gt_text'}
        assert predictions[0]['gt_text'] == 'hello world'

        metric = WordErrorRate()
        metric.process(None, predictions)
        assert metric.compute_metrics(metric.results)['wer'] >= 0.0

    def test_requires_fusion_for_audiovisual(self):
        try:
            AVSRLLMClassifier(
                modality='audiovisual',
                audio_backbone=dict(type='_MockBackbone'),
                head=dict(type='_MockHead'),
                fusion=None)
        except ValueError:
            return
        raise AssertionError('expected a ValueError')
