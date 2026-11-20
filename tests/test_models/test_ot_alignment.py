# Copyright (c) OTAVSR. All rights reserved.
"""Unit tests of the OT alignment conventions (guide §3.2 - §3.4)."""
import torch

from otavsr.models.losses import OTAlignmentLoss
from otavsr.registry import MODELS
from otavsr.utils.sinkhorn import (NEG, cosine_sim, log_sinkhorn,
                                   ot_diagnostics, solve_ot)


def _toy(batch=2, num_frames=40, num_tokens=6, dim=32, seed=0):
    torch.manual_seed(seed)
    z_src = torch.randn(batch, num_frames, dim)
    z_tgt = torch.randn(batch, num_tokens, dim)
    src_mask = torch.ones(batch, num_frames, dtype=torch.bool)
    if batch > 1:
        src_mask[1, 30:] = False
    tgt_mask = torch.ones(batch, num_tokens, dtype=torch.bool)
    return z_src, z_tgt, src_mask, tgt_mask


class TestSinkhorn:

    def test_marginals(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        coupling, _ = solve_ot(z_src, z_tgt, src_mask, tgt_mask, s_tilde=None)
        for b in range(z_src.shape[0]):
            rows = src_mask[b].sum()
            cols = tgt_mask[b].sum()
            assert (coupling[b][src_mask[b]].sum(-1) - 1.0 / rows).abs().max() < 1e-4
            assert (coupling[b][:, tgt_mask[b]].sum(0) -
                    1.0 / cols).abs().max() < 1e-4

    def test_padding_carries_no_mass(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        coupling, _ = solve_ot(z_src, z_tgt, src_mask, tgt_mask)
        assert coupling[~src_mask].abs().max() < 1e-8
        assert torch.isfinite(coupling).all()

    def test_kernel_sign(self):
        """The literal Eq.(7) gives an anti-aligned coupling."""
        z_src, z_tgt, _, _ = _toy(batch=1)
        sim = cosine_sim(z_src, z_tgt)[0]
        log_p = torch.full((sim.shape[0], ), 1.0 / sim.shape[0]).log()
        log_q = torch.full((sim.shape[1], ), 1.0 / sim.shape[1]).log()
        wrong = log_sinkhorn(-sim / 0.1, log_p, log_q).exp()
        right = log_sinkhorn(-(1.0 - sim) / 0.1, log_p, log_q).exp()
        corr_wrong = torch.corrcoef(
            torch.stack([wrong.flatten(), sim.flatten()]))[0, 1]
        corr_right = torch.corrcoef(
            torch.stack([right.flatten(), sim.flatten()]))[0, 1]
        assert corr_wrong < 0
        assert corr_right > 0

    def test_cost_shift_invariance(self):
        z_src, z_tgt, _, _ = _toy(batch=1)
        sim = cosine_sim(z_src, z_tgt)[0]
        log_p = torch.full((sim.shape[0], ), 1.0 / sim.shape[0]).log()
        log_q = torch.full((sim.shape[1], ), 1.0 / sim.shape[1]).log()
        q_one_minus_cos = log_sinkhorn(-(1.0 - sim) / 0.1, log_p, log_q).exp()
        q_neg_cos = log_sinkhorn(-(-sim) / 0.1, log_p, log_q).exp()
        assert (q_one_minus_cos - q_neg_cos).abs().max() < 1e-6

    def test_bucket_budget(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        for s_tilde in (0.1, 0.5, 0.9):
            coupling, dust = solve_ot(
                z_src, z_tgt, src_mask, tgt_mask, s_tilde=s_tilde, rho=0.2)
            # rho is a hard marginal: exactly rho of the mass is discarded
            assert dust.shape == src_mask.shape
            assert (dust.sum() / z_src.shape[0] - 0.2).abs() < 1e-4
            upper = 1.0 / src_mask.sum(-1)
            assert (coupling.sum(-1) <= upper.unsqueeze(-1) + 1e-5).all()

    def test_inf_sentinel_breaks_the_coupling(self):
        """Why NEG and not -inf: a fully masked row yields NaN."""
        z_src, z_tgt, _, _ = _toy(batch=1)
        sim = cosine_sim(z_src, z_tgt)[0]
        log_p = torch.full((sim.shape[0], ), 1.0 / sim.shape[0]).log()
        log_q = torch.full((sim.shape[1], ), 1.0 / sim.shape[1]).log()
        bad = -(1.0 - sim) / 0.1
        bad[30:] = float('-inf')
        assert torch.isnan(log_sinkhorn(bad, log_p, log_q)).any()
        good = -(1.0 - sim) / 0.1
        good[30:] = NEG
        assert not torch.isnan(log_sinkhorn(good, log_p, log_q)).any()


class TestOTAlignmentLoss:

    def test_registered_and_forward(self):
        assert 'OTAlignmentLoss' in MODELS.module_dict
        loss_module = OTAlignmentLoss(lam=0.1, tau=0.07, s_tilde=0.5)
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        out = loss_module(z_src, z_tgt, src_mask, tgt_mask)
        assert set(out) == {'loss_align'}
        assert torch.isfinite(out['loss_align'])
        assert out['loss_align'] >= 0

    def test_gradient_reaches_the_features(self):
        loss_module = OTAlignmentLoss(lam=0.1, tau=0.07, s_tilde=0.5)
        z_src = torch.randn(1, 40, 32, requires_grad=True)
        z_tgt = torch.randn(1, 6, 32)
        loss = loss_module(z_src, z_tgt)['loss_align']
        loss.backward()
        assert z_src.grad is not None
        assert torch.isfinite(z_src.grad).all()
        assert z_src.grad.abs().sum() > 0

    def test_loss_weight_scales_the_loss(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        base = OTAlignmentLoss(loss_weight=0.1)
        double = OTAlignmentLoss(loss_weight=0.2)
        value = base.align_loss(z_src, z_tgt, src_mask, tgt_mask)[0]
        assert torch.allclose(
            base(z_src, z_tgt, src_mask, tgt_mask)['loss_align'], value * 0.1)
        assert torch.allclose(
            double(z_src, z_tgt, src_mask, tgt_mask)['loss_align'], value * 0.2)

    def test_reduction_conventions(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy(batch=1, num_frames=40)
        summed = OTAlignmentLoss(reduction='sum').align_loss(
            z_src, z_tgt, src_mask, tgt_mask)[0]
        mean = OTAlignmentLoss(reduction='mean').align_loss(
            z_src, z_tgt, src_mask, tgt_mask)[0]
        assert summed > mean * 10  # sum is O(L) larger than the mean

    def test_no_bucket_has_no_dust(self):
        z_src, z_tgt, src_mask, tgt_mask = _toy()
        coupling, dust = solve_ot(
            z_src, z_tgt, src_mask, tgt_mask, s_tilde=None)
        assert torch.allclose(dust, torch.zeros_like(dust))
        stats = ot_diagnostics(
            cosine_sim(z_src, z_tgt), coupling, dust, src_mask)
        assert stats['bucket_mass'] == 0.0
        assert stats['q_rowsum'] > 0.0