#!/usr/bin/env python
# Copyright (c) OTAVSR. All rights reserved.
"""Numeric checks of the OT conventions against the paper (guide §3.2 / §3.3.1).

Run ``python tools/analysis_tools/check_ot_numerics.py``. Every block prints a
number and the expected qualitative behaviour, so it doubles as documentation
of the four decisions the paper leaves open:

1. the sign of the Gibbs kernel (Eq.(7) is inconsistent with Eq.(2));
2. ``cost = 1 - cos`` vs ``cost = -cos`` (a constant shift, so identical);
3. how the virtual bucket marginals change the absorbed mass;
4. masked rows, and why ``-inf`` must not be used.
"""
import os.path as _osp
import sys as _sys

_root = _osp.dirname(_osp.abspath(__file__))
while not _osp.isdir(_osp.join(_root, 'otavsr')) and _osp.dirname(_root) != _root:
    _root = _osp.dirname(_root)
_sys.path.insert(0, _root)
import torch

from otavsr.utils.sinkhorn import (NEG, cosine_sim, log_sinkhorn,
                                   ot_diagnostics, solve_ot)


def block(title):
    print('\n' + '=' * 72)
    print(title)


def main():
    torch.manual_seed(0)
    num_frames, num_tokens, dim = 200, 30, 64
    z_src = torch.randn(1, num_frames, dim)
    z_tgt = torch.randn(1, num_tokens, dim)
    sim = cosine_sim(z_src, z_tgt)[0]
    lam = 0.1

    log_p = torch.full((num_frames,), 1.0 / num_frames).log()
    log_q = torch.full((num_tokens,), 1.0 / num_tokens).log()

    block('[1] Eq.(7) as written  vs  cost = 1 - cos')
    q_literal = log_sinkhorn(-sim / lam, log_p, log_q).exp()
    q_fixed = log_sinkhorn(-(1.0 - sim) / lam, log_p, log_q).exp()
    corr_literal = torch.corrcoef(
        torch.stack([q_literal.flatten(), sim.flatten()]))[0, 1].item()
    corr_fixed = torch.corrcoef(
        torch.stack([q_fixed.flatten(), sim.flatten()]))[0, 1].item()
    print(f'  corr(Q, S) with K=exp(-S/lam)    : {corr_literal:+.4f}  <- anti-alignment')
    print(f'  corr(Q, S) with K=exp(-(1-S)/lam) : {corr_fixed:+.4f}  <- correct')

    block('[2] 1 - cos and -cos must give the same coupling')
    q_neg = log_sinkhorn(-(-sim) / lam, log_p, log_q).exp()
    print(f'  max |Q(1-cos) - Q(-cos)| = {(q_fixed - q_neg).abs().max():.2e}')

    block('[3] virtual bucket marginals: the paper defines S~ but not p~/q~')
    print('  recipe A (bucket row AND column, equal mass 1/(n+1)):')
    for s_tilde in (0.9, 0.5, 0.1):
        sim_aug = sim.new_full((num_frames + 1, num_tokens + 1), s_tilde)
        sim_aug[:num_frames, :num_tokens] = sim
        log_p_a = sim.new_full((num_frames + 1,), 1.0 / (num_frames + 1)).log()
        log_q_a = sim.new_full((num_tokens + 1,), 1.0 / (num_tokens + 1)).log()
        q_full = log_sinkhorn(-(1.0 - sim_aug) / lam, log_p_a, log_q_a).exp()
        absorbed = 1.0 - q_full[:num_frames, :num_tokens].sum().item()
        print(f'    s_tilde={s_tilde:<4} absorbed={absorbed * 100:5.1f}%  '
              '(pinned by the marginals: s_tilde does not matter)')

    print('  recipe B (bucket column, hard marginal rho=0.2) -- shipped default:')
    print('  rho pins HOW MUCH is discarded; s_tilde decides WHICH frames.')
    print('  A healthy bucket discards low-similarity frames only, i.e.')
    print('  dusted_row_max_cos << row_max_cos.')
    z_corr = z_src[:, :num_tokens] + 0.08 * torch.randn(1, num_tokens, dim)
    for tag, src, tgt in (('random', z_src, z_tgt),
                          ('correlated', z_src, z_corr)):
        for s_tilde in (0.1, 0.5, 0.9):
            coupling, dust = solve_ot(
                src, tgt, lam=lam, s_tilde=s_tilde, rho=0.2, num_iters=200)
            stats = ot_diagnostics(cosine_sim(src, tgt), coupling, dust)
            print(f'    {tag:<10} s_tilde={s_tilde:<4} '
                  f'bucket_mass={stats["bucket_mass"]:.3f} '
                  f'row_max={stats["row_max_cos"]:.3f} '
                  f'dusted={stats["dusted_row_max_cos"]:.3f}')

    block('[4] padding: finite sentinel vs -inf')
    mask = torch.ones(num_frames, dtype=torch.bool)
    mask[180:] = False
    log_kernel = -(1.0 - sim) / lam
    log_kernel[~mask] = NEG
    log_p_masked = torch.where(
        mask, torch.full((num_frames,), 1.0 / mask.sum()),
        torch.zeros(num_frames)).clamp_min(1e-30).log()
    q_ok = log_sinkhorn(log_kernel, log_p_masked, log_q).exp()

    bad_kernel = -(1.0 - sim) / lam
    bad_kernel[~mask] = -float('inf')
    q_bad = log_sinkhorn(bad_kernel, log_p_masked, log_q).exp()
    print(f'  padded rows carry {q_ok[~mask].sum():.2e} of the mass, '
          f'NaN={bool(torch.isnan(q_ok).any())}')
    print(f'  with -inf: NaN={bool(torch.isnan(q_bad).any())}  '
          '<- one fully masked row destroys the coupling')

    block('[5] alignment loss scale: sum (paper Eq.10) vs per-frame mean')
    coupling, _ = solve_ot(z_src, z_tgt, lam=0.1, s_tilde=0.5, rho=0.2)
    log_prob = torch.log_softmax(cosine_sim(z_src, z_tgt) / 0.07, dim=-1)
    per_frame = -(coupling * log_prob).sum(-1)
    print('  sum over i, j (paper Eq.10, reduction="sum") : '
          f'{per_frame.sum().item():.4f} per sample')
    print('  per-frame mean (reduction="mean")            : '
          f'{per_frame.mean().item():.4f}')
    print('  => alpha=0.1 is meaningful only with the sum, where L_align is')
    print('     on the same order as the AR loss (~1-3). The two conventions')
    print('     are equivalent up to alpha; the gradient is (P-Q)/tau anyway.')


if __name__ == '__main__':
    main()
