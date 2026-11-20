# Copyright (c) OTAVSR. All rights reserved.
"""Log-domain balanced Sinkhorn solvers used by the OT alignment branch.

Design notes (see ``docs/zh_cn/replication_guide.md`` §3.2 / §7):

* the paper's Eq.(7) writes ``K = exp(-S / lam)`` although Eq.(2) maximises
  ``<Q, S>``.  The consistent kernel is ``K = exp(+S / lam)``; we therefore
  always pass a **cost** matrix ``C = 1 - cos`` so that ``K = exp(-C / lam)``.
* padding is killed with the finite sentinel :data:`NEG`.  Using ``-inf`` makes
  a fully masked row produce ``logsumexp = -inf`` and hence ``log(u) = +inf``,
  which turns the whole coupling into ``NaN`` one iteration later.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ['NEG', 'EPS', 'cosine_sim', 'log_sinkhorn', 'solve_ot']

NEG = -1e4
EPS = 1e-12


def cosine_sim(z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two feature sequences in the LLM space.

    Args:
        z_src (Tensor): ``[B, L, d]`` projected acoustic / visual features.
        z_tgt (Tensor): ``[B, Lt, d]`` ``FC(Embed(target))``.

    Returns:
        Tensor: ``[B, L, Lt]`` cosine similarities in ``[-1, 1]``.
    """
    return torch.einsum(
        'bld,bmd->blm',
        F.normalize(z_src, dim=-1),
        F.normalize(z_tgt, dim=-1))


def log_sinkhorn(log_kernel: torch.Tensor,
                 log_p: torch.Tensor,
                 log_q: torch.Tensor,
                 num_iters: int = 50) -> torch.Tensor:
    """Balanced entropy-regularised OT in the log domain.

    Args:
        log_kernel (Tensor): ``[m, n]`` log Gibbs kernel, i.e. ``-cost / lam``.
        log_p (Tensor): ``[m]`` log row marginals.
        log_q (Tensor): ``[n]`` log column marginals.
        num_iters (int): number of scaling-vector updates. 50 iterations leave
            a marginal residual of ~1e-5, which is far below any gradient
            signal the alignment loss carries.

    Returns:
        Tensor: ``[m, n]`` ``log(Q)``.
    """
    log_u = torch.zeros_like(log_p)
    log_v = torch.zeros_like(log_q)
    for _ in range(num_iters):
        log_u = log_p - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_q - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0)
    return log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)


def _margin_logits(mask: torch.Tensor) -> torch.Tensor:
    """Uniform log marginals that put all the mass on the valid entries."""
    weight = mask.to(torch.float32)
    weight = weight / weight.sum().clamp_min(1.0)
    return torch.where(mask, weight, weight.new_full((), EPS)).log()


def solve_ot(z_src: torch.Tensor,
             z_tgt: torch.Tensor,
             src_mask: torch.Tensor = None,
             tgt_mask: torch.Tensor = None,
             lam: float = 0.1,
             s_tilde: float = 0.5,
             rho: float = 0.2,
             bucket_row: bool = False,
             num_iters: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """OT coupling between a modality sequence and the LLM token embeddings.

    The virtual bucket is one extra row and/or column whose similarity is
    ``s_tilde`` and whose total mass is ``rho``.  The marginals of the *real*
    entries are kept at ``(1 - rho) / n`` so that switching the bucket on or
    off does not rescale the alignment loss (the paper does not define these
    marginals, see the guide §3.3.1).

    Args:
        z_src (Tensor): ``[B, L, d]``.
        z_tgt (Tensor): ``[B, Lt, d]``.
        src_mask (Tensor, optional): ``[B, L]`` bool, ``True`` for valid frames.
        tgt_mask (Tensor, optional): ``[B, Lt]`` bool, ``True`` for valid tokens.
        lam (float): entropy regularisation of the OT problem.
        s_tilde (float, optional): similarity margin of the virtual bucket.
            ``None`` disables the bucket entirely.
        rho (float): fraction of the frame mass routed to the bucket,
            ``0 <= rho < 1``. Note that this is a *hard* marginal: exactly
            ``rho`` of the mass is discarded, and ``s_tilde`` only decides
            *which* frames are discarded. That is what Eq.(9) does once the
            augmented marginals are balanced; making the discarded amount
            adaptive requires unbalanced OT instead.
        bucket_row (bool): also add the bucket *row* (Eq.(9) draws both).
        num_iters (int): Sinkhorn iterations.

    Returns:
        Tuple[Tensor, Tensor]: ``Q`` of shape ``[B, L, Lt]`` (the truncated
        block) and the per-frame mass sent to the bucket, shape ``[B, L]``
        (all zeros when the bucket is disabled).
    """
    B, L, _ = z_src.shape
    Lt = z_tgt.shape[1]
    device = z_src.device
    if src_mask is None:
        src_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    if tgt_mask is None:
        tgt_mask = torch.ones(B, Lt, dtype=torch.bool, device=device)

    sim = cosine_sim(z_src, z_tgt)

    use_bucket = s_tilde is not None and rho > 0
    keep = (1.0 - rho) if use_bucket else 1.0
    num_row = L + (1 if (use_bucket and bucket_row) else 0)
    num_col = Lt + (1 if use_bucket else 0)

    sim_aug = sim.new_full((B, num_row, num_col),
                           float(s_tilde) if use_bucket else 0.0)
    sim_aug[:, :L, :Lt] = sim
    log_kernel = -(1.0 - sim_aug) / lam

    row_mask = torch.ones(B, num_row, dtype=torch.bool, device=device)
    row_mask[:, :L] = src_mask
    col_mask = torch.ones(B, num_col, dtype=torch.bool, device=device)
    col_mask[:, :Lt] = tgt_mask
    valid = row_mask.unsqueeze(-1) & col_mask.unsqueeze(1)
    log_kernel = torch.where(valid, log_kernel,
                             log_kernel.new_full((), NEG))

    couplings = []
    dust = []
    for b in range(B):
        log_q = torch.where(tgt_mask[b],
                            sim.new_full((Lt,), keep / tgt_mask[b].sum()),
                            sim.new_full((Lt,), EPS)).log()
        if use_bucket:
            log_q = torch.cat([log_q, sim.new_full((1,), rho).log()])
        row_scale = keep if bucket_row else 1.0
        log_p = torch.where(src_mask[b],
                            sim.new_full((L,), row_scale / src_mask[b].sum()),
                            sim.new_full((L,), EPS)).log()
        if use_bucket and bucket_row:
            log_p = torch.cat([log_p, sim.new_full((1,), rho).log()])

        log_q_mat = log_sinkhorn(log_kernel[b], log_p, log_q, num_iters)
        coupling = log_q_mat.exp()
        if use_bucket:
            row_dust = coupling[:L, Lt] * src_mask[b]
        else:
            row_dust = sim.new_zeros((L, ))
        dust.append(row_dust)
        block = coupling[:L, :Lt]
        block = torch.where(src_mask[b].unsqueeze(1) & tgt_mask[b].unsqueeze(0),
                            block, block.new_zeros(()))
        couplings.append(block)
    return torch.stack(couplings), torch.stack(dust)


def ot_diagnostics(sim: torch.Tensor, coupling: torch.Tensor,
                   dust: torch.Tensor,
                   src_mask: torch.Tensor = None) -> dict:
    """Scalars that should be logged to know whether the OT branch is alive.

    ``bucket_mass`` is pinned to ``rho`` by construction, so it is not a health
    signal by itself; what matters is *which* frames are discarded:

    * ``row_max_cos``: mean ``max_j cos(z_i, z_j^t)``. If it sits well below
      ``s_tilde`` the margin discards informative frames -- the margin is
      mis-scaled for your feature space.
    * ``dusted_row_max_cos``: the same average restricted to the discarded
      frames (weighted by the discarded mass). A bucket that "works" has
      ``dusted_row_max_cos << row_max_cos``; equality means it discards at
      random.
    * ``q_entropy``: if it approaches ``log(Lt)`` the coupling is uniform, i.e.
      there is no alignment at all.
    """
    if src_mask is None:
        src_mask = torch.ones(
            sim.shape[:2], dtype=torch.bool, device=sim.device)
    batch = sim.shape[0]
    row_max = sim.masked_fill(~src_mask.unsqueeze(-1), -1.0).max(-1).values
    num_rows = src_mask.float().sum().clamp_min(1.0)
    entropy = -(coupling.clamp_min(1e-12).log() * coupling).sum((-1, -2))
    dust = dust * src_mask
    dust_total = dust.sum().clamp_min(1e-12)
    return {
        'bucket_mass': (dust.sum() / batch).item(),
        'row_max_cos': ((row_max * src_mask).sum() / num_rows).item(),
        'dusted_row_max_cos': ((row_max * dust).sum() / dust_total).item(),
        'q_rowsum': ((coupling.sum(-1) * src_mask).sum() / num_rows).item(),
        'q_entropy': (entropy / num_rows).mean().item(),
    }
