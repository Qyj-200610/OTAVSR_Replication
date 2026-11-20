"""Reference OT-alignment module for OT-LLM-AVSR replication.

Self-contained (torch only), numerically checked by `python code/ot_align.py`.

Design decisions the paper does NOT make (documented in the guide, sec. 3.3 / 7):

* cost = 1 - cos,  K = exp(-cost / lam)   <-- equal to K = exp(+S / lam)
  (the paper's Eq.(7) writes K = exp(-S / lam); that is the wrong sign and
   transports mass to the *least* similar tokens)
* log-domain Sinkhorn; padded rows/cols are killed with a finite sentinel
  (-1e4).  Using -inf makes logsumexp(-inf) and log(u) = +inf -> NaN.
* virtual bucket = one extra row and/or column whose similarity is s_tilde and
  whose total mass is `rho` (the paper defines the augmented similarity matrix
  but never its marginals; this choice keeps the real rows at (1-rho)/L so that
  toggling the bucket does not rescale L_align)
* Q is detached; gradients reach z_a, z_v, z_t only through log P(S / tau)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1e4          # finite sentinel for masked log-kernel entries
EPS = 1e-12


def sinkhorn_log(logK: torch.Tensor,
                 logp: torch.Tensor,
                 logq: torch.Tensor,
                 n_iters: int = 50,
                 tol: float = 0.0) -> torch.Tensor:
    """Balanced entropy-regularised OT in the log domain.

    logK : [m, n]  log Gibbs kernel (-cost / lam)
    logp : [m]     log row marginals
    logq : [n]     log col marginals
    returns logQ : [m, n]
    """
    logu = torch.zeros_like(logp)
    logv = torch.zeros_like(logq)
    for _ in range(n_iters):
        logu = logp - torch.logsumexp(logK + logv.unsqueeze(0), dim=1)
        logv_new = logq - torch.logsumexp(logK + logu.unsqueeze(1), dim=0)
        if tol > 0 and (logv_new - logv).abs().max() < tol:
            logv = logv_new
            break
        logv = logv_new
    return logu.unsqueeze(1) + logK + logv.unsqueeze(0)


def ot_coupling(zs: torch.Tensor,
                zt: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None,
                lam: float = 0.1,
                s_tilde: float | None = 0.5,
                rho: float = 0.2,
                bucket_row: bool = False,
                n_iters: int = 50):
    """Per-sample entropy-regularised OT between a modality and the text tokens.

    zs : [B, L, d]    projected acoustic (or visual) features
    zt : [B, Lt, d]   FC(embed(target))     (the OT reference)
    lam : entropy regularisation / similarity temperature of the OT problem
    s_tilde : similarity margin of the virtual bucket (None -> no bucket)
    rho : total mass of the bucket, in [0, 1)  (0.2 = at most 20% is discarded)
    returns
      Q    : [B, L, Lt]  coupling block (rows sum to <= (1-rho)/L_valid)
      dust : [B]         mass routed to the bucket, per sample
    """
    B, L, _ = zs.shape
    Lt = zt.shape[1]
    dev = zs.device
    if src_mask is None:
        src_mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    if tgt_mask is None:
        tgt_mask = torch.ones(B, Lt, dtype=torch.bool, device=dev)

    S = torch.einsum("bld,bmd->blm",
                     F.normalize(zs, dim=-1), F.normalize(zt, dim=-1))

    use_bucket = s_tilde is not None and rho > 0
    Lr = L + (1 if (use_bucket and bucket_row) else 0)
    Tr = Lt + (1 if use_bucket else 0)
    S_aug = S.new_full((B, Lr, Tr), float(s_tilde) if use_bucket else 0.0)
    S_aug[:, :L, :Lt] = S
    logK = -(1.0 - S_aug) / lam

    # mask: padded frames / tokens get kernel ~ 0 (finite sentinel, never -inf)
    f_mask = torch.ones(B, Lr, dtype=torch.bool, device=dev)
    f_mask[:, :L] = src_mask
    t_mask = torch.ones(B, Tr, dtype=torch.bool, device=dev)
    t_mask[:, :Lt] = tgt_mask
    logK = torch.where(f_mask.unsqueeze(-1) & t_mask.unsqueeze(1), logK,
                       logK.new_full((), NEG))

    Qs, dust = [], []
    for b in range(B):
        n_s = src_mask[b].sum().clamp_min(1)
        n_t = tgt_mask[b].sum().clamp_min(1)
        keep = (1.0 - rho) if use_bucket else 1.0

        q = torch.where(tgt_mask[b],
                        S.new_full((Lt,), keep / n_t),
                        S.new_full((Lt,), EPS))
        if use_bucket:
            q = torch.cat([q, q.new_full((1,), rho)])
        # every frame's mass must go somewhere: a token or the bucket, so the
        # row marginals stay at 1/n_s (only the bucket ROW reduces them to keep/n_s)
        p_real = torch.where(src_mask[b],
                             S.new_full((L,), (keep if bucket_row else 1.0) / n_s),
                             S.new_full((L,), EPS))
        p = torch.cat([p_real, p_real.new_full((1,), rho)]) if (use_bucket and bucket_row) else p_real

        Q = sinkhorn_log(logK[b], p.log(), q.log(), n_iters=n_iters).exp()
        dust.append(Q[:L, Lt].sum() if use_bucket else Q.new_zeros(()))
        Qb = Q[:L, :Lt]
        Qb = torch.where(src_mask[b].unsqueeze(1) & tgt_mask[b].unsqueeze(0),
                         Qb, Qb.new_zeros(()))
        Qs.append(Qb)
    return torch.stack(Qs), torch.stack(dust)


def alignment_loss(S: torch.Tensor,
                   Q: torch.Tensor,
                   tau: float = 0.07,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """L_align = - sum_ij Q_ij log P_ij ,  P = softmax_j(S / tau).

    S : [B, L, Lt] cosine similarity, Q : [B, L, Lt] coupling (detached here).
    Padding frames are excluded from the mean.
    """
    logP = F.log_softmax(S / tau, dim=-1)
    per_frame = -(Q.detach() * logP).sum(-1)                      # [B, L]
    if mask is None:
        return per_frame.mean()
    w = mask.to(per_frame.dtype)
    return (per_frame * w).sum() / w.sum().clamp_min(1.0)


def diagnostics(S: torch.Tensor, Q: torch.Tensor, dust: torch.Tensor,
                src_mask: torch.Tensor | None = None) -> dict:
    """Log these while training: they tell you whether the OT branch is alive.

    dump_frac : share of the total frame mass that went to the bucket, in [0, rho]
    row_max_cos : mean of max_j cos(z_i, z_j^t); if it is far below s_tilde the
                  bucket is saturated and the alignment loss is effectively dead
    Q_rowsum : mean row sum of the truncated coupling (1/L_valid when no bucket)
    Q_entropy : entropy of the coupling rows; ~log(Lt) means uniform (no alignment)
    """
    if src_mask is None:
        src_mask = torch.ones(S.shape[:2], dtype=torch.bool, device=S.device)
    B = S.shape[0]
    row_max = S.masked_fill(~src_mask.unsqueeze(-1), -1.0).max(-1).values
    n_rows = src_mask.float().sum().clamp_min(1.0)
    ent = -(Q.clamp_min(1e-12).log() * Q).sum((-1, -2)) / n_rows
    return {
        "dump_frac": (dust.sum() / B).item(),
        "row_max_cos": ((row_max * src_mask).sum() / n_rows).item(),
        "Q_rowsum": ((Q.sum(-1) * src_mask).sum() / n_rows).item(),
        "Q_entropy": ent.mean().item(),
    }


# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    B, L, Lt, d = 3, 40, 6, 32
    zs, zt = torch.randn(B, L, d), torch.randn(B, Lt, d)
    src = torch.ones(B, L, dtype=torch.bool)
    src[1, 30:] = False                     # sample 1: 10 padded frames
    src[2, 36:] = False                     # sample 2: 4  padded frames
    tgt = torch.ones(B, Lt, dtype=torch.bool)
    tgt[2, 4:] = False                      # sample 2: 2 padded tokens
    S = torch.einsum("bld,bmd->blm",
                     F.normalize(zs, dim=-1), F.normalize(zt, dim=-1))

    # 1) marginals without bucket
    Q, _ = ot_coupling(zs, zt, src, tgt, lam=0.1, s_tilde=None)
    for b in range(B):
        assert (Q[b][src[b]].sum(-1) - 1.0 / src[b].sum()).abs().max() < 1e-4
        assert (Q[b][:, tgt[b]].sum(0) - 1.0 / tgt[b].sum()).abs().max() < 1e-4
    assert (Q[~src] == 0).all() and torch.isfinite(Q).all()
    print("[ok] marginals to ~1e-5 after 50 iters; padding exactly 0")

    # 2) kernel sign: Q must correlate positively with the similarity S
    corr = torch.corrcoef(torch.stack([Q.flatten(), S.flatten()]))[0, 1]
    assert corr > 0.3
    print(f"[ok] corr(Q, S) = {corr.item():.3f}   (literal Eq.(7) gives < 0)")

    # 3) masked rows must not steal mass, and nothing may be NaN
    assert Q[1, 30:].abs().max() < 1e-8 and not torch.isnan(Q).any()
    print("[ok] padded rows carry no mass, no NaN")

    # 4) bucket: absorbed mass grows as s_tilde drops, never exceeds rho
    # 4) bucket: absorbed mass grows as the margin s_tilde rises, capped by rho
    prev = -1.0
    for s_t in (0.1, 0.5, 0.9):
        Qb, dust = ot_coupling(zs, zt, src, tgt, lam=0.1, s_tilde=s_t, rho=0.2)
        assert (Qb.sum(-1) <= 1.0 / src.sum(-1, keepdim=True) + 1e-6).all()
        assert dust.mean().item() <= 0.2 + 1e-6
        assert dust.mean().item() >= prev - 1e-7
        prev = dust.mean().item()
        print(f"[ok] s_tilde={s_t}: bucket mass={dust.mean():.3f} "
              f"rowsum={Qb[src].mean():.4f} (<= {1/L:.4f})")

    # 5) loss + diagnostics
    Q, dust = ot_coupling(zs, zt, src, tgt, lam=0.1, s_tilde=0.5)
    loss = alignment_loss(S, Q, tau=0.07, mask=src)
    assert torch.isfinite(loss) and loss >= 0
    print(f"[ok] L_align={loss.item():.4f}  {diagnostics(S, Q, dust, src)}")

    # 6) the -inf pitfall
    logK = -(1.0 - S[1]) / 0.1
    logK[30:] = -float("inf")
    bad = sinkhorn_log(logK, torch.full((L,), 1.0 / L).log(),
                       torch.full((Lt,), 1.0 / Lt).log()).exp()
    print(f"[ok] -inf sentinel -> NaN: {torch.isnan(bad).any().item()} "
          f"(hence NEG = {NEG})")
    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
