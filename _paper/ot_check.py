"""Numerical checks for the OT-alignment recipe proposed in the replication guide.

Run:  python _paper/ot_check.py
Everything here is CPU-only and deterministic.
"""
import torch

torch.manual_seed(0)
NEG = -1e4          # finite sentinel used instead of -inf so that logsumexp stays finite


def sinkhorn_log(logK, logp, logq, n_iters=200):
    """logK: [m,n]; logp: [m]; logq: [n]  -> logQ: [m,n]"""
    logu = torch.zeros_like(logp)
    logv = torch.zeros_like(logq)
    for _ in range(n_iters):
        logu = logp - torch.logsumexp(logK + logv.unsqueeze(0), dim=1)
        logv = logq - torch.logsumexp(logK + logu.unsqueeze(1), dim=0)
    return logu.unsqueeze(1) + logK + logv.unsqueeze(0)


def cos_sim(Za, Zt):
    Za = torch.nn.functional.normalize(Za, dim=-1)
    Zt = torch.nn.functional.normalize(Zt, dim=-1)
    return Za @ Zt.T


m, n, d = 40, 6, 32          # 40 frames (1.6 s @25 Hz), 6 text tokens
Za, Zt = torch.randn(m, d), torch.randn(n, d)
S = cos_sim(Za, Zt)
p, q = torch.full((m,), 1.0 / m), torch.full((n,), 1.0 / n)
lam = 0.1

print("=" * 70)
print("[1] sign convention: K = exp(-S/lam)  vs  K = exp(+(1-S)/lam)")
logQ_paper = sinkhorn_log(-S / lam, p.log(), q.log())          # literal Eq.(7) of the paper
logQ_fix = sinkhorn_log(-(1.0 - S) / lam, p.log(), q.log())    # cost = 1 - cos
Q_paper, Q_fix = logQ_paper.exp(), logQ_fix.exp()
print("  corr(Q_paper, S) =", round(torch.corrcoef(torch.stack([Q_paper.flatten(), S.flatten()]))[0, 1].item(), 4))
print("  corr(Q_fix  , S) =", round(torch.corrcoef(torch.stack([Q_fix.flatten(), S.flatten()]))[0, 1].item(), 4))
print("  => literal Eq.(7) transports mass to the LEAST similar tokens (anti-alignment)")

print("\n[2] equivalence of cost 1-cos and -cos (constant shift must cancel)")
logQ_b = sinkhorn_log(-(-S) / lam, p.log(), q.log())
print("  max |Q(1-cos) - Q(-cos)| =", (logQ_fix.exp() - logQ_b.exp()).abs().max().item())

print("\n[3] marginal constraints of the plain (no bucket) coupling")
print("  max |rowsum - 1/m| =", (Q_fix.sum(1) - 1.0 / m).abs().max().item())
print("  max |colsum - 1/n| =", (Q_fix.sum(0) - 1.0 / n).abs().max().item())

print("\n[4] numerical range at lam=0.05 (fp32 ok, bf16 not)")
print("  max exp(-S/lam)      lam=0.05 :", round((-S / 0.05).exp().max().item(), 1))
print("  max exp(-(1-S)/lam)  lam=0.05 :", (-(1 - S) / 0.05).exp().max().item())

print("\n" + "=" * 70)
print("[5] virtual bucket, recipe A (hard bucket ROW + COL, equal mass): the marginal\n"
      "    copy pins the absorbed mass, so s_tilde has no effect -> wrong recipe")


def bucket_q_hard(S, lam, s_tilde, n_iters=300):
    m, n = S.shape
    St = S.new_full((m + 1, n + 1), s_tilde)
    St[:m, :n] = S
    pt = S.new_full((m + 1,), 1.0 / (m + 1))
    qt = S.new_full((n + 1,), 1.0 / (n + 1))
    Q = sinkhorn_log(-(1.0 - St) / lam, pt.log(), qt.log(), n_iters).exp()
    return Q[:m, :n]


for s_t in (1.0, 0.7, 0.5, 0.3, 0.0):
    Qr = bucket_q_hard(S, 0.1, s_t)
    print(f"  s_tilde={s_t:<4} -> bucket mass {100*(1-Qr.sum().item()):5.1f}%")

print("\n[6] recipe B (bucket COLUMN only, slack capacity rho): absorbed mass is driven\n"
      "    by s_tilde, real marginals stay 1/L, real columns stay 1/Lt")


def bucket_q(S, lam, s_tilde, rho=1.0, n_iters=300):
    """S: [m,n] cosine sim. Returns truncated Q [m,n] and the mass sent to the bucket."""
    m, n = S.shape
    St = S.new_full((m, n + 1), s_tilde)
    St[:, :n] = S
    p_ = S.new_full((m,), (1.0 + rho) / m)            # real rows, scaled so totals match
    q_ = torch.cat([S.new_full((n,), 1.0 / n), S.new_full((1,), rho)])
    Q = sinkhorn_log(-(1.0 - St) / lam, p_.log(), q_.log(), n_iters).exp()
    return Q[:, :n], Q[:, n]


for s_t in (1.0, 0.7, 0.5, 0.3, 0.0):
    Qr, dust = bucket_q(S, 0.1, s_t)
    print(f"  s_tilde={s_t:<4} -> bucket mass {100*dust.sum().item():5.1f}%   "
          f"rowsum in [{Qr.sum(1).min():.4f},{Qr.sum(1).max():.4f}]   1/L={1/m:.4f}")

print("\n[7] same, sweeping lam at fixed s_tilde=0.5 (shows the lam / s_tilde interaction)")
for lam_i in (0.05, 0.1, 0.2, 0.5):
    Qr, dust = bucket_q(S, lam_i, 0.5)
    print(f"  lam={lam_i:<5} -> bucket mass {100*dust.sum().item():5.1f}%")

print("\n[8] effect of rho (bucket capacity) at lam=0.1, s_tilde=0.5")
for rho in (0.25, 1.0, 4.0):
    Qr, dust = bucket_q(S, 0.1, 0.5, rho=rho)
    print(f"  rho={rho:<5} -> bucket mass {100*dust.sum().item():5.1f}%")

print("\n[9] what truncation does to -sum_ij Q log P (tau=0.07)")
tau = 0.07
logP = torch.log_softmax(S / tau, dim=1).detach()
print(f"  no bucket      L_align = {-(Q_fix * logP).sum().item():8.4f}")
Qr, _ = bucket_q(S, 0.1, 0.5)
print(f"  with bucket    L_align = {-(Qr * logP).sum().item():8.4f}   (smaller: mass moved off hard frames)")

print("\n[10] masked Sinkhorn with padded rows (5 of 40 frames are padding)")
mask = torch.ones(m, dtype=torch.bool)
mask[35:] = False
S2 = S.clone()
S2[~mask] = 0.0
p2 = torch.where(mask, torch.full_like(p, 1.0 / mask.sum()), torch.zeros_like(p))
logK = -(1.0 - S2) / lam
logK[~mask] = NEG                                   # finite sentinel, NOT -inf
logQ = sinkhorn_log(logK, p2.clamp_min(1e-30).log(), q.log())
Q2 = logQ.exp()
print("  max rowsum of padded rows :", Q2[~mask].sum(1).max().item())
print("  rowsum of valid rows       :", round(Q2[mask].sum(1).min().item(), 4),
      round(Q2[mask].sum(1).max().item(), 4), " (target 1/35 =", round(1 / 35, 4), ")")
print("  any NaN:", torch.isnan(Q2).any().item())

print("\n[11] same with -inf instead of a sentinel (what a naive implementation gets)")
logK_bad = -(1.0 - S2) / lam
logK_bad[~mask] = -float("inf")
Q_bad = sinkhorn_log(logK_bad, p2.clamp_min(1e-30).log(), q.log()).exp()
print("  any NaN:", torch.isnan(Q_bad).any().item(),
      " -> a fully padded row makes logsumexp=-inf and log(u)=+inf, so NaN")
