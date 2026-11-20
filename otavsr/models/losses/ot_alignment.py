# Copyright (c) OTAVSR. All rights reserved.
"""Alignment loss of OT-LLM-AVSR (paper Eq.(10)-(11))."""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from otavsr.registry import MODELS
from otavsr.utils.sinkhorn import (cosine_sim, ot_diagnostics, solve_ot)

__all__ = ['OTAlignmentLoss']


@MODELS.register_module()
class OTAlignmentLoss(BaseModule):
    """OT coupling as a soft pseudo-label for a temperature-scaled contrastive
    distribution.

    ``L_align = - sum_ij Q_ij log P_ij`` with ``P = softmax_j(S / tau)``, where
    ``S`` is the cosine similarity between the modality features and
    ``FC(Embed(target))``, and ``Q`` is the entropy-regularised OT coupling.

    Gradient only flows through ``P`` (``Q`` is detached by default), i.e. the
    useful signal is ``(P - Q) / tau``, which is O(1) regardless of the loss
    value.  Never size ``alpha`` by comparing loss values.

    Args:
        lam (float): entropy regularisation ``lambda`` of Eq.(2). The paper
            sweeps 0.05 / 0.1 / 0.2.
        tau (float): temperature of Eq.(11). Not reported by the paper.
        s_tilde (float, optional): similarity margin of the virtual bucket.
            ``None`` disables the bucket (Table III "wo bucket").
        rho (float): fraction of the frame mass discarded by the bucket,
            ``0 <= rho < 1``. This is a hard marginal (exactly ``rho`` is
            discarded), so ``s_tilde`` chooses *which* frames, not how many.
        bucket_row (bool): also add the bucket row of Eq.(9).
        sinkhorn_iters (int): Sinkhorn iterations.
        detach_q (bool): treat ``Q`` as a fixed soft label.
        loss_weight (float): ``alpha`` of ``L_total = L_AR + alpha * L_align``.
            It is kept here so that the weight lives next to the loss, as in
            mmpretrain; the classifier passes ``alpha`` through the config.
        fp32 (bool): run the OT branch in fp32 even under autocast. bf16/fp16
            loses too much precision in the scaling-vector iterations.
        reduction (str): ``'sum'`` (paper Eq.(10), default) or ``'mean'``.
            ``'sum'`` gives ``L_align ~ 2-6`` for a 16 s clip, so ``alpha=0.1``
            matches the paper's scale; ``'mean'`` divides by the number of
            frames and makes ``alpha`` ~L times larger for the same effect.
            The two are equivalent up to ``alpha``; say which one you used.
    """

    def __init__(self,
                 lam: float = 0.1,
                 tau: float = 0.07,
                 s_tilde: Optional[float] = 0.5,
                 rho: float = 0.2,
                 bucket_row: bool = False,
                 sinkhorn_iters: int = 50,
                 detach_q: bool = True,
                 loss_weight: float = 0.1,
                 fp32: bool = True,
                 reduction: str = 'sum',
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.lam = lam
        self.tau = tau
        self.s_tilde = s_tilde
        self.rho = rho
        self.reduction = reduction
        self.bucket_row = bucket_row
        self.sinkhorn_iters = sinkhorn_iters
        self.detach_q = detach_q
        self.loss_weight = loss_weight
        self.fp32 = fp32

    def _coupled(self, z_src, z_tgt, src_mask, tgt_mask):
        with torch.autocast(device_type=z_src.device.type,
                            enabled=False) if self.fp32 else _nullcontext():
            coupling, dust = solve_ot(
                z_src.float() if self.fp32 else z_src,
                z_tgt.float() if self.fp32 else z_tgt,
                src_mask,
                tgt_mask,
                lam=self.lam,
                s_tilde=self.s_tilde,
                rho=self.rho,
                bucket_row=self.bucket_row,
                num_iters=self.sinkhorn_iters)
        return coupling, dust

    def align_loss(self, z_src, z_tgt, src_mask=None, tgt_mask=None):
        """Return the unweighted alignment term and the OT diagnostics."""
        if src_mask is None:
            src_mask = z_src.new_ones(z_src.shape[:2], dtype=torch.bool)
        if tgt_mask is None:
            tgt_mask = z_tgt.new_ones(z_tgt.shape[:2], dtype=torch.bool)

        coupling, dust = self._coupled(z_src, z_tgt, src_mask, tgt_mask)
        sim = cosine_sim(
            z_src.float() if self.fp32 else z_src,
            z_tgt.float() if self.fp32 else z_tgt)
        log_prob = F.log_softmax(sim / self.tau, dim=-1)
        weight = coupling.detach() if self.detach_q else coupling
        per_frame = -(weight * log_prob).sum(-1)
        valid = src_mask.to(per_frame.dtype)
        if self.reduction == 'sum':
            # paper Eq.(10) literally: sum over i and j, mean over the batch.
            # L_align is then O(L) -- ~2-6 for a 16 s clip -- so alpha=0.1 puts
            # it on the same scale as the AR loss.
            loss = per_frame.sum() / per_frame.shape[0]
        elif self.reduction == 'mean':
            # per-frame average: L_align drops to ~1e-2, which forces alpha to
            # be ~L times larger for the same effect.
            loss = (per_frame * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            raise ValueError(f"unknown reduction {self.reduction!r}")
        stats = ot_diagnostics(sim, coupling, dust, src_mask)
        return loss, stats

    def forward(self,
                z_src: torch.Tensor,
                z_tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Args are ``z`` of shape ``[B, L, d]`` and ``[B, Lt, d]``.

        Returns ``{'loss_align': tensor}`` so that it can be fed straight into
        ``mmengine.model.parse_losses``; the diagnostics are available from
        :meth:`align_loss` and are consumed by ``OTMonitorHook``.
        """
        loss, _ = self.align_loss(z_src, z_tgt, src_mask, tgt_mask)
        return {'loss_align': loss * self.loss_weight}


class _nullcontext:
    """Tiny local replacement of ``contextlib.nullcontext`` (py3.7 friendly)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False
