# Copyright (c) OTAVSR. All rights reserved.
"""Hook that logs the OT diagnostics of the alignment branch.

The alignment loss is small compared with the AR loss and can silently become
a no-op. This hook watches the scalars that reveal it (guide §3.3.1):

* ``ot_row_max_cos``: mean of ``max_j cos(z_i, z_j^t)``. If it sits well below
  ``s_tilde`` the bucket is discarding informative frames -- the margin is
  mis-scaled for this feature space.
* ``ot_dusted_row_max_cos``: the same average over the discarded frames. It
  must be clearly smaller than ``ot_row_max_cos``, otherwise the bucket
  discards at random.
* ``ot_bucket_mass``: pinned to ``rho`` by construction (hard marginal), kept
  only as a sanity number.
* ``ot_q_entropy``: close to ``log(Lt)`` means a uniform coupling, i.e. the
  alignment has collapsed.
"""
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from otavsr.registry import HOOKS

__all__ = ['OTMonitorHook']


@HOOKS.register_module()
class OTMonitorHook(Hook):
    """Write ``train/ot_*`` scalars to the message hub every ``interval`` iters.

    Args:
        interval (int): logging interval in iterations.
        keys (Sequence[str]): diagnostic names produced by
            :meth:`OTAlignmentLoss.align_loss`.
        s_tilde (float, optional): similarity margin used in training; enables
            the "discarding informative frames" warning.
        margin_tol (float): how far below ``s_tilde`` the mean row maximum may
            fall before warning.
    """

    priority = 'LOW'

    def __init__(self,
                 interval: int = 100,
                 keys: Sequence[str] = ('bucket_mass', 'row_max_cos',
                                        'dusted_row_max_cos', 'q_rowsum',
                                        'q_entropy'),
                 s_tilde: Optional[float] = 0.5,
                 margin_tol: float = 0.1) -> None:
        self.interval = interval
        self.keys = tuple(keys)
        self.s_tilde = s_tilde
        self.margin_tol = margin_tol
        self._warned = False

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        model = getattr(runner.model, 'module', runner.model)
        stats = getattr(model, '_ot_stats', None)
        if not stats:
            return
        if (runner.iter + 1) % max(self.interval, 1) != 0:
            return
        for key in self.keys:
            full_key = key if key.startswith('ot_') else f'ot_{key}'
            if full_key in stats:
                runner.message_hub.update_scalar(f'train/{full_key}',
                                                 float(stats[full_key]))

        if self._warned or self.s_tilde is None:
            return
        row_max = float(stats.get('ot_row_max_cos', 1.0))
        bucketed = float(stats.get('ot_dusted_row_max_cos', 0.0))
        logger = MMLogger.get_current_instance()
        if row_max < self.s_tilde - self.margin_tol:
            logger.warning(
                'OT margin mis-scaled: mean row_max_cos=%.3f < s_tilde=%.3f. '
                'The bucket is discarding informative frames; lower s_tilde '
                'to roughly the 10th percentile of row_max_cos (guide §3.3.1).',
                row_max, self.s_tilde)
            self._warned = True
        elif bucketed >= row_max - 1e-3:
            logger.warning(
                'OT bucket discards at random: dusted_row_max_cos=%.3f equals '
                'row_max_cos=%.3f. Check the bucket marginals (guide §3.3).',
                bucketed, row_max)
            self._warned = True
