# Copyright (c) OTAVSR. All rights reserved.
"""Testing entry, mirroring ``mmpretrain/apis/test.py``."""
import copy
from typing import Dict, Optional

from mmengine.config import Config
from mmengine.runner import Runner

__all__ = ['test_model']


def test_model(cfg: Config,
               checkpoint: Optional[str] = None,
               work_dir: Optional[str] = None,
               show_dir: Optional[str] = None) -> Dict:
    """Run ``Runner.test()`` with the model of ``checkpoint``.

    Args:
        cfg (Config): experiment config (only ``model`` / dataloaders /
            ``test_evaluator`` are used).
        checkpoint (str, optional): weights to load; ``None`` uses
            ``cfg.load_from``.
        work_dir (str, optional): directory for logs and predictions.
        show_dir (str, optional): kept for API parity with mmpretrain.

    Returns:
        dict: ``{metric_name: value}``, e.g. ``{'wer': 0.74}``.
    """
    cfg = copy.deepcopy(cfg)
    if work_dir is not None:
        cfg.work_dir = work_dir
    if checkpoint is not None:
        cfg.load_from = checkpoint
    runner = Runner.from_cfg(cfg)
    return runner.test()
