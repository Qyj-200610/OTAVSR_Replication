# Copyright (c) OTAVSR. All rights reserved.
"""Training entry, mirroring ``mmpretrain/apis/train.py``."""
import copy
from typing import Optional

from mmengine.config import Config
from mmengine.runner import Runner

__all__ = ['train_model']


def train_model(cfg: Config,
                work_dir: Optional[str] = None,
                auto_scale_lr: Optional[dict] = None,
                resume: bool = False,
                load_from: Optional[str] = None) -> Runner:
    """Build a runner from ``cfg`` and train.

    Unlike mmpretrain's ``train_model(model, dataset, cfg)``, everything here is
    built from the config (the encoders / LLM are lazily instantiated, so a
    config is a full description of the experiment).

    Args:
        cfg (Config): full experiment config.
        work_dir (str, optional): override ``cfg.work_dir``.
        auto_scale_lr (dict, optional): ``dict(base_batch_size=..., enable=...)``.
            Accepted for API parity; the paper's schedule is fixed, so it is
            not applied automatically.
        resume (bool): resume from ``work_dir/last_checkpoint``. Implemented by
            setting ``cfg.resume`` and letting ``Runner.train()`` call
            ``load_or_resume()`` -- ``Runner.resume`` expects a *file* path, not
            a directory, so calling it with ``work_dir`` raises.
        load_from (str, optional): load weights before training.

    Returns:
        Runner: the runner after ``train()``.
    """
    cfg = copy.deepcopy(cfg)
    if work_dir is not None:
        cfg.work_dir = work_dir
    if load_from is not None:
        cfg.load_from = load_from
    if resume:
        cfg.resume = True
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner
