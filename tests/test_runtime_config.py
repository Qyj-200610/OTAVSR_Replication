# Copyright (c) OTAVSR. All rights reserved.
"""Every optimiser / scheduler / hook entry of the configs must be buildable.

Typos in ``default_runtime.py`` or the schedule only surface when a run starts,
i.e. after the data has been loaded -- the most expensive possible moment.
"""
import glob
import os.path as osp

import pytest
from mmengine.config import Config
import torch
from mmengine.registry import (HOOKS, OPTIMIZERS, PARAM_SCHEDULERS,
                               init_default_scope)

import otavsr  # noqa: F401  (registers OTMonitorHook)


def _project_root() -> str:
    root = osp.dirname(osp.abspath(__file__))
    while (not osp.isdir(osp.join(root, 'configs'))
           and osp.dirname(root) != root):
        root = osp.dirname(root)
    return root


ROOT = _project_root()
CONFIGS = sorted(
    glob.glob(osp.join(ROOT, 'configs', '**', '*.py'), recursive=True))
EXPERIMENTS = [p for p in CONFIGS if '_base_' not in p]


def test_all_hooks_build():
    init_default_scope('otavsr')  # what Runner.from_cfg does
    cfg = Config.fromfile(osp.join(ROOT, 'configs', 'table2_noisy',
                                   'c_concat.py'))
    for name, hook_cfg in cfg.default_hooks.items():
        hook = HOOKS.build(hook_cfg)
        assert hook is not None, name


def test_optimizer_and_scheduler_build():
    init_default_scope('otavsr')
    cfg = Config.fromfile(osp.join(ROOT, 'configs', 'table2_noisy',
                                   'c_concat.py'))
    params = torch.nn.Linear(2, 2).parameters()
    optimizer = OPTIMIZERS.build(
        cfg.optim_wrapper.optimizer, default_args=dict(params=params))
    assert optimizer.param_groups[0]['lr'] == 1e-4

    for scheduler_cfg in cfg.param_scheduler:
        # LinearLR / CosineAnnealingLR already fix param_name='lr'
        scheduler = PARAM_SCHEDULERS.build(
            scheduler_cfg, default_args=dict(optimizer=optimizer))
        assert scheduler is not None


def test_warmup_and_total_steps_match_the_paper():
    cfg = Config.fromfile(osp.join(ROOT, 'configs', 'table2_noisy',
                                   'c_concat.py'))
    train_cfg = cfg.train_cfg
    assert train_cfg['max_iters'] == 30000
    assert train_cfg['by_epoch'] is False
    warmup = cfg.param_scheduler[0]
    assert warmup['type'] == 'LinearLR'
    assert warmup['end'] == 5000
    cosine = cfg.param_scheduler[1]
    assert cosine['begin'] == 5000
    assert cosine['end'] == 30000


@pytest.mark.parametrize('path', EXPERIMENTS,
                         ids=lambda p: osp.relpath(p, ROOT))
def test_eval_pipeline_exposes_the_snr_knob(path):
    """``--cfg-options test_dataloader.dataset.pipeline.1.eval_snr=0`` must
    keep working: the noise transform has to be second in the test pipeline."""
    cfg = Config.fromfile(path)
    pipeline = cfg.test_dataloader.dataset.pipeline
    if not any(step['type'] == 'AddBabbleNoise' for step in pipeline):
        pytest.skip('clean pipeline')
    assert pipeline[1]['type'] == 'AddBabbleNoise'
    assert pipeline[1]['seed'] is not None  # reproducible evaluation noise
