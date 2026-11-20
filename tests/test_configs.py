# Copyright (c) OTAVSR. All rights reserved.
"""Every experiment config must be loadable and internally consistent."""
import glob
import os.path as osp

import pytest
from mmengine.config import Config


def _project_root() -> str:
    """Walk up until the directory that contains ``configs/`` is found.

    ``__file__`` may be relative (pytest runs from the repo root) or absolute,
    and this file sits at a different depth than the other test modules, so
    ``dirname(dirname(__file__))`` is not reliable.
    """
    root = osp.dirname(osp.abspath(__file__))
    while (not osp.isdir(osp.join(root, 'configs'))
           and osp.dirname(root) != root):
        root = osp.dirname(root)
    return root


ROOT = _project_root()
CONFIGS = sorted(
    glob.glob(osp.join(ROOT, 'configs', '**', '*.py'), recursive=True))
EXPERIMENTS = [path for path in CONFIGS if '_base_' not in path]


def _rel(path):
    return osp.relpath(path, ROOT)


def test_configs_are_found():
    assert len(EXPERIMENTS) >= 20


@pytest.mark.parametrize('path', CONFIGS, ids=_rel)
def test_load(path):
    cfg = Config.fromfile(path)

    if '_base_' in path:
        return  # fragments: nothing to assert beyond loadability

    assert cfg.model.type == 'AVSRLLMClassifier'
    assert 'train_dataloader' in cfg or 'test_dataloader' in cfg
    assert cfg.model.head.type == 'LLMDecoder'
    assert cfg.test_evaluator.type == 'WordErrorRate'


@pytest.mark.parametrize('path', EXPERIMENTS, ids=_rel)
def test_ot_branch_is_configured(path):
    cfg = Config.fromfile(path)
    ot_loss = cfg.model.ot_loss
    if ot_loss is None:
        return
    assert ot_loss.type == 'OTAlignmentLoss'
    assert ot_loss.lam > 0
    assert 0 <= ot_loss.rho < 1
    if ot_loss.s_tilde is not None:
        assert -1 <= ot_loss.s_tilde <= 1
    assert ot_loss.reduction in ('sum', 'mean')


@pytest.mark.parametrize('path', EXPERIMENTS, ids=_rel)
def test_runtime_is_sane(path):
    """Guards the two config bugs that silently ruin a 30k-step run:
    saving a checkpoint every iteration, and maximising WER as 'best'."""
    cfg = Config.fromfile(path)
    checkpoint = cfg.default_hooks.checkpoint
    assert checkpoint.interval >= 1000
    assert checkpoint.save_best == 'wer'
    assert checkpoint.rule == 'less'
    assert checkpoint.max_keep_ckpts <= 5


def test_fusion_matrix_covers_the_paper_tables():
    names = {
        osp.splitext(osp.basename(path))[0]
        for path in EXPERIMENTS if 'table1_clean' in path
    }
    assert names == {'c_concat', 'add', 'cross_att', 'q_former', 't_concat'}


def test_alpha_sweep_is_complete():
    alphas = sorted(
        osp.splitext(osp.basename(path))[0]
        for path in EXPERIMENTS if 'table4_alpha' in path)
    assert alphas == ['alpha_0.05', 'alpha_0.1', 'alpha_0.3', 'alpha_0.5',
                      'alpha_0.7', 'alpha_0.9']
