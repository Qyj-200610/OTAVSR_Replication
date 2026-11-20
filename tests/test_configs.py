# Copyright (c) OTAVSR. All rights reserved.
"""Every experiment config must be loadable and internally consistent."""
import glob
import os.path as osp

import pytest
from mmengine.config import Config

CONFIGS = sorted(glob.glob(
    osp.join(osp.dirname(osp.dirname(__file__)), 'configs', '**', '*.py'),
    recursive=True))
EXPERIMENTS = [path for path in CONFIGS if '_base_' not in path]


def _rel(path):
    return osp.relpath(path, osp.dirname(osp.dirname(__file__)))


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
