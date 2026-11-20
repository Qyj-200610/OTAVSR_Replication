# Copyright (c) OTAVSR. All rights reserved.
"""Every experiment config must survive ``MODELS.build``.

The encoders / LLM are lazily instantiated, so building the whole model needs
neither ``transformers`` nor ``fairseq`` -- but it *does* catch constructor
signature errors (e.g. a config key the classifier does not accept), which is
exactly the class of bug that only shows up on the first real run.
"""
import glob
import os.path as osp

import pytest
from mmengine.config import Config

import otavsr
from otavsr.models.data_preprocessors import AVSRDataPreprocessor

def _project_root() -> str:
    """Walk up until the directory that contains ``configs/`` is found."""
    root = osp.dirname(osp.abspath(__file__))
    while (not osp.isdir(osp.join(root, 'configs'))
           and osp.dirname(root) != root):
        root = osp.dirname(root)
    return root


ROOT = _project_root()
CONFIGS = sorted(glob.glob(osp.join(ROOT, 'configs', '**', '*.py'),
                           recursive=True))
EXPERIMENTS = [p for p in CONFIGS if '_base_' not in p]


def _rel(path):
    return osp.relpath(path, ROOT)


@pytest.mark.parametrize('path', EXPERIMENTS, ids=_rel)
def test_build_model(path):
    cfg = Config.fromfile(path)
    model = otavsr.MODELS.build(cfg.model)

    assert model.modality in ('audiovisual', 'audio')
    assert isinstance(model.data_preprocessor, AVSRDataPreprocessor)
    # the frozen encoders are built lazily: no heavy import at build time
    assert model.audio_backbone.out_channels == 1024
    assert model.head.llm_path.startswith('meta-llama/')
    if model.modality == 'audiovisual':
        assert model.fusion is not None
    else:
        assert model.fusion is None


def test_unknown_kwarg_is_rejected():
    """A typo in a config must fail loudly at build time."""
    with pytest.raises(TypeError):
        otavsr.MODELS.build(
            dict(type='AVSRLLMClassifier', audio_backbone=dict(
                type='WhisperEncoder'), head=dict(type='LLMDecoder'),
                not_a_real_argument=1))


def test_audio_only_base_builds():
    """``_base_/models/avsr_llm_audio_only.py`` is a fragment, so the
    parametrised test above does not cover it."""
    cfg = Config.fromfile(
        osp.join(ROOT, 'configs', '_base_', 'models',
                 'avsr_llm_audio_only.py'))
    model = otavsr.MODELS.build(cfg.model)
    assert model.modality == 'audio'
    assert model.video_backbone is None
    assert model.fusion is None
