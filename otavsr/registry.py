# Copyright (c) OTAVSR. All rights reserved.
"""Registries of OTAVSR.

Mirrors ``mmpretrain/registry.py``: every registry has mmengine's registry as
parent, so third-party modules (transforms, metrics, hooks, optimizers coming
from mmengine / mmcv) can be used without re-registering them.
"""
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

__all__ = [
    'MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'HOOKS', 'OPTIM_WRAPPERS',
    'SAMPLERS'
]

MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['otavsr.models'])

DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['otavsr.datasets'])

TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['otavsr.datasets.transforms'])

METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['otavsr.evaluation'])

HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['otavsr.engine.hooks'])

OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['otavsr.engine.optimizers'])

SAMPLERS = Registry(
    'sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['otavsr.datasets'])
