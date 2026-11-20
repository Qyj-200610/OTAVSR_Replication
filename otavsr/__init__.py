# Copyright (c) OTAVSR. All rights reserved.
"""OTAVSR: replication of *Optimal Transport-based Semantic Alignment for
LLM-based Audio-Visual Speech Recognition* (arXiv:2607.09001)."""
from .version import __author__, __version__
from .registry import (DATASETS, HOOKS, METRICS, MODELS, OPTIM_WRAPPERS,
                       SAMPLERS, TRANSFORMS)
# structures / utils first: models and datasets depend on them
from . import structures, utils
from .structures import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from . import datasets, evaluation, models
from .datasets import *  # noqa: F401,F403
from .evaluation import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from . import apis, engine
from .engine import *  # noqa: F401,F403
from .apis import *  # noqa: F401,F403

__all__ = [
    '__version__', '__author__', 'MODELS', 'DATASETS', 'TRANSFORMS',
    'METRICS', 'HOOKS', 'OPTIM_WRAPPERS', 'SAMPLERS'
] + structures.__all__ + utils.__all__ + datasets.__all__ + \
    evaluation.__all__ + models.__all__ + engine.__all__ + apis.__all__
