# Copyright (c) OTAVSR. All rights reserved.
"""Model components, mirroring ``mmpretrain/models/__init__.py``."""
from . import (backbones, classifiers, data_preprocessors, heads, losses,
               necks)
from .backbones import *  # noqa: F401,F403
from .classifiers import *  # noqa: F401,F403
from .data_preprocessors import *  # noqa: F401,F403
from .heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403

__all__ = (backbones.__all__ + classifiers.__all__ +
           data_preprocessors.__all__ + heads.__all__ + losses.__all__ +
           necks.__all__)
