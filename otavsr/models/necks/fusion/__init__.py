# Copyright (c) OTAVSR. All rights reserved.
"""Modality fusion modules of Eq.(12).

``C_Concat`` is the fusion finally adopted by the paper; ``Add`` /
``Cross_att`` / ``Q_former`` are the ablations of Table I / Table II.  These
four are exactly the ``modality_fuse`` switch of the official MMS-LLaMA code
(``src/model.py``), which is where the paper took them from.
"""
from .add import AddFusion
from .c_concat import ChannelConcatFusion
from .cross_att import CrossAttentionFusion
from .q_former import QFormerFusion

__all__ = [
    'ChannelConcatFusion', 'AddFusion', 'CrossAttentionFusion', 'QFormerFusion'
]
