# Copyright (c) OTAVSR. All rights reserved.
"""Table V setting 3: alpha=0.1, s_tilde=0.5, lambda=0.05."""
_base_ = [
    '../_base_/models/avsr_llm_c_concat.py',
    '../_base_/datasets/lrs3_433h_seg16s_noise75.py',
    '../_base_/schedules/30k_lr1e-4.py',
    '../_base_/default_runtime.py',
]
model = dict(ot_loss=dict(loss_weight=0.1, lam=0.05, s_tilde=0.5))
