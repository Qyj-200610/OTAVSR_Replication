# Copyright (c) OTAVSR. All rights reserved.
"""Table III, OT with the virtual bucket (s_tilde=0.5, rho=0.2, lambda=0.1, alpha=0.1)."""
_base_ = [
    '../_base_/models/avsr_llm_c_concat.py',
    '../_base_/datasets/lrs3_433h_seg16s_noise75.py',
    '../_base_/schedules/30k_lr1e-4.py',
    '../_base_/default_runtime.py',
]
model = dict(ot_loss=dict(s_tilde=0.5, rho=0.2))
