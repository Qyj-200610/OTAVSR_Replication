# Copyright (c) OTAVSR. All rights reserved.
"""Add fusion ablation (Table I / II). (clean training)"""
_base_ = [
    '../_base_/models/avsr_llm_c_concat.py',
    '../_base_/datasets/lrs3_433h_seg16s.py',
    '../_base_/schedules/30k_lr1e-4.py',
    '../_base_/default_runtime.py',
]
model = dict(fusion=dict(type='AddFusion'))
