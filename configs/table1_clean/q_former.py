# Copyright (c) OTAVSR. All rights reserved.
"""Q-Former fusion ablation (Table I / II), queries_per_sec=3 as in MMS-LLaMA. (clean training)"""
_base_ = [
    '../_base_/models/avsr_llm_c_concat.py',
    '../_base_/datasets/lrs3_433h_seg16s.py',
    '../_base_/schedules/30k_lr1e-4.py',
    '../_base_/default_runtime.py',
]
model = dict(fusion=dict(type='QFormerFusion', in_channels=3072, out_channels=3072, queries_per_sec=3.0, num_layers=2, num_heads=8))
