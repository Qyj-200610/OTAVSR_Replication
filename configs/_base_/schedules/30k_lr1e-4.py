# Copyright (c) OTAVSR. All rights reserved.
"""Optimiser and schedule of the paper §III.

Adam, lr 1e-4, 5k warm-up steps out of 30k total.  Note that neither official
repository uses this exact schedule (Llama-AVSR counts warm-up in *epochs*,
MMS-LLaMA uses 500 steps), so this file is the single source of truth for the
paper's numbers; see the guide §2.4.

bf16 does not need loss scaling: ``AmpOptimWrapper`` only accepts
``'dynamic'``/float/dict for ``loss_scale``, and the dict form below disables
the GradScaler (``enabled=False``) instead of scaling by 65536 and undoing it
every step.
"""
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=dict(init_scale=1.0, enabled=False),
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=0.0, betas=(0.9, 0.98)),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=5000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=False,
        begin=5000,
        end=30000),
]

train_cfg = dict(by_epoch=False, max_iters=30000, val_interval=3000)
val_cfg = dict()
test_cfg = dict()
