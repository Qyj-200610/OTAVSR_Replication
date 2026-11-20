# Copyright (c) OTAVSR. All rights reserved.
"""Default runtime settings (logging, checkpoints, distributed)."""
import platform

default_scope = 'otavsr'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        # 30k iters: saving every iteration would write 30k files. The paper
        # evaluates the 30k checkpoint and the val interval is 3k, so 3k.
        interval=3000,
        by_epoch=False,
        save_best='wer',
        # WER must be MINIMISED; mmengine's default rule is 'greater', which
        # would keep the worst checkpoint.
        rule='less',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    ot_monitor=dict(
        type='OTMonitorHook',
        interval=100,
        s_tilde=0.5,
    ),
)

#: 'fork' does not exist on Windows; the dataloader workers would crash there.
_MP_START_METHOD = 'spawn' if platform.system() == 'Windows' else 'fork'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method=_MP_START_METHOD, opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
log_processor = dict(by_epoch=False, window_size=50)
load_from = None
resume = False
