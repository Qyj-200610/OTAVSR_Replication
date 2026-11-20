# Copyright (c) OTAVSR. All rights reserved.
"""Default runtime settings (logging, checkpoints, distributed)."""
default_scope = 'otavsr'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=False,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    ot_monitor=dict(
        type='OTMonitorHook',
        interval=100,
        s_tilde=0.5,
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
log_processor = dict(by_epoch=False, window_size=50)
load_from = None
resume = False
