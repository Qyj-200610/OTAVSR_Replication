# Copyright (c) OTAVSR. All rights reserved.
"""Evaluation-only config (Table I/II/III/IV/V numbers are all produced by it).

The SNR is chosen on the command line; ``AddBabbleNoise`` is the second step of
``test_pipeline``, so::

    python tools/test.py configs/eval_lrs3_433h_snr.py \
        --checkpoint work_dirs/table5_setting3/iter_30000.pth \
        --cfg-options test_dataloader.dataset.pipeline.1.eval_snr=0

SNR values used by the paper: 999 (clean), -5, 0, 5. Always keep the same
``eval_noise_seed`` when comparing two models, otherwise the random noise crop
alone moves WER by ~0.3 (guide §2.5).
"""
_base_ = [
    './_base_/models/avsr_llm_c_concat.py',
    './_base_/datasets/lrs3_433h_seg16s_noise75.py',
    './_base_/default_runtime.py',
]

test_cfg = dict()
