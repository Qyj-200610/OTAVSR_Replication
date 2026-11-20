# Copyright (c) OTAVSR. All rights reserved.
"""LRS3-TED, 433 h, noisy training (Table II onward, the paper's default).

75 % of the training clips get babble noise at an SNR drawn uniformly from
{-5, 0, 5, 10, 15, 20} dB (MMS-LLaMA's setting, which the paper follows).
Llama-AVSR instead puts ``999999`` in the SNR list, which yields 1/7 clean
clips (86 %, not 75 %) -- see the guide §2.5.

The pipelines are intentionally duplicated instead of inherited from
``lrs3_433h_seg16s.py``: mmengine *replaces* lists when merging, so a child
config would have to repeat them anyway, and repeating keeps every step of the
noisy pipeline visible in one screen.

Evaluation: the noise crop is random, so every run must pin it. ``seed`` below
does that. To evaluate at a fixed SNR use::

    --cfg-options test_dataloader.dataset.pipeline.1.eval_snr=0

(the ``AddBabbleNoise`` entry is the second element of ``test_pipeline``);
``999`` means clean.
"""
data_root = 'data/LRS3'
ann_train = 'labels/lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv'
ann_val = 'labels/lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv'
ann_test = 'labels/lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv'

sample_rate = 16000
max_frames = 400  # 16 s at 25 fps
tokenizer_path = 'meta-llama/Llama-3.2-3B'
noise_path = 'data/noise/babble_noise.wav'
eval_snr = 999.0  # clean by default; override per run
eval_noise_seed = 1234

train_pipeline = [
    dict(type='LoadAudioVideo', sample_rate=16000, max_frames=400),
    dict(
        type='AddBabbleNoise',
        noise_path=noise_path,
        prob=0.75,
        snr_levels=(-5, 0, 5, 10, 15, 20)),
    dict(type='MouthCrop', size=88, random=True),
    dict(type='RandomHorizontalFlip', prob=0.5),
    dict(type='WhisperLogMel'),
    dict(
        type='TokenizeTranscript',
        tokenizer_path=tokenizer_path,
        max_length=64),
    dict(type='PackAVSRInputs'),
]

test_pipeline = [
    dict(type='LoadAudioVideo', sample_rate=16000, max_frames=400),
    dict(
        type='AddBabbleNoise',
        noise_path=noise_path,
        eval_snr=eval_snr,
        seed=eval_noise_seed),
    dict(type='MouthCrop', size=88, random=False),
    dict(type='WhisperLogMel'),
    dict(
        type='TokenizeTranscript',
        tokenizer_path=tokenizer_path,
        max_length=64),
    dict(type='PackAVSRInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='LRS3Dataset',
        data_root=data_root,
        ann_file=ann_train,
        pipeline=train_pipeline,
        modality='audiovisual'),
    batch_sampler=dict(
        type='MaxFramesBatchSampler', max_frames=1000, shuffle=True),
    collate_fn=dict(type='pseudo_collate'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='LRS3Dataset',
        data_root=data_root,
        ann_file=ann_val,
        pipeline=test_pipeline,
        modality='audiovisual',
        test_mode=True),
    collate_fn=dict(type='pseudo_collate'),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='LRS3Dataset',
        data_root=data_root,
        ann_file=ann_test,
        pipeline=test_pipeline,
        modality='audiovisual',
        test_mode=True),
    collate_fn=dict(type='pseudo_collate'),
)

val_evaluator = dict(type='WordErrorRate')
test_evaluator = dict(type='WordErrorRate')
