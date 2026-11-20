# Copyright (c) OTAVSR. All rights reserved.
"""LRS3-TED, 433 h, clean training (Table I).

Every path below is relative to ``data_root``.  Run
``tools/analysis_tools/browse_dataset.py`` once: it prints the parsed
``(num_frames, transcript)`` pairs and the tensor shapes, which catches a wrong
CSV field order or a broken mouth-crop pipeline before a 30 k-step run.
"""
data_root = 'data/LRS3'
ann_train = 'labels/lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv'
ann_val = 'labels/lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv'
ann_test = 'labels/lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv'

sample_rate = 16000
max_frames = 400  # 16 s at 25 fps
tokenizer_path = 'meta-llama/Llama-3.2-3B'

train_pipeline = [
    dict(type='LoadAudioVideo', sample_rate=sample_rate, max_frames=max_frames),
    dict(type='MouthCrop', size=88, random=True),
    dict(type='RandomHorizontalFlip', prob=0.5),
    dict(type='WhisperLogMel'),
    dict(type='TokenizeTranscript', tokenizer_path=tokenizer_path, max_length=64),
    dict(type='PackAVSRInputs'),
]

test_pipeline = [
    dict(type='LoadAudioVideo', sample_rate=sample_rate, max_frames=max_frames),
    dict(type='MouthCrop', size=88, random=False),
    dict(type='WhisperLogMel'),
    dict(type='TokenizeTranscript', tokenizer_path=tokenizer_path, max_length=64),
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
