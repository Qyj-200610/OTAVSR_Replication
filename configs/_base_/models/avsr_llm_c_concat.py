# Copyright (c) OTAVSR. All rights reserved.
"""The paper's model: Whisper-medium + AV-HuBERT-large + LLaMA-3.2-3B + OT.

Every block maps to a section of the guide:

* ``audio_backbone``   Whisper-medium.en encoder, frozen, 50 Hz (§2.2)
* ``audio_downsample`` one stride-2 Conv1d, 50 Hz -> 25 Hz (§3.1)
* ``audio_projector``  ``Proj_a``: 1024 -> 3072, linear (§3.1)
* ``video_backbone``   AV-HuBERT large, frozen, 25 Hz / 1024 (§2.2)
* ``video_projector``  ``Proj_v``: 1024 -> 3072 (§3.1)
* ``fusion``           ``C_Concat`` + ``Proj_av`` 6144 -> 3072 (§3.6)
* ``head``             LLaMA-3.2-3B + LoRA r=16 / alpha=32 (§2.4)
* ``ot_loss``          OT alignment on ``z_a``, ``z_v`` vs ``z_t`` (§3.2-3.4)
"""
model = dict(
    type='AVSRLLMClassifier',
    modality='audiovisual',
    data_preprocessor=dict(type='AVSRDataPreprocessor', pad_value=0.0),
    audio_backbone=dict(
        type='WhisperEncoder',
        model_name='openai/whisper-medium.en',
        frozen=True,
        feature_rate=50),
    audio_downsample=dict(
        type='TemporalDownsample',
        channels=1024,
        kernel_size=3,
        stride=2,
        trainable=True),
    audio_projector=dict(
        type='ModalityProjector',
        in_channels=1024,
        out_channels=3072,
        layernorm=False),
    video_backbone=dict(
        type='AVHuBERTEncoder',
        checkpoint='pretrained_models/avhubert/large_vox_iter5.pt',
        frozen=True,
        frame_rate=25),
    video_projector=dict(
        type='ModalityProjector',
        in_channels=1024,
        out_channels=3072,
        layernorm=False),
    fusion=dict(
        type='ChannelConcatFusion', in_channels=3072, out_channels=3072),
    head=dict(
        type='LLMDecoder',
        llm_path='meta-llama/Llama-3.2-3B',
        prompt='Transcribe speech and video to text.',
        frozen=True,
        # peft semantics: r is the true rank, lora_alpha is peft's numerator.
        # Llama-AVSR's `--reduction-lora` is a *divisor* -- see guide §2.4.
        lora=dict(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias='none'),
        layout='bos_av_prompt',
        max_dec_tokens=32,
        num_beams=15),
    ot_loss=dict(
        type='OTAlignmentLoss',
        lam=0.1,
        tau=0.07,
        s_tilde=0.5,
        rho=0.2,
        bucket_row=False,
        sinkhorn_iters=50,
        detach_q=True,
        loss_weight=0.1,
        fp32=True,
        reduction='sum'),
)
