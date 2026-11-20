# Copyright (c) OTAVSR. All rights reserved.
"""Audio-only variant, used for the M0 sanity run before touching AV data.

With ``modality='audio'`` there is no fusion and only the ``a -> t`` branch of
the alignment loss is active, so this config answers two questions cheaply:

1. does the pipeline train at all (overfit 20 utterances to WER < 5 %)?
2. does ``L_align`` have the right magnitude before the visual encoder is in
   the loop?

It must reproduce Llama-AVSR's ASR numbers if you need an absolute check, but
that is not the point -- use it as a smoke test.
"""
model = dict(
    type='AVSRLLMClassifier',
    modality='audio',
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
    video_backbone=None,
    video_projector=None,
    fusion=None,
    head=dict(
        type='LLMDecoder',
        llm_path='meta-llama/Llama-3.2-3B',
        prompt='Transcribe speech and video to text.',
        frozen=True,
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
        s_tilde=None,
        rho=0.0,
        sinkhorn_iters=20,
        detach_q=True,
        loss_weight=0.1,
        fp32=True,
        reduction='sum'),
)
