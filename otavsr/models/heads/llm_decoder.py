# Copyright (c) OTAVSR. All rights reserved.
"""The LLaMA decoder: prompt assembly, LoRA fine-tuning, teacher forcing."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from otavsr.registry import MODELS

__all__ = ['LLMDecoder', 'assemble_avsr_inputs']

#: LLaMA-3.2 instruct special tokens
LLAMA3_BOS = 128000
LLAMA3_EOS = 128001


def assemble_avsr_inputs(av_embeds: torch.Tensor,
                         prompt_embeds: torch.Tensor,
                         target_ids: torch.Tensor,
                         embed_fn,
                         bos_token_id: int = LLAMA3_BOS,
                         layout: str = 'bos_av_prompt',
                         ignore_index: int = -100
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``inputs_embeds`` and ``labels`` for teacher forcing.

    Two layouts are supported because the paper and the official code disagree
    (see ``docs/zh_cn/replication_guide.md`` §2.3):

    * ``'bos_av_prompt'`` -- Llama-AVSR: ``[BOS][av][prompt][target]``
    * ``'prompt_av_bos'`` -- paper Fig.1: ``[prompt][av][BOS][target]``

    Only the target tokens are supervised; everything before them is
    ``ignore_index``.

    Args:
        av_embeds (Tensor): ``[B, L, d]`` fused audio-visual embeddings.
        prompt_embeds (Tensor): ``[1 or B, P, d]`` instruction embeddings.
        target_ids (Tensor): ``[B, T]`` target token ids.
        embed_fn (Callable): maps ``ids -> embeds`` (the LLM token embedding).
        bos_token_id (int): begin-of-sequence token id.
        layout (str): one of ``'bos_av_prompt'`` / ``'prompt_av_bos'``.
        ignore_index (int): value used for unsupervised positions.

    Returns:
        Tuple[Tensor, Tensor]: ``inputs_embeds`` ``[B, 1+P+L+T, d]`` and
        ``labels`` with the same sequence length.
    """
    batch = av_embeds.shape[0]
    if prompt_embeds.shape[0] == 1 and batch > 1:
        prompt_embeds = prompt_embeds.expand(batch, -1, -1)
    target_embeds = embed_fn(target_ids)
    bos_ids = torch.full((batch, 1),
                         bos_token_id,
                         dtype=torch.long,
                         device=av_embeds.device)
    bos_embeds = embed_fn(bos_ids)

    def ignored(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_full(tensor.shape[:2], ignore_index)

    if layout == 'bos_av_prompt':
        pieces = [bos_embeds, av_embeds, prompt_embeds, target_embeds]
        labels = [
            ignored(bos_embeds), ignored(av_embeds), ignored(prompt_embeds),
            target_ids
        ]
    elif layout == 'prompt_av_bos':
        pieces = [prompt_embeds, av_embeds, bos_embeds, target_embeds]
        labels = [
            ignored(prompt_embeds), ignored(av_embeds), ignored(bos_embeds),
            target_ids
        ]
    else:
        raise ValueError(f'unknown layout {layout!r}')
    return torch.cat(pieces, dim=1), torch.cat(labels, dim=1)


@MODELS.register_module()
class LLMDecoder(BaseModule):
    """LLaMA-3.2-3B decoder with LoRA, used as the ASR head.

    Args:
        llm_path (str): HF hub id or local path.
        prompt (str): the instruction text. Both official baselines use
            ``"Transcribe speech and video to text."`` for AVSR.
        frozen (bool): freeze the base LLM weights (LoRA is added on top).
        lora (dict, optional): **peft** semantics, i.e. ``r`` is the true rank
            and ``lora_alpha`` the peft scaling numerator (``alpha / r`` is the
            effective scale). The paper reports ``rank 16, scaling 32`` ->
            ``r=16, lora_alpha=32``. NOTE: Llama-AVSR's own
            ``--reduction-lora`` is a *divisor* of the hidden size, not the
            rank; see the guide §2.4 before porting a config.
        target_modules (list): LoRA modules; ``q/v`` is what both baselines do.
        layout (str): see :func:`assemble_avsr_inputs`.
        max_dec_tokens (int): generation budget (Llama-AVSR uses 32).
        num_beams (int): beam width (Llama-AVSR uses 15, MMS-LLaMA 5).
    """

    def __init__(self,
                 llm_path: str = 'meta-llama/Llama-3.2-3B',
                 prompt: str = 'Transcribe speech and video to text.',
                 frozen: bool = True,
                 lora: Optional[dict] = None,
                 bos_token_id: int = LLAMA3_BOS,
                 eos_token_id: int = LLAMA3_EOS,
                 layout: str = 'bos_av_prompt',
                 ignore_index: int = -100,
                 max_dec_tokens: int = 32,
                 num_beams: int = 15,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.llm_path = llm_path
        self.prompt = prompt
        self.frozen = frozen
        self.lora = dict(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias='none') if lora is None else lora
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.layout = layout
        self.ignore_index = ignore_index
        self.max_dec_tokens = max_dec_tokens
        self.num_beams = num_beams
        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # lazy HF model
    # ------------------------------------------------------------------ #
    def _build(self):
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoTokenizer, LlamaForCausalLM
        except ImportError as error:  # pragma: no cover - env dependent
            raise ImportError(
                'LLMDecoder needs `transformers` and `peft` plus an accepted '
                f'licence for {self.llm_path!r}.') from error
        llm = LlamaForCausalLM.from_pretrained(
            self.llm_path, torch_dtype=torch.bfloat16)
        if self.frozen:
            llm.requires_grad_(False)
        llm = get_peft_model(llm, LoraConfig(**self.lora))
        return llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._build()
        return self._llm

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        return self._tokenizer

    # ------------------------------------------------------------------ #
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings of the (frozen or LoRA-wrapped) LLM."""
        return self.llm.model.embed_tokens(token_ids)

    def prompt_embeds(self, device=None) -> torch.Tensor:
        """Instruction embeddings with the leading/trailing specials removed.

        ``input_ids[:, 1:-1]`` is what Llama-AVSR does, so the prompt does not
        inject a second BOS/EOS pair.
        """
        ids = self.tokenizer(self.prompt, return_tensors='pt').input_ids
        ids = ids[:, 1:-1]
        if device is not None:
            ids = ids.to(device)
        return self.embed_tokens(ids)

    def forward(self, av_embeds: torch.Tensor,
                target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Teacher-forced cross entropy (``L_AR`` of Eq.(14))."""
        embeds, labels = assemble_avsr_inputs(
            av_embeds,
            self.prompt_embeds(av_embeds.device),
            target_ids,
            self.embed_tokens,
            bos_token_id=self.bos_token_id,
            layout=self.layout,
            ignore_index=self.ignore_index)
        out = self.llm(inputs_embeds=embeds, labels=labels)
        return {'loss_ar': out.loss}

    @torch.no_grad()
    def generate(self, av_embeds: torch.Tensor) -> List[torch.Tensor]:
        """Greedy/beam decoding from ``[prompt][av][BOS]`` only."""
        batch = av_embeds.shape[0]
        prompt = self.prompt_embeds(av_embeds.device)
        if prompt.shape[0] == 1 and batch > 1:
            prompt = prompt.expand(batch, -1, -1)
        bos = self.embed_tokens(
            torch.full((batch, 1),
                       self.bos_token_id,
                       dtype=torch.long,
                       device=av_embeds.device))
        prefix = torch.cat([prompt, av_embeds, bos], dim=1)
        generated = self.llm.generate(
            inputs_embeds=prefix,
            max_new_tokens=self.max_dec_tokens,
            num_beams=self.num_beams,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
            do_sample=False)
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated
        ]
