# Copyright (c) OTAVSR. All rights reserved.
"""Transcript tokenisation with the LLaMA tokenizer."""
from typing import Optional

import torch

from otavsr.registry import TRANSFORMS

__all__ = ['TokenizeTranscript']


@TRANSFORMS.register_module()
class TokenizeTranscript:
    """Turn ``gt_text`` into ``target_ids``.

    The target does **not** contain BOS/EOS: the head adds the BOS and the
    EOS is appended here only if ``append_eos=True``. Keeping the token
    sequence free of BOS/EOS matters for the OT branch, because ``z_t`` would
    otherwise contain a token that every frame tries to match (guide §7.4).

    Args:
        tokenizer_path (str): HF hub id / local path of the LLaMA tokenizer.
        max_length (int): truncate longer transcripts (Llama-AVSR filters the
            label files at 25 tokens instead).
        append_eos (bool): append the EOS token to the target.
    """

    def __init__(self,
                 tokenizer_path: str = 'meta-llama/Llama-3.2-3B',
                 max_length: int = 64,
                 append_eos: bool = False) -> None:
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.append_eos = append_eos
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def transform(self, results: dict) -> dict:
        ids = self.tokenizer(
            results['gt_text'],
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True).input_ids
        target = torch.as_tensor(ids, dtype=torch.long)
        if self.append_eos:
            eos = self.tokenizer.eos_token_id
            target = torch.cat([target, torch.as_tensor([eos])])
        results['target_ids'] = target
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
