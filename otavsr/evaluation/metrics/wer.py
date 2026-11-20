# Copyright (c) OTAVSR. All rights reserved.
"""WER metric, implemented without third-party edit-distance packages."""
import string
from typing import Any, Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric

from otavsr.registry import METRICS

__all__ = ['WordErrorRate', 'levenshtein', 'normalize_text', 'compute_wer']

_PUNCTUATION = str.maketrans('', '', string.punctuation)


def normalize_text(text: str) -> str:
    """Lower-case and strip punctuation, as the official baselines do."""
    return ' '.join(text.lower().translate(_PUNCTUATION).split())


def levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Token-level edit distance (two-row DP, no external dependency)."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_token != hyp_token)))  # substitution
        previous = current
    return previous[-1]


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER in percent."""
    total_errors = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = normalize_text(reference).split()
        hyp_tokens = normalize_text(hypothesis).split()
        total_errors += levenshtein(ref_tokens, hyp_tokens)
        total_words += len(ref_tokens)
    return 100.0 * total_errors / max(total_words, 1)


@METRICS.register_module()
class WordErrorRate(BaseMetric):
    """Corpus WER of ASR / AVSR predictions.

    Args:
        collect_device (str): ``'cpu'`` or ``'gpu'``.
        prefix (str): metric name prefix. Empty by default so the key is plain
            ``wer`` (``None`` triggers mmengine's "prefix is not set" warning).
        strip_punctuation (bool): apply :func:`normalize_text`.
    """

    default_prefix = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 strip_punctuation: bool = True) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.strip_punctuation = strip_punctuation

    def _norm(self, text: str) -> str:
        return normalize_text(text) if self.strip_punctuation else text

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Collect ``pred_text`` / ``gt_text`` of every sample."""
        for sample in data_samples:
            if not hasattr(sample, 'get'):
                raise TypeError(
                    'WordErrorRate expects mapping-like predictions with '
                    "'pred_text'/'gt_text', got "
                    f'{type(sample).__name__}. A model whose `predict` returns '
                    'bare strings will silently break evaluation.')
            prediction = sample.get('pred_text', '')
            reference = sample.get('gt_text', '')
            self.results.append((reference, prediction))

    def compute_metrics(self, results: List[Any]) -> Dict[str, float]:
        total_errors = 0
        total_words = 0
        for reference, prediction in results:
            ref_tokens = self._norm(reference).split()
            hyp_tokens = self._norm(prediction).split()
            total_errors += levenshtein(ref_tokens, hyp_tokens)
            total_words += len(ref_tokens)
        return dict(wer=100.0 * total_errors / max(total_words, 1))
