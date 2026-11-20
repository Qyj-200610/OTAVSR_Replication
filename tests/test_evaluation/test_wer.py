# Copyright (c) OTAVSR. All rights reserved.
"""Tests of the WER metric and its text normalisation."""
import torch

from otavsr.evaluation.metrics import (WordErrorRate, compute_wer, levenshtein,
                                       normalize_text)
from otavsr.registry import METRICS


def _sample(prediction, reference):
    return dict(pred_text=prediction, gt_text=reference)


class TestTextNormalisation:

    def test_lowercase_and_punctuation(self):
        assert normalize_text('Hello, World!') == 'hello world'
        assert normalize_text('  a   b ') == 'a b'

    def test_levenshtein(self):
        assert levenshtein(['a', 'b'], ['a', 'b']) == 0
        assert levenshtein(['a', 'b'], ['a']) == 1
        assert levenshtein(['a'], ['a', 'b']) == 1
        assert levenshtein(['a', 'b'], ['b', 'a']) == 2

    def test_corpus_wer(self):
        references = ['the cat sat', 'a dog']
        hypotheses = ['the cat sits', 'a dog']
        # 1 substitution out of 5 reference words -> 20 %
        assert abs(compute_wer(references, hypotheses) - 20.0) < 1e-6


class TestWordErrorRateMetric:

    def test_registered(self):
        assert 'WordErrorRate' in METRICS.module_dict

    def test_compute(self):
        metric = WordErrorRate()
        metric.process(None, [_sample('HELLO, world!', 'hello world')])
        results = metric.compute_metrics(metric.results)
        assert results['wer'] == 0.0

    def test_error_is_counted(self):
        metric = WordErrorRate()
        metric.process(None, [_sample('hello there', 'hello world')],
                       )
        metric.process(None, [_sample('hello world', 'hello world')])
        results = metric.compute_metrics(metric.results)
        assert abs(results['wer'] - 25.0) < 1e-6  # 1 error in 4 reference words
