# Copyright (c) OTAVSR. All rights reserved.
"""Babble-noise augmentation (paper §II-A / §III-A).

The paper follows MMS-LLaMA: 75 % of the training clips get babble noise at an
SNR drawn uniformly from ``{-5, 0, 5, 10, 15, 20}`` dB, and the test set is
evaluated at a *fixed* SNR.

Three details that are easy to get wrong and that this class makes explicit:

* the noise segment is a **random crop** of the noise file, so evaluating the
  same model twice gives different WER. Pass ``seed`` for a reproducible
  evaluation, or pre-generate the noisy wavs once.
* with ``num_workers > 1`` every worker inherits the same seeded RNG, so all
  workers would draw the *same* noise segments. The seed is therefore mixed
  with ``torch.initial_seed()`` (which differs per worker) on first use.
* Llama-AVSR instead puts ``999999`` in the SNR list, which gives 1/7 clean
  clips (86 %, not 75 %). Use ``prob`` if you want the paper's setting.
"""
import random
from typing import Optional, Sequence

import torch

from otavsr.registry import TRANSFORMS

__all__ = ['AddBabbleNoise', 'mix_at_snr']

#: the six SNR levels of the paper
DEFAULT_SNR_LEVELS = (-5, 0, 5, 10, 15, 20)


def mix_at_snr(speech: torch.Tensor, noise: torch.Tensor,
               snr_db: float) -> torch.Tensor:
    """Add ``noise`` to ``speech`` at the requested SNR (RMS ratio)."""
    speech_power = speech.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    scale = torch.sqrt(speech_power / (noise_power * 10**(snr_db / 10.0)))
    return speech + scale * noise


@TRANSFORMS.register_module()
class AddBabbleNoise:
    """Mix a random babble segment into ``results['audio']``.

    Args:
        noise_path (str): ``babble_noise.wav`` (bundled by both official repos).
        prob (float): probability of adding noise in *training*; the paper uses
            0.75. Ignored when ``eval_snr`` is not ``None``.
        snr_levels (Sequence[int]): training SNR candidates.
        eval_snr (float, optional): fixed SNR for evaluation. ``None`` keeps
            the random training behaviour.
        seed (int, optional): fixes the noise crop and the SNR draw. **Use it
            for every evaluation run**, otherwise WER is not comparable.
        sample_rate (int): sample rate of the noise file.
    """

    def __init__(self,
                 noise_path: str,
                 prob: float = 0.75,
                 snr_levels: Sequence[int] = DEFAULT_SNR_LEVELS,
                 eval_snr: Optional[float] = None,
                 seed: Optional[int] = None,
                 sample_rate: int = 16000) -> None:
        self.noise_path = noise_path
        self.prob = prob
        self.snr_levels = tuple(snr_levels)
        self.eval_snr = eval_snr
        self.sample_rate = sample_rate
        self.seed = seed
        self._noise = None
        self._rng = None

    # ------------------------------------------------------------------ #
    def _get_rng(self) -> random.Random:
        """Seeded RNG, unique per dataloader worker.

        ``torch.initial_seed()`` is ``base_seed + worker_id`` inside a worker
        and the base seed in the main process, so mixing it in keeps the run
        reproducible while giving each worker a different noise sequence.
        """
        if self._rng is None:
            seed = self.seed
            if seed is not None:
                seed = (int(seed) + torch.initial_seed()) % (2**31 - 1)
            self._rng = random.Random(seed)
        return self._rng

    @property
    def noise(self) -> torch.Tensor:
        if self._noise is None:
            import torchaudio
            wav, sr = torchaudio.load(self.noise_path)
            if wav.dim() > 1:
                wav = wav.mean(0)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, int(sr),
                                                     self.sample_rate)
            self._noise = wav
        return self._noise

    def _sample_segment(self, num_samples: int) -> torch.Tensor:
        noise = self.noise
        if noise.numel() < num_samples:  # tile short noise files
            repeats = int(num_samples / noise.numel()) + 1
            noise = noise.repeat(repeats)
        start = self._get_rng().randint(0, noise.numel() - num_samples)
        return noise[start:start + num_samples]

    def transform(self, results: dict) -> dict:
        speech = results['audio']
        if speech.dim() > 1:
            speech = speech.squeeze(0)
        rng = self._get_rng()
        if self.eval_snr is not None:
            snr = float(self.eval_snr)
        else:
            if rng.random() >= self.prob:
                return results
            snr = float(rng.choice(self.snr_levels))
        segment = self._sample_segment(speech.numel())
        results['audio'] = mix_at_snr(speech, segment.to(speech.device), snr)
        results['snr_db'] = snr
        return results

    def __call__(self, results: dict) -> dict:
        return self.transform(results)
