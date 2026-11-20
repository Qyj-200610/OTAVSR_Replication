# Requirements

Three tiers, because the heavy dependencies are only needed once you have the
data and the checkpoints:

| file | when you need it |
|---|---|
| `runtime.txt` | always: `mmengine`, `torch`, `numpy`. Enough for the OT module, the configs and `tests/` (84 tests, no GPU). |
| `mptest.txt` | `pytest` suite. |
| `optional.txt` | the full AVSR stack: `transformers` (Whisper + LLaMA), `peft` (LoRA), `torchaudio`/`decord` (IO), `av_hubert` + `fairseq` (visual encoder). |

## Known-good combination

The visual encoder is the constraint: `av_hubert` builds against `fairseq`,
which does not compile on new Python/torch. MMS-LLaMA pins this stack, and it
is the one to copy:

```
python 3.9
torch 2.1.2 / torchvision 0.16.2 / torchaudio 2.1.2 (cu121)
transformers 4.47.1
peft 0.14.0
omegaconf 2.0.6   # pip > 24.1 needs `pip==24.0` for these two
hydra-core 1.0.7
```

Install the AV-HuBERT submodule with

```bash
cd third_party/av_hubert && pip install --editable ./fairseq
```

(the submodule is not vendored in this repo on purpose -- see
`docs/zh_cn/evidence/upstream_sources.md`).
