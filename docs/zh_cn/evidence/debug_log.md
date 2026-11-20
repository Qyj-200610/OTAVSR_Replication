# Debug 记录（2026-11-20）

对整个仓库做的第一轮系统性 debug。结论：**8 个真实缺陷**（4 个会直接崩、4 个静默出错），
全部已修复并固化为回归用例。当前 `pytest tests -q` → **171 passed / 5 skipped**。

## 一、会直接崩的（首次真跑就会发现）

| # | 位置 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | `models/classifiers/base.py` | `MODELS.build(cfg.model)` → `TypeError: unexpected keyword argument 'data_preprocessor'` | 所有 config 都写了 `data_preprocessor=dict(type=...)`，但 `BaseClassifier.__init__` 没有这个参数 | 把 `data_preprocessor` 透传给 `BaseModel` |
| 2 | `models/data_preprocessors/avsr_preprocessor.py` | `KeyError: 'AVSRDataPreprocessor is not in the mmengine::model registry'` | `BaseModel` 用 **mmengine 自己的** `MODELS` registry 构建 preprocessor，而我们只注册在 `otavsr.MODELS`（子 registry） | 额外注册进 mmengine 的 registry |
| 3 | `datasets/transforms/formatting.py` | `AssertionError: 40 should be a <class 'torch.Tensor'> but got <class 'int'>` | `PackAVSRInputs` 把 CSV 里的 int `num_frames` 塞进带 `dtype=torch.Tensor` 校验的 `set_field` | `num_frames` 不走 `meta_keys`，显式转 tensor |
| 4 | `models/classifiers/avsr_llm.py` | 评测抛 `AttributeError: 'str' object has no attribute 'get'` | `predict` 返回 `List[str]`，而 mmengine metric 需要 mapping（`sample['pred_text']`） | `predict` 返回 `dict(pred_text, gt_text, sample_id)`；metric 对非 mapping 给明确报错 |

## 二、静默出错的（不崩，但结果不对）

| # | 位置 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 5 | `models/classifiers/avsr_llm.py` + `models/backbones/avhubert.py` | 视觉 mask 把几乎全部帧标成 padding，相当于只用前 2 帧的视觉信息 | 长度契约不一致：分类器传的是**秒**，AV-HuBERT 当**帧数**用（1.6 → 只有 2 帧有效） | 全项目统一为"秒"，两个 backbone 各自乘自己的帧率 |
| 6 | `datasets/transforms/loading.py` + `transforms.py` | AV-HuBERT 收到 `[T,H,W,C]` 而不是 `[B,T,C,H,W]` | 解码器输出 `[T,H,W,C]`，但编码器/fairseq 要 channel-first | `LoadAudioVideo` 里 permute 成 `[T,C,H,W]`，crop/flip 改到 dim 2:4 / dim 3 |
| 7 | `datasets/samplers/max_frames.py` | 用真实 dataset 时 batch 数为 0（epoch 空转） | `BaseDataset(serialize_data=True)` 会把 `data_list` **清空**，`getattr(dataset,'data_list')` 返回 `[]`（不是 `None`），fallback 没触发 | 空列表也走 `get_data_info(i)` |
| 8 | `models/heads/llm_decoder.py` | prompt 少最后一个 token（`"…to text"` 而不是 `"…to text."`） | 照抄 Llama-AVSR 的 `input_ids[:, 1:-1]`，但 LLaMA-3 tokenizer 不加 EOS | 只在首/尾确实是特殊 token 时才剥 |

## 三、配置层面的两个"跑完 30k 步才发现"的坑

| # | 现象 | 修复 |
|---|---|---|
| 9 | `CheckpointHook(interval=1, by_epoch=False)` → 30k 次存盘；`save_best='auto'` 对未知指标默认 `rule='greater'` → **保存 WER 最差的那个 ckpt** | `interval=3000`、`save_best='wer'`、`rule='less'` |
| 10 | `mp_start_method='fork'` 在 Windows 上不存在 → dataloader worker 起不来 | 按平台选 `spawn` |

## 四、其他健壮性修正（不改变结果，但省排查时间）

- `AmpOptimWrapper(dtype='bfloat16')` 仍会做动态 loss scaling：改成 `loss_scale=dict(init_scale=1.0, enabled=False)`（bf16 不需要缩放）。
- `apis/train.py` 用 `runner.resume(work_dir)` —— `Runner.resume` 要的是**文件**不是目录；改成 `cfg.resume=True` 交给 `train()`。
- `AddBabbleNoise` 的 seed 在多 worker 下每个 worker 相同（噪声段重复）：混入 `torch.initial_seed()`。
- 音频截断原先按 `max_frames` 而不是按**视频实际帧数**：短片会音画错位。
- 声学/视觉序列长度差 1 帧（mel 的 `1 + T/hop`）会让融合 `cat` 崩：`extract_feat` 里裁到较短的一侧。
- `WhisperLogMel` 补了说明：torchaudio 的 mel 滤波器与 Whisper 官方（librosa/slaney）差几个 dB，需要 bit-exact 就换 `WhisperFeatureExtractor`。
- `LRS3Dataset.METAINFO` 里的字符串被 mmengine 当成"meta info 文件路径"并告警 → 清空。
- `WordErrorRate` 未设 `default_prefix` → 每次构造都告警 → 设为 `''`。
- 清理 8 处未使用的 import；删除 `sinkhorn.py` 里的死函数。

## 五、新增的回归用例（85 → 176）

| 文件 | 覆盖 |
|---|---|
| `tests/test_models/test_build.py` | 22 个实验 config 全部 `MODELS.build`；未知参数必须报错；audio-only 分支 |
| `tests/test_models/test_backbones.py` | 长度按秒换算（`1.6 s → 40 帧`）、Whisper padding 归零、显式 mask 优先 |
| `tests/test_models/test_fusion.py` | 新增 predict 返回 dict、音画长度差 1 帧时自动裁剪 |
| `tests/test_datasets/test_pipeline.py` | `PackAVSRInputs`(int num_frames)、`MouthCrop`/flip 的 `[T,C,H,W]` 布局、preprocessor padding |
| `tests/test_datasets/test_lrs3.py` | CSV 两种字段布局、`load_data_list`、空标注报错 |
| `tests/test_datasets/test_sampler_noise.py` | 序列化 dataset 下按帧 batch（缺陷 7） |
| `tests/test_runtime_config.py` | hook / optimizer / scheduler 全部可构建；checkpoint 规则；SNR 旋钮位置 |

## 六、还没验证的部分

- **真实权重路径**：`transformers` / `peft` / `fairseq` 都没装，Whisper / AV-HuBERT / LLaMA 的
  forward 只用 fake 模块验证了**长度与 mask 语义**，没有跑过真实的 fairseq `extract_features`
  —— 那里的签名在不同 av_hubert 版本上可能不同（代码里有 TODO）。
- **真实数据路径**：`decord` / `torchaudio` 缺失，`LoadAudioVideo` 只做了布局层面的单元测试。
- **多卡**：`dist_train.sh` 未在真实机器上跑过。
