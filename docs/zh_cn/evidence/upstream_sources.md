# 上游源码清单（不再 vendoring）

复现指南里的每条"代码事实"都来自下面几个仓库。**它们不放进本仓库**：可以直接
`git clone`，而 vendoring 第三方代码既有许可证问题，也会让"这份代码到底是我的还是
别人的"变得模糊。本文件记录"哪个仓库、哪个文件、看什么"，以及我核实到的结论。

## 1. Llama-AVSR

```bash
git clone https://github.com/umbertocappellazzo/Llama-AVSR
```

| 文件 | 核实到的内容 |
|---|---|
| `train.py` L130–144 | 三个 prompt 的默认值：`"Transcribe speech to text."` / `"Transcribe video to text."` / `"Transcribe speech and video to text."`（最后一个是 AVSR） |
| `train.py` L34 | `precision='bf16-true'`（Lightning，主权重就是 bf16） |
| `train.py` `--compression-mode` | 默认 `stack`（把连续 r 帧拼起来过线性层），不是卷积下采样 |
| `train.py` `--warmup-epochs` | `default=0`，但 help 里写 "Default: 5" —— help 是错的 |
| `models/Llama_LoRA.py` L36 | `self.scaling = ALPHA / self.rank`，且 `self.rank = lora_config.RANK`；瓶颈宽度 = `hidden_size / rank` ⇒ `--reduction-lora` 是**除数** |
| `models/Llama_LoRA.py` | LoRA 只加在 self-attn 的 **Q 和 V**（K/O 被注释掉），V 按 GQA 折半 |
| `models/modeling_AVSRLLM.py` `prepare_inputs()` | 序列布局是 `[BOS][av][prompt][target]`，prompt 取 `tokenizer(...).input_ids[:, 1:-1]` |
| `datamodule/transforms.py` L67–86 | `AddNoise`：`snr_levels = [-5,0,5,10,15,20,999999]`，用 `999999` 充当"不加噪" ⇒ 实际 1/7 clean（86%，不是论文的 75%） |
| `datamodule/data_module.py` L70 | `_batch_by_token_count()`：按帧数组 batch |
| `datamodule/babble_noise.wav` | 噪声文件直接放在仓库里（15 MB） |
| Model Zoo | `AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42`，1756 h，WER 0.748，可训参数 57 M |
| 2025-11 更新 | 新增 `--add-sink-loss`、`--layernorm-projector`（对应 arXiv:2510.22603） |

## 2. MMS-LLaMA

```bash
git clone https://github.com/JeongHun0716/MMS-LLaMA   # 默认分支是 master，不是 main
```

| 文件 | 核实到的内容 |
|---|---|
| `src/conf/mms-llama.yaml` L104–105 | `lora_rank: 16`、`lora_alpha: 32` ⇒ **论文"rank 16 / scaling 32"的出处**（peft 口径） |
| `src/conf/mms-llama.yaml` L63 | `max_update: 30000`、`max_epoch: 7` |
| `src/conf/mms-llama.yaml` L77 | `warmup_updates: 500`（与论文的 5k 不一致） |
| `src/conf/mms-llama.yaml` L109 | `modality_fuse: concat` ⇒ Table I/II 的 C_Concat |
| `src/conf/mms-llama.yaml` L111–114 | `use_qformer`、`queries_per_sec: 3`、`qformer_layers: 2`、`qformer_dim: 1024` ⇒ Q_former 那一行 |
| `src/model.py` L102–109 | `concat` 是 `torch.cat([...], dim=2)`（通道拼接）；`cross-att` 用 8 头 `Multimodal_Attention` |
| `src/model.py` L115 / L152–153 | 音频 50→25 Hz（k=2,s=2）；**无 Q-Former 时**音频 k=4/s=4、视频 k=2/s=2 ⇒ 12.5 Hz（论文是 25 Hz） |
| `src/dataset.py` L195–248 | `snr_levels = [-5,0,5,10,15,20]` + 独立的 `noise_prob`；噪声段 `random.randint` 随机截取（训练和评测都是） |
| `scripts/eval_snr.sh` | 评测：`generation.beam=5`、`generation.temperature=0.3`、`noise_prob=1`、`noise_snr=SNR` |
| README | 8×RTX3090 上 433 h 约 6 小时、1759 h 约 20 小时（它是 6.25 Hz token，本论文是 25 Hz） |

## 3. 其他

| 资源 | 用途 / 关键信息 |
|---|---|
| [av_hubert](https://github.com/facebookresearch/av_hubert) | `large_vox_iter5.pt`；配置 `conf/pretrain/large_vox_iter5.yaml`：24 层 / 1024 维 / 16 头 / `label_rate: 25` |
| 权重直链 | `https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt` |
| [auto_avsr](https://github.com/mpc001/auto_avsr) | `preparation/`：LRS3 嘴部 96×96 裁剪、landmark、16 s 分段、csv 标签 |
| [POT](https://pythonot.github.io/) | 交叉验证 Sinkhorn：`ot.bregman.sinkhorn_log(a, b, M, reg)`，**M 是代价矩阵**（传 `1-cos`），`reg` 就是 λ |
| [LAVCap (arXiv:2501.09291)](https://arxiv.org/abs/2501.09291) | 同样把 OT 耦合当软标签做 AV 对比学习，最值得对照的邻居工作 |
| [arXiv:2510.22603](https://arxiv.org/pdf/2510.22603) | attention sink / massive activation + pre-fusion LayerNorm（ICASSP 2026） |

## 4. 本地为什么不留副本

1. 这些仓库都能一条命令克隆，且都是 public；
2. 复现结论已经写进 `docs/zh_cn/replication_guide.md` 和本文件，**结论入库、代码不入库**；
3. 第三方源码进 git 会让 `git blame` / 许可证边界都变脏。

需要离线查阅时，按上面的 URL clone 到 `third_party/`（已在 `.gitignore` 里）即可。
