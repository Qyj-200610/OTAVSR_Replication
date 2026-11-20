# OTAVSR · OT 语义对齐 LLM-AVSR 复现

> 论文：**Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition**
> [arXiv:2607.09001](https://arxiv.org/abs/2607.09001)｜Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai（NICT / Academia Sinica）｜SLT 2026
> 官方**未开源**。本仓库按论文 + 两个上游基线（[Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR)、[MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA)）重建。

---

## 30 秒了解

| 维度 | 说明 |
|---|---|
| **是什么** | 一个可运行、可测试的 OT 对齐模块（论文唯一的新东西）+ 完整的主干接线（Whisper / AV-HuBERT / LLaMA-3.2-3B + LoRA） |
| **不是什么** | 不是"已经复现出论文数字"的仓库 —— 端到端训练需要 LRS3 433 h + 8 卡，本机（Windows/CPU）只验证了纯 PyTorch 部分 |
| **工程骨架** | 对齐 [mmpretrain](https://github.com/open-mmlab/mmpretrain)：`configs/` 用 `_base_` 继承、`otavsr/` 用 registry 注册、`tools/` 入口、`tests/` pytest |
| **当前状态** | `pytest tests -q` → **85 passed**（纯 CPU，5 秒）；28 个 config 全部可加载；13 个组件注册到 `MODELS` |
| **代码量** | 48 个 Python 文件，约 625 KB（不含数据/权重） |

---

## 先读哪份文档

| 你想做的事 | 读这个 | 大概多久 |
|---|---|---|
| **搞清论文到底改了什么、复现要注意什么** | [docs/zh_cn/replication_guide.md](docs/zh_cn/replication_guide.md) ← **主文档** | 40 min |
| 看懂代码结构、想加一个融合/损失 | [docs/zh_cn/project_structure.md](docs/zh_cn/project_structure.md) | 10 min |
| 核对论文数字、判断"复现到什么程度算成功" | [docs/zh_cn/evidence/paper_wer_logs.md](docs/zh_cn/evidence/paper_wer_logs.md) | 10 min |
| 对照官方仓库的某个实现细节 | [docs/zh_cn/evidence/upstream_sources.md](docs/zh_cn/evidence/upstream_sources.md) | 5 min |

指南里的每个 `✅/⚠️/🆕` 标记都对应一条证据；**先读指南的 §0（15 行修订摘要）和 §4.3（方差分析）**，其余按需。

---

## 5 分钟自检（不需要数据、不需要 GPU）

```bash
pip install -r requirements/runtime.txt -r requirements/mptest.txt

python -m pytest tests -q                      # 85 passed
python tools/analysis_tools/check_ot_numerics.py   # 验证论文式(7) 的符号、bucket、mask
python tools/misc/print_config.py configs/table5_sota/setting3.py --keys model.ot_loss
```

三条都通过，说明 registry、config 继承、OT 的数值约定都没被破坏。

---

## 代码结构

```
OTAVSR_Replication/
├── configs/                  # 一个 .py = 论文表格里的一格（22 个实验 + 6 个 _base_）
│   ├── _base_/{models,datasets,schedules,default_runtime.py}
│   ├── table1_clean/         # Table I  ：5 种融合 × 干净训练
│   ├── table2_noisy/         # Table II ：5 种融合 × 75% 加噪（论文默认）
│   ├── table3_bucket/        # Table III：bucket 开/关
│   ├── table4_alpha/         # Table IV ：α = 0.05…0.9
│   ├── table5_sota/          # Table V  ：三组 (α, s̃, λ)
│   └── eval_lrs3_433h_snr.py # 评测入口，SNR 由命令行给
├── otavsr/                   # 主包
│   ├── utils/sinkhorn.py             # ★ log 域 Sinkhorn + bucket + 诊断
│   ├── models/losses/ot_alignment.py # ★ L_align（式 10-11）
│   ├── models/classifiers/avsr_llm.py# ★ 整机：backbone→neck→fusion→head→loss
│   ├── models/necks/fusion/          # ★ 五种融合（式 12）
│   ├── models/{backbones,heads}/     # Whisper / AV-HuBERT / LLMDecoder
│   ├── datasets/{transforms,samplers} # LRS3 + 噪声/裁剪/mel + 按帧 batch
│   ├── evaluation/metrics/wer.py     # WER（自带编辑距离）
│   └── engine/hooks/ot_monitor.py    # OT 诊断量的监控
├── tools/                    # train.py / test.py / print_config.py / check_ot_numerics.py
├── tests/                    # 85 用例，5 个文件
├── docs/zh_cn/               # 4 份文档（见上表）
└── resources/                # 论文原图
```

**只看四个文件就能读懂全部**：`otavsr/utils/sinkhorn.py`（数学）→ `otavsr/models/losses/ot_alignment.py`（损失）→ `otavsr/models/classifiers/avsr_llm.py`（整机）→ `configs/_base_/models/avsr_llm_c_concat.py`（超参）。

---

## 论文公式 → 代码

| 论文 | 内容 | 代码位置 |
|---|---|---|
| 式(1) | `z_a / z_v / z_t = FC(Embed(target))` | `classifiers/avsr_llm.py::extract_feat` |
| 式(2)(3) | 熵正则 OT 目标 | `utils/sinkhorn.py::solve_ot` |
| 式(4) | 余弦相似度 | `utils/sinkhorn.py::cosine_sim` |
| 式(5)(6) | 边缘约束（非方阵 `L×L_t`） | `solve_ot` 里的 `log_p` / `log_q` |
| 式(7) | Gibbs 核 ⚠️ **论文符号有误** | `log_kernel = -(1 - sim_aug) / lam` |
| 式(8) | Sinkhorn 迭代 | `utils/sinkhorn.py::log_sinkhorn` |
| 式(9) | virtual bucket 扩边 | `solve_ot` 里的 `sim_aug` + `rho` |
| 式(10)(11) | `L_align`（softmax 沿 text 维） | `losses/ot_alignment.py::align_loss` |
| 式(12) | 五种融合 | `models/necks/fusion/*.py` |
| 式(13) | `h_av = Proj_av(h_fusion)` | 各 fusion 里的 `self.proj` |
| 式(14) | `L_AR`（只对 target） | `heads/llm_decoder.py::forward` |
| 式(15) | `L_total = L_AR + α·L_align` | `classifiers/avsr_llm.py::loss` |

---

## 四处"必须自己定"的口径

论文没写清的地方，选择都在这里，**改之前请先读 docstring**：

| 口径 | 论文写法 | 本项目 | 为什么 |
|---|---|---|---|
| **Gibbs 核符号** | 式(7) `K=exp(-S/λ)`，与式(2) 的 `max` 矛盾 | `cost=1-cos` ⇒ `K=exp(-(1-S)/λ)` | 照抄式(7) 会做**反向传输**（实测 corr(Q,S)=−0.75） |
| **bucket 边缘** | 只给 \(\tilde S\)，没给 \(\tilde p/\tilde q\) | bucket 列取硬边缘 ρ；真实 token 取 (1−ρ)/L_t | 平衡 OT 下 **ρ 决定丢多少、s̃ 决定丢谁** |
| **L_align 归一化** | 式(10) 无 `1/L` | `reduction='sum'` | `'mean'` 会让 L_align 小 L 倍，α=0.1 就失去论文含义 |
| **评测噪声** | 只说加噪 | 噪声段**随机截取**，用 `seed` 固定 | 否则同一模型两次评测 WER 差 0.3 |

---

## 已经核到的坑（摘要，详细见指南 §8）

| # | 坑 | 后果 |
|---|---|---|
| 1 | 式(7) 的符号 | 训练必崩，`L_align` 卡住不降 |
| 2 | Llama-AVSR 的 `--reduction-lora` 是"隐藏维 ÷ 秩"的**除数** | 填 16 得到秩 192 而非 16；论文口径要填 **192** |
| 3 | Llama-AVSR 默认跑出来是 **T_Concat**（11.41），不是 C_Concat（8.34） | baseline 目标一开始就错 |
| 4 | 加噪 75% 两个仓库实现不同（Llama-AVSR 用 `999999` 凑出 1/7 clean） | noisy 数据分布对不上 |
| 5 | 评测噪声段随机 | 同模型两次 −5 dB WER 差 0.3–0.5 |
| 6 | 训练 / 测试标签分段不同（16 s vs 24 s） | L 差 50% |
| 7 | `warm-up` 论文说 5k，两个官方实现分别是 0 epoch / 500 step | 训练曲线形状不同 |
| 8 | bucket 在 clean 上反而掉点（0.76→0.80），作者仍称 "slight improvement" | 别拿它当主收益 |

还有一个**方差事实**：作者留在 LaTeX 注释里的 5 组 run，clean WER 在 **0.657–0.789**（mean±sd = 0.740±0.054）。所以"0.89→0.71"成立（≈2.7σ），但"0.80 vs 0.76"不成立（0.7σ）。

---

## 跑真实验

```bash
# 训练（单机多卡用 tools/dist_train.sh）
python tools/train.py configs/table2_noisy/c_concat.py --work-dir work_dirs/t2_cconcat

# 评测：999 = clean，论文用 -5 / 0 / 5
python tools/test.py configs/eval_lrs3_433h_snr.py \
    --checkpoint work_dirs/t2_cconcat/iter_30000.pth \
    --cfg-options test_dataloader.dataset.pipeline.1.eval_snr=0
```

前置条件：auto-avsr 预处理好的 LRS3（`data/LRS3/labels/*.csv` + `lrs3/lrs3_video_seg16s/`）、
`large_vox_iter5.pt`、`openai/whisper-medium.en`、`meta-llama/Llama-3.2-3B`；
`av_hubert` 的 fairseq 子模块需按 [requirements/README.md](requirements/README.md) 指定的版本组合自行编译。

算力：一次 433 h / 30 k step 约 18–30 h（8×3090）；跑完全部 5 张表 ≈ 20 次训练 ≈ 5–7 周，详见指南 §5.0。

---

## 常见问题

**Q: `ModuleNotFoundError: mmengine`？**
`pip install mmengine`（本项目只依赖 mmengine，不依赖 mmcv / mmpretrain）。

**Q: 没有 GPU / 没有权重，能跑什么？**
`pytest tests`（85 用例）、`check_ot_numerics.py`、`print_config.py` 全都不需要 GPU 和权重。主干背板用 mock 验证过 wiring。

**Q: `s_tilde` 该设多少？**
不是固定的 0.5。先看 `OTMonitorHook` 打出的 `row_max_cos`；若它明显低于 s̃（说明在丢有用的帧），把 s̃ 设成 `row_max_cos` 的 10–20 分位数。训练初期 `row_max_cos` 只有 0.2–0.3，此时 s̃=0.5 等于随机丢帧。

**Q: 为什么 `L_align` 有 2~6 这么大？**
论文式(10) 是不带 `1/L` 的双重求和。这是刻意的：这样 α=0.1 才和 Table IV 的扫描区间一致。想换成按帧均值就同时把 α ×400。

**Q: 能直接用 Llama-AVSR 的预训练 ckpt 吗？**
不建议当初始化。它是 Llama-3.1-8B（hidden 4096）+ T_Concat（2L token），本仓库是 Llama-3.2-3B（3072）+ C_Concat（L token），`Proj_a/Proj_v/Proj_av` 的形状都对不上。它的价值是"确认自己的数据管线和 WER 计算是通的"，不是拿来初始化。

**Q: 为什么仓库里没有 paper PDF / 上游源码？**
结论入库、代码不入库。论文可 `https://arxiv.org/src/2607.09001` 一键重下；上游仓库的"哪个文件哪一行"记在 [upstream_sources.md](docs/zh_cn/evidence/upstream_sources.md)。

---

## 引用

```bibtex
@article{lu2026otavsr,
  title={Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition},
  author={Lu, Xugang and Shen, Peng and Tsao, Yu and Kawai, Hisashi},
  journal={arXiv preprint arXiv:2607.09001},
  year={2026}
}
```

致谢：[Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR)、[MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA)、
[av_hubert](https://github.com/facebookresearch/av_hubert)、[auto_avsr](https://github.com/mpc001/auto_avsr)、[mmpretrain](https://github.com/open-mmlab/mmpretrain)。
