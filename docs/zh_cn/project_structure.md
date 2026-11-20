# 工程结构（对齐 mmpretrain）

本项目的目录骨架照抄 [mmpretrain](https://github.com/open-mmlab/mmpretrain) v1.x：
`configs/` 用 `_base_` 继承、`otavsr/` 用 registry 注册组件、`tools/` 是入口脚本、`tests/` 是 pytest。
换成 mmpretrain 的读者可以按"概念映射表"直接定位。

```
OTAVSR_Replication/
├── configs/                       # 实验配置：一个文件 = 论文表格里的一格
│   ├── _base_/
│   │   ├── default_runtime.py     # hook / 日志 / checkpoint
│   │   ├── datasets/              # LRS3 干净训练、噪声训练（含 SNR 评测）
│   │   ├── models/                # 完整模型定义 + 纯音频冒烟版
│   │   └── schedules/             # Adam 1e-4 + 5k warmup + 30k steps
│   ├── table1_clean/              # Table I：5 种融合（干净训练）
│   ├── table2_noisy/              # Table II：5 种融合（75% 加噪训练）
│   ├── table3_bucket/             # Table III：bucket 开关
│   ├── table4_alpha/              # Table IV：alpha 扫描
│   ├── table5_sota/               # Table V：3 组 setting
│   └── eval_lrs3_433h_snr.py      # 评测入口（SNR 由命令行指定）
├── otavsr/                        # 主包（= mmpretrain/）
│   ├── registry.py                # MODELS/DATASETS/TRANSFORMS/METRICS/HOOKS/SAMPLERS
│   ├── apis/                      # train_model / test_model
│   ├── datasets/                  # LRS3Dataset + samplers + transforms
│   │   ├── transforms/            # loading / noise / transforms / tokenize / formatting
│   │   └── samplers/              # MaxFramesBatchSampler（按帧数组 batch）
│   ├── structures/                # AVSRDataSample
│   ├── models/
│   │   ├── backbones/             # WhisperEncoder, AVHuBERTEncoder（冻结）
│   │   ├── necks/                 # TemporalDownsample, ModalityProjector
│   │   │   └── fusion/            # C_Concat / Add / Cross_att / Q_former / T_Concat
│   │   ├── heads/                 # LLMDecoder（LLaMA + LoRA + prompt 拼装）
│   │   ├── losses/                # OTAlignmentLoss（式 10-11）
│   │   ├── data_preprocessors/    # 变长 batch padding
│   │   └── classifiers/           # AVSRLLMClassifier（= ImageClassifier 的对应物）
│   ├── evaluation/metrics/        # WordErrorRate（自带编辑距离，不依赖 editdistance）
│   ├── engine/hooks/              # OTMonitorHook（dump/bucket 诊断）
│   └── utils/                     # sinkhorn.py（log 域 OT + bucket + 诊断）
├── tools/
│   ├── train.py                   # python tools/train.py <cfg>
│   ├── test.py                    # python tools/test.py <cfg> --checkpoint ...
│   ├── dist_train.sh / dist_test.sh
│   ├── misc/print_config.py       # 展开 _base_，检查覆盖是否符合预期
│   └── analysis_tools/
│       └── check_ot_numerics.py   # 论文式(7) 符号 / bucket 边缘 / mask 的数值验证
├── tests/                         # 176 个用例，纯 CPU，无需数据
├── docs/zh_cn/
│   ├── replication_guide.md       # 复现指南（原 v2 文档）
│   ├── project_structure.md       # 本文件
│   └── evidence/                  # 论文/仓库里核到的"证据"，不再放第三方文件
├── resources/                     # 文档引用的论文原图
├── requirements/
├── data/  work_dirs/              # gitignore，空目录占位
├── README.md
├── pyproject.toml
└── .gitignore
```

## 1. 概念映射：mmpretrain → 本项目

| mmpretrain | 本项目 | 说明 |
|---|---|---|
| `models/backbones/` | `models/backbones/` | Whisper = 声学 backbone，AV-HuBERT = 视觉 backbone，都冻结 |
| `models/necks/` | `models/necks/` | `Proj_a/Proj_v`、`Proj_av`、时序降采样；**融合也放在这里**（它作用在 backbone 输出和 head 之间） |
| `models/heads/` | `models/heads/LLMDecoder` | 对应 mmpretrain 的 `ClsHead`：把特征变成预测，并持有 loss 所需的拼装逻辑 |
| `models/losses/` | `models/losses/OTAlignmentLoss` | 对齐损失；`loss_weight` 就是论文的 α |
| `models/classifiers/ImageClassifier` | `models/classifiers/AVSRLLMClassifier` | `extract_feat` / `loss` / `predict` 三件套完全保留 |
| `models/data_preprocessors/` | `models/data_preprocessors/AVSRDataPreprocessor` | 变长序列 padding（图像里不需要，序列任务必须要） |
| `datasets/` | `datasets/` | `LRS3Dataset(BaseDataset)`；`transforms/` 四件套拆法一致 |
| `datasets/samplers.py` | `datasets/samplers/` | 多了 `MaxFramesBatchSampler`（按帧数组 batch，论文口径） |
| `evaluation/metrics/` | `evaluation/metrics/` | `BaseMetric` 子类，`process` + `compute_metrics` |
| `engine/hooks/` | `engine/hooks/OTMonitorHook` | 监控 `dump/bucket/entropy`，论文没有但复现必须有 |
| `apis/` | `apis/` | `train_model` / `test_model`，内部都是 `Runner.from_cfg` |
| `tools/train.py` | `tools/train.py` | `--config/--work-dir/--resume/--cfg-options` 同名 |
| `configs/_base_/` | `configs/_base_/` | 模型 / 数据 / 调度 / runtime 四层继承 |
| `tests/` | `tests/` | `test_configs.py` 会遍历所有 config，防止 `_base_` 覆盖失效 |

## 2. 迁移表（旧路径 → 新路径）

| 旧 | 新 | 备注 |
|---|---|---|
| `OT-LLM-AVSR复现指南.md` | `docs/zh_cn/replication_guide.md` | 内容不变，仅改内部链接 |
| `code/ot_align.py` | `otavsr/utils/sinkhorn.py` + `otavsr/models/losses/ot_alignment.py` | 拆成"数值内核"和"注册到 MODELS 的 loss" |
| `code/ot_align.py::_selftest` | `tests/test_models/test_ot_alignment.py` | 自检变成 10 个 pytest 用例 |
| `_paper/ot_check.py` | `tools/analysis_tools/check_ot_numerics.py` | 改成调用包内实现，顺带验证 `solve_ot` |
| `_paper/Fig1.png` / `Fig2.png` | `resources/paper_fig1.png` / `paper_fig2.png` | 指南里引用 |

## 3. 删除的文件及理由

| 删除 | 理由 |
|---|---|
| `_paper/SLT2026_2_June29.tex` | LaTeX 源码可从 `https://arxiv.org/src/2607.09001` 一键重新下载；其中**唯一不可再生的信息**（作者遗留的 5 组 WER 日志、5 张表的原始 tabular）已摘录到 `docs/zh_cn/evidence/paper_wer_logs.md` |
| `_paper/2607.09001v2.pdf`、`IEEEtran.cls`、`Figs/*.pdf`、`src.tar.gz` | PDF/类文件/压缩包都是源码包的派生物；图已存 `resources/` |
| `_refs/llamaavsr/*.py`、`_refs/mmsllama/*` | 第三方仓库源码直接 vendoring 既占体积也有许可证问题；改用 `docs/zh_cn/evidence/upstream_sources.md` 记录"哪个仓库、哪个文件、哪一行" |
| `work_dirs/`、`data/` 内容 | 训练产物与数据，`.gitignore` 已覆盖 |

保留原则：**能一条命令重新下载的不入库；不能再生的结论入库。**

## 4. 加一个新组件（registry 工作流）

mmpretrain 的约定是"注册 → 配置里引用 → 测试里验证"，三步：

```python
# 1) otavsr/models/necks/fusion/my_fusion.py
from otavsr.registry import MODELS

@MODELS.register_module()
class MyFusion(BaseModule):
    def __init__(self, in_channels, out_channels): ...
    def forward(self, x: dict): ...   # x = {'a': [B,L,d], 'v': [B,L,d]}
```

```python
# 2) 在 __init__.py 里导出，让 registry 能被发现
#    otavsr/models/necks/fusion/__init__.py
from .my_fusion import MyFusion
__all__ = [..., 'MyFusion']

# 3) 新配置只写差异，其余从 _base_ 继承（dict 是递归合并）
_base_ = ['../_base_/models/avsr_llm_c_concat.py', ...]
model = dict(fusion=dict(type='MyFusion'))
```

`python tools/misc/print_config.py configs/xxx.py --keys model.fusion` 用来确认覆盖结果。

## 5. 五分钟自检（改完任何东西都跑一遍）

```bash
python -m pytest tests -q                       # 176 用例，纯 CPU
python tools/misc/print_config.py configs/table5_sota/setting3.py --keys model.ot_loss
python tools/analysis_tools/check_ot_numerics.py
```

三个都通过，说明 registry、config 继承、OT 数值约定都没被破坏。
