# MMSA 多模态情感分析实验平台

基于 [MMSA 框架](https://github.com/thuiar/MMSA) 的多模态情感分析实验项目，支持在 CMU-MOSI 数据集上运行多种单模态和多模态融合模型，并进行多随机种子实验。

## ✨ 特性

- 🔧 **自动设备检测**：自动适配 NVIDIA GPU (CUDA)、Apple Silicon (MPS) 和 CPU
- 📊 **多种模型**：支持 7 种模型（3 种单模态 + 4 种融合模型）
- 🎲 **多种子实验**：支持多随机种子实验，便于统计分析
- 📈 **结果分析工具**：自动汇总实验结果，支持导出 LaTeX 表格

---

## 📁 项目结构

```
mmsa/
├── mmsa_mac_utils.py              # 核心工具：设备检测、路径管理
│
├── 📂 单模态实验
│   ├── run_text_only.py            # 文本 MLP 模型
│   ├── run_audio_only.py           # 音频 MLP 模型
│   ├── run_visual_only.py          # 视觉 MLP 模型
│   ├── run_text_lstm.py            # 文本 BiLSTM 模型
│   ├── run_text_attention.py       # 文本 BiLSTM + Attention 模型
│   └── run_text_transformer.py     # 文本 Transformer 模型
│
├── 📂 多模态融合实验
│   ├── run_early_fusion.py         # 早期融合 (EF-LSTM)
│   ├── run_late_fusion.py          # 晚期融合 (LF-DNN)
│   ├── run_lmf.py                  # 低秩多模态融合 (LMF)
│   ├── run_mult.py                 # 多模态 Transformer (MulT)
│   └── run_self_mm.py              # Self-MM
│
├── 📂 多种子实验（每个模型的多种子版本）
│   ├── run_text_only_multiseed.py
│   ├── run_audio_only_multiseed.py
│   ├── run_visual_only_multiseed.py
│   ├── run_text_lstm_multiseed.py
│   ├── run_text_attention_multiseed.py
│   ├── run_text_transformer_multiseed.py
│   ├── run_unimodal_multiseed.py   # 通用单模态多种子实验
│   ├── run_early_fusion_multiseed.py
│   ├── run_late_fusion_multiseed.py
│   ├── run_lmf_multiseed.py
│   ├── run_mult_multiseed.py
│   ├── run_self_mm_multiseed.py
│   └── run_mmsa_multiseed.py       # 通用 MMSA 模型多种子实验
│
├── 📂 工具脚本
│   ├── summarize_results.py        # 汇总实验结果
│   ├── check_model_data.py         # 查询模型数据需求
│   ├── show_model_config.py        # 查看模型配置
│   └── run_standard_seeds.sh       # 一键运行全部实验
│
├── MOSI/                           # 数据集目录（需自行下载）
├── MOSEI/                          # 数据集目录（需自行下载）
├── experiment_results/             # 实验结果输出目录
└── .gitignore
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone -b feature/gpu-support https://github.com/zynaaaaa/mmsa.git
cd mmsa
```

### 2. 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install MMSA numpy scipy scikit-learn
```

> **注意**：`MMSA` 包会自动安装 PyTorch。如果你需要 CUDA 支持，建议先按 [PyTorch 官网](https://pytorch.org/) 的指引安装对应版本的 PyTorch，再安装 MMSA。

### 4. 准备数据集

从 MMSA 官方下载 CMU-MOSI 数据集特征文件，放入 `MOSI/` 目录：

```
MOSI/
├── unaligned_50.pkl    # 非对齐数据（大部分模型使用）
└── aligned_50.pkl      # 对齐数据（Early Fusion 使用）
```

> **数据集下载**：可以从 [MMSA 官方仓库](https://github.com/thuiar/MMSA) 的说明中获取下载链接。

### 5. 验证安装

```bash
# 测试 MMSA 是否正确安装
python -c "from MMSA import MMSA_run; print('MMSA 安装成功')"

# 测试设备检测
python -c "from mmsa_mac_utils import get_device; print(f'检测到的设备: {get_device()}')"
```

预期输出：
| 环境 | 输出 |
|---|---|
| NVIDIA GPU | `检测到的设备: cuda` |
| Apple Silicon Mac | `检测到的设备: mps` |
| 无 GPU | `检测到的设备: cpu` |

---

## 🧪 运行实验

### 单次实验（快速测试）

每个单次实验脚本默认使用一个随机种子运行一次实验：

```bash
# 单模态实验
python run_text_only.py           # 文本 MLP
python run_audio_only.py          # 音频 MLP
python run_visual_only.py         # 视觉 MLP
python run_text_lstm.py           # 文本 BiLSTM
python run_text_attention.py      # 文本 BiLSTM + Attention
python run_text_transformer.py    # 文本 Transformer

# 多模态融合实验
python run_early_fusion.py        # 早期融合 (EF-LSTM)
python run_late_fusion.py         # 晚期融合 (LF-DNN)
python run_lmf.py                 # 低秩多模态融合 (LMF)
python run_mult.py                # MulT
python run_self_mm.py             # Self-MM
```

### 多种子实验（用于论文）

多种子实验会使用多个随机种子运行，输出每个种子的结果以及均值 ± 标准差：

```bash
# 默认 5 个种子 [0, 1, 2, 3, 4]
python run_text_only_multiseed.py
python run_text_lstm_multiseed.py
python run_text_attention_multiseed.py
python run_text_transformer_multiseed.py
python run_audio_only_multiseed.py
python run_visual_only_multiseed.py

# MMSA 融合模型多种子
python run_early_fusion_multiseed.py
python run_late_fusion_multiseed.py
python run_lmf_multiseed.py
python run_mult_multiseed.py
python run_self_mm_multiseed.py
```

**自定义种子列表**：

```bash
python run_text_only_multiseed.py --seeds 0 1 2 3 4 5 6 7 8 9
python run_lmf_multiseed.py --seeds 42 123 456
```

**通用单模态脚本**（可指定模态）：

```bash
python run_unimodal_multiseed.py text --seeds 0 1 2 3 4
python run_unimodal_multiseed.py audio --seeds 0 1 2 3 4
python run_unimodal_multiseed.py vision --seeds 0 1 2 3 4
```

**通用 MMSA 融合模型脚本**（可指定模型）：

```bash
python run_mmsa_multiseed.py ef_lstm --seeds 0 1 2 3 4
python run_mmsa_multiseed.py lf_dnn --seeds 0 1 2 3 4
python run_mmsa_multiseed.py lmf --seeds 0 1 2 3 4
python run_mmsa_multiseed.py mult --seeds 0 1 2 3 4
```

### 一键运行全部实验

```bash
bash run_standard_seeds.sh
```

> ⚠️ 注意：一键运行脚本中的路径是硬编码的，请先修改 `run_standard_seeds.sh` 中的 `VENV_PYTHON` 和 `WORK_DIR` 变量为你自己的路径。

---

## 📊 查看与分析结果

### 汇总实验结果

多种子实验的日志保存在 `experiment_results/` 目录下。使用汇总工具查看：

```bash
python summarize_results.py experiment_results
```

输出内容包括：
- 每个模型每个种子的详细指标
- 均值 ± 标准差汇总
- 失败种子检测
- **对比表格**（横向对比所有模型）
- **LaTeX 表格**（可直接复制到论文中）

### 评估指标说明

| 指标 | 含义 | 方向 |
|---|---|---|
| Has0_acc_2 | 二分类准确率（含 0） | ↑ 越高越好 |
| Has0_F1_score | F1 分数（含 0） | ↑ 越高越好 |
| Non0_acc_2 | 二分类准确率（排除 0） | ↑ 越高越好 |
| Non0_F1_score | F1 分数（排除 0） | ↑ 越高越好 |
| Mult_acc_5 | 5 分类准确率 | ↑ 越高越好 |
| Mult_acc_7 | 7 分类准确率 | ↑ 越高越好 |
| MAE | 平均绝对误差 | ↓ 越低越好 |
| Corr | 皮尔逊相关系数 | ↑ 越高越好 |

---

## 🔧 工具脚本

### 查询模型数据需求

```bash
# 查看所有模型的数据对齐需求
python check_model_data.py

# 查看特定模型
python check_model_data.py ef_lstm
python check_model_data.py lmf
```

### 查看模型完整配置

```bash
python show_model_config.py lmf
python show_model_config.py mult mosi
```

---

## 🖥️ 设备支持

本项目自动检测运行环境，无需手动配置设备：

| 环境 | 设备 | 说明 |
|---|---|---|
| NVIDIA GPU | `cuda` | 自动使用 GPU 加速 |
| Apple Silicon Mac | `mps` | 使用 Metal Performance Shaders |
| 其他 | `cpu` | 使用 CPU 运行 |

如需手动查看当前检测到的设备：

```bash
python -c "from mmsa_mac_utils import get_device; print(get_device())"
```

---

## 📋 模型概览

### 单模态模型

| 模型 | 模态 | 结构 | 数据文件 |
|---|---|---|---|
| Text MLP | 文本 | 均值池化 + MLP | unaligned_50.pkl |
| Audio MLP | 音频 | 均值池化 + MLP | unaligned_50.pkl |
| Visual MLP | 视觉 | 均值池化 + MLP | unaligned_50.pkl |
| Text BiLSTM | 文本 | 双向 LSTM + 均值池化 + MLP | unaligned_50.pkl |
| Text BiLSTM-Attention | 文本 | 双向 LSTM + 注意力池化 + MLP | unaligned_50.pkl |
| Text Transformer | 文本 | Transformer Encoder + 均值池化 + MLP | unaligned_50.pkl |

### 多模态融合模型

| 模型 | 说明 | 数据文件 |
|---|---|---|
| EF-LSTM | 早期融合，拼接三模态后用 LSTM 建模 | **aligned_50.pkl** |
| LF-DNN | 晚期融合，分别编码后拼接 | unaligned_50.pkl |
| LMF | 低秩多模态融合 | unaligned_50.pkl |
| MulT | 多模态 Transformer，跨模态注意力 | unaligned_50.pkl |
| Self-MM | 自监督多模态多任务学习 | unaligned_50.pkl |

> ⚠️ **注意**：Early Fusion (EF-LSTM) 需要**对齐数据** `aligned_50.pkl`，其他模型使用 `unaligned_50.pkl`。

---

## ❓ 常见问题

### Q: `ModuleNotFoundError: No module named 'MMSA'`
安装 MMSA：
```bash
pip install MMSA
```

### Q: `FileNotFoundError: 找不到文件: .../MOSI/unaligned_50.pkl`
需要下载 CMU-MOSI 数据集特征文件，放到项目根目录下的 `MOSI/` 文件夹。

### Q: Mac 上运行报 `RuntimeError: MPS backend...`
某些操作在 MPS 上可能不支持。如遇到此问题，可以在运行前强制使用 CPU：
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python run_xxx.py
```

### Q: 如何在有 GPU 的服务器上加速？
项目会自动检测 CUDA 并使用 GPU，无需额外配置。确保安装了 CUDA 版本的 PyTorch 即可：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📄 License

本项目仅供学术研究使用。
