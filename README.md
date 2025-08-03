# MOEQ  
**MoQE: Improve Quantization Model Performance via Mixture of Quantization Experts**

<p align="center">
  <img alt="Python 3.10.9+" src="https://img.shields.io/badge/python-3.10.9+-blue.svg"/>
  <img alt="PyTorch 2.7+" src="https://img.shields.io/badge/PyTorch-2.7+-orange.svg"/>
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green.svg"/>
</p>

---

## 📌 About
This repository provides an **efficient framework** for training **Mixture-of-Quantization-Experts (MoQE)** models that are built from **pre-trained, frozen, quantization experts**.  
The key idea is simple: keep all expert parameters frozen and only train a **lightweight gating network** that dynamically routes and combines the knowledge of these quantized experts .

| Advantages |
| --- |
| ✅ **Dramatically reduces training compute** |
| ✅ **Accelerates inference** |
| ✅ **Outperforms single quantized models** |

> NLP experiments were run on **A100 80 GB**; CV experiments on **V100S**.


## 🚀 Core Features

| Feature | Description |
| --- | --- |
| **Shallow-Sharing** | Unified embedding layer shared across all experts; expert bodies remain frozen → minimal trainable params. |
| **Heterogeneous Experts** | Native support for **GGUF**, **Safetensors**, and more—mix-and-match open-source experts freely. |
| **Dynamic Gating (MoERouter)** | TransformerEncoder + self-attention router for **context-aware** routing decisions. |
| **Memory-Efficient Training** | • Gradient Checkpointing<br>• 8-bit AdamW (`bitsandbytes`) |
| **Smart GPU Allocation** | Auto VRAM profiling → large experts go model-parallel, small ones land on the least-busy device. |
| **Curriculum Learning** | Start on a **smaller** dataset, then switch to a **larger** corpus for stable & efficient convergence. |

---
## 🖥️ System Environment
| Component | Version |
| --- | --- |
| **Python** | 3.10.9 |
| **CUDA** | 12.2 |
| **GPU** | NVIDIA A100-SXM4-80GB |
| **Driver** | 535.54.03 |

## 📦 Quick Setup
```bash
git clone https://best-jett/MOEQ.git

cd MOEQ

pip install -r requirements.txt

pip install torch==2.7.1 transformers==4.53.3 \
            bitsandbytes==0.47.0.dev0 pandas==2.3.1 \
            tqdm==4.67.1 accelerate==1.9.0
```
## Dependent library version See requirements.txt for the full list.


## 🛠️ Usage
### 1️⃣ Prepare Data & Models
- Put .parquet datasets in the specified directory.
- Prepare your quantized experts and note their paths.
- We use WikiText-2, OpenWebText, and C4 for experiments.

### 2️⃣ Launch Training
- Basic Training
```bash
python train_model.py \
    --train \
    --expert_paths /path/to/expert1 /path/to/expert2 \
    --data_dir /path/to/data \
    --save_dir /path/to/save/checkpoints \
    --batch_size 8 \
    --gradient_accumulation_steps 6 \
    --learning_rate 5e-5 \
    --epochs 10
```
- Resume from Checkpoint
```bash
python train_model.py \
    --train \
    --checkpoint_path /path/to/save/checkpoints/checkpoint.pt \
    --expert_paths /path/to/expert1 /path/to/expert2 \
    --data_dir /path/to/data \
    --save_dir /path/to/save/checkpoints
```
⚙️ Argument Reference
| Argument            | Default         | Description                     |
| ------------------- | --------------- | ------------------------------- |
| `--train`           | `False`         | Enable training mode            |
| `--eval`            | `False`         | Enable evaluation mode          |
| `--expert_paths`    | `None`          | Space-separated expert paths    |
| `--data_dir`        | `./data`        | Directory with `.parquet` files |
| `--save_dir`        | `./checkpoints` | Output directory                |
| `--from_scratch`    | `False`         | Ignore existing checkpoints     |
| `--checkpoint_path` | `None`          | Resume from checkpoint          |


# MOEQ
MoQE Training Code
# MoQE: Improve Quantization Model performance via Mixture of Quantization Experts.

本项目提供了一个用于混合量化专家（MoQE）模型的高效框架。其核心思想是利用多个预训练、冻结的经过量化的“专家”大语言模型，并仅训练一个轻量级的门控网络来学习如何根据输入动态地路由和组合这些量化专家的知识。

这种方法显著降低了训练所需的计算资源，加快推理速度并且有着比单一的量化模型更加强的性能。论文中的NLP实验在A100 80GB的显卡上完成，CV实验在V100S上完成。

## 核心特性

- **浅层共享架构 (Shallow-Sharing)**: 所有专家模型共享一个统一的词嵌入层，而专家自身的主体参数保持冻结，极大地减少了可训练参数量。
- **异构专家支持 (Heterogeneous Experts)**: 框架原生支持加载不同格式的专家模型，包括 GGUF 和 Safetensors，允许灵活组合来自开源社区的各类模型。
- **动态门控网络 (Dynamic Gating Network)**: 采用一个包含`TransformerEncoder`和自注意力机制的复杂`MoERouter`，能够捕捉输入序列的深层上下文信息以做出更精准的路由决策。。
- **显存高效训练 (Memory-Efficient Training)**: 默认启用**梯度检查点 (Gradient Checkpointing)**、**8-bit AdamW 优化器**以在标准单卡上实现多专家模型的稳定训练。
- **智能GPU分配 (Smart GPU Allocation)**: 训练前自动评估专家模型的显存占用，并将大型模型配置为跨多GPU的模型并行模式，小型模型则分配至当前最空闲的设备。
- **课程学习策略 (Curriculum Learning)**: 支持在训练初期使用较小的数据集，在后期切换到更大的数据集，以实现更稳定和高效的收敛。

## 环境要求

在运行前，请确保已安装以下核心依赖库：

```bash
cuda 12.2
pip install torch==2.7.1 transformers==4.53.3 bitsandbytes==0.47.0.dev0 pandas==2.3.1 tqdm==4.67.1 accelerate==1.9.0
```
更多详细信息见requirement.txt

## 使用方法

### 1. 准备数据和模型
- 将数据集（`.parquet`格式）放入指定的数据目录。
- 准备好您要用作专家的模型，并记下它们的路径。
- 本实验使用的是wikitext2,openwebtext,C4
### 2. 开始训练
使用以下命令启动训练。您可以根据需求修改命令行参数。

**基础训练命令：**
```bash
python train_model.py \
    --train \
    --expert_paths /path/to/expert1 /path/to/expert2 \
    --data_dir /path/to/your/data \
    --save_dir /path/to/save/checkpoints \
    --batch_size 8 \
    --gradient_accumulation_steps 6 \
    --learning_rate 5e-5 \
    --epochs 10
```

**从检查点继续训练：**
```bash
python train_model.py \
    --train \
    --checkpoint_path /path/to/save/checkpoints/checkpoint.pt \
    --expert_paths /path/to/expert1 /path/to/expert2 \
    --data_dir /path/to/your/data \
    --save_dir /path/to/save/checkpoints
```

## 参数配置详解

以下是所有可用的命令行参数及其说明。

| 参数 | 默认值 | 描述 |
|:---|:---:|:---|
| **核心配置** | | |
| `--train` | `False` | 启动训练模式。 |
| `--eval` | `False` | 启动评估模式。 |
| `--expert_paths` | `None` | 一系列专家模型的路径，以空格分隔。 |
| `--data_dir` | `/path/to/data` | 包含`.parquet`数据集文件的目录。 |
| `--save_dir` | `/path/to/moe_output` | 保存模型检查点和输出的目录。 |
| `--from_scratch` | `False` | 从头开始训练，即使存在检查点也忽略。 |
| `--checkpoint_path` | `

📜 License
This project is released under the MIT License.
