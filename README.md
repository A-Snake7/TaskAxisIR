# TaskAxisIR

这是从服务器 `/root/ProMamba_Project` 提取的只含源码的阅读备份。备份日期：2026-09-20；50 个原始文件，共 628,855 字节（约 614 KiB），不含本次新增说明文档。每个文件与下载时读取的服务器内容进行了 SHA-256 一致性校验。

项目用于去雨、去雾、去噪三任务联合图像恢复，包含旧版轻量模型、分层模型以及 Blind-v3 全局/局部退化路由。建议先读 [项目介绍与代码地图](docs/项目介绍与代码地图.md)，再看原始 [README](source/README.md)。

## 文件夹分类

```text
ProMamba_核心代码备份_20260920/
├─ README_本地导读.md           本文：备份范围与入口
├─ source/                    项目主目录中的核心源码，保留原始布局
│  ├─ net/                    模型、PLFB、FDMB、退化编码器、路由融合
│  ├─ utils/                  损失、分块推理、EMA、检查点等
│  ├─ scripts/                数据审计、正式评测、效率与冒烟测试
│  ├─ tests/                  单元测试
│  ├─ train.py / dataset.py   训练入口、数据加载与采样
│  └─ test.py / test_single.py
├─ ablation_overrides/        当前消融实验相对主源码的6个新增或变更文件
├─ experiment_tools/          最近实验的评测、队列、续训控制脚本
└─ docs/
   ├─ 项目介绍与代码地图.md
   └─ backup_manifest.json    服务器来源、文件长度和SHA-256
```

`source/` 内不随意移动文件，避免破坏 `net`、`utils` 等导入路径。分类通过原有功能目录和本文导航实现。

## 备份边界

已备份核心 Python 源码、测试、依赖清单、原始 README 和必要运行脚本。没有下载数据集、图像、权重、检查点、训练日志、评测输出、数据划分 CSV、缓存、字体、压缩包或历史备份；也没有下载服务器 SSH 配置或本机私钥。

日期固定的历史训练启动脚本、下载脚本和论文绘图脚本不属于本次核心阅读备份。原始 README 可能提及这些未备份文件，不能据此认为本地拥有完整服务器工程。

## 主源码与消融源码的区别

服务器当前消融训练使用冻结目录 `experiments/chapter6_priority_20260917/code`，并非直接使用项目主目录。为避免保存多个重复工程，这里只把冻结目录中与主源码不同或新增的文件放入 `ablation_overrides/`。

该目录包含：`train.py`、`net/promamba.py`、`net/promamba_v2.py`、`utils/losses.py`、`utils/chapter6_inference.py`、`scripts/check_ablations.py`。查看当前消融逻辑时优先阅读这些文件；它们依赖 `source/` 中的其他模块，不能单独运行。

若以后需要还原一个可编辑的消融源码副本，应先将 `source/` 复制到新的工作目录，再将 `ablation_overrides/` 按相同相对路径覆盖进去。保留本备份原样，以便日后核对。这里没有恢复完整实验控制目录和其配置文件，因此这种组合是核心源码副本，不是可直接重启整个实验队列的打包环境。

## 本地能做什么

- 可直接阅读、检索、比较模型实现及训练与推理逻辑。
- 要实际运行，需要另行准备 Linux/CUDA/PyTorch/Mamba 环境、数据、权重和路径配置。本备份不保证 Windows 原生可运行。
- `experiment_tools/` 保留了服务器绝对路径，供理解实验流程和恢复方式；不要在本地直接执行。主训练和评测入口见 `source/`。

本次操作只读取服务器文件，没有修改训练源码、检查点或训练进程。

