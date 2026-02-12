# Codex Notes

本文件记录项目特定的问题与解决方案。

---

## [P001] 2026-02-04 Test Run

- 全量测试 27 passed
- COCO per-class AP 计算修复（numpy 数组使用 `.size` 替代 `numel()`）

---

## [P002] Baseline 模型不收敛

**日期**: 2026-02-06
**状态**: 已解决
**严重程度**: 高

### 问题描述

baseline 实验 50 epochs，mAP 仅 0.12%

### 根因分析

1. 预训练权重未使用
2. num_classes=21 未含背景类
3. 学习率过高

### 解决方案

使用预训练权重 + 替换分类头 + num_classes=22

---

## 项目特定修复记录

### 预训练权重下载失败
- 下载 fasterrcnn_resnet50_fpn 权重时报 `invalid hash value`
- 修复：增加自动清理缓存并重试下载的逻辑

### COCO Evaluator + Subset
- 小样本验证时报 `Results do not correspond to current coco set`
- 修复：自动 unwrap Subset，按 subset image_ids 过滤 GT / 预测

### Normalize 修复
- torchvision 检测模型内部会做标准化，外部再 Normalize 会造成双重归一
- 修复：数据增强仅缩放到 [0,1]

### Resize/Padding 修复
- 预先 resize/pad 导致模型输出框坐标基于 800×800，COCO eval 用原图尺寸，IoU 失配
- 修复：禁用数据管道 resize/pad，交给 torchvision 内部处理

### COCOEval stats 空数组问题
- 静默 COCOEval 输出时，若不调用 `summarize()`，`stats` 可能为空
- 修复：在静默上下文里调用 `summarize()`，并对 `stats` 做空值兜底

### Train mAP50 评估 Subset 问题
- train_loader.dataset 是 Subset 时再包 Subset，导致 evaluator 无法匹配 GT
- 修复：从 base dataset + base indices 构建子集，避免 Subset-of-Subset

### Albumentations bbox 警告
- 关闭 resize 时没有 bbox 变换会触发警告
- 修复：追加 `HorizontalFlip(p=0.0)` 作为空操作

### DataLoader BBox Clamp
- 训练时 albumentations 报 bboxes y_max > 1.0
- 修复：在 `TongueCocoDataset.__getitem__` 中 clamp COCO bboxes 到图像边界

### Truncated Image Handling
- 训练时 PIL 报 `image file is truncated`
- 修复：启用 `ImageFile.LOAD_TRUNCATED_IMAGES = True`

### Scripts Import Fix
- `python scripts/train.py` 报 `ModuleNotFoundError: tcm_tongue`
- 修复：在 `scripts/train.py` 中添加 `src/` 到 `sys.path`

### GPU 显存口径差异（WSL/WDDM）
- `nvidia-smi` 显示全局显存占用（含驱动/系统保留）
- `torch.cuda.memory_allocated()` 仅统计 PyTorch 进程内 tensor 分配
