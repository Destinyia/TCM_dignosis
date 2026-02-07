# Codex Notes

This file records issues encountered in this repo and practical tips to resolve them.

## Python environment
- Use the `cv` conda environment to run Python scripts in this repo.
- Example: `conda run -n cv python analyze_shezhenv3_coco.py`
## Testing
- `pytest` is not currently available in the `cv` env (`pytest: command not found`).
  - Install via `conda install -n cv pytest` or `pip install pytest` if allowed.
  - `conda install -n cv pytest` failed due to DNS/network (`repo.anaconda.com` name resolution).
- `pytest -q` on repo root timed out (>60s); running a specific test file worked: `pytest -q tests/test_imports.py`.

## Dependency downloads
- Internet access may be blocked for `pip install`.
- If `pip` fails with "No matching distribution found", install via an offline wheel or use the `cv` environment where the package is already installed.

## Image/plot outputs
- PNG output requires Pillow. If Pillow is missing, use `conda run -n cv ...` where it is available.
- SVG outputs are a safe fallback when plotting libraries (matplotlib, cv2, etc.) are not available.

## 2026-02-04 Test Run
- Full test suite `conda run -n cv pytest tests/ -v` completed with 27 passed.
- No new issues discovered.

## 2026-02-04 COCO Evaluator Fix
- COCO per-class AP calculation used numpy arrays; `numel()` caused AttributeError.
- Fixed by switching to `valid.size` and `np.mean` in `COCOEvaluator._compute_per_class_ap`.

## 2026-02-04 CUDA Check (explicit conda activate)
- Running with explicit `conda activate cv` in a fresh shell still reports `torch.cuda.is_available() == False` with Error 304.
- In this Codex session, CUDA is treated as unavailable; use CPU for automated runs.

## 2026-02-04 Scripts Import Fix
- `python scripts/train.py` failed with `ModuleNotFoundError: tcm_tongue` when repo not installed as package.
- Fixed by adding `src/` to `sys.path` inside `scripts/train.py`.

## 2026-02-04 Train Script Logging Fix
- `scripts/train.py` raised `NameError: _print_env_config` due to function defined after use.
- Fixed by moving `_print_env_config` above `main`.

## 2026-02-04 DataLoader BBox Clamp
- Training run failed in albumentations with bboxes having y_max > 1.0.
- Fixed by clamping COCO bboxes to image bounds in `TongueCocoDataset.__getitem__` before augmentation.

## 2026-02-04 Truncated Image Handling
- Training crashed on `image file is truncated` from PIL.
- Enabled `ImageFile.LOAD_TRUNCATED_IMAGES = True` in dataset image loader to allow reading truncated files.

## 2026-02-06 [P002] Baseline 模型不收敛

**日期**: 2026-02-06
**状态**: 待处理
**严重程度**: 高

### 问题描述
运行 `scripts/run_experiments.py` 后，baseline 实验 50 epochs 训练完成，但 mAP 仅为 0.12%，模型基本没有收敛。

实验结果 (`runs/experiments/baseline_20260205_194956/metrics.json`):
- mAP: 0.0012 (0.12%)
- mAP_50: 0.0057 (0.57%)
- 大部分类别 AP 为 0

### 根因分析

1. **预训练权重未使用** (`src/tcm_tongue/models/head.py:49-53`)
   ```python
   self.model = fasterrcnn_resnet50_fpn(
       weights=None,           # ❌ 没有使用预训练权重
       weights_backbone=None,  # ❌ backbone 也没有预训练
       num_classes=num_classes,
   )
   ```
   从头训练目标检测模型非常困难，需要大量数据和训练时间。

2. **num_classes 未包含背景类**
   - 配置中 `num_classes=21`，但 torchvision 检测模型需要 `num_classes` 包含背景类
   - 正确值应为 `21 + 1 = 22`

3. **学习率可能过高**
   - 当前使用 SGD lr=0.001，对于从头训练可能不稳定

### 解决方案

修改 `src/tcm_tongue/models/head.py`，使用预训练权重并替换分类头：

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 加载预训练模型
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)

# 替换分类头以匹配自定义类别数
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

### 验证方法

创建快速调试脚本，使用少量样本验证收敛：
- 训练样本: 200 张
- 验证样本: 50 张
- Epochs: 5
- 预期: loss 应明显下降，检测数量应增加

## 2026-02-06 Baseline 修复已落地
- Faster R-CNN 使用预训练权重并替换分类头
- `model.num_classes` 设置为 22（含背景类）
- 训练标签 `label_offset=1`；评估/推理时做反向映射

## 2026-02-06 预训练权重下载失败
- 下载 fasterrcnn_resnet50_fpn 权重时报 `invalid hash value`（文件损坏/中断）。
- 已增加自动清理缓存并重试下载的逻辑（`models/head.py`）。

## 2026-02-06 COCO Evaluator + Subset
- 小样本验证时报 `Results do not correspond to current coco set`。
- 原因：`Subset` 传入 evaluator 时，GT 集与预测 image_ids 不一致。
- 修复：自动 unwrap Subset，并按 subset image_ids 过滤 GT / 预测。

## 2026-02-06 Normalize 修复
- torchvision 检测模型内部会做标准化；外部再 Normalize 会造成双重归一。
- 数据增强改为仅缩放到 [0,1]（Normalize mean=0, std=1），由模型内部再归一。

## 2026-02-06 Resize/Padding 修复
- 预先 resize/pad 会导致模型输出框坐标基于 800×800，COCO eval 用原图尺寸，IoU 失配。
- 基线实验禁用数据管道 resize/pad，交给 torchvision 内部 transform 处理。

## 2026-02-06 COCOEval stats 空数组问题
- 在静默 COCOEval 输出时，若不调用 `summarize()`，`stats` 可能为空导致 IndexError。
- 修复：在静默上下文里调用 `summarize()`，并对 `stats` 做空值兜底。

## 2026-02-06 Train mAP50 评估 Subset 问题
- 当 train_loader.dataset 是 Subset 时再包 Subset，会导致 evaluator 无法匹配 GT。
- 修复：从 base dataset + base indices 构建子集，避免 Subset-of-Subset。

## 2026-02-06 Albumentations bbox 警告
- 关闭 resize 时没有 bbox 变换会触发警告。
- 修复：追加 `HorizontalFlip(p=0.0)` 作为空操作。

## 2026-02-06 训练日志与指标输出
- 进度条显存显示使用 `memory_allocated/reserved` + `mem_get_info free/total`。
- train mAP50 仅在 epoch 结束后计算一次，并写入 `metrics.json`/`metrics_history.jsonl`。

## 2026-02-06 GPU 显存口径差异（WSL/WDDM）
- `nvidia-smi` 显示的是全局显存占用（含驱动/系统保留），空闲时也可能有 ~1GB 常驻。
- `torch.cuda.memory_allocated()` 仅统计 PyTorch 进程内 tensor 分配，数值较小属正常。

## 工作约定（通用）
- 每完成一个阶段运行全量单元测试（用户在 CUDA 环境执行）。
- 发现问题要更新 DEVELOPMENT_PLAN 或 codex.MD。
- 需要安装新依赖或调整计划前先征询用户。
