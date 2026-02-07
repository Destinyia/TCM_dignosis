# 基线模型说明文档

**日期**: 2026-02-04  
**基线配置**: `configs/experiments/baseline.yaml`（继承 `configs/base.yaml`）

## 1) 总览
基线模型使用 **torchvision Faster R-CNN（ResNet-50 + FPN）**。当前实现中检测头直接封装 torchvision 模型，基线实验不走自定义 `backbone/neck` 前向，因此结构等价于 torchvision 官方实现。

## 2) 输入尺寸与图片处理管道
**输入尺寸**: 使用**原始尺寸输入**（不在数据管道中 resize/pad），由 torchvision 内部 transform 负责缩放与归一。

### 训练阶段
- **数据集**: COCO 格式（`datasets/shezhenv3-coco`）
- **图片读取**: PIL → RGB
- **BBox 处理**:
  - 将 COCO bbox 裁剪到图像边界内
  - 重新计算面积（避免非法框）
- **数据增强（TrainTransform）**:
  1. `HorizontalFlip(p=0.5)`
  2. `RandomBrightnessContrast(p=0.3)`
  3. `HueSaturationValue(p=0.3)`
  4. `GaussNoise(p=0.1)`
  5. `Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0])`（仅缩放到 [0,1]）
  6. `ToTensorV2()`

### 验证/推理阶段
- 相同的 normalize（仅缩放到 [0,1]） + tensor（无随机增强）
- 使用 `ValTransform`

## 3) 模型结构
**检测器**: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`

核心组件：
- **Backbone**: ResNet-50
- **Neck**: FPN
- **RPN**: torchvision 默认 anchors & proposals
- **ROI Head**: 分类 + 回归

> 说明：虽然代码中包含自定义 `backbone/neck/head` 抽象，但基线实验使用 torchvision 内置 Faster R-CNN，结构与官方实现一致。

## 4) 关键超参数（基线配置）
来自 `configs/base.yaml` + `configs/experiments/baseline.yaml`：
- **Batch size**: 8
- **Epochs**: 50（可通过 CLI 覆盖）
- **Optimizer**: SGD（当 scheduler=cosine）
- **LR**: 0.001
- **Weight decay**: 1e-4
- **LR scheduler**: cosine
- **Sampler**: default（不重采样）
- **Loss**: 配置为 CE（Faster R-CNN 实际使用其内部损失）
- **类别数**: 数据集 21 类，模型 `num_classes=22`（包含背景类）
- **label_offset**: 1（训练时标签整体 +1，评估/推理时再映射回原类别）
- **resize_in_dataset**: false（不在数据管道中做 resize/pad）

## 5) 参数量
在当前环境测得：
```
PYTHONPATH=src python - <<'PY'
from tcm_tongue.config import Config
from tcm_tongue.models import build_detector
cfg = Config.from_yaml('configs/experiments/baseline.yaml')
model = build_detector(cfg)
print(sum(p.numel() for p in model.parameters()))
PY
```

- **总参数量**: **41,449,656**
- **可训练参数量**: **41,449,656**

> 该数值对应 `num_classes=21` 的 torchvision Faster R-CNN。

## 6) 显存占用（需实测）
显存占用与 GPU 型号、驱动、CUDA、数据增强以及 batch size 有关。请在你的 CUDA shell 中运行一小段训练并记录峰值显存。建议命令：  

```
python scripts/train.py --config configs/experiments/baseline.yaml --epochs 1 --device cuda
```

运行后用以下方式之一获取峰值显存：
- **PyTorch**：在训练脚本内打印 `torch.cuda.max_memory_allocated()/1024**3`（可选追加）
- **nvidia-smi**：`nvidia-smi --query-gpu=memory.used --format=csv`

填写模板（实测后补充）：  
- **Batch size=8，输入 800×800**：____ GB（峰值）  

如果你需要，我可以在 `scripts/train.py` 里加“峰值显存打印”并自动写入日志。

## 7) 执行入口
- **训练**: `python scripts/train.py --config configs/experiments/baseline.yaml`
- **运行实验集**: `python scripts/run_experiments.py`
- **实验分析**: `python scripts/analyze_experiments.py`

## 8) 已知约束
- 未安装包时，脚本会将 `src/` 注入 `sys.path` 以保证导入成功。
- 部分图片可能截断，数据集加载时允许读取截断图片（`ImageFile.LOAD_TRUNCATED_IMAGES = True`）。
