# Classifier 相关脚本 / 模型 / 优化实现索引

## 1) 训练与实验脚本
- `scripts/train_classifier.py`  
  训练主入口：模型构建、数据加载、loss/scheduler/AMP/早停配置。
- `scripts/run_ablation_imbalance.py`  
  不均衡消融的批量运行与结果汇总（Phase1–Phase4）。
- `scripts/run_ablation_crop_geo.py`  
  传统增强/几何增强/one‑of 增强的实验配置参考。
- `scripts/visualize_aug_grid.py`  
  增强可视化（训练外）与参数标注。

## 2) 模型与骨干
- 分类模型定义：`src/tcm_tongue/models/classifier.py`  
  - 支持：ResNet34/50/101、EfficientNet‑B0/B2/B3
  - 入口：`build_classifier(...)` 与 `build_model(...)`
- 分割注意力（mask attention）：`src/tcm_tongue/models/seg_attention.py`  
  - 如果启用 `--model-type seg_attention` 或 mask‑aug 相关分支

**模型构建调用示例（项目内）**
```python
# src/tcm_tongue/models/classifier.py
from tcm_tongue.models.classifier import build_classifier

model = build_classifier(
    backbone="efficientnet_b3",
    num_classes=13,
    pretrained=True,
    dropout=0.0,
)
```
```python
# src/tcm_tongue/models/classifier.py
from tcm_tongue.models.classifier import build_model

# baseline / seg_attention / multiscale 等
model = build_model(
    model_type="baseline",
    backbone="resnet50",
    num_classes=13,
    pretrained=True,
)
```

**调用示例（CLI）**
```bash
# 选择骨干与模型类型
python scripts/train_classifier.py --backbone efficientnet_b3 --model-type baseline
python scripts/train_classifier.py --backbone resnet34 --model-type baseline
```
## 3) 数据集与增强
- 数据集实现：`src/tcm_tongue/data/classification_dataset.py`
  - 核心：`TongueClassificationDataset`
  - 变换：`ClassificationTransform`
  - Mask 增强：`mask_aug` / `mask_aug_prob` / `mask_aug_mode`
  - One‑of 增强：`aug_oneof`
- 训练批次可视化：`src/tcm_tongue/engine/cls_trainer.py`
  - 第 1 个 epoch 可输出真实 batch 的增强样例与 mask 网格

## 4) 不均衡优化（Loss / Sampler）

### 4.1 损失函数
- Focal / CB‑Focal：`src/tcm_tongue/losses/focal_loss.py`
  - `FocalLoss` / `ClassBalancedFocalLoss`
- Seesaw：`src/tcm_tongue/losses/seesaw_loss.py`
  - 已加 `torch.no_grad()` 防止 AMP inplace 错误
- 入口选择逻辑：`scripts/train_classifier.py`
  - `--loss {ce,focal,cb_focal,seesaw}`、`--focal-gamma`

**实现代码（项目内）**
```python
# src/tcm_tongue/losses/focal_loss.py
loss = FocalLoss(gamma=2.0)
loss = ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)
```
```python
# src/tcm_tongue/losses/seesaw_loss.py
loss = SeesawLoss(num_classes=num_classes, p=0.6, q=1.5)
```

**调用示例（CLI）**
```bash
# Focal
python scripts/train_classifier.py --loss focal --focal-gamma 2.0

# CB-Focal
python scripts/train_classifier.py --loss cb_focal --focal-gamma 2.0

# Seesaw
python scripts/train_classifier.py --loss seesaw
```

### 4.2 采样器（用于不均衡）
- 采样器实现：`src/tcm_tongue/data/sampler.py`
  - `ClassBalancedSampler`（oversample）
  - `UnderSampler`（undersample）
  - `WeightedRandomSampler`（sqrt / inverse）
- 数据加载器入口：`create_classification_dataloaders(...)`
  - `src/tcm_tongue/data/classification_dataset.py`
- CLI 参数：`scripts/train_classifier.py`
  - `--weighted-sampler` + `--sampler-strategy {sqrt,inverse}`
  - `--sampler-type {default,oversample,undersample}`
  - `--oversample-factor` / `--undersample-ratio`

**实现代码（项目内）**
```python
# src/tcm_tongue/data/sampler.py
sampler = ClassBalancedSampler(dataset, oversample_factor=2.0)
sampler = UnderSampler(dataset, target_ratio=0.5)
```

**调用示例（官方库）**
```python
# torch.utils.data.WeightedRandomSampler
weights = train_dataset.get_sample_weights(strategy="sqrt")
sampler = torch.utils.data.WeightedRandomSampler(
    weights, len(weights), replacement=True
)
```

**调用示例（CLI）**
```bash
# weighted sampler (sqrt / inverse)
python scripts/train_classifier.py --weighted-sampler --sampler-strategy sqrt
python scripts/train_classifier.py --weighted-sampler --sampler-strategy inverse

# oversample / undersample
python scripts/train_classifier.py --sampler-type oversample --oversample-factor 3
python scripts/train_classifier.py --sampler-type undersample --undersample-ratio 0.5
```

## 5) 增强策略（实验用到的关键实现）
- Mask 背景增强（gray/noise/blur）  
  - 实现：`classification_dataset.py` 中 `apply_mask_aug`
  - 参数：
    - `--mask-aug`
    - `--mask-aug-mode background`
    - `--mask-aug-bg-mode {solid,noise,blur}`
    - `--mask-aug-bg-color gray`
    - `--mask-aug-prob`
- 几何增强（旋转 / 缩放）
  - 实现：`ClassificationTransform`（Albumentations）
  - 参数：
    - `--aug-rotate --aug-rotate-limit --aug-rotate-prob --aug-rotate-fill`
    - `--aug-scale --aug-scale-limit --aug-scale-prob`
- One‑of 增强（D02 等）
  - 实现：`ClassificationTransform` 中 `aug_oneof`
  - 典型组合：
    - mask0.3 + scale0.05 + rotate8（各 p=0.1）

**实现代码（官方库）**
```python
# src/tcm_tongue/data/classification_dataset.py
# Albumentations 组合（示例）
ops = [
    A.LongestMaxSize(max_size=max(image_size)),
    A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT),
    A.RandomScale(scale_limit=0.05, p=0.1),
    A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, value=114, p=0.1),
]
```

**调用示例（CLI）**
```bash
# mask(gray)
python scripts/train_classifier.py --mask-aug \
  --mask-aug-mode background --mask-aug-bg-mode solid --mask-aug-bg-color gray \
  --mask-aug-prob 0.5 --mask-aug-dilate 15

# rotate 5deg
python scripts/train_classifier.py --aug-rotate --aug-rotate-limit 5 \
  --aug-rotate-prob 0.2 --aug-rotate-fill gray

# D02 one-of (mask0.3 + scale0.05 + rotate8, p=0.1 each)
python scripts/train_classifier.py --aug-oneof \
  --mask-aug --mask-aug-mode background --mask-aug-bg-mode solid --mask-aug-bg-color gray \
  --mask-aug-prob 0.3 --mask-aug-dilate 15 \
  --aug-scale --aug-scale-limit 0.05 --aug-scale-prob 0.1 \
  --aug-rotate --aug-rotate-limit 8 --aug-rotate-prob 0.1 --aug-rotate-fill gray
```

## 6) 训练优化细节
- Scheduler：`scripts/train_classifier.py`
  - `--scheduler {cosine_warmup,cosine,none}`
  - `--warmup-epochs --min-lr`
- AMP：`--amp / --no-amp`
- 早停：`--early-stop`
- 梯度裁剪：`--grad-clip`

**调用示例（官方库）**
```python
# torch.cuda.amp
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
with torch.cuda.amp.autocast(enabled=use_amp):
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**调用示例（CLI）**
```bash
python scripts/train_classifier.py --scheduler cosine_warmup --warmup-epochs 5 --min-lr 1e-6
python scripts/train_classifier.py --amp
python scripts/train_classifier.py --early-stop 10 --grad-clip 1.0
```

## 7) 指标与结果输出
- 训练/验证指标统计：`src/tcm_tongue/engine/cls_trainer.py`
- 结果保存：
  - `metrics.json`
  - `metrics_history.jsonl`

---

## 常用入口速查
- 训练：`python scripts/train_classifier.py ...`
- 不均衡消融：`python scripts/run_ablation_imbalance.py --phase phaseX --seeds ...`
- 增强消融：`python scripts/run_ablation_crop_geo.py --only ...`

## 一键训练脚本配置案例

### 不均衡消融（Phase2）
```bash
python scripts/run_ablation_imbalance.py --phase phase2 --seeds 45,46,47 --skip-completed
```

### 头部复测（Phase3）
```bash
python scripts/run_ablation_imbalance.py --phase phase3 --seeds 48,49 --skip-completed
```

### 增强 × 不均衡（Phase4）
```bash
python scripts/run_ablation_imbalance.py --phase phase4 --seeds 45,46,47 --skip-completed
```

## 各 Phase 完整实验键与参数快照

### 公共训练参数（全部 Phase 共用）
```text
--data-root datasets/shezhenv3-coco
--image-size 640
--batch-size 16
--backbone efficientnet_b3
--epochs 50
--lr 0.0005
--scheduler cosine_warmup
--warmup-epochs 5
--min-lr 0.000001
--early-stop 10
--amp
--model-type baseline
```

---

### Phase1（单因素：loss / sampler）
**实验键列表**
```
B0, L1, L2, L3, S1, S2
```

**参数快照**
```text
B0: (仅公共参数)
L1: --loss focal --focal-gamma 2.0
L2: --loss cb_focal --focal-gamma 2.0
L3: --loss seesaw
S1: --weighted-sampler --sampler-strategy sqrt
S2: --weighted-sampler --sampler-strategy inverse
```

---

### Phase2（组合：B0/L1/L2 × 采样）
**实验键列表（完整展开）**
```
P2_B0_S1, P2_B0_S2, P2_B0_U05, P2_B0_O2, P2_B0_O3, P2_B0_O4,
P2_L1_S1, P2_L1_S2, P2_L1_U05, P2_L1_O2, P2_L1_O3, P2_L1_O4,
P2_L2_S1, P2_L2_S2, P2_L2_U05, P2_L2_O2, P2_L2_O3, P2_L2_O4
```

**参数快照**
```text
基础损失部分：
P2_B0_*: (仅公共参数)
P2_L1_*: --loss focal --focal-gamma 2.0
P2_L2_*: --loss cb_focal --focal-gamma 2.0

采样部分：
*_S1:  --weighted-sampler --sampler-strategy sqrt
*_S2:  --weighted-sampler --sampler-strategy inverse
*_U05: --sampler-type undersample --undersample-ratio 0.5
*_O2:  --sampler-type oversample --oversample-factor 2
*_O3:  --sampler-type oversample --oversample-factor 3
*_O4:  --sampler-type oversample --oversample-factor 4
```

---

### Phase3（头部候选复测）
**实验键列表**
```
B0, L2, S2, P2_L1_S1, P2_L2_O2, P2_B0_S1
```

**参数快照**
```text
B0: (仅公共参数)
L2: --loss cb_focal --focal-gamma 2.0
S2: --weighted-sampler --sampler-strategy inverse
P2_L1_S1: --loss focal --focal-gamma 2.0 + --weighted-sampler --sampler-strategy sqrt
P2_L2_O2: --loss cb_focal --focal-gamma 2.0 + --sampler-type oversample --oversample-factor 2
P2_B0_S1: --weighted-sampler --sampler-strategy sqrt
```

---

### Phase4（增强 × 不均衡）
**实验键列表（完整展开）**
```
P4_P2_L1_S1_gray,   P4_P2_L1_S1_rot5,   P4_P2_L1_S1_d02,
P4_L2_gray,         P4_L2_rot5,         P4_L2_d02,
P4_P2_L2_O2_gray,   P4_P2_L2_O2_rot5,   P4_P2_L2_O2_d02
```

**参数快照**
```text
基础不均衡部分：
P4_P2_L1_S1_*: --loss focal --focal-gamma 2.0 + --weighted-sampler --sampler-strategy sqrt
P4_L2_*:       --loss cb_focal --focal-gamma 2.0
P4_P2_L2_O2_*: --loss cb_focal --focal-gamma 2.0 + --sampler-type oversample --oversample-factor 2

增强部分：
*_gray:
  --mask-aug --mask-aug-mode background --mask-aug-bg-mode solid --mask-aug-bg-color gray
  --mask-aug-prob 0.5 --mask-aug-dilate 15

*_rot5:
  --aug-rotate --aug-rotate-limit 5 --aug-rotate-prob 0.2 --aug-rotate-fill gray

*_d02:
  --aug-oneof
  --mask-aug --mask-aug-mode background --mask-aug-bg-mode solid --mask-aug-bg-color gray
  --mask-aug-prob 0.3 --mask-aug-dilate 15
  --aug-scale --aug-scale-limit 0.05 --aug-scale-prob 0.1
  --aug-rotate --aug-rotate-limit 8 --aug-rotate-prob 0.1 --aug-rotate-fill gray
```
