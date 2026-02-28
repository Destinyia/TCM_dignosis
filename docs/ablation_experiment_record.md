# 消融实验记录（分类：不均衡 × 增强）

## 数据集与目标
- 数据集：`datasets/shezhenv3-coco`
- 任务：舌象分类（13类）
- 主要指标：Macro‑F1；辅助观察：Train/Val Acc 与 Gap

## 训练与评估基线
- 脚本入口：`scripts/train_classifier.py`
- 基线模型：EfficientNet‑B3（`--backbone efficientnet_b3`）
- 训练设置：Cosine Warmup，AMP，早停
- 结果统计：`metrics_history.jsonl`（由 `scripts/run_ablation_imbalance.py --analyze` 汇总）

## 方法与关键代码

### 1) 损失函数（长尾不均衡）
**方法**
- CE / Focal / Class‑Balanced Focal / Seesaw

**关键代码**
- 损失选择逻辑：`scripts/train_classifier.py`
  - 参数：`--loss {ce,focal,cb_focal,seesaw}`、`--focal-gamma`
- Seesaw 实现：`src/tcm_tongue/losses/seesaw_loss.py`
  - 已做 AMP 兼容修复：seesaw 权重在 `torch.no_grad()` 下计算

**实验设计**
- Phase1：单独对比 L1/Focal、L2/CB‑Focal、L3/Seesaw

---

### 2) 采样策略（类不均衡）
**方法**
- WeightedRandomSampler（sqrt / inverse）
- Oversample（2/3/4）
- Undersample（ratio=0.5）

**关键代码**
- CLI 参数：`scripts/train_classifier.py`
  - `--weighted-sampler`、`--sampler-strategy {sqrt,inverse}`
  - `--sampler-type {default,oversample,undersample}`
  - `--oversample-factor`、`--undersample-ratio`
- 采样器实现：`src/tcm_tongue/data/sampler.py`
  - `ClassBalancedSampler`（oversample）
  - `UnderSampler`（undersample）
  - `WeightedRandomSampler`（sqrt / inverse）
- 与分类数据匹配：`src/tcm_tongue/data/sampler.py` 中 `_select_image_infos` 使用 `valid_images`

**实验设计**
- Phase2：B0/L1/L2 × {undersample(0.5), oversample(2/3/4), weighted sqrt/inverse}

---

### 3) 背景增强（mask‑aug）
**方法**
- 背景纯色 gray（A3）
- 背景 noise / blur（A2/A1）

**关键代码**
- Mask 增强入口：`src/tcm_tongue/data/classification_dataset.py`
  - `mask_aug`、`mask_aug_prob`、`mask_aug_mode/background`
- 可视化工具：`scripts/visualize_aug_grid.py`
- 训练期可视化：`src/tcm_tongue/engine/cls_trainer.py`（epoch1 batch）

**实验设计**
- A0/A1/A2/A3 对比（固定模型）

---

### 4) 几何增强（旋转 / 缩放）
**方法**
- 旋转 3° / 5°
- 缩放 ±0.05

**关键代码**
- 变换配置：`src/tcm_tongue/data/classification_dataset.py`
  - `aug_rotate`, `aug_rotate_limit`, `aug_rotate_prob`
  - `aug_scale`, `aug_scale_limit`, `aug_scale_prob`
- CLI 参数：`scripts/train_classifier.py`

**实验设计**
- C0/C1/C03 对比；结果显示单独几何增强提升有限

---

### 5) One‑of 组合增强（D0/D02 等）
**方法**
- One‑of：mask‑aug / scale / rotate 三者随机取一
- D02：mask0.3 + scale0.05 + rotate8°（各 p=0.1）

**关键代码**
- One‑of 逻辑：`src/tcm_tongue/data/classification_dataset.py`
  - `aug_oneof` 与 `mask_aug` 协同控制
- CLI：`scripts/train_classifier.py`（`--aug-oneof`）
- 配置参考：`scripts/run_ablation_crop_geo.py`（D0/D02）

**实验设计**
- D0/D01/D02 对比；D02 为增强端最优之一

---

## 实验设计概览
1) **Phase1**：损失与采样单因素对比  
2) **Phase2**：B0/L1/L2 × 采样策略组合  
3) **Phase3**：头部候选复测（稳健性确认）  
4) **Phase4**：增强端与不均衡策略组合  
   - 增强对照：A3（mask‑gray）、C1（rot5）、D02（one‑of）
   - 不均衡基组合：P2_L1_S1、L2、P2_L2_O2

实验管理脚本：`scripts/run_ablation_imbalance.py`

---

## 各阶段消融思路说明

### Phase1（单因素：损失 / 采样）
**目标**：先找到单一维度上对 F1 最有效的策略，建立“强基线”。  
**设计**：
- 损失函数：CE / Focal / CB‑Focal / Seesaw
- 采样：weighted sqrt / weighted inverse  
**判据**：Macro‑F1 为主，观察 Val Acc 与 Gap 作为过拟合提醒。
**实验键**：`B0`, `L1`, `L2`, `L3`, `S1`, `S2`

### Phase2（组合：损失 × 采样）
**目标**：验证“单因素最优”是否在组合后继续增益，确定不均衡端最佳方案。  
**设计**：
- 基线损失：B0 / L1 / L2
- 采样：undersample(0.5)、oversample(2/3/4)、weighted(sqrt/inverse)  
**判据**：F1 提升 + Gap 可控；淘汰提升小但不稳定的组合。
**实验键（自动生成）**：`P2_*`  
覆盖：`P2_B0_*`, `P2_L1_*`, `P2_L2_*` × `{S1,S2,U05,O2,O3,O4}`

### Phase3（头部候选复测）
**目标**：在新的种子上验证头部候选的稳定性，避免偶然最优。  
**设计**：只保留 Phase2 的 Top 组合 + 原始基线。  
**判据**：F1 均值与方差、Gap 稳定性。
**实验键**：`B0`, `L2`, `S2`, `P2_L1_S1`, `P2_L2_O2`, `P2_B0_S1`

### Phase4（增强 × 不均衡）
**目标**：验证最有效增强能否叠加在稳健不均衡策略上形成增益。  
**设计**：
- 不均衡端：P2_L1_S1 / L2 / P2_L2_O2  
- 增强端：A3（mask‑gray）、C1（rot5）、D02（one‑of）  
**判据**：是否在 F1 提升的同时保持 Val Acc 稳定与 Gap 控制。
**实验键（自动生成）**：`P4_*`  
覆盖：`{P2_L1_S1, L2, P2_L2_O2}` × `{gray, rot5, d02}`

---

## 最终结论（当前最优）
1) **不均衡端最稳健**：`P2_L1_S1`（Focal + weighted sqrt）
2) **增强端最有效**：`D02_oneof_mask_scale_rotate8`
3) **组合最优**：`P2_L1_S1 + D02`  
   - 在 Phase4 中达到最高 F1 且 Gap 可控，是当前最推荐的组合

**建议的最终对照组**
- `B0`（无增强）
- `D02`（增强端单独）
- `P2_L1_S1`（不均衡端单独）
- `P2_L1_S1 + D02`（最终推荐）
