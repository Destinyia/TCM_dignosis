# 少样本不均衡消融实验计划（以 B0 为基线）

## 1. 数据集基本情况（高层说明）
- 数据集：`datasets/shezhenv3-coco`（COCO 格式）
- 任务：舌象分类（13 类，明显长尾/不均衡）
- 划分：固定 train/val，不在本计划中变更

---

## 2. 实验目标
- 以 **B0** 作为固定 baseline
- 评估 **不均衡策略**（loss / sampler）对尾类表现的影响
- 评估 **增强策略**（mask / 几何 / one‑of）在不均衡场景下的稳定性
- 用 **Macro F1** 作为主指标，辅以 Train/Val Acc 与 Gap 监控过拟合

---

## 3. Baseline 定义（B0）
- 模型：EffNet‑B3 baseline
- Loss：CE
- Sampler：无
- 增强：无额外策略（仅基础预处理）

---

## 4. 消融策略设计

### Phase 1 — 不均衡策略消融（仅算法手段）
目的：剥离增强干扰，纯评估 loss / sampler

**Loss-only（无 sampler）**
- L1: Focal（`--loss focal --focal-gamma 2.0`）
- L2: Class-Balanced Focal（`--loss cb_focal --focal-gamma 2.0`）
- L3: Seesaw（`--loss seesaw`）

**Sampler-only（loss=CE）**
- S1: WeightedSampler + sqrt（`--weighted-sampler --sampler-strategy sqrt`）
- S2: WeightedSampler + inverse（`--weighted-sampler --sampler-strategy inverse`）

**Loss + Sampler（取 Loss-only 最优 2 个）**
- C1: best loss + sampler(sqrt)
- C2: best loss + sampler(inverse)
- C3: 2nd best loss + sampler(sqrt)
- C4: 2nd best loss + sampler(inverse)

---

### Phase 2 — 增强策略消融（B0 + 增强）
目的：验证增强在不均衡任务中的稳定性

**Mask 背景类**
- A1: 背景 blur
- A2: 背景 noise
- A3: 背景 solid gray

**几何类**
- G1: rotate 3° (p=0.2)
- G2: rotate 5° (p=0.2)
- G3: scale ±0.05 (p=0.2)

**One‑of 组合（互斥）**
> 每样本只触发 mask/rotate/scale 之一  
> 概率由手动配置，要求总和 ≤ 1
- O1: mask/rotate/scale 三选一
- O2: mask/rotate 二选一
- O3: mask/scale 二选一

---

### Phase 3 — 组合验证（不均衡策略 × 最佳增强）
目的：验证最佳不均衡策略与最佳增强的叠加效应

- 选择 Phase 1 的最佳策略
- 选择 Phase 2 的最佳增强
- 组合 1–2 组作为最终验证

---

## 5. 固定训练设置（保持一致）
- `--epochs 50`
- `--early-stop 10`
- `--scheduler cosine_warmup --warmup-epochs 5 --min-lr 1e-6`
- `--lr 5e-4`
- `--batch-size 16`
- `--backbone efficientnet_b3`
- `--model-type baseline`

---

## 6. Seeds 设置
- 3 seeds（建议：45/46/47）
- 所有实验共享相同 seeds

---

## 7. 输出与汇总
- 每组记录：Best Macro F1 / Train Acc / Val Acc / Gap / Best Epoch
- 汇总为统一表格（Markdown）
- 结论输出：最佳策略及其相对 B0 的提升

---

## 8. 实验流程（执行顺序）
1) 跑 B0 baseline  
2) Phase 1：不均衡策略消融  
3) Phase 2：增强策略消融  
4) Phase 3：组合验证  
5) 汇总结果表，输出结论  

---

## 9. 评价标准（验收）
- 至少 1 个策略在 Macro F1 上稳定优于 B0  
- 最优策略在 3 seeds 上波动可控（std ≤ 0.01）
