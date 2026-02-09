# 实验对比说明（更新至 2026-02-09）

## 说明与范围

- 目标：对比不同实验配置的性能，重点关注**样本不均衡处理方法**。
- 模型：`faster_rcnn_v2 + resnet50 + fpn`
- 数据：`shezhenv3-coco`
- 训练规模：默认使用 `train_size=4000`、`val_size=1000`（以日志记录为准）
- 当前 baseline 已切换为 **strong_augment**（强增强 + mosaic + mixup）。

> 注：表格中使用**每个实验前缀的最新一次 run**进行对比。

---

## 总体对比（最新 run）

| experiment | run | mAP | mAP50 | mAP_small | mAP_medium | mAP_large |
| --- | --- | --- | --- | --- | --- | --- |
| class_balanced_focal | class_balanced_focal_20260208_232803 | 0.1909 | 0.2697 | 0.0767 | 0.2047 | 0.2094 |
| strong_augment | strong_augment_20260208_122147 | 0.1880 | 0.2684 | 0.0764 | 0.1534 | 0.2053 |
| augment_test | augment_test_20260208_100144 | 0.1864 | 0.2712 | 0.0747 | 0.2494 | 0.2020 |
| weighted_ce_sqrt | weighted_ce_sqrt_20260208_214236 | 0.1840 | 0.2701 | 0.0597 | 0.2074 | 0.2071 |
| focal_loss | focal_loss_20260208_183529 | 0.1823 | 0.2610 | 0.0658 | 0.1563 | 0.2025 |
| weighted_ce | weighted_ce_20260208_201807 | 0.1783 | 0.2581 | 0.0702 | 0.1475 | 0.1966 |
| baseline | baseline_20260208_171600 | 0.1736 | 0.2598 | 0.0606 | 0.2334 | 0.1914 |
| tcm_prior_augment | tcm_prior_augment_20260208_140841 | 0.1692 | 0.2379 | 0.0895 | 0.1129 | 0.1891 |

---

## 不均衡方法对比（相对 baseline）

baseline run：`baseline_20260208_171600`

### mAP / mAP50 变化

| 方法 | mAP | ΔmAP | mAP50 | ΔmAP50 |
| --- | --- | --- | --- | --- |
| class_balanced_focal | 0.1909 | **+0.0173** | 0.2697 | **+0.0099** |
| weighted_ce_sqrt | 0.1840 | +0.0104 | 0.2701 | **+0.0103** |
| focal_loss | 0.1823 | +0.0088 | 0.2610 | +0.0012 |
| weighted_ce | 0.1783 | +0.0048 | 0.2581 | -0.0017 |

### Small / Medium / Large

| 方法 | mAP_small | Δ | mAP_medium | Δ | mAP_large | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| class_balanced_focal | 0.0767 | **+0.0161** | 0.2047 | -0.0287 | 0.2094 | +0.0180 |
| weighted_ce_sqrt | 0.0597 | -0.0009 | 0.2074 | -0.0260 | 0.2071 | +0.0157 |
| focal_loss | 0.0658 | +0.0051 | 0.1563 | -0.0771 | 0.2025 | +0.0112 |
| weighted_ce | 0.0702 | +0.0096 | 0.1475 | -0.0860 | 0.1966 | +0.0052 |

**解读**
- `class_balanced_focal` 整体最好，mAP 与 mAP50 均领先 baseline。
- 多数不均衡方法**提升小目标 mAP**，但**牺牲中等目标 mAP**。
- `weighted_ce_sqrt` 对 mAP50 有优势，但 small 改善有限。

---

## 结论与建议

1) **当前最优：`class_balanced_focal`**  
   若目标是提升整体精度与长尾类，优先使用该配置。

2) **强增强 baseline 仍是稳定基线**  
   `strong_augment` 与 `augment_test` 在 mAP50 上表现稳定，适合作为主线 baseline。

3) **tcm_prior_augment 仍不足以替代强增强**  
   小目标有所提升，但整体落后于 strong/augment_test。

---

## 待补充实验（尚未生成 metrics）

以下不均衡方法尚未生成 `metrics.json`，建议重新跑（已修复 Subset + sampler 问题）：
- `oversampling`
- `stratified`
- `oversample_focal`
- `combined`

完成后可按相同表格继续补充对比。
