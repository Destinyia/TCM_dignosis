# Mask 背景增强消融实验报告（2026-02-16）

## 实验说明
- 损失函数：普通交叉熵 (CE)
- 数据来源：`runs/ablation_mask_bg/*/*/metrics_history.jsonl`
- 指标口径：按 `val_macro_f1` 选最佳 epoch

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |
| --- | --- | --- | --- | --- |
| B0_baseline_ce | Baseline CE损失，无mask增强 | 0.2624 | 0.6426 | 14 |
| A1_bg_blur | 背景高斯模糊 | 0.2651 | 0.6444 | 23 |
| A2_bg_solid_gray | 固定灰色背景 (114,114,114) | 0.2612 | 0.6338 | 27 |
| A3_bg_solid_random | 随机纯色背景 | 0.2861 | 0.6620 | 14 |
| A4_bg_noise | 随机噪声背景 | 0.2915 | 0.6444 | 26 |
| A5_bg_random_mode | 随机背景模式 (blur/solid/noise随机) | 0.2806 | 0.6320 | 14 |

## 结论摘要
- 最佳实验：`A4_bg_noise`（Best Macro F1=0.2915）
