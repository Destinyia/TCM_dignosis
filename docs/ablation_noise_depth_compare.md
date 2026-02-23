# Noise 背景增强深度消融实验报告（2026-02-16）

## 实验说明
- 损失函数：普通交叉熵 (CE)
- 基于前序实验结论：noise 背景效果最佳 (F1=0.2915)
- 本轮探索：不同概率、组合策略

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |
| --- | --- | --- | --- | --- |
| B0_baseline_ce | Baseline CE损失，无mask增强 | 0.2711 | 0.6303 | 25 |
| N1_noise_p50 | noise背景，概率0.5 | 0.2777 | 0.6549 | 32 |
| N2_noise_p70 | noise背景，概率0.7 | 0.2790 | 0.6549 | 15 |
| N3_noise_p90 | noise背景，概率0.9 | 0.2763 | 0.6496 | 26 |
| N4_noise_crop | noise背景0.5 + crop裁剪 | 0.2890 | 0.6373 | 20 |
| N5_noise_strong | noise背景0.5 + 强增强 | 0.2764 | 0.6215 | 18 |

## 结论摘要
- 最佳实验：`N4_noise_crop`（Best Macro F1=0.2890）
- 相比 Baseline 提升：+0.0178
