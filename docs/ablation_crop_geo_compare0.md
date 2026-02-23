# Crop 裁剪与几何变换消融实验报告（2026-02-18）

## 实验说明
- Baseline：noise 背景 0.5
- 探索：crop 裁剪与几何变换的组合效果（包含不同损失/骨干的对照）

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Train Acc | Val Acc | Gap | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- |
| B0_noise_baseline | noise背景0.5 (baseline) | 0.2921 | 0.8468 | 0.6232 | 0.2236 | 14 |
| S1_EfficientNet_B0 | EffB0 + CB + mask + noise0.5 | 0.2464 | 0.8524 | 0.6461 | 0.2062 | 12 |
| S2_EfficientNet_B3 | EffB3 + CB + mask + noise0.5 | 0.2694 | 0.8498 | 0.6303 | 0.2195 | 8 |
| C01_crop | noise0.5 + crop裁剪 | 0.2684 | 0.8917 | 0.6426 | 0.2491 | 14 |
| C02_scale | noise0.5 + 随机缩放 | 0.2691 | 0.9499 | 0.6514 | 0.2985 | 23 |
| C03_rotate5_noise | noise0.5 + 随机旋转5度(p=0.2, noise填充) | 0.2797 | 0.9609 | 0.6496 | 0.3112 | 30 |
| C04_crop_scale_rotate5_noise | noise0.5 + crop + 缩放 + 旋转5度(p=0.1, noise填充) | 0.2764 | 0.8611 | 0.6567 | 0.2044 | 17 |
| C05_crop_scale_rotate5_noise | noise0.5 + crop + 缩放 + 旋转5度(p=0.2, noise填充) | 0.2901 | 0.8164 | 0.6285 | 0.1879 | 14 |

## 结论摘要
- 最佳实验：`B0_noise_baseline`（Best Macro F1=0.2921）
- 相比 noise baseline 提升：+0.0000
