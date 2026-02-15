# Mask 增强与多任务对比实验报告（2026-02-15）

## 实验说明
- 数据来源：`runs/ablation_mask_multitask/*/*/metrics_history.jsonl`
- 指标口径：按 `val_macro_f1` 选最佳 epoch，并记录对应的 `val_accuracy`

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |
| --- | --- | --- | --- | --- |
| B0_baseline | Baseline 无mask增强 | 0.2688 | 0.6514 | 15 |
| A1_mask_crop | Mask裁剪增强（真实分割mask） | 0.2005 | 0.6303 | 7 |
| A2_mask_bg_blur | Mask背景模糊（dilate=25） | 0.2909 | 0.6391 | 16 |
| M1_multitask_seg_v2 | 分类+分割联合训练 | 未运行 | 未运行 | - |

## 结论摘要
- 最佳实验：`A2_mask_bg_blur`（Best Macro F1=0.2909）。
