# 注意力机制消融对比实验报告（2026-02-14）

## 实验说明
- 数据来源：`runs/ablation_attention/*/*/metrics_history.jsonl`
- 指标口径：按 `val_macro_f1` 选最佳 epoch，并记录对应的 `val_accuracy`
- 基线：无注意力模型（Best Macro F1=0.3402，Accuracy=0.6391）

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |
| --- | --- | --- | --- | --- |
| E1a_soft_0.3 | 更高的 soft_floor（保留更多背景） | 0.2632 | 0.6426 | 22 |
| E1b_alpha_neg1 | 负 init_alpha（初始更依赖注意力） | 0.2660 | 0.6303 | 17 |
| E1c_no_channel | 无通道注意力（简化模型） | 0.2715 | 0.6408 | 19 |
| E1d_gate_mode | gate 模式（加性增强） | 0.2909 | 0.6461 | 20 |
| E1e_weighted | soft attention + weighted sampler | 0.2677 | 0.6356 | 10 |
| E2_multiscale | 多尺度注意力 | 0.2744 | 0.6408 | 10 |
| E2a_multiscale_weighted | 多尺度注意力 + weighted sampler | 0.2808 | 0.6180 | 3 |

## 结论摘要
- 最佳实验：`E1d_gate_mode`（Best Macro F1=0.2909）。
- 与基线相比：下降 14.49%（0.2909 vs 0.3402）。
- 整体趋势：注意力变体未超过基线，准确率大多在 0.6180–0.6461 区间。

## 建议（与当前结果匹配的下一步）
- 不再加深同类注意力微调，转向数据与任务层面改进。
- 优先尝试：
  1) 用分割 mask 做数据增强（裁剪/背景替换）而非特征加权。
  2) 多任务学习（分类 + 分割联合训练）。
  3) 更强的不平衡处理（过采样、class-balanced loss、合并极少类）。
