# TCM Diagnosis 项目记录

## 2026-02-04: shezhenv3-coco 数据集分析

### 运行命令
```bash
conda run -n cv python analyze_shezhenv3_coco.py
```

### 数据集概述

| 分割 | 图像数量 | 标注数量 |
|------|----------|----------|
| train | 5594 | 14677 |
| val | 572 | 1874 |
| test | 553 | 1672 |
| **总计** | **6719** | **18223** |

### 类别分布（21个类别）

| 类别名称 | 总数 | 中文含义 |
|----------|------|----------|
| baitaishe | 5040 | 白苔舌 |
| hongdianshe | 3653 | 红点舌 |
| liewenshe | 1886 | 裂纹舌 |
| chihenshe | 1482 | 齿痕舌 |
| hongshe | 1466 | 红舌 |
| huangtaishe | 1119 | 黄苔舌 |
| pangdashe | 689 | 胖大舌 |
| botaishe | 581 | 薄苔舌 |
| shenquao | 495 | 肾区凹 |
| xinfeiao | 372 | 心肺凹 |
| gandanao | 292 | 肝胆凹 |
| shoushe | 285 | 瘦舌 |
| huataishe | 283 | 滑苔舌 |
| zishe | 231 | 紫舌 |
| piweiao | 186 | 脾胃凹 |
| heitaishe | 122 | 黑苔舌 |
| jiankangshe | 21 | 健康舌 |
| gandantu | 9 | 肝胆凸 |
| unknown_21 | 7 | 未知类别 |
| shenqutu | 2 | 肾区凸 |
| xinfeitu | 2 | 心肺凸 |

### 类别不平衡问题

数据集存在严重的类别不平衡：
- **头部类别**: baitaishe(5040), hongdianshe(3653) 占总标注的 47.7%
- **尾部类别**: xinfeitu(2), shenqutu(2), gandantu(9) 样本极少
- 最大/最小比例: 5040:2 = 2520:1

### 生成的可视化文件

输出目录: `/mnt/d/workspace/DM/TCM_dignosis/runs/shezhenv3_coco_analysis_2026-02-04/`

| 文件名 | 描述 |
|--------|------|
| class_distribution_overall.svg | 总体类别分布柱状图 |
| class_distribution_train.svg | 训练集类别分布 |
| class_distribution_val.svg | 验证集类别分布 |
| class_distribution_test.svg | 测试集类别分布 |
| bbox_area_hist_overall.svg | bbox面积直方图(对数刻度) |
| bbox_width_hist_overall.svg | bbox宽度直方图 |
| bbox_height_hist_overall.svg | bbox高度直方图 |
| sample_5x5_train.png | 训练集5x5样本网格图 |
| sample_5x5_xinfeitu_shenqutu_gandantu.png | 稀有类别样本网格图 |

### 分析结论

1. 数据集为中医舌诊目标检测数据集，包含舌象特征标注
2. 类别涵盖舌质（红舌、紫舌等）、舌苔（白苔、黄苔等）、舌形（胖大、齿痕等）和脏腑区域特征
3. 类别严重不平衡，训练时需考虑过采样或损失函数加权
4. 部分稀有类别（凸类特征）样本过少，可能影响模型泛化能力

---

## 2026-02-04: 开发规划制定

### 生成文件

- `DEVELOPMENT_PLAN.md`: 完整的模块化舌诊检测模型开发规划

### 规划概述

**目标**: 开发基于 shezhenv3-coco 数据集的模块化舌诊目标检测模型

**开发阶段** (共17天):
1. **阶段1 (Day 1)**: 环境准备与数据探索
2. **阶段2-3 (Day 2-3)**: 数据层模块开发 (Dataset, Transforms, Sampler)
3. **阶段4-6 (Day 4-6)**: 模型层模块开发 (Backbone, Neck, Head, Detector)
4. **阶段7 (Day 7)**: 损失函数模块 (Focal Loss, Weighted CE)
5. **阶段8-10 (Day 8-10)**: 训练与评估模块 (Trainer, Evaluator)
6. **阶段11 (Day 11)**: 推理接口与API
7. **阶段12-15 (Day 12-15)**: 类别不平衡专项优化实验 (11个实验)
8. **阶段16-17 (Day 16-17)**: 文档与收尾

### 核心模块

```
src/tcm_tongue/
├── config/     # 配置系统
├── data/       # 数据集、增强、采样器
├── models/     # Backbone、Neck、Head、Detector
├── losses/     # Focal Loss、Weighted CE
├── engine/     # Trainer、Evaluator
├── api/        # 推理接口
└── utils/      # 工具函数
```

### 类别不平衡优化策略

| 策略类型 | 方法 |
|---------|------|
| 数据采样 | 过采样、欠采样、分层采样 |
| 损失函数 | Focal Loss、加权交叉熵、类别平衡Focal Loss |
| 数据增强 | 基础增强、强增强、舌象先验增强 |

### 开发方法

- 测试驱动开发 (TDD)
- 模块化设计，标准化接口
- 使用官方预训练模型 (ResNet-50/101, Swin Transformer)
- 运行环境: `conda activate cv`

---

## 2026-02-04

- 新增项目结构 `src/tcm_tongue/` 与基础包初始化
- 添加 `requirements.txt` 与 `setup.py`
- 实现配置系统 `Config` 并补充单元测试
- 实现 COCO 数据集模块、数据增强与采样器模块及相应测试

## 2026-02-04 (continued)

- 实现模型层基础：Backbone、Neck、Head、Detector
- 添加 backbone/detector 单元测试

## 2026-02-04 (warnings fix)

- 更新 PadIfNeeded 参数为 fill，避免 albumentations 警告
- 在 tests/conftest.py 设置 NO_ALBUMENTATIONS_UPDATE 以静默版本检查警告

## 2026-02-04 (stage 4)

- 实现损失函数模块：FocalLoss、WeightedCrossEntropyLoss
- 添加损失函数单元测试

## 2026-02-04 (stage 5)

- 实现训练器与评估器（COCOEvaluator、Trainer）
- 添加训练/评估单元测试

## 2026-02-04 (stage 5 fix)

- 修复 COCOEvaluator per-class AP 统计对 numpy 数组使用 numel 的错误

## 2026-02-04 (stage 6)

- 实现推理接口 TongueDetectorInference
- 添加推理接口单元测试

## 2026-02-04 (cuda switch)

- 训练/评估测试改为优先使用 CUDA（可用时），否则回退 CPU

## 2026-02-04 (amp warning fix)

- 使用 torch.amp GradScaler/autocast 替代 torch.cuda.amp，消除 FutureWarning

## 2026-02-04 (stage 7)

- 添加实验配置与脚本（configs/experiments、scripts/run_experiments.py、scripts/analyze_experiments.py）
- 添加基础训练脚本 scripts/train.py

## 2026-02-04 (experiment runner fix)

- 修复 scripts/train.py 运行时无法导入 tcm_tongue（补充 src 路径）

## 2026-02-04 (train script logging fix)

- 修复 scripts/train.py 中 _print_env_config 定义顺序导致的 NameError

## 2026-02-04 (train crash fix)

- 数据集读取时对 COCO bbox 做边界裁剪，避免 Albumentations 报 y_max 超界

## 2026-02-04 (train crash fix 2)

- 允许 PIL 读取截断图片（ImageFile.LOAD_TRUNCATED_IMAGES）

## 2026-02-06: Baseline 不收敛问题诊断

### 问题现象

运行 baseline 实验 50 epochs 后，mAP 仅 0.12%，模型未收敛。

### 根因分析

| 问题 | 位置 | 影响 |
|------|------|------|
| 未使用预训练权重 | `head.py:49-53` | 从头训练极难收敛 |
| num_classes 未含背景类 | 配置 21，应为 22 | 分类头维度错误 |
| 学习率过高 | SGD lr=0.001 | 训练不稳定 |

### 计划调整

暂停当前实验，优先修复 baseline 收敛问题。采用快速迭代策略验证修复效果。

## 2026-02-06 (baseline 修复已实施)

- 模型 num_classes 调整为 22（包含背景类）
- 数据集标签 label_offset=1（训练时 +1，评估/推理时回映射）
- Faster R-CNN 使用预训练权重并替换分类头

---

## 2026-02-06: 开发计划调整

### 新增阶段: Baseline 修复 (插入在阶段 8 之前)

**目标**: 构建可收敛的 baseline 模型

**任务清单**:
1. 修改 `head.py`，添加预训练权重支持
2. 创建快速调试脚本 `scripts/debug_baseline.py`
3. 验证模型收敛 (少量样本快速迭代)
4. 确认 baseline mAP > 30% 后再进行后续实验

**快速迭代参数**:
- 训练样本: 200 张
- 验证样本: 50 张
- Epochs: 5-10
- Batch size: 4
- 学习率: 0.0001 (AdamW)

## 2026-02-06 (train logging enhancement)

- 训练脚本新增关键配置输出（num_classes、label_offset、pretrained 等）

## 2026-02-06 (baseline debug)

- 新增小样本快速验证脚本 scripts/debug_baseline.py

## 2026-02-06 (run_experiments options)

- run_experiments.py 支持 --only / --skip / --list 参数控制实验集合

## 2026-02-06 (run_experiments parameters)

- run_experiments.py 增加 batch-size、train-size、val-size、num-workers、seed 等参数透传
- train.py 支持 batch-size/num-workers/子集大小/随机种子覆盖

## 2026-02-06 (run_experiments defaults)

- run_experiments.py 默认只跑 baseline，并采用小样本参数（train=200/val=50, bs=4, epochs=5）

## 2026-02-06 (pretrained download retry)

- 预训练权重哈希不匹配时自动清理缓存并重试下载

## 2026-02-06 (evaluator subset fix)

- COCOEvaluator 支持 Subset，按子集 image_id 过滤 GT 与预测，避免 loadRes 断言失败

## 2026-02-06 (normalize fix)

- 针对 torchvision 检测模型，数据管道仅缩放到 [0,1]，避免与模型内部归一重复

## 2026-02-06 (resize pipeline fix)

- 对 torchvision 检测模型禁用数据管道中的 resize/pad，避免预测框坐标与原图不一致

## 2026-02-06 (baseline 进一步调整)

- 默认检测头切换为 Faster R-CNN v2（torchvision 官方实现）
- 配置新增 `data.image_size`，默认 `[800, 800]`
- Albumentations 在关闭 resize 时加入空操作以消除 bbox 警告

## 2026-02-06 (训练日志与指标)

- COCOEval 输出精简为表格（两行表头 + 分隔线），仅展示验证集指标
- 训练进度条显示显存：`allocated/reserved` + `free/total`
- train mAP50 仅在 epoch 结束后计算一次（基于训练子集）
- 追加 `metrics_history.jsonl`，用于分析过拟合趋势
- `metrics.json` 同步记录 `last_train_mAP_50`

## 协作约定（通用）

- 每完成一个阶段需运行全量单元测试（用户在 CUDA 环境执行）
- 发现问题需更新 DEVELOPMENT_PLAN 或 codex.MD
- 需要安装新依赖或调整计划时先征询用户
