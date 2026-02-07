# 前期阶段开发报告（阶段1-3）

**日期**: 2026-02-04
**范围**: 阶段1（环境与结构）、阶段2（数据层）、阶段3（模型层）

## 概述
已按 `DEVELOPMENT_PLAN.md` 完成前3个阶段的基础开发工作：项目结构搭建、配置系统、数据集/增强/采样器模块，以及模型层（backbone/neck/head/detector）与对应测试。

## 阶段1：环境与项目结构
- 完成 `src/tcm_tongue/` 包结构与基础 `__init__.py` 导出
- 添加依赖清单与安装配置：`requirements.txt`、`setup.py`
- 新增测试路径自动注入：`tests/conftest.py`

## 阶段2：数据层模块
- 配置系统：`src/tcm_tongue/config/default.py`
  - 支持 YAML 读写、_base_ 继承与覆盖
  - 单元测试：`tests/test_config.py`
- COCO 数据集：`src/tcm_tongue/data/dataset.py`
  - 读取 COCO annotations 与 images
  - 返回标准 target（boxes/labels/image_id/area/iscrowd）
  - 单元测试：`tests/test_dataset.py`
- 数据增强与先验增强：`src/tcm_tongue/data/transforms.py`
  - Albumentations 训练/验证增强
  - 体质颜色先验增强（cold/heat/damp）
  - 单元测试：`tests/test_transforms.py`（依赖 albumentations）
- 采样器：`src/tcm_tongue/data/sampler.py`
  - 过采样（ClassBalancedSampler）
  - 分层采样（StratifiedSampler）
  - 欠采样（UnderSampler）
  - 单元测试：`tests/test_sampler.py`

## 阶段3：模型层模块
- Backbone：`src/tcm_tongue/models/backbone.py`
  - ResNet50/101、Swin-T/S
  - 支持冻结前几个stage
- Neck：`src/tcm_tongue/models/neck.py`
  - FPN 实现（BiFPN/PAFPN 占位）
- Head：`src/tcm_tongue/models/head.py`
  - 采用 torchvision 内置检测模型封装（Faster R-CNN/FCOS/RetinaNet）
  - 当前实现为直接调用 torchvision detector，绕过自定义 backbone/neck
- Detector：`src/tcm_tongue/models/detector.py`
  - 统一 forward / predict / 后处理
  - 检测头需特征时使用自定义 backbone/neck；否则直接走 torchvision
- 单元测试：`tests/test_backbone.py`、`tests/test_detector.py`

## 测试结果
运行命令：
```
conda run -n cv pytest tests/ -v
```
结果：23 passed, 4 skipped
- 跳过项：`tests/test_transforms.py`（缺少 albumentations 依赖时自动 skip）

## 关键说明/已知限制
- Head 目前使用 torchvision detector wrapper，暂未实现自定义 RPN/ROI 或 FCOS/RetinaNet head 细节
- BiFPN/PAFPN 尚未实现（占位）
- 自定义 backbone/neck 的 end-to-end 组合后续需要与 head 进一步对接

## 下一步建议（阶段4）
- 实现损失函数模块：Focal Loss、Weighted CE
- 补齐模型层细节（若计划替换 torchvision 内置检测头）
- 增强训练/评估模块并补充测试
