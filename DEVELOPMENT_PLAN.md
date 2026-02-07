# 舌诊目标检测模型开发规划

## 项目概述

**项目名称**: TCM Tongue Diagnosis Detection System
**目标**: 基于 shezhenv3-coco 数据集开发模块化舌诊目标检测模型，用于健康评估和疾病风险检测
**数据集**: 6719 张图像，18223 个标注，21 个类别
**开发方法**: 测试驱动开发 (TDD)
**运行环境**: `conda activate cv`

---

## 项目结构

```
TCM_dignosis/
├── src/
│   └── tcm_tongue/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── default.py          # 默认配置
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py          # COCO数据集加载器
│       │   ├── transforms.py       # 数据增强
│       │   └── sampler.py          # 采样器(过采样/欠采样/分层)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbone.py         # 骨干网络(预训练模型)
│       │   ├── neck.py             # 特征融合(FPN等)
│       │   ├── head.py             # 检测头
│       │   └── detector.py         # 完整检测器
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── focal_loss.py       # Focal Loss
│       │   └── weighted_ce.py      # 加权交叉熵
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── trainer.py          # 训练器
│       │   └── evaluator.py        # 评估器
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── metrics.py          # 评估指标
│       │   ├── visualization.py    # 可视化工具
│       │   └── checkpoint.py       # 模型保存/加载
│       └── api/
│           ├── __init__.py
│           └── inference.py        # 推理接口
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_transforms.py
│   ├── test_sampler.py
│   ├── test_backbone.py
│   ├── test_detector.py
│   ├── test_losses.py
│   ├── test_trainer.py
│   └── test_inference.py
├── configs/
│   ├── base.yaml                   # 基础配置
│   ├── faster_rcnn_r50.yaml        # Faster R-CNN + ResNet50
│   ├── fcos_r50.yaml               # FCOS + ResNet50
│   └── experiments/
│       ├── baseline.yaml
│       ├── focal_loss.yaml
│       ├── weighted_ce.yaml
│       ├── oversampling.yaml
│       └── combined.yaml
├── scripts/
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── inference.py                # 推理脚本
│   └── export.py                   # 模型导出
├── runs/                           # 实验输出
├── datasets/                       # 数据集
├── docs/                           # 文档
├── requirements.txt
├── setup.py
├── claude.MD                       # 开发日志
└── codex.MD                        # 问题记录
```

---

## 开发阶段

### 状态更新 (2026-02-06)

**已完成阶段**:
- [x] 阶段1: 环境准备与数据探索基础
- [x] 阶段2: 数据层模块（dataset/transforms/sampler）
- [x] 阶段3: 模型层模块（backbone/neck/head/detector）
- [x] 阶段4: 损失函数模块
- [x] 阶段5: 训练与评估模块
- [x] 阶段6: 推理接口与 API
- [x] 阶段7: 实验配置与脚本

**新增修复**:
- [x] COCO bbox 越界修复（训练时裁剪至图像边界）
- [x] PIL 读取截断图片容错
- [x] AMP API 由 `torch.cuda.amp` 迁移至 `torch.amp`
- [x] 标签偏移与背景类修正：`label_offset=1`，`model.num_classes=22`
- [x] torchvision 检测模型外部 Normalize/Resize 关闭（避免双重归一与评估坐标失配）
- [x] COCOEvaluator 支持 Subset 过滤 + 静默冗余输出 + 指标表格化
- [x] 训练日志新增显存占用（allocated/reserved + free/total）与 epoch 级 train mAP50
- [x] 训练指标追加到 `metrics_history.jsonl`，`metrics.json` 记录 `last_train_mAP_50`
- [x] Albumentations 关闭 resize 时加入空操作以消除 bbox 警告
- [x] Faster R-CNN v2 检测头加入并设为默认配置
- [x] 配置新增 `data.image_size`（默认 `[800, 800]`）

**待处理重点**:
- [ ] Baseline 进一步优化（长尾类别/小目标召回，必要时调整采样器、损失与 anchor）
- [ ] 训练日志趋势分析（基于 `metrics_history.jsonl` 判断过拟合）
- [ ] 样本分布与标注一致性复核（val 集稀有类别覆盖）

**阶段性结果**:
- Baseline（Faster R-CNN v2）在 20 epochs 达到约 mAP=0.117 / mAP50=0.182，较早期显著提升，但长尾类别仍偏弱。


### 阶段 1: 环境准备与数据探索 (Day 1)

#### 1.1 环境配置

**任务清单**:
- [ ] 创建项目目录结构
- [ ] 编写 `requirements.txt`
- [ ] 编写 `setup.py` 使项目可安装
- [ ] 验证环境依赖

**requirements.txt 内容**:
```
torch>=2.0.0
torchvision>=0.15.0
pycocotools>=2.0.6
albumentations>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
pytest>=7.3.0
pytest-cov>=4.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

**验收标准**:
- `pip install -e .` 成功安装
- `python -c "import tcm_tongue"` 无报错

#### 1.2 数据集深度分析

**任务清单**:
- [ ] 分析各分割集的类别分布差异
- [ ] 分析 bbox 尺寸分布（小/中/大目标比例）
- [ ] 分析每张图像的平均标注数
- [ ] 分析类别共现关系（哪些类别经常同时出现）
- [ ] 生成数据分析报告

**输出文件**: `runs/data_analysis/detailed_report.md`

**关键数据点**:
```
类别不平衡统计:
- 头部类别 (>1000): baitaishe(5040), hongdianshe(3653), liewenshe(1886),
                    chihenshe(1482), hongshe(1466), huangtaishe(1119)
- 中部类别 (100-1000): pangdashe(689), botaishe(581), shenquao(495),
                       xinfeiao(372), gandanao(292), shoushe(285),
                       huataishe(283), zishe(231), piweiao(186), heitaishe(122)
- 尾部类别 (<100): jiankangshe(21), gandantu(9), unknown_21(7),
                   shenqutu(2), xinfeitu(2)

类别语义分组:
- 舌苔类: baitaishe, huangtaishe, botaishe, huataishe, heitaishe
- 舌质类: hongdianshe, hongshe, zishe
- 舌形类: liewenshe, chihenshe, pangdashe, shoushe
- 脏腑凹陷: shenquao, xinfeiao, gandanao, piweiao
- 脏腑凸起: shenqutu, xinfeitu, gandantu
- 健康: jiankangshe
```

---

### 阶段 2: 核心模块开发 - 数据层 (Day 2-3)

#### 2.1 配置系统

**文件**: `src/tcm_tongue/config/default.py`

**接口设计**:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class DataConfig:
    root: str = "datasets/shezhenv3-coco"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_classes: int = 21
    label_offset: int = 1
    image_size: List[int] = field(default_factory=lambda: [800, 800])
    batch_size: int = 8
    num_workers: int = 4
    normalize: bool = False
    resize_in_dataset: bool = False

@dataclass
class ModelConfig:
    backbone: str = "resnet50"  # resnet50, resnet101, swin_t, swin_s
    pretrained: bool = True
    neck: str = "fpn"           # fpn, bifpn, panet
    head: str = "faster_rcnn_v2"   # faster_rcnn, faster_rcnn_v2, fcos, retinanet
    num_classes: int = 22

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"  # step, cosine, warmup_cosine
    warmup_epochs: int = 3
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_metric: str = "mAP"

@dataclass
class LossConfig:
    type: str = "focal"         # ce, weighted_ce, focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None

@dataclass
class SamplerConfig:
    type: str = "default"       # default, oversample, undersample, stratified
    oversample_factor: float = 2.0

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从YAML文件加载配置"""
        pass

    def to_yaml(self, path: str) -> None:
        """保存配置到YAML文件"""
        pass
```

**单元测试**: `tests/test_config.py`
```python
def test_config_default():
    """测试默认配置创建"""

def test_config_from_yaml():
    """测试从YAML加载配置"""

def test_config_to_yaml():
    """测试保存配置到YAML"""

def test_config_override():
    """测试配置覆盖"""
```

#### 2.2 数据集模块

**文件**: `src/tcm_tongue/data/dataset.py`

**接口设计**:
```python
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable
import torch

class TongueCocoDataset(Dataset):
    """舌诊COCO格式数据集"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            split: 数据分割 (train/val/test)
            transforms: 数据增强变换
        """
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: (C, H, W) 图像张量
            target: {
                "boxes": (N, 4) xyxy格式
                "labels": (N,) 类别标签
                "image_id": int
                "area": (N,) bbox面积
                "iscrowd": (N,) 是否crowd
            }
        """
        pass

    def get_category_counts(self) -> Dict[int, int]:
        """获取各类别的标注数量"""
        pass

    def get_category_names(self) -> Dict[int, str]:
        """获取类别ID到名称的映射"""
        pass

    @staticmethod
    def collate_fn(batch: List) -> Tuple[List[torch.Tensor], List[Dict]]:
        """批次整理函数"""
        pass
```

**单元测试**: `tests/test_dataset.py`
```python
import pytest

class TestTongueCocoDataset:

    @pytest.fixture
    def dataset(self):
        """创建测试数据集实例"""
        return TongueCocoDataset(root="datasets/shezhenv3-coco", split="train")

    def test_dataset_length(self, dataset):
        """测试数据集长度"""
        assert len(dataset) == 5594

    def test_getitem_returns_correct_format(self, dataset):
        """测试__getitem__返回正确格式"""
        image, target = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # C, H, W
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].shape[1] == 4

    def test_category_counts(self, dataset):
        """测试类别计数"""
        counts = dataset.get_category_counts()
        assert sum(counts.values()) == 14677  # train标注总数

    def test_collate_fn(self, dataset):
        """测试批次整理函数"""
        batch = [dataset[i] for i in range(4)]
        images, targets = TongueCocoDataset.collate_fn(batch)
        assert len(images) == 4
        assert len(targets) == 4
```

#### 2.3 数据增强模块

**文件**: `src/tcm_tongue/data/transforms.py`

**接口设计**:
```python
from typing import Dict, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class BaseTransform:
    """基础变换类"""

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        pass

class TrainTransform(BaseTransform):
    """训练时数据增强"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 800),
        use_mosaic: bool = False,
        use_mixup: bool = False,
    ):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                value=(114, 114, 114)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1
        ))

class ValTransform(BaseTransform):
    """验证/测试时变换"""

    def __init__(self, image_size: Tuple[int, int] = (800, 800)):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                value=(114, 114, 114)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1
        ))

class TonguePriorAugment:
    """基于舌象先验知识的数据增强

    舌诊先验知识:
    1. 舌象颜色变化反映体质状态
    2. 舌苔厚薄反映病邪深浅
    3. 脏腑区域位置相对固定
    """

    def __init__(self):
        # 模拟不同体质的颜色变化
        self.color_shifts = {
            'cold': {'hue': -10, 'sat': -20},      # 寒证：舌色偏淡
            'heat': {'hue': 10, 'sat': 20},        # 热证：舌色偏红
            'damp': {'hue': 5, 'sat': -10},        # 湿证：舌苔厚腻
        }

    def apply_constitution_shift(
        self,
        image: np.ndarray,
        constitution: str
    ) -> np.ndarray:
        """应用体质相关的颜色变换"""
        pass
```

**单元测试**: `tests/test_transforms.py`
```python
class TestTransforms:

    def test_train_transform_output_shape(self):
        """测试训练变换输出形状"""

    def test_val_transform_output_shape(self):
        """测试验证变换输出形状"""

    def test_bbox_preserved_after_transform(self):
        """测试变换后bbox正确保留"""

    def test_augmentation_increases_diversity(self):
        """测试数据增强增加多样性"""
```

#### 2.4 采样器模块

**文件**: `src/tcm_tongue/data/sampler.py`

**接口设计**:
```python
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Iterator, List, Dict
import numpy as np

class ClassBalancedSampler(Sampler):
    """类别平衡采样器 - 过采样策略"""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        oversample_factor: float = 2.0,
    ):
        """
        Args:
            dataset: 数据集实例
            oversample_factor: 过采样因子，对尾部类别过采样的倍数
        """
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self._compute_weights()

    def _compute_weights(self):
        """计算每个样本的采样权重"""
        pass

    def __iter__(self) -> Iterator[int]:
        pass

    def __len__(self) -> int:
        pass

class StratifiedSampler(Sampler):
    """分层采样器 - 确保每个batch包含各类别样本"""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        batch_size: int,
        drop_last: bool = True,
    ):
        pass

class UnderSampler(Sampler):
    """欠采样器 - 对头部类别进行欠采样"""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        target_ratio: float = 0.5,  # 头部类别保留比例
    ):
        pass

def create_sampler(
    sampler_type: str,
    dataset: TongueCocoDataset,
    **kwargs
) -> Sampler:
    """采样器工厂函数"""
    samplers = {
        "default": None,
        "oversample": ClassBalancedSampler,
        "undersample": UnderSampler,
        "stratified": StratifiedSampler,
    }
    if sampler_type == "default":
        return None
    return samplers[sampler_type](dataset, **kwargs)
```

**单元测试**: `tests/test_sampler.py`
```python
class TestSamplers:

    def test_class_balanced_sampler_weights(self):
        """测试类别平衡采样器权重计算"""

    def test_class_balanced_sampler_distribution(self):
        """测试过采样后类别分布更均衡"""

    def test_stratified_sampler_batch_composition(self):
        """测试分层采样器每批次包含多类别"""

    def test_undersampler_reduces_head_classes(self):
        """测试欠采样器减少头部类别样本"""
```

---

### 阶段 3: 核心模块开发 - 模型层 (Day 4-6)

#### 3.1 骨干网络模块

**文件**: `src/tcm_tongue/models/backbone.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, resnet101,
    swin_t, swin_s,
    ResNet50_Weights, ResNet101_Weights,
    Swin_T_Weights, Swin_S_Weights
)
from typing import Dict, List, Tuple

class BackboneBase(nn.Module):
    """骨干网络基类"""

    def __init__(self, out_channels: List[int]):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            features: 多尺度特征图字典
                {"feat0": (B, C0, H/4, W/4),
                 "feat1": (B, C1, H/8, W/8),
                 "feat2": (B, C2, H/16, W/16),
                 "feat3": (B, C3, H/32, W/32)}
        """
        raise NotImplementedError

class ResNetBackbone(BackboneBase):
    """ResNet骨干网络"""

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        frozen_stages: int = 1,
    ):
        if depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
            out_channels = [256, 512, 1024, 2048]
        elif depth == 101:
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet101(weights=weights)
            out_channels = [256, 512, 1024, 2048]

        super().__init__(out_channels)

        # 提取各阶段
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int):
        """冻结前num_stages个阶段"""
        pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

class SwinBackbone(BackboneBase):
    """Swin Transformer骨干网络"""

    def __init__(
        self,
        variant: str = "tiny",  # tiny, small
        pretrained: bool = True,
    ):
        pass

def create_backbone(
    name: str,
    pretrained: bool = True,
    **kwargs
) -> BackboneBase:
    """骨干网络工厂函数"""
    backbones = {
        "resnet50": lambda: ResNetBackbone(50, pretrained, **kwargs),
        "resnet101": lambda: ResNetBackbone(101, pretrained, **kwargs),
        "swin_t": lambda: SwinBackbone("tiny", pretrained),
        "swin_s": lambda: SwinBackbone("small", pretrained),
    }
    return backbones[name]()
```

**单元测试**: `tests/test_backbone.py`
```python
class TestBackbone:

    @pytest.mark.parametrize("backbone_name", ["resnet50", "resnet101", "swin_t"])
    def test_backbone_output_channels(self, backbone_name):
        """测试骨干网络输出通道数"""

    def test_backbone_output_shapes(self):
        """测试骨干网络输出形状"""
        backbone = create_backbone("resnet50")
        x = torch.randn(2, 3, 800, 800)
        features = backbone(x)
        assert features["feat0"].shape == (2, 256, 200, 200)
        assert features["feat1"].shape == (2, 512, 100, 100)
        assert features["feat2"].shape == (2, 1024, 50, 50)
        assert features["feat3"].shape == (2, 2048, 25, 25)

    def test_pretrained_weights_loaded(self):
        """测试预训练权重正确加载"""

    def test_frozen_stages(self):
        """测试冻结阶段参数不更新"""
```

#### 3.2 特征融合模块

**文件**: `src/tcm_tongue/models/neck.py`

**接口设计**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class FPN(nn.Module):
    """Feature Pyramid Network"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 5,
    ):
        """
        Args:
            in_channels: 骨干网络各层输出通道数 [256, 512, 1024, 2048]
            out_channels: FPN输出通道数
            num_outs: FPN输出层数
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )

        # 额外的下采样层
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(num_outs - len(in_channels)):
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            features: 骨干网络多尺度特征
        Returns:
            fpn_features: [P2, P3, P4, P5, P6] 多尺度特征金字塔
        """
        pass

class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network"""
    pass

class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network"""
    pass

def create_neck(
    name: str,
    in_channels: List[int],
    out_channels: int = 256,
    **kwargs
) -> nn.Module:
    """特征融合模块工厂函数"""
    necks = {
        "fpn": FPN,
        "bifpn": BiFPN,
        "pafpn": PAFPN,
    }
    return necks[name](in_channels, out_channels, **kwargs)
```

#### 3.3 检测头模块

**文件**: `src/tcm_tongue/models/head.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops
from typing import Dict, List, Tuple, Optional

class DetectionHead(nn.Module):
    """检测头基类"""

    def __init__(self, num_classes: int, in_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(
        self,
        features: List[torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        """
        Args:
            features: FPN多尺度特征
            targets: 训练时的目标标注
        Returns:
            losses: 训练时返回损失字典
            detections: 推理时返回检测结果
        """
        raise NotImplementedError

class FasterRCNNHead(DetectionHead):
    """Faster R-CNN检测头 (RPN + ROI Head)"""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        # RPN参数
        rpn_anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,)),
        rpn_aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 5,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        # ROI参数
        box_roi_pool_output_size: int = 7,
        box_roi_pool_sampling_ratio: int = 2,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
    ):
        super().__init__(num_classes, in_channels)
        # 构建RPN和ROI Head
        pass

class FCOSHead(DetectionHead):
    """FCOS检测头 (Anchor-free)"""
    pass

class RetinaNetHead(DetectionHead):
    """RetinaNet检测头 (Anchor-based, 使用Focal Loss)"""
    pass

def create_head(
    name: str,
    num_classes: int,
    in_channels: int = 256,
    **kwargs
) -> DetectionHead:
    """检测头工厂函数"""
    heads = {
        "faster_rcnn": FasterRCNNHead,
        "fcos": FCOSHead,
        "retinanet": RetinaNetHead,
    }
    return heads[name](num_classes, in_channels, **kwargs)
```

#### 3.4 完整检测器

**文件**: `src/tcm_tongue/models/detector.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class TongueDetector(nn.Module):
    """舌诊目标检测器 - 端到端模型"""

    def __init__(
        self,
        backbone: BackboneBase,
        neck: nn.Module,
        head: DetectionHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        """
        Args:
            images: 图像列表 [(C, H, W), ...]
            targets: 目标列表 (训练时)
        Returns:
            losses: 损失字典 (训练时)
            detections: 检测结果列表 (推理时)
        """
        # 提取特征
        features = self.backbone(images)

        # 特征融合
        fpn_features = self.neck(features)

        # 检测头
        losses, detections = self.head(fpn_features, targets)

        return losses, detections

    @torch.no_grad()
    def predict(
        self,
        images: List[torch.Tensor],
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
    ) -> List[Dict]:
        """推理接口"""
        self.eval()
        _, detections = self.forward(images)
        # 后处理: 过滤低置信度和NMS
        return self._postprocess(detections, score_thresh, nms_thresh)

def build_detector(config: Config) -> TongueDetector:
    """根据配置构建检测器"""
    backbone = create_backbone(
        config.model.backbone,
        pretrained=config.model.pretrained
    )
    neck = create_neck(
        config.model.neck,
        in_channels=backbone.out_channels
    )
    head = create_head(
        config.model.head,
        num_classes=config.model.num_classes
    )
    return TongueDetector(backbone, neck, head)
```

**单元测试**: `tests/test_detector.py`
```python
class TestTongueDetector:

    @pytest.fixture
    def detector(self):
        config = Config()
        return build_detector(config)

    def test_detector_forward_train(self, detector):
        """测试训练模式前向传播"""
        images = [torch.randn(3, 800, 800) for _ in range(2)]
        targets = [
            {"boxes": torch.tensor([[100, 100, 200, 200]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[50, 50, 150, 150]]), "labels": torch.tensor([2])},
        ]
        detector.train()
        losses, _ = detector(images, targets)
        assert "loss_classifier" in losses or "loss_cls" in losses
        assert "loss_box_reg" in losses or "loss_bbox" in losses

    def test_detector_forward_eval(self, detector):
        """测试推理模式前向传播"""
        images = [torch.randn(3, 800, 800) for _ in range(2)]
        detector.eval()
        _, detections = detector(images)
        assert len(detections) == 2
        assert "boxes" in detections[0]
        assert "labels" in detections[0]
        assert "scores" in detections[0]

    def test_detector_predict(self, detector):
        """测试预测接口"""
        images = [torch.randn(3, 800, 800)]
        results = detector.predict(images)
        assert len(results) == 1
```

---

### 阶段 4: 核心模块开发 - 损失函数 (Day 7)

#### 4.1 Focal Loss

**文件**: `src/tcm_tongue/losses/focal_loss.py`

**接口设计**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: 平衡因子，用于调节正负样本权重
            gamma: 聚焦参数，gamma越大，对易分类样本的惩罚越小
            reduction: 损失聚合方式 (none, mean, sum)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) 预测logits
            targets: (N,) 目标类别
        Returns:
            loss: 标量损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class ClassBalancedFocalLoss(FocalLoss):
    """类别平衡的Focal Loss

    在Focal Loss基础上，根据各类别样本数量自动计算alpha权重
    """

    def __init__(
        self,
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            class_counts: 各类别样本数量
            beta: 有效样本数计算参数
            gamma: 聚焦参数
        """
        # 计算类别平衡权重
        # effective_num = 1 - beta^n
        # weights = (1 - beta) / effective_num
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)

        super().__init__(alpha=1.0, gamma=gamma, reduction=reduction)
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
```

#### 4.2 加权交叉熵损失

**文件**: `src/tcm_tongue/losses/weighted_ce.py`

**接口设计**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失"""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.reduction = reduction

    @classmethod
    def from_class_counts(
        cls,
        class_counts: List[int],
        weighting: str = "inverse",  # inverse, inverse_sqrt, effective_num
    ) -> "WeightedCrossEntropyLoss":
        """根据类别数量计算权重

        weighting strategies:
        - inverse: w_i = N / n_i (反比)
        - inverse_sqrt: w_i = sqrt(N / n_i) (平方根反比)
        - effective_num: 基于有效样本数
        """
        counts = torch.tensor(class_counts, dtype=torch.float)
        total = counts.sum()

        if weighting == "inverse":
            weights = total / counts
        elif weighting == "inverse_sqrt":
            weights = torch.sqrt(total / counts)
        elif weighting == "effective_num":
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / effective_num

        # 归一化
        weights = weights / weights.sum() * len(weights)
        return cls(weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction=self.reduction
        )
```

**单元测试**: `tests/test_losses.py`
```python
class TestLosses:

    def test_focal_loss_reduces_easy_sample_loss(self):
        """测试Focal Loss降低易分类样本损失"""

    def test_focal_loss_gamma_effect(self):
        """测试gamma参数对损失的影响"""

    def test_weighted_ce_class_weights(self):
        """测试加权交叉熵类别权重"""

    def test_class_balanced_focal_loss(self):
        """测试类别平衡Focal Loss"""
```

---

### 阶段 5: 核心模块开发 - 训练与评估 (Day 8-10)

#### 5.1 训练器模块

**文件**: `src/tcm_tongue/engine/trainer.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Callable
import logging
from tqdm import tqdm

class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "runs/train",
        log_interval: int = 50,
        eval_interval: int = 1,
        save_interval: int = 5,
        max_epochs: int = 50,
        grad_clip: Optional[float] = None,
        amp: bool = True,  # 自动混合精度
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.amp = amp

        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        self.logger = logging.getLogger(__name__)
        self.best_metric = 0.0
        self.current_epoch = 0

        # 回调函数
        self.callbacks: Dict[str, Callable] = {}

    def train(self) -> Dict[str, float]:
        """执行完整训练流程"""
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self._train_epoch()

            # 评估
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._evaluate()

                # 保存最佳模型
                if val_metrics["mAP"] > self.best_metric:
                    self.best_metric = val_metrics["mAP"]
                    self._save_checkpoint("best.pth")

            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth")

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

        return {"best_mAP": self.best_metric}

    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            if self.amp:
                with torch.cuda.amp.autocast():
                    losses, _ = self.model(images, targets)
                    loss = sum(losses.values())

                self.scaler.scale(loss).backward()

                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses, _ = self.model(images, targets)
                loss = sum(losses.values())
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({"loss": loss.item()})

        return {"train_loss": total_loss / len(self.train_loader)}

    def _evaluate(self) -> Dict[str, float]:
        """评估模型"""
        from .evaluator import COCOEvaluator
        evaluator = COCOEvaluator(self.val_loader.dataset)
        metrics = evaluator.evaluate(self.model, self.val_loader, self.device)
        return metrics

    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        pass

    def load_checkpoint(self, path: str):
        """加载检查点"""
        pass

    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        self.callbacks[event] = callback
```

#### 5.2 评估器模块

**文件**: `src/tcm_tongue/engine/evaluator.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, List
import json
import tempfile

class COCOEvaluator:
    """COCO格式评估器"""

    def __init__(self, dataset: TongueCocoDataset):
        self.dataset = dataset
        self.coco_gt = self._build_coco_gt()

    def _build_coco_gt(self) -> COCO:
        """构建COCO格式的ground truth"""
        pass

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        评估模型性能

        Returns:
            metrics: {
                "mAP": float,           # mAP@0.5:0.95
                "mAP_50": float,        # mAP@0.5
                "mAP_75": float,        # mAP@0.75
                "mAP_small": float,     # 小目标mAP
                "mAP_medium": float,    # 中目标mAP
                "mAP_large": float,     # 大目标mAP
                "per_class_AP": Dict[str, float],  # 各类别AP
            }
        """
        model.eval()
        predictions = []

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            _, detections = model(images)

            for target, detection in zip(targets, detections):
                image_id = target["image_id"].item()
                boxes = detection["boxes"].cpu()
                scores = detection["scores"].cpu()
                labels = detection["labels"].cpu()

                # 转换为COCO格式
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                        "score": score.item(),
                    })

        # COCO评估
        coco_dt = self.coco_gt.loadRes(predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
        }

        # 计算各类别AP
        per_class_ap = self._compute_per_class_ap(coco_eval)
        metrics["per_class_AP"] = per_class_ap

        return metrics

    def _compute_per_class_ap(self, coco_eval: COCOeval) -> Dict[str, float]:
        """计算各类别的AP"""
        pass

class ClassImbalanceAnalyzer:
    """类别不平衡分析器"""

    def __init__(self, evaluator: COCOEvaluator):
        self.evaluator = evaluator

    def analyze_head_tail_performance(
        self,
        per_class_ap: Dict[str, float],
        class_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """
        分析头部/中部/尾部类别的性能差异

        Returns:
            {
                "head_mAP": float,    # 头部类别(>1000样本)平均AP
                "medium_mAP": float,  # 中部类别(100-1000样本)平均AP
                "tail_mAP": float,    # 尾部类别(<100样本)平均AP
                "head_tail_gap": float,  # 头尾差距
            }
        """
        pass
```

**单元测试**: `tests/test_trainer.py`
```python
class TestTrainer:

    def test_trainer_one_epoch(self):
        """测试训练一个epoch"""

    def test_trainer_checkpoint_save_load(self):
        """测试检查点保存和加载"""

    def test_trainer_amp(self):
        """测试混合精度训练"""

    def test_evaluator_coco_metrics(self):
        """测试COCO评估指标计算"""

    def test_evaluator_per_class_ap(self):
        """测试各类别AP计算"""
```

---

### 阶段 6: 推理接口与API (Day 11)

#### 6.1 推理接口

**文件**: `src/tcm_tongue/api/inference.py`

**接口设计**:
```python
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path

class TongueDetectorInference:
    """舌诊检测推理接口"""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
    ):
        """
        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
            device: 推理设备
            score_thresh: 置信度阈值
            nms_thresh: NMS阈值
        """
        self.device = device
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.config = Config.from_yaml(config_path) if config_path else Config()
        self.model = self._load_model(model_path)
        self.transform = ValTransform()
        self.category_names = self._load_category_names()

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        model = build_detector(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict:
        """
        单张图像预测

        Args:
            image: 图像路径、PIL Image或numpy数组

        Returns:
            {
                "boxes": List[List[float]],  # [[x1,y1,x2,y2], ...]
                "labels": List[str],         # 类别名称
                "label_ids": List[int],      # 类别ID
                "scores": List[float],       # 置信度
                "health_assessment": Dict,   # 健康评估结果
            }
        """
        # 预处理
        img_array = self._preprocess(image)
        img_tensor = self.transform(img_array)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 推理
        _, detections = self.model([img_tensor[0]])

        # 后处理
        result = self._postprocess(detections[0])

        # 健康评估
        result["health_assessment"] = self._assess_health(result)

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """批量预测"""
        pass

    def _preprocess(self, image) -> np.ndarray:
        """图像预处理"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image

    def _postprocess(self, detection: Dict) -> Dict:
        """后处理检测结果"""
        boxes = detection["boxes"].cpu().numpy().tolist()
        scores = detection["scores"].cpu().numpy().tolist()
        label_ids = detection["labels"].cpu().numpy().tolist()
        labels = [self.category_names.get(lid, f"unknown_{lid}") for lid in label_ids]

        # 过滤低置信度
        filtered = [
            (b, l, lid, s)
            for b, l, lid, s in zip(boxes, labels, label_ids, scores)
            if s >= self.score_thresh
        ]

        if filtered:
            boxes, labels, label_ids, scores = zip(*filtered)
        else:
            boxes, labels, label_ids, scores = [], [], [], []

        return {
            "boxes": list(boxes),
            "labels": list(labels),
            "label_ids": list(label_ids),
            "scores": list(scores),
        }

    def _assess_health(self, detection_result: Dict) -> Dict:
        """
        基于检测结果进行健康评估

        中医舌诊评估逻辑:
        - 舌质: 红舌(热证)、紫舌(血瘀)、淡舌(虚证)
        - 舌苔: 白苔(表证/寒证)、黄苔(里证/热证)、厚苔(湿浊)
        - 舌形: 齿痕(脾虚)、裂纹(阴虚)、胖大(水湿)
        - 脏腑区域: 凹陷(虚)、凸起(实)

        Returns:
            {
                "risk_level": str,      # low/medium/high
                "findings": List[str],  # 发现的舌象特征
                "suggestions": List[str],  # 健康建议
                "organ_status": Dict[str, str],  # 脏腑状态评估
            }
        """
        labels = detection_result["labels"]

        findings = []
        suggestions = []
        organ_status = {}
        risk_score = 0

        # 舌质分析
        if "hongshe" in labels:
            findings.append("红舌 - 可能存在热证")
            suggestions.append("建议清热降火，多饮水")
            risk_score += 1
        if "zishe" in labels:
            findings.append("紫舌 - 可能存在血瘀")
            suggestions.append("建议活血化瘀，适当运动")
            risk_score += 2

        # 舌苔分析
        if "huangtaishe" in labels:
            findings.append("黄苔 - 可能存在里热证")
            suggestions.append("建议清热解毒")
            risk_score += 1
        if "heitaishe" in labels:
            findings.append("黑苔 - 可能存在重症或服药影响")
            suggestions.append("建议及时就医检查")
            risk_score += 3

        # 舌形分析
        if "chihenshe" in labels:
            findings.append("齿痕舌 - 可能存在脾虚湿盛")
            suggestions.append("建议健脾祛湿")
            risk_score += 1
        if "liewenshe" in labels:
            findings.append("裂纹舌 - 可能存在阴虚")
            suggestions.append("建议滋阴润燥")
            risk_score += 1

        # 脏腑区域分析
        for label in labels:
            if "xinfeiao" in label:
                organ_status["心肺"] = "虚"
            if "gandanao" in label:
                organ_status["肝胆"] = "虚"
            if "piweiao" in label:
                organ_status["脾胃"] = "虚"
            if "shenquao" in label:
                organ_status["肾"] = "虚"

        # 健康舌判断
        if "jiankangshe" in labels and len(labels) == 1:
            findings.append("健康舌象")
            risk_score = 0

        # 风险等级
        if risk_score == 0:
            risk_level = "low"
        elif risk_score <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"

        if not findings:
            findings.append("未检测到明显异常舌象")
        if not suggestions:
            suggestions.append("保持良好生活习惯")

        return {
            "risk_level": risk_level,
            "findings": findings,
            "suggestions": suggestions,
            "organ_status": organ_status,
        }

    def visualize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        result: Dict,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """可视化检测结果"""
        pass
```

**单元测试**: `tests/test_inference.py`
```python
class TestInference:

    def test_predict_single_image(self):
        """测试单张图像预测"""

    def test_predict_batch(self):
        """测试批量预测"""

    def test_health_assessment(self):
        """测试健康评估逻辑"""

    def test_visualize_output(self):
        """测试可视化输出"""
```

---

### 阶段 7: 类别不平衡专项优化 (Day 12-15)

#### 7.1 实验设计

| 实验编号 | 实验名称 | 采样策略 | 损失函数 | 数据增强 |
|---------|---------|---------|---------|---------|
| EXP-01 | Baseline | default | CE | 基础增强 |
| EXP-02 | Focal Loss | default | Focal (γ=2) | 基础增强 |
| EXP-03 | Weighted CE | default | Weighted CE (inverse) | 基础增强 |
| EXP-04 | Weighted CE (sqrt) | default | Weighted CE (inverse_sqrt) | 基础增强 |
| EXP-05 | Oversample | oversample (2x) | CE | 基础增强 |
| EXP-06 | Oversample + Focal | oversample (2x) | Focal (γ=2) | 基础增强 |
| EXP-07 | Stratified | stratified | CE | 基础增强 |
| EXP-08 | Class-Balanced Focal | default | CB-Focal (β=0.9999) | 基础增强 |
| EXP-09 | Strong Augment | default | CE | 强增强 |
| EXP-10 | TCM Prior Augment | default | CE | 舌象先验增强 |
| EXP-11 | Combined Best | best_sampler | best_loss | 舌象先验增强 |

#### 7.2 实验配置文件

**configs/experiments/baseline.yaml**:
```yaml
experiment:
  name: "baseline"
  description: "基线实验：默认采样 + 交叉熵损失"

data:
  root: "datasets/shezhenv3-coco"
  batch_size: 8
  num_workers: 4

model:
  backbone: "resnet50"
  pretrained: true
  neck: "fpn"
  head: "faster_rcnn"
  num_classes: 21

train:
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  lr_scheduler: "cosine"
  warmup_epochs: 3

loss:
  type: "ce"

sampler:
  type: "default"

augmentation:
  type: "basic"
  horizontal_flip: true
  brightness_contrast: true
  hue_saturation: false
  mosaic: false
  mixup: false
```

**configs/experiments/focal_loss.yaml**:
```yaml
experiment:
  name: "focal_loss"
  description: "Focal Loss实验"

# 继承基线配置
_base_: "baseline.yaml"

loss:
  type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**configs/experiments/oversampling.yaml**:
```yaml
experiment:
  name: "oversampling"
  description: "过采样实验"

_base_: "baseline.yaml"

sampler:
  type: "oversample"
  oversample_factor: 2.0
```

#### 7.3 实验运行脚本

**scripts/run_experiments.py**:
```python
import subprocess
import yaml
from pathlib import Path
import datetime

EXPERIMENTS = [
    "baseline",
    "focal_loss",
    "weighted_ce",
    "weighted_ce_sqrt",
    "oversampling",
    "oversample_focal",
    "stratified",
    "class_balanced_focal",
    "strong_augment",
    "tcm_prior_augment",
]

def run_experiment(config_name: str):
    """运行单个实验"""
    config_path = f"configs/experiments/{config_name}.yaml"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/experiments/{config_name}_{timestamp}"

    cmd = [
        "python", "scripts/train.py",
        "--config", config_path,
        "--output-dir", output_dir,
    ]

    subprocess.run(cmd)

def run_all_experiments():
    """运行所有实验"""
    for exp_name in EXPERIMENTS:
        print(f"Running experiment: {exp_name}")
        run_experiment(exp_name)

if __name__ == "__main__":
    run_all_experiments()
```

#### 7.4 实验结果分析

**scripts/analyze_experiments.py**:
```python
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_experiment_results(exp_dir: Path) -> dict:
    """加载实验结果"""
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)

def generate_comparison_table(results: dict) -> pd.DataFrame:
    """生成实验对比表格"""
    rows = []
    for exp_name, metrics in results.items():
        rows.append({
            "实验": exp_name,
            "mAP": metrics["mAP"],
            "mAP@50": metrics["mAP_50"],
            "mAP@75": metrics["mAP_75"],
            "头部mAP": metrics.get("head_mAP", "-"),
            "中部mAP": metrics.get("medium_mAP", "-"),
            "尾部mAP": metrics.get("tail_mAP", "-"),
            "头尾差距": metrics.get("head_tail_gap", "-"),
        })
    return pd.DataFrame(rows)

def plot_class_ap_comparison(results: dict, output_path: str):
    """绘制各类别AP对比图"""
    pass

def generate_experiment_report(results: dict, output_path: str):
    """生成实验报告"""
    pass
```

---

### 阶段 8: 文档与收尾 (Day 16-17)

#### 8.1 文档结构

```
docs/
├── README.md                    # 项目概述
├── installation.md              # 安装指南
├── quickstart.md                # 快速开始
├── api_reference/
│   ├── dataset.md              # 数据集API
│   ├── models.md               # 模型API
│   ├── training.md             # 训练API
│   └── inference.md            # 推理API
├── experiments/
│   ├── class_imbalance.md      # 类别不平衡实验报告
│   └── ablation_study.md       # 消融实验报告
└── tcm_knowledge/
    └── tongue_diagnosis.md     # 舌诊知识背景
```

#### 8.2 API文档模板

**docs/api_reference/inference.md**:
```markdown
# 推理API

## TongueDetectorInference

舌诊检测推理接口，提供端到端的图像检测和健康评估功能。

### 初始化

\`\`\`python
from tcm_tongue.api import TongueDetectorInference

detector = TongueDetectorInference(
    model_path="runs/best_model.pth",
    config_path="configs/base.yaml",
    device="cuda",
    score_thresh=0.5,
    nms_thresh=0.5,
)
\`\`\`

### 单张图像预测

\`\`\`python
result = detector.predict("path/to/tongue_image.jpg")
print(result)
# {
#     "boxes": [[100, 100, 200, 200], ...],
#     "labels": ["baitaishe", "chihenshe", ...],
#     "scores": [0.95, 0.87, ...],
#     "health_assessment": {
#         "risk_level": "medium",
#         "findings": ["白苔 - 可能存在表证", "齿痕舌 - 可能脾虚"],
#         "suggestions": ["建议解表散寒", "建议健脾祛湿"],
#     }
# }
\`\`\`

### 批量预测

\`\`\`python
results = detector.predict_batch(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    batch_size=8
)
\`\`\`

### 可视化

\`\`\`python
detector.visualize(
    "path/to/image.jpg",
    result,
    output_path="output/visualized.jpg",
    show=True
)
\`\`\`
```

#### 8.3 codex.MD 模板

**codex.MD**:
```markdown
# 问题记录与解决方案

## 格式说明

每个问题条目包含：
- 问题描述
- 根因分析
- 解决方案
- 参考链接

---

## [P001] 示例问题标题

**日期**: 2026-02-04
**状态**: 已解决 / 进行中 / 待处理
**严重程度**: 高 / 中 / 低

### 问题描述
描述具体问题现象...

### 根因分析
分析问题产生的原因...

### 解决方案
\`\`\`python
# 代码修复示例
\`\`\`

### 参考链接
- [相关Issue](https://...)

---
```

---

## 里程碑与验收标准

### M1: 基础框架完成 (Day 1-6)

**验收标准**:
- [ ] 项目结构完整，可 `pip install -e .` 安装
- [ ] 数据集模块通过所有单元测试
- [ ] 数据增强模块通过所有单元测试
- [ ] 采样器模块通过所有单元测试
- [ ] 骨干网络模块通过所有单元测试
- [ ] 检测器模块可前向传播

**验证命令**:
```bash
conda run -n cv pytest tests/ -v --cov=src/tcm_tongue --cov-report=term-missing
```

### M2: 训练流程完成 (Day 7-11)

**验收标准**:
- [x] 损失函数模块通过所有单元测试
- [x] 训练器可完成完整训练流程
- [x] 评估器可计算COCO指标
- [x] 推理接口可预测并返回健康评估
- [ ] ~~Baseline实验完成，mAP > 30%~~ (移至 M2.5)

**验证命令**:
```bash
conda run -n cv pytest tests/ -v --cov=src/tcm_tongue --cov-report=term-missing
```

### M2.5: Baseline 收敛修复 (紧急插入)

**背景**: 2026-02-06 发现 baseline 实验 mAP 仅 0.12%，模型未收敛。

**根因**:
1. `head.py` 未使用预训练权重 (`weights=None`)
2. `num_classes=21` 未包含背景类 (应为 22)
3. 学习率 SGD lr=0.001 过高

**任务清单**:
- [ ] 修改 `src/tcm_tongue/models/head.py`，使用 COCO 预训练权重
- [ ] 替换分类头以适配自定义类别数
- [ ] 创建 `scripts/debug_baseline.py` 快速验证脚本
- [ ] 验证收敛：少量样本 (200张)，5 epochs，loss 应下降
- [ ] 完整训练验证：mAP > 30%

**代码修改要点** (`head.py`):
```python
# 修改前
self.model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=num_classes,
)

# 修改后
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
in_features = self.model.roi_heads.box_predictor.cls_score.in_features
self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

**快速验证参数**:
| 参数 | 值 |
|------|-----|
| 训练样本 | 200 |
| 验证样本 | 50 |
| Epochs | 5 |
| Batch size | 4 |
| 学习率 | 0.0001 (AdamW) |
| 梯度裁剪 | 1.0 |

**验证命令**:
```bash
conda run -n cv python scripts/debug_baseline.py --num-samples 200 --epochs 5
```

**验收标准**:
- [ ] 快速验证: loss 明显下降，检测数量增加
- [ ] 完整训练: mAP > 30%

### M3: 优化实验完成 (Day 12-15)

**验收标准**:
- [ ] 所有11个实验配置完成
- [ ] 实验结果对比表格生成
- [ ] 最佳策略组合确定
- [ ] 尾部类别AP显著提升（相比baseline）

**验证命令**:
```bash
conda run -n cv python scripts/run_experiments.py
conda run -n cv python scripts/analyze_experiments.py --output-dir runs/experiments
```

### M4: 文档与发布 (Day 16-17)

**验收标准**:
- [ ] 所有API文档完成
- [ ] 实验报告完成
- [ ] codex.MD 记录所有遇到的问题
- [ ] claude.MD 更新完整开发日志
- [ ] README.md 包含完整使用说明

---

## 关键技术决策

### D1: 检测框架选择

**决策**: 使用 torchvision 内置检测模型作为基础

**理由**:
1. 官方维护，稳定可靠
2. 预训练模型开箱即用
3. 与PyTorch生态兼容性好
4. 代码可读性高，便于定制

**备选方案**: MMDetection, Detectron2

### D2: 骨干网络选择

**决策**: 默认使用 ResNet-50，可选 Swin Transformer

**理由**:
1. ResNet-50 平衡了性能和效率
2. ImageNet预训练权重可用
3. Swin Transformer 可用于追求更高精度

### D3: 类别不平衡处理策略

**主要策略**:
1. **数据层**: 过采样尾部类别、基于舌象先验的数据增强
2. **损失层**: Focal Loss、类别平衡Focal Loss
3. **评估层**: 关注尾部类别指标、头尾差距分析

---

## 风险与缓解措施

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|-----|---------|
| 尾部类别样本过少导致模型无法学习 | 高 | 高 | 数据增强、过采样、Few-shot学习 |
| 类别间视觉特征相似导致混淆 | 中 | 中 | 类别层级结构、多任务学习 |
| 训练不稳定 | 低 | 中 | 梯度裁剪、学习率warmup |
| 推理速度不满足需求 | 低 | 低 | 模型量化、TensorRT优化 |

---

## 附录

### A1: 类别中英文对照表

| 拼音 | 中文 | 英文 | 样本数 |
|-----|------|------|-------|
| baitaishe | 白苔舌 | White coating tongue | 5040 |
| hongdianshe | 红点舌 | Red dot tongue | 3653 |
| liewenshe | 裂纹舌 | Cracked tongue | 1886 |
| chihenshe | 齿痕舌 | Teeth-marked tongue | 1482 |
| hongshe | 红舌 | Red tongue | 1466 |
| huangtaishe | 黄苔舌 | Yellow coating tongue | 1119 |
| pangdashe | 胖大舌 | Enlarged tongue | 689 |
| botaishe | 薄苔舌 | Thin coating tongue | 581 |
| shenquao | 肾区凹 | Kidney area depression | 495 |
| xinfeiao | 心肺凹 | Heart-lung area depression | 372 |
| gandanao | 肝胆凹 | Liver-gallbladder area depression | 292 |
| shoushe | 瘦舌 | Thin tongue | 285 |
| huataishe | 滑苔舌 | Slippery coating tongue | 283 |
| zishe | 紫舌 | Purple tongue | 231 |
| piweiao | 脾胃凹 | Spleen-stomach area depression | 186 |
| heitaishe | 黑苔舌 | Black coating tongue | 122 |
| jiankangshe | 健康舌 | Healthy tongue | 21 |
| gandantu | 肝胆凸 | Liver-gallbladder area protrusion | 9 |
| shenqutu | 肾区凸 | Kidney area protrusion | 2 |
| xinfeitu | 心肺凸 | Heart-lung area protrusion | 2 |

### A2: 舌诊脏腑分区图

```
        ┌─────────────────┐
        │    心肺区       │  ← 舌尖
        │  (xinfeiao/tu)  │
        ├─────────────────┤
        │    脾胃区       │  ← 舌中
        │  (piweiao)      │
        ├────────┬────────┤
        │ 肝胆区 │ 肝胆区 │  ← 舌边
        │(gandn) │(gandn) │
        ├────────┴────────┤
        │    肾区         │  ← 舌根
        │  (shenquao/tu)  │
        └─────────────────┘
```

---

## 执行说明

1. 按阶段顺序执行任务
2. 每个模块开发前先编写单元测试 (TDD)
3. 遇到问题及时更新 codex.MD
4. 每日更新 claude.MD 开发日志
5. 使用 `conda activate cv` 环境运行所有命令
6. 所有实验使用相同的随机种子 (42) 保证可复现性
