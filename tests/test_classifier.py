"""分类模型单元测试"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Tongue_segment"))


class TestClassifierModels:
    """测试分类模型"""

    def test_baseline_classifier(self):
        """测试基线分类器"""
        from tcm_tongue.models.classifier import BaselineClassifier

        model = BaselineClassifier(num_classes=8, backbone="resnet50", pretrained=False)
        x = torch.randn(2, 3, 640, 640)
        out = model(x)

        assert out.shape == (2, 8)

    def test_seg_attention_classifier_no_weights(self):
        """测试分割注意力分类器（无预训练权重）"""
        from tcm_tongue.models.classifier import SegAttentionClassifier

        model = SegAttentionClassifier(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
            freeze_seg=False,
        )
        x = torch.randn(2, 3, 640, 640)
        out = model(x)

        assert out.shape == (2, 8)

    def test_seg_attention_classifier_with_attention(self):
        """测试分割注意力分类器返回注意力图"""
        from tcm_tongue.models.classifier import SegAttentionClassifier

        model = SegAttentionClassifier(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
        )
        x = torch.randn(2, 3, 640, 640)
        out, attention = model(x, return_attention=True)

        assert out.shape == (2, 8)
        assert attention.shape[0] == 2
        assert attention.shape[1] == 1

    def test_seg_attention_classifier_with_mask_refiner(self):
        """测试分割注意力分类器带掩码精炼层"""
        from tcm_tongue.models.classifier import SegAttentionClassifier

        model = SegAttentionClassifier(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
            freeze_seg=False,
            use_mask_refiner=True,
        )
        x = torch.randn(2, 3, 224, 224)
        out, mask = model(x, return_mask=True)

        assert out.shape == (2, 8)
        assert mask.shape[0] == 2
        assert mask.shape[1] == 1
        # 验证mask_refiner存在
        assert model.mask_refiner is not None

    def test_dual_stream_classifier(self):
        """测试双流融合分类器"""
        from tcm_tongue.models.classifier import DualStreamClassifier

        model = DualStreamClassifier(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
            freeze_seg=False,
            fusion_type="concat",
        )
        x = torch.randn(2, 3, 640, 640)
        out = model(x)

        assert out.shape == (2, 8)

    def test_build_classifier_factory(self):
        """测试模型工厂函数"""
        from tcm_tongue.models.classifier import build_classifier

        for model_type in ["baseline", "seg_attention", "seg_attention_v2",
                           "seg_attention_multiscale", "dual_stream"]:
            model = build_classifier(
                model_type=model_type,
                num_classes=8,
                backbone="resnet50",
                pretrained=False,
                seg_weights_path=None,
                soft_floor=0.1,
            )
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
            assert out.shape == (1, 8), f"Failed for {model_type}"


class TestSegAttentionModules:
    """测试分割注意力模块"""

    def test_spatial_attention(self):
        """测试空间注意力模块"""
        from tcm_tongue.models.seg_attention import SpatialAttention

        module = SpatialAttention(refine=True)
        mask = torch.rand(2, 1, 64, 64)
        out = module(mask, target_size=(32, 32))

        assert out.shape == (2, 1, 32, 32)
        assert out.min() >= 0 and out.max() <= 1

    def test_mask_guided_attention(self):
        """测试掩码引导注意力模块"""
        from tcm_tongue.models.seg_attention import MaskGuidedAttention

        module = MaskGuidedAttention(in_channels=256, use_channel_attention=True)
        features = torch.randn(2, 256, 32, 32)
        mask = torch.rand(2, 1, 64, 64)
        out = module(features, mask)

        assert out.shape == features.shape

    def test_residual_soft_attention(self):
        """测试残差软注意力模块"""
        from tcm_tongue.models.seg_attention import ResidualSoftAttention

        module = ResidualSoftAttention(
            in_channels=2048,
            soft_floor=0.1,
            use_channel_attention=True,
        )
        features = torch.randn(2, 2048, 20, 20)
        mask = torch.rand(2, 1, 640, 640)
        out = module(features, mask)

        assert out.shape == features.shape
        # 验证 alpha 参数存在且可学习
        assert hasattr(module, 'alpha')
        assert module.alpha.requires_grad

    def test_multi_scale_attention(self):
        """测试多尺度注意力模块"""
        from tcm_tongue.models.seg_attention import MultiScaleAttention

        feature_dims = [256, 512, 1024, 2048]
        module = MultiScaleAttention(feature_dims=feature_dims, soft_floor=0.1)
        mask = torch.rand(2, 1, 640, 640)

        # 测试每个层级
        for i, dim in enumerate(feature_dims):
            h = 160 // (2 ** i)
            features = torch.randn(2, dim, h, h)
            out = module.forward_single(features, mask, level=i)
            assert out.shape == features.shape, f"Failed at level {i}"

    def test_seg_attention_classifier_v2(self):
        """测试改进版分割注意力分类器"""
        from tcm_tongue.models.classifier import SegAttentionClassifierV2

        model = SegAttentionClassifierV2(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
            soft_floor=0.1,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 8)

        # 测试返回注意力
        out, mask = model(x, return_attention=True)
        assert out.shape == (2, 8)
        assert mask.shape[0] == 2

    def test_multiscale_seg_attention_classifier(self):
        """测试多尺度分割注意力分类器"""
        from tcm_tongue.models.classifier import MultiScaleSegAttentionClassifier

        model = MultiScaleSegAttentionClassifier(
            num_classes=8,
            backbone="resnet50",
            pretrained=False,
            seg_weights_path=None,
            soft_floor=0.1,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 8)


class TestLossFunctions:
    """测试损失函数"""

    def test_focal_loss(self):
        """测试Focal Loss"""
        from tcm_tongue.losses import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4,))
        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_class_balanced_focal_loss(self):
        """测试类别平衡Focal Loss"""
        from tcm_tongue.losses import ClassBalancedFocalLoss

        class_counts = [100, 50, 30, 20, 15, 10, 8, 5]
        criterion = ClassBalancedFocalLoss(class_counts=class_counts, gamma=2.0)
        logits = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4,))
        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_seesaw_loss(self):
        """测试Seesaw Loss"""
        from tcm_tongue.losses import SeesawLoss

        criterion = SeesawLoss(num_classes=8)
        logits = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4,))
        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_mask_bbox_loss(self):
        """测试MaskBBoxLoss"""
        from tcm_tongue.losses import MaskBBoxLoss

        criterion = MaskBBoxLoss(
            iou_weight=1.0,
            coverage_weight=1.0,
            boundary_weight=0.5,
            use_distance_penalty=True,
            distance_scale=5.0,
        )
        mask = torch.rand(2, 1, 64, 64)
        bbox = torch.tensor([
            [0.1, 0.1, 0.9, 0.9],
            [0.2, 0.2, 0.8, 0.8],
        ])
        loss = criterion(mask, bbox)

        assert loss.ndim == 0
        assert loss >= 0

    def test_mask_bbox_loss_metrics(self):
        """测试MaskBBoxLoss指标获取"""
        from tcm_tongue.losses import MaskBBoxLoss

        criterion = MaskBBoxLoss()
        mask = torch.rand(2, 1, 64, 64)
        bbox = torch.tensor([
            [0.1, 0.1, 0.9, 0.9],
            [0.2, 0.2, 0.8, 0.8],
        ])
        metrics = criterion.get_metrics(mask, bbox)

        assert "mask_iou_loss" in metrics
        assert "mask_coverage_loss" in metrics
        assert "mask_boundary_loss" in metrics
        assert "mask_iou" in metrics
        assert "mask_coverage" in metrics


class TestClassificationTransform:
    """测试数据增强"""

    def test_train_transform(self):
        """测试训练时数据增强"""
        import numpy as np
        from tcm_tongue.data.classification_dataset import ClassificationTransform

        transform = ClassificationTransform(
            image_size=(640, 640),
            is_train=True,
            strong_aug=False,
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = transform(image=image)

        assert "image" in result
        assert result["image"].shape == (3, 640, 640)

    def test_val_transform(self):
        """测试验证时数据增强"""
        import numpy as np
        from tcm_tongue.data.classification_dataset import ClassificationTransform

        transform = ClassificationTransform(
            image_size=(640, 640),
            is_train=False,
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = transform(image=image)

        assert "image" in result
        assert result["image"].shape == (3, 640, 640)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
