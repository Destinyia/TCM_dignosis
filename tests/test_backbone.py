import torch
import pytest

from tcm_tongue.models import ResNetBackbone, create_backbone


class TestBackbone:

    @pytest.mark.parametrize("backbone_name", ["resnet50", "resnet101", "swin_t"])
    def test_backbone_output_channels(self, backbone_name):
        backbone = create_backbone(backbone_name, pretrained=False)
        assert len(backbone.out_channels) == 4

    def test_backbone_output_shapes(self):
        backbone = create_backbone("resnet50", pretrained=False)
        x = torch.randn(2, 3, 800, 800)
        features = backbone(x)
        assert features["feat0"].shape == (2, 256, 200, 200)
        assert features["feat1"].shape == (2, 512, 100, 100)
        assert features["feat2"].shape == (2, 1024, 50, 50)
        assert features["feat3"].shape == (2, 2048, 25, 25)

    def test_pretrained_weights_loaded(self):
        backbone = create_backbone("resnet50", pretrained=True)
        x = torch.randn(1, 3, 224, 224)
        _ = backbone(x)

    def test_frozen_stages(self):
        backbone = ResNetBackbone(depth=50, pretrained=False, frozen_stages=2)
        assert not any(p.requires_grad for p in backbone.conv1.parameters())
        assert not any(p.requires_grad for p in backbone.bn1.parameters())
        assert not any(p.requires_grad for p in backbone.layer1.parameters())
        assert any(p.requires_grad for p in backbone.layer2.parameters())
