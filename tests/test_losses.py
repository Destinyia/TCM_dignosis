import torch

from tcm_tongue.losses import FocalLoss, WeightedCrossEntropyLoss


def test_focal_loss_matches_ce_when_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))

    ce = torch.nn.functional.cross_entropy(logits, targets)
    focal = FocalLoss(alpha=None, gamma=0.0)(logits, targets)

    assert torch.allclose(ce, focal, atol=1e-6)


def test_focal_loss_reduction_none():
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss = FocalLoss(alpha=0.5, gamma=2.0, reduction="none")(logits, targets)
    assert loss.shape == targets.shape


def test_focal_loss_ignore_index():
    logits = torch.randn(3, 4)
    targets = torch.tensor([1, -1, 2])
    loss = FocalLoss(alpha=None, gamma=2.0, ignore_index=-1)(logits, targets)
    assert torch.isfinite(loss)


def test_weighted_ce_biases_classes():
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]])
    targets = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 3.0])

    loss_unweighted = WeightedCrossEntropyLoss()(logits, targets)
    loss_weighted = WeightedCrossEntropyLoss(class_weights=weights)(logits, targets)

    assert loss_weighted > loss_unweighted
