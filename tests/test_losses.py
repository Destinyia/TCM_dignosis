import torch

from tcm_tongue.losses import FocalLoss, WeightedCrossEntropyLoss, ClassBalancedFocalLoss, SeesawLoss


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


def test_class_balanced_focal_loss_basic():
    """Test ClassBalancedFocalLoss with basic inputs."""
    torch.manual_seed(42)
    num_classes = 5
    class_counts = [1000, 500, 100, 50, 10]  # Imbalanced distribution

    logits = torch.randn(8, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (8,))

    loss_fn = ClassBalancedFocalLoss(class_counts=class_counts, beta=0.9999, gamma=2.0)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_class_balanced_focal_loss_weights():
    """Test that CB Focal Loss gives higher weights to rare classes."""
    class_counts = [1000, 10]  # Very imbalanced
    loss_fn = ClassBalancedFocalLoss(class_counts=class_counts, beta=0.9999, gamma=0.0)

    # Weight for rare class should be higher
    weights = loss_fn.class_weights
    assert weights[1] > weights[0]


def test_class_balanced_focal_loss_reduction_none():
    """Test CB Focal Loss with reduction='none'."""
    class_counts = [100, 50, 25]
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])

    loss_fn = ClassBalancedFocalLoss(class_counts=class_counts, reduction="none")
    loss = loss_fn(logits, targets)

    assert loss.shape == targets.shape


def test_seesaw_loss_basic():
    """Test SeesawLoss with basic inputs."""
    torch.manual_seed(42)
    num_classes = 5

    logits = torch.randn(8, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (8,))

    loss_fn = SeesawLoss(num_classes=num_classes, p=0.8, q=2.0)
    loss_fn.train()
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_seesaw_loss_updates_counts():
    """Test that SeesawLoss updates cumulative counts during training."""
    num_classes = 3
    loss_fn = SeesawLoss(num_classes=num_classes)
    loss_fn.train()

    # Initial counts should be zero
    assert loss_fn.cum_counts.sum() == 0

    logits = torch.randn(4, num_classes)
    targets = torch.tensor([0, 0, 1, 2])
    loss_fn(logits, targets)

    # Counts should be updated
    assert loss_fn.cum_counts[0] == 2
    assert loss_fn.cum_counts[1] == 1
    assert loss_fn.cum_counts[2] == 1


def test_seesaw_loss_reduction_none():
    """Test SeesawLoss with reduction='none'."""
    num_classes = 4
    logits = torch.randn(6, num_classes)
    targets = torch.randint(0, num_classes, (6,))

    loss_fn = SeesawLoss(num_classes=num_classes, reduction="none")
    loss = loss_fn(logits, targets)

    assert loss.shape == targets.shape


def test_seesaw_loss_ignore_index():
    """Test SeesawLoss with ignore_index."""
    num_classes = 3
    logits = torch.randn(4, num_classes)
    targets = torch.tensor([0, -1, 1, 2])

    loss_fn = SeesawLoss(num_classes=num_classes, ignore_index=-1)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
