import torch

from src.attention_unet.losses import (
    HybridBCEDiceLoss,
)

from src.attention_unet.metrics import (
    segmentation_metrics,
)


def test_perfect_prediction():
    prediction = torch.tensor(
        [[[[1.0, 0.0],
           [1.0, 0.0]]]]
    )

    target = torch.tensor(
        [[[[1.0, 0.0],
           [1.0, 0.0]]]]
    )

    metrics = segmentation_metrics(
        prediction,
        target,
    )

    assert abs(
        metrics["iou"] - 1.0
    ) < 1e-6

    assert abs(
        metrics["dice"] - 1.0
    ) < 1e-6

    assert abs(
        metrics["precision"] - 1.0
    ) < 1e-6

    assert abs(
        metrics["recall"] - 1.0
    ) < 1e-6


def test_invalid_negative_pixels_ignored():
    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5
    )

    target = torch.tensor(
        [[[[1.0, 0.0],
           [-1.0, 1.0]]]]
    )

    prediction_a = torch.tensor(
        [[[[0.9, 0.1],
           [0.0, 0.9]]]]
    )

    prediction_b = torch.tensor(
        [[[[0.9, 0.1],
           [1.0, 0.9]]]]
    )

    loss_a = loss_fn(
        prediction_a,
        target,
    )

    loss_b = loss_fn(
        prediction_b,
        target,
    )

    # Changing prediction at an invalid pixel
    # must not affect the loss.
    assert torch.allclose(
        loss_a,
        loss_b,
        atol=1e-7,
    )


def test_loss_backward():
    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5
    )

    raw = torch.randn(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )

    prediction = torch.sigmoid(raw)

    target = torch.randint(
        0,
        2,
        (2, 1, 8, 8),
    ).float()

    loss = loss_fn(
        prediction,
        target,
    )

    loss.backward()

    assert torch.isfinite(loss)