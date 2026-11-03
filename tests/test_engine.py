import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.attention_unet.engine import (
    prepare_batch,
    run_epoch,
)

from src.attention_unet.losses import (
    HybridBCEDiceLoss,
)

from src.attention_unet.model import (
    AttentionUNet,
)


def test_prepare_augmented_batch():
    x = torch.randn(
        2, 40, 32, 32
    )

    e = torch.randn(
        2, 64, 32, 32
    )

    y = torch.randint(
        0,
        2,
        (2, 32, 32),
    ).float()

    inputs, target, mask = (
        prepare_batch(
            (x, e, y),
            use_aef=True,
        )
    )

    assert inputs.shape == (
        2,
        104,
        32,
        32,
    )

    assert target.shape == (
        2,
        32,
        32,
    )

    assert mask is None


def test_prepare_baseline_from_same_batch():
    x = torch.randn(
        2, 40, 32, 32
    )

    e = torch.randn(
        2, 64, 32, 32
    )

    y = torch.randint(
        0,
        2,
        (2, 32, 32),
    ).float()

    inputs, _, _ = prepare_batch(
        (x, e, y),
        use_aef=False,
    )

    assert inputs.shape == (
        2,
        40,
        32,
        32,
    )


def test_augmented_training_epoch():
    x = torch.randn(
        4, 40, 32, 32
    )

    e = torch.randn(
        4, 64, 32, 32
    )

    y = torch.randint(
        0,
        2,
        (4, 32, 32),
    ).float()

    loader = DataLoader(
        TensorDataset(x, e, y),
        batch_size=2,
    )

    model = AttentionUNet(
        n_in_channels=104,
        base_channels=4,
    )

    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    result = run_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        use_aef=True,
        threshold=0.5,
        optimizer=optimizer,
    )

    assert result.loss >= 0.0
    assert 0.0 <= result.iou <= 1.0
    assert 0.0 <= result.dice <= 1.0


def test_validation_epoch():
    x = torch.randn(
        2, 40, 32, 32
    )

    y = torch.randint(
        0,
        2,
        (2, 32, 32),
    ).float()

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=1,
    )

    model = AttentionUNet(
        n_in_channels=40,
        base_channels=4,
    )

    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5
    )

    result = run_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        use_aef=False,
        threshold=0.5,
        optimizer=None,
    )

    assert result.loss >= 0.0
    assert 0.0 <= result.iou <= 1.0