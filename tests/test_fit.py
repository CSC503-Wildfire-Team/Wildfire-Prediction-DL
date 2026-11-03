import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.attention_unet.fit import fit
from src.attention_unet.losses import (
    HybridBCEDiceLoss,
)
from src.attention_unet.model import (
    AttentionUNet,
)


def test_fit_saves_best_checkpoint(tmp_path):
    torch.manual_seed(0)

    x = torch.randn(
        2,
        40,
        16,
        16,
    )

    y = torch.randint(
        0,
        2,
        (2, 16, 16),
    ).float()

    train_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    val_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    model = AttentionUNet(
        n_in_channels=40,
        base_channels=2,
    )

    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    checkpoint = (
        tmp_path / "best_model.pt"
    )

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        use_aef=False,
        threshold=0.5,
        checkpoint_path=checkpoint,
    )

    assert len(result.history) == 2

    assert result.best_epoch in (
        1,
        2,
    )

    assert (
        0.0
        <= result.best_val_iou
        <= 1.0
    )

    assert checkpoint.exists()

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert (
        saved["epoch"]
        == result.best_epoch
    )

    assert (
        saved["val_iou"]
        == result.best_val_iou
    )

    assert "model_state_dict" in saved
    assert "optimizer_state_dict" in saved


def test_fit_without_checkpoint():
    x = torch.randn(
        2,
        40,
        16,
        16,
    )

    y = torch.randint(
        0,
        2,
        (2, 16, 16),
    ).float()

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    model = AttentionUNet(
        n_in_channels=40,
        base_channels=2,
    )

    loss_fn = HybridBCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    result = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        use_aef=False,
        threshold=0.5,
    )

    assert len(result.history) == 1
    assert result.checkpoint_path is None