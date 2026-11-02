import torch

from src.attention_unet.model import AttentionUNet


def test_attention_unet_output_shape_64():
    # Smaller base_channels makes the test fast.
    model = AttentionUNet(
        n_in_channels=12,
        base_channels=8,
    )

    x = torch.randn(
        2,
        12,
        64,
        64,
    )

    y = model(x)

    assert y.shape == (
        2,
        1,
        64,
        64,
    )

    assert torch.all(
        (y >= 0.0) & (y <= 1.0)
    )


def test_attention_unet_output_shape_128():
    model = AttentionUNet(
        n_in_channels=20,
        base_channels=8,
    )

    x = torch.randn(
        1,
        20,
        128,
        128,
    )

    y = model(x)

    assert y.shape == (
        1,
        1,
        128,
        128,
    )


def test_attention_unet_backward_pass():
    model = AttentionUNet(
        n_in_channels=5,
        base_channels=8,
    )

    x = torch.randn(
        2,
        5,
        64,
        64,
    )

    output = model(x)

    loss = output.mean()

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )