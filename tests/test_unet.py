import torch

from src.attention_unet.model import (
    AttentionGate,
    AttentionUNet,
    SEBlock,
)


def test_attention_unet_output_shape_64():
    # Embedding-augmented configuration:
    # 40 WSTS + 64 AEF channels.
    model = AttentionUNet(
        n_in_channels=104,
        base_channels=8,
        use_se=True,
    )

    x = torch.randn(
        2,
        104,
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
    # WSTS-only baseline and original architecture
    # without SE channel attention.
    model = AttentionUNet(
        n_in_channels=40,
        base_channels=8,
        use_se=False,
        use_spatial_attention=False,
    )

    assert all(
        isinstance(decoder.attention, torch.nn.Identity)
        for decoder in (
            model.dec1,
            model.dec2,
            model.dec3,
            model.dec4,
        )
    )

    x = torch.randn(
        1,
        40,
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


def test_attention_unet_spatial_attention_enabled_by_default():
    model = AttentionUNet(
        n_in_channels=5,
        base_channels=8,
    )

    assert all(
        isinstance(decoder.attention, AttentionGate)
        for decoder in (
            model.dec1,
            model.dec2,
            model.dec3,
            model.dec4,
        )
    )


def test_attention_unet_backward_pass():
    model = AttentionUNet(
        n_in_channels=5,
        base_channels=8,
        use_se=True,
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


def test_se_block_preserves_shape_and_backward():
    # reduction > channels intentionally tests the
    # small-width edge case used in lightweight tests.
    block = SEBlock(
        channels=8,
        reduction=16,
    )

    x = torch.randn(
        2,
        8,
        16,
        16,
        requires_grad=True,
    )

    y = block(x)

    assert y.shape == x.shape

    y.mean().backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()