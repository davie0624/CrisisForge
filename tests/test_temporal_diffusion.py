from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
diffusion_module = pytest.importorskip("crisisforge.diffusion")
ConditionalTemporalDDPM = diffusion_module.ConditionalTemporalDDPM


def _model() -> ConditionalTemporalDDPM:
    torch.manual_seed(101)
    return ConditionalTemporalDDPM(
        horizon=6,
        factor_dim=2,
        context_dim=3,
        regime_dim=2,
        num_diffusion_steps=4,
        hidden_channels=8,
        time_embedding_dim=8,
        num_residual_blocks=2,
    )


def _conditioning(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(9)
    context = torch.randn(batch_size, 5, 3, generator=generator)
    regimes = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])[:batch_size]
    return context, regimes


def test_sample_generates_joint_one_shot_path_shape() -> None:
    model = _model()
    context, regimes = _conditioning()
    paths = model.sample(context, regimes, seed=44)
    assert paths.shape == (3, 6, 2)
    assert torch.isfinite(paths).all()
    assert "future_context" not in inspect.signature(model.sample).parameters
    assert "future_context" not in inspect.signature(model.training_loss).parameters


def test_conditioning_changes_path_under_paired_noise() -> None:
    model = _model()
    context = torch.zeros(1, 5, 3)
    initial_noise = torch.randn(1, 6, 2, generator=torch.Generator().manual_seed(7))
    first = model.sample(
        context,
        torch.tensor([[1.0, 0.0]]),
        seed=55,
        initial_noise=initial_noise,
    )
    second = model.sample(
        context,
        torch.tensor([[0.0, 1.0]]),
        seed=55,
        initial_noise=initial_noise,
    )
    assert not torch.allclose(first, second)


def test_sampling_is_deterministic_and_restores_training_mode() -> None:
    model = _model()
    model.train()
    context, regimes = _conditioning(batch_size=2)
    first = model.sample(context, regimes, seed=87)
    assert model.training
    second = model.sample(context, regimes, seed=87)
    assert model.training
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    model.eval()
    _ = model.sample(context, regimes, seed=2)
    assert not model.training


def test_training_loss_has_finite_nonzero_gradients() -> None:
    model = _model()
    context, regimes = _conditioning()
    clean = torch.randn(3, 6, 2, generator=torch.Generator().manual_seed(17))
    noise = torch.randn(3, 6, 2, generator=torch.Generator().manual_seed(18))
    timesteps = torch.tensor([0, 1, 3], dtype=torch.long)
    weights = torch.tensor([1.0, 2.0, 4.0])
    loss = model.training_loss(
        clean,
        context,
        regimes,
        sample_weights=weights,
        timesteps=timesteps,
        noise=noise,
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_shape_checks_and_future_context_rejection() -> None:
    model = _model()
    context, regimes = _conditioning()
    clean = torch.zeros(3, 5, 2)
    with pytest.raises(ValueError, match="clean_paths"):
        model.training_loss(clean, context, regimes)
    with pytest.raises(TypeError):
        model.sample(
            context,
            regimes,
            future_context=torch.zeros(3, 6, 1),
        )
