"""One-shot conditional diffusion for complete future factor paths.

The public interface intentionally accepts only information available when a
forecast is issued: a past-context tensor and current filtered regime
probabilities. There is no future-covariate argument. If genuinely known future
controls are added later, they require a separate, explicitly audited interface.

This module implements a compact DDPM research baseline. It is not an
autoregressive forecaster: every denoising call receives and updates the entire
``(horizon, factor_dim)`` tensor at once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Choose the largest small GroupNorm divisor."""

    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _linear_beta_schedule(
    num_steps: int,
    *,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_diffusion_steps must be at least two")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


class SinusoidalDiffusionEmbedding(nn.Module):
    """Deterministic sinusoidal embedding for integer diffusion time."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("embedding_dim must be even and at least four")
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (batch,)")
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * exponent
        )
        angles = timesteps.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class PastContextEncoder(nn.Module):
    """Encode observed history without accessing a forecast-horizon tensor."""

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.gru = nn.GRU(
            input_size=context_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, past_context: torch.Tensor) -> torch.Tensor:
        if past_context.ndim != 3:
            raise ValueError("past_context must have shape (batch, history, context_dim)")
        if past_context.shape[1] < 1:
            raise ValueError("past_context must contain at least one historical row")
        if past_context.shape[2] != self.context_dim:
            raise ValueError("past_context has an unexpected feature dimension")
        _, final_hidden = self.gru(past_context)
        return final_hidden[-1]


class ConditionedTemporalResidualBlock(nn.Module):
    """Dilated temporal block with FiLM conditioning."""

    def __init__(self, channels: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be positive")
        groups = _group_count(channels)
        self.normalization_1 = nn.GroupNorm(groups, channels)
        self.convolution_1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.normalization_2 = nn.GroupNorm(groups, channels)
        self.condition_projection = nn.Linear(condition_dim, 2 * channels)
        self.convolution_2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.convolution_1(F.silu(self.normalization_1(values)))
        scale, shift = self.condition_projection(condition).chunk(2, dim=1)
        hidden = self.normalization_2(hidden)
        hidden = hidden * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden = self.convolution_2(F.silu(hidden))
        return (values + hidden) / math.sqrt(2.0)


class ConditionalTemporalDenoiser(nn.Module):
    """Predict joint path noise from past context and soft regime probabilities."""

    def __init__(
        self,
        *,
        factor_dim: int,
        context_dim: int,
        regime_dim: int,
        hidden_channels: int = 64,
        time_embedding_dim: int = 64,
        num_residual_blocks: int = 4,
    ) -> None:
        super().__init__()
        if factor_dim < 1 or regime_dim < 1:
            raise ValueError("factor_dim and regime_dim must be positive")
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least four")
        if num_residual_blocks < 1:
            raise ValueError("num_residual_blocks must be positive")

        self.factor_dim = int(factor_dim)
        self.context_dim = int(context_dim)
        self.regime_dim = int(regime_dim)
        self.input_projection = nn.Conv1d(
            factor_dim,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.time_embedding = SinusoidalDiffusionEmbedding(time_embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.context_encoder = PastContextEncoder(context_dim, hidden_channels)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.regime_projection = nn.Sequential(
            nn.Linear(regime_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.residual_blocks = nn.ModuleList(
            ConditionedTemporalResidualBlock(
                hidden_channels,
                hidden_channels,
                dilation=2 ** (block_index % 4),
            )
            for block_index in range(num_residual_blocks)
        )
        self.output_normalization = nn.GroupNorm(
            _group_count(hidden_channels),
            hidden_channels,
        )
        self.output_projection = nn.Conv1d(
            hidden_channels,
            factor_dim,
            kernel_size=3,
            padding=1,
        )

    def _validate_inputs(
        self,
        noisy_paths: torch.Tensor,
        timesteps: torch.Tensor,
        past_context: torch.Tensor,
        regime_probabilities: torch.Tensor,
    ) -> None:
        if noisy_paths.ndim != 3:
            raise ValueError("noisy_paths must have shape (batch, horizon, factor_dim)")
        batch_size, horizon, factor_dim = noisy_paths.shape
        if horizon < 1 or factor_dim != self.factor_dim:
            raise ValueError("noisy_paths has an unexpected horizon or factor dimension")
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape (batch,)")
        if past_context.ndim != 3 or past_context.shape[0] != batch_size:
            raise ValueError("past_context must share the noisy-path batch dimension")
        if past_context.shape[2] != self.context_dim:
            raise ValueError("past_context has an unexpected feature dimension")
        if regime_probabilities.shape != (batch_size, self.regime_dim):
            raise ValueError("regime_probabilities must have shape (batch, regime_dim)")

        tensors: Sequence[torch.Tensor] = (
            noisy_paths,
            timesteps,
            past_context,
            regime_probabilities,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all model inputs must be on the same device")
        float_tensors = (noisy_paths, past_context, regime_probabilities)
        if any(not tensor.is_floating_point() for tensor in float_tensors):
            raise TypeError("paths, context, and regime probabilities must be floating point")
        if len({tensor.dtype for tensor in float_tensors}) != 1:
            raise ValueError("floating-point inputs must use the same dtype")
        if not all(torch.isfinite(tensor).all() for tensor in float_tensors):
            raise ValueError("model inputs contain non-finite values")
        if torch.any(regime_probabilities < 0.0):
            raise ValueError("regime probabilities cannot be negative")
        if torch.any(regime_probabilities.sum(dim=1) <= 0.0):
            raise ValueError("each regime-probability row must have positive mass")

    def forward(
        self,
        noisy_paths: torch.Tensor,
        timesteps: torch.Tensor,
        past_context: torch.Tensor,
        regime_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            noisy_paths,
            timesteps,
            past_context,
            regime_probabilities,
        )
        normalized_regimes = regime_probabilities / regime_probabilities.sum(
            dim=1,
            keepdim=True,
        )
        condition = (
            self.time_projection(self.time_embedding(timesteps).to(dtype=noisy_paths.dtype))
            + self.context_projection(self.context_encoder(past_context))
            + self.regime_projection(normalized_regimes)
        )
        hidden = self.input_projection(noisy_paths.transpose(1, 2))
        for block in self.residual_blocks:
            hidden = block(hidden, condition)
        prediction = self.output_projection(F.silu(self.output_normalization(hidden)))
        return prediction.transpose(1, 2)


class ConditionalTemporalDDPM(nn.Module):
    """DDPM that denoises a complete future factor path in one shot.

    Parameters are deliberately compact so CPU smoke tests are practical.
    Production experiments still require fold-specific normalization,
    validation-selected capacity, convergence diagnostics, and comparison with
    non-neural scenario generators.
    """

    def __init__(
        self,
        *,
        horizon: int,
        factor_dim: int,
        context_dim: int,
        regime_dim: int,
        num_diffusion_steps: int = 100,
        hidden_channels: int = 64,
        time_embedding_dim: int = 64,
        num_residual_blocks: int = 4,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.factor_dim = int(factor_dim)
        self.context_dim = int(context_dim)
        self.regime_dim = int(regime_dim)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.denoiser = ConditionalTemporalDenoiser(
            factor_dim=factor_dim,
            context_dim=context_dim,
            regime_dim=regime_dim,
            hidden_channels=hidden_channels,
            time_embedding_dim=time_embedding_dim,
            num_residual_blocks=num_residual_blocks,
        )

        betas = _linear_beta_schedule(
            num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]))
        posterior_variance = (betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)).clamp_min(
            1.0e-20
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("posterior_variance", posterior_variance)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @property
    def dtype(self) -> torch.dtype:
        return self.betas.dtype

    def _validate_paths(self, paths: torch.Tensor, *, name: str) -> None:
        expected = (self.horizon, self.factor_dim)
        if paths.ndim != 3 or paths.shape[1:] != expected:
            raise ValueError(f"{name} must have shape (batch, {self.horizon}, {self.factor_dim})")
        if paths.device != self.device:
            raise ValueError(f"{name} must be on model device {self.device}")
        if paths.dtype != self.dtype:
            raise ValueError(f"{name} must use model dtype {self.dtype}")
        if not torch.isfinite(paths).all():
            raise ValueError(f"{name} contains non-finite values")

    def _validate_conditioning(
        self,
        past_context: torch.Tensor,
        regime_probabilities: torch.Tensor,
    ) -> int:
        if past_context.ndim != 3:
            raise ValueError("past_context must have shape (batch, history, context_dim)")
        batch_size = past_context.shape[0]
        if past_context.shape[1] < 1 or past_context.shape[2] != self.context_dim:
            raise ValueError("past_context has an unexpected history or feature dimension")
        if regime_probabilities.shape != (batch_size, self.regime_dim):
            raise ValueError("regime_probabilities must have shape (batch, regime_dim)")
        for name, values in (
            ("past_context", past_context),
            ("regime_probabilities", regime_probabilities),
        ):
            if values.device != self.device:
                raise ValueError(f"{name} must be on model device {self.device}")
            if values.dtype != self.dtype:
                raise ValueError(f"{name} must use model dtype {self.dtype}")
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        if torch.any(regime_probabilities < 0.0):
            raise ValueError("regime probabilities cannot be negative")
        if torch.any(regime_probabilities.sum(dim=1) <= 0.0):
            raise ValueError("each regime-probability row must have positive mass")
        return batch_size

    @staticmethod
    def _extract(coefficients: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return coefficients.gather(0, timesteps).view(-1, 1, 1)

    def q_sample(
        self,
        clean_paths: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the closed-form forward diffusion at selected time steps."""

        self._validate_paths(clean_paths, name="clean_paths")
        self._validate_paths(noise, name="noise")
        if timesteps.shape != (clean_paths.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if timesteps.device != self.device or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a long tensor on the model device")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.num_diffusion_steps):
            raise ValueError("timesteps fall outside the diffusion schedule")
        alpha_bar = self._extract(self.alpha_bars, timesteps)
        return alpha_bar.sqrt() * clean_paths + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(
        self,
        clean_paths: torch.Tensor,
        past_context: torch.Tensor,
        regime_probabilities: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return normalized weighted DDPM noise-prediction loss.

        ``sample_weights`` is intended for a second, tail-focused fine-tuning
        stage. It must be computed from the training fold only.
        """

        self._validate_paths(clean_paths, name="clean_paths")
        batch_size = self._validate_conditioning(
            past_context,
            regime_probabilities,
        )
        if clean_paths.shape[0] != batch_size:
            raise ValueError("clean_paths and conditioning batch sizes differ")
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.num_diffusion_steps,
                (batch_size,),
                generator=generator,
                device=self.device,
            )
        if noise is None:
            noise = torch.randn(
                clean_paths.shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        noisy_paths = self.q_sample(clean_paths, timesteps, noise)
        predicted_noise = self.denoiser(
            noisy_paths,
            timesteps,
            past_context,
            regime_probabilities,
        )
        per_sample = (predicted_noise - noise).square().mean(dim=(1, 2))

        if sample_weights is None:
            return per_sample.mean()
        if (
            sample_weights.shape != (batch_size,)
            or sample_weights.device != self.device
            or sample_weights.dtype != self.dtype
        ):
            raise ValueError("sample_weights must have shape (batch,) and match model device/dtype")
        if not torch.isfinite(sample_weights).all() or torch.any(sample_weights < 0.0):
            raise ValueError("sample_weights must be finite and non-negative")
        total_weight = sample_weights.sum()
        if total_weight <= 0.0:
            raise ValueError("sample_weights must contain positive total mass")
        normalized_weights = sample_weights * batch_size / total_weight
        return (normalized_weights * per_sample).mean()

    def _make_generator(
        self,
        *,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if seed is not None and generator is not None:
            raise ValueError("provide either seed or generator, not both")
        if seed is None:
            return generator
        local_generator = torch.Generator(device=self.device)
        local_generator.manual_seed(int(seed))
        return local_generator

    @torch.no_grad()
    def sample(
        self,
        past_context: torch.Tensor,
        regime_probabilities: torch.Tensor,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a joint future path for each conditioning row.

        Sampling loops over diffusion time, never over forecast horizon. A
        fixed ``seed`` reproduces the starting noise and reverse transitions.
        ``initial_noise`` supports paired conditioning diagnostics.
        """

        batch_size = self._validate_conditioning(
            past_context,
            regime_probabilities,
        )
        random_generator = self._make_generator(seed=seed, generator=generator)
        if initial_noise is None:
            values = torch.randn(
                (batch_size, self.horizon, self.factor_dim),
                generator=random_generator,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._validate_paths(initial_noise, name="initial_noise")
            if initial_noise.shape[0] != batch_size:
                raise ValueError("initial_noise and conditioning batch sizes differ")
            values = initial_noise.clone()

        was_training = self.training
        self.eval()
        try:
            for step in reversed(range(self.num_diffusion_steps)):
                timesteps = torch.full(
                    (batch_size,),
                    step,
                    device=self.device,
                    dtype=torch.long,
                )
                predicted_noise = self.denoiser(
                    values,
                    timesteps,
                    past_context,
                    regime_probabilities,
                )
                alpha = self._extract(self.alphas, timesteps)
                alpha_bar = self._extract(self.alpha_bars, timesteps)
                beta = self._extract(self.betas, timesteps)
                posterior_mean = (
                    values - beta * predicted_noise / (1.0 - alpha_bar).sqrt()
                ) / alpha.sqrt()
                if step > 0:
                    transition_noise = torch.randn(
                        values.shape,
                        generator=random_generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    variance = self._extract(self.posterior_variance, timesteps)
                    values = posterior_mean + variance.sqrt() * transition_noise
                else:
                    values = posterior_mean
        finally:
            self.train(was_training)

        if not torch.isfinite(values).all():
            raise FloatingPointError("reverse diffusion produced non-finite values")
        return values
