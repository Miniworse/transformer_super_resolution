"""Bayesian Visibility Transformer v1 model and srdata helpers.

This first version works directly in the irregular visibility/uv region. It
does not require a provided noise level. Instead, the network infers per-token
noise uncertainty from noisy visibility values, uv geometry, known/unknown
masks, and redundancy time.

Expected input tensors:
    values:       [batch, n_token, 2]   noisy real/imag visibility
                                         use zeros for unknown uv query tokens
    coords:       [batch, n_token, D]   irregular uv or uvw coordinates
    known_mask:   [batch, n_token]      True where visibility is observed
    redundancy:   [batch, n_token, 2]   original and virtual redundancy counts
    token_mask:   [batch, n_token]      True for valid tokens, False for padding

Expected training target:
    target_values: [batch, n_token, 2]  noisy or clean target visibility
    target_mask:   [batch, n_token]     tokens supervised by the target

The model predicts the clean visibility posterior for both known and unknown uv:
    clean_mean:      [batch, n_token, 2]
    clean_logvar:    [batch, n_token, 1]  complex variance log E[|e|^2]
    noise_logvar:    [batch, n_token, 1]  complex variance log E[|n|^2]

For clean targets, keep ``target_is_noisy=False``. For noisy training targets,
use ``target_is_noisy=True`` so the likelihood marginalizes clean uncertainty
and inferred label noise:
    target_noisy ~ N(clean_mean, clean_var + noise_var)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class VisibilityRegionInput:
    """Model-ready tensors packed from native visibility-region arrays."""

    values: Tensor
    coords: Tensor
    known_mask: Tensor
    redundancy: Tensor
    token_mask: Tensor
    target_values: Tensor
    target_mask: Tensor
    original_mask: Tensor
    virtual_mask: Tensor


@dataclass
class BVTOutput:
    """Predictive posterior and latent Bayesian state."""

    clean_mean: Tensor
    clean_logvar: Tensor
    noise_logvar: Tensor
    latent_mean: Tensor
    latent_logvar: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor

    @property
    def total_logvar(self) -> Tensor:
        """Log variance of a noisy visibility observation."""
        clean_var = torch.exp(self.clean_logvar)
        noise_var = torch.exp(self.noise_logvar)
        return torch.log(clean_var + noise_var + 1e-8)


def masked_mean(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.mean(dim=dim)
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def gaussian_nll(
    target: Tensor,
    mean: Tensor,
    logvar: Tensor,
    mask: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Masked diagonal Gaussian negative log likelihood for complex visibility."""
    logvar = logvar.clamp(-14.0, 8.0)
    loss = 0.5 * (math.log(2.0 * math.pi) + logvar + (target - mean).pow(2) * torch.exp(-logvar))
    loss = loss.sum(dim=-1)
    if mask is None and weight is None:
        return loss.mean()
    if mask is None:
        weights = torch.ones_like(loss)
    else:
        weights = mask.to(loss.dtype)
    if weight is not None:
        weights = weights * weight.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def complex_gaussian_nll(
    target: Tensor,
    mean: Tensor,
    logvar: Tensor,
    mask: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Masked circular complex Gaussian NLL.

    ``logvar`` is one scalar per complex visibility token and represents the
    complex variance E[|V - mean|^2]. The density is
    p(V) = 1 / (pi * var) * exp(-|V - mean|^2 / var).
    """
    logvar = logvar.clamp(-14.0, 8.0).squeeze(-1)
    squared_error = (target - mean).pow(2).sum(dim=-1)
    loss = math.log(math.pi) + logvar + squared_error * torch.exp(-logvar)
    if mask is None and weight is None:
        return loss.mean()
    if mask is None:
        weights = torch.ones_like(loss)
    else:
        weights = mask.to(loss.dtype)
    if weight is not None:
        weights = weights * weight.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def kl_normal(mean: Tensor, logvar: Tensor, prior_mean: Tensor, prior_logvar: Tensor) -> Tensor:
    """KL(q || p) for diagonal normal distributions."""
    logvar = logvar.clamp(-14.0, 8.0)
    prior_logvar = prior_logvar.clamp(-14.0, 8.0)
    var_ratio = torch.exp(logvar - prior_logvar)
    mean_diff = (mean - prior_mean).pow(2) * torch.exp(-prior_logvar)
    kl = 0.5 * (prior_logvar - logvar + var_ratio + mean_diff - 1.0)
    return kl.sum(dim=-1).mean()


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UVFourierEncoding(nn.Module):
    """Fourier features for irregular uv/uvw coordinates.

    The deterministic frequency bank makes nearby and far-away uv relations
    visible to attention without assuming a rectangular grid.
    """

    def __init__(
        self,
        coord_dim: int,
        num_frequencies: int = 16,
        max_frequency: float = 64.0,
        normalize_coords: bool = False,
    ) -> None:
        super().__init__()
        if coord_dim < 2:
            raise ValueError("coord_dim must include at least u and v.")
        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies
        self.normalize_coords = normalize_coords
        freqs = torch.logspace(0.0, math.log10(max_frequency), num_frequencies)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        # coords + radius + angle + optional scale + sin/cos for every coordinate/frequency
        scale_dim = 1 if self.normalize_coords else 0
        return self.coord_dim + 2 + scale_dim + 2 * self.coord_dim * self.num_frequencies

    def forward(self, coords: Tensor) -> Tensor:
        if self.normalize_coords:
            raw_radius = torch.linalg.norm(coords[..., :2], dim=-1, keepdim=True)
            coord_scale = raw_radius.amax(dim=1, keepdim=True).clamp_min(1e-6)
            encoded_coords = coords / coord_scale
        else:
            coord_scale = None
            encoded_coords = coords

        uv = encoded_coords[..., :2]
        radius = torch.linalg.norm(uv, dim=-1, keepdim=True)
        angle = torch.atan2(uv[..., 1:2], uv[..., 0:1])

        scaled = encoded_coords.unsqueeze(-1) * self.freqs.to(coords.dtype) * (2.0 * math.pi)
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        fourier = fourier.flatten(start_dim=-2)
        if coord_scale is None:
            return torch.cat([encoded_coords, radius, angle, fourier], dim=-1)

        log_scale = torch.log1p(coord_scale).expand(*encoded_coords.shape[:2], 1)
        return torch.cat([encoded_coords, radius, angle, log_scale, fourier], dim=-1)


class VisibilityTokenEmbedder(nn.Module):
    """Embeds noisy visibility values, uv geometry, masks, and redundancy."""

    def __init__(
        self,
        coord_dim: int,
        model_dim: int,
        redundancy_dim: int = 2,
        num_frequencies: int = 16,
        max_frequency: float = 64.0,
        normalize_coords: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.coord_encoding = UVFourierEncoding(coord_dim, num_frequencies, max_frequency, normalize_coords)
        self.value_proj = MLP(2, model_dim, model_dim, dropout)
        self.coord_proj = MLP(self.coord_encoding.out_dim, model_dim, model_dim, dropout)
        self.redundancy_proj = MLP(redundancy_dim, model_dim, model_dim, dropout)
        self.known_embed = nn.Embedding(2, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, values: Tensor, coords: Tensor, known_mask: Tensor, redundancy: Tensor) -> Tensor:
        redundancy = torch.log1p(redundancy.clamp_min(0.0))
        token = (
            self.value_proj(values)
            + self.coord_proj(self.coord_encoding(coords))
            + self.redundancy_proj(redundancy)
            + self.known_embed(known_mask.to(torch.long))
        )
        return self.norm(token)


class LatentNoiseState(nn.Module):
    """Global latent state for unobserved scene/noise conditions."""

    def __init__(self, model_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.posterior = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2 * latent_dim),
        )
        self.to_model = nn.Linear(latent_dim, model_dim)

    def forward(self, tokens: Tensor, token_mask: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pooled = masked_mean(tokens, token_mask, dim=1)
        latent_mean, latent_logvar = self.posterior(pooled).chunk(2, dim=-1)
        prior_mean = torch.zeros_like(latent_mean)
        prior_logvar = torch.zeros_like(latent_logvar)

        latent_logvar = latent_logvar.clamp(-10.0, 5.0)
        if self.training:
            eps = torch.randn_like(latent_mean)
            z = latent_mean + eps * torch.exp(0.5 * latent_logvar)
        else:
            z = latent_mean
        return self.to_model(z), latent_mean, latent_logvar, prior_mean, prior_logvar


class BayesianVisibilityTransformer(nn.Module):
    """First-version transformer for visibility denoising and uv SR.

    Known and unknown uv positions are encoded as one irregular token set.
    Unknown/query tokens should have zero values and ``known_mask=False``. The
    model predicts clean visibility at every valid token, so super-resolution is
    realized inside the same transformer instead of a separate module.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        redundancy_dim: int = 2,
        model_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 8,
        num_heads: int = 8,
        num_frequencies: int = 16,
        max_frequency: float = 64.0,
        normalize_coords: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            dropout=dropout,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.latent = LatentNoiseState(model_dim, latent_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 4),
        )

    def forward(
        self,
        values: Tensor,
        coords: Tensor,
        known_mask: Tensor,
        redundancy: Tensor,
        token_mask: Optional[Tensor] = None,
    ) -> BVTOutput:
        if token_mask is None:
            token_mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)

        tokens = self.embed(values, coords, known_mask, redundancy)
        key_padding_mask = ~token_mask.bool()
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        z_token, latent_mean, latent_logvar, prior_mean, prior_logvar = self.latent(encoded, token_mask)
        encoded = encoded + z_token.unsqueeze(1)

        pred = self.head(encoded)
        clean_mean = pred[..., :2]
        clean_logvar = pred[..., 2:3].clamp(-12.0, 6.0)
        noise_logvar = pred[..., 3:4].clamp(-12.0, 6.0)

        return BVTOutput(
            clean_mean=clean_mean,
            clean_logvar=clean_logvar,
            noise_logvar=noise_logvar,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )


class BayesianVisibilityEncoderDecoder(nn.Module):
    """Observed-visibility encoder with expanded-uv query decoder.

    The encoder only receives observed/original visibility evidence. The decoder
    predicts clean visibility at every valid uv token as a query. This keeps the
    information path physically clearer:

        observed noisy visibility -> encoder memory -> uv query decoder
    """

    def __init__(
        self,
        coord_dim: int = 2,
        redundancy_dim: int = 2,
        model_dim: int = 256,
        latent_dim: int = 64,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        num_frequencies: int = 16,
        max_frequency: float = 64.0,
        normalize_coords: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            dropout=dropout,
        )
        self.query_embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            dropout=dropout,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.latent = LatentNoiseState(model_dim, latent_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 4),
        )

    def forward(
        self,
        values: Tensor,
        coords: Tensor,
        known_mask: Tensor,
        redundancy: Tensor,
        token_mask: Optional[Tensor] = None,
    ) -> BVTOutput:
        if token_mask is None:
            token_mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)

        context_mask = token_mask.bool() & known_mask.bool()
        context_values = values.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        context_tokens = self.context_embed(context_values, coords, context_mask, redundancy)
        memory_key_padding_mask = ~context_mask
        memory = self.encoder(context_tokens, src_key_padding_mask=memory_key_padding_mask)

        z_token, latent_mean, latent_logvar, prior_mean, prior_logvar = self.latent(memory, context_mask)
        memory = memory + z_token.unsqueeze(1)

        query_values = values.masked_fill(~known_mask.bool().unsqueeze(-1), 0.0)
        query_tokens = self.query_embed(query_values, coords, known_mask.bool(), redundancy)
        decoded = self.decoder(
            query_tokens,
            memory,
            tgt_key_padding_mask=~token_mask.bool(),
            memory_key_padding_mask=memory_key_padding_mask,
        )

        pred = self.head(decoded)
        delta_or_value = pred[..., :2]
        clean_mean = torch.where(
            known_mask.bool().unsqueeze(-1),
            values + delta_or_value,
            delta_or_value,
        )
        clean_logvar = pred[..., 2:3].clamp(-12.0, 6.0)
        noise_logvar = pred[..., 3:4].clamp(-12.0, 6.0)

        return BVTOutput(
            clean_mean=clean_mean,
            clean_logvar=clean_logvar,
            noise_logvar=noise_logvar,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )


def training_objective(
    output: BVTOutput,
    target_values: Tensor,
    target_mask: Optional[Tensor],
    target_is_noisy: bool = False,
    beta_kl: float = 1e-3,
    beta_noise_prior: float = 1e-4,
    target_weight: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Default objective for noisy-label denoising and virtual-array SR.

    If expanded virtual visibility labels are noisy, keep ``target_is_noisy`` as
    True. If you later have clean simulated targets, set it to False so the
    clean posterior is trained directly.
    """

    likelihood_logvar = output.total_logvar if target_is_noisy else output.clean_logvar
    nll = complex_gaussian_nll(target_values, output.clean_mean, likelihood_logvar, target_mask, target_weight)
    kl = kl_normal(output.latent_mean, output.latent_logvar, output.prior_mean, output.prior_logvar)

    # Mildly discourages explaining every error as measurement noise while still
    # allowing the Bayesian branch to infer high-noise visibility regions.
    if target_mask is None:
        noise_prior = torch.exp(output.noise_logvar).mean()
    else:
        mask = target_mask.to(output.noise_logvar.dtype).unsqueeze(-1)
        denom = (mask.sum() * output.noise_logvar.shape[-1]).clamp_min(1.0)
        noise_prior = (torch.exp(output.noise_logvar) * mask).sum() / denom

    loss = nll + beta_kl * kl + beta_noise_prior * noise_prior
    metrics = {
        "loss": loss.detach(),
        "nll": nll.detach(),
        "kl": kl.detach(),
        "noise_prior": noise_prior.detach(),
    }
    return loss, metrics


def radial_frequency_weights(
    coords: Tensor,
    token_mask: Optional[Tensor],
    alpha: float = 2.0,
    gamma: float = 1.0,
) -> Tensor:
    """Weight outer uv radius more strongly for super-resolution learning."""
    rho = torch.linalg.norm(coords[..., :2], dim=-1)
    if token_mask is None:
        rho_max = rho.amax(dim=1, keepdim=True)
    else:
        masked_rho = rho.masked_fill(~token_mask.bool(), 0.0)
        rho_max = masked_rho.amax(dim=1, keepdim=True)
    rho_norm = rho / rho_max.clamp_min(1e-6)
    return 1.0 + alpha * rho_norm.pow(gamma)


def radial_bin_balanced_weights(
    coords: Tensor,
    balance_mask: Tensor,
    num_bins: int = 8,
) -> Tensor:
    """Give each occupied uv-radius bin equal total weight."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    rho = torch.linalg.norm(coords[..., :2], dim=-1)
    weights = torch.zeros_like(rho)
    for batch_idx in range(coords.shape[0]):
        valid = balance_mask[batch_idx].bool()
        if valid.sum() == 0:
            continue

        rho_valid = rho[batch_idx, valid]
        rho_norm = rho_valid / rho_valid.max().clamp_min(1e-6)
        bin_index = torch.clamp((rho_norm * num_bins).long(), max=num_bins - 1)
        occupied = torch.unique(bin_index)
        valid_count = valid.sum().to(rho.dtype)
        for bin_id in occupied:
            in_bin = bin_index == bin_id
            bin_weight = valid_count / (occupied.numel() * in_bin.sum().clamp_min(1).to(rho.dtype))
            valid_positions = torch.where(valid)[0][in_bin]
            weights[batch_idx, valid_positions] = bin_weight
    return weights


def hermitian_symmetry_loss(
    pred_values: Tensor,
    coords: Tensor,
    token_mask: Optional[Tensor],
    tolerance: float = 1e-4,
) -> Tensor:
    """Penalize violations of V(-u, -v) = conj(V(u, v))."""
    losses = []
    batch_size = pred_values.shape[0]
    for batch_idx in range(batch_size):
        if token_mask is None:
            valid = torch.ones(pred_values.shape[1], device=pred_values.device, dtype=torch.bool)
        else:
            valid = token_mask[batch_idx].bool()
        if valid.sum() < 2:
            continue

        uv = coords[batch_idx, valid, :2]
        pred = pred_values[batch_idx, valid]
        distance = torch.cdist(uv, -uv)
        min_distance, pair_index = distance.min(dim=1)
        pair_mask = min_distance <= tolerance
        if pair_mask.sum() == 0:
            continue

        pred_pair = pred[pair_index[pair_mask]]
        pred_conj = torch.stack([pred[pair_mask, 0], -pred[pair_mask, 1]], dim=-1)
        losses.append((pred_pair - pred_conj).abs().sum(dim=-1).mean())

    if not losses:
        return pred_values.new_zeros(())
    return torch.stack(losses).mean()


def complex_energy_loss(pred_values: Tensor, target_values: Tensor, mask: Tensor) -> Tensor:
    """Match total complex visibility energy in a masked region."""
    if mask.sum() == 0:
        return pred_values.new_zeros(())
    weights = mask.unsqueeze(-1).to(pred_values.dtype)
    pred_energy = ((pred_values.pow(2)) * weights).sum().sqrt()
    target_energy = ((target_values.pow(2)) * weights).sum().sqrt().clamp_min(1e-8)
    return (pred_energy / target_energy - 1.0).abs()


def complex_normalized_mse(
    pred_values: Tensor,
    target_values: Tensor,
    mask: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Complex MSE normalized by target power in a masked region."""
    if mask.sum() == 0:
        return pred_values.new_zeros(())
    weights = mask.to(pred_values.dtype)
    if weight is not None:
        weights = weights * weight.to(pred_values.dtype)
    squared_error = (pred_values - target_values).pow(2).sum(dim=-1)
    target_power = target_values.pow(2).sum(dim=-1)
    numerator = (squared_error * weights).sum()
    denominator = (target_power * weights).sum().clamp_min(1e-8)
    return numerator / denominator


def complex_phase_loss(
    pred_values: Tensor,
    target_values: Tensor,
    mask: Tensor,
    amplitude_floor: float = 1e-6,
) -> Tensor:
    """Amplitude-weighted wrapped phase loss for complex visibility."""
    target_amp = torch.linalg.norm(target_values, dim=-1)
    phase_weight = mask.to(pred_values.dtype) * target_amp
    phase_weight = phase_weight.masked_fill(target_amp <= amplitude_floor, 0.0)
    if phase_weight.sum() == 0:
        return pred_values.new_zeros(())

    pred_phase = torch.atan2(pred_values[..., 1], pred_values[..., 0])
    target_phase = torch.atan2(target_values[..., 1], target_values[..., 0])
    phase_error = 1.0 - torch.cos(pred_phase - target_phase)
    return (phase_error * phase_weight).sum() / phase_weight.sum().clamp_min(1e-8)


def visibility_physical_objective(
    output: BVTOutput,
    batch: VisibilityRegionInput,
    target_is_noisy: bool = False,
    beta_kl: float = 1e-3,
    beta_noise_prior: float = 1e-4,
    lambda_orig: float = 1.0,
    lambda_virtual: float = 2.0,
    lambda_expanded: float = 3.0,
    lambda_high_freq: float = 1.0,
    lambda_radial_bins: float = 0.0,
    lambda_sym: float = 0.1,
    lambda_energy_orig: float = 0.5,
    lambda_energy_virtual: float = 0.5,
    lambda_phase: float = 0.1,
    lambda_phase_expanded: float = 0.0,
    lambda_expanded_nmse: float = 0.0,
    freq_alpha: float = 2.0,
    freq_gamma: float = 1.0,
    num_radial_bins: int = 8,
    symmetry_tolerance: float = 1e-4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Region-separated physical objective for visibility denoising and SR."""
    likelihood_logvar = output.total_logvar if target_is_noisy else output.clean_logvar
    expanded_mask = batch.virtual_mask & ~batch.original_mask
    freq_weights = radial_frequency_weights(batch.coords, batch.token_mask, freq_alpha, freq_gamma)
    radial_bin_weights = radial_bin_balanced_weights(batch.coords, expanded_mask, num_radial_bins)

    nll_orig = complex_gaussian_nll(batch.target_values, output.clean_mean, likelihood_logvar, batch.original_mask)
    nll_virtual = complex_gaussian_nll(batch.target_values, output.clean_mean, likelihood_logvar, batch.virtual_mask)
    nll_expanded = complex_gaussian_nll(batch.target_values, output.clean_mean, likelihood_logvar, expanded_mask)
    nll_high_freq = complex_gaussian_nll(
        batch.target_values,
        output.clean_mean,
        likelihood_logvar,
        expanded_mask,
        weight=freq_weights,
    )
    nll_radial_bins = complex_gaussian_nll(
        batch.target_values,
        output.clean_mean,
        likelihood_logvar,
        expanded_mask,
        weight=radial_bin_weights,
    )

    kl = kl_normal(output.latent_mean, output.latent_logvar, output.prior_mean, output.prior_logvar)
    mask = batch.target_mask.to(output.noise_logvar.dtype).unsqueeze(-1)
    denom = (mask.sum() * output.noise_logvar.shape[-1]).clamp_min(1.0)
    noise_prior = (torch.exp(output.noise_logvar) * mask).sum() / denom
    sym = hermitian_symmetry_loss(output.clean_mean, batch.coords, batch.token_mask, symmetry_tolerance)
    energy_orig = complex_energy_loss(output.clean_mean, batch.target_values, batch.original_mask)
    energy_virtual = complex_energy_loss(output.clean_mean, batch.target_values, batch.virtual_mask)
    expanded_nmse = complex_normalized_mse(output.clean_mean, batch.target_values, expanded_mask)
    expanded_radial_nmse = complex_normalized_mse(
        output.clean_mean,
        batch.target_values,
        expanded_mask,
        weight=radial_bin_weights,
    )
    phase = complex_phase_loss(output.clean_mean, batch.target_values, batch.target_mask)
    phase_orig = complex_phase_loss(output.clean_mean, batch.target_values, batch.original_mask)
    phase_virtual = complex_phase_loss(output.clean_mean, batch.target_values, batch.virtual_mask)
    phase_expanded = complex_phase_loss(output.clean_mean, batch.target_values, expanded_mask)

    region_nll = (
        lambda_orig * nll_orig
        + lambda_virtual * nll_virtual
        + lambda_expanded * nll_expanded
        + lambda_high_freq * nll_high_freq
        + lambda_radial_bins * nll_radial_bins
    )
    energy = lambda_energy_orig * energy_orig + lambda_energy_virtual * energy_virtual
    sr_mean_loss = lambda_expanded_nmse * expanded_radial_nmse
    phase_loss = lambda_phase * phase + lambda_phase_expanded * phase_expanded
    loss = region_nll + lambda_sym * sym + energy + sr_mean_loss + phase_loss + beta_kl * kl + beta_noise_prior * noise_prior
    metrics = {
        "loss": loss.detach(),
        "region_nll": region_nll.detach(),
        "nll_original": nll_orig.detach(),
        "nll_virtual": nll_virtual.detach(),
        "nll_expanded_only": nll_expanded.detach(),
        "nll_high_freq": nll_high_freq.detach(),
        "nll_radial_bins": nll_radial_bins.detach(),
        "hermitian": sym.detach(),
        "energy": energy.detach(),
        "energy_original": energy_orig.detach(),
        "energy_virtual": energy_virtual.detach(),
        "expanded_nmse_loss": expanded_nmse.detach(),
        "expanded_radial_nmse_loss": expanded_radial_nmse.detach(),
        "sr_mean_loss": sr_mean_loss.detach(),
        "phase": phase.detach(),
        "phase_original": phase_orig.detach(),
        "phase_virtual": phase_virtual.detach(),
        "phase_expanded_only": phase_expanded.detach(),
        "phase_loss": phase_loss.detach(),
        "kl": kl.detach(),
        "noise_prior": noise_prior.detach(),
    }
    return loss, metrics


def _as_float_tensor(x: Tensor | object) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _visibility_array_to_batch_last(x: Tensor | object, name: str, channels: int = 2) -> Tensor:
    """Convert [V, C], [C, V], [B, V, C], or [B, C, V] to [B, V, C]."""
    x = _as_float_tensor(x)
    if x.ndim == 2:
        if x.shape[-1] == channels:
            x = x.unsqueeze(0)
        elif x.shape[0] == channels:
            x = x.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError(f"{name} must have shape [V, {channels}] or [{channels}, V], got {tuple(x.shape)}.")
    elif x.ndim == 3:
        if x.shape[-1] == channels:
            pass
        elif x.shape[1] == channels:
            x = x.transpose(1, 2)
        else:
            raise ValueError(
                f"{name} must have shape [B, V, {channels}] or [B, {channels}, V], got {tuple(x.shape)}."
            )
    else:
        raise ValueError(f"{name} must be 2D or 3D, got {x.ndim}D.")
    return x.contiguous()


def _channel_first_to_batch_last(x: Tensor | object, name: str, channels: int = 2) -> Tensor:
    """Backward-compatible alias; current data should use [V, 2]."""
    return _visibility_array_to_batch_last(x, name, channels)


def build_visibility_region_inputs(
    uv: Tensor | object,
    visibility: Tensor | object,
    redu: Tensor | object,
    valid_mask: Optional[Tensor | object] = None,
    target_visibility: Optional[Tensor | object] = None,
    include_virtual_context: bool = False,
) -> VisibilityRegionInput:
    """Pack native visibility-region arrays into the transformer's token API.

    Args:
        uv: [V, 2] or [B, V, 2], columns are u and v. The old [2, V] and
            [B, 2, V] conventions are still accepted.
        visibility: [V, 2] or [B, V, 2], columns are real and imaginary noisy
            input visibility.
        redu: [V, 2] or [B, V, 2]. Column 0 is original visibility redundancy
            and is 0 where original visibility is unknown. Column 1 is
            introduced virtual-array redundancy.
        valid_mask: Optional [V] or [B, V] mask for padded positions.
        target_visibility: Optional clean or noisy target visibility with the
            same shape as ``visibility``. If omitted, ``visibility`` is used as
            the target.
        include_virtual_context: If False, only original-known visibility is
            visible to the model input. Virtual visibility remains available as
            target supervision but is hidden from context.

    Returns:
        VisibilityRegionInput with [B, V, *] tensors. Use ``values``, ``coords``,
        ``known_mask``, ``redundancy``, and ``token_mask`` for model.forward().
        Use ``target_values`` and ``target_mask`` for training_objective().
    """

    coords = _visibility_array_to_batch_last(uv, "uv")
    input_values = _visibility_array_to_batch_last(visibility, "visibility")
    target_values = input_values if target_visibility is None else _visibility_array_to_batch_last(
        target_visibility,
        "target_visibility",
    )
    redundancy = _visibility_array_to_batch_last(redu, "redu")

    if coords.shape[:2] != input_values.shape[:2] or coords.shape[:2] != redundancy.shape[:2]:
        raise ValueError("uv, visibility, and redu must describe the same batch size and V_all length.")
    if target_values.shape != input_values.shape:
        raise ValueError(f"target_visibility must have shape {tuple(input_values.shape)}, got {tuple(target_values.shape)}.")

    original_mask = redundancy[..., 0] > 0
    virtual_mask = redundancy[..., 1] > 0

    if valid_mask is None:
        token_mask = torch.ones(coords.shape[:2], device=coords.device, dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=coords.device)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != coords.shape[:2]:
            raise ValueError(f"valid_mask must have shape {tuple(coords.shape[:2])}, got {tuple(valid_mask.shape)}.")
        token_mask = valid_mask.to(torch.bool).to(coords.device)

    known_mask = original_mask | virtual_mask if include_virtual_context else original_mask
    target_mask = (original_mask | virtual_mask) & token_mask
    known_mask = known_mask & token_mask
    original_mask = original_mask & token_mask
    virtual_mask = virtual_mask & token_mask

    values = input_values.masked_fill(~known_mask.unsqueeze(-1), 0.0)

    return VisibilityRegionInput(
        values=values,
        coords=coords,
        known_mask=known_mask,
        redundancy=redundancy,
        token_mask=token_mask,
        target_values=target_values,
        target_mask=target_mask,
        original_mask=original_mask,
        virtual_mask=virtual_mask,
    )


def parse_srdata_filename(path: str | Path) -> tuple[str, int, int, str]:
    """Parse uv/visibility/redundant_####_expand_#_suffix.npy filenames."""
    match = re.match(
        r"^(uv|visibility|redundant)_(\d+)_expand_(\d+)_(unnoised|noised_\d+)\.npy$",
        Path(path).name,
    )
    if match is None:
        raise ValueError(f"Unsupported srdata filename: {Path(path).name}")
    kind, scene_id, expand_id, suffix = match.groups()
    return kind, int(scene_id), int(expand_id), suffix


def srdata_path(root: str | Path, kind: str, scene_id: int, expand_id: int, suffix: str = "unnoised") -> Path:
    return Path(root) / f"{kind}_{scene_id:04d}_expand_{expand_id}_{suffix}.npy"


def _srdata_candidate_paths(
    root: str | Path,
    kind: str,
    scene_id: int,
    expand_id: int,
    suffix: str,
) -> list[Path]:
    root = Path(root)
    paths = [srdata_path(root, kind, scene_id, expand_id, suffix)]
    if suffix != "unnoised":
        paths.append(srdata_path(root, kind, scene_id, expand_id, "unnoised"))

    if kind in {"uv", "redundant"}:
        paths.append(root / f"{kind}_expand_{expand_id}_{suffix}.npy")
        if suffix != "unnoised":
            paths.append(root / f"{kind}_expand_{expand_id}_unnoised.npy")
        paths.append(root / f"{kind}_expand_{expand_id}.npy")

    return paths


def _first_existing_path(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_npy(path: str | Path) -> Tensor:
    import numpy as np

    return torch.from_numpy(np.load(path)).to(torch.float32)


def _load_npy_from_candidates(
    root: str | Path,
    kind: str,
    scene_id: int,
    expand_id: int,
    suffix: str,
) -> Tensor:
    candidates = _srdata_candidate_paths(root, kind, scene_id, expand_id, suffix)
    path = _first_existing_path(candidates)
    if path is None:
        path = candidates[0]
    return _load_npy(path)


def _synthetic_redundancy_like(visibility: Tensor) -> Tensor:
    visibility_batch = _visibility_array_to_batch_last(visibility, "visibility")
    redundancy = torch.ones((*visibility_batch.shape[:2], 2), dtype=visibility_batch.dtype)
    if visibility.ndim == 2:
        return redundancy.squeeze(0)
    return redundancy


def load_srdata_arrays(
    root: str | Path,
    scene_id: int,
    expand_id: int,
    input_suffix: str = "unnoised",
    target_suffix: Optional[str] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load one simulated scene/expand sample.

    Returns ``uv``, input ``visibility``, ``redundant``, and target visibility,
    all with the current [V, 2] layout. Datasets may store uv/redundancy either
    per scene or once per expansion as ``uv_expand_#.npy``. If redundancy is not
    provided, every token is treated as both observed and supervised, which is
    appropriate for denoising-only grids with no explicit expanded-only region.
    """
    target_suffix = input_suffix if target_suffix is None else target_suffix
    uv = _load_npy_from_candidates(root, "uv", scene_id, expand_id, input_suffix)
    visibility = _load_npy(srdata_path(root, "visibility", scene_id, expand_id, input_suffix))
    redundancy_path = _first_existing_path(
        _srdata_candidate_paths(root, "redundant", scene_id, expand_id, input_suffix)
    )
    redundancy = _load_npy(redundancy_path) if redundancy_path is not None else _synthetic_redundancy_like(visibility)
    target_visibility = _load_npy(srdata_path(root, "visibility", scene_id, expand_id, target_suffix))
    return uv, visibility, redundancy, target_visibility


class SRVisibilityDataset(torch.utils.data.Dataset):
    """Dataset for the simulated srdata npy triplets."""

    def __init__(
        self,
        root: str | Path,
        scene_ids: Optional[list[int]] = None,
        expand_ids: Optional[list[int]] = None,
        input_suffix: str = "unnoised",
        target_suffix: Optional[str] = None,
        include_virtual_context: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.input_suffix = input_suffix
        self.target_suffix = target_suffix
        self.include_virtual_context = include_virtual_context

        scene_filter = None if scene_ids is None else set(scene_ids)
        expand_filter = None if expand_ids is None else set(expand_ids)
        samples: list[tuple[int, int]] = []

        for path in self.root.glob(f"visibility_*_expand_*_{input_suffix}.npy"):
            _, scene_id, expand_id, _ = parse_srdata_filename(path)
            if scene_filter is not None and scene_id not in scene_filter:
                continue
            if expand_filter is not None and expand_id not in expand_filter:
                continue

            uv_path = _first_existing_path(_srdata_candidate_paths(self.root, "uv", scene_id, expand_id, input_suffix))
            required = [uv_path]
            if target_suffix is not None:
                required.append(srdata_path(self.root, "visibility", scene_id, expand_id, target_suffix))
            if all(item is not None and item.exists() for item in required):
                samples.append((scene_id, expand_id))

        self.samples = sorted(samples)
        if not self.samples:
            raise ValueError(f"No srdata samples found in {self.root} for suffix {input_suffix!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VisibilityRegionInput:
        scene_id, expand_id = self.samples[index]
        uv, visibility, redundancy, target_visibility = load_srdata_arrays(
            self.root,
            scene_id,
            expand_id,
            input_suffix=self.input_suffix,
            target_suffix=self.target_suffix,
        )
        return build_visibility_region_inputs(
            uv,
            visibility,
            redundancy,
            target_visibility=target_visibility,
            include_virtual_context=self.include_virtual_context,
        )


def visibility_collate_fn(samples: list[VisibilityRegionInput]) -> VisibilityRegionInput:
    """Pad mixed expand levels into a batch."""
    max_tokens = max(sample.values.shape[1] for sample in samples)

    def pad_last(x: Tensor, pad_value: float = 0.0) -> Tensor:
        pad_tokens = max_tokens - x.shape[1]
        if pad_tokens > 0:
            pad = torch.full((x.shape[0], pad_tokens, *x.shape[2:]), pad_value, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x.squeeze(0)

    def pad_mask(x: Tensor) -> Tensor:
        pad_tokens = max_tokens - x.shape[1]
        if pad_tokens > 0:
            pad = torch.zeros((x.shape[0], pad_tokens), dtype=torch.bool, device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x.squeeze(0)

    return VisibilityRegionInput(
        values=torch.stack([pad_last(sample.values) for sample in samples], dim=0),
        coords=torch.stack([pad_last(sample.coords) for sample in samples], dim=0),
        known_mask=torch.stack([pad_mask(sample.known_mask) for sample in samples], dim=0),
        redundancy=torch.stack([pad_last(sample.redundancy) for sample in samples], dim=0),
        token_mask=torch.stack([pad_mask(sample.token_mask) for sample in samples], dim=0),
        target_values=torch.stack([pad_last(sample.target_values) for sample in samples], dim=0),
        target_mask=torch.stack([pad_mask(sample.target_mask) for sample in samples], dim=0),
        original_mask=torch.stack([pad_mask(sample.original_mask) for sample in samples], dim=0),
        virtual_mask=torch.stack([pad_mask(sample.virtual_mask) for sample in samples], dim=0),
    )


def make_visibility_tokens(
    known_values: Tensor,
    known_coords: Tensor,
    known_redundancy: Tensor,
    query_coords: Tensor,
    query_redundancy: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack observed and unknown uv into the single-token-set API.

    This helper is useful at inference time:
        known noisy visibility -> first tokens
        unknown expanded uv    -> later tokens with zero values
    """

    batch, n_query, _ = query_coords.shape
    query_values = torch.zeros(batch, n_query, 2, device=known_values.device, dtype=known_values.dtype)
    if query_redundancy is None:
        query_redundancy = torch.zeros(batch, n_query, 1, device=known_values.device, dtype=known_values.dtype)

    known_virtual_redundancy = torch.zeros_like(known_redundancy)
    query_original_redundancy = torch.zeros_like(query_redundancy)
    known_redundancy_pair = torch.cat([known_redundancy, known_virtual_redundancy], dim=-1)
    query_redundancy_pair = torch.cat([query_original_redundancy, query_redundancy], dim=-1)

    values = torch.cat([known_values, query_values], dim=1)
    coords = torch.cat([known_coords, query_coords], dim=1)
    redundancy = torch.cat([known_redundancy_pair, query_redundancy_pair], dim=1)

    known_flag = torch.ones(known_values.shape[:2], device=known_values.device, dtype=torch.bool)
    query_flag = torch.zeros((batch, n_query), device=known_values.device, dtype=torch.bool)
    known_mask = torch.cat([known_flag, query_flag], dim=1)
    return values, coords, known_mask, redundancy


def apply_context_dropout(
    values: Tensor,
    known_mask: Tensor,
    drop_probability: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Hide a random subset of known uv values during training.

    This is useful when targets are noisy and clean labels are unavailable. The
    dropped tokens keep their uv coordinates and redundancy, but their input
    visibility is set to zero and their known flag becomes False. They can still
    be included in ``target_mask`` for supervised/noisy-label likelihood.
    """

    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must be in [0, 1).")
    if drop_probability == 0.0:
        return values, known_mask

    keep = torch.rand(known_mask.shape, device=known_mask.device) >= drop_probability
    context_known_mask = known_mask & keep
    context_values = values.masked_fill(~context_known_mask.unsqueeze(-1), 0.0)
    return context_values, context_known_mask
