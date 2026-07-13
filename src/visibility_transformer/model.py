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
    visibility_scale: Tensor
    gram_context_values: Tensor


def complex_visibility_features(values: Tensor, known_mask: Optional[Tensor] = None) -> Tensor:
    """Augment real/imag values with compressed amplitude and circular phase."""
    amplitude = torch.linalg.norm(values, dim=-1, keepdim=True)
    unit = values / amplitude.clamp_min(1e-8)
    features = torch.cat([values, torch.log1p(amplitude), unit], dim=-1)
    if known_mask is not None:
        features = features * known_mask.to(values.dtype).unsqueeze(-1)
    return features


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


def conjugate_real_imag(values: Tensor) -> Tensor:
    """Return the real/imag representation of the complex conjugate."""
    return torch.stack([values[..., 0], -values[..., 1]], dim=-1)


def hermitian_symmetrize_complex_values(
    values: Tensor,
    coords: Tensor,
    token_mask: Optional[Tensor],
    tolerance: float = 1e-4,
) -> Tensor:
    """Project complex values onto V(-u, -v) = conj(V(u, v)).

    The projection is applied only to valid tokens that have a matching
    conjugate uv partner inside ``tolerance``. Self-conjugate zero-baseline
    tokens are projected to real values by construction.
    """
    projected = values.clone()
    batch_size = values.shape[0]
    for batch_idx in range(batch_size):
        if token_mask is None:
            valid = torch.ones(values.shape[1], device=values.device, dtype=torch.bool)
        else:
            valid = token_mask[batch_idx].bool()
        if valid.sum() == 0:
            continue

        valid_indices = torch.where(valid)[0]
        uv = coords[batch_idx, valid, :2]
        distance = torch.cdist(uv, -uv)
        min_distance, pair_index = distance.min(dim=1)
        pair_mask = min_distance <= tolerance
        if pair_mask.sum() == 0:
            continue

        source_index = valid_indices[pair_mask]
        partner_index = valid_indices[pair_index[pair_mask]]
        source_values = values[batch_idx, source_index]
        partner_conj = conjugate_real_imag(values[batch_idx, partner_index])
        projected[batch_idx, source_index] = 0.5 * (source_values + partner_conj)
    return projected


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
        value_dim: int = 2,
        use_complex_features: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.coord_encoding = UVFourierEncoding(coord_dim, num_frequencies, max_frequency, normalize_coords)
        self.use_complex_features = use_complex_features
        self.value_proj = MLP(value_dim, model_dim, model_dim, dropout)
        self.coord_proj = MLP(self.coord_encoding.out_dim, model_dim, model_dim, dropout)
        self.redundancy_proj = MLP(redundancy_dim, model_dim, model_dim, dropout)
        self.known_embed = nn.Embedding(2, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: Tensor,
        coords: Tensor,
        known_mask: Tensor,
        redundancy: Tensor,
        gram_context_values: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_complex_features:
            values = complex_visibility_features(values, known_mask)
            if gram_context_values is not None:
                gram_context_values = complex_visibility_features(gram_context_values)
        if gram_context_values is not None:
            values = torch.cat([values, gram_context_values], dim=-1)
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
        use_gram_prior: bool = False,
        gram_prior_mode: str = "feature",
        use_complex_features: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if gram_prior_mode not in {"feature", "residual"}:
            raise ValueError(f"Unsupported gram_prior_mode: {gram_prior_mode!r}.")
        self.use_gram_prior = use_gram_prior
        self.gram_prior_mode = gram_prior_mode
        self.embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            value_dim=(5 if use_complex_features else 2) * (2 if use_gram_prior else 1),
            use_complex_features=use_complex_features,
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
        gram_context_values: Optional[Tensor] = None,
    ) -> BVTOutput:
        if token_mask is None:
            token_mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        if self.use_gram_prior and gram_context_values is None:
            gram_context_values = torch.zeros_like(values)
        elif not self.use_gram_prior:
            gram_context_values = None

        tokens = self.embed(values, coords, known_mask, redundancy, gram_context_values)
        key_padding_mask = ~token_mask.bool()
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        z_token, latent_mean, latent_logvar, prior_mean, prior_logvar = self.latent(encoded, token_mask)
        encoded = encoded + z_token.unsqueeze(1)

        pred = self.head(encoded)
        delta_or_value = pred[..., :2]
        if self.use_gram_prior and self.gram_prior_mode == "residual":
            clean_mean = torch.where(
                known_mask.bool().unsqueeze(-1),
                values + delta_or_value,
                gram_context_values + delta_or_value,
            )
        else:
            clean_mean = delta_or_value
        clean_mean = hermitian_symmetrize_complex_values(clean_mean, coords, token_mask)
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
        use_gram_prior: bool = False,
        gram_prior_mode: str = "feature",
        use_complex_features: bool = False,
        separate_denoising_head: bool = False,
        use_gram_attention_bias: bool = False,
        gram_attention_strength: float = 1.0,
        gram_image_half_width: float = math.sin(math.radians(4.0)),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if gram_prior_mode not in {"feature", "residual"}:
            raise ValueError(f"Unsupported gram_prior_mode: {gram_prior_mode!r}.")
        self.use_gram_prior = use_gram_prior
        self.gram_prior_mode = gram_prior_mode
        self.separate_denoising_head = separate_denoising_head
        self.use_gram_attention_bias = use_gram_attention_bias
        self.gram_attention_strength = gram_attention_strength
        self.gram_image_half_width = gram_image_half_width
        self.num_heads = num_heads
        value_dim = (5 if use_complex_features else 2) * (2 if use_gram_prior else 1)
        self.context_embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            value_dim=value_dim,
            use_complex_features=use_complex_features,
            dropout=dropout,
        )
        self.query_embed = VisibilityTokenEmbedder(
            coord_dim=coord_dim,
            model_dim=model_dim,
            redundancy_dim=redundancy_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            normalize_coords=normalize_coords,
            value_dim=value_dim,
            use_complex_features=use_complex_features,
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
        self.denoising_head = None
        if separate_denoising_head:
            self.denoising_head = nn.Sequential(
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
        gram_context_values: Optional[Tensor] = None,
    ) -> BVTOutput:
        if token_mask is None:
            token_mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        if self.use_gram_prior and gram_context_values is None:
            gram_context_values = torch.zeros_like(values)
        elif not self.use_gram_prior:
            gram_context_values = None

        context_mask = token_mask.bool() & known_mask.bool()
        context_values = values.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        context_gram_values = None
        if gram_context_values is not None:
            context_gram_values = gram_context_values.masked_fill(~token_mask.bool().unsqueeze(-1), 0.0)
        context_tokens = self.context_embed(context_values, coords, context_mask, redundancy, context_gram_values)
        memory_key_padding_mask = ~context_mask
        memory = self.encoder(context_tokens, src_key_padding_mask=memory_key_padding_mask)

        z_token, latent_mean, latent_logvar, prior_mean, prior_logvar = self.latent(memory, context_mask)
        memory = memory + z_token.unsqueeze(1)

        query_values = values.masked_fill(~known_mask.bool().unsqueeze(-1), 0.0)
        query_tokens = self.query_embed(query_values, coords, known_mask.bool(), redundancy, gram_context_values)
        memory_mask = None
        decoder_memory_key_padding_mask = memory_key_padding_mask
        if self.use_gram_attention_bias:
            gram_corr = fourier_column_correlation(
                coords[..., :2],
                coords[..., :2],
                self.gram_image_half_width,
            )
            memory_mask = self.gram_attention_strength * torch.log(gram_corr.clamp_min(1e-6))
            memory_mask = memory_mask.repeat_interleave(self.num_heads, dim=0)
            decoder_memory_key_padding_mask = torch.zeros_like(memory_key_padding_mask, dtype=memory.dtype)
            decoder_memory_key_padding_mask = decoder_memory_key_padding_mask.masked_fill(
                memory_key_padding_mask,
                float("-inf"),
            )
        decoded = self.decoder(
            query_tokens,
            memory,
            tgt_key_padding_mask=~token_mask.bool(),
            memory_key_padding_mask=decoder_memory_key_padding_mask,
            memory_mask=memory_mask,
        )

        expansion_pred = self.head(decoded)
        denoising_pred = self.denoising_head(memory) if self.denoising_head is not None else expansion_pred
        delta_or_value = expansion_pred[..., :2]
        query_baseline = (
            gram_context_values
            if self.use_gram_prior and self.gram_prior_mode == "residual"
            else torch.zeros_like(delta_or_value)
        )
        observed_mean = values + denoising_pred[..., :2]
        clean_mean = torch.where(known_mask.bool().unsqueeze(-1), observed_mean, query_baseline + delta_or_value)
        clean_mean = hermitian_symmetrize_complex_values(clean_mean, coords, token_mask)
        clean_logvar = torch.where(
            known_mask.bool().unsqueeze(-1),
            denoising_pred[..., 2:3],
            expansion_pred[..., 2:3],
        ).clamp(-12.0, 6.0)
        noise_logvar = torch.where(
            known_mask.bool().unsqueeze(-1),
            denoising_pred[..., 3:4],
            expansion_pred[..., 3:4],
        ).clamp(-12.0, 6.0)

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


def complex_amplitude_loss(pred_values: Tensor, target_values: Tensor, mask: Tensor) -> Tensor:
    """Relative L1 amplitude loss in a masked complex visibility region."""
    if mask.sum() == 0:
        return pred_values.new_zeros(())
    weights = mask.to(pred_values.dtype)
    pred_amp = torch.linalg.norm(pred_values, dim=-1)
    target_amp = torch.linalg.norm(target_values, dim=-1)
    numerator = ((pred_amp - target_amp).abs() * weights).sum()
    denominator = (target_amp * weights).sum().clamp_min(1e-8)
    return numerator / denominator


def complex_normalized_mse(
    pred_values: Tensor,
    target_values: Tensor,
    mask: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Complex MSE normalized by target complex power in a masked region."""
    if mask.sum() == 0:
        return pred_values.new_zeros(())
    weights = mask.to(pred_values.dtype)
    if weight is not None:
        weights = weights * weight.to(pred_values.dtype)
    component_weights = weights.unsqueeze(-1)
    diff_power = ((pred_values - target_values).pow(2) * component_weights).sum()
    target_power = (target_values.pow(2) * component_weights).sum().clamp_min(1e-8)
    return diff_power / target_power


def complex_correlation_loss(pred_values: Tensor, target_values: Tensor, mask: Tensor) -> Tensor:
    """One minus normalized real-vector correlation over complex visibility components."""
    if mask.sum() == 0:
        return pred_values.new_zeros(())
    weights = mask.unsqueeze(-1).to(pred_values.dtype)
    pred = pred_values * weights
    target = target_values * weights
    numerator = (pred * target).sum()
    denominator = pred.pow(2).sum().sqrt() * target.pow(2).sum().sqrt()
    return 1.0 - numerator / denominator.clamp_min(1e-8)


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


def uncertainty_calibration_loss(output: BVTOutput, target_values: Tensor, mask: Tensor) -> Tensor:
    """Match predicted complex variance to realized squared complex error."""
    if mask.sum() == 0:
        return output.clean_mean.new_zeros(())
    error_power = (output.clean_mean.detach() - target_values).pow(2).sum(dim=-1).clamp_min(1e-10)
    predicted_var = torch.exp(output.clean_logvar).squeeze(-1).clamp_min(1e-10)
    log_error = torch.log(error_power)
    log_var = torch.log(predicted_var)
    weights = mask.to(output.clean_mean.dtype)
    return ((log_var - log_error).pow(2) * weights).sum() / weights.sum().clamp_min(1.0)


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
    lambda_phase_expanded: float = 1.0,
    lambda_amp_all: float = 0.05,
    lambda_amp_expanded: float = 0.2,
    lambda_expanded_nmse: float = 0.5,
    lambda_expanded_corr: float = 0.3,
    lambda_uncertainty_calibration: float = 0.02,
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
    amp_all = complex_amplitude_loss(output.clean_mean, batch.target_values, batch.target_mask)
    amp_orig = complex_amplitude_loss(output.clean_mean, batch.target_values, batch.original_mask)
    amp_virtual = complex_amplitude_loss(output.clean_mean, batch.target_values, batch.virtual_mask)
    amp_expanded = complex_amplitude_loss(output.clean_mean, batch.target_values, expanded_mask)
    expanded_nmse = complex_normalized_mse(output.clean_mean, batch.target_values, expanded_mask)
    expanded_radial_nmse = complex_normalized_mse(
        output.clean_mean,
        batch.target_values,
        expanded_mask,
        weight=radial_bin_weights,
    )
    expanded_corr = complex_correlation_loss(output.clean_mean, batch.target_values, expanded_mask)
    uncertainty_cal = uncertainty_calibration_loss(output, batch.target_values, expanded_mask)
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
    amplitude = lambda_amp_all * amp_all + lambda_amp_expanded * amp_expanded
    sr_structure = lambda_expanded_nmse * expanded_radial_nmse + lambda_expanded_corr * expanded_corr
    phase_loss = lambda_phase * phase + lambda_phase_expanded * phase_expanded
    calibration = lambda_uncertainty_calibration * uncertainty_cal
    loss = (
        region_nll
        + lambda_sym * sym
        + energy
        + amplitude
        + sr_structure
        + phase_loss
        + calibration
        + beta_kl * kl
        + beta_noise_prior * noise_prior
    )
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
        "amplitude": amplitude.detach(),
        "amplitude_all": amp_all.detach(),
        "amplitude_original": amp_orig.detach(),
        "amplitude_virtual": amp_virtual.detach(),
        "amplitude_expanded_only": amp_expanded.detach(),
        "expanded_nmse_loss": expanded_nmse.detach(),
        "expanded_radial_nmse_loss": expanded_radial_nmse.detach(),
        "expanded_corr_loss": expanded_corr.detach(),
        "sr_structure": sr_structure.detach(),
        "phase": phase.detach(),
        "phase_original": phase_orig.detach(),
        "phase_virtual": phase_virtual.detach(),
        "phase_expanded_only": phase_expanded.detach(),
        "phase_loss": phase_loss.detach(),
        "uncertainty_calibration": uncertainty_cal.detach(),
        "calibration": calibration.detach(),
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


def _single_visibility_array(x: Tensor | object, name: str) -> Tensor:
    x = _visibility_array_to_batch_last(x, name)
    if x.shape[0] != 1:
        raise ValueError(f"{name} must describe one scene, got batch size {x.shape[0]}.")
    return x.squeeze(0)


def _uv_key(coord: Tensor, decimals: int = 5) -> tuple[float, float]:
    return (round(float(coord[0]), decimals), round(float(coord[1]), decimals))


def align_visibility_to_target_uv(
    target_uv: Tensor | object,
    source_uv: Tensor | object,
    source_visibility: Tensor | object,
    target_original_mask: Tensor | object,
) -> Tensor:
    """Place source visibility values on matching target uv positions.

    ``target_original_mask`` should mark the target-grid tokens that correspond
    to the observed source support, usually redundancy column 0.
    """
    target_coords = _single_visibility_array(target_uv, "target_uv")
    source_coords = _single_visibility_array(source_uv, "source_uv")
    source_values = _single_visibility_array(source_visibility, "source_visibility")
    original_mask = torch.as_tensor(target_original_mask, dtype=torch.bool)
    if original_mask.ndim == 2:
        if original_mask.shape[0] != 1:
            raise ValueError(f"target_original_mask must describe one scene, got shape {tuple(original_mask.shape)}.")
        original_mask = original_mask.squeeze(0)

    if source_coords.shape[:1] != source_values.shape[:1]:
        raise ValueError("source_uv and source_visibility must have the same token count.")
    if target_coords.shape[0] != original_mask.shape[0]:
        raise ValueError("target_uv and target_original_mask must have the same token count.")

    target_indices = torch.where(original_mask)[0]
    aligned = torch.zeros_like(target_coords)
    if source_coords.shape[0] == target_indices.numel():
        ordered_target_coords = target_coords[target_indices]
        max_order_delta = (source_coords - ordered_target_coords).abs().amax()
        # Some srdata uv files regenerate the same original support with tiny
        # trig/rounding differences, so exact coordinate keys are too brittle.
        if max_order_delta <= 1e-2:
            aligned[target_indices] = source_values
            return aligned

    source_by_uv = {_uv_key(coord): source_values[index] for index, coord in enumerate(source_coords)}
    missing = 0
    for target_index in target_indices.tolist():
        value = source_by_uv.get(_uv_key(target_coords[target_index]))
        if value is None:
            missing += 1
            continue
        aligned[target_index] = value
    if missing:
        raise ValueError(f"Could not align {missing} source uv points onto the target expansion grid.")
    return aligned


def align_source_visibility_to_target_uv(
    target_uv: Tensor | object,
    source_uv: Tensor | object,
    source_visibility: Tensor | object,
    tolerance: float = 1e-2,
) -> tuple[Tensor, Tensor]:
    """Align a lower-expansion observation onto a higher-expansion uv grid."""
    target_coords = _single_visibility_array(target_uv, "target_uv")
    source_coords = _single_visibility_array(source_uv, "source_uv")
    source_values = _single_visibility_array(source_visibility, "source_visibility")
    if source_coords.shape[0] != source_values.shape[0]:
        raise ValueError("source_uv and source_visibility must have the same token count.")

    distance = torch.cdist(source_coords[:, :2], target_coords[:, :2])
    min_distance, target_indices = distance.min(dim=1)
    if (min_distance > tolerance).any():
        missing = int((min_distance > tolerance).sum())
        raise ValueError(f"Could not align {missing} source uv points onto the target expansion grid.")

    aligned = torch.zeros_like(target_coords)
    observed_mask = torch.zeros(target_coords.shape[0], dtype=torch.bool)
    aligned[target_indices] = source_values
    observed_mask[target_indices] = True
    return aligned, observed_mask


def _compute_visibility_scale(
    input_values: Tensor,
    original_mask: Tensor,
    visibility_normalization: str,
) -> Tensor:
    if visibility_normalization == "none":
        return input_values.new_ones((*input_values.shape[:1], 1, 1))
    if visibility_normalization != "original-rms":
        raise ValueError(f"Unsupported visibility normalization: {visibility_normalization!r}.")

    weights = original_mask.to(input_values.dtype)
    complex_power = input_values.pow(2).sum(dim=-1)
    scale = ((complex_power * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).sqrt()
    return scale.clamp_min(1e-8).view(-1, 1, 1)


def fourier_column_correlation(
    query_uv: Tensor,
    source_uv: Tensor,
    image_half_width: float = math.sin(math.radians(4.0)),
) -> Tensor:
    """Continuous rectangular-FOV normalized Fourier-column correlation.

    For image coordinates in ``[-image_half_width, image_half_width]`` along
    both axes, the normalized Gram magnitude factorizes into sinc terms:
    ``|sinc(2L du) sinc(2L dv)|``.
    """
    delta = query_uv.unsqueeze(-2) - source_uv.unsqueeze(-3)
    corr_u = torch.sinc(2.0 * image_half_width * delta[..., 0])
    corr_v = torch.sinc(2.0 * image_half_width * delta[..., 1])
    return (corr_u * corr_v).abs()


def gram_neighbor_context_values(
    coords: Tensor,
    values: Tensor,
    source_mask: Tensor,
    token_mask: Tensor,
    top_k: int = 32,
    image_half_width: float = math.sin(math.radians(4.0)),
    min_corr: float = 0.0,
) -> Tensor:
    """Top-k PSF/Gram-weighted visibility context for every uv token.

    The aggregation is a soft physical prior, not a target. Source visibility
    comes only from currently visible tokens, and exact self-neighbors are
    excluded so original tokens cannot simply copy their noisy measurement.
    """
    if top_k <= 0:
        return torch.zeros_like(values)

    coords = coords.to(values.dtype)
    source_mask = source_mask.bool() & token_mask.bool()
    query_mask = token_mask.bool()
    corr = fourier_column_correlation(coords[..., :2], coords[..., :2], image_half_width)

    valid_pair = query_mask.unsqueeze(-1) & source_mask.unsqueeze(-2)
    same_point = torch.cdist(coords[..., :2], coords[..., :2]) <= 1e-6
    valid_pair = valid_pair & ~same_point
    if min_corr > 0.0:
        valid_pair = valid_pair & (corr >= min_corr)
    corr = corr.masked_fill(~valid_pair, 0.0)

    k = min(top_k, corr.shape[-1])
    top_values, top_indices = corr.topk(k=k, dim=-1)
    source_values = values.unsqueeze(1).expand(-1, values.shape[1], -1, -1)
    gathered_values = source_values.gather(
        dim=2,
        index=top_indices.unsqueeze(-1).expand(*top_indices.shape, values.shape[-1]),
    )
    weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (gathered_values * weights.unsqueeze(-1)).sum(dim=-2)


def build_visibility_region_inputs(
    uv: Tensor | object,
    visibility: Tensor | object,
    redu: Tensor | object,
    valid_mask: Optional[Tensor | object] = None,
    target_visibility: Optional[Tensor | object] = None,
    include_virtual_context: bool = False,
    visibility_normalization: str = "none",
    use_gram_prior: bool = False,
    gram_top_k: int = 32,
    gram_image_half_width: float = math.sin(math.radians(4.0)),
    gram_min_corr: float = 0.0,
    observed_support_mask: Optional[Tensor | object] = None,
    virtual_support_mask: Optional[Tensor | object] = None,
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
        use_gram_prior: If True, compute a top-k PSF/Gram-weighted visibility
            context from known tokens for every uv query.

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
    if observed_support_mask is not None:
        original_mask = torch.as_tensor(observed_support_mask, dtype=torch.bool, device=coords.device)
        if original_mask.ndim == 1:
            original_mask = original_mask.unsqueeze(0)
        if original_mask.shape != coords.shape[:2]:
            raise ValueError(
                f"observed_support_mask must have shape {tuple(coords.shape[:2])}, got {tuple(original_mask.shape)}."
            )
    virtual_mask = redundancy[..., 1] > 0
    if virtual_support_mask is not None:
        virtual_mask = torch.as_tensor(virtual_support_mask, dtype=torch.bool, device=coords.device)
        if virtual_mask.ndim == 1:
            virtual_mask = virtual_mask.unsqueeze(0)
        if virtual_mask.shape != coords.shape[:2]:
            raise ValueError(
                f"virtual_support_mask must have shape {tuple(coords.shape[:2])}, got {tuple(virtual_mask.shape)}."
            )

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

    visibility_scale = _compute_visibility_scale(input_values, original_mask, visibility_normalization)
    input_values = input_values / visibility_scale
    target_values = target_values / visibility_scale
    values = input_values.masked_fill(~known_mask.unsqueeze(-1), 0.0)
    if use_gram_prior:
        gram_context_values = gram_neighbor_context_values(
            coords,
            values,
            known_mask,
            token_mask,
            top_k=gram_top_k,
            image_half_width=gram_image_half_width,
            min_corr=gram_min_corr,
        )
    else:
        gram_context_values = torch.zeros_like(values)

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
        visibility_scale=visibility_scale,
        gram_context_values=gram_context_values,
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
    context_expand_id: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load one simulated scene/expand sample.

    Returns ``uv``, input ``visibility``, ``redundant``, and target visibility,
    all with the current [V, 2] layout. Datasets may store uv/redundancy either
    per scene or once per expansion as ``uv_expand_#.npy``. If redundancy is not
    provided, every token is treated as both observed and supervised, which is
    appropriate for denoising-only grids with no explicit expanded-only region.
    """
    uv, visibility, redundancy, target_visibility, _ = _load_srdata_arrays_with_observed_mask(
        root,
        scene_id,
        expand_id,
        input_suffix=input_suffix,
        target_suffix=target_suffix,
        context_expand_id=context_expand_id,
    )
    return uv, visibility, redundancy, target_visibility


def _load_srdata_arrays_with_observed_mask(
    root: str | Path,
    scene_id: int,
    expand_id: int,
    input_suffix: str = "unnoised",
    target_suffix: Optional[str] = None,
    context_expand_id: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    target_suffix = input_suffix if target_suffix is None else target_suffix
    uv = _load_npy_from_candidates(root, "uv", scene_id, expand_id, input_suffix)
    redundancy_path = _first_existing_path(
        _srdata_candidate_paths(root, "redundant", scene_id, expand_id, input_suffix)
    )
    target_visibility = _load_npy(srdata_path(root, "visibility", scene_id, expand_id, target_suffix))
    redundancy = (
        _load_npy(redundancy_path)
        if redundancy_path is not None
        else _synthetic_redundancy_like(target_visibility)
    )
    if context_expand_id is None or context_expand_id == expand_id:
        visibility = _load_npy(srdata_path(root, "visibility", scene_id, expand_id, input_suffix))
        observed_mask = _visibility_array_to_batch_last(redundancy, "redu").squeeze(0)[:, 0] > 0
    else:
        source_uv = _load_npy_from_candidates(root, "uv", scene_id, context_expand_id, input_suffix)
        source_visibility = _load_npy(srdata_path(root, "visibility", scene_id, context_expand_id, input_suffix))
        visibility, observed_mask = align_source_visibility_to_target_uv(
            uv,
            source_uv,
            source_visibility,
        )
    return uv, visibility, redundancy, target_visibility, observed_mask


def _load_cross_expansion_arrays(
    root: str | Path,
    scene_id: int,
    source_expand_id: int,
    target_expand_id: int,
    input_suffix: str,
    target_suffix: Optional[str],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build separate source-context and target-query tokens on one union grid.

    Expansion uv arrays need not be nested. Concatenating their supports keeps
    source visibility at its own physical coordinates and lets the decoder
    query the target expansion at its distinct coordinates.
    """
    clean_suffix = input_suffix if target_suffix is None else target_suffix
    source_uv = _single_visibility_array(
        _load_npy_from_candidates(root, "uv", scene_id, source_expand_id, input_suffix),
        "source_uv",
    )
    source_values = _single_visibility_array(
        _load_npy(srdata_path(root, "visibility", scene_id, source_expand_id, input_suffix)),
        "source_visibility",
    )
    source_target = _single_visibility_array(
        _load_npy(srdata_path(root, "visibility", scene_id, source_expand_id, clean_suffix)),
        "source_target_visibility",
    )
    source_redundancy_path = _first_existing_path(
        _srdata_candidate_paths(root, "redundant", scene_id, source_expand_id, input_suffix)
    )
    source_redundancy = _single_visibility_array(
        _load_npy(source_redundancy_path)
        if source_redundancy_path is not None
        else _synthetic_redundancy_like(source_target),
        "source_redundancy",
    )

    target_uv = _single_visibility_array(
        _load_npy_from_candidates(root, "uv", scene_id, target_expand_id, input_suffix),
        "target_uv",
    )
    target_target = _single_visibility_array(
        _load_npy(srdata_path(root, "visibility", scene_id, target_expand_id, clean_suffix)),
        "target_visibility",
    )
    target_redundancy_path = _first_existing_path(
        _srdata_candidate_paths(root, "redundant", scene_id, target_expand_id, input_suffix)
    )
    target_redundancy = _single_visibility_array(
        _load_npy(target_redundancy_path)
        if target_redundancy_path is not None
        else _synthetic_redundancy_like(target_target),
        "target_redundancy",
    )

    source_count = source_uv.shape[0]
    target_count = target_uv.shape[0]
    return (
        torch.cat([source_uv, target_uv], dim=0),
        torch.cat([source_values, torch.zeros_like(target_uv)], dim=0),
        torch.cat([source_redundancy, target_redundancy], dim=0),
        torch.cat([source_target, target_target], dim=0),
        torch.cat([torch.ones(source_count, dtype=torch.bool), torch.zeros(target_count, dtype=torch.bool)]),
        torch.cat([torch.zeros(source_count, dtype=torch.bool), torch.ones(target_count, dtype=torch.bool)]),
    )


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
        context_expand_id: Optional[int] = None,
        visibility_normalization: str = "none",
        use_gram_prior: bool = False,
        gram_top_k: int = 32,
        gram_image_half_width: float = math.sin(math.radians(4.0)),
        gram_min_corr: float = 0.0,
        cross_expansion: bool = False,
        curriculum_epochs: int = 30,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.input_suffix = input_suffix
        self.target_suffix = target_suffix
        self.include_virtual_context = include_virtual_context
        self.context_expand_id = context_expand_id
        self.visibility_normalization = visibility_normalization
        self.use_gram_prior = use_gram_prior
        self.gram_top_k = gram_top_k
        self.gram_image_half_width = gram_image_half_width
        self.gram_min_corr = gram_min_corr
        self.cross_expansion = cross_expansion
        self.curriculum_epochs = max(int(curriculum_epochs), 1)
        self.sampling_seed = int(sampling_seed)
        self.epoch = 0

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
            if context_expand_id is not None:
                source_uv_path = _first_existing_path(
                    _srdata_candidate_paths(self.root, "uv", scene_id, context_expand_id, input_suffix)
                )
                required.extend([
                    source_uv_path,
                    srdata_path(self.root, "visibility", scene_id, context_expand_id, input_suffix),
                ])
            if target_suffix is not None:
                required.append(srdata_path(self.root, "visibility", scene_id, expand_id, target_suffix))
            if all(item is not None and item.exists() for item in required):
                samples.append((scene_id, expand_id))

        available_by_scene: dict[int, set[int]] = {}
        for scene_id, expand_id in samples:
            available_by_scene.setdefault(scene_id, set()).add(expand_id)
        if cross_expansion:
            samples = [
                (scene_id, target_expand_id)
                for scene_id, target_expand_id in samples
                if any(source_expand_id < target_expand_id for source_expand_id in available_by_scene[scene_id])
            ]
        self.available_expand_ids = {
            scene_id: sorted(expand_ids_for_scene)
            for scene_id, expand_ids_for_scene in available_by_scene.items()
        }
        self.samples = sorted(samples)
        if not self.samples:
            raise ValueError(f"No srdata samples found in {self.root} for suffix {input_suffix!r}.")
        expand_levels = sorted({item[1] for item in self.samples})
        self.expand_rank = {expand_id: rank + 1 for rank, expand_id in enumerate(expand_levels)}

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(int(epoch), 0)

    def sample_weight(self, index: int, power: float = 1.0) -> float:
        """Favor samples with wider target expansion and longer uv coverage."""
        _, expand_id = self.samples[index]
        return float(self.expand_rank[expand_id] ** max(power, 0.0))

    def _source_expand_id(self, index: int, scene_id: int, target_expand_id: int) -> Optional[int]:
        if not self.cross_expansion:
            return self.context_expand_id
        candidates = [item for item in self.available_expand_ids[scene_id] if item < target_expand_id]
        max_possible_gap = target_expand_id - min(candidates)
        progress = min((self.epoch + 1) / self.curriculum_epochs, 1.0)
        max_gap = max(1, math.ceil(progress * max_possible_gap))
        candidates = [item for item in candidates if target_expand_id - item <= max_gap]
        generator = torch.Generator().manual_seed(
            self.sampling_seed + self.epoch * 1_000_003 + index * 97 + scene_id
        )
        return candidates[int(torch.randint(len(candidates), (1,), generator=generator))]

    def __getitem__(self, index: int) -> VisibilityRegionInput:
        scene_id, expand_id = self.samples[index]
        context_expand_id = self._source_expand_id(index, scene_id, expand_id)
        virtual_mask = None
        if self.cross_expansion:
            if context_expand_id is None or context_expand_id >= expand_id:
                raise RuntimeError("Cross-expansion samples require a lower source expansion.")
            try:
                # Prefer the target grid when source coordinates are an exact
                # subset, such as expand_3 -> expand_8.
                uv, visibility, redundancy, target_visibility, observed_mask = _load_srdata_arrays_with_observed_mask(
                    self.root,
                    scene_id,
                    expand_id,
                    input_suffix=self.input_suffix,
                    target_suffix=self.target_suffix,
                    context_expand_id=context_expand_id,
                )
            except ValueError:
                # Some lower expansions cover the same uv region on a shifted
                # grid. Preserve their source coordinates as separate context
                # tokens instead of forcing an invalid pointwise alignment.
                uv, visibility, redundancy, target_visibility, observed_mask, virtual_mask = _load_cross_expansion_arrays(
                    self.root,
                    scene_id,
                    context_expand_id,
                    expand_id,
                    input_suffix=self.input_suffix,
                    target_suffix=self.target_suffix,
                )
        else:
            uv, visibility, redundancy, target_visibility, observed_mask = _load_srdata_arrays_with_observed_mask(
                self.root,
                scene_id,
                expand_id,
                input_suffix=self.input_suffix,
                target_suffix=self.target_suffix,
                context_expand_id=context_expand_id,
            )
        return build_visibility_region_inputs(
            uv,
            visibility,
            redundancy,
            target_visibility=target_visibility,
            include_virtual_context=self.include_virtual_context,
            visibility_normalization=self.visibility_normalization,
            use_gram_prior=self.use_gram_prior,
            gram_top_k=self.gram_top_k,
            gram_image_half_width=self.gram_image_half_width,
            gram_min_corr=self.gram_min_corr,
            observed_support_mask=observed_mask,
            virtual_support_mask=virtual_mask,
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
        visibility_scale=torch.stack([sample.visibility_scale.squeeze(0) for sample in samples], dim=0),
        gram_context_values=torch.stack([pad_last(sample.gram_context_values) for sample in samples], dim=0),
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
