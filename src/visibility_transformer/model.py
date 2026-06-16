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
    clean_logvar:    [batch, n_token, 2]
    noise_logvar:    [batch, n_token, 2]

For noisy training targets, use ``training_objective(..., target_is_noisy=True)``.
The likelihood then marginalizes clean uncertainty and inferred label noise:
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


def gaussian_nll(target: Tensor, mean: Tensor, logvar: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Masked diagonal Gaussian negative log likelihood for complex visibility."""
    logvar = logvar.clamp(-14.0, 8.0)
    loss = 0.5 * (math.log(2.0 * math.pi) + logvar + (target - mean).pow(2) * torch.exp(-logvar))
    loss = loss.sum(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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
        self.prior = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2 * latent_dim),
        )
        self.to_model = nn.Linear(latent_dim, model_dim)

    def forward(self, tokens: Tensor, token_mask: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pooled = masked_mean(tokens, token_mask, dim=1)
        latent_mean, latent_logvar = self.posterior(pooled).chunk(2, dim=-1)
        prior_mean, prior_logvar = self.prior(pooled.detach()).chunk(2, dim=-1)

        latent_logvar = latent_logvar.clamp(-10.0, 5.0)
        prior_logvar = prior_logvar.clamp(-10.0, 5.0)
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
            nn.Linear(model_dim, 6),
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
        clean_logvar = pred[..., 2:4].clamp(-12.0, 6.0)
        noise_logvar = pred[..., 4:6].clamp(-12.0, 6.0)

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
    target_is_noisy: bool = True,
    beta_kl: float = 1e-3,
    beta_noise_prior: float = 1e-4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Default objective for noisy-label denoising and virtual-array SR.

    If expanded virtual visibility labels are noisy, keep ``target_is_noisy`` as
    True. If you later have clean simulated targets, set it to False so the
    clean posterior is trained directly.
    """

    likelihood_logvar = output.total_logvar if target_is_noisy else output.clean_logvar
    nll = gaussian_nll(target_values, output.clean_mean, likelihood_logvar, target_mask)
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


def _load_npy(path: str | Path) -> Tensor:
    import numpy as np

    return torch.from_numpy(np.load(path)).to(torch.float32)


def _load_npy_with_unnoised_fallback(
    root: str | Path,
    kind: str,
    scene_id: int,
    expand_id: int,
    suffix: str,
) -> Tensor:
    path = srdata_path(root, kind, scene_id, expand_id, suffix)
    if not path.exists() and suffix != "unnoised":
        path = srdata_path(root, kind, scene_id, expand_id, "unnoised")
    return _load_npy(path)


def load_srdata_arrays(
    root: str | Path,
    scene_id: int,
    expand_id: int,
    input_suffix: str = "unnoised",
    target_suffix: Optional[str] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load one simulated scene/expand sample.

    Returns ``uv``, input ``visibility``, ``redundant``, and target visibility,
    all with the current [V, 2] layout. When fixed-noise files arrive, use
    ``input_suffix="noised_1", target_suffix="unnoised"`` for clean supervised
    denoising, or leave ``target_suffix=None`` for noisy-label training.
    """
    target_suffix = input_suffix if target_suffix is None else target_suffix
    uv = _load_npy_with_unnoised_fallback(root, "uv", scene_id, expand_id, input_suffix)
    visibility = _load_npy(srdata_path(root, "visibility", scene_id, expand_id, input_suffix))
    redundancy = _load_npy_with_unnoised_fallback(root, "redundant", scene_id, expand_id, input_suffix)
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

            required = [
                srdata_path(self.root, "uv", scene_id, expand_id, input_suffix)
                if srdata_path(self.root, "uv", scene_id, expand_id, input_suffix).exists()
                else srdata_path(self.root, "uv", scene_id, expand_id, "unnoised"),
                srdata_path(self.root, "redundant", scene_id, expand_id, input_suffix)
                if srdata_path(self.root, "redundant", scene_id, expand_id, input_suffix).exists()
                else srdata_path(self.root, "redundant", scene_id, expand_id, "unnoised"),
            ]
            if target_suffix is not None:
                required.append(srdata_path(self.root, "visibility", scene_id, expand_id, target_suffix))
            if all(item.exists() for item in required):
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
