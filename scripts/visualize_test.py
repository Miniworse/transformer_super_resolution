"""Visualize matched uv visibility and image-domain test predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import compute_metrics, move_batch  # noqa: E402
from visibility_transformer import (  # noqa: E402
    BayesianVisibilityEncoderDecoder,
    BayesianVisibilityTransformer,
    SRVisibilityDataset,
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            "Falling back to weights_only=False for a legacy trusted checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def create_model_from_args(ckpt_args: dict):
    common = {
        "coord_dim": 2,
        "redundancy_dim": 2,
        "model_dim": int(ckpt_args.get("model_dim", 256)),
        "latent_dim": int(ckpt_args.get("latent_dim", 64)),
        "num_heads": int(ckpt_args.get("num_heads", 8)),
        "num_frequencies": int(ckpt_args.get("num_frequencies", 16)),
        "normalize_coords": bool(ckpt_args.get("normalize_coords", False)),
        "dropout": float(ckpt_args.get("dropout", 0.1)),
    }
    architecture = ckpt_args.get("architecture", "encoder")
    if architecture == "encoder-decoder":
        return BayesianVisibilityEncoderDecoder(
            num_encoder_layers=int(ckpt_args.get("num_encoder_layers", 6)),
            num_decoder_layers=int(ckpt_args.get("num_decoder_layers", 4)),
            **common,
        )
    return BayesianVisibilityTransformer(num_layers=int(ckpt_args.get("num_layers", 8)), **common)


def complex_from_ri(values: Tensor) -> np.ndarray:
    array = values.detach().cpu().numpy()
    return array[..., 0] + 1j * array[..., 1]


def mask_to_numpy(mask: Tensor) -> np.ndarray:
    return mask.detach().cpu().numpy().astype(bool)


def visibility_metrics(pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    if mask.sum().item() == 0:
        return {"mse": math.nan, "rmse": math.nan, "mae": math.nan, "nmse_db": math.nan}
    diff = pred[mask] - target[mask]
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    power = target[mask].pow(2).mean().clamp_min(1e-12)
    nmse = mse / power
    return {
        "mse": float(mse.cpu()),
        "rmse": float(torch.sqrt(mse).cpu()),
        "mae": float(mae.cpu()),
        "nmse_db": float((10.0 * torch.log10(nmse)).cpu()),
    }


def inverse_visibility_image(
    uv: np.ndarray,
    visibility: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
    weights: np.ndarray | None = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """Direct irregular inverse Fourier visualization on the xi/eta grid.

    This is only for visualization and is not used by the model.
    """
    if weights is None:
        weights = np.ones(len(visibility), dtype=np.float64)
    weights = weights.astype(np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    weighted_visibility = visibility * weights

    xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="xy")
    flat_xi = xi_grid.reshape(-1)
    flat_eta = eta_grid.reshape(-1)
    image = np.empty(flat_xi.shape, dtype=np.complex128)

    u = uv[:, 0].astype(np.float64)
    v = uv[:, 1].astype(np.float64)
    for start in range(0, flat_xi.size, chunk_size):
        stop = min(start + chunk_size, flat_xi.size)
        phase = 2.0j * np.pi * (
            np.outer(flat_xi[start:stop], u)
            + np.outer(flat_eta[start:stop], v)
        )
        image[start:stop] = np.exp(phase) @ weighted_visibility

    return image.reshape(len(eta), len(xi)).real


def normalize_image_limits(*images: np.ndarray) -> tuple[float, float]:
    stacked = np.concatenate([image.reshape(-1) for image in images])
    low, high = np.percentile(stacked, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(stacked)), float(np.nanmax(stacked))
    return float(low), float(high)


def plot_visualization(
    output_path: Path,
    coords: Tensor,
    input_values: Tensor,
    pred_values: Tensor,
    target_values: Tensor,
    original_mask: Tensor,
    target_mask: Tensor,
    image_mask: Tensor,
    redundancy: Tensor,
    scene_id: int,
    expand_id: int,
    image_size: int,
    chunk_size: int,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    coords_np = coords.detach().cpu().numpy()
    original_np = mask_to_numpy(original_mask)
    target_np = mask_to_numpy(target_mask)
    image_np = mask_to_numpy(image_mask)

    input_complex = complex_from_ri(input_values)
    pred_complex = complex_from_ri(pred_values)
    target_complex = complex_from_ri(target_values)

    pred_error = np.abs(pred_complex - target_complex)
    xi_eta_extent = math.sin(math.radians(4.0))
    xi = np.linspace(-xi_eta_extent, xi_eta_extent, image_size)
    eta = np.linspace(-xi_eta_extent, xi_eta_extent, image_size)

    original_image = inverse_visibility_image(
        coords_np[original_np],
        input_complex[original_np],
        xi,
        eta,
        weights=redundancy.detach().cpu().numpy()[original_np, 0],
        chunk_size=chunk_size,
    )
    pred_image = inverse_visibility_image(
        coords_np[image_np],
        pred_complex[image_np],
        xi,
        eta,
        weights=redundancy.detach().cpu().numpy()[image_np].max(axis=1),
        chunk_size=chunk_size,
    )
    target_image = inverse_visibility_image(
        coords_np[image_np],
        target_complex[image_np],
        xi,
        eta,
        weights=redundancy.detach().cpu().numpy()[image_np].max(axis=1),
        chunk_size=chunk_size,
    )
    image_error = np.abs(pred_image - target_image)

    vis_vmax = np.percentile(
        np.concatenate([np.abs(input_complex[original_np]), np.abs(pred_complex[target_np]), np.abs(target_complex[target_np])]),
        99.0,
    )
    err_vmax = np.percentile(pred_error[target_np], 99.0)
    img_vmin, img_vmax = normalize_image_limits(original_image, pred_image, target_image)
    img_err_vmax = np.percentile(image_error, 99.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    fig.suptitle(f"Scene {scene_id:04d}, expand_{expand_id}: matched uv visibility and image view", fontsize=13)

    scatter_panels = [
        ("original input |V|", original_np, np.abs(input_complex), "viridis", 0.0, vis_vmax),
        ("predicted clean |V|", target_np, np.abs(pred_complex), "viridis", 0.0, vis_vmax),
        ("ideal target |V|", target_np, np.abs(target_complex), "viridis", 0.0, vis_vmax),
        ("visibility error |dV|", target_np, pred_error, "magma", 0.0, err_vmax),
    ]
    for ax, (title, mask, color, cmap, vmin, vmax) in zip(axes[0], scatter_panels):
        sc = ax.scatter(coords_np[mask, 0], coords_np[mask, 1], c=color[mask], s=6, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)

    image_panels = [
        ("image from original input", original_image, "coolwarm", img_vmin, img_vmax),
        ("image from predicted expanded", pred_image, "coolwarm", img_vmin, img_vmax),
        ("image from ideal expanded", target_image, "coolwarm", img_vmin, img_vmax),
        ("image abs error", image_error, "magma", 0.0, img_err_vmax),
    ]
    extent = [xi[0], xi[-1], eta[0], eta[-1]]
    for ax, (title, image, cmap, vmin, vmax) in zip(axes[1], image_panels):
        im = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("xi")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    image_mse = float(np.mean((pred_image - target_image) ** 2))
    image_mae = float(np.mean(np.abs(pred_image - target_image)))
    return {
        "image_mse": image_mse,
        "image_mae": image_mae,
        "image_rmse": float(math.sqrt(image_mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "bvt_v1" / "checkpoints" / "best.pt")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-id", type=int, default=91)
    parser.add_argument("--expand-id", type=int, default=4)
    parser.add_argument("--input-suffix", default=None)
    parser.add_argument("--target-suffix", default=None)
    parser.add_argument("--include-virtual-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-expand-id", type=int, default=None)
    parser.add_argument("--visibility-normalization", choices=["none", "original-rms"], default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    ckpt_args = checkpoint.get("args", {})
    data_root = args.data_root or Path(ckpt_args.get("data_root", PROJECT_ROOT / "srdata" / "srdata16Juin" / "interval5_AMtown01"))
    output_dir = args.output_dir or args.checkpoint.parent.parent / "visualizations"
    input_suffix = args.input_suffix or ckpt_args.get("input_suffix", "unnoised")
    target_suffix = args.target_suffix if args.target_suffix is not None else ckpt_args.get("target_suffix")
    context_expand_id = (
        args.context_expand_id
        if args.context_expand_id is not None
        else ckpt_args.get("context_expand_id")
    )
    visibility_normalization = args.visibility_normalization or ckpt_args.get("visibility_normalization", "none")
    include_virtual_context = (
        bool(ckpt_args.get("include_virtual_context", False))
        if args.include_virtual_context is None
        else args.include_virtual_context
    )

    dataset = SRVisibilityDataset(
        data_root,
        scene_ids=[args.scene_id],
        expand_ids=[args.expand_id],
        input_suffix=input_suffix,
        target_suffix=target_suffix,
        include_virtual_context=include_virtual_context,
        context_expand_id=None if context_expand_id is None else int(context_expand_id),
        visibility_normalization=visibility_normalization,
    )
    batch = dataset[0]

    model = create_model_from_args(ckpt_args)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        device_batch = move_batch(batch, device)
        output = model(
            device_batch.values,
            device_batch.coords,
            device_batch.known_mask,
            device_batch.redundancy,
            device_batch.token_mask,
        )

    batch_cpu = move_batch(device_batch, torch.device("cpu"))
    pred_cpu = output.clean_mean.detach().cpu()[0]
    scale_cpu = batch_cpu.visibility_scale[0]
    input_values = batch_cpu.values[0] * scale_cpu
    pred_values = pred_cpu * scale_cpu
    target_values = batch_cpu.target_values[0] * scale_cpu
    image_mask = batch_cpu.virtual_mask[0] | batch_cpu.original_mask[0]
    output_path = output_dir / f"scene_{args.scene_id:04d}_expand_{args.expand_id}_comparison.png"

    image_metrics = plot_visualization(
        output_path,
        batch_cpu.coords[0],
        input_values,
        pred_values,
        target_values,
        batch_cpu.original_mask[0],
        batch_cpu.target_mask[0],
        image_mask,
        batch_cpu.redundancy[0],
        args.scene_id,
        args.expand_id,
        args.image_size,
        args.chunk_size,
    )

    metrics = compute_metrics(batch_cpu, pred_cpu.unsqueeze(0))
    metrics.update({f"image/{key}": value for key, value in image_metrics.items()})
    metrics_path = output_path.with_suffix(".json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps({"figure": str(output_path), "metrics": str(metrics_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
