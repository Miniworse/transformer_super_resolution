"""Visualize true and predicted visibility noise for one held-out sample."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import move_batch  # noqa: E402
from visualize_test import create_model_from_args  # noqa: E402
from visibility_transformer import SRVisibilityDataset  # noqa: E402


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


def as_complex(values: np.ndarray) -> np.ndarray:
    return values[..., 0] + 1j * values[..., 1]


def unwrap_phase(values: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(values))


def wrapped_phase(values: np.ndarray) -> np.ndarray:
    return np.angle(values)


def symmetric_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    stacked = np.concatenate([np.ravel(array) for array in arrays])
    limit = float(np.nanpercentile(np.abs(stacked), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(np.abs(stacked))) if stacked.size else 1.0
    return max(limit, 1e-8)


def positive_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    stacked = np.concatenate([np.ravel(array) for array in arrays])
    limit = float(np.nanpercentile(stacked, percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(stacked)) if stacked.size else 1.0
    return max(limit, 1e-8)


def component_corr(pred: np.ndarray, target: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    denom = np.linalg.norm(pred_flat) * np.linalg.norm(target_flat)
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(pred_flat, target_flat) / denom)


def complex_phase_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.angle(pred) - np.angle(target))))


def amplitude_weighted_mean_abs(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return math.nan
    return float(np.sum(np.abs(values) * weights) / denom)


def plot_noise_lines(
    output_path: Path,
    coords: np.ndarray,
    true_noise: np.ndarray,
    pred_noise: np.ndarray,
    scene_id: int,
    expand_id: int,
) -> None:
    import matplotlib.pyplot as plt

    radius = np.linalg.norm(coords, axis=1)
    order = np.argsort(radius)
    x = np.arange(order.size)
    radius_sorted = radius[order]
    true_sorted = true_noise[order]
    pred_sorted = pred_noise[order]
    error_sorted = pred_sorted - true_sorted

    true_complex = as_complex(true_sorted)
    pred_complex = as_complex(pred_sorted)
    error_complex = pred_complex - true_complex

    panels = [
        ("real(delta V)", true_sorted[:, 0], pred_sorted[:, 0], error_sorted[:, 0], "component"),
        ("imag(delta V)", true_sorted[:, 1], pred_sorted[:, 1], error_sorted[:, 1], "component"),
        ("amp(delta V)", np.abs(true_complex), np.abs(pred_complex), np.abs(error_complex), "amplitude"),
        ("phase(delta V)", unwrap_phase(true_complex), unwrap_phase(pred_complex), complex_phase_error(pred_complex, true_complex), "phase"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Scene {scene_id:04d}, expand_{expand_id}: true vs predicted noise, sorted by uv radius",
        fontsize=13,
    )

    for ax, (title, true_values, pred_values, err_values, scale_kind) in zip(axes[:4], panels):
        ax.plot(x, true_values, color="#1f77b4", linewidth=1.0, label="ground truth delta V")
        ax.plot(x, pred_values, color="#d62728", linewidth=1.0, alpha=0.85, label="predicted delta V")
        ax.plot(x, err_values, color="#2ca02c", linewidth=0.8, alpha=0.75, label="prediction error")
        if scale_kind == "component":
            limit = symmetric_limit(true_values, pred_values, err_values)
            ax.set_ylim(-limit, limit)
        elif scale_kind == "amplitude":
            limit = positive_limit(true_values, pred_values, err_values)
            ax.set_ylim(0.0, limit)
        else:
            ax.set_ylim(-math.pi, math.pi)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    axes[4].plot(x, radius_sorted, color="#444444", linewidth=1.0)
    axes[4].set_ylabel("uv radius")
    axes[4].set_xlabel("token index sorted by uv radius")
    axes[4].grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_noise_phase_lines(
    output_path: Path,
    coords: np.ndarray,
    true_noise: np.ndarray,
    pred_noise: np.ndarray,
    scene_id: int,
    expand_id: int,
    amp_percentile: float = 20.0,
    error_percentile: float = 95.0,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    radius = np.linalg.norm(coords, axis=1)
    order = np.argsort(radius)
    x = np.arange(order.size)
    radius_sorted = radius[order]

    true_complex = as_complex(true_noise)
    pred_complex = as_complex(pred_noise)
    true_amp = np.abs(true_complex)
    pred_amp = np.abs(pred_complex)
    amp_threshold = float(np.nanpercentile(true_amp, amp_percentile))
    if not np.isfinite(amp_threshold) or amp_threshold <= 0.0:
        amp_threshold = 1e-8

    stable = true_amp >= amp_threshold
    if not np.any(stable):
        stable = np.ones_like(true_amp, dtype=bool)

    true_phase = wrapped_phase(true_complex)
    pred_phase = wrapped_phase(pred_complex)
    phase_error = complex_phase_error(pred_complex, true_complex)

    true_amp_sorted = true_amp[order]
    pred_amp_sorted = pred_amp[order]
    stable_sorted = stable[order]
    stable_x = x[stable_sorted]

    true_phase_sorted = true_phase[order]
    pred_phase_sorted = pred_phase[order]
    phase_error_sorted = phase_error[order]

    stable_error = phase_error[stable]
    error_limit = symmetric_limit(stable_error, percentile=error_percentile)
    error_limit = min(math.pi, max(0.15, error_limit * 1.15))
    amp_limit = positive_limit(true_amp, pred_amp, percentile=99.0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Scene {scene_id:04d}, expand_{expand_id}: noise phase details, sorted by uv radius",
        fontsize=13,
    )

    axes[0].scatter(
        x[~stable_sorted],
        true_phase_sorted[~stable_sorted],
        s=5,
        color="#bbbbbb",
        alpha=0.35,
        label=f"low amp truth < p{amp_percentile:g}",
    )
    axes[0].plot(
        stable_x,
        true_phase_sorted[stable_sorted],
        color="#1f77b4",
        linewidth=1.0,
        marker=".",
        markersize=2.5,
        label="ground truth phase",
    )
    axes[0].plot(
        stable_x,
        pred_phase_sorted[stable_sorted],
        color="#d62728",
        linewidth=1.0,
        marker=".",
        markersize=2.5,
        alpha=0.85,
        label="predicted phase",
    )
    axes[0].set_ylabel("wrapped phase")
    axes[0].set_ylim(-math.pi, math.pi)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].scatter(
        x[~stable_sorted],
        phase_error_sorted[~stable_sorted],
        s=5,
        color="#bbbbbb",
        alpha=0.25,
        label="low amp tokens",
    )
    axes[1].plot(
        stable_x,
        phase_error_sorted[stable_sorted],
        color="#2ca02c",
        linewidth=1.0,
        marker=".",
        markersize=2.5,
        label="wrapped phase error",
    )
    axes[1].set_ylabel("phase error")
    axes[1].set_ylim(-error_limit, error_limit)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    axes[2].plot(x, true_amp_sorted, color="#1f77b4", linewidth=1.0, label="ground truth |delta V|")
    axes[2].plot(x, pred_amp_sorted, color="#d62728", linewidth=1.0, alpha=0.85, label="predicted |delta V|")
    axes[2].axhline(amp_threshold, color="#444444", linewidth=0.9, linestyle="--", label="stable phase threshold")
    axes[2].set_ylabel("|delta V|")
    axes[2].set_ylim(0.0, amp_limit)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    axes[3].plot(x, radius_sorted, color="#444444", linewidth=1.0)
    axes[3].fill_between(x, 0.0, radius_sorted, where=stable_sorted, color="#1f77b4", alpha=0.12, label="stable phase tokens")
    axes[3].set_ylabel("uv radius")
    axes[3].set_xlabel("token index sorted by uv radius")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper left", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "noise/phase_stable_amp_threshold": amp_threshold,
        "noise/phase_stable_fraction": float(np.mean(stable)),
        "noise/phase_mae_rad_stable": float(np.mean(np.abs(phase_error[stable]))),
        "noise/phase_mae_rad_amp_weighted": amplitude_weighted_mean_abs(phase_error, true_amp),
    }


def plot_delta_v_uv_image(
    output_path: Path,
    coords: np.ndarray,
    true_noise: np.ndarray,
    pred_noise: np.ndarray,
    scene_id: int,
    expand_id: int,
) -> None:
    import matplotlib.pyplot as plt

    true_complex = as_complex(true_noise)
    pred_complex = as_complex(pred_noise)
    error_complex = pred_complex - true_complex

    amp_vmax = positive_limit(np.abs(true_complex), np.abs(pred_complex), percentile=99.0)
    err_vmax = positive_limit(np.abs(error_complex), percentile=99.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    fig.suptitle(f"Scene {scene_id:04d}, expand_{expand_id}: delta V in uv coordinates", fontsize=13)
    panels = [
        ("ground truth |delta V|", np.abs(true_complex), amp_vmax),
        ("predicted |delta V|", np.abs(pred_complex), amp_vmax),
        ("|predicted - truth|", np.abs(error_complex), err_vmax),
    ]
    for ax, (title, color, vmax) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, s=7, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def noise_metrics(true_noise: np.ndarray, pred_noise: np.ndarray) -> dict[str, float]:
    diff = pred_noise - true_noise
    true_complex = as_complex(true_noise)
    pred_complex = as_complex(pred_noise)
    diff_complex = pred_complex - true_complex
    component_rmse = float(np.sqrt(np.mean(diff**2)))
    component_mae = float(np.mean(np.abs(diff)))
    power = float(np.mean(true_noise**2))
    return {
        "noise/component_rmse": component_rmse,
        "noise/component_mae": component_mae,
        "noise/component_nmse_db": float(10.0 * np.log10((component_rmse**2) / max(power, 1e-12))),
        "noise/component_corr": component_corr(pred_noise, true_noise),
        "noise/complex_abs_rmse": float(np.sqrt(np.mean(np.abs(diff_complex) ** 2))),
        "noise/amp_rmse": float(np.sqrt(np.mean((np.abs(pred_complex) - np.abs(true_complex)) ** 2))),
        "noise/phase_mae_rad": float(np.mean(np.abs(complex_phase_error(pred_complex, true_complex)))),
        "noise/true_real_mean": float(true_noise[:, 0].mean()),
        "noise/true_imag_mean": float(true_noise[:, 1].mean()),
        "noise/pred_real_mean": float(pred_noise[:, 0].mean()),
        "noise/pred_imag_mean": float(pred_noise[:, 1].mean()),
        "noise/true_real_std": float(true_noise[:, 0].std()),
        "noise/true_imag_std": float(true_noise[:, 1].std()),
        "noise/pred_real_std": float(pred_noise[:, 0].std()),
        "noise/pred_imag_std": float(pred_noise[:, 1].std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-id", type=int, default=1001)
    parser.add_argument("--expand-id", type=int, default=0)
    parser.add_argument("--input-suffix", default=None)
    parser.add_argument("--target-suffix", default=None)
    parser.add_argument("--denoise-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-virtual-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-expand-id", type=int, default=None)
    parser.add_argument("--visibility-normalization", choices=["none", "original-rms"], default=None)
    parser.add_argument("--phase-amp-percentile", type=float, default=20.0)
    parser.add_argument("--phase-error-percentile", type=float, default=95.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    ckpt_args = checkpoint.get("args", {})
    data_root = args.data_root or Path(ckpt_args.get("data_root", PROJECT_ROOT / "srdata" / "srdata25Juin"))
    output_dir = args.output_dir or args.checkpoint.parent.parent / "noise_visualizations"
    input_suffix = args.input_suffix or ckpt_args.get("input_suffix", "noised_1")
    target_suffix = args.target_suffix if args.target_suffix is not None else ckpt_args.get("target_suffix", "noised_0")
    context_expand_id = (
        args.context_expand_id
        if args.context_expand_id is not None
        else ckpt_args.get("eval_context_expand_id", ckpt_args.get("context_expand_id"))
    )
    visibility_normalization = args.visibility_normalization or ckpt_args.get("visibility_normalization", "none")
    denoise_only = bool(ckpt_args.get("denoise_only", False)) if args.denoise_only is None else args.denoise_only
    include_virtual_context = (
        bool(ckpt_args.get("include_virtual_context", False))
        if args.include_virtual_context is None
        else args.include_virtual_context
    )
    use_gram_prior = bool(ckpt_args.get("use_gram_prior", False))

    dataset = SRVisibilityDataset(
        data_root,
        scene_ids=[args.scene_id],
        expand_ids=[args.expand_id],
        input_suffix=input_suffix,
        target_suffix=target_suffix,
        include_virtual_context=include_virtual_context,
        context_expand_id=None if context_expand_id is None else int(context_expand_id),
        visibility_normalization=visibility_normalization,
        use_gram_prior=use_gram_prior,
        gram_top_k=int(ckpt_args.get("gram_top_k", 32)),
        gram_image_half_width=math.sin(math.radians(float(ckpt_args.get("gram_image_half_angle_deg", 4.0)))),
        gram_min_corr=float(ckpt_args.get("gram_min_corr", 0.0)),
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one sample, found {len(dataset)}.")

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
            device_batch.gram_context_values,
        )

    batch_cpu = move_batch(device_batch, torch.device("cpu"))
    scale = batch_cpu.visibility_scale[0]
    input_values = (batch_cpu.values[0] * scale).numpy()
    target_values = (batch_cpu.target_values[0] * scale).numpy()
    pred_values = (output.clean_mean.detach().cpu()[0] * scale).numpy()
    coords = batch_cpu.coords[0].numpy()
    mask = batch_cpu.original_mask[0].numpy().astype(bool) if denoise_only else batch_cpu.target_mask[0].numpy().astype(bool)

    coords = coords[mask]
    true_noise = input_values[mask] - target_values[mask]
    pred_noise = input_values[mask] - pred_values[mask]

    stem = f"scene_{args.scene_id:04d}_expand_{args.expand_id}_noise"
    line_path = output_dir / f"{stem}_lines.png"
    phase_path = output_dir / f"{stem}_phase_lines.png"
    uv_path = output_dir / f"{stem}_uv_delta_v.png"
    metrics_path = output_dir / f"{stem}.json"

    plot_noise_lines(line_path, coords, true_noise, pred_noise, args.scene_id, args.expand_id)
    phase_metrics = plot_noise_phase_lines(
        phase_path,
        coords,
        true_noise,
        pred_noise,
        args.scene_id,
        args.expand_id,
        amp_percentile=args.phase_amp_percentile,
        error_percentile=args.phase_error_percentile,
    )
    plot_delta_v_uv_image(uv_path, coords, true_noise, pred_noise, args.scene_id, args.expand_id)
    metrics = noise_metrics(true_noise, pred_noise)
    metrics.update(phase_metrics)
    metrics.update({
        "scene_id": args.scene_id,
        "expand_id": args.expand_id,
        "num_tokens": int(mask.sum()),
        "line_figure": str(line_path),
        "phase_figure": str(phase_path),
        "uv_delta_v_figure": str(uv_path),
    })
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
