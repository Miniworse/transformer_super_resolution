"""Train the Bayesian Visibility Transformer on simulated srdata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visibility_transformer import (  # noqa: E402
    BayesianVisibilityEncoderDecoder,
    BayesianVisibilityTransformer,
    SRVisibilityDataset,
    VisibilityRegionInput,
    apply_context_dropout,
    gram_neighbor_context_values,
    visibility_physical_objective,
    visibility_collate_fn,
)


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def move_batch(batch: VisibilityRegionInput, device: torch.device) -> VisibilityRegionInput:
    return VisibilityRegionInput(
        values=batch.values.to(device),
        coords=batch.coords.to(device),
        known_mask=batch.known_mask.to(device),
        redundancy=batch.redundancy.to(device),
        token_mask=batch.token_mask.to(device),
        target_values=batch.target_values.to(device),
        target_mask=batch.target_mask.to(device),
        original_mask=batch.original_mask.to(device),
        virtual_mask=batch.virtual_mask.to(device),
        visibility_scale=batch.visibility_scale.to(device),
        gram_context_values=batch.gram_context_values.to(device),
    )


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def region_metrics(pred: Tensor, target: Tensor, mask: Tensor, prefix: str) -> dict[str, float]:
    component_mask = mask.unsqueeze(-1).expand_as(pred)
    count = component_mask.sum()
    if count.item() == 0:
        return {
            f"{prefix}/mse": math.nan,
            f"{prefix}/rmse": math.nan,
            f"{prefix}/mae": math.nan,
            f"{prefix}/nmse_db": math.nan,
            f"{prefix}/corr": math.nan,
        }

    diff = pred - target
    mse = masked_mean(diff.pow(2), component_mask)
    mae = masked_mean(diff.abs(), component_mask)
    power = masked_mean(target.pow(2), component_mask).clamp_min(1e-12)
    nmse = mse / power
    selected = component_mask.bool()
    corr_num = (pred * target).masked_select(selected).sum()
    corr_den = torch.sqrt(
        pred.pow(2).masked_select(selected).sum().clamp_min(1e-12)
        * target.pow(2).masked_select(selected).sum().clamp_min(1e-12)
    )
    corr = corr_num / corr_den
    return {
        f"{prefix}/mse": float(mse.detach().cpu()),
        f"{prefix}/rmse": float(torch.sqrt(mse).detach().cpu()),
        f"{prefix}/mae": float(mae.detach().cpu()),
        f"{prefix}/nmse_db": float((10.0 * torch.log10(nmse)).detach().cpu()),
        f"{prefix}/corr": float(corr.detach().cpu()),
    }


def compute_metrics(batch: VisibilityRegionInput, pred: Tensor, denoise_only: bool = False) -> dict[str, float]:
    scale = batch.visibility_scale.to(pred.device)
    pred = pred * scale
    target = batch.target_values * scale
    expanded_only = batch.virtual_mask & ~batch.original_mask
    metrics = {}
    regions = [
        ("all", batch.original_mask if denoise_only else batch.target_mask),
        ("original", batch.original_mask),
    ]
    if not denoise_only:
        regions.extend([
            ("virtual", batch.virtual_mask),
            ("expanded_only", expanded_only),
        ])
    for name, mask in regions:
        metrics.update(region_metrics(pred, target, mask, name))
    return metrics


def average_metric_dict(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for item in metrics:
        for key, value in item.items():
            if not math.isnan(value):
                values.setdefault(key, []).append(value)
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def log_tensorboard_scalars(writer, metrics: dict[str, float], step: int, prefix: str) -> None:
    for key, value in metrics.items():
        if not math.isnan(value):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    """Convert argparse values to checkpoint-safe Python primitives."""
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def create_model(args: argparse.Namespace):
    common = {
        "coord_dim": 2,
        "redundancy_dim": 2,
        "model_dim": args.model_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "num_frequencies": args.num_frequencies,
        "normalize_coords": args.normalize_coords,
        "use_gram_prior": args.use_gram_prior,
        "gram_prior_mode": args.gram_prior_mode,
        "use_complex_features": args.use_complex_features,
        "dropout": args.dropout,
    }
    if args.architecture == "encoder":
        return BayesianVisibilityTransformer(num_layers=args.num_layers, **common)
    if args.architecture == "encoder-decoder":
        return BayesianVisibilityEncoderDecoder(
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            separate_denoising_head=args.separate_denoising_head,
            noise_residual_denoising=args.noise_residual_denoising,
            shared_denoising_noise_logvar=args.shared_denoising_noise_logvar,
            use_expanded_residual_head=args.use_expanded_residual_head,
            expanded_residual_start_radius=args.expanded_residual_start_radius,
            expanded_residual_radius_power=args.expanded_residual_radius_power,
            use_gram_attention_bias=args.use_gram_attention_bias,
            gram_attention_strength=args.gram_attention_strength,
            gram_image_half_width=math.sin(math.radians(args.gram_image_half_angle_deg)),
            **common,
        )
    raise ValueError(f"Unsupported architecture: {args.architecture}")


def make_uv_figure(batch: VisibilityRegionInput, pred: Tensor, max_points: int = 2500):
    import matplotlib.pyplot as plt

    coords = batch.coords[0].detach().cpu()
    target = batch.target_values[0].detach().cpu()
    prediction = pred[0].detach().cpu()
    mask = batch.target_mask[0].detach().cpu()

    idx = torch.where(mask)[0]
    if idx.numel() > max_points:
        idx = idx[torch.linspace(0, idx.numel() - 1, max_points).long()]

    uv = coords[idx]
    target_amp = torch.linalg.norm(target[idx], dim=-1)
    pred_amp = torch.linalg.norm(prediction[idx], dim=-1)
    err_amp = torch.linalg.norm(prediction[idx] - target[idx], dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    panels = [("target |V|", target_amp), ("pred |V|", pred_amp), ("error |dV|", err_amp)]
    for ax, (title, color) in zip(axes, panels):
        scatter = ax.scatter(uv[:, 0], uv[:, 1], c=color, s=7, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    return fig


def evaluate(
    model: BayesianVisibilityTransformer,
    loader: DataLoader,
    device: torch.device,
    target_is_noisy: bool,
    denoise_only: bool,
    objective_kwargs: dict[str, float],
) -> dict[str, float]:
    model.eval()
    losses = []
    metric_items = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(
                batch.values,
                batch.coords,
                batch.known_mask,
                batch.redundancy,
                batch.token_mask,
                batch.gram_context_values,
            )
            loss, loss_metrics = visibility_physical_objective(
                out,
                batch,
                target_is_noisy=target_is_noisy,
                denoise_only=denoise_only,
                **objective_kwargs,
            )
            losses.append(float(loss.detach().cpu()))
            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item.update(compute_metrics(batch, out.clean_mean, denoise_only=denoise_only))
            metric_items.append(item)
    metrics = average_metric_dict(metric_items)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "srdata" / "srdata16Juin" / "interval5_AMtown01")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "bvt_v1")
    parser.add_argument("--input-suffix", default="unnoised")
    parser.add_argument("--target-suffix", default=None)
    parser.add_argument("--target-is-noisy", action="store_true")
    parser.add_argument("--denoise-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-virtual-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-expand-id", type=int, default=None)
    parser.add_argument("--eval-context-expand-id", type=int, default=0)
    parser.add_argument("--cross-expansion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cross-expansion-curriculum-epochs", type=int, default=30)
    parser.add_argument("--expansion-sampling-power", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--visibility-normalization", choices=["none", "original-rms"], default="original-rms")
    parser.add_argument("--use-gram-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gram-prior-mode", choices=["feature", "residual"], default="feature")
    parser.add_argument("--gram-top-k", type=int, default=32)
    parser.add_argument("--gram-image-half-angle-deg", type=float, default=4.0)
    parser.add_argument("--gram-min-corr", type=float, default=0.0)
    parser.add_argument("--use-gram-attention-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gram-attention-strength", type=float, default=1.0)
    parser.add_argument("--expand-ids", default="0,1,2,3,4")
    parser.add_argument("--train-scenes", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,"
                        "21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,"
                        "41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,"
                        "61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80")
    parser.add_argument("--val-scenes", default="81,82,83,84,85,86,87,88,89,90")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--beta-noise-prior", type=float, default=1e-4)
    parser.add_argument("--lambda-orig", type=float, default=1.0)
    parser.add_argument("--lambda-virtual", type=float, default=0.0)
    parser.add_argument("--lambda-expanded", type=float, default=3.0)
    parser.add_argument("--lambda-high-freq", type=float, default=1.0)
    parser.add_argument("--lambda-radial-bins", type=float, default=0.0)
    parser.add_argument("--lambda-high-freq-charbonnier", type=float, default=0.0)
    parser.add_argument("--lambda-high-freq-phase", type=float, default=0.0)
    parser.add_argument("--lambda-sym", type=float, default=0.0)
    parser.add_argument("--lambda-energy-orig", type=float, default=0.5)
    parser.add_argument("--lambda-energy-virtual", type=float, default=0.5)
    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--lambda-phase-expanded", type=float, default=1.0)
    parser.add_argument("--lambda-amp-all", type=float, default=0.05)
    parser.add_argument("--lambda-amp-expanded", type=float, default=0.2)
    parser.add_argument("--lambda-expanded-nmse", type=float, default=0.5)
    parser.add_argument("--lambda-expanded-corr", type=float, default=0.3)
    parser.add_argument("--lambda-uncertainty-calibration", type=float, default=0.02)
    parser.add_argument("--lambda-noise-zero-mean", type=float, default=0.0)
    parser.add_argument("--lambda-denoise-clean-charbonnier", type=float, default=0.0)
    parser.add_argument("--freq-alpha", type=float, default=2.0)
    parser.add_argument("--freq-gamma", type=float, default=1.0)
    parser.add_argument("--num-radial-bins", type=int, default=8)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-4)
    parser.add_argument("--architecture", choices=["encoder", "encoder-decoder"], default="encoder-decoder")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-frequencies", type=int, default=16)
    parser.add_argument("--use-complex-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--separate-denoising-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-residual-denoising", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shared-denoising-noise-logvar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-expanded-residual-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expanded-residual-start-radius", type=float, default=0.55)
    parser.add_argument("--expanded-residual-radius-power", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--init-direct-clean-denoiser", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--figure-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["expanded-rmse", "expanded-corr", "denoise-rmse", "denoise-corr"],
        default="expanded-corr",
    )
    args = parser.parse_args()
    if args.noise_residual_denoising and not args.separate_denoising_head:
        parser.error("--noise-residual-denoising requires --separate-denoising-head.")
    if args.init_direct_clean_denoiser and not args.noise_residual_denoising:
        parser.error("--init-direct-clean-denoiser requires --noise-residual-denoising.")
    if args.denoise_only and args.cross_expansion:
        parser.error("--denoise-only cannot be combined with --cross-expansion.")
    if args.denoise_only and args.context_expand_id is not None:
        parser.error("--denoise-only expects source and target to use the same expansion grid.")
    if args.denoise_only and args.selection_metric.startswith("expanded"):
        args.selection_metric = "denoise-rmse"
    if args.denoise_only:
        args.eval_context_expand_id = None
    if args.cross_expansion and args.include_virtual_context:
        parser.error("--cross-expansion requires --no-include-virtual-context to prevent target leakage.")
    if args.use_gram_attention_bias and args.architecture != "encoder-decoder":
        parser.error("--use-gram-attention-bias requires --architecture encoder-decoder.")

    from torch.utils.tensorboard import SummaryWriter

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    with (args.run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    train_dataset = SRVisibilityDataset(
        args.data_root,
        scene_ids=parse_int_list(args.train_scenes),
        expand_ids=parse_int_list(args.expand_ids),
        input_suffix=args.input_suffix,
        target_suffix=args.target_suffix,
        include_virtual_context=args.include_virtual_context,
        context_expand_id=args.context_expand_id,
        visibility_normalization=args.visibility_normalization,
        use_gram_prior=args.use_gram_prior,
        gram_top_k=args.gram_top_k,
        gram_image_half_width=math.sin(math.radians(args.gram_image_half_angle_deg)),
        gram_min_corr=args.gram_min_corr,
        cross_expansion=args.cross_expansion,
        curriculum_epochs=args.cross_expansion_curriculum_epochs,
        sampling_seed=args.sampling_seed,
    )
    val_dataset = SRVisibilityDataset(
        args.data_root,
        scene_ids=parse_int_list(args.val_scenes),
        expand_ids=parse_int_list(args.expand_ids),
        input_suffix=args.input_suffix,
        target_suffix=args.target_suffix,
        include_virtual_context=args.include_virtual_context,
        context_expand_id=args.eval_context_expand_id,
        visibility_normalization=args.visibility_normalization,
        use_gram_prior=args.use_gram_prior,
        gram_top_k=args.gram_top_k,
        gram_image_half_width=math.sin(math.radians(args.gram_image_half_angle_deg)),
        gram_min_corr=args.gram_min_corr,
    )

    sample_weights = [
        train_dataset.sample_weight(index, args.expansion_sampling_power)
        for index in range(len(train_dataset))
    ]
    sampler_generator = torch.Generator().manual_seed(args.sampling_seed)
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=visibility_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=visibility_collate_fn,
    )

    device = torch.device(args.device)
    model = create_model(args).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        if args.init_direct_clean_denoiser:
            state_dict = dict(state_dict)
            for key in ["denoising_head.4.weight", "denoising_head.4.bias"]:
                if key not in state_dict:
                    raise RuntimeError(f"Cannot sign-flip direct-clean denoiser initialization; missing {key!r}.")
                state_dict[key] = state_dict[key].clone()
                state_dict[key][:2].mul_(-1.0)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {key for key in incompatible.missing_keys if key.startswith("expanded_residual_head.")}
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or set(incompatible.missing_keys) != allowed_missing:
            raise RuntimeError(
                "Initial checkpoint is incompatible with the requested model. "
                f"Missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )
        init_note = " with sign-flipped direct-clean denoising head" if args.init_direct_clean_denoiser else ""
        print(f"initialized from {args.init_checkpoint}{init_note}; new residual parameters={sorted(allowed_missing)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(args.run_dir / "tensorboard")
    global_step = 0
    best_score = -float("inf")
    objective_kwargs = {
        "beta_kl": args.beta_kl,
        "beta_noise_prior": args.beta_noise_prior,
        "lambda_orig": args.lambda_orig,
        "lambda_virtual": args.lambda_virtual,
        "lambda_expanded": args.lambda_expanded,
        "lambda_high_freq": args.lambda_high_freq,
        "lambda_radial_bins": args.lambda_radial_bins,
        "lambda_high_freq_charbonnier": args.lambda_high_freq_charbonnier,
        "lambda_high_freq_phase": args.lambda_high_freq_phase,
        "lambda_sym": args.lambda_sym,
        "lambda_energy_orig": args.lambda_energy_orig,
        "lambda_energy_virtual": args.lambda_energy_virtual,
        "lambda_phase": args.lambda_phase,
        "lambda_phase_expanded": args.lambda_phase_expanded,
        "lambda_amp_all": args.lambda_amp_all,
        "lambda_amp_expanded": args.lambda_amp_expanded,
        "lambda_expanded_nmse": args.lambda_expanded_nmse,
        "lambda_expanded_corr": args.lambda_expanded_corr,
        "lambda_uncertainty_calibration": args.lambda_uncertainty_calibration,
        "denoise_noise_residual": args.noise_residual_denoising,
        "lambda_noise_zero_mean": args.lambda_noise_zero_mean,
        "lambda_denoise_clean_charbonnier": args.lambda_denoise_clean_charbonnier,
        "freq_alpha": args.freq_alpha,
        "freq_gamma": args.freq_gamma,
        "num_radial_bins": args.num_radial_bins,
        "symmetry_tolerance": args.symmetry_tolerance,
    }

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        model.train()
        train_items = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            values, known_mask = apply_context_dropout(batch.values, batch.known_mask, args.context_dropout)
            gram_context_values = batch.gram_context_values
            if args.use_gram_prior and args.context_dropout > 0.0:
                gram_context_values = gram_neighbor_context_values(
                    batch.coords,
                    values,
                    known_mask,
                    batch.token_mask,
                    top_k=args.gram_top_k,
                    image_half_width=math.sin(math.radians(args.gram_image_half_angle_deg)),
                    min_corr=args.gram_min_corr,
                )
            out = model(values, batch.coords, known_mask, batch.redundancy, batch.token_mask, gram_context_values)
            loss, loss_metrics = visibility_physical_objective(
                out,
                batch,
                target_is_noisy=args.target_is_noisy,
                denoise_only=args.denoise_only,
                **objective_kwargs,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item.update(compute_metrics(batch, out.clean_mean.detach(), denoise_only=args.denoise_only))
            train_items.append(item)

            if global_step % args.log_every == 0:
                log_tensorboard_scalars(writer, item, global_step, "train_step")
                writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        train_metrics = average_metric_dict(train_items)
        val_metrics = evaluate(model, val_loader, device, args.target_is_noisy, args.denoise_only, objective_kwargs)
        log_tensorboard_scalars(writer, train_metrics, epoch, "train_epoch")
        log_tensorboard_scalars(writer, val_metrics, epoch, "val_epoch")

        if epoch % args.figure_every == 0:
            model.eval()
            with torch.no_grad():
                sample = move_batch(next(iter(val_loader)), device)
                out = model(
                    sample.values,
                    sample.coords,
                    sample.known_mask,
                    sample.redundancy,
                    sample.token_mask,
                    sample.gram_context_values,
                )
                fig = make_uv_figure(sample, out.clean_mean)
                writer.add_figure("val/uv_amplitude", fig, epoch)

        if args.selection_metric == "expanded-corr":
            best_metric_name = "expanded_only/corr"
        elif args.selection_metric == "expanded-rmse":
            best_metric_name = "expanded_only/rmse"
        elif args.selection_metric == "denoise-corr":
            best_metric_name = "original/corr"
        else:
            best_metric_name = "original/rmse"
        best_metric = val_metrics.get(best_metric_name, math.nan)
        if math.isnan(best_metric):
            best_metric_name = "all/rmse"
            best_metric = val_metrics.get(best_metric_name, math.nan)
            metric_mode = "min"
        else:
            metric_mode = "max" if args.selection_metric in {"expanded-corr", "denoise-corr"} else "min"
        if math.isnan(best_metric):
            best_metric_name = "loss"
            best_metric = val_metrics[best_metric_name]
            metric_mode = "min"

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "val_metrics": val_metrics,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
        }
        torch.save(checkpoint, args.run_dir / "checkpoints" / "last.pt")
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(checkpoint, args.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        selection_score = best_metric if metric_mode == "max" else -best_metric
        if selection_score > best_score:
            best_score = selection_score
            torch.save(checkpoint, args.run_dir / "checkpoints" / "best.pt")

        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics.get('loss', math.nan):.6f} "
            f"val_loss={val_metrics.get('loss', math.nan):.6f} "
            f"val_all_rmse={val_metrics.get('all/rmse', math.nan):.6f} "
            f"val_orig_rmse={val_metrics.get('original/rmse', math.nan):.6f} "
            f"val_exp_rmse={val_metrics.get('expanded_only/rmse', math.nan):.6f} "
            f"best_metric={best_metric_name}:{best_metric:.6f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
