"""Evaluate a trained Bayesian Visibility Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys
import warnings

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import (  # noqa: E402
    average_metric_dict,
    compute_metrics,
    make_uv_figure,
    move_batch,
    parse_int_list,
)
from visibility_transformer import (  # noqa: E402
    BayesianVisibilityEncoderDecoder,
    BayesianVisibilityTransformer,
    SRVisibilityDataset,
    visibility_physical_objective,
    visibility_collate_fn,
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            "Falling back to weights_only=False for a legacy trusted checkpoint. "
            "Re-save the checkpoint with the current training script to avoid this fallback.",
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
        "use_gram_prior": bool(ckpt_args.get("use_gram_prior", False)),
        "gram_prior_mode": ckpt_args.get("gram_prior_mode", "feature"),
        "use_complex_features": bool(ckpt_args.get("use_complex_features", False)),
        "dropout": float(ckpt_args.get("dropout", 0.1)),
    }
    architecture = ckpt_args.get("architecture", "encoder")
    if architecture == "encoder-decoder":
        return BayesianVisibilityEncoderDecoder(
            num_encoder_layers=int(ckpt_args.get("num_encoder_layers", 6)),
            num_decoder_layers=int(ckpt_args.get("num_decoder_layers", 4)),
            separate_denoising_head=bool(ckpt_args.get("separate_denoising_head", False)),
            noise_residual_denoising=bool(ckpt_args.get("noise_residual_denoising", False)),
            dual_clean_noise_head=bool(ckpt_args.get("dual_clean_noise_head", False)),
            shared_denoising_noise_logvar=bool(ckpt_args.get("shared_denoising_noise_logvar", False)),
            use_expanded_residual_head=bool(ckpt_args.get("use_expanded_residual_head", False)),
            expanded_residual_start_radius=float(ckpt_args.get("expanded_residual_start_radius", 0.55)),
            expanded_residual_radius_power=float(ckpt_args.get("expanded_residual_radius_power", 1.0)),
            use_gram_attention_bias=bool(ckpt_args.get("use_gram_attention_bias", False)),
            gram_attention_strength=float(ckpt_args.get("gram_attention_strength", 1.0)),
            gram_image_half_width=math.sin(
                math.radians(float(ckpt_args.get("gram_image_half_angle_deg", 4.0)))
            ),
            **common,
        )
    return BayesianVisibilityTransformer(num_layers=int(ckpt_args.get("num_layers", 8)), **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "bvt_v1" / "checkpoints" / "best.pt")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-suffix", default=None)
    parser.add_argument("--target-suffix", default=None)
    parser.add_argument("--target-is-noisy", action="store_true")
    parser.add_argument("--denoise-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-virtual-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-expand-id", type=int, default=None)
    parser.add_argument("--visibility-normalization", choices=["none", "original-rms"], default=None)
    parser.add_argument("--expand-ids", default=None)
    parser.add_argument("--test-scenes", default="91,92,93,94,95,96,97,98,99,100")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--write-tensorboard", action="store_true")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    ckpt_args = checkpoint.get("args", {})

    data_root = args.data_root or Path(ckpt_args.get("data_root", PROJECT_ROOT / "srdata" / "srdata16Juin" / "interval5_AMtown01"))
    output_dir = args.output_dir or args.checkpoint.parent.parent / "eval"
    input_suffix = args.input_suffix or ckpt_args.get("input_suffix", "unnoised")
    target_suffix = args.target_suffix if args.target_suffix is not None else ckpt_args.get("target_suffix")
    expand_ids = parse_int_list(args.expand_ids or ckpt_args.get("expand_ids", "0,1,2,3,4"))
    context_expand_id = (
        args.context_expand_id
        if args.context_expand_id is not None
        else ckpt_args.get("eval_context_expand_id", ckpt_args.get("context_expand_id"))
    )
    visibility_normalization = args.visibility_normalization or ckpt_args.get("visibility_normalization", "none")
    target_is_noisy = args.target_is_noisy or bool(ckpt_args.get("target_is_noisy", False))
    denoise_only = bool(ckpt_args.get("denoise_only", False)) if args.denoise_only is None else args.denoise_only
    objective_kwargs = {
        "beta_kl": float(ckpt_args.get("beta_kl", 1e-3)),
        "beta_noise_prior": float(ckpt_args.get("beta_noise_prior", 1e-4)),
        "lambda_orig": float(ckpt_args.get("lambda_orig", 5.0)),
        "lambda_virtual": float(ckpt_args.get("lambda_virtual", 1.0)),
        "lambda_expanded": float(ckpt_args.get("lambda_expanded", 1.0)),
        "lambda_high_freq": float(ckpt_args.get("lambda_high_freq", 0.5)),
        "lambda_radial_bins": float(ckpt_args.get("lambda_radial_bins", 0.0)),
        "lambda_high_freq_charbonnier": float(ckpt_args.get("lambda_high_freq_charbonnier", 0.0)),
        "lambda_high_freq_phase": float(ckpt_args.get("lambda_high_freq_phase", 0.0)),
        "lambda_sym": float(ckpt_args.get("lambda_sym", 0.0)),
        "lambda_energy_orig": float(ckpt_args.get("lambda_energy_orig", 0.5)),
        "lambda_energy_virtual": float(ckpt_args.get("lambda_energy_virtual", 0.5)),
        "lambda_phase": float(ckpt_args.get("lambda_phase", 0.1)),
        "lambda_phase_expanded": float(ckpt_args.get("lambda_phase_expanded", 0.0)),
        "lambda_amp_all": float(ckpt_args.get("lambda_amp_all", 0.0)),
        "lambda_amp_expanded": float(ckpt_args.get("lambda_amp_expanded", 0.0)),
        "lambda_expanded_nmse": float(ckpt_args.get("lambda_expanded_nmse", 0.0)),
        "lambda_expanded_corr": float(ckpt_args.get("lambda_expanded_corr", 0.0)),
        "lambda_uncertainty_calibration": float(ckpt_args.get("lambda_uncertainty_calibration", 0.0)),
        "denoise_noise_residual": bool(ckpt_args.get("noise_residual_denoising", False)),
        "lambda_noise_zero_mean": float(ckpt_args.get("lambda_noise_zero_mean", 0.0)),
        "lambda_denoise_clean_charbonnier": float(ckpt_args.get("lambda_denoise_clean_charbonnier", 0.0)),
        "lambda_clean_noise_consistency": float(ckpt_args.get("lambda_clean_noise_consistency", 0.0)),
        "freq_alpha": float(ckpt_args.get("freq_alpha", 2.0)),
        "freq_gamma": float(ckpt_args.get("freq_gamma", 1.0)),
        "num_radial_bins": int(ckpt_args.get("num_radial_bins", 8)),
        "symmetry_tolerance": float(ckpt_args.get("symmetry_tolerance", 1e-4)),
    }
    include_virtual_context = (
        bool(ckpt_args.get("include_virtual_context", False))
        if args.include_virtual_context is None
        else args.include_virtual_context
    )
    use_gram_prior = bool(ckpt_args.get("use_gram_prior", False))
    gram_top_k = int(ckpt_args.get("gram_top_k", 32))
    gram_image_half_angle_deg = float(ckpt_args.get("gram_image_half_angle_deg", 4.0))
    gram_min_corr = float(ckpt_args.get("gram_min_corr", 0.0))

    dataset = SRVisibilityDataset(
        data_root,
        scene_ids=parse_int_list(args.test_scenes),
        expand_ids=expand_ids,
        input_suffix=input_suffix,
        target_suffix=target_suffix,
        include_virtual_context=include_virtual_context,
        context_expand_id=None if context_expand_id is None else int(context_expand_id),
        visibility_normalization=visibility_normalization,
        use_gram_prior=use_gram_prior,
        gram_top_k=gram_top_k,
        gram_image_half_width=math.sin(math.radians(gram_image_half_angle_deg)),
        gram_min_corr=gram_min_corr,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=visibility_collate_fn,
    )

    model = create_model_from_args(ckpt_args)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_items = []
    with torch.no_grad():
        first_batch = None
        first_pred = None
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
            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item["loss"] = float(loss.cpu())
            item.update(compute_metrics(batch, out.clean_mean, denoise_only=denoise_only))
            metric_items.append(item)
            if first_batch is None:
                first_batch = batch
                first_pred = out.clean_mean

    metrics = average_metric_dict(metric_items)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if args.write_tensorboard and first_batch is not None and first_pred is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output_dir / "tensorboard")
        for key, value in metrics.items():
            writer.add_scalar(f"test/{key}", value, 0)
        writer.add_figure("test/uv_amplitude", make_uv_figure(first_batch, first_pred), 0)
        writer.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
