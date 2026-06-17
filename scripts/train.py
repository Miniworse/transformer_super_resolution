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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visibility_transformer import (  # noqa: E402
    BayesianVisibilityTransformer,
    SRVisibilityDataset,
    VisibilityRegionInput,
    apply_context_dropout,
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
        }

    diff = pred - target
    mse = masked_mean(diff.pow(2), component_mask)
    mae = masked_mean(diff.abs(), component_mask)
    power = masked_mean(target.pow(2), component_mask).clamp_min(1e-12)
    nmse = mse / power
    return {
        f"{prefix}/mse": float(mse.detach().cpu()),
        f"{prefix}/rmse": float(torch.sqrt(mse).detach().cpu()),
        f"{prefix}/mae": float(mae.detach().cpu()),
        f"{prefix}/nmse_db": float((10.0 * torch.log10(nmse)).detach().cpu()),
    }


def compute_metrics(batch: VisibilityRegionInput, pred: Tensor) -> dict[str, float]:
    target = batch.target_values
    expanded_only = batch.virtual_mask & ~batch.original_mask
    metrics = {}
    for name, mask in [
        ("all", batch.target_mask),
        ("original", batch.original_mask),
        ("virtual", batch.virtual_mask),
        ("expanded_only", expanded_only),
    ]:
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
    objective_kwargs: dict[str, float],
) -> dict[str, float]:
    model.eval()
    losses = []
    metric_items = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch.values, batch.coords, batch.known_mask, batch.redundancy, batch.token_mask)
            loss, loss_metrics = visibility_physical_objective(
                out,
                batch,
                target_is_noisy=target_is_noisy,
                **objective_kwargs,
            )
            losses.append(float(loss.detach().cpu()))
            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item.update(compute_metrics(batch, out.clean_mean))
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
    parser.add_argument("--include-virtual-context", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--context-dropout", type=float, default=0.15)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--beta-noise-prior", type=float, default=1e-4)
    parser.add_argument("--lambda-orig", type=float, default=1.0)
    parser.add_argument("--lambda-virtual", type=float, default=2.0)
    parser.add_argument("--lambda-expanded", type=float, default=3.0)
    parser.add_argument("--lambda-high-freq", type=float, default=1.0)
    parser.add_argument("--lambda-sym", type=float, default=0.1)
    parser.add_argument("--freq-alpha", type=float, default=2.0)
    parser.add_argument("--freq-gamma", type=float, default=1.0)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-frequencies", type=int, default=16)
    parser.add_argument("--normalize-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--figure-every", type=int, default=5)
    args = parser.parse_args()

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
    )
    val_dataset = SRVisibilityDataset(
        args.data_root,
        scene_ids=parse_int_list(args.val_scenes),
        expand_ids=parse_int_list(args.expand_ids),
        input_suffix=args.input_suffix,
        target_suffix=args.target_suffix,
        include_virtual_context=args.include_virtual_context,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    model = BayesianVisibilityTransformer(
        coord_dim=2,
        redundancy_dim=2,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_frequencies=args.num_frequencies,
        normalize_coords=args.normalize_coords,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(args.run_dir / "tensorboard")
    global_step = 0
    best_val = float("inf")
    objective_kwargs = {
        "beta_kl": args.beta_kl,
        "beta_noise_prior": args.beta_noise_prior,
        "lambda_orig": args.lambda_orig,
        "lambda_virtual": args.lambda_virtual,
        "lambda_expanded": args.lambda_expanded,
        "lambda_high_freq": args.lambda_high_freq,
        "lambda_sym": args.lambda_sym,
        "freq_alpha": args.freq_alpha,
        "freq_gamma": args.freq_gamma,
        "symmetry_tolerance": args.symmetry_tolerance,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_items = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            values, known_mask = apply_context_dropout(batch.values, batch.known_mask, args.context_dropout)
            out = model(values, batch.coords, known_mask, batch.redundancy, batch.token_mask)
            loss, loss_metrics = visibility_physical_objective(
                out,
                batch,
                target_is_noisy=args.target_is_noisy,
                **objective_kwargs,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item.update(compute_metrics(batch, out.clean_mean.detach()))
            train_items.append(item)

            if global_step % args.log_every == 0:
                log_tensorboard_scalars(writer, item, global_step, "train_step")
                writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        train_metrics = average_metric_dict(train_items)
        val_metrics = evaluate(model, val_loader, device, args.target_is_noisy, objective_kwargs)
        log_tensorboard_scalars(writer, train_metrics, epoch, "train_epoch")
        log_tensorboard_scalars(writer, val_metrics, epoch, "val_epoch")

        if epoch % args.figure_every == 0:
            model.eval()
            with torch.no_grad():
                sample = move_batch(next(iter(val_loader)), device)
                out = model(sample.values, sample.coords, sample.known_mask, sample.redundancy, sample.token_mask)
                fig = make_uv_figure(sample, out.clean_mean)
                writer.add_figure("val/uv_amplitude", fig, epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, args.run_dir / "checkpoints" / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, args.run_dir / "checkpoints" / "best.pt")

        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics.get('loss', math.nan):.6f} "
            f"val_loss={val_metrics.get('loss', math.nan):.6f} "
            f"val_exp_rmse={val_metrics.get('expanded_only/rmse', math.nan):.6f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
