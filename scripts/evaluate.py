"""Evaluate a trained Bayesian Visibility Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    BayesianVisibilityTransformer,
    SRVisibilityDataset,
    training_objective,
    visibility_collate_fn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs" / "bvt_v1" / "checkpoints" / "best.pt")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-suffix", default=None)
    parser.add_argument("--target-suffix", default=None)
    parser.add_argument("--target-is-noisy", action="store_true")
    parser.add_argument("--expand-ids", default=None)
    parser.add_argument("--test-scenes", default="91,92,93,94,95,96,97,98,99,100")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--write-tensorboard", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    data_root = args.data_root or Path(ckpt_args.get("data_root", PROJECT_ROOT / "srdata" / "srdata16Juin" / "interval5_AMtown01"))
    output_dir = args.output_dir or args.checkpoint.parent.parent / "eval"
    input_suffix = args.input_suffix or ckpt_args.get("input_suffix", "unnoised")
    target_suffix = args.target_suffix if args.target_suffix is not None else ckpt_args.get("target_suffix")
    expand_ids = parse_int_list(args.expand_ids or ckpt_args.get("expand_ids", "0,1,2,3,4"))
    target_is_noisy = args.target_is_noisy or bool(ckpt_args.get("target_is_noisy", False))

    dataset = SRVisibilityDataset(
        data_root,
        scene_ids=parse_int_list(args.test_scenes),
        expand_ids=expand_ids,
        input_suffix=input_suffix,
        target_suffix=target_suffix,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=visibility_collate_fn,
    )

    model = BayesianVisibilityTransformer(
        coord_dim=2,
        redundancy_dim=2,
        model_dim=int(ckpt_args.get("model_dim", 256)),
        latent_dim=int(ckpt_args.get("latent_dim", 64)),
        num_layers=int(ckpt_args.get("num_layers", 8)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
        num_frequencies=int(ckpt_args.get("num_frequencies", 16)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
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
            out = model(batch.values, batch.coords, batch.known_mask, batch.redundancy, batch.token_mask)
            loss, loss_metrics = training_objective(
                out,
                batch.target_values,
                batch.target_mask,
                target_is_noisy=target_is_noisy,
            )
            item = {key: float(value.cpu()) for key, value in loss_metrics.items()}
            item["loss"] = float(loss.cpu())
            item.update(compute_metrics(batch, out.clean_mean))
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
