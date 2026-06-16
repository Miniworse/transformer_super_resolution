# Bayesian Visibility Transformer

First-version transformer for visibility-region denoising and uv super-resolution.

## Work Tree

```text
.
├── README.md
├── requirements.txt
├── scripts
│   ├── evaluate.py
│   └── train.py
├── src
│   └── visibility_transformer
│       ├── __init__.py
│       └── model.py
└── srdata
    └── srdata16Juin
        └── interval5_AMtown01
```

The data arrays are expected as `[V, 2]`:

```text
uv[:, 0], uv[:, 1]                 -> u, v
visibility[:, 0], visibility[:, 1] -> real, imag
redundant[:, 0]                    -> original-array redundancy
redundant[:, 1]                    -> introduced virtual-array redundancy
```

`redundant[:, 0] > 0` marks original known visibility used as model context.
`redundant[:, 1] > 0` marks the expanded virtual region used as target support.

## Install

Install PyTorch for your CUDA/CPU environment first, then:

```powershell
pip install -r requirements.txt
```

## Train On Current Unnoised Data

This trains scene `0001-0080`, validates on `0081-0090`, and uses expand levels `0-4`.

```powershell
python scripts/train.py `
  --data-root srdata/srdata16Juin/interval5_AMtown01 `
  --input-suffix unnoised `
  --expand-ids 0,1,2,3,4 `
  --run-dir runs/bvt_v1
```

Open TensorBoard:

```powershell
tensorboard --logdir runs/bvt_v1/tensorboard
```

## Train Later With Fixed Noise

When files named `_noised_1.npy` exist, use noisy input and unnoised clean target:

```powershell
python scripts/train.py `
  --data-root srdata/srdata16Juin/interval5_AMtown01 `
  --input-suffix noised_1 `
  --target-suffix unnoised `
  --expand-ids 0,1,2,3,4 `
  --run-dir runs/bvt_noised_1
```

If you intentionally train with noisy targets only, omit `--target-suffix` and add:

```powershell
--target-is-noisy
```

## Test

Evaluate the best checkpoint on held-out scenes `0091-0100`:

```powershell
python scripts/evaluate.py `
  --checkpoint runs/bvt_v1/checkpoints/best.pt `
  --test-scenes 91,92,93,94,95,96,97,98,99,100 `
  --write-tensorboard
```

Results are written to:

```text
runs/bvt_v1/eval/metrics.json
runs/bvt_v1/eval/tensorboard
```

## Metrics

The scripts log these metrics for four regions:

```text
all            -> every supervised token
original       -> original known visibility region
virtual        -> all virtual-array supported tokens
expanded_only  -> virtual tokens that are not original known tokens
```

Each region includes:

```text
mse
rmse
mae
nmse_db
```

Training also logs:

```text
loss
nll
kl
noise_prior
```

TensorBoard includes scalar curves and uv scatter figures showing target amplitude,
predicted amplitude, and visibility error amplitude.
