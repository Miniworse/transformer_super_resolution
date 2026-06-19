# Bayesian Visibility Transformer

First-version transformer for visibility-region denoising and uv super-resolution.

## Work Tree

```text
.
+-- README.md
+-- requirements.txt
+-- scripts
|   +-- evaluate.py
|   +-- train.py
|   +-- visualize_test.py
+-- src
|   +-- visibility_transformer
|       +-- __init__.py
|       +-- model.py
+-- srdata
    +-- srdata16Juin
        +-- interval5_AMtown01
```

The data arrays are expected as `[V, 2]`:

```text
uv[:, 0], uv[:, 1]                 -> u, v
visibility[:, 0], visibility[:, 1] -> real, imag
redundant[:, 0]                    -> original-array redundancy
redundant[:, 1]                    -> introduced virtual-array redundancy
```

`redundant[:, 0] > 0` marks original known visibility. `redundant[:, 1] > 0`
marks the expanded virtual region used as target support.

By default, the model uses only original known visibility as context. You can
allow virtual visibility as context with `--include-virtual-context`.

Training now defaults to the encoder-decoder architecture:

```text
observed visibility encoder -> expanded uv query decoder -> predicted visibility
```

Use `--architecture encoder` to run the older single-encoder baseline.

## Install

Install PyTorch for your CUDA/CPU environment first, then:

```powershell
pip install -r requirements.txt
```

## Train On Current Unnoised Data

This trains scene `0001-0080`, validates on `0081-0090`, and uses expand levels
`0-4`.

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

This is the recommended denoising setup. The default objective treats the target
as clean, so `clean_mean` is trained directly toward `visibility_*_unnoised.npy`.

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

## Visualize One Test Scene

Create a compact comparison of matched uv visibility and image-domain views:

```powershell
python scripts/visualize_test.py `
  --checkpoint runs/bvt_v1/checkpoints/best.pt `
  --scene-id 91 `
  --expand-id 4
```

The figure contains:

```text
uv panels: original input |V|, predicted clean |V|, ideal target |V|, |error|
image panels: original-input image, predicted-expanded image, ideal-expanded image, image error
```

The image panels are only for visualization. They use a direct irregular inverse
Fourier sum on:

```text
xi = eta = linspace(-sin(4 deg), sin(4 deg), 256)
```

Outputs are written by default to:

```text
runs/bvt_v1/visualizations/
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
region_nll
nll_original
nll_virtual
nll_expanded_only
nll_high_freq
nll_radial_bins
hermitian
kl
noise_prior
```

TensorBoard includes scalar curves and uv scatter figures showing target
amplitude, predicted amplitude, and visibility error amplitude.

## Physical Loss Defaults

The training script uses region-separated losses:

```text
lambda_orig = 1.0
lambda_virtual = 2.0
lambda_expanded = 3.0
lambda_high_freq = 1.0
lambda_radial_bins = 0.0
lambda_sym = 0.1
freq_alpha = 2.0
freq_gamma = 1.0
num_radial_bins = 8
```

These defaults restore the gentler former weighting while keeping Hermitian
symmetry regularization:

```text
V(-u, -v) = conj(V(u, v))
```

The radial-bin term is still available with `--lambda-radial-bins`, but it is
disabled by default because it was too aggressive in the latest run.
