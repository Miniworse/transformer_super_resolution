from pathlib import Path
import tempfile

import numpy as np
import torch

from visibility_transformer import (
    BVTOutput,
    BayesianVisibilityEncoderDecoder,
    SRVisibilityDataset,
    VisibilityRegionInput,
    align_source_visibility_to_target_uv,
    complex_visibility_features,
    visibility_physical_objective,
)


def test_complex_features_are_compressed_and_masked():
    values = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    known = torch.tensor([[True, False]])
    features = complex_visibility_features(values, known)

    assert features.shape == (1, 2, 5)
    assert torch.allclose(features[0, 0], torch.tensor([3.0, 4.0, torch.log1p(torch.tensor(5.0)), 0.6, 0.8]))
    assert torch.count_nonzero(features[0, 1]) == 0


def test_source_uv_alignment_builds_observed_mask():
    target_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    source_uv = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    source_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    aligned, observed = align_source_visibility_to_target_uv(target_uv, source_uv, source_values)

    assert observed.tolist() == [True, False, True]
    assert torch.allclose(aligned[observed], source_values)


def test_dual_heads_and_gram_attention_receive_gradients():
    model = BayesianVisibilityEncoderDecoder(
        model_dim=32,
        latent_dim=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        num_frequencies=2,
        use_complex_features=True,
        separate_denoising_head=True,
        use_expanded_residual_head=True,
        expanded_residual_start_radius=0.5,
        use_gram_attention_bias=True,
        dropout=0.0,
    )
    coords = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]])
    values = torch.tensor([[[1.0, 0.2], [1.0, -0.2], [0.0, 0.0], [0.0, 0.0]]])
    known = torch.tensor([[True, True, False, False]])
    token_mask = torch.ones_like(known)
    redundancy = torch.ones(1, 4, 2)

    output = model(values, coords, known, redundancy, token_mask)
    loss = output.clean_mean[known].sum() + output.clean_mean[~known].sum()
    loss.backward()

    assert torch.isfinite(output.clean_mean).all()
    assert model.denoising_head[-1].weight.grad is not None
    assert model.head[-1].weight.grad is not None
    assert output.expanded_residual is not None
    assert torch.count_nonzero(output.expanded_residual[known]) == 0
    assert model.expanded_residual_head[-1].weight.grad is not None
    assert torch.allclose(output.clean_mean[0, 1], output.clean_mean[0, 0] * torch.tensor([1.0, -1.0]))


def test_shared_denoising_noise_logvar_uses_one_observed_variance():
    model = BayesianVisibilityEncoderDecoder(
        model_dim=32,
        latent_dim=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        num_frequencies=2,
        separate_denoising_head=True,
        noise_residual_denoising=True,
        shared_denoising_noise_logvar=True,
        dropout=0.0,
    )
    model.eval()
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    values = torch.tensor([[[1.0, 0.2], [0.8, -0.1], [0.0, 0.0]]])
    known = torch.tensor([[True, True, False]])
    token_mask = torch.ones_like(known)
    redundancy = torch.ones(1, 3, 2)

    with torch.no_grad():
        output = model(values, coords, known, redundancy, token_mask)

    assert torch.allclose(output.noise_logvar[0, 0], output.noise_logvar[0, 1])


def test_dual_clean_noise_head_uses_decoder_clean_and_noise_residual():
    model = BayesianVisibilityEncoderDecoder(
        model_dim=32,
        latent_dim=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        num_frequencies=2,
        separate_denoising_head=True,
        noise_residual_denoising=True,
        dual_clean_noise_head=True,
        dropout=0.0,
    )
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    values = torch.tensor([[[1.0, 0.2], [0.8, -0.1], [0.0, 0.0]]])
    known = torch.tensor([[True, True, False]])
    token_mask = torch.ones_like(known)
    redundancy = torch.ones(1, 3, 2)

    output = model(values, coords, known, redundancy, token_mask)
    loss = output.clean_mean[known].sum() + output.noise_mean[known].sum()
    loss.backward()

    assert torch.isfinite(output.clean_mean).all()
    assert torch.isfinite(output.noise_mean).all()
    assert model.head[-1].weight.grad is not None
    assert model.denoising_head[-1].weight.grad is not None
    assert torch.count_nonzero(output.noise_mean[~known]) == 0


def _write_sample(root: Path, expand_id: int, uv: np.ndarray) -> None:
    scene = 1
    noisy = np.stack([uv[:, 0] + 1.0, uv[:, 1] - 0.5], axis=-1).astype(np.float32)
    clean = noisy * np.float32(0.9)
    redundancy = np.zeros((len(uv), 2), dtype=np.float32)
    redundancy[:2, 0] = 1.0
    redundancy[:, 1] = 1.0
    np.save(root / f"uv_{scene:04d}_expand_{expand_id}_noised_1.npy", uv)
    np.save(root / f"visibility_{scene:04d}_expand_{expand_id}_noised_1.npy", noisy)
    np.save(root / f"visibility_{scene:04d}_expand_{expand_id}_noised_0.npy", clean)
    np.save(root / f"redundant_{scene:04d}_expand_{expand_id}_noised_1.npy", redundancy)


def test_cross_expansion_curriculum_uses_lower_source_support():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_sample(root, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        _write_sample(
            root,
            1,
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]], dtype=np.float32),
        )
        _write_sample(
            root,
            2,
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [3.0, 0.0], [-3.0, 0.0]],
                dtype=np.float32,
            ),
        )
        dataset = SRVisibilityDataset(
            root,
            scene_ids=[1],
            expand_ids=[0, 1, 2],
            input_suffix="noised_1",
            target_suffix="noised_0",
            cross_expansion=True,
            curriculum_epochs=10,
        )
        target_two_index = dataset.samples.index((1, 2))
        dataset.set_epoch(0)
        sample = dataset[target_two_index]

        assert sample.values.shape[1] == 6
        assert sample.known_mask.sum().item() == 4
        assert sample.original_mask.sum().item() == 4
        assert (sample.virtual_mask & ~sample.original_mask).sum().item() == 2


def test_cross_expansion_falls_back_to_union_for_offset_grids():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_sample(root, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        _write_sample(root, 1, np.array([[0.5, 0.0], [1.5, 0.0]], dtype=np.float32))
        for expand_id in [0, 1]:
            redundancy_path = root / f"redundant_0001_expand_{expand_id}_noised_1.npy"
            np.save(redundancy_path, np.load(redundancy_path).T)
        dataset = SRVisibilityDataset(
            root,
            scene_ids=[1],
            expand_ids=[0, 1],
            input_suffix="noised_1",
            target_suffix="noised_0",
            context_expand_id=0,
        )
        sample = dataset[dataset.samples.index((1, 1))]

        assert sample.values.shape[1] == 4
        assert sample.known_mask.sum().item() == 2
        assert (sample.virtual_mask & ~sample.original_mask).sum().item() == 2


def test_denoise_only_objective_ignores_expanded_tokens():
    clean_mean = torch.tensor([[[1.0, 0.1], [100.0, 50.0]]], requires_grad=True)
    zeros = torch.zeros(1, 2, 1)
    latent = torch.zeros(1, 2)
    batch = VisibilityRegionInput(
        values=torch.tensor([[[0.9, 0.0], [0.0, 0.0]]]),
        coords=torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        known_mask=torch.tensor([[True, False]]),
        redundancy=torch.ones(1, 2, 2),
        token_mask=torch.tensor([[True, True]]),
        target_values=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        target_mask=torch.tensor([[True, True]]),
        original_mask=torch.tensor([[True, False]]),
        virtual_mask=torch.tensor([[True, True]]),
        visibility_scale=torch.ones(1, 1, 1),
        gram_context_values=torch.zeros(1, 2, 2),
    )
    output = BVTOutput(
        clean_mean=clean_mean,
        clean_logvar=zeros,
        noise_logvar=zeros,
        latent_mean=latent,
        latent_logvar=latent,
        prior_mean=latent,
        prior_logvar=latent,
    )

    loss, metrics = visibility_physical_objective(
        output,
        batch,
        denoise_only=True,
        beta_kl=0.0,
        beta_noise_prior=0.0,
        lambda_orig=1.0,
        lambda_sym=0.0,
        lambda_energy_orig=0.0,
        lambda_phase=0.0,
        lambda_amp_all=0.0,
        lambda_uncertainty_calibration=0.0,
    )
    loss.backward()

    assert metrics["sr_structure"].item() == 0.0
    assert metrics["high_freq_structure"].item() == 0.0
    assert torch.count_nonzero(clean_mean.grad[0, 1]) == 0


def test_noise_residual_denoising_trains_noise_on_original_tokens():
    predicted_noise = torch.tensor([[[0.0, 0.0], [5.0, -5.0]]], requires_grad=True)
    noisy_values = torch.tensor([[[1.2, 0.1], [0.0, 0.0]]])
    clean_target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    zeros = torch.zeros(1, 2, 1)
    latent = torch.zeros(1, 2)
    batch = VisibilityRegionInput(
        values=noisy_values,
        coords=torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        known_mask=torch.tensor([[True, False]]),
        redundancy=torch.ones(1, 2, 2),
        token_mask=torch.tensor([[True, True]]),
        target_values=clean_target,
        target_mask=torch.tensor([[True, True]]),
        original_mask=torch.tensor([[True, False]]),
        virtual_mask=torch.tensor([[True, True]]),
        visibility_scale=torch.ones(1, 1, 1),
        gram_context_values=torch.zeros(1, 2, 2),
    )
    output = BVTOutput(
        clean_mean=noisy_values - predicted_noise,
        clean_logvar=zeros,
        noise_logvar=zeros,
        latent_mean=latent,
        latent_logvar=latent,
        prior_mean=latent,
        prior_logvar=latent,
        noise_mean=predicted_noise,
    )

    loss, metrics = visibility_physical_objective(
        output,
        batch,
        denoise_only=True,
        denoise_noise_residual=True,
        beta_kl=0.0,
        beta_noise_prior=0.0,
        lambda_orig=1.0,
        lambda_sym=0.0,
        lambda_energy_orig=0.0,
        lambda_phase=0.0,
        lambda_amp_all=0.0,
        lambda_uncertainty_calibration=0.0,
        lambda_noise_zero_mean=0.0,
    )
    loss.backward()

    assert metrics["noise_nll_original"].item() == metrics["region_nll"].item()
    assert torch.count_nonzero(predicted_noise.grad[0, 0]) > 0
    assert torch.count_nonzero(predicted_noise.grad[0, 1]) == 0


def test_clean_noise_consistency_trains_clean_and_noise_predictions():
    clean_mean = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    predicted_noise = torch.tensor([[[0.1, 0.0], [5.0, -5.0]]], requires_grad=True)
    noisy_values = torch.tensor([[[1.4, 0.2], [0.0, 0.0]]])
    clean_target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    zeros = torch.zeros(1, 2, 1)
    latent = torch.zeros(1, 2)
    batch = VisibilityRegionInput(
        values=noisy_values,
        coords=torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        known_mask=torch.tensor([[True, False]]),
        redundancy=torch.ones(1, 2, 2),
        token_mask=torch.tensor([[True, True]]),
        target_values=clean_target,
        target_mask=torch.tensor([[True, True]]),
        original_mask=torch.tensor([[True, False]]),
        virtual_mask=torch.tensor([[True, True]]),
        visibility_scale=torch.ones(1, 1, 1),
        gram_context_values=torch.zeros(1, 2, 2),
    )
    output = BVTOutput(
        clean_mean=clean_mean,
        clean_logvar=zeros,
        noise_logvar=zeros,
        latent_mean=latent,
        latent_logvar=latent,
        prior_mean=latent,
        prior_logvar=latent,
        noise_mean=predicted_noise,
    )

    loss, metrics = visibility_physical_objective(
        output,
        batch,
        denoise_only=True,
        denoise_noise_residual=True,
        beta_kl=0.0,
        beta_noise_prior=0.0,
        lambda_orig=0.0,
        lambda_sym=0.0,
        lambda_energy_orig=0.0,
        lambda_phase=0.0,
        lambda_amp_all=0.0,
        lambda_uncertainty_calibration=0.0,
        lambda_noise_zero_mean=0.0,
        lambda_clean_noise_consistency=1.0,
    )
    loss.backward()

    assert metrics["consistency"].item() > 0.0
    assert torch.count_nonzero(clean_mean.grad[0, 0]) > 0
    assert torch.count_nonzero(predicted_noise.grad[0, 0]) > 0
    assert torch.count_nonzero(predicted_noise.grad[0, 1]) == 0


if __name__ == "__main__":
    test_complex_features_are_compressed_and_masked()
    test_source_uv_alignment_builds_observed_mask()
    test_dual_heads_and_gram_attention_receive_gradients()
    test_shared_denoising_noise_logvar_uses_one_observed_variance()
    test_dual_clean_noise_head_uses_decoder_clean_and_noise_residual()
    test_cross_expansion_curriculum_uses_lower_source_support()
    test_cross_expansion_falls_back_to_union_for_offset_grids()
    test_denoise_only_objective_ignores_expanded_tokens()
    test_noise_residual_denoising_trains_noise_on_original_tokens()
    test_clean_noise_consistency_trains_clean_and_noise_predictions()
    print("cross-expansion training assertions passed")
