import torch

from visibility_transformer import hermitian_symmetrize_complex_values


def test_hermitian_projection_pairs_and_zero_baseline():
    coords = torch.tensor(
        [
            [
                [1.0, 2.0],
                [-1.0, -2.0],
                [0.0, 0.0],
                [3.0, 0.0],
            ]
        ]
    )
    values = torch.tensor(
        [
            [
                [2.0, 4.0],
                [6.0, 8.0],
                [1.5, -7.0],
                [9.0, -3.0],
            ]
        ]
    )
    token_mask = torch.tensor([[True, True, True, True]])

    projected = hermitian_symmetrize_complex_values(values, coords, token_mask)

    assert torch.allclose(projected[0, 1], torch.stack([projected[0, 0, 0], -projected[0, 0, 1]]))
    assert projected[0, 2, 1].abs() < 1e-6
    assert torch.allclose(projected[0, 3], values[0, 3])
