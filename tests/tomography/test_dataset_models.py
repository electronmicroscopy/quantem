"""Tests for ``quantem.tomography.dataset_models``.

Covers constraint parsing, the pixelated dataset (validation, normalisation, tilt-angle
convention, pose-parameter materialisation) and the INR / pretrain datasets.
"""

import numpy as np
import pytest
import torch

from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetValue,
    TomographyINRDataset,
    TomographyINRPretrainDataset,
    TomographyPixDataset,
)

from .conftest import requires_torch


class TestDatasetConstraintParse:
    def test_parse_base_by_name(self):
        c = DatasetConstraintParams.parse_dict({"name": "base_tomography_dataset", "tv_zs": 0.1})
        assert isinstance(c, DatasetConstraintParams.BaseTomographyDatasetConstraints)
        assert c.tv_zs == 0.1

    def test_parse_base_by_type_key(self):
        c = DatasetConstraintParams.parse_dict(
            {"type": "base_tomography_dataset", "tv_shifts": 0.2}
        )
        assert c.tv_shifts == 0.2

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            DatasetConstraintParams.parse_dict({"name": "nope"})


def _stack(nang=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((nang, n, n)) * 10).astype(np.float32)


class TestTomographyPixDataset:
    def test_wrong_projection_axis_raises(self):
        # projections must live on axis 0 (i.e. fewer than the image dims).
        bad = np.zeros((20, 5, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            TomographyPixDataset.from_data(bad, np.linspace(-60, 60, 20).astype(np.float32))

    def test_tilt_angles_are_negated(self):
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(), angles)
        np.testing.assert_allclose(d.tilt_angles.numpy(), -angles, atol=1e-5)

    def test_normalised_by_95th_quantile(self):
        d = TomographyPixDataset.from_data(_stack(), np.linspace(-60, 60, 5).astype(np.float32))
        q95 = torch.quantile(d.tilt_stack, 0.95)
        assert torch.isclose(q95, torch.tensor(1.0), atol=1e-4)

    def test_reference_idx_and_learnable_tilts(self):
        # negated angles -> [40, 15, -10, -35, -60]; smallest |angle| is index 2.
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(), angles)
        assert d.reference_tilt_idx == 2
        assert d.learnable_tilts == 4

    def test_forward_returns_dataset_value(self):
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(nang=5, n=12), angles)
        out = d.forward(0)
        assert isinstance(out, DatasetValue)
        assert out.target.shape == (12, 12)
        assert out.tilt_angle == pytest.approx(float(-angles[0]))

    def test_to_materialises_pose_parameters(self):
        d = TomographyPixDataset.from_data(_stack(), np.linspace(-60, 60, 5).astype(np.float32))
        d.to("cpu")
        assert isinstance(d.z1_params, torch.nn.Parameter)
        assert d.shifts_params.shape == (d.learnable_tilts, 2)


@requires_torch
class TestTomographyINRDataset:
    def test_len_is_projections_times_pixels(self):
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12), np.linspace(-60, 60, 5, dtype="f4")
        )
        assert len(d) == 5 * 12 * 12

    def test_getitem_keys(self):
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12), np.linspace(-60, 60, 5, dtype="f4")
        )
        item = d[0]
        assert {"phi", "pixel_i", "pixel_j", "projection_idx", "target_value"} <= set(item.keys())

    @pytest.mark.parametrize(
        "learn_shift,learn_tilt_axis",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_forward_gates_shift_and_tilt(self, learn_shift, learn_tilt_axis):
        """``forward`` zeros the disabled component and passes the enabled one through.

        Guards the gating after removing the unreachable duplicate branch: shifts are
        controlled by ``learn_shift``; the z1/z3 Euler angles by ``learn_tilt_axis``.
        """
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12),
            np.linspace(-60, 60, 5, dtype="f4"),
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
        )
        d.to("cpu")
        # Make every pose parameter non-zero so the gating is observable by value.
        for p in (d.z1_params, d.z3_params, d.shifts_params):
            p.data.fill_(1.0)
        d._z1_ref = torch.ones_like(d._z1_ref)
        d._z3_ref = torch.ones_like(d._z3_ref)
        d._shifts_ref = torch.ones_like(d._shifts_ref)

        shifts, z1, z3 = d.forward(None)
        assert bool(shifts.any()) == learn_shift
        assert bool(z1.any()) == learn_tilt_axis
        assert bool(z3.any()) == learn_tilt_axis


class TestTomographyINRPretrainDataset:
    def test_len_and_getitem(self):
        vol = torch.rand(1, 8, 8, 8)
        ds = TomographyINRPretrainDataset(pretrain_target=vol)
        assert len(ds) == 8**3
        item = ds[0]
        assert set(item.keys()) == {"coords", "target"}
        assert item["coords"].shape == (3,)


@requires_torch
class TestINRRayMath:
    """The static ray helpers are pure tensor math (CPU), exercised here without a recon."""

    def test_create_batch_rays_shape_and_endpoints(self):
        N, S = 8, 5
        rays = TomographyINRDataset.create_batch_rays(
            torch.tensor([0, N - 1]), torch.tensor([0, N - 1]), N=N, num_samples_per_ray=S
        )
        assert rays.shape == (2, S, 3)
        # pixel 0 maps to -1, pixel N-1 maps to +1 on both x (j) and y (i).
        assert torch.allclose(rays[0, :, 0], torch.full((S,), -1.0))
        assert torch.allclose(rays[1, :, 0], torch.full((S,), 1.0))
        # z spans the full -1..1 sampling range.
        assert torch.isclose(rays[0, 0, 2], torch.tensor(-1.0))
        assert torch.isclose(rays[0, -1, 2], torch.tensor(1.0))

    def test_transform_batch_rays_identity_at_zero_pose(self):
        rays = torch.rand(4, 6, 3)
        zero = torch.zeros(4)
        out = TomographyINRDataset.transform_batch_rays(
            rays, z1=zero, x=zero, z3=zero, shifts=torch.zeros(4, 2), N=8, sampling_rate=1.0
        )
        # No rotation and no shift -> rays pass through unchanged.
        assert torch.allclose(out, rays, atol=1e-5)

    def test_integrate_rays_sums_with_step_size(self):
        B, S = 3, 5
        out = TomographyINRDataset.integrate_rays(
            torch.ones(B, S), num_samples_per_ray=S, target_values_len=B
        )
        step = 2.0 / (S - 1)
        assert out.shape == (B,)
        assert torch.allclose(out, torch.full((B,), S * step))

    def test_getitem_index_mapping(self):
        # n=4 -> H*W=16 per projection. idx=21 -> projection 1, remaining 5 -> i=1, j=1.
        stack = _stack(nang=3, n=4)
        d = TomographyINRDataset.from_data(stack, np.linspace(-60, 60, 3, dtype="f4"))
        item = d[21]
        assert int(item["projection_idx"]) == 1
        assert int(item["pixel_i"]) == 1
        assert int(item["pixel_j"]) == 1
        assert torch.isclose(item["target_value"], d.tilt_stack[1, 1, 1])

    def test_len_and_getitem_non_square(self):
        # Regression: pixel_i/pixel_j must decompose by the width (shape[2]) and __len__
        # by H*W, not max(shape)^2 -- both were wrong for rectangular tilt images.
        rng = np.random.default_rng(0)
        stack = rng.random((3, 4, 6)).astype(np.float32)  # nang=3, H=4, W=6
        d = TomographyINRDataset.from_data(stack, np.linspace(-60, 60, 3, dtype="f4"))
        assert len(d) == 3 * 4 * 6
        # idx = proj*(H*W) + i*W + j = 1*24 + 2*6 + 3 = 39
        item = d[39]
        assert int(item["projection_idx"]) == 1
        assert int(item["pixel_i"]) == 2
        assert int(item["pixel_j"]) == 3
        assert torch.isclose(item["target_value"], d.tilt_stack[1, 2, 3])
        # every index in range maps to a valid pixel
        last = d[len(d) - 1]
        assert int(last["projection_idx"]) == 2
        assert int(last["pixel_i"]) == 3
        assert int(last["pixel_j"]) == 5

    def test_save_load_parameters_roundtrip(self, tmp_path):
        angles = np.linspace(-60, 60, 5, dtype="f4")
        d = TomographyINRDataset.from_data(_stack(), angles)
        d.to("cpu")
        d.z1_params.data.fill_(0.37)
        path = str(tmp_path / "params.pt")
        d.save_parameters(path)

        d2 = TomographyINRDataset.from_data(_stack(), angles)
        d2.to("cpu")
        d2.load_parameters(path)
        assert torch.allclose(d2.z1_params.detach(), d.z1_params.detach())
