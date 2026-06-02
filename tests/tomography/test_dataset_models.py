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
