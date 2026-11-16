"""ModelDiffraction on a synthetic disk lattice with a known strain ramp."""

import warnings

import numpy as np
import pytest

from quantem.core.datastructures import Dataset4dstem
from quantem.core.fitting import DCBackground, DiskTemplate, SyntheticDiskLattice
from quantem.diffraction import ModelDiffraction

N, R = 64, 3.0
U, V = np.array([9.0, 3.0]), np.array([-3.0, 10.0])
S = 1.0 + 0.005 * np.arange(4)  # diffraction-vector scale per scan row


def _disk(d):
    return 1.0 / (1.0 + np.exp((d - R) / 0.7))  # smooth edge, so sub-pixel shifts are fit


def _pattern(s):
    rr, cc = np.mgrid[:N, :N]
    dp = np.full((N, N), 0.01)
    for a in range(-2, 3):
        for b in range(-2, 3):
            p = N / 2 + s * (a * U + b * V)
            dp += (4.0 if a == b == 0 else 1.0) * _disk(np.hypot(rr - p[0], cc - p[1]))
    return dp


def _dataset():
    ds = Dataset4dstem.from_array(np.stack([[_pattern(s)] * 2 for s in S]).astype(np.float32))
    ds.metadata.update(q_to_r_rotation_ccw_deg=0.0, q_transpose=False)
    return ds


@pytest.fixture(scope="module")
def fitted():
    tr, tc = np.mgrid[-5:6, -5:6]
    disk = DiskTemplate.from_array(
        name="disk0", array=_disk(np.hypot(tr, tc)), refine_all_pixels=False
    )
    lat = SyntheticDiskLattice(
        name="lat0",
        disk=disk,
        u_row=(U[0], 1.0),
        u_col=(U[1], 1.0),
        v_row=(V[0], 1.0),
        v_col=(V[1], 1.0),
        u_max=2,
        v_max=2,
        per_disk_intensity=False,
        intensity_0=1.0,
        center_intensity_0=4.0,
    )
    md = ModelDiffraction.from_dataset(_dataset()).preprocess()
    md.define_model(
        origin_row=(N / 2, 1.0),
        origin_col=(N / 2, 1.0),
        components=[disk, DCBackground(intensity=0.01), lat],
        device="cpu",
    )
    md.set_loss(loss_fn="mse")
    opt = {"lat0": {"type": "adam", "lr": 1e-2}}
    md.fit_mean_diffraction_pattern(reset=True, n_steps=100, optimizer_params=opt, progress=False)
    md.fit_individual_diffraction_pattern(
        n_steps=300, batch_size=8, optimizer_params=opt, progress=False
    )
    return md


def test_batched_fit_recovers_strain_ramp(fitted):
    # scalar lattice intensity (per_disk_intensity=False) is the create_mask branch that used to fail
    fitted.create_mask(plot=False)
    assert fitted.mask_weight.min() == 0.0 and fitted.mask_weight.max() == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = fitted.calculate_strain_map(g1_ref=U, g2_ref=V)
    expected = (1.0 / S - 1.0)[:, None] * np.ones((1, 2))
    np.testing.assert_allclose(sm.e_rr.array, expected, atol=1e-3)
    np.testing.assert_allclose(sm.e_cc.array, expected, atol=1e-3)
    np.testing.assert_allclose(sm.phi.array, 0.0, atol=1e-3)


def test_preprocess_twice_starts_from_the_raw_data():
    md = ModelDiffraction.from_dataset(_dataset())
    raw = md.dataset.array.copy()
    md.preprocess(mode="gamma", gamma=0.5).preprocess(mode="gamma", gamma=0.5)
    np.testing.assert_allclose(md.dataset.array, np.sqrt(raw), rtol=1e-6)


def test_force_positive_render_hard_constraint_can_be_turned_off():
    # a steep per-disk slope pushes one edge of every disk below zero
    tr, tc = np.mgrid[-5:6, -5:6]
    disk = DiskTemplate.from_array(name="disk0", array=_disk(np.hypot(tr, tc)))
    lat = SyntheticDiskLattice(
        name="lat0",
        disk=disk,
        u_row=U[0],
        u_col=U[1],
        v_row=V[0],
        v_col=V[1],
        u_max=2,
        v_max=2,
        per_disk_intensity=True,
        per_disk_slopes=True,
        intensity_0=1.0,
        intensity_row=0.5,
    )
    md = ModelDiffraction.from_dataset(_dataset()).preprocess()
    md.define_model(origin_row=N / 2, origin_col=N / 2, components=[disk, lat], device="cpu")
    assert float(lat(md.ctx).min()) == 0.0  # clipped by default
    md.model.apply_constraint_params({"lat0": {"force_positive_render": False}})
    assert float(lat(md.ctx).min()) < 0.0
