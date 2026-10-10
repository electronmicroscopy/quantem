import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem


def _ring_dataset(radius=9.0, center=(16.0, 15.0), shape=(2, 3, 33, 33)):
    rr, cc = np.mgrid[0 : shape[2], 0 : shape[3]].astype(float)
    r = np.hypot(rr - center[0], cc - center[1])
    ring = np.exp(-0.5 * ((r - radius) / 1.0) ** 2)
    array = np.broadcast_to(ring, shape).astype(np.float32).copy()
    return array, Dataset4dstem.from_array(
        array, sampling=[1.0, 1.0, 0.05, 0.05], units=["A", "A", "1/A", "1/A"]
    )


def test_polar_transform_ring_peaks_at_radius():
    _, ds = _ring_dataset()
    polar = ds.polar_transform(origin_row=16.0, origin_col=15.0, num_annular_bins=36)
    assert isinstance(polar, Polar4dstem)
    assert polar.shape[:2] == (2, 3)
    assert polar.n_phi == 36
    profile = polar.array.mean(axis=(0, 1, 2))
    assert int(np.argmax(profile)) == 9  # radial_step 1 px from radial_min 0
    # the ring is isotropic: every azimuthal bin peaks at the same radius
    assert np.all(np.argmax(polar.array[0, 0], axis=1) == 9)
    assert polar.sampling[3] == pytest.approx(0.05)
    assert polar.sampling[2] == pytest.approx(10.0)
    assert polar.units[2] == "deg"
    assert polar.metadata["polar_origin_row"] == 16.0
    assert polar.n_r == len(profile)


def test_polar_transform_two_fold_and_radial_range():
    _, ds = _ring_dataset()
    polar = ds.polar_transform(
        16.0, 15.0, num_annular_bins=18, radial_min=4.0, radial_max=12.0,
        radial_step=0.5, two_fold_rotation_symmetry=True,
    )  # fmt: skip
    assert polar.n_r == 16
    assert polar.sampling[2] == pytest.approx(10.0)  # 180 deg over 18 bins
    profile = polar.array.mean(axis=(0, 1, 2))
    assert 4.0 + 0.5 * int(np.argmax(profile)) == pytest.approx(9.0)


def test_polar_transform_tensor_backed():
    array, ds = _ring_dataset()
    ds_t = Dataset4dstem.from_tensor(torch.as_tensor(array))
    a = ds.polar_transform(16.0, 15.0, num_annular_bins=12).array
    b = ds_t.polar_transform(16.0, 15.0, num_annular_bins=12).array
    assert np.allclose(a, b)
