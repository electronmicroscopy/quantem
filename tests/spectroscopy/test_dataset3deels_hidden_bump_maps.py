"""Tests for ``build_hidden_bump_spatial_maps()`` (quantem.spectroscopy)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quantem.spectroscopy.dataset3deels import Dataset3deels  # noqa: E402
from quantem.spectroscopy.spectroscopy_visualzitions import (  # noqa: E402
    build_hidden_bump_spatial_maps,
)

NY, NX = 6, 8
DE = 0.02
E_LL = -1.0 + DE * np.arange(250)  # spans E = 0
E_HL = 0.5 + DE * np.arange(250)  # 0.5 .. 5.48 eV
BUMP_AT = 2.5


def _datasets(bump_amp=0.02):
    """LL: a ZLP whose height varies per pixel; HL: a decaying tail scaled by the same
    height, plus a small broad Gaussian bump in the left half of the scan only."""
    height = 100.0 + 50.0 * np.arange(NX)[None, :] / NX + np.zeros((NY, 1))
    ll = height[..., None] * np.exp(-0.5 * (E_LL / 0.05) ** 2)
    tail = E_HL**-2.0
    bump = bump_amp * np.exp(-0.5 * ((E_HL - BUMP_AT) / 0.3) ** 2)
    has_bump = (np.arange(NX) < NX // 2)[None, :] & np.ones((NY, 1), bool)
    hl = height[..., None] * (tail + np.where(has_bump[..., None], bump, 0.0))
    mk = lambda a, e0: Dataset3deels.from_array(  # noqa: E731
        a, sampling=[1.0, 1.0, DE], origin=[0, 0, e0], units=["nm", "nm", "eV"]
    )
    return mk(hl, E_HL[0]), mk(ll, E_LL[0]), has_bump


def test_bump_region_has_higher_dip_prominence():
    hl, ll, has_bump = _datasets()
    out = build_hidden_bump_spatial_maps(
        hl,
        ll,
        np.ones((NY, NX)),
        None,
        candidate_energies_eV=[BUMP_AT],
        window_samples=21,
        show=False,
    )
    m = out["map_2p500eV"]
    assert np.all(np.isfinite(m))
    assert m[has_bump].mean() > m[~has_bump].mean()
    assert out["failed_pixel_counts"]["map_2p500eV"] == 0
    assert out["correlation_with_adf"] == {}


def test_zero_zlp_pixel_is_nan_masked():
    hl, ll, _ = _datasets()
    arr = np.array(ll.array)
    arr[0, 0, :] = 0.0
    ll.array = arr
    out = build_hidden_bump_spatial_maps(
        hl,
        ll,
        np.ones((NY, NX)),
        np.ones((NY, NX)),
        candidate_energies_eV=[BUMP_AT],
        window_samples=21,
        show=False,
    )
    assert np.isnan(out["map_2p500eV"][0, 0])
    assert out["failed_pixel_counts"]["map_2p500eV"] == 1


def test_candidate_at_axis_edge_raises():
    hl, ll, _ = _datasets()
    with pytest.raises(ValueError, match="edge margin"):
        build_hidden_bump_spatial_maps(
            hl,
            ll,
            np.ones((NY, NX)),
            None,
            candidate_energies_eV=[float(E_HL[2])],
            window_samples=21,
            show=False,
        )
