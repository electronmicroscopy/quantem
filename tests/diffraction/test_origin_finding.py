import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantem.core.datastructures import Vector
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
import quantem.diffraction.bragg_peaks as bragg_peaks_module
from quantem.diffraction.bragg_peaks import (
    BraggPeaksPolymer,
    _central_peak_index,
    _display_center,
    _intensity_display_limits,
    _mean_intensity_map,
    _normalized_dp,
    _polar_peak_bins,
    _resolve_intensity_map,
    _zoom_peak_overlay,
)
from quantem.diffraction.polar_transform import (
    find_origin,
    find_origin_angular_descent,
    find_origin_angular_grid,
)


def _ring_pattern(ny, nx, cy, cx, radii=(10, 20, 30), beam_sigma=2.5):
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    pattern = np.zeros((ny, nx), dtype=np.float32)
    for radius in radii:
        pattern += 80.0 * np.exp(-((r - radius) ** 2) / (2 * 1.5**2))
    pattern += 500.0 * np.exp(-(r**2) / (2 * beam_sigma**2))
    return pattern.astype(np.float32)


def _dataset(array):
    return Dataset4dstem.from_array(
        array=np.asarray(array, dtype=np.float32),
        name="origin_finding_test",
        origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 1.0, 1.0),
        units=["pixels", "pixels", "pixels", "pixels"],
        signal_units="counts",
    )


def _bragg_for_plotting():
    arr = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8)
    ds = _dataset(arr)
    bp = BraggPeaksPolymer.from_data(
        ds,
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )

    peaks = Vector.from_shape(
        shape=(2, 2),
        fields=["y_pixels", "x_pixels"],
        units=["pixels", "pixels"],
        name="cartesian_peaks",
    )
    polar = Vector.from_shape(
        shape=(2, 2),
        fields=["r_invA", "theta"],
        units=["1/Å", "radians"],
        name="polar_peaks",
    )
    intensities = Vector.from_shape(
        shape=(2, 2),
        fields=["intensities"],
        units=["counts"],
        name="peak_intensities",
    )
    for i in range(2):
        for j in range(2):
            peaks.set_data(
                np.array([[4.0, 4.0], [2.0, 6.0], [6.0, 2.0]], dtype=float),
                i,
                j,
            )
            polar.set_data(
                np.array([[0.1, 0.0], [0.4, np.pi / 2], [0.7, np.pi]], dtype=float),
                i,
                j,
            )
            intensities.set_data(np.array([[10.0], [20.0], [30.0]], dtype=float), i, j)

    bp.peak_coordinates_cartesian = peaks
    bp.polar_peaks = polar
    bp.peak_intensities = intensities
    bp.polar_data = {"intensity": np.ones((2, 2, 5, 6), dtype=np.float32)}
    bp.image_centers = np.zeros((2, 2, 2), dtype=float)
    bp.image_centers[:, :, :] = np.array([4.0, 4.0])[:, None, None]
    bp.max_radius_invA = 1.0
    bp.num_radial_bins = 5
    bp.num_annular_bins = 3
    bp.two_fold_symmetry = True
    return bp


def test_descent_recovers_subpixel_centers():
    ny = nx = 96
    true_centers = [(47.3, 48.7), (48.4, 47.6), (46.8, 49.1), (49.0, 46.9)]
    arr = np.stack([_ring_pattern(ny, nx, cy, cx) for cy, cx in true_centers]).reshape(
        2, 2, ny, nx
    )

    origins = find_origin_angular_descent(
        _dataset(arr),
        radial_min=4,
        radial_max=36,
        n_phi=96,
        device="cpu",
    )

    assert origins.shape == (2, 2, 2)
    for idx, (cy, cx) in enumerate(true_centers):
        row, col = origins[idx // 2, idx % 2]
        assert np.hypot(row - cy, col - cx) < 0.35


def test_grid_recovers_center_on_small_detector():
    ny = nx = 64
    cy, cx = 30.4, 31.6
    arr = _ring_pattern(ny, nx, cy, cx, radii=(8, 16, 24))[None, None]

    origins = find_origin_angular_grid(
        _dataset(arr),
        radial_min=3,
        radial_max=26,
        num_annular_bins=72,
        device="cpu",
    )

    assert origins.shape == (1, 1, 2)
    assert np.hypot(origins[0, 0, 0] - cy, origins[0, 0, 1] - cx) < 0.75


def test_dispatch_accepts_2d_arrays_and_rejects_bad_method():
    pattern = _ring_pattern(64, 64, 31.5, 31.5)

    origins = find_origin(pattern, method="descent", radial_min=4, radial_max=26, device="cpu")

    assert origins.shape == (1, 1, 2)
    assert np.hypot(origins[0, 0, 0] - 31.5, origins[0, 0, 1] - 31.5) < 0.35
    with pytest.raises(ValueError, match="method"):
        find_origin(pattern, method="peaks")


def test_descent_blank_pattern_returns_image_center():
    origins = find_origin_angular_descent(
        np.zeros((32, 32), dtype=np.float32),
        radial_min=4,
        radial_max=12,
        device="cpu",
    )

    assert origins.shape == (1, 1, 2)
    assert np.allclose(origins[0, 0], [(32 - 1) / 2.0, (32 - 1) / 2.0], atol=1.0)


def test_bragg_peak_polar_transform_matches_image_polar_convention():
    ds = _dataset(np.zeros((1, 1, 16, 16), dtype=np.float32))
    peaks = Vector.from_shape(
        shape=(1, 1),
        fields=["y_pixels", "x_pixels", "y_invA", "x_invA"],
        units=["pixels", "pixels", "1/Å", "1/Å"],
        name="peaks",
    )
    peaks.set_data(
        np.array(
            [
                [8.0, 11.0, 0.0, 0.0],  # +x axis -> theta 0
                [12.0, 8.0, 0.0, 0.0],  # +y axis -> theta pi/2
                [4.0, 8.0, 0.0, 0.0],   # -y axis -> theta 3pi/2, folded to pi/2
                [8.0, 5.0, 0.0, 0.0],   # -x axis -> theta pi, folded to 0
            ],
            dtype=float,
        ),
        0,
        0,
    )
    bp = BraggPeaksPolymer.from_data(
        ds,
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )

    polar = bp.polar_transform_peaks(
        peaks,
        centers=np.array([[[8.0]], [[8.0]]]),
        two_fold_symmetry=True,
        use_tqdm=False,
    )

    got = polar[0, 0]
    assert np.allclose(got[:, 0], [3.0, 4.0, 4.0, 3.0])
    assert np.allclose(got[:, 1], [0.0, np.pi / 2.0, np.pi / 2.0, 0.0])


def test_bragg_polar_transform_two_fold_sums_opposite_angles():
    arr = np.zeros((1, 1, 7, 9), dtype=np.float32)
    arr[0, 0, 5, 4] = 0.2  # 90 degrees at r=2
    arr[0, 0, 1, 4] = 0.3  # 270 degrees at r=2
    ds = _dataset(arr)
    bp = BraggPeaksPolymer.from_data(
        ds,
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    centers = np.array([[[3.0]], [[4.0]]])

    full = bp.polar_transform_4d(
        ds,
        centers=centers,
        num_r=5,
        num_theta=4,
        two_fold_symmetry=False,
        use_tqdm=False,
    )
    folded = bp.polar_transform_4d(
        ds,
        centers=centers,
        num_r=5,
        num_theta=4,
        two_fold_symmetry=True,
        use_tqdm=False,
    )

    assert full["intensity"].shape == (1, 1, 5, 4)
    assert folded["intensity"].shape == (1, 1, 5, 2)
    assert np.allclose(
        folded["intensity"][0, 0],
        full["intensity"][0, 0, :, :2] + full["intensity"][0, 0, :, 2:],
    )
    assert folded["intensity"][0, 0, 2, 1] == pytest.approx(0.5, abs=1e-6)


def test_bragg_peak_polar_transform_inverts_ellipse_mapping():
    ds = _dataset(np.zeros((1, 1, 17, 17), dtype=np.float32))
    peaks = Vector.from_shape(
        shape=(1, 1),
        fields=["y_pixels", "x_pixels"],
        units=["pixels", "pixels"],
        name="peaks",
    )
    peaks.set_data(np.array([[8.0, 14.0]], dtype=float), 0, 0)
    bp = BraggPeaksPolymer.from_data(
        ds,
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )

    polar = bp.polar_transform_peaks(
        peaks,
        centers=np.array([[[8.0]], [[8.0]]]),
        two_fold_symmetry=False,
        ellipse_params=(2.0, 1.0, 0.0),
        use_tqdm=False,
    )

    got = polar[0, 0]
    assert got.shape == (1, 3)
    assert got[0, 0] == pytest.approx(3.0)
    assert got[0, 1] == pytest.approx(0.0)


def test_bragg_private_helpers_characterize_shared_plotting_behavior():
    ds = _dataset(np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4))

    mean_map = _mean_intensity_map(ds, (2, 2))
    assert mean_map.shape == (2, 2)
    assert mean_map[0, 0] == pytest.approx(np.mean(ds[0, 0].array))

    resolved, upsample = _resolve_intensity_map(ds, None, (2, 2))
    assert upsample == 1
    assert np.allclose(resolved, mean_map)
    custom = np.zeros((4, 4), dtype=float)
    resolved, upsample = _resolve_intensity_map(ds, custom, (2, 2), validate=True)
    assert resolved is custom
    assert upsample == 2
    with pytest.raises(ValueError, match="integer multiple"):
        _resolve_intensity_map(ds, np.zeros((5, 4)), (2, 2), validate=True)

    is_rgb, vmin, vmax = _intensity_display_limits(np.dstack([custom, custom, custom]))
    assert is_rgb is True
    assert vmin is None and vmax is None
    is_rgb, vmin, vmax = _intensity_display_limits(np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert is_rgb is False
    assert vmin == pytest.approx(0.03)
    assert vmax == pytest.approx(2.97)

    normalized = _normalized_dp(
        ds,
        0,
        0,
        norm_upper_quantile=0.5,
        norm_power=2.0,
    )
    clipped = np.clip(ds[0, 0].array, 0, np.quantile(ds[0, 0].array, 0.5))
    expected = (clipped / np.nanmax(clipped)) ** 2.0 * np.nanmax(clipped)
    assert np.allclose(normalized, expected)

    assert _display_center(None, 0, 0, (4, 6)) == (2.0, 3.0)
    centers = np.zeros((2, 2, 2), dtype=float)
    centers[:, 1, 1] = [1.5, 2.5]
    assert _display_center(centers, 1, 1, (4, 6)) == pytest.approx((1.5, 2.5))

    peaks_x = np.array([3.0, 5.0])
    peaks_y = np.array([3.0, 1.0])
    peaks_r = np.array([1.0, 0.5])
    assert _central_peak_index(peaks_x, peaks_y, peaks_r, (3.0, 3.0)) == 1
    assert _central_peak_index(peaks_x, peaks_y, None, (3.0, 3.0)) == 0

    cropped, zx, zy, zr, zi, zcentral = _zoom_peak_overlay(
        np.zeros((6, 6)),
        peaks_x,
        peaks_y,
        peaks_r,
        np.array([10.0, 20.0]),
        0,
        2,
        (3.0, 3.0),
    )
    assert cropped.shape == (3, 3)
    assert np.allclose(zx, [1.0])
    assert np.allclose(zy, [1.0])
    assert np.allclose(zr, [1.0])
    assert np.allclose(zi, [10.0])
    assert zcentral == 0

    r_bins, theta_bins = _polar_peak_bins(
        np.array([1.0, 2.0]),
        np.array([np.pi / 2, np.pi]),
        max_radius_invA=2.0,
        num_radial_bins=10,
        num_annular_bins=180,
        two_fold_symmetry=True,
    )
    assert np.allclose(r_bins, [5.0, 10.0])
    assert np.allclose(theta_bins, [90.0, 180.0])


def test_bragg_plotting_and_save_smoke(monkeypatch, tmp_path):
    bp = _bragg_for_plotting()

    def fake_interactive_output(fn, controls):
        fn(**{name: widget.value for name, widget in controls.items()})
        return bragg_peaks_module.widgets.Output()

    monkeypatch.setattr(bragg_peaks_module, "interactive_output", fake_interactive_output)
    monkeypatch.setattr(bragg_peaks_module, "display", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(bragg_peaks_module, "clear_output", lambda *args, **kwargs: None)

    bp.plot_interactive_image_map(ry=0, rx=0, show_polar=False)
    bp.plot_interactive_peak_map(ry=0, rx=0, show_polar=True)

    bp.save_diffraction_figures(0, 0, save_dir=tmp_path / "diff", show_polar=True)
    assert (tmp_path / "diff" / "diffraction_ry0_rx0_intensity_map.pdf").exists()
    assert (tmp_path / "diff" / "diffraction_ry0_rx0_diffraction.pdf").exists()
    assert (tmp_path / "diff" / "diffraction_ry0_rx0_polar.pdf").exists()
    assert (tmp_path / "diff" / "diffraction_ry0_rx0_combined.pdf").exists()

    bp.save_peak_figures(0, 0, save_dir=tmp_path / "peaks", show_polar=True)
    assert (tmp_path / "peaks" / "peaks_ry0_rx0_intensity_map.pdf").exists()
    assert (tmp_path / "peaks" / "peaks_ry0_rx0_diffraction.pdf").exists()
    assert (tmp_path / "peaks" / "peaks_ry0_rx0_polar.pdf").exists()
    plt.close("all")
