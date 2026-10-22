import ipywidgets
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quantem.diffraction.bragg_peaks as bragg_peaks_module
import quantem.diffraction.peak_visualization as peak_visualization_module
from quantem.core.datastructures import Vector
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction.bragg_peaks import (
    BraggPeaksPolymer,
    _central_peak_index,
    _display_center,
    _polar_peak_bins,
    _zoom_peak_overlay,
)
from quantem.diffraction.peak_visualization import (
    _intensity_display_limits,
    _mean_intensity_map,
    _normalized_dp,
    _resolve_intensity_map,
)
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
)
from quantem.diffraction.polymer_polar import as_dataset4dstem, find_origin


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


def _bragg(array):
    return BraggPeaksPolymer.from_data(
        _dataset(array),
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )


def _vector(shape, fields, units, name, values):
    """A Vector whose every scan position holds ``values``."""
    vec = Vector.from_shape(shape=shape, fields=fields, units=units, name=name)
    for idx in np.ndindex(*shape):
        vec[idx] = np.asarray(values, dtype=float)
    return vec


def _bragg_for_plotting():
    bp = _bragg(np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8))
    bp.peak_coordinates_cartesian = _vector(
        (2, 2),
        ["y_pixels", "x_pixels"],
        ["pixels", "pixels"],
        "cartesian_peaks",
        [[4.0, 4.0], [2.0, 6.0], [6.0, 2.0]],
    )
    bp.polar_peaks = _vector(
        (2, 2),
        ["r_invA", "theta"],
        ["1/Å", "radians"],
        "polar_peaks",
        [[0.1, 0.0], [0.4, np.pi / 2], [0.7, np.pi]],
    )
    bp.peak_intensities = _vector(
        (2, 2), ["intensities"], ["counts"], "peak_intensities", [[10.0], [20.0], [30.0]]
    )
    bp.polar_data = {"intensity": np.ones((2, 2, 5, 6), dtype=np.float32)}
    bp.image_centers = np.full((2, 2, 2), 4.0)
    bp.max_radius_invA = 1.0
    bp.num_radial_bins = 5
    bp.num_annular_bins = 3
    bp.two_fold_symmetry = True
    return bp


def test_origin_finders_recover_centers():
    descent_centers = [(47.3, 48.7), (48.4, 47.6), (46.8, 49.1), (49.0, 46.9)]
    descent_arr = np.stack([_ring_pattern(96, 96, cy, cx) for cy, cx in descent_centers]).reshape(
        2, 2, 96, 96
    )
    cases = {
        # Subpixel centres that differ per scan position.
        "descent": (
            lambda: find_origin_angular_descent(
                _dataset(descent_arr), radial_min=4, radial_max=36, n_phi=96, device="cpu"
            ),
            np.array(descent_centers).reshape(2, 2, 2),
            0.35,
        ),
        "grid": (
            lambda: find_origin_angular_grid(
                _dataset(_ring_pattern(64, 64, 30.4, 31.6, radii=(8, 16, 24))[None, None]),
                radial_min=3,
                radial_max=26,
                num_annular_bins=72,
                device="cpu",
            ),
            np.array([[[30.4, 31.6]]]),
            0.75,
        ),
        # The dispatcher must accept a bare 2D pattern.
        "dispatch_2d": (
            lambda: find_origin(
                _ring_pattern(64, 64, 31.5, 31.5),
                method="descent",
                radial_min=4,
                radial_max=26,
                device="cpu",
            ),
            np.array([[[31.5, 31.5]]]),
            0.35,
        ),
    }
    for name, (run, expected, tol) in cases.items():
        origins = run()
        assert origins.shape == expected.shape, name
        err = np.hypot(*(origins - expected).transpose(2, 0, 1))
        assert np.all(err < tol), f"{name}: error {err}"

    with pytest.raises(ValueError, match="method"):
        find_origin(_ring_pattern(64, 64, 31.5, 31.5), method="peaks")


@pytest.mark.xfail(
    strict=True,
    reason="polar_transform._com_anchor has no zero-total guard, so a blank pattern "
    "descends to an off-detector origin; the golden snapshot's copy returned the image center.",
)
def test_descent_blank_pattern_returns_image_center():
    origins = find_origin_angular_descent(
        as_dataset4dstem(np.zeros((32, 32), dtype=np.float32)),
        radial_min=4,
        radial_max=12,
        device="cpu",
    )

    assert origins.shape == (1, 1, 2)
    assert np.allclose(origins[0, 0], [(32 - 1) / 2.0, (32 - 1) / 2.0], atol=1.0)


def test_bragg_peak_polar_transform_conventions():
    polar_fields = ["r_pixels", "theta", "r_invA", "theta_unfolded"]
    cases = {
        # Peak theta matches the image polar convention, folded into [0, pi).
        "two_fold": dict(
            size=16,
            fields=["y_pixels", "x_pixels", "y_invA", "x_invA"],
            peaks=[
                [8.0, 11.0, 0.0, 0.0],  # +x axis -> theta 0
                [12.0, 8.0, 0.0, 0.0],  # +y axis -> theta pi/2
                [4.0, 8.0, 0.0, 0.0],  # -y axis -> theta 3pi/2, folded to pi/2
                [8.0, 5.0, 0.0, 0.0],  # -x axis -> theta pi, folded to 0
            ],
            kwargs=dict(two_fold_symmetry=True),
            r=[3.0, 4.0, 4.0, 3.0],
            theta=[0.0, np.pi / 2.0, np.pi / 2.0, 0.0],
            unfolded=[0.0, np.pi / 2.0, 3 * np.pi / 2.0, np.pi],
        ),
        # ellipse_params (2, 1, 0) maps x=+6 back to r=3.
        "ellipse": dict(
            size=17,
            fields=["y_pixels", "x_pixels"],
            peaks=[[8.0, 14.0]],
            kwargs=dict(two_fold_symmetry=False, ellipse_params=(2.0, 1.0, 0.0)),
            r=[3.0],
            theta=[0.0],
            unfolded=[0.0],
        ),
    }
    for name, case in cases.items():
        bp = _bragg(np.zeros((1, 1, case["size"], case["size"]), dtype=np.float32))
        peaks = _vector(
            (1, 1),
            case["fields"],
            ["pixels", "pixels", "1/Å", "1/Å"][: len(case["fields"])],
            "peaks",
            case["peaks"],
        )
        polar = bp.polar_transform_peaks(
            peaks, centers=np.array([[[8.0]], [[8.0]]]), use_tqdm=False, **case["kwargs"]
        )
        got = polar[0, 0].numpy()
        assert polar.fields[:4] == polar_fields, name
        assert np.allclose(got[:, 0], case["r"]), f"{name}: r {got[:, 0]}"
        assert np.allclose(got[:, 1], case["theta"]), f"{name}: theta {got[:, 1]}"
        assert np.allclose(got[:, 3], case["unfolded"]), f"{name}: unfolded {got[:, 3]}"


def test_bragg_polar_transform_two_fold_sums_opposite_angles():
    arr = np.zeros((1, 1, 7, 9), dtype=np.float32)
    arr[0, 0, 5, 4] = 0.2  # 90 degrees at r=2
    arr[0, 0, 1, 4] = 0.3  # 270 degrees at r=2
    bp = _bragg(arr)
    full, folded = (
        bp.polar_transform_4d(
            _dataset(arr),
            centers=np.array([[[3.0]], [[4.0]]]),
            num_r=5,
            num_theta=4,
            two_fold_symmetry=fold,
            use_tqdm=False,
        )["intensity"]
        for fold in (False, True)
    )

    assert full.shape == (1, 1, 5, 4)
    assert folded.shape == (1, 1, 5, 2)
    assert np.allclose(folded[0, 0], full[0, 0, :, :2] + full[0, 0, :, 2:])
    assert folded[0, 0, 2, 1] == pytest.approx(0.5, abs=1e-6)


def _bragg_orientation_histogram(theta_values, normalize_stack=False):
    bp = _bragg(np.zeros((1, 1, 8, 8), dtype=np.float32))
    theta_values = np.asarray(theta_values, dtype=float)
    bp.polar_peaks = _vector(
        (1, 1),
        ["r_invA", "theta"],
        ["1/Å", "radians"],
        "polar_peaks",
        np.column_stack([np.ones_like(theta_values), theta_values]),
    )
    bp.peak_intensities = _vector(
        (1, 1),
        ["intensities"],
        ["counts"],
        "peak_intensities",
        np.ones((theta_values.size, 1)),
    )
    return bp.make_orientation_histogram(
        radial_ranges=np.array([0.5, 1.5]),
        upsample_factor=2,
        theta_step_deg=90,
        sigma_x=None,
        sigma_y=None,
        sigma_theta=None,
        normalize_intensity_image=False,
        normalize_intensity_stack=normalize_stack,
        progress_bar=False,
    )


def test_bragg_orientation_histogram_angles():
    cases = {
        # Peak angles land in their own bins.
        "karen_angles": ([0.0, np.pi / 2], False, [1.0, 1.0]),
        # theta and theta + pi fold onto the same bin.
        "unwrapped": ([0.0, np.pi, np.pi / 2, 3 * np.pi / 2], False, [2.0, 2.0]),
        # No peaks with stack normalization must not divide by zero.
        "empty": ([], True, [0.0, 0.0]),
    }
    for name, (thetas, normalize_stack, expected) in cases.items():
        hist = _bragg_orientation_histogram(thetas, normalize_stack=normalize_stack)
        assert np.isfinite(hist).all(), name
        assert hist[0, 0, 0] == pytest.approx(expected), f"{name}: {hist[0, 0, 0]}"
        if not thetas:
            assert not np.any(hist), name


def test_bragg_private_helpers_characterize_shared_plotting_behavior():
    ds = _dataset(np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4))

    mean_map = _mean_intensity_map(ds, (2, 2))
    assert np.allclose(mean_map, ds.array.mean(axis=(2, 3)))

    resolved, upsample = _resolve_intensity_map(ds, None, (2, 2))
    assert upsample == 1
    assert np.allclose(resolved, mean_map)
    custom = np.zeros((4, 4), dtype=float)
    resolved, upsample = _resolve_intensity_map(ds, custom, (2, 2), validate=True)
    assert resolved is custom
    assert upsample == 2
    with pytest.raises(ValueError, match="integer multiple"):
        _resolve_intensity_map(ds, np.zeros((5, 4)), (2, 2), validate=True)

    is_rgb, vmin, vmax = _intensity_display_limits(np.dstack([custom] * 3))
    assert is_rgb is True
    assert vmin is None and vmax is None
    is_rgb, vmin, vmax = _intensity_display_limits(np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert is_rgb is False
    assert (vmin, vmax) == pytest.approx((0.03, 2.97))

    normalized = _normalized_dp(ds, 0, 0, norm_upper_quantile=0.5, norm_power=2.0)
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
    assert _central_peak_index(peaks_x, peaks_y, peaks_r, (3.0, 3.0)) == 0
    assert _central_peak_index(peaks_x, peaks_y, None, (3.0, 3.0)) == 0

    zoom_cases = {
        # The central peak stays in the crop and keeps its index.
        "central_kept": (
            (np.zeros((6, 6)), peaks_x, peaks_y, peaks_r, (3.0, 3.0)),
            dict(shape=(3, 3), xy=1.0, r=1.0, i=10.0, central=0, center=(1.0, 1.0)),
        ),
        # The central peak falls outside the crop, so no central index survives.
        "central_cropped": (
            (
                np.zeros((8, 8)),
                np.array([1.0, 4.0]),
                np.array([1.0, 4.0]),
                np.array([0.1, 2.0]),
                (4.0, 4.0),
            ),
            dict(shape=(4, 4), xy=2.0, r=2.0, i=20.0, central=None, center=(2.0, 2.0)),
        ),
    }
    for name, ((image, px, py, pr, center), exp) in zoom_cases.items():
        cropped, zx, zy, zr, zi, zcentral, display_center = _zoom_peak_overlay(
            image, px, py, pr, np.array([10.0, 20.0]), 0, 2, center
        )
        assert cropped.shape == exp["shape"], name
        assert np.allclose(zx, [exp["xy"]]) and np.allclose(zy, [exp["xy"]]), name
        assert np.allclose(zr, [exp["r"]]), name
        assert np.allclose(zi, [exp["i"]]), name
        assert zcentral == exp["central"], name
        assert display_center == pytest.approx(exp["center"]), name

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
        return ipywidgets.Output()

    for module in (bragg_peaks_module, peak_visualization_module):
        monkeypatch.setattr(module, "interactive_output", fake_interactive_output, raising=False)
        monkeypatch.setattr(module, "display", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(module, "clear_output", lambda *args, **kwargs: None, raising=False)

    bp.plot_interactive_image_map(ry=0, rx=0, show_polar=False)
    bp.plot_interactive_peak_map(ry=0, rx=0, show_polar=True)
    # zoom crops peaks from the Cartesian panel only; the polar panel still shows all of them.
    bp.plot_interactive_peak_map(ry=0, rx=0, show_polar=True, zoom=2)

    bp.save_diffraction_figures(0, 0, save_dir=tmp_path / "diff", show_polar=True)
    bp.save_peak_figures(0, 0, save_dir=tmp_path / "peaks", show_polar=True)
    expected = [
        f"diff/diffraction_ry0_rx0_{s}.pdf"
        for s in ("intensity_map", "diffraction", "polar", "combined")
    ]
    expected += [f"peaks/peaks_ry0_rx0_{s}.pdf" for s in ("intensity_map", "diffraction", "polar")]
    for path in expected:
        assert (tmp_path / path).exists(), path
    plt.close("all")
