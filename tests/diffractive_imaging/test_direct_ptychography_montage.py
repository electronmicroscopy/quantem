"""Tests for the real-space (shadow montage) direct-ptychography reconstruction.

The headline check is equivalence with the Fourier-space parallax kernel of
``DirectPtychography``: the two are the same linear operator written in different domains,
so on a raster scan with periodic wraparound they must agree.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import load
from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging import (
    DirectPtychography,
    DirectPtychographyMontage,
    OptimizationParameter,
)
from quantem.diffractive_imaging.complex_probe import FourierProbe, spatial_frequencies
from quantem.diffractive_imaging.direct_ptycho_utils import (
    allocate_splat_buffers,
    estimate_frame_drift,
    scatter_add_convolve,
    scatter_add_splat,
    splat_and_convolve,
    splat_stack,
)

from .conftest import (
    ACCELERATORS,
    CTF_SAMPLING,
    CTF_SCAN_SAMPLING,
    ORIGIN,
    PROBE_ENERGY,
    RECIPROCAL_SAMPLING,
    SCAN_SAMPLING,
    SEMIANGLE_CUTOFF,
    N,
    analytic_probe_array,
    band_limited_phase,
    correlation,
    ctf_interleaved_frames,
    ctf_kwargs,
    integer_shift_defocus,
    make_model_vbf_stack,
    make_tilted_dataset4d,
    model_vbf_kwargs,
    scan_positions_px,
)
from .conftest import (
    direct_ptycho_kwargs as _common_kwargs,
)

#: defocus and scan size at which the per-patch estimator is well conditioned; see
#: `make_model_vbf_stack` for why the 32x32 4D fixture is not
MODEL_DEFOCUS = 3000.0
MODEL_SCAN_GPTS = (96, 96)
MODEL_C10_GRID = np.linspace(1500.0, 4500.0, 13)


def _model_montage(defocus_gradient, defocus=MODEL_DEFOCUS):
    """A montage over a model vBF stack with a seeded defocus plane, plus the ground truth."""
    vbf, bf_mask, obj = make_model_vbf_stack(defocus, defocus_gradient, scan_gpts=MODEL_SCAN_GPTS)
    montage = DirectPtychographyMontage.from_virtual_bfs(vbf, bf_mask, **model_vbf_kwargs(defocus))
    return montage, obj


def _build_pair(dataset4d, defocus):
    """A `DirectPtychography` and a `DirectPtychographyMontage` over the same data."""
    fourier = DirectPtychography.from_dataset4d(
        dataset4d, edge_blend_pixels=0, **_common_kwargs(defocus)
    )
    montage = DirectPtychographyMontage.from_dataset4d(
        dataset4d, edge_blend_pixels=0, boundary="wrap", **_common_kwargs(defocus)
    )
    return fourier, montage


def _relative_error(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


class TestSplatKernel:
    """Unit tests for `scatter_add_splat`, independent of any reconstruction."""

    def test_integer_coordinates_deposit_one_pixel(self):
        values = torch.tensor([[2.0]])
        coords = torch.tensor([[[3.0, 4.0]]])
        sum_w, sum_wv, sum_wv2 = scatter_add_splat(values, coords, (8, 8))

        assert sum_w.sum().item() == pytest.approx(1.0)
        assert sum_w.reshape(8, 8)[3, 4].item() == pytest.approx(1.0)
        assert sum_wv.reshape(8, 8)[3, 4].item() == pytest.approx(2.0)
        assert sum_wv2.reshape(8, 8)[3, 4].item() == pytest.approx(4.0)

    def test_half_pixel_splits_four_ways(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.5, 4.5]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8))

        nonzero = sum_w[sum_w > 0]
        assert nonzero.numel() == 4
        assert torch.allclose(nonzero, torch.full((4,), 0.25, dtype=nonzero.dtype))

    def test_wrap_is_periodic(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[-0.5, 8.25]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), boundary="wrap")

        assert sum_w.sum().item() == pytest.approx(1.0)
        touched = {tuple(ij) for ij in torch.nonzero(sum_w.reshape(8, 8)).tolist()}
        assert touched == {(7, 0), (7, 1), (0, 0), (0, 1)}

    def test_pad_drops_out_of_bounds_without_corrupting_the_edge(self):
        values = torch.tensor([[100.0, 2.0]])
        coords = torch.tensor([[[-5.0, 4.0], [3.0, 4.0]]])
        sum_w, sum_wv, _ = scatter_add_splat(values, coords, (8, 8), boundary="pad")

        # only the in-bounds point contributes, and the clamped row picks up nothing
        assert sum_w.sum().item() == pytest.approx(1.0)
        assert sum_wv.sum().item() == pytest.approx(2.0)
        assert sum_w.reshape(8, 8)[0].max().item() == pytest.approx(0.0)

    def test_weights_are_a_partition_of_unity(self):
        generator = torch.Generator().manual_seed(0)
        values = torch.randn(4, 100, generator=generator)
        coords = torch.rand(4, 100, 2, generator=generator) * 8
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), boundary="wrap")

        assert sum_w.sum().item() == pytest.approx(400.0)

    def test_matches_brute_force(self):
        generator = torch.Generator().manual_seed(1)
        values = torch.randn(3, 50, generator=generator)
        coords = torch.rand(3, 50, 2, generator=generator) * 10 - 1

        sum_w, sum_wv, sum_wv2 = scatter_add_splat(values, coords, (10, 10), boundary="pad")

        # the splat accumulates in float64, so the reference must too
        values_np = to_numpy(values).astype(np.float64)
        coords_np = to_numpy(coords).astype(np.float64)
        ref_w, ref_wv, ref_wv2 = (np.zeros(100) for _ in range(3))
        for b in range(3):
            for t in range(50):
                r0 = int(np.floor(coords_np[b, t, 0]))
                c0 = int(np.floor(coords_np[b, t, 1]))
                fr = coords_np[b, t, 0] - r0
                fc = coords_np[b, t, 1] - c0
                for d_row, d_col in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    row, col = r0 + d_row, c0 + d_col
                    if not (0 <= row < 10 and 0 <= col < 10):
                        continue
                    weight = (fr if d_row else 1 - fr) * (fc if d_col else 1 - fc)
                    value = values_np[b, t]
                    ref_w[row * 10 + col] += weight
                    ref_wv[row * 10 + col] += weight * value
                    ref_wv2[row * 10 + col] += weight * value**2

        assert np.allclose(to_numpy(sum_w), ref_w)
        assert np.allclose(to_numpy(sum_wv), ref_wv)
        assert np.allclose(to_numpy(sum_wv2), ref_wv2)

    def test_nearest_rounds(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.4, 4.6]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), interpolation="nearest")

        assert torch.nonzero(sum_w.reshape(8, 8)).tolist() == [[3, 5]]

    def test_accumulates_into_provided_buffers(self):
        buffers = allocate_splat_buffers((8, 8), "cpu")
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.0, 4.0]]])

        for _ in range(3):
            scatter_add_splat(values, coords, (8, 8), out=buffers)

        assert buffers[0].reshape(8, 8)[3, 4].item() == pytest.approx(3.0)

    def test_rejects_unknown_modes(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.0, 4.0]]])
        with pytest.raises(ValueError, match="boundary"):
            scatter_add_splat(values, coords, (8, 8), boundary="reflect")
        with pytest.raises(ValueError, match="interpolation"):
            scatter_add_splat(values, coords, (8, 8), interpolation="cubic")

    def test_convolve_with_a_unit_tap_is_a_plain_splat(self):
        values = torch.tensor([[1.0, 2.0]])
        coords = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

        out = scatter_add_convolve(
            values,
            coords,
            (8, 8),
            torch.tensor([[0, 0]]),
            torch.ones(1, 1, dtype=torch.complex64),
        ).reshape(8, 8)

        assert out[3, 4].item() == pytest.approx(1.0)
        assert out[5, 6].item() == pytest.approx(2.0)

    def test_convolve_places_each_tap_at_its_offset(self):
        offsets = torch.tensor([[0, 0], [1, 0], [0, -2]])
        weights = torch.tensor([[1.0, 2.0j, -3.0]], dtype=torch.complex64)

        out = scatter_add_convolve(
            torch.tensor([[1.0]]), torch.tensor([[[4.0, 4.0]]]), (8, 8), offsets, weights
        ).reshape(8, 8)

        assert out[4, 4].item() == pytest.approx(1.0)
        assert out[5, 4].item() == pytest.approx(2.0j)
        assert out[4, 2].item() == pytest.approx(-3.0)
        assert int((out != 0).sum()) == 3

    def test_convolve_wraps_taps_at_the_boundary(self):
        out = scatter_add_convolve(
            torch.tensor([[1.0]]),
            torch.tensor([[[0.0, 0.0]]]),
            (8, 8),
            torch.tensor([[-1, -1]]),
            torch.ones(1, 1, dtype=torch.complex64),
            boundary="wrap",
        ).reshape(8, 8)

        assert out[7, 7].item() == pytest.approx(1.0)


class TestIntegerShiftConstruction:
    """The defocus used below must put every BF pixel on an exact canvas pixel."""

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    @pytest.mark.parametrize("pixel_shift", [1, 2])
    def test_shifts_land_on_pixel_centers(self, dataset4d, upsampling_factor, pixel_shift):
        defocus = integer_shift_defocus(pixel_shift, upsampling_factor)
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, boundary="wrap", **_common_kwargs(defocus)
        )
        shifts, _ = montage._return_shifts_px(
            0.0, {"C10": defocus}, montage.bf_mask, upsampling_factor
        )
        residual = (shifts - shifts.round()).abs().max().item()

        assert residual < 1e-4, f"shifts are not integral: max residual {residual:.2e} px"
        assert shifts.abs().max().item() > 1.0, "test would be vacuous with zero shifts"


class TestFourierEquivalence:
    """`DirectPtychographyMontage` must reproduce `DirectPtychography`'s parallax kernel."""

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    @pytest.mark.parametrize("pixel_shift", [1, 2])
    def test_matches_parallax_kernel(self, dataset4d, upsampling_factor, pixel_shift):
        defocus = integer_shift_defocus(pixel_shift, upsampling_factor)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            upsampling_factor=upsampling_factor,
            verbose=False,
        )
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(
            boundary="wrap", interpolation="bilinear", weight_normalize=False, **recon_kwargs
        )

        assert montage.obj.shape == fourier.obj.shape
        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_matches_with_phase_flip(self, dataset4d):
        """The phase-flip filter is BF-independent, so post-hoc application is exact."""
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        fourier.reconstruct(deconvolution_kernel="prlx", parallax_flip_phase=True, verbose=False)
        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=True,
            weight_normalize=False,
            verbose=False,
        )

        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_matches_with_butterworth_filters(self, dataset4d):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            q_lowpass=0.2,
            q_highpass=0.02,
            verbose=False,
        )
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(weight_normalize=False, **recon_kwargs)

        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_weight_normalization_differs_only_by_a_constant(self, dataset4d):
        """On a full grid at U=1 the accumulated weight is exactly num_bf everywhere."""
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        unnormalized = montage.obj
        bf_weights = float(montage._bf_weights)

        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=True,
            verbose=False,
        )
        normalized = montage.obj

        assert np.allclose(to_numpy(montage.weights), montage.num_bf, rtol=1e-6)
        rescaled = normalized * montage.num_bf / bf_weights
        assert _relative_error(rescaled, unnormalized) < 1e-4

    def test_variance_loss_tracks_the_fourier_one(self, dataset4d):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(deconvolution_kernel="prlx", parallax_flip_phase=False, verbose=False)
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(weight_normalize=False, **recon_kwargs)

        bf_weights = float(montage._bf_weights)
        expected = float(fourier.variance_loss()) * bf_weights**2
        assert float(montage.variance_loss()) == pytest.approx(expected, rel=1e-3)

    def test_bf_mask_subsets_are_additive(self, dataset4d):
        """Checkerboard half-sets, rescaled by their BF weights, sum to the whole."""
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        montage.reconstruct(**recon_kwargs)
        full = montage.obj * float(montage._bf_weights)

        halves = []
        for mask in montage._make_checkerboard_bf_masks(montage.gpts, montage.bf_mask):
            montage.reconstruct(bf_mask=mask, **recon_kwargs)
            halves.append(montage.obj * float(montage._bf_weights))

        assert _relative_error(halves[0] + halves[1], full) < 1e-4

    @pytest.mark.parametrize("kernel", ["ssb", "obf", "mf"])
    def test_accepts_the_deconvolution_kernels(self, dataset4d, kernel):
        """Available as truncated real-space convolutions; see `TestRealSpaceKernels`."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        obj = montage.reconstruct(deconvolution_kernel=kernel, stencil_radius=6, verbose=False).obj

        assert np.isfinite(obj).all()

    def test_icom_matches_the_fourier_class(self, dataset4d):
        """iCoM has no compact real-space form, but the FFT route never truncates."""
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        fourier.reconstruct(deconvolution_kernel="icom", verbose=False)
        montage.reconstruct(deconvolution_kernel="icom", convolution_mode="fft", verbose=False)

        assert _relative_error(montage.obj, fourier.obj) < 1e-5


class TestRotationConvention:
    """`spatial_frequencies` rotates the k-grid; shifts must follow it, unflipped."""

    def test_ninety_degrees_rotates_the_shifts(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)
        coefs = {"C10": defocus}

        unrotated, _ = montage._return_shifts_px(0.0, coefs, montage.bf_mask, 1)
        rotated, _ = montage._return_shifts_px(90.0, coefs, montage.bf_mask, 1)

        # _passively_rotate_grid sends (kx, ky) -> (kx cos a - ky sin a, kx sin a + ky cos a),
        # so at 90 deg the shifts, which are parallel to k for pure defocus, map (r, c) -> (-c, r)
        expected = torch.stack((-unrotated[:, 1], unrotated[:, 0]), dim=-1)
        assert torch.allclose(rotated, expected, atol=1e-4)

    def test_rotation_changes_the_reconstruction(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        montage.reconstruct(override_rotation_angle=0.0, verbose=False)
        unrotated = montage.obj.copy()
        montage.reconstruct(override_rotation_angle=30.0, verbose=False)

        assert not np.allclose(unrotated, montage.obj)


class TestReconstructDefaults:
    """The defaults are behaviour; pin them."""

    def test_interpolation_defaults_to_nearest(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        default = montage.reconstruct(upsampling_factor=2, verbose=False).obj.copy()
        nearest = montage.reconstruct(
            upsampling_factor=2, interpolation="nearest", verbose=False
        ).obj

        assert np.array_equal(default, nearest)

    def test_nearest_is_a_roll_of_the_bright_field_images(self, dataset4d):
        """On a raster scan, snapping moves every position of a BF image by one integer.

        `positions_px * U` is an exact integer, so `round(n + s) == n + round(s)`.
        """
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        upsampling_factor = 4
        shifts, _ = montage._return_shifts_px(
            0.0, montage.aberration_coefs, montage.bf_mask, upsampling_factor
        )
        positions = montage.positions_px * upsampling_factor
        coords = positions[None] + shifts[:, None]

        offsets = coords.round() - positions[None]
        # every position of a given BF image moves by the same integer
        assert torch.equal(offsets, shifts.round()[:, None].expand_as(offsets))

    def test_gridded_constructors_flag_the_scan_as_gridded(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        assert montage.gridded_scan is True

    def test_weight_normalize_defaults_off_for_a_raster_scan(self, dataset4d):
        """Uniform density needs no correction, and normalizing amplifies edge noise."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        default = montage.reconstruct(boundary="pad", verbose=False).obj.copy()
        unnormalized = montage.reconstruct(
            boundary="pad", weight_normalize=False, verbose=False
        ).obj

        assert np.array_equal(default, unnormalized)

    def test_weight_normalize_defaults_on_for_an_ungridded_scan(self, dataset4d):
        dataset3d, positions = TestNonGridScan._dataset3d_and_positions(dataset4d)
        recon = DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            positions,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            **_common_kwargs(integer_shift_defocus(1)),
        )

        assert recon.gridded_scan is False
        default = recon.reconstruct(verbose=False).obj.copy()
        normalized = recon.reconstruct(weight_normalize=True, verbose=False).obj
        assert np.array_equal(default, normalized)


class TestPadBoundary:
    """`boundary="pad"` grows the canvas instead of wrapping."""

    def test_interior_matches_wrap(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        montage.reconstruct(boundary="wrap", **recon_kwargs)
        wrapped = montage.obj

        shifts, _ = montage._return_shifts_px(0.0, {"C10": defocus}, montage.bf_mask, 1)
        margin = int(np.ceil(float(shifts.abs().max()))) + 1

        montage.reconstruct(boundary="pad", **recon_kwargs)
        padded = montage.obj
        row0, col0 = (-to_numpy(montage._canvas_origin_px)).astype(int)

        # far enough from every edge, the wrap modulo is a no-op and the two agree exactly
        n_rows, n_cols = wrapped.shape
        interior_wrap = wrapped[margin : n_rows - margin, margin : n_cols - margin]
        interior_pad = padded[
            row0 + margin : row0 + n_rows - margin, col0 + margin : col0 + n_cols - margin
        ]

        assert interior_pad.shape == interior_wrap.shape
        assert _relative_error(interior_pad, interior_wrap) < 1e-5

    def test_canvas_covers_the_shifted_positions(self, dataset4d):
        defocus = integer_shift_defocus(2)
        _, montage = _build_pair(dataset4d, defocus)
        montage.reconstruct(boundary="pad", verbose=False)

        # nothing was dropped: every (BF pixel, position) pair landed on the canvas
        assert float(montage.weights.sum()) == pytest.approx(
            montage.num_bf * montage.num_positions, rel=1e-6
        )
        assert montage.obj.shape[0] > N and montage.obj.shape[1] > N

    def test_pad_px_freezes_the_canvas(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        shapes = set()
        for pixel_shift in (1, 2):
            montage.reconstruct(
                override_aberration_coefs={"C10": integer_shift_defocus(pixel_shift)},
                boundary="pad",
                pad_px=12,
                verbose=False,
            )
            shapes.add(montage.obj.shape)

        assert len(shapes) == 1, f"canvas resized across trials: {shapes}"


class TestNonGridScan:
    """`from_dataset3d` with raster positions must reproduce the gridded path."""

    @staticmethod
    def _dataset3d_and_positions(dataset4d):
        n_scan = dataset4d.shape[0]
        dataset3d = Dataset3d.from_array(
            dataset4d.array.reshape(-1, N, N),
            name="synthetic 3D stack",
            units=("index", "A^-1", "A^-1"),
            sampling=(1, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        )
        positions = scan_positions_px()[: n_scan * n_scan] * SCAN_SAMPLING
        return dataset3d, positions

    def test_matches_from_dataset4d(self, dataset4d):
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        gridded = DirectPtychographyMontage.from_dataset4d(
            dataset4d, edge_blend_pixels=0, boundary="wrap", **_common_kwargs(defocus)
        )
        ungridded = DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            positions,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="wrap",
            **_common_kwargs(defocus),
        )

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        gridded.reconstruct(**recon_kwargs)
        ungridded.reconstruct(**recon_kwargs)

        assert ungridded.obj.shape == gridded.obj.shape
        assert _relative_error(ungridded.obj, gridded.obj) < 1e-5

    def test_position_axis_order_is_row_col(self, dataset4d):
        """Swapping the position columns must transpose the reconstruction."""
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        common = dict(
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="wrap",
            **_common_kwargs(defocus),
        )

        straight = DirectPtychographyMontage.from_dataset3d(dataset3d, positions, **common)
        swapped = DirectPtychographyMontage.from_dataset3d(
            dataset3d, positions[:, ::-1].copy(), **common
        )
        straight.reconstruct(**recon_kwargs)
        swapped.reconstruct(**recon_kwargs)

        assert not np.allclose(straight.obj, swapped.obj)

    def test_scattered_positions_reconstruct(self, dataset4d):
        """A jittered, shuffled scan still produces a supported montage."""
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        rng = np.random.default_rng(0)
        order = rng.permutation(len(positions))
        jittered = positions[order] + rng.normal(
            scale=0.25 * SCAN_SAMPLING, size=(len(positions), 2)
        )
        shuffled = Dataset3d.from_array(
            dataset3d.array[order],
            name="shuffled",
            units=dataset3d.units,
            sampling=dataset3d.sampling,
        )

        recon = DirectPtychographyMontage.from_dataset3d(
            shuffled,
            jittered,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="pad",
            **_common_kwargs(defocus),
        )
        recon.reconstruct(parallax_flip_phase=False, verbose=False)

        assert np.isfinite(recon.obj).all()
        assert float(recon.weights.max()) > 0

    def test_auto_scan_sampling_warns_and_infers(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        with pytest.warns(UserWarning, match="Inferred scan_sampling"):
            recon = DirectPtychographyMontage.from_dataset3d(
                dataset3d,
                positions,
                scan_sampling="auto",
                **_common_kwargs(integer_shift_defocus(1)),
            )

        assert recon.scan_sampling[0] == pytest.approx(SCAN_SAMPLING, rel=1e-6)

    def test_accepts_a_dataset2d_of_positions(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        positions_dataset = Dataset2d.from_array(positions, name="positions", units=("A", "A"))

        recon = DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            positions_dataset,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            **_common_kwargs(integer_shift_defocus(1)),
        )
        assert recon.num_positions == len(positions)

    def test_rejects_positions_in_the_wrong_units(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        positions_dataset = Dataset2d.from_array(
            positions, name="positions", units=("pixels", "pixels")
        )

        with pytest.raises(ValueError, match="must be given in 'A'"):
            DirectPtychographyMontage.from_dataset3d(
                dataset3d,
                positions_dataset,
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **_common_kwargs(integer_shift_defocus(1)),
            )

    def test_rejects_mismatched_position_count(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        with pytest.raises(ValueError, match="rows but `dataset` has"):
            DirectPtychographyMontage.from_dataset3d(
                dataset3d,
                positions[:-1],
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **_common_kwargs(integer_shift_defocus(1)),
            )

    def test_requires_an_explicit_rotation_angle(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        kwargs = _common_kwargs(integer_shift_defocus(1))
        kwargs["rotation_angle"] = None

        with pytest.raises(ValueError, match="must be given for non-raster scans"):
            DirectPtychographyMontage.from_dataset3d(
                dataset3d,
                positions,
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **kwargs,
            )


class TestHyperparameterSearch:
    def test_grid_search_recovers_the_seeded_defocus(self, dataset4d):
        true_defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, true_defocus)

        montage.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(
                    low=0.4 * true_defocus, high=1.6 * true_defocus, n_points=7
                )
            },
            parallax_flip_phase=False,
            verbose=False,
        )
        best = montage.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * true_defocus / 6
        assert abs(best - true_defocus) <= step

    def test_variance_loss_is_minimized_at_the_true_defocus(self, dataset4d):
        true_defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, true_defocus)

        losses = {}
        for scale in (0.5, 0.8, 1.0, 1.2, 1.5):
            montage.reconstruct(
                override_aberration_coefs={"C10": scale * true_defocus},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = float(montage.variance_loss())

        assert all(value > 0 for value in losses.values())
        assert min(losses, key=losses.get) == pytest.approx(1.0)

    def test_rms_gradient_drives_a_search_over_a_convolution_kernel(self, dataset4d):
        """`variance_loss` is undefined for `ssb` here; the sharpness objective is not.

        That is the reason it exists on this class: without it there is no way to tune
        aberrations for the real-space SSB/OBF/MF kernels at all.
        """
        true_defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, true_defocus)

        montage.reconstruct(deconvolution_kernel="ssb", stencil_radius=6, verbose=False)
        with pytest.raises(NotImplementedError):
            montage.variance_loss()

        montage.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(
                    low=0.4 * true_defocus, high=1.6 * true_defocus, n_points=5
                )
            },
            loss="rms_gradient",
            deconvolution_kernel="ssb",
            stencil_radius=6,
            verbose=False,
        )
        best = montage.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * true_defocus / 4
        assert abs(best - true_defocus) <= step


class TestSerialization:
    """Both classes must survive a save/load round-trip and stay usable afterwards."""

    @pytest.mark.parametrize("cls_name", ["fourier", "montage"])
    def test_round_trip(self, dataset4d, tmp_path, cls_name):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)
        recon = fourier if cls_name == "fourier" else montage

        recon.hyperparameter_state.optimized_aberrations = {"C10": 123.0}
        recon.hyperparameter_state.optimized_rotation_angle = 7.5
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)
        before = recon.obj.copy()

        path = str(tmp_path / f"{cls_name}.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert type(restored) is type(recon)
        assert np.array_equal(restored.obj, before)
        assert restored.hyperparameter_state.optimized_aberrations == {"C10": 123.0}
        assert restored.hyperparameter_state.optimized_rotation_angle == 7.5
        assert restored.gpts == recon.gpts
        assert float(restored.variance_loss()) == pytest.approx(float(recon.variance_loss()))

        # and it must still be able to reconstruct
        restored.reconstruct(deconvolution_kernel="prlx", verbose=False)
        assert np.allclose(restored.obj, before)

    def test_torch_size_attributes_round_trip_as_tuples(self, dataset4d):
        """`torch.Size` is a tuple subclass; AutoSerialize used to choke on the subclass name."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        assert type(montage.gpts) is tuple
        assert type(montage.scan_gpts) is tuple


class TestVisualization:
    """The object sampling, and hence `visualize`'s scalebar, must follow the upsampling."""

    @pytest.mark.parametrize("cls_name", ["fourier", "montage"])
    @pytest.mark.parametrize("upsampling_factor", [1, 2, 3])
    def test_scalebar_follows_the_upsampling(self, dataset4d, cls_name, upsampling_factor):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1, upsampling_factor))
        recon = fourier if cls_name == "fourier" else montage
        recon.reconstruct(
            deconvolution_kernel="prlx", upsampling_factor=upsampling_factor, verbose=False
        )

        expected = SCAN_SAMPLING / upsampling_factor
        assert recon._obj_sampling[0] == pytest.approx(expected)
        # the reported sampling must span the same field of view as the image itself
        assert recon.obj.shape[0] * recon._obj_sampling[0] == pytest.approx(N * SCAN_SAMPLING)

    def test_sampling_resets_between_reconstructions(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        montage.reconstruct(upsampling_factor=3, verbose=False)
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING / 3)
        montage.reconstruct(upsampling_factor=1, verbose=False)
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_sampling_is_defined_before_reconstructing(self, dataset4d):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        for recon in (fourier, montage):
            assert recon._obj_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_fov_matches_the_scan_extent(self, dataset4d):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        for recon in (fourier, montage):
            assert recon.fov == pytest.approx((N * SCAN_SAMPLING, N * SCAN_SAMPLING))

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    def test_padded_canvas_keeps_the_scan_sampling(self, dataset4d, upsampling_factor):
        """A padded canvas spans more than the scan, so its sampling is not fov/shape.

        Deriving from the *scan* field of view would under-report the pixel size by the
        padding fraction; `_obj_fov` reports the canvas extent instead.
        """
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1, upsampling_factor))
        montage.reconstruct(boundary="pad", upsampling_factor=upsampling_factor, verbose=False)

        assert montage.obj.shape[0] > N * upsampling_factor  # canvas really did grow
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING / upsampling_factor)
        assert montage._obj_fov[0] > montage.fov[0]
        # and the reported extent still matches the image it describes
        assert montage.obj.shape[0] * montage._obj_sampling[0] == pytest.approx(
            montage._obj_fov[0]
        )

    def test_wrapped_canvas_spans_exactly_the_scan(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        montage.reconstruct(boundary="wrap", upsampling_factor=2, verbose=False)

        assert montage._obj_fov == pytest.approx(montage.fov)

    def test_visualize_before_reconstruct_raises(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        with pytest.raises(RuntimeError, match="Run reconstruct"):
            montage.visualize()


class TestFourierProbe:
    """An empirical ``psi(k)`` in place of an aperture plus aberrations."""

    DEFOCUS = staticmethod(lambda: integer_shift_defocus(1))

    def _pair(self, dataset4d, cls, **probe_kwargs):
        """The same reconstruction twice: analytic, and with its own probe fed back in."""
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0)
        if cls is DirectPtychographyMontage:
            kwargs["boundary"] = "wrap"

        analytic = cls.from_dataset4d(dataset4d, **kwargs)
        psi = analytic_probe_array(analytic, {"C10": defocus})
        probe = FourierProbe.from_array(
            psi, analytic.reciprocal_sampling, analytic.wavelength, **probe_kwargs
        )
        empirical = cls.from_dataset4d(dataset4d, fourier_probe=probe, **kwargs)
        return analytic, empirical

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    @pytest.mark.parametrize("kernel", ["ssb", "obf", "mf"])
    def test_empirical_probe_reproduces_the_analytic_one(self, dataset4d, cls, kernel):
        """The headline: the same probe, described two ways, must reconstruct the same.

        This is what makes the empirical path trustworthy -- there is no ground truth for a
        measured probe, so the only check available is that a probe we *can* write down
        analytically goes through the array path unchanged.
        """
        analytic, empirical = self._pair(dataset4d, cls, normalize=False)
        extra = {} if cls is DirectPtychography else {"stencil_radius": 5}

        analytic.reconstruct(deconvolution_kernel=kernel, verbose=False, **extra)
        expected = analytic.obj.copy()
        empirical.reconstruct(deconvolution_kernel=kernel, verbose=False, **extra)

        assert _relative_error(empirical.obj, expected) < 1e-5

    def test_is_analytic_reports_which_path_is_live(self, dataset4d):
        analytic, empirical = self._pair(dataset4d, DirectPtychographyMontage)

        assert analytic.fourier_probe is None
        assert empirical.fourier_probe is not None
        assert empirical.fourier_probe.is_analytic is False

    def test_zero_outside_the_detector(self, dataset4d):
        """Beyond the detector's Nyquist nothing was measured, so `psi` is zero there.

        Wrapping instead would fold the opposite edge of the aperture back in, which moved
        the reconstruction by 13% on this fixture before it was fixed -- the bright-field
        mask is cropped tight to the disk, so `k -/+ q` leaves the grid constantly.
        """
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage)
        probe = empirical.fourier_probe
        dq_row, dq_col = probe.reciprocal_sampling
        n_rows = probe.array.shape[0]

        just_outside = torch.tensor([(n_rows // 2 + 1) * dq_row])
        zero = torch.zeros(1)

        assert probe.at(just_outside, zero).abs().item() == 0.0
        assert probe.at(-just_outside, zero).abs().item() == 0.0
        # and the centre is emphatically not zero
        assert probe.at(zero, zero).abs().item() > 0

    def test_resampling_is_exact_on_the_shared_lattice(self):
        """Zero-padding a compact real-space probe refines `psi` without changing it."""
        torch.manual_seed(0)
        real = torch.zeros(32, 32, dtype=torch.complex64)
        real[:8, :8] = torch.randn(8, 8, dtype=torch.complex64)  # confined, so band limited
        psi = torch.fft.fft2(real)
        probe = FourierProbe.from_array(psi, (0.25, 0.25), 0.02, normalize=False)

        refined = probe.resampled_to((0.0625, 0.0625))

        assert refined.array.shape == (128, 128)
        # every fourth sample of the fine grid is the original, exactly
        assert _relative_error(to_numpy(refined.array[::4, ::4]), to_numpy(psi)) < 1e-5

    def test_resampling_handles_anisotropy_and_odd_sizes(self):
        torch.manual_seed(0)
        real = torch.zeros(31, 16, dtype=torch.complex64)
        real[:6, :4] = torch.randn(6, 4, dtype=torch.complex64)
        psi = torch.fft.fft2(real)
        probe = FourierProbe.from_array(psi, (0.25, 0.5), 0.02, normalize=False)

        refined = probe.resampled_to((0.125, 0.125))

        assert refined.array.shape == (62, 64)
        assert _relative_error(to_numpy(refined.array[::2, ::4]), to_numpy(psi)) < 1e-5

    def test_resampling_rejects_a_non_integer_ratio(self):
        probe = FourierProbe.from_array(
            np.ones((8, 8), dtype=np.complex64), (0.25, 0.25), 0.02, normalize=False
        )
        with pytest.raises(ValueError, match="whole number"):
            probe.resampled_to((0.1, 0.1))

    def test_a_canvas_matching_the_probe_needs_no_resampling(self, dataset4d):
        """Canvas field of view == probe field of view is the exact, assumption-free case."""
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage, normalize=False)
        empirical.reconstruct(deconvolution_kernel="ssb", convolution_mode="fft", verbose=False)

        _, _, refined = empirical._resampled_probe
        assert refined.array.shape == empirical.fourier_probe.array.shape

    def test_refined_sampling_converges_on_the_exact_shift(self):
        """Bilinear on a refined grid must approach the exact band-limited evaluation.

        Isolated from any reconstruction, and on a probe whose real-space form really is
        confined, so that zero-padding is exact and only the interpolation is under test.
        Error falls as the square of the refinement.
        """
        torch.manual_seed(0)
        n = 64
        real = torch.zeros(n, n, dtype=torch.complex64)
        real[:10, :10] = torch.randn(10, 10, dtype=torch.complex64)
        psi = torch.fft.fft2(real)
        step = 0.25

        # ground truth at a generic sub-pixel offset: a phase ramp in real space
        drow, dcol = 0.31, 0.17
        ramp_axis = torch.fft.fftfreq(n)
        ramp = torch.exp(-2j * np.pi * (drow * ramp_axis[:, None] + dcol * ramp_axis[None, :]))
        truth = torch.fft.fft2(real * ramp)

        index = torch.arange(n)
        centered = torch.where(index < n // 2, index, index - n).to(torch.float32)
        kx = centered[:, None].expand(n, n) * step
        ky = centered[None, :].expand(n, n) * step

        errors = []
        for oversample in (1, 4, 16):
            probe = FourierProbe.from_array(
                psi, (step, step), 0.02, normalize=False, interpolation="bilinear"
            )
            if oversample > 1:
                probe = probe.resampled_to((step / oversample, step / oversample))
            sampled = probe.at(kx + drow * step, ky + dcol * step)
            errors.append(float((sampled - truth).abs().max() / truth.abs().max()))

        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 0.01

    @pytest.mark.parametrize("rotation_angle", [15.0, -30.0])
    def test_rotation_off_the_lattice_is_supported(self, dataset4d, rotation_angle):
        """A rotation carries `q` off the probe's lattice, which is interpolated, not refused.

        It is a real experimental parameter. What it costs is a sampling error that the
        refinement controls -- and, on this fixture, a floor of about 6% that no refinement
        removes: the aperture is cropped to an 11x11 grid, so its real-space probe is not
        confined and the band-limited interpolant is not the true probe. That floor is a
        property of the data, not of the method.
        """
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        kwargs["rotation_angle"] = rotation_angle

        analytic = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        psi = analytic_probe_array(analytic, {"C10": defocus})
        analytic.reconstruct(deconvolution_kernel="ssb", convolution_mode="fft", verbose=False)

        empirical = DirectPtychographyMontage.from_dataset4d(
            dataset4d,
            fourier_probe=FourierProbe.from_array(
                psi, analytic.reciprocal_sampling, analytic.wavelength, normalize=False
            ),
            **kwargs,
        )
        empirical.reconstruct(
            deconvolution_kernel="ssb", convolution_mode="fft", probe_oversample=8, verbose=False
        )

        assert np.isfinite(empirical.obj).all()
        assert correlation(empirical.obj, analytic.obj) > 0.99

    def test_low_oversampling_warns(self, dataset4d):
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        kwargs["rotation_angle"] = 15.0
        analytic = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        empirical = DirectPtychographyMontage.from_dataset4d(
            dataset4d,
            fourier_probe=FourierProbe.from_array(
                analytic_probe_array(analytic, {"C10": defocus}),
                analytic.reciprocal_sampling,
                analytic.wavelength,
            ),
            **kwargs,
        )

        with pytest.warns(UserWarning, match="off the probe's reciprocal lattice"):
            empirical.reconstruct(
                deconvolution_kernel="ssb",
                convolution_mode="fft",
                probe_oversample=2,
                verbose=False,
            )

    def test_off_grid_sampling_raises(self, dataset4d):
        """A canvas incommensurate with the probe would need interpolation; say so."""
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage)
        probe = empirical.fourier_probe
        half_pixel = torch.tensor([0.5 * probe.reciprocal_sampling[0]])

        with pytest.raises(ValueError, match="own reciprocal grid"):
            probe.at(half_pixel, torch.zeros(1))

    def test_bilinear_accepts_off_grid_sampling(self, dataset4d):
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage, interpolation="bilinear")
        probe = empirical.fourier_probe
        dq_row = probe.reciprocal_sampling[0]

        midpoint = probe.at(torch.tensor([0.5 * dq_row]), torch.zeros(1))
        ends = probe.at(torch.tensor([0.0, dq_row]), torch.zeros(2))

        assert midpoint.item() == pytest.approx(complex(ends.mean()), rel=1e-5)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"deconvolution_kernel": "prlx"}, "parallax kernel"),
            (
                {
                    "deconvolution_kernel": "ssb",
                    "stencil_radius": 5,
                    "defocus_gradient": (1.0, 0.0),
                },
                "defocus gradient",
            ),
        ],
    )
    def test_aberration_only_features_raise(self, dataset4d, kwargs, match):
        """Everything that reads chi(k) has no meaning without aberration coefficients."""
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage)

        with pytest.raises(NotImplementedError, match=match):
            empirical.reconstruct(verbose=False, **kwargs)

    def test_semiangle_cutoff_becomes_optional(self, dataset4d):
        """The empirical probe carries its own aperture, whatever shape it is."""
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        analytic = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        psi = analytic_probe_array(analytic, {"C10": defocus})

        kwargs["semiangle_cutoff"] = None
        empirical = DirectPtychographyMontage.from_dataset4d(
            dataset4d,
            fourier_probe=FourierProbe.from_array(
                psi, analytic.reciprocal_sampling, analytic.wavelength
            ),
            **kwargs,
        )

        assert empirical.semiangle_cutoff is None
        empirical.reconstruct(deconvolution_kernel="ssb", stencil_radius=5, verbose=False)
        assert np.isfinite(empirical.obj).all()

    def test_normalize_gives_unit_intensity(self, dataset4d):
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage, normalize=True)

        assert float(empirical.fourier_probe.array.abs().square().sum()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "array, match",
        [
            (np.ones((8, 8)), "must be a complex probe"),
            (np.ones((8, 8, 2), dtype=complex), "must be 2D"),
        ],
    )
    def test_array_validation(self, array, match):
        with pytest.raises(ValueError, match=match):
            FourierProbe.from_array(array, (0.1, 0.1), 0.02)

    def test_shape_must_match_the_detector(self, dataset4d):
        analytic, _ = self._pair(dataset4d, DirectPtychographyMontage)
        wrong = FourierProbe.from_array(
            np.ones((4, 4), dtype=np.complex64), analytic.reciprocal_sampling, analytic.wavelength
        )

        with pytest.raises(ValueError, match="detector grid"):
            analytic.fourier_probe = wrong

    def test_survives_a_round_trip(self, dataset4d, tmp_path):
        _, empirical = self._pair(dataset4d, DirectPtychographyMontage)
        empirical.reconstruct(deconvolution_kernel="ssb", stencil_radius=5, verbose=False)
        before = empirical.obj.copy()

        path = str(tmp_path / "empirical.zip")
        empirical.save(path, mode="o")
        restored = load(path)

        assert restored.fourier_probe is not None
        restored.reconstruct(deconvolution_kernel="ssb", stencil_radius=5, verbose=False)
        assert np.allclose(restored.obj, before)


class TestWavelength:
    """`wavelength` given directly, for anything that is not an electron."""

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    def test_energy_and_wavelength_agree(self, dataset4d, cls):
        """The two routes must land on the same geometry, and hence the same image."""
        common = dict(
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            aberration_coefs={"C10": integer_shift_defocus(1)},
            rotation_angle=0.0,
            force_fitted_origin=ORIGIN,
            edge_blend_pixels=0,
            verbose=False,
        )
        from_energy = cls.from_dataset4d(dataset4d, energy=PROBE_ENERGY, **common)
        from_wavelength = cls.from_dataset4d(
            dataset4d, wavelength=from_energy.wavelength, **common
        )

        assert from_wavelength.wavelength == pytest.approx(from_energy.wavelength)
        assert from_wavelength.angular_sampling == pytest.approx(from_energy.angular_sampling)

        for recon in (from_energy, from_wavelength):
            recon.reconstruct(deconvolution_kernel="prlx", verbose=False)
        assert np.array_equal(from_wavelength.obj, from_energy.obj)

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # neither
            {"energy": PROBE_ENERGY, "wavelength": 0.02},  # both
        ],
    )
    def test_exactly_one_of_energy_or_wavelength(self, dataset4d, cls, kwargs):
        with pytest.raises(ValueError, match="exactly one of `energy`"):
            cls.from_dataset4d(
                dataset4d,
                semiangle_cutoff=SEMIANGLE_CUTOFF,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
                **kwargs,
            )

    def test_photon_wavelength_is_not_the_electron_one(self):
        """The reason this exists: the de Broglie formula is wrong for photons.

        `electron_wavelength_angstrom` is relativistic and correct for electrons -- 0.019687
        Angstrom at 300 kV -- but it is the wrong physics for a 7.9 keV photon, where the
        answer is `hc/E` = 1.5694 rather than 0.1375.
        """
        from quantem.core.utils.utils import electron_wavelength_angstrom

        assert electron_wavelength_angstrom(300e3) == pytest.approx(0.019687, abs=1e-6)

        photon_energy_ev = 7900.08
        assert electron_wavelength_angstrom(photon_energy_ev) == pytest.approx(0.1375, abs=1e-3)
        assert 12398.42 / photon_energy_ev == pytest.approx(1.5694, abs=1e-3)

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    def test_wavelength_survives_a_round_trip(self, dataset4d, cls, tmp_path):
        recon = cls.from_dataset4d(
            dataset4d,
            wavelength=1.5694,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            force_fitted_origin=ORIGIN,
            edge_blend_pixels=0,
            verbose=False,
        )
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)

        path = str(tmp_path / f"{cls.__name__}.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert restored.wavelength == pytest.approx(1.5694)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_wavelength_raises(self, dataset4d, bad):
        with pytest.raises(ValueError):
            DirectPtychographyMontage.from_dataset4d(
                dataset4d,
                wavelength=bad,
                semiangle_cutoff=SEMIANGLE_CUTOFF,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
            )


class TestSemiangleCutoff:
    """`semiangle_cutoff` sets the probe aperture and is never optional."""

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    def test_from_virtual_bfs_requires_it(self, cls, dataset4d):
        """Enforced at runtime rather than by the signature.

        It used to be a required positional, but `wavelength` has to sit alongside `energy`
        with a default, and Python will not take a defaulted parameter before an
        undefaulted one. The guarantee is the error, so test the error.
        """
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        vbf_dataset = Dataset3d.from_array(
            to_numpy(montage.vbf_stack).reshape(montage.num_bf, N, N),
            name="vBF stack",
            units=("index", "A", "A"),
            sampling=(1, SCAN_SAMPLING, SCAN_SAMPLING),
        )
        bf_mask_dataset = Dataset2d.from_array(
            to_numpy(montage.bf_mask),
            name="BF mask",
            units=("A^-1", "A^-1"),
            sampling=tuple(montage.reciprocal_sampling),
        )

        with pytest.raises(ValueError, match="`semiangle_cutoff` is required"):
            cls.from_virtual_bfs(
                vbf_dataset,
                bf_mask_dataset,
                energy=PROBE_ENERGY,
                rotation_angle=0.0,
                crop_bf_mask=False,
                verbose=False,
            )

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    def test_none_raises_a_clear_error(self, dataset4d, cls):
        with pytest.raises(ValueError, match="`semiangle_cutoff` is required"):
            cls.from_dataset4d(
                dataset4d,
                energy=PROBE_ENERGY,
                semiangle_cutoff=None,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
            )

    @pytest.mark.parametrize("cls", [DirectPtychography, DirectPtychographyMontage])
    def test_non_positive_raises(self, dataset4d, cls):
        with pytest.raises(ValueError):
            cls.from_dataset4d(
                dataset4d,
                energy=PROBE_ENERGY,
                semiangle_cutoff=-1.0,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
            )


class TestDefocusGradient:
    """Position-dependent defocus, for a tilted sample.

    The montage shifts each scan position by its own local defocus. A Fourier multiplier is
    global over the scan by construction, so `DirectPtychography` has no counterpart to
    compare against; these check the model relation directly instead.
    """

    def test_none_and_zero_are_the_same_reconstruction(self, dataset4d):
        """The gradient must be a no-op when absent -- guards the fast path."""
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")

        without = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        with_zero = DirectPtychographyMontage.from_dataset4d(
            dataset4d, defocus_gradient=(0.0, 0.0), **kwargs
        )

        assert np.array_equal(
            without.reconstruct(verbose=False).obj,
            with_zero.reconstruct(verbose=False).obj,
        )

    def test_defocus_rate_is_the_analytic_lambda_k(self, dataset4d):
        """`d shift / d C10 = wavelength * k`, independent of the other aberrations."""
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        rate = montage._return_defocus_rate_px(0.0, montage.bf_mask, 1)

        kxa, kya = spatial_frequencies(montage.gpts, montage.sampling, device=montage.device)
        scan_sampling = torch.as_tensor(
            tuple(montage.scan_sampling), dtype=torch.float64, device=montage.device
        )
        expected = (
            torch.stack((kxa[montage.bf_mask], kya[montage.bf_mask]), -1).to(torch.float64)
            * montage.wavelength
            / scan_sampling
        )

        assert torch.allclose(rate, expected, atol=1e-9)

    def test_defocus_rate_ignores_other_aberrations(self, dataset4d):
        """chi is linear in every magnitude, so the rate cannot depend on the rest."""
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        rate = montage._return_defocus_rate_px(0.0, montage.bf_mask, 1)

        base = {"C10": 500.0, "C12": 40.0, "phi12": 0.7, "C30": 1.2e5}
        shifted = montage._return_shifts_px(0.0, {**base, "C10": 501.0}, montage.bf_mask, 1)[0]
        unshifted = montage._return_shifts_px(0.0, base, montage.bf_mask, 1)[0]

        assert torch.allclose(shifted - unshifted, rate, atol=1e-6)

    def test_delta_defocus_is_mean_zero(self, dataset4d):
        """Measuring from the centroid keeps the gradient orthogonal to the global C10."""
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        delta = montage._return_delta_c10((7.0, -3.0))

        assert delta is not None
        assert float(delta.mean().abs()) < 1e-9
        assert float(delta.abs().max()) > 0

    def test_zero_gradient_short_circuits(self, dataset4d):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        assert montage._return_delta_c10(None) is None
        assert montage._return_delta_c10((0.0, 0.0)) is None

    def test_padded_canvas_covers_the_gradient(self, dataset4d):
        """A gradient widens the range of shifts, so `"pad"` must grow to match."""
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0)

        flat = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        tilted = DirectPtychographyMontage.from_dataset4d(
            dataset4d, defocus_gradient=(30.0, -10.0), **kwargs
        )

        flat_shape = flat.reconstruct(boundary="pad", verbose=False).obj.shape
        tilted_shape = tilted.reconstruct(boundary="pad", verbose=False).obj.shape

        assert tilted_shape[0] > flat_shape[0]
        assert tilted_shape[1] > flat_shape[1]

    def test_shift_extrema_match_a_brute_force_scan(self, dataset4d):
        """The closed form must bound every (BF pixel, position) pair, with no slack."""
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        gradient = (30.0, -10.0)
        shifts = montage._return_shifts_px(0.0, montage.aberration_coefs, montage.bf_mask, 1)[0]
        rate = montage._return_defocus_rate_px(0.0, montage.bf_mask, 1)
        delta = montage._return_delta_c10(gradient)

        lo, hi = DirectPtychographyMontage._return_shift_extrema(shifts, rate, delta)
        brute = shifts[:, None, :] + rate[:, None, :] * delta[None, :, None]

        assert torch.allclose(lo, brute.amin((0, 1)))
        assert torch.allclose(hi, brute.amax((0, 1)))

    def test_sign_convention_on_simulated_tilted_data(self):
        """On real 4D data a negated gradient must be worse than the true one.

        Only the *ordering* is asserted. At this fixture's 32 Angstrom field of view a
        visible gradient needs a defocus swing so large that the probe size varies threefold
        across the scan, and the parallax model itself starts to break down -- so
        "better than no correction at all" is not true here, and is checked on the model
        stack below instead.
        """
        defocus = integer_shift_defocus(1)
        gradient = (40.0, 0.0)
        dataset = make_tilted_dataset4d(defocus, gradient)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0)

        def corr(g):
            montage = DirectPtychographyMontage.from_dataset4d(
                dataset, defocus_gradient=g, **kwargs
            )
            return correlation(montage.reconstruct(verbose=False).obj, band_limited_phase())

        assert corr(gradient) > corr((-gradient[0], -gradient[1]))

    @pytest.mark.parametrize("gradient", [(20.0, 0.0), (20.0, -10.0), (-15.0, 25.0)])
    def test_correcting_the_gradient_sharpens_the_reconstruction(self, gradient):
        montage, obj = _model_montage(gradient)
        reconstruct = dict(parallax_flip_phase=False, interpolation="bilinear", verbose=False)

        uncorrected = correlation(montage.reconstruct(**reconstruct).obj, obj)
        corrected = correlation(
            montage.reconstruct(defocus_gradient=gradient, **reconstruct).obj, obj
        )

        assert corrected > uncorrected
        assert corrected > 0.98

    def test_defocus_map_tracks_a_seeded_plane(self):
        gradient = (20.0, -10.0)
        montage, _ = _model_montage(gradient)

        results = montage.defocus_map(
            MODEL_C10_GRID, patch_grid=(3, 3), interpolation="bilinear", verbose=False
        )
        expected = MODEL_DEFOCUS + results["centers_A"] @ np.asarray(gradient)

        assert results["valid"].all()
        # the estimator carries a small uniform offset (~170 A, see the flat control below),
        # so compare the spatial variation rather than the absolute value.
        # 0.97 rather than 0.99: scoring only pixels near the peak accumulated weight used
        # to reach 0.995 on this gridded fixture, but on a real ungridded scan the weight is
        # uneven everywhere and that cut selects density hot-spots instead of the interior --
        # see `_patch_variance_loss`. The weight-averaged mean is unbiased on both.
        recovered = results["c10_best"] - results["c10_best"].mean()
        assert np.corrcoef(recovered, expected - expected.mean())[0, 1] > 0.97

    def test_defocus_map_is_flat_without_a_gradient(self):
        """The control that makes the test above meaningful."""
        montage, _ = _model_montage((0.0, 0.0))

        results = montage.defocus_map(
            MODEL_C10_GRID, patch_grid=(3, 3), interpolation="bilinear", verbose=False
        )

        assert np.ptp(results["c10_best"]) < 0.05 * MODEL_DEFOCUS

    @pytest.mark.parametrize("gradient", [(0.0, 0.0), (20.0, -10.0), (-15.0, 25.0)])
    def test_fit_defocus_gradient_recovers_the_seed(self, gradient):
        montage, _ = _model_montage(gradient)

        montage.fit_defocus_gradient(
            MODEL_C10_GRID, patch_grid=(3, 3), interpolation="bilinear", verbose=False
        )

        assert montage.defocus_gradient is not None
        scale = max(np.hypot(*gradient), 1.0)
        assert np.hypot(*np.subtract(montage.defocus_gradient, gradient)) < 0.15 * scale

    def test_fit_defocus_gradient_updates_the_global_defocus(self):
        montage, _ = _model_montage((20.0, 0.0))
        montage.hyperparameter_state.optimized_aberrations = {}

        montage.fit_defocus_gradient(
            MODEL_C10_GRID, patch_grid=(3, 3), interpolation="bilinear", verbose=False
        )

        assert "C10" in montage.hyperparameter_state.optimized_aberrations
        assert "C10" in montage.hyperparameter_state.optimized_keys

    def test_fit_defocus_gradient_can_leave_the_defocus_alone(self):
        montage, _ = _model_montage((20.0, 0.0))

        montage.fit_defocus_gradient(
            MODEL_C10_GRID,
            patch_grid=(3, 3),
            interpolation="bilinear",
            update_defocus=False,
            verbose=False,
        )

        assert montage.hyperparameter_state.optimized_aberrations == {}

    def test_endpoint_pinned_patches_are_invalid(self):
        """A grid that does not bracket the local defocus must be reported, not fitted."""
        montage, _ = _model_montage((20.0, 0.0))

        results = montage.defocus_map(
            np.linspace(3400.0, 4500.0, 6),
            patch_grid=(3, 3),
            interpolation="bilinear",
            verbose=False,
        )

        assert not results["valid"].all()
        assert np.isnan(results["c10_best"][~results["valid"]]).all()

    def test_fit_raises_when_too_few_patches_bracket(self):
        montage, _ = _model_montage((20.0, 0.0))

        with pytest.raises(RuntimeError, match="bracketed minimum"):
            montage.fit_defocus_gradient(
                np.linspace(4200.0, 4500.0, 4),
                patch_grid=(2, 2),
                interpolation="bilinear",
                verbose=False,
            )

    def test_defocus_map_allows_a_one_dimensional_grid(self):
        """A (P, 1) grid is a profile along one axis -- only the plane fit needs three."""
        montage, _ = _model_montage((20.0, 0.0))

        results = montage.defocus_map(
            MODEL_C10_GRID, patch_grid=(3, 1), interpolation="bilinear", verbose=False
        )

        assert results["c10_best"].shape == (3,)

    def test_defocus_map_rejects_a_degenerate_grid(self):
        montage, _ = _model_montage((0.0, 0.0))

        with pytest.raises(ValueError, match="must be positive"):
            montage.defocus_map(MODEL_C10_GRID, patch_grid=(0, 3), verbose=False)

    def test_defocus_map_needs_enough_trial_values(self):
        montage, _ = _model_montage((0.0, 0.0))

        with pytest.raises(ValueError, match="at least 3 points"):
            montage.defocus_map([2000.0, 3000.0], patch_grid=(2, 2), verbose=False)

    def test_gradient_is_orthogonal_to_a_global_defocus_search(self):
        """The API worry: a grid search over C10 must stay well posed with a gradient set."""
        gradient = (20.0, -10.0)
        montage, _ = _model_montage(gradient)

        montage.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(
                    low=MODEL_DEFOCUS - 900, high=MODEL_DEFOCUS + 900, n_points=7
                )
            },
            defocus_gradient=gradient,
            interpolation="bilinear",
            parallax_flip_phase=False,
            verbose=False,
        )

        fitted = montage.hyperparameter_state.current_aberrations()["C10"]
        assert abs(fitted - MODEL_DEFOCUS) < 0.2 * MODEL_DEFOCUS

    def test_rejects_a_malformed_gradient(self, dataset4d):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, **_common_kwargs(integer_shift_defocus(1))
        )
        with pytest.raises(ValueError, match="must be a \\(row, col\\) pair"):
            montage.defocus_gradient = (1.0, 2.0, 3.0)

    def test_survives_a_serialization_round_trip(self, dataset4d, tmp_path):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, defocus_gradient=(7.0, -3.0), **_common_kwargs(integer_shift_defocus(1))
        )
        path = tmp_path / "montage.zip"
        montage.save(path, mode="o")

        assert load(path).defocus_gradient == (7.0, -3.0)


class TestRealSpaceKernels:
    """SSB / OBF / MF as truncated real-space convolutions.

    `obj = sum_m ifft2(G_m * K_m) = sum_m (v_m conv kappa_m)` is an identity, so truncating
    `kappa_m` to a box stencil is the *only* approximation: with a large enough stencil these
    must reproduce `DirectPtychography` exactly.

    They converge slowly. Dividing by `|gamma|` leaves a unit-magnitude phase on a hard-edged
    support, whose transform has `r**-1.5` tails, so the error falls like 1/radius -- and
    being in focus does not help, since there is no chirp to concentrate the kernel.
    """

    KERNELS = ("ssb", "obf", "mf")
    MAX_RADIUS = 15  # canvas half-width for the 32x32 fixture

    @staticmethod
    def _pair(dataset4d):
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0)
        return (
            DirectPtychography.from_dataset4d(dataset4d, **kwargs),
            DirectPtychographyMontage.from_dataset4d(dataset4d, boundary="wrap", **kwargs),
        )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_converges_to_the_fourier_kernel(self, dataset4d, kernel):
        fourier, montage = self._pair(dataset4d)
        reference = fourier.reconstruct(deconvolution_kernel=kernel, verbose=False).obj

        errors = []
        for radius in (4, 8, self.MAX_RADIUS):
            obj = montage.reconstruct(
                deconvolution_kernel=kernel,
                stencil_radius=radius,
                interpolation="nearest",
                verbose=False,
            ).obj
            errors.append(_relative_error(obj, reference))

        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 2e-2

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_reported_error_bounds_the_measured_one(self, dataset4d, kernel):
        """The estimate assumes a white object spectrum, so it must be conservative."""
        fourier, montage = self._pair(dataset4d)
        reference = fourier.reconstruct(deconvolution_kernel=kernel, verbose=False).obj

        obj = montage.reconstruct(
            deconvolution_kernel=kernel,
            stencil_radius=8,
            interpolation="nearest",
            verbose=False,
        ).obj

        assert _relative_error(obj, reference) < montage._stencil_info["mean_error"]

    def test_a_box_stencil_beats_a_tapered_one(self, dataset4d):
        """Pins a counterintuitive choice: tapering the stencil measures consistently worse.

        A Hann taper discards mid-radius content that matters more than the ringing it
        suppresses, so do not "improve" the box away.
        """
        fourier, montage = self._pair(dataset4d)
        reference = fourier.reconstruct(deconvolution_kernel="ssb", verbose=False).obj
        radius = 8

        boxed = montage.reconstruct(
            deconvolution_kernel="ssb",
            stencil_radius=radius,
            interpolation="nearest",
            verbose=False,
        ).obj

        # rebuild the same stencil, taper it, and run the same accumulation by hand
        bf = montage._return_bf_context(montage.bf_mask)
        shifts, _ = montage._return_shifts_px(0.0, montage.aberration_coefs, bf.bf_mask, 1)
        offsets, weights, bf_weights, _ = montage._return_kernel_stencil(
            bf,
            kernel="ssb",
            rotation_angle=0.0,
            aberration_coefs=montage.aberration_coefs,
            canvas_shape=montage._canvas_shape,
            upsampling_factor=1,
            shift_centers=shifts.round(),
            stencil_radius=radius,
            truncation_tolerance=1.0,
            max_stencil_radius=radius,
            matched_filter_norm_epsilon=1e-1,
            kernel_batch_size=16,
            verbose=False,
        )
        distance = offsets.to(torch.float64).abs().amax(-1)
        taper = 0.5 * (1 + torch.cos(np.pi * distance / (radius + 1)))
        canvas_shape = montage._canvas_shape

        accumulator = torch.zeros(
            canvas_shape[0] * canvas_shape[1], device=montage.device, dtype=torch.complex64
        )
        coords = montage.positions_px[None] + shifts.round()[:, None]
        scatter_add_convolve(
            montage.vbf_stack,
            coords,
            canvas_shape,
            offsets,
            weights * taper[None, :],
            boundary="wrap",
            interpolation="nearest",
            out=accumulator,
        )
        tapered = to_numpy(accumulator.real.reshape(canvas_shape) / bf_weights)

        assert _relative_error(boxed, reference) < _relative_error(tapered, reference)

    def test_warns_when_the_tolerance_cannot_be_met(self, dataset4d):
        _, montage = self._pair(dataset4d)

        with pytest.warns(UserWarning, match="truncation error"):
            montage.reconstruct(deconvolution_kernel="ssb", stencil_radius=2, verbose=False)

    def test_auto_respects_the_radius_cap(self, dataset4d):
        _, montage = self._pair(dataset4d)

        with pytest.warns(UserWarning, match="truncation error"):
            montage.reconstruct(
                deconvolution_kernel="ssb",
                convolution_mode="stencil",
                stencil_radius="auto",
                max_stencil_radius=3,
                verbose=False,
            )

        assert montage._stencil_info["stencil_radius"] <= 3

    def test_auto_reports_what_it_chose(self, dataset4d):
        _, montage = self._pair(dataset4d)
        montage.reconstruct(deconvolution_kernel="ssb", convolution_mode="stencil", verbose=False)

        info = montage._stencil_info
        assert info["stencil_radius"] >= 1
        assert 0.0 <= info["mean_error"] <= info["max_error"]

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_variance_loss_is_undefined(self, dataset4d, kernel):
        _, montage = self._pair(dataset4d)
        montage.reconstruct(deconvolution_kernel=kernel, stencil_radius=4, verbose=False)

        with pytest.raises(NotImplementedError, match="only defined for the parallax"):
            montage.variance_loss()

    def test_variance_loss_returns_after_a_parallax_reconstruction(self, dataset4d):
        _, montage = self._pair(dataset4d)
        montage.reconstruct(deconvolution_kernel="ssb", stencil_radius=4, verbose=False)
        montage.reconstruct(verbose=False)

        assert float(montage.variance_loss()) > 0

    def test_icom_runs_in_either_mode(self, dataset4d):
        _, montage = self._pair(dataset4d)
        for mode in ("fft", "stencil"):
            montage.reconstruct(
                deconvolution_kernel="icom",
                convolution_mode=mode,
                stencil_radius=8,
                verbose=False,
            )
            assert np.isfinite(montage.obj).all()

    def test_icom_ignores_an_empirical_probe(self, dataset4d):
        """It never reads psi, so an empirical probe changes nothing -- not even a bit.

        `normalize=False` matters only because the probe's total intensity sets
        `bf_weights`, which scales the finished object; the image itself is identical
        either way.
        """
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        analytic = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        empirical = DirectPtychographyMontage.from_dataset4d(
            dataset4d,
            fourier_probe=FourierProbe.from_array(
                analytic_probe_array(analytic, {"C10": defocus}),
                analytic.reciprocal_sampling,
                analytic.wavelength,
                normalize=False,
            ),
            **kwargs,
        )
        for recon in (analytic, empirical):
            recon.reconstruct(deconvolution_kernel="icom", convolution_mode="fft", verbose=False)

        assert np.array_equal(empirical.obj, analytic.obj)

    def test_truncated_icom_is_ricom(self):
        """A truncated iCoM stencil *is* riCOM (Yu et al., Microsc Microanal 28, 1526).

        riCOM cross-correlates the centre-of-mass shift map with a kernel
        `(r_p - r_xy) / |r_p - r_xy|**2` truncated to an n x n box. That kernel is the
        real-space form of the iCoM operator: for `G = ln|r| / 2pi`, `grad G = r / (2 pi
        |r|**2)`, whose transform is `-i q / |q|**2`. Since each bright-field pixel's kernel
        is linear in `k_m`, summing over the detector collapses the montage's per-pixel
        convolutions into one convolution of the COM shift -- which is riCOM exactly.

        Two consequences, both checked below: a kernel spanning the canvas reproduces
        untruncated iCoM, and shrinking it acts as a high pass. The latter is riCOM's whole
        point -- it is what suppresses the long-range drift that blurs an iCoM image.
        """
        vbf, bf_mask, _ = make_model_vbf_stack(MODEL_DEFOCUS, (0.0, 0.0), scan_gpts=(96, 96))
        montage = DirectPtychographyMontage.from_virtual_bfs(
            vbf, bf_mask, **model_vbf_kwargs(MODEL_DEFOCUS)
        )

        size = montage.scan_gpts[0]
        axis = np.minimum(np.arange(size), size - np.arange(size))
        radius = np.hypot(axis[:, None], axis[None, :]).astype(int)

        def bands(image):
            spectrum = np.abs(np.fft.fft2(image - image.mean()))
            profile = np.bincount(radius.ravel(), spectrum.ravel()) / np.maximum(
                np.bincount(radius.ravel()), 1
            )
            return profile[1:4].mean(), profile[10:25].mean()

        common = dict(deconvolution_kernel="icom", boundary="wrap", verbose=False)
        montage.reconstruct(convolution_mode="fft", **common)
        untruncated = montage.obj.copy()

        ratios = []
        for stencil_radius in (5, 10, 20, 40):
            montage.reconstruct(
                convolution_mode="stencil", stencil_radius=stencil_radius, **common
            )
            low, high = bands(montage.obj)
            ratios.append(high / low)
            if stencil_radius == 40:  # spans the canvas, so nothing is truncated away
                assert correlation(montage.obj, untruncated) > 0.99

        # smaller kernel -> more weight at high frequency, monotonically
        assert ratios == sorted(ratios, reverse=True)
        assert ratios[0] > 1.2 * ratios[-1]

    def test_collapsing_the_detector_changes_nothing(self, dataset4d):
        """riCOM sums over the detector first; that must be an optimization, not a change.

        The per-detector-pixel route still runs when a defocus gradient gives each position
        its own shift, which the collapse cannot represent -- so a gradient small enough to
        be physically negligible exercises the slow path and the two must agree.
        """
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        montage = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)

        montage.reconstruct(deconvolution_kernel="icom", convolution_mode="fft", verbose=False)
        collapsed = montage.obj.copy()
        montage.reconstruct(
            deconvolution_kernel="icom",
            convolution_mode="fft",
            defocus_gradient=(1e-9, 0.0),
            verbose=False,
        )

        assert _relative_error(montage.obj, collapsed) < 1e-5

    def test_com_shift_is_the_k_weighted_first_moment(self, dataset4d):
        """The collapse itself, against an explicit sum over the detector."""
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0, boundary="wrap")
        montage = DirectPtychographyMontage.from_dataset4d(dataset4d, **kwargs)
        bf = montage._return_bf_context(montage.bf_mask)

        collapsed = montage._return_com_shift(bf, 0.0, 1)

        kxa, kya, _, _ = montage._return_k_grid(0.0)
        k_vectors = torch.stack((kxa[bf.bf_mask], kya[bf.bf_mask]), dim=-1)
        values = montage._vbf_stack[bf.vbf_index_mapping]
        expected = torch.stack([(values * k_vectors[:, i : i + 1]).sum(0) for i in (0, 1)])

        assert _relative_error(to_numpy(collapsed), to_numpy(expected)) < 1e-5

    def test_the_icom_kernel_is_the_ricom_kernel(self):
        """`ifft2(-i q / |q|**2)` is `r / (2 pi |r|**2)`, the kernel riCOM writes down."""
        size = 128
        frequency = np.fft.fftfreq(size)
        qx, qy = frequency[:, None], frequency[None, :]
        q_square = qx**2 + qy**2
        q_square[0, 0] = 1.0
        operator = -1j * qx / q_square
        operator[0, 0] = 0.0
        transformed = np.real(np.fft.ifft2(operator))

        position = np.fft.fftfreq(size, 1 / size)
        rx, ry = position[:, None], position[None, :]
        r_square = rx**2 + ry**2
        r_square[0, 0] = 1.0
        analytic = rx / (2 * np.pi * r_square)

        # away from the singular origin and the periodic seam
        distance = np.hypot(rx, ry)
        inside = (distance > 3) & (distance < size / 4)
        assert np.corrcoef(transformed[inside], analytic[inside])[0, 1] > 0.98

    def test_phase_flip_is_not_applied_to_deconvolution_kernels(self, dataset4d):
        """The kernels already invert the contrast transfer, as in `DirectPtychography`."""
        _, montage = self._pair(dataset4d)
        common = dict(deconvolution_kernel="ssb", stencil_radius=4, verbose=False)

        flipped = montage.reconstruct(parallax_flip_phase=True, **common).obj
        unflipped = montage.reconstruct(parallax_flip_phase=False, **common).obj

        assert np.array_equal(flipped, unflipped)

    def test_stencil_covers_the_parallax_shift_without_growing(self, dataset4d):
        """The shift is divided out of the kernel, so the stencil sizes the residual only."""
        _, montage = self._pair(dataset4d)
        montage.reconstruct(deconvolution_kernel="ssb", convolution_mode="stencil", verbose=False)
        modest_defocus = montage._stencil_info["stencil_radius"]

        montage.reconstruct(
            deconvolution_kernel="ssb",
            convolution_mode="stencil",
            override_aberration_coefs={"C10": integer_shift_defocus(3)},
            verbose=False,
        )

        assert montage._stencil_info["stencil_radius"] <= modest_defocus + 2


class TestConvolutionModes:
    """`convolution_mode` picks how the SSB / OBF / MF convolutions are evaluated.

    `"fft"` multiplies in `q`, which is exact; `"stencil"` truncates to a box, which is not.
    Both are the same operator, so the only differences that may appear are the truncation
    and the circular-versus-linear boundary.
    """

    KERNELS = ("ssb", "obf", "mf")

    @staticmethod
    def _pair(dataset4d, boundary="wrap"):
        defocus = integer_shift_defocus(1)
        kwargs = dict(_common_kwargs(defocus), edge_blend_pixels=0)
        return (
            DirectPtychography.from_dataset4d(dataset4d, **kwargs),
            DirectPtychographyMontage.from_dataset4d(dataset4d, boundary=boundary, **kwargs),
        )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_fft_mode_is_exact(self, dataset4d, kernel):
        """The headline: no truncation at all, so it must equal the Fourier class.

        For contrast, the truncated stencil on this fixture is 20-34% off at radius 5 and
        still 2-5% off at radius 12 -- these kernels are not compact and never converge fast.
        """
        fourier, montage = self._pair(dataset4d)
        fourier.reconstruct(deconvolution_kernel=kernel, verbose=False)
        montage.reconstruct(deconvolution_kernel=kernel, convolution_mode="fft", verbose=False)

        assert _relative_error(montage.obj, fourier.obj) < 1e-5

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_fft_mode_beats_a_truncated_stencil(self, dataset4d, kernel):
        fourier, montage = self._pair(dataset4d)
        fourier.reconstruct(deconvolution_kernel=kernel, verbose=False)
        reference = fourier.obj.copy()

        montage.reconstruct(deconvolution_kernel=kernel, convolution_mode="fft", verbose=False)
        exact = _relative_error(montage.obj, reference)
        montage.reconstruct(
            deconvolution_kernel=kernel,
            convolution_mode="stencil",
            stencil_radius=5,
            verbose=False,
        )
        truncated = _relative_error(montage.obj, reference)

        assert exact < 0.01 * truncated

    def test_pad_boundary_stays_linear(self, dataset4d):
        """A Fourier convolution wraps; `"pad"` doubles the canvas so that it does not.

        Compared against a stencil wide enough that its own truncation is the larger error.
        """
        _, montage = self._pair(dataset4d, boundary="pad")
        montage.reconstruct(
            deconvolution_kernel="ssb",
            convolution_mode="stencil",
            stencil_radius=14,
            pad_px=4,
            verbose=False,
        )
        stencil = montage.obj.copy()
        montage.reconstruct(
            deconvolution_kernel="ssb", convolution_mode="fft", pad_px=4, verbose=False
        )

        assert montage.obj.shape == stencil.shape
        assert _relative_error(montage.obj, stencil) < 0.05

    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
    @pytest.mark.parametrize("boundary", ["wrap", "pad"])
    def test_conv2d_stencil_matches_the_scatter(self, dataset4d, boundary, interpolation):
        """`splat_and_convolve` is a reorganization of `scatter_add_convolve`, not a change.

        The subtlety it has to reproduce is that the scatter tests the boundary at the
        *deposit* position, so a point outside the canvas still contributes inward through
        the kernel -- which is why the splat happens on a canvas grown by the radius.
        """
        torch.manual_seed(0)
        radius = 3
        span = torch.arange(-radius, radius + 1)
        offsets = torch.stack(torch.meshgrid(span, span, indexing="ij"), dim=-1).reshape(-1, 2)
        weights = torch.randn(4, offsets.shape[0], dtype=torch.complex64)
        values = torch.randn(4, 30)
        # deliberately spill outside the canvas on both sides
        coords = torch.rand(4, 30, 2) * torch.tensor([28.0, 24.0]) - torch.tensor([4.0, 4.0])
        shape = (20, 16)

        scattered = torch.zeros(shape[0] * shape[1], dtype=torch.complex64)
        scatter_add_convolve(
            values,
            coords,
            shape,
            offsets,
            weights,
            boundary=boundary,
            interpolation=interpolation,
            out=scattered,
        )
        convolved = splat_and_convolve(
            values,
            coords,
            shape,
            weights,
            radius,
            boundary=boundary,
            interpolation=interpolation,
        ).sum(0)

        assert _relative_error(to_numpy(convolved), to_numpy(scattered.view(shape))) < 1e-5

    def test_splat_stack_matches_a_shared_canvas(self):
        """Summing the per-image canvases must give what the shared-canvas splat gives."""
        torch.manual_seed(1)
        values = torch.randn(6, 25)
        coords = torch.rand(6, 25, 2) * 12.0
        shape = (12, 12)

        stack = splat_stack(values, coords, shape, boundary="wrap", interpolation="bilinear")
        _, shared, _ = scatter_add_splat(
            values, coords, shape, boundary="wrap", interpolation="bilinear"
        )

        assert _relative_error(to_numpy(stack.sum(0)), to_numpy(shared.view(shape))) < 1e-5

    @pytest.mark.parametrize("stencil_radius, expected", [("auto", "fft"), (6, "stencil")])
    def test_auto_reads_the_stencil_radius(self, stencil_radius, expected):
        """Naming a radius is a request to truncate; leaving it 'auto' takes the exact route."""
        resolve = DirectPtychographyMontage._resolve_convolution_mode
        assert resolve("auto", "ssb", stencil_radius) == expected

    def test_parallax_ignores_the_mode(self):
        resolve = DirectPtychographyMontage._resolve_convolution_mode
        assert resolve("fft", "prlx", "auto") == "splat"

    def test_unknown_mode_raises(self, dataset4d):
        _, montage = self._pair(dataset4d)
        with pytest.raises(ValueError, match="must be 'auto', 'fft' or 'stencil'"):
            montage.reconstruct(
                deconvolution_kernel="ssb", convolution_mode="direct", verbose=False
            )

    def test_fft_mode_leaves_no_stencil_info(self, dataset4d):
        _, montage = self._pair(dataset4d)
        montage.reconstruct(deconvolution_kernel="ssb", convolution_mode="fft", verbose=False)

        assert montage._stencil_info is None


class TestSharedCanvas:
    """`obj_origin` / `obj_fov` pin the canvas to a window of the *specimen*.

    Without them the canvas follows each acquisition's own position bounding box, so two
    scans of the same region reconstruct onto canvases of different shapes, offset from one
    another by a fraction of the drift you are trying to measure.
    """

    @staticmethod
    def _montage(dataset4d, defocus, position_offset=(0.0, 0.0), **kwargs):
        dataset3d = Dataset3d.from_array(
            dataset4d.array.reshape(-1, N, N),
            name="synthetic 3D stack",
            units=("index", "A^-1", "A^-1"),
            sampling=(1, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        )
        n_scan = dataset4d.shape[0]
        positions = scan_positions_px()[: n_scan * n_scan] * SCAN_SAMPLING
        positions = positions + np.asarray(position_offset, dtype=np.float64)
        return DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            positions,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="pad",
            **_common_kwargs(defocus),
            **kwargs,
        )

    def test_scan_origin_recovers_the_input_positions(self, dataset4d):
        """`scan_origin + positions_px * scan_sampling` must be what was passed in."""
        offset = (37.0, -12.5)
        montage = self._montage(dataset4d, integer_shift_defocus(1), position_offset=offset)

        n_scan = dataset4d.shape[0]
        expected = scan_positions_px()[: n_scan * n_scan] * SCAN_SAMPLING + np.asarray(offset)
        recovered = np.asarray(montage.scan_origin) + to_numpy(montage.positions_px) * np.asarray(
            montage.scan_sampling
        )

        assert recovered == pytest.approx(expected, abs=1e-4)

    def test_pinned_window_is_reported_back(self, dataset4d):
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        origin = (-40.0, -30.0)
        fov = (SCAN_SAMPLING * 20, SCAN_SAMPLING * 24)

        montage.reconstruct(
            deconvolution_kernel="prlx", obj_origin=origin, obj_fov=fov, verbose=False
        )

        assert montage.obj.shape == (20, 24)
        assert montage.obj_origin == pytest.approx(origin, abs=1e-4)
        assert montage._obj_fov == pytest.approx(fov, rel=1e-6)

    def test_offset_acquisitions_land_on_the_same_canvas(self, dataset4d):
        """The headline: the same specimen scanned in shifted coordinates must agree.

        The two montages differ only in a rigid offset applied to *both* the positions and
        the requested window, so the reconstructions have to be the same image -- which is
        exactly what makes a cross-correlation between frames measure drift and nothing else.
        """
        defocus = integer_shift_defocus(1)
        # deliberately not a whole number of scan pixels, which is the case that used to
        # leave the two canvases a fraction of a pixel apart
        offset = np.array([2.3 * SCAN_SAMPLING, -1.7 * SCAN_SAMPLING])
        origin = np.array([-3.0 * SCAN_SAMPLING, -3.0 * SCAN_SAMPLING])
        fov = (SCAN_SAMPLING * 24, SCAN_SAMPLING * 24)

        plain = self._montage(dataset4d, defocus)
        moved = self._montage(dataset4d, defocus, position_offset=offset)

        plain.reconstruct(
            deconvolution_kernel="prlx", obj_origin=tuple(origin), obj_fov=fov, verbose=False
        )
        moved.reconstruct(
            deconvolution_kernel="prlx",
            obj_origin=tuple(origin + offset),
            obj_fov=fov,
            verbose=False,
        )

        assert moved.obj.shape == plain.obj.shape
        assert moved.obj_origin == pytest.approx(tuple(origin + offset), abs=1e-4)
        assert _relative_error(moved.obj, plain.obj) < 1e-5

    def test_unpinned_canvas_is_reproduced_by_pinning_it(self, dataset4d):
        """Reading the window back out and passing it in must be a no-op."""
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        montage.reconstruct(deconvolution_kernel="prlx", pad_px=4, verbose=False)
        automatic = montage.obj.copy()

        montage.reconstruct(
            deconvolution_kernel="prlx",
            obj_origin=montage.obj_origin,
            obj_fov=montage._obj_fov,
            verbose=False,
        )

        assert montage.obj.shape == automatic.shape
        assert _relative_error(montage.obj, automatic) < 1e-6

    def test_obj_fov_and_pad_px_are_mutually_exclusive(self, dataset4d):
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        with pytest.raises(ValueError, match="both size the canvas"):
            montage.reconstruct(
                deconvolution_kernel="prlx", pad_px=4, obj_fov=(100.0, 100.0), verbose=False
            )

    def test_wrap_boundary_accepts_a_pinned_canvas(self, dataset4d):
        """Pinning the canvas is orthogonal to the boundary rule.

        ``"wrap"`` defaults to the scan grid, but that is a default rather than a
        restriction: given a window, it wraps into that window instead. Needed whenever the
        canvas has to be a particular size for reasons other than the scan -- matching an
        empirical probe's reciprocal grid, say.
        """
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        fov = (SCAN_SAMPLING * 20, SCAN_SAMPLING * 24)

        montage.reconstruct(
            deconvolution_kernel="prlx",
            boundary="wrap",
            obj_origin=(0.0, 0.0),
            obj_fov=fov,
            verbose=False,
        )

        assert montage.obj.shape == (20, 24)
        assert montage.obj_origin == pytest.approx((0.0, 0.0), abs=1e-4)
        assert np.isfinite(montage.obj).all()

    def test_a_wrapped_canvas_smaller_than_the_scan_warns(self, dataset4d):
        """Positions outside a wrapped canvas fold a second copy of the specimen over it.

        The failure looks like a real reconstruction with a ghost in it, which is easy to
        blame on something else -- so it has to say so.
        """
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        span = np.ptp(to_numpy(montage.positions_px), axis=0) * SCAN_SAMPLING

        with pytest.warns(UserWarning, match="wrap around"):
            montage.reconstruct(
                deconvolution_kernel="prlx",
                boundary="wrap",
                obj_origin=tuple(np.asarray(montage.scan_origin)),
                obj_fov=(span[0] / 2, span[1]),
                verbose=False,
            )

    @pytest.mark.parametrize("boundary", ["wrap", "pad"])
    def test_a_canvas_covering_the_scan_is_silent(self, dataset4d, boundary, recwarn):
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        span = np.ptp(to_numpy(montage.positions_px), axis=0) * SCAN_SAMPLING

        montage.reconstruct(
            deconvolution_kernel="prlx",
            boundary=boundary,
            obj_origin=tuple(np.asarray(montage.scan_origin) - 4 * SCAN_SAMPLING),
            obj_fov=tuple(span + 8 * SCAN_SAMPLING),
            verbose=False,
        )

        assert not [w for w in recwarn if "wrap around" in str(w.message)]

    def test_wrap_defaults_to_the_scan_grid(self, dataset4d):
        """The default is unchanged: no window given, the canvas is the scan."""
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        montage.reconstruct(deconvolution_kernel="prlx", boundary="wrap", verbose=False)

        assert montage.obj.shape == tuple(montage.scan_gpts)

    def test_obj_fov_must_be_positive(self, dataset4d):
        montage = self._montage(dataset4d, integer_shift_defocus(1))
        with pytest.raises(ValueError, match="positive"):
            montage.reconstruct(deconvolution_kernel="prlx", obj_fov=(0.0, 10.0), verbose=False)

    def test_scan_origin_survives_a_round_trip(self, dataset4d, tmp_path):
        montage = self._montage(dataset4d, integer_shift_defocus(1), position_offset=(11.0, -3.0))
        montage.reconstruct(deconvolution_kernel="prlx", verbose=False)

        path = str(tmp_path / "montage.zip")
        montage.save(path, mode="o")
        restored = load(path)

        assert restored.scan_origin == pytest.approx(montage.scan_origin)
        assert restored.obj_origin == pytest.approx(montage.obj_origin)

    def test_gridded_construction_has_a_zero_scan_origin(self, dataset4d):
        """A raster acquisition's grid *defines* the coordinates, so its origin is zero."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        assert montage.scan_origin == (0.0, 0.0)


class TestFrameDrift:
    """`estimate_frame_drift` on interleaved frames with a seeded, known displacement."""

    #: a window well inside the object, so every frame's canvas is fully supported and the
    #: correlation is not measuring which frame happened to reach nearer the edge
    ORIGIN = (8 * CTF_SAMPLING, 8 * CTF_SAMPLING)
    FOV = (48 * CTF_SAMPLING, 48 * CTF_SAMPLING)
    DRIFT = np.array([[0.0, 0.0], [1.6, -0.8], [3.2, -1.6], [4.8, -2.4]]) * CTF_SAMPLING

    def _frames(self, ctf_scene, drift=None):
        complex_obj, _, probe, _, _ = ctf_scene
        drift = self.DRIFT if drift is None else drift
        datasets, positions = ctf_interleaved_frames(complex_obj, probe, drift)

        montages = []
        for dataset, pos in zip(datasets, positions):
            montage = DirectPtychographyMontage.from_dataset3d(
                dataset,
                pos,
                scan_sampling=(CTF_SCAN_SAMPLING, CTF_SCAN_SAMPLING),
                boundary="pad",
                **ctf_kwargs(),
            )
            montage.reconstruct(
                deconvolution_kernel="prlx",
                obj_origin=self.ORIGIN,
                obj_fov=self.FOV,
                verbose=False,
            )
            montages.append(montage)
        return montages, positions

    def test_recovers_a_seeded_drift(self, ctf_scene):
        montages, _ = self._frames(ctf_scene)
        measured = estimate_frame_drift(montages, verbose=False)

        # the estimate is referred to the mean over frames, so compare mean-centered
        expected = self.DRIFT - self.DRIFT.mean(axis=0)
        assert measured == pytest.approx(expected, abs=0.35 * CTF_SCAN_SAMPLING)

    def test_drift_sums_to_zero(self, ctf_scene):
        montages, _ = self._frames(ctf_scene)
        measured = estimate_frame_drift(montages, verbose=False)
        assert measured.mean(axis=0) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_correcting_the_positions_sharpens_the_combined_reconstruction(self, ctf_scene):
        """The point of the exercise: `positions - drift` must beat `positions`.

        Also pins the sign, which no amount of reasoning about correlation conventions
        substitutes for.
        """
        complex_obj, _, probe, ctf_full, _ = ctf_scene
        datasets, positions = ctf_interleaved_frames(complex_obj, probe, self.DRIFT)
        montages, _ = self._frames(ctf_scene)
        drift = estimate_frame_drift(montages, verbose=False)

        combined = Dataset3d.from_array(
            np.concatenate([d.array for d in datasets]),
            name="combined frames",
            units=datasets[0].units,
            sampling=datasets[0].sampling,
        )

        def reconstruct(position_list):
            montage = DirectPtychographyMontage.from_dataset3d(
                combined,
                np.concatenate(position_list),
                scan_sampling=(CTF_SCAN_SAMPLING, CTF_SCAN_SAMPLING),
                boundary="pad",
                **ctf_kwargs(),
            )
            return montage.reconstruct(
                deconvolution_kernel="prlx",
                obj_origin=self.ORIGIN,
                obj_fov=self.FOV,
                verbose=False,
            )

        uncorrected = reconstruct(positions)
        corrected = reconstruct([p - d for p, d in zip(positions, drift)])

        # drift smears the montage, so undoing it has to raise the contrast
        assert corrected.obj.std() > uncorrected.obj.std()
        # and the wrong sign must make it worse, not better
        wrong_sign = reconstruct([p + d for p, d in zip(positions, drift)])
        assert corrected.obj.std() > wrong_sign.obj.std()

    def test_zero_drift_is_recovered_as_zero(self, ctf_scene):
        no_drift = np.zeros((3, 2))
        montages, _ = self._frames(ctf_scene, drift=no_drift)
        measured = estimate_frame_drift(montages, verbose=False)
        assert np.abs(measured).max() < 0.35 * CTF_SCAN_SAMPLING

    def test_rejects_frames_on_different_canvases(self, ctf_scene):
        montages, _ = self._frames(ctf_scene)
        montages[1].reconstruct(
            deconvolution_kernel="prlx",
            obj_origin=self.ORIGIN,
            obj_fov=(self.FOV[0] + 4 * CTF_SAMPLING, self.FOV[1]),
            verbose=False,
        )
        with pytest.raises(ValueError, match="share a canvas"):
            estimate_frame_drift(montages, verbose=False)

    def test_rejects_frames_at_different_origins(self, ctf_scene):
        """The subtle failure: same shape, wrong place, silently measuring the offset."""
        montages, _ = self._frames(ctf_scene)
        montages[1].reconstruct(
            deconvolution_kernel="prlx",
            obj_origin=(self.ORIGIN[0] + 3 * CTF_SAMPLING, self.ORIGIN[1]),
            obj_fov=self.FOV,
            verbose=False,
        )
        with pytest.raises(ValueError, match="not a canvas origin"):
            estimate_frame_drift(montages, verbose=False)

    def test_rejects_unreconstructed_frames(self, ctf_scene):
        montages, _ = self._frames(ctf_scene)
        montages[1]._reset_reconstruction()
        with pytest.raises(ValueError, match="not been reconstructed"):
            estimate_frame_drift(montages, verbose=False)

    def test_needs_at_least_two_frames(self, ctf_scene):
        montages, _ = self._frames(ctf_scene)
        with pytest.raises(ValueError, match="at least two"):
            estimate_frame_drift(montages[:1], verbose=False)

    def test_works_on_gridded_reconstructions(self, dataset4d):
        """Nothing about it is montage-specific -- two raster reconstructions also align."""
        defocus = integer_shift_defocus(1)
        first, second = (
            DirectPtychography.from_dataset4d(
                dataset4d, edge_blend_pixels=0, **_common_kwargs(defocus)
            )
            for _ in range(2)
        )
        first.reconstruct(deconvolution_kernel="prlx", verbose=False)
        second.reconstruct(deconvolution_kernel="prlx", verbose=False)

        drift = estimate_frame_drift([first, second], verbose=False)
        assert np.abs(drift).max() < 1e-6


@pytest.mark.skipif(not ACCELERATORS, reason="no accelerator available")
@pytest.mark.parametrize("device", ACCELERATORS)
class TestAccelerators:
    """Both classes must run, and agree with CPU, on whatever accelerator is present.

    MPS has no float64 at all, so every positional tensor -- coordinates, shifts, canvas
    origins -- has to take its dtype from the device rather than hardcode one.
    """

    @staticmethod
    def _kwargs(dataset4d):
        return dict(_common_kwargs(integer_shift_defocus(1)), edge_blend_pixels=0)

    @staticmethod
    def _ungridded(dataset4d):
        """A `Dataset3d` and positions, with the origin *fitted* rather than forced.

        Forcing the origin skips the centre-of-mass fit entirely, which is where the
        device mismatch lived, so these deliberately let it run.
        """
        dataset3d = Dataset3d.from_array(
            np.asarray(dataset4d.array).reshape(-1, N, N),
            name="ungridded",
            sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
            units=("index", "A^-1", "A^-1"),
        )
        return dataset3d, scan_positions_px() * SCAN_SAMPLING

    @pytest.mark.parametrize("boundary", ["wrap", "pad"])
    def test_montage_matches_cpu(self, dataset4d, device, boundary):
        def run(where):
            montage = DirectPtychographyMontage.from_dataset4d(
                dataset4d, device=where, boundary=boundary, **self._kwargs(dataset4d)
            )
            return montage.reconstruct(verbose=False).obj

        on_device, on_cpu = run(device), run("cpu")

        assert on_device.shape == on_cpu.shape
        assert np.abs(on_device - on_cpu).max() / np.abs(on_cpu).max() < 1e-4

    def test_montage_upsampled_matches_cpu(self, dataset4d, device):
        def run(where):
            montage = DirectPtychographyMontage.from_dataset4d(
                dataset4d, device=where, **self._kwargs(dataset4d)
            )
            return montage.reconstruct(
                upsampling_factor=2, interpolation="bilinear", verbose=False
            ).obj

        assert np.abs(run(device) - run("cpu")).max() / np.abs(run("cpu")).max() < 1e-4

    def test_defocus_gradient_matches_cpu(self, dataset4d, device):
        def run(where):
            montage = DirectPtychographyMontage.from_dataset4d(
                dataset4d,
                device=where,
                boundary="pad",
                defocus_gradient=(30.0, -10.0),
                **self._kwargs(dataset4d),
            )
            return montage.reconstruct(verbose=False).obj

        on_device, on_cpu = run(device), run("cpu")

        assert on_device.shape == on_cpu.shape
        assert np.abs(on_device - on_cpu).max() / np.abs(on_cpu).max() < 1e-4

    def test_real_space_kernel_runs(self, dataset4d, device):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, device=device, **self._kwargs(dataset4d)
        )
        obj = montage.reconstruct(deconvolution_kernel="ssb", stencil_radius=4, verbose=False).obj

        assert np.isfinite(obj).all()

    def test_variance_loss_and_search_run(self, dataset4d, device):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d, device=device, **self._kwargs(dataset4d)
        )
        montage.reconstruct(verbose=False)

        assert np.isfinite(float(montage.variance_loss()))

    @pytest.mark.parametrize(
        "cls", [DirectPtychography, DirectPtychographyMontage], ids=["fourier", "montage"]
    )
    def test_ungridded_fits_the_origin_on_device(self, dataset4d, device, cls):
        """Regression: probe positions arrive as numpy and met the measured origin on MPS."""
        dataset3d, positions = self._ungridded(dataset4d)

        reconstruction = cls.from_dataset3d(
            dataset3d,
            positions,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            device=device,
            verbose=False,
        )
        obj = reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False).obj

        assert np.isfinite(obj).all()


@pytest.mark.skipif(not ACCELERATORS, reason="no accelerator available")
@pytest.mark.parametrize("device", ACCELERATORS)
def test_padded_canvas_shape_is_device_independent(dataset4d, device):
    """Float noise must not push a canvas bound across an integer on one device only."""

    def shape(where):
        montage = DirectPtychographyMontage.from_dataset4d(
            dataset4d,
            device=where,
            boundary="pad",
            **dict(_common_kwargs(integer_shift_defocus(1)), edge_blend_pixels=0),
        )
        return montage.reconstruct(verbose=False).obj.shape

    assert shape(device) == shape("cpu")
