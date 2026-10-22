"""``device="gpu"`` resolves to CUDA or MPS, and every reconstruction path matches the CPU.

MPS has no float64 or complex128, so the per-frequency solves leave the device and come back;
these tests run only where a GPU is present.
"""

import warnings

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset3d
from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging import (
    DirectPtychography,
    DirectPtychographyMontage,
    TiltCorrectedDarkField,
    bright_field_transfer,
    fuse_full_field,
)

from .conftest import (
    ML_N_DET,
    PROBE_ENERGY,
    SCAN_SAMPLING,
    SEMIANGLE_CUTOFF,
    N,
    direct_ptycho_kwargs,
    integer_shift_defocus,
    ml_kwargs,
    scan_positions_px,
    thin_layer_4dstem,
    tilted_layers_4dstem,
    twisted_square_cell,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="needs a CUDA or MPS device",
)

C10 = integer_shift_defocus(1)
KERNELS = ("ssb", "obf", "wiener", "wdd", "prlx", "icom")


def _build(dataset, device, cls=DirectPtychography):
    kwargs = dict(direct_ptycho_kwargs(C10), device=device, verbose=False)
    if cls is DirectPtychography:
        kwargs["edge_blend_pixels"] = 0
    return cls.from_dataset4d(dataset, **kwargs)


@pytest.fixture(scope="module")
def pair(dataset4d):
    return _build(dataset4d, "cpu"), _build(dataset4d, "gpu")


def _assert_close(image, expected, rtol=1e-4, case=""):
    image, expected = to_numpy(image), to_numpy(expected)
    assert image.shape == expected.shape, case
    assert np.isfinite(image).all(), case
    assert np.abs(image - expected).max() <= rtol * np.abs(expected).max(), case


def test_gpu_resolves_to_a_device(pair):
    _, gpu = pair
    assert gpu.device in ("mps", "cuda:0")


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_matches_the_cpu(pair, kernel):
    cpu, gpu = pair
    expected = cpu.reconstruct(deconvolution_kernel=kernel).corrected_bf
    _assert_close(gpu.reconstruct(deconvolution_kernel=kernel).corrected_bf, expected)


@pytest.mark.parametrize("kernel", ["obf", "wiener"])
def test_a_complex_object_matches_the_cpu(pair, kernel):
    cpu, gpu = pair
    kwargs = dict(deconvolution_kernel=kernel, obj_type="complex")
    cpu.reconstruct(**kwargs)
    gpu.reconstruct(**kwargs)
    _assert_close(gpu.corrected_bf, cpu.corrected_bf)
    _assert_close(gpu.corrected_amplitude, cpu.corrected_amplitude)
    assert gpu.residual_loss() == pytest.approx(cpu.residual_loss(), rel=1e-5)


SPACING = 6.0


@pytest.fixture(scope="module")
def layered_pair():
    dataset = tilted_layers_4dstem(spacing=SPACING)
    return tuple(
        DirectPtychography.from_dataset4d(dataset, device=device, **ml_kwargs())
        for device in ("cpu", "gpu")
    )


@pytest.mark.parametrize("kernel", ["obf", "wiener"])
def test_slices_match_the_cpu(layered_pair, kernel):
    cpu, gpu = layered_pair
    kwargs = dict(
        deconvolution_kernel=kernel,
        obj_type="complex",
        num_slices=2,
        slice_mode="joint",
        slice_thicknesses=SPACING,
    )
    cpu.reconstruct(**kwargs)
    gpu.reconstruct(**kwargs)
    # the data are noise-free, so the Wiener estimate's noise variance is float32 round-off,
    # which the two devices accumulate in different orders
    _assert_close(gpu.corrected_bf, cpu.corrected_bf, rtol=1e-3)
    _assert_close(gpu.corrected_amplitude, cpu.corrected_amplitude, rtol=1e-3)
    assert gpu.residual_loss() == pytest.approx(cpu.residual_loss(), abs=1e-6)


def test_the_gradient_fit_matches_the_cpu(dataset4d):
    fitted = {}
    for device in ("cpu", "gpu"):
        recon = _build(dataset4d, device)
        recon.hyperparameter_state.optimized_aberrations = {"C10": C10 + 20.0}
        recon.fit_hyperparameters_gradient(("C10",), num_iterations=5, verbose=False)
        fitted[device] = recon.hyperparameter_state.current_aberrations()["C10"]
    assert np.isfinite(fitted["gpu"])
    assert fitted["gpu"] == pytest.approx(fitted["cpu"], abs=0.05)


def test_the_montage_fft_route_matches_the_cpu(dataset4d):
    images = {}
    for device in ("cpu", "gpu"):
        montage = _build(dataset4d, device, DirectPtychographyMontage)
        montage.reconstruct(
            deconvolution_kernel="obf", obj_type="complex", convolution_mode="fft", verbose=False
        )
        images[device] = montage.corrected_bf
    _assert_close(images["gpu"], images["cpu"])


def test_the_montage_splat_and_stencil_routes_match_the_cpu(dataset4d):
    """MPS has no float64 at all, so every positional tensor -- coordinates, shifts, canvas
    origins -- has to take its dtype from the device rather than hardcode one."""
    for constructor, reconstruct in (
        (dict(boundary="wrap"), {}),
        # float noise must not push a canvas bound across an integer on one device only
        (dict(boundary="pad"), {}),
        (dict(boundary="wrap"), dict(upsampling_factor=2, interpolation="bilinear")),
        (dict(boundary="pad", defocus_gradient=(30.0, -10.0)), {}),
        (dict(boundary="wrap"), dict(deconvolution_kernel="ssb", stencil_radius=4)),
    ):
        case = f"{constructor=} {reconstruct=}"
        results = {}
        for device in ("cpu", "gpu"):
            montage = DirectPtychographyMontage.from_dataset4d(
                dataset4d,
                device=device,
                edge_blend_pixels=0,
                **direct_ptycho_kwargs(C10),
                **constructor,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                montage.reconstruct(verbose=False, **reconstruct)
            residual = None if "deconvolution_kernel" in reconstruct else montage.residual_loss()
            results[device] = montage.obj, residual
        _assert_close(results["gpu"][0], results["cpu"][0], case=case)
        if results["cpu"][1] is not None:
            assert results["gpu"][1] == pytest.approx(results["cpu"][1], rel=1e-4), case


def test_an_ungridded_scan_fits_its_origin_on_the_device(dataset4d):
    """Regression: probe positions arrive as numpy and met the measured origin on MPS.

    Forcing the origin skips the centre-of-mass fit entirely, which is where the device
    mismatch lived, so this deliberately lets it run.
    """
    dataset3d = Dataset3d.from_array(
        np.asarray(dataset4d.array).reshape(-1, N, N),
        name="ungridded",
        sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
        units=("index", "A^-1", "A^-1"),
    )
    images = {}
    for device in ("cpu", "gpu"):
        reconstruction = DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            scan_positions_px() * SCAN_SAMPLING,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            aberration_coefs={"C10": C10},
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            device=device,
            verbose=False,
        )
        images[device] = reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
    assert np.abs(images["cpu"]).max() > 0
    _assert_close(images["gpu"], images["cpu"])


def test_a_tilt_series_matches_the_cpu():
    datasets = [tilted_layers_4dstem(tilt, SPACING) for tilt in ((0, 0), (10, 0))]
    images = {}
    for device in ("cpu", "gpu"):
        series = DirectPtychography.from_tilt_series(
            datasets, [(0, 0), (10, 0)], device=device, **ml_kwargs()
        )
        series.register(max_shift=1.5)
        series.reconstruct(
            deconvolution_kernel="obf",
            obj_type="complex",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=SPACING,
        )
        images[device] = series.corrected_bf
    _assert_close(images["gpu"], images["cpu"])


def test_tilt_corrected_dark_field_matches_the_cpu():
    _, layers, _ = twisted_square_cell()
    defocus = {"C10": 150.0}
    dataset = thin_layer_4dstem([1j * 0.5 * layers[0]], [0.0], defocus)
    kwargs = dict(ml_kwargs(defocus), force_fitted_origin=(ML_N_DET // 2, ML_N_DET // 2))
    kwargs.pop("edge_blend_pixels")
    images = {}
    for device in ("cpu", "gpu"):
        tcdf = TiltCorrectedDarkField.from_dataset4d(dataset, device=device, **kwargs)
        dark = tcdf.reconstruct(upsampling_factor=2, verbose=False).corrected_bf
        direct = DirectPtychography.from_dataset4d(dataset, device=device, **ml_kwargs(defocus))
        bright = direct.reconstruct(deconvolution_kernel="obf", upsampling_factor=2).corrected_bf
        transfer = bright_field_transfer(direct, upsampling_factor=2)
        images[device] = (dark, fuse_full_field(to_numpy(bright), to_numpy(dark), transfer))
    _assert_close(images["gpu"][0], images["cpu"][0])
    _assert_close(images["gpu"][1], images["cpu"][1])


def test_sections_match_the_cpu(pair):
    cpu, gpu = pair
    kwargs = dict(
        deconvolution_kernel="obf", num_slices=3, slice_thicknesses=50.0, slice_mode="sectioned"
    )
    cpu.reconstruct(**kwargs)
    gpu.reconstruct(**kwargs)
    _assert_close(gpu.corrected_bf, cpu.corrected_bf)
    assert np.allclose(gpu.section_residuals, cpu.section_residuals, rtol=1e-4)
