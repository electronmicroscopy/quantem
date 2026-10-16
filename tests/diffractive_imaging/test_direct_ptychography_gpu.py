"""``device="gpu"`` resolves to CUDA or MPS, and every reconstruction path matches the CPU.

MPS has no float64 or complex128, so the per-frequency solves leave the device and come back;
these tests run only where a GPU is present.
"""

import numpy as np
import pytest
import torch

from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging import (
    DirectPtychography,
    DirectPtychographyMontage,
    TiltCorrectedDarkField,
    bright_field_transfer,
    fuse_full_field,
)

from .conftest import (
    ML_ABERRATIONS,
    ML_N_DET,
    ML_SAMPLING,
    direct_ptycho_kwargs,
    integer_shift_defocus,
    ml_kwargs,
    thin_layer_4dstem,
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


def _assert_close(image, expected, rtol=1e-4):
    image, expected = to_numpy(image), to_numpy(expected)
    assert np.isfinite(image).all()
    assert np.abs(image - expected).max() <= rtol * np.abs(expected).max()


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


def _shifted(image, shift):
    q = np.fft.fftfreq(image.shape[0], ML_SAMPLING)
    ramp = np.exp(-2j * np.pi * (q[:, None] * shift[0] + q[None, :] * shift[1]))
    return np.fft.ifft2(np.fft.fft2(image) * ramp).real


def _layered(tilt_mrad=(0, 0)):
    """Two twisted layers 6 A apart, each displaced by its depth times the tilt."""
    _, layers, _ = twisted_square_cell()
    theta = np.asarray(tilt_mrad, dtype=float) * 1e-3
    depths = (-SPACING / 2, SPACING / 2)
    logs = [1j * _shifted(0.3 * layer, z * theta) for layer, z in zip(layers, depths)]
    return thin_layer_4dstem(logs, list(depths), ML_ABERRATIONS)


@pytest.fixture(scope="module")
def layered_pair():
    dataset = _layered()
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


def test_a_tilt_series_matches_the_cpu():
    datasets = [_layered((0, 0)), _layered((10, 0))]
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


@pytest.mark.parametrize("kernel", ["ssb", "obf", "prlx"])
def test_sections_match_the_cpu(pair, kernel):
    cpu, gpu = pair
    kwargs = dict(
        deconvolution_kernel=kernel, num_slices=3, slice_thicknesses=50.0, slice_mode="sectioned"
    )
    cpu.reconstruct(**kwargs)
    gpu.reconstruct(**kwargs)
    _assert_close(gpu.corrected_bf, cpu.corrected_bf)
    assert np.allclose(gpu.section_residuals, cpu.section_residuals, rtol=1e-4)
