"""
Property-style tests for ptychography state management and serialization.

Gradient equivalence between autograd and the analytic path lives in
``test_ptychography_gradients.py``.
"""

import numpy as np
import pytest

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml import OptimizerParams
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography

if config.NUM_DEVICES > 0:
    config.set_device("gpu")

N = 64
Q_MAX = 0.5  # inverse Angstroms
Q_PROBE = Q_MAX / 2  # inverse Angstroms
PROBE_ENERGY = 300e3  # eV

SCAN_STEP_SIZE = 1  # pixels
sx = sy = N // SCAN_STEP_SIZE
C10 = 50


@pytest.fixture
def white_noise_4d_array():
    """Create a white noise 4D array for ptychography testing."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    arr = rng.random((N, N))
    arr -= arr.mean()
    return arr.astype(np.float32)


@pytest.fixture
def complex_obj(white_noise_4d_array):
    """Create a complex object for ptychography testing."""
    return np.exp(1.0j * white_noise_4d_array)


def return_patch_indices(positions_px, roi_shape, obj_shape):
    """ """
    x0 = np.round(positions_px[:, 0]).astype("int")
    y0 = np.round(positions_px[:, 1]).astype("int")

    x_ind = np.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
    y_ind = np.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

    row = (x0[:, None, None] + x_ind[None, :, None]) % obj_shape[0]
    col = (y0[:, None, None] + y_ind[None, None, :]) % obj_shape[1]

    return row, col


def simulate_exit_waves(
    complex_obj,
    probe,
    row,
    col,
):
    """ """
    obj_patches = complex_obj[row, col]
    exit_waves = obj_patches * probe
    return obj_patches, exit_waves


def simulate_intensities(
    complex_obj,
    probe,
    row,
    col,
):
    """ """
    obj_patches, exit_waves = simulate_exit_waves(complex_obj, probe, row, col)
    fourier_exit_waves = np.fft.fft2(exit_waves)
    intensities = np.abs(fourier_exit_waves) ** 2
    return obj_patches, exit_waves, fourier_exit_waves, intensities


@pytest.fixture
def probe_array(complex_obj):
    """Create a probe array for ptychography testing."""
    sampling = 1 / Q_MAX / 2  # Angstroms
    reciprocal_sampling = 2 * Q_MAX / N  # inverse Angstroms

    qx = qy = np.fft.fftfreq(N, sampling)
    q2 = qx[:, None] ** 2 + qy[None, :] ** 2
    q = np.sqrt(q2)

    aperture_fourier = np.sqrt(
        np.clip(
            (Q_PROBE - q) / reciprocal_sampling + 0.5,
            0,
            1,
        ),
    )

    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * C10
    exp_chi = np.exp(-1j * chi)
    probe_array_fourier = aperture_fourier * exp_chi
    probe_array_fourier /= np.sqrt(np.sum(np.abs(probe_array_fourier) ** 2))
    probe_array = np.fft.ifft2(probe_array_fourier) * N
    return probe_array


@pytest.fixture
def ptycho_dataset(complex_obj, probe_array):
    """Create a Dataset4dstem from white noise for testing."""

    x = y = np.arange(0.0, N, SCAN_STEP_SIZE)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    positions = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    reciprocal_sampling = 2 * Q_MAX / N  # inverse Angstroms

    sim_row, sim_col = return_patch_indices(positions, (N, N), (N, N))

    obj_patches, exit_waves, fourier_exit_waves, intensities = simulate_intensities(
        complex_obj, probe_array, sim_row, sim_col
    )

    dset = Dataset4dstem.from_array(
        array=np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((sx, sy, N, N)),
        sampling=(
            SCAN_STEP_SIZE,
            SCAN_STEP_SIZE,
            reciprocal_sampling,
            reciprocal_sampling,
        ),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)

    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=True,
        plot_com=True,
        probe_energy=PROBE_ENERGY,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    return pdset


@pytest.fixture
def single_probe_ptycho_model(ptycho_dataset, probe_array):
    """Create ptychography model components for testing."""
    obj_model = ObjectPixelated.from_uniform(num_slices=1, obj_type="complex", slice_thicknesses=1)

    probe_params = {
        "energy": PROBE_ENERGY,
        "C10": C10,
        "semiangle_cutoff": electron_wavelength_angstrom(PROBE_ENERGY) * 1e3,
    }

    probe_model = ProbePixelated.from_array(
        num_probes=1,
        probe_params=probe_params,
        probe_array=probe_array,
    )

    detector_model = DetectorPixelated()

    ptycho = Ptychography.from_models(
        dset=ptycho_dataset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        rng=42,
    )

    ptycho.preprocess(
        obj_padding_px=(0, 0),
    )
    return ptycho


@pytest.fixture
def small_ptycho_dataset():
    """A small random dataset model, enough for property and serialization checks."""
    array = np.random.default_rng(42).random((8, 8, 16, 16)).astype(np.float32)
    dset = Dataset4dstem.from_array(
        array, sampling=(1.0, 1.0, 0.05, 0.05), units=("A", "A", "A^-1", "A^-1")
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)
    return pdset


class TestTargetResidency:
    """Property + serialization behavior for the streaming-target knob."""

    def test_default_and_setter(self, small_ptycho_dataset):
        assert small_ptycho_dataset.target_residency == "device"
        small_ptycho_dataset.target_residency = "cpu"
        assert small_ptycho_dataset.target_residency == "cpu"
        small_ptycho_dataset.target_residency = "device"
        assert small_ptycho_dataset.target_residency == "device"

        for bad in ["gpu", "GPU", "CPU", "", "cuda", "Device"]:
            with pytest.raises(ValueError, match="target_residency"):
                small_ptycho_dataset.target_residency = bad
            # value should be unchanged after a rejected set
            assert small_ptycho_dataset.target_residency == "device", bad

    def test_save_load_roundtrip(self, small_ptycho_dataset, tmp_path):
        small_ptycho_dataset.target_residency = "cpu"
        path = tmp_path / "pdset.zip"
        small_ptycho_dataset.save(str(path))
        reloaded = autoserialize_load(str(path))
        assert reloaded.target_residency == "cpu"


def _build_aspect_ratio_ptycho(complex_obj, probe_array, gpts, com_rotation, transpose):
    """Build a Ptychography on a (possibly non-square) scan grid with a forced rotation
    and transpose. Reconstruction quality is irrelevant here — the fixture only exercises
    scan-position placement and object sizing."""
    scan_x, scan_y = gpts
    x = np.arange(0.0, scan_x * SCAN_STEP_SIZE, SCAN_STEP_SIZE)
    y = np.arange(0.0, scan_y * SCAN_STEP_SIZE, SCAN_STEP_SIZE)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    positions = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    reciprocal_sampling = 2 * Q_MAX / N

    sim_row, sim_col = return_patch_indices(positions, (N, N), (N, N))
    _, _, _, intensities = simulate_intensities(complex_obj, probe_array, sim_row, sim_col)

    dset = Dataset4dstem.from_array(
        array=np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((scan_x, scan_y, N, N)),
        sampling=(SCAN_STEP_SIZE, SCAN_STEP_SIZE, reciprocal_sampling, reciprocal_sampling),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=PROBE_ENERGY,
        force_com_rotation=com_rotation,
        force_com_transpose=transpose,
    )

    probe_params = {
        "energy": PROBE_ENERGY,
        "C10": C10,
        "semiangle_cutoff": electron_wavelength_angstrom(PROBE_ENERGY) * 1e3,
    }
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(
            num_slices=1, obj_type="complex", slice_thicknesses=1
        ),
        probe_model=ProbePixelated.from_array(
            num_probes=1, probe_params=probe_params, probe_array=probe_array
        ),
        detector_model=DetectorPixelated(),
        rng=42,
    )
    ptycho.preprocess(obj_padding_px=(0, 0))
    return ptycho


class TestAspectRatioRotatedFOV:
    """Regression for the non-square / rotated / transposed FOV bug.

    A high aspect-ratio scan grid combined with a large com_rotation (and/or transpose)
    used to (a) place scan positions outside the object array — wrapping the object and
    pinning positions to the FOV edge — and (b) crash ``obj_cropped`` inside
    ``_crop_rotate_obj_fov`` because the un-rotated FOV could not fit the object frame.
    """

    def test_positions_inside_object_and_obj_cropped_shape(self, complex_obj, probe_array):
        for transpose in (False, True):
            ptycho = _build_aspect_ratio_ptycho(
                complex_obj, probe_array, gpts=(32, 64), com_rotation=89, transpose=transpose
            )
            obj_full = ptycho.dset._obj_shape_full_2d(ptycho.obj_padding_px)
            pos = ptycho.dset.initial_scan_positions_px.cpu().detach().numpy()
            # positions must sit strictly inside the object (no wrap / no edge pile-up)
            assert pos[:, 0].min() >= 0 and pos[:, 1].min() >= 0, transpose
            assert pos[:, 0].max() < obj_full[-2], transpose
            assert pos[:, 1].max() < obj_full[-1], transpose
            # obj_cropped must not raise and returns the non-transposed display FOV shape
            cropped = ptycho.obj_cropped
            assert tuple(cropped.shape) == tuple(ptycho.obj_shape_crop), transpose


@pytest.mark.slow
class TestPtychographySaveLoadRoundtrip:
    """Reconstruct → save → load → continue training preserves training state.

    The 0.3 threshold reflects that, on this synthetic ducky-style dataset with the
    analytical probe already in hand, a well-formed reconstruction should drive the
    loss down by at least 70% in 20 iterations on the right configuration. The bar
    is deliberately strict — if you tune optimizer settings and this fires, the
    config probably regressed.
    """

    NUM_ITERS = 50  # enough headroom for the strict 0.3 threshold at lr=5e-3

    def test_reconstruct_save_load_and_continue(self, single_probe_ptycho_model, tmp_path):
        """Reconstruct, then reload: losses, positions and object match exactly, and continued
        training extends the history without regressing -- confirms optimizer state and
        parameter bindings survive the save/load roundtrip end-to-end."""
        ptycho = single_probe_ptycho_model
        ptycho.reconstruct(
            num_iters=self.NUM_ITERS,
            reset=True,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=5e-3),
                "probe": OptimizerParams.Adam(lr=5e-3),
            },
            batch_size=N**2,
            device=config.get_device(),
        )
        path = tmp_path / "ptycho.zip"
        ptycho.save(str(path), save_raw_data=True)
        reloaded = autoserialize_load(str(path))

        np.testing.assert_array_equal(reloaded._iter_losses, ptycho._iter_losses)
        np.testing.assert_allclose(
            reloaded.dset.scan_positions_px.detach().cpu().numpy(),
            ptycho.dset.scan_positions_px.detach().cpu().numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            reloaded.obj_model._obj.detach().cpu().numpy(),
            ptycho.obj_model._obj.detach().cpu().numpy(),
            rtol=0,
            atol=0,
        )

        losses = ptycho._iter_losses
        assert losses[-1] < 0.3 * losses[0], (
            f"loss should drop below 30% of initial in {self.NUM_ITERS} iters: "
            f"initial={losses[0]:.3e}, final={losses[-1]:.3e}, "
            f"ratio={losses[-1] / losses[0]:.2f}"
        )

        loss_after_reload = reloaded._iter_losses[-1]
        n_continue = 10
        reloaded.reconstruct(
            num_iters=n_continue,
            reset=False,
            batch_size=N**2,
            device=config.get_device(),
        )
        assert len(reloaded._iter_losses) == self.NUM_ITERS + n_continue, (
            "continuation must not reset history"
        )
        assert reloaded._iter_losses[-1] <= loss_after_reload, (
            "loss must not regress after reload — optimizer state likely lost"
        )
