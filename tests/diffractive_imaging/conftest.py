"""Shared synthetic 4D-STEM data for the direct-ptychography test modules.

Mirrors the simulation idiom in ``test_ptychography.py`` (white-noise phase object,
soft-aperture defocused probe, ``|FFT(obj_patch * probe)|**2``) but on a smaller grid and
with a probe small enough that parallax shifts stay well inside the scan.

Imported by ``test_direct_ptychography.py`` and ``test_direct_ptychography_montage.py`` via
``from .conftest import ...``, matching the pattern used by the tomography suite.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset3d, Dataset4d
from quantem.core.utils.utils import electron_wavelength_angstrom

N = 32  # detector / object gridpoints
Q_MAX = 0.5  # inverse Angstroms
Q_PROBE = Q_MAX / 4  # inverse Angstroms -> BF disk radius of N/8 = 4 px
PROBE_ENERGY = 300e3  # eV
SCAN_STEP_SIZE = 1  # pixels

SAMPLING = 1 / Q_MAX / 2  # Angstroms
RECIPROCAL_SAMPLING = 2 * Q_MAX / N  # inverse Angstroms
SEMIANGLE_CUTOFF = Q_PROBE * electron_wavelength_angstrom(PROBE_ENERGY) * 1e3  # mrad
SCAN_SAMPLING = SAMPLING * SCAN_STEP_SIZE  # Angstroms

#: fixing the fitted origin keeps the center-of-mass step exact and deterministic
ORIGIN = (N // 2, N // 2)

DECONVOLUTION_KERNELS = ("ssb", "obf", "prlx", "icom")


def integer_shift_defocus(pixel_shift_per_k_step: int, upsampling_factor: int = 1) -> float:
    """``C10`` (in Angstrom) placing every BF pixel's parallax shift on an exact pixel.

    For pure defocus the aberration gradient is ``dchi/dk = 2*pi*wavelength*C10*k``, so the
    lateral shift is ``wavelength*C10*k`` Angstrom. Because ``k = m * reciprocal_sampling``
    exactly (``spatial_frequencies`` uses ``fftfreq(n, 1/(dk*n))``), choosing

        C10 = p * scan_sampling / (U * wavelength * dk)

    gives a shift of exactly ``p * m`` upsampled pixels for integer detector index ``m``.
    """
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    return (
        pixel_shift_per_k_step
        * SCAN_SAMPLING
        / (upsampling_factor * wavelength * RECIPROCAL_SAMPLING)
    )


def make_complex_obj(seed: int = 42) -> np.ndarray:
    """White-noise pure-phase object."""
    rng = np.random.default_rng(seed)
    arr = rng.random((N, N))
    arr -= arr.mean()
    return np.exp(1.0j * arr.astype(np.float32))


def band_limited_phase(seed: int = 42, cutoff: float = 2 * Q_PROBE) -> np.ndarray:
    """Ground-truth phase, low-passed to what the bright-field disk can actually transfer.

    Correlating a reconstruction against the raw white-noise phase caps out near 0.4 simply
    because most of its power sits above the aperture cutoff; band-limiting first makes
    "did this recover the object" a meaningful question.
    """
    phase = np.angle(make_complex_obj(seed))
    qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
    q = np.hypot(qx[:, None], qy[None, :])
    limited = np.fft.ifft2(np.fft.fft2(phase) * (q <= cutoff)).real
    return limited - limited.mean()


def make_probe_array(defocus: float) -> np.ndarray:
    """Soft-aperture probe with ``C10 = defocus`` Angstrom."""
    qx = qy = np.fft.fftfreq(N, SAMPLING)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)

    aperture_fourier = np.sqrt(
        np.clip((Q_PROBE - q) / RECIPROCAL_SAMPLING + 0.5, 0, 1),
    )
    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * defocus
    probe_fourier = aperture_fourier * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return np.fft.ifft2(probe_fourier) * N


def scan_positions_px() -> np.ndarray:
    """``(N_pos, 2)`` raster positions in object pixels, "ij" (row, col) ordering."""
    n = N // SCAN_STEP_SIZE
    ii, jj = np.meshgrid(
        np.arange(n) * SCAN_STEP_SIZE,
        np.arange(n) * SCAN_STEP_SIZE,
        indexing="ij",
    )
    return np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)


def simulate_intensities(complex_obj: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """``(N_pos, N, N)`` diffraction intensities, corner-centered in reciprocal space.

    ``probe`` is either a single ``(N, N)`` array or a ``(N_pos, N, N)`` stack, which
    broadcasts against the extracted object patches and gives every scan position its own
    probe -- how a tilted sample is simulated.
    """
    positions_px = scan_positions_px()
    x0 = np.round(positions_px[:, 0]).astype(int)
    y0 = np.round(positions_px[:, 1]).astype(int)

    x_ind = np.fft.fftfreq(N, d=1 / N).astype(int)
    y_ind = np.fft.fftfreq(N, d=1 / N).astype(int)

    row = (x0[:, None, None] + x_ind[None, :, None]) % N
    col = (y0[:, None, None] + y_ind[None, None, :]) % N

    exit_waves = complex_obj[row, col] * probe
    return np.abs(np.fft.fft2(exit_waves)) ** 2


def make_dataset4d(defocus: float | None = None, seed: int = 42) -> Dataset4d:
    """``(n_scan, n_scan, N, N)`` 4D-STEM dataset, reciprocal axes fftshifted."""
    if defocus is None:
        defocus = integer_shift_defocus(1)

    complex_obj = make_complex_obj(seed)
    probe = make_probe_array(defocus)
    intensities = simulate_intensities(complex_obj, probe)

    n = N // SCAN_STEP_SIZE
    array = np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((n, n, N, N))

    return Dataset4d.from_array(
        array.astype(np.float32),
        name="synthetic 4D-STEM",
        sampling=(SCAN_SAMPLING, SCAN_SAMPLING, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A", "A", "A^-1", "A^-1"),
    )


def defocus_per_position(mean_defocus: float, defocus_gradient: tuple[float, float]) -> np.ndarray:
    """``(N_pos,)`` local defocus for a tilted sample, mean-zero about the scan centroid.

    Matches ``DirectPtychographyMontage._return_delta_c10``: the offset is measured from the
    centroid of the scan positions in Angstrom, in the unrotated scan frame.
    """
    positions_ang = scan_positions_px() * SCAN_SAMPLING
    offsets = positions_ang - positions_ang.mean(axis=0)
    return mean_defocus + offsets @ np.asarray(defocus_gradient, dtype=np.float64)


def make_tilted_dataset4d(
    mean_defocus: float,
    defocus_gradient: tuple[float, float],
    seed: int = 42,
) -> Dataset4d:
    """4D-STEM dataset whose defocus varies linearly across the field of view.

    The gradients used in the tests look enormous as tilts -- at this fixture's 32 Angstrom
    field of view a gradient of 12.7 A/A is needed to swing the parallax shift by a single
    scan pixel. That is an artifact of the tiny synthetic scan, not of the method: it is only
    a +/-12% swing on the ~1625 A baseline defocus.
    """
    complex_obj = make_complex_obj(seed)
    defocus = defocus_per_position(mean_defocus, defocus_gradient)
    probes = np.stack([make_probe_array(float(d)) for d in defocus])
    intensities = simulate_intensities(complex_obj, probes)

    n = N // SCAN_STEP_SIZE
    array = np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((n, n, N, N))

    return Dataset4d.from_array(
        array.astype(np.float32),
        name="synthetic tilted 4D-STEM",
        sampling=(SCAN_SAMPLING, SCAN_SAMPLING, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A", "A", "A^-1", "A^-1"),
    )


def _bilinear_sample_periodic(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional ``(..., 2)`` coordinates, wrapping at the edges."""
    n_rows, n_cols = image.shape
    base = np.floor(coords)
    frac = coords - base
    base = base.astype(np.int64)

    out = np.zeros(coords.shape[:-1], dtype=np.float64)
    for d_row, d_col in ((0, 0), (1, 0), (0, 1), (1, 1)):
        w_row = frac[..., 0] if d_row else 1 - frac[..., 0]
        w_col = frac[..., 1] if d_col else 1 - frac[..., 1]
        out += (
            w_row * w_col * image[(base[..., 0] + d_row) % n_rows, (base[..., 1] + d_col) % n_cols]
        )
    return out


def make_model_vbf_stack(
    mean_defocus: float,
    defocus_gradient: tuple[float, float] = (0.0, 0.0),
    scan_gpts: tuple[int, int] = (96, 96),
    seed: int = 42,
):
    """Virtual bright-field stack built directly from the parallax model, at any scan size.

    Returns ``(vbf_dataset, bf_mask_dataset, object_phase)``.

    Every image is ``v_m(r) = object[r + shift_m(C10(r))]``, the relation the montage inverts
    -- the ``+`` sign was measured against the 4D pipeline, where a bright-field image
    correlates with ``obj[r + shift]`` at 0.30 versus 0.09 for ``obj[r - shift]``.

    The full 4D simulator cannot be used for the position-dependent tests: its 32x32 scan
    only affords ~16-position patches, and at that size the per-patch defocus estimator has a
    ~430 Angstrom bias -- larger than the signal, as the zero-gradient control shows. This
    builds the same relation at whatever scan size the estimator needs, cheaply.

    Bright-field pixels are ordered by ``np.nonzero`` over the corner-centered mask, which is
    the order ``torch.nonzero`` gives in ``_return_bf_context``. Pass ``crop_bf_mask=False``
    so the class uses this mask verbatim and the ordering is preserved.
    """
    from quantem.core.datastructures import Dataset2d, Dataset3d

    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)

    # detector grid just large enough to hold the bright-field disk
    radius_px = int(round(Q_PROBE / RECIPROCAL_SAMPLING))
    n_det = 2 * radius_px + 3
    centered = np.hypot(*np.meshgrid(*(np.arange(n_det) - n_det // 2,) * 2, indexing="ij"))
    bf_mask = np.fft.ifftshift(centered <= radius_px)

    # k = m * reciprocal_sampling exactly, independent of the grid size
    freqs = np.fft.fftfreq(n_det, 1 / (RECIPROCAL_SAMPLING * n_det))
    inds_i, inds_j = np.nonzero(bf_mask)
    k_vec = np.stack((freqs[inds_i], freqs[inds_j]), axis=-1)  # (num_bf, 2)

    rng = np.random.default_rng(seed)
    phase = rng.random(scan_gpts)
    qx = np.fft.fftfreq(scan_gpts[0], SCAN_SAMPLING)
    qy = np.fft.fftfreq(scan_gpts[1], SCAN_SAMPLING)
    q = np.hypot(qx[:, None], qy[None, :])
    obj = np.fft.ifft2(np.fft.fft2(phase) * (q <= 2 * Q_PROBE)).real
    obj -= obj.mean()

    ii, jj = np.meshgrid(np.arange(scan_gpts[0]), np.arange(scan_gpts[1]), indexing="ij")
    positions_px = np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)

    offsets_ang = (positions_px - positions_px.mean(0)) * SCAN_SAMPLING
    c10 = mean_defocus + offsets_ang @ np.asarray(defocus_gradient, dtype=np.float64)

    # shift_px[m, n] = wavelength * C10[n] * k[m] / scan_sampling
    shifts = (
        wavelength * c10[None, :, None] * k_vec[:, None, :] / SCAN_SAMPLING
    )  # (num_bf, N_pos, 2)
    stack = _bilinear_sample_periodic(obj, positions_px[None] + shifts)

    vbf_dataset = Dataset3d.from_array(
        stack.reshape(len(k_vec), *scan_gpts).astype(np.float32),
        name="model virtual BF stack",
        sampling=(1.0, SCAN_SAMPLING, SCAN_SAMPLING),
        units=("index", "A", "A"),
    )
    bf_mask_dataset = Dataset2d.from_array(
        bf_mask,
        name="BF mask",
        sampling=(RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A^-1", "A^-1"),
    )
    return vbf_dataset, bf_mask_dataset, obj


def model_vbf_kwargs(defocus: float) -> dict:
    """Constructor kwargs for a stack from :func:`make_model_vbf_stack`."""
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        rotation_angle=0.0,
        aberration_coefs={"C10": defocus},
        crop_bf_mask=False,
        verbose=False,
    )


def direct_ptycho_kwargs(defocus: float) -> dict:
    """Constructor kwargs shared by both direct-ptychography classes."""
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        rotation_angle=0.0,
        aberration_coefs={"C10": defocus},
        force_fitted_origin=ORIGIN,
        verbose=False,
    )


def correlation(image: np.ndarray, reference: np.ndarray) -> float:
    """Pearson correlation between a reconstruction and a reference, means removed."""
    a = np.asarray(image, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def dataset4d():
    """Read-only synthetic dataset at the integer-shift defocus."""
    return make_dataset4d()


# ---------------------------------------------------------------------------
# white-noise / analytical-CTF harness
#
# The object has constant Fourier amplitude, so |FFT(reconstruction)| *is* the contrast
# transfer function and can be compared against an analytical one. This is the only fixture
# here that can see whether `upsampling_factor` extends the CTF ("unfolding") or merely
# replicates it: the scan pitch is deliberately coarser than the aperture's transfer limit,
# leaving a genuine factor of 2 above the scan Nyquist to recover. Ported from
# ipynb-playground/white-noise-ptycho.ipynb.
# ---------------------------------------------------------------------------

CTF_N = 64  # object gridpoints
CTF_K_MAX = 2.0  # inverse Angstroms
CTF_K_PROBE = 1.0  # inverse Angstroms -> transfers to 2 * K_PROBE
CTF_SCAN_STEP = 2  # object pixels -> scan Nyquist is half the transfer limit
CTF_SAMPLING = 1 / CTF_K_MAX / 2
CTF_RECIPROCAL_SAMPLING = 2 * CTF_K_MAX / CTF_N
CTF_SCAN_SAMPLING = CTF_SAMPLING * CTF_SCAN_STEP
CTF_SEMIANGLE = CTF_K_PROBE * electron_wavelength_angstrom(PROBE_ENERGY) * 1e3
CTF_ABERRATIONS = {"C10": 100, "C12": 50, "phi12": np.deg2rad(11)}


def white_noise_object_2D(n: int = CTF_N, phi0: float = 1.0, seed: int = 0) -> np.ndarray:
    """Real 2D array whose FFT has random phase and *constant* amplitude."""
    rng = np.random.default_rng(seed)
    even = n % 2 == 0
    pos = np.arange(1, (n if even else n + 1) // 2)
    neg = np.flip(np.arange(n // 2 + 1, n))

    arr = rng.standard_normal((n, n))
    arr[pos[:, None], pos[None, :]] = -arr[neg[:, None], neg[None, :]]
    arr[pos[:, None], neg[None, :]] = -arr[neg[:, None], pos[None, :]]
    arr[0, pos] = -arr[0, neg]
    arr[pos, 0] = -arr[neg, 0]
    if even:
        arr[n // 2, :] = 0
        arr[:, n // 2] = 0
    arr[0, 0] = 0

    return np.fft.ifft2(np.exp(2j * np.pi * arr) * phi0).real


def _ctf_grids():
    kx = ky = np.fft.fftfreq(CTF_N, CTF_SAMPLING)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    phi = np.arctan2(ky[None, :], kx[:, None])
    aperture = np.clip((CTF_K_PROBE - np.sqrt(k2)) / CTF_RECIPROCAL_SAMPLING + 0.5, 0, 1)
    return k2, phi, aperture


def ctf_probe(aberrations: dict = CTF_ABERRATIONS):
    """``(chi, probe_array)`` for the soft-aperture probe used by the CTF harness."""
    k2, phi, aperture = _ctf_grids()
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)

    chi = k2 * wavelength * np.pi * aberrations.get("C10", 0.0)
    chi = chi + (
        k2
        * wavelength
        * np.pi
        * aberrations.get("C12", 0.0)
        * np.cos(2 * (phi - aberrations.get("phi12", 0.0)))
    )
    probe_fourier = aperture * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return chi, np.fft.ifft2(probe_fourier) * CTF_N


def analytical_parallax_ctf(chi: np.ndarray):
    """``(full_band, scan_band)`` analytical parallax CTF, fftshifted."""
    _, _, aperture = _ctf_grids()
    aperture_0 = aperture / np.sqrt(np.sum(np.abs(aperture) ** 2))
    autocorr = np.real(np.fft.ifft2(np.abs(np.fft.fft2(aperture_0)) ** 2))
    full = np.fft.fftshift(np.abs(autocorr * -np.sin(chi)))
    return full, full[CTF_N // 4 : -CTF_N // 4, CTF_N // 4 : -CTF_N // 4]


def ctf_raster_positions() -> np.ndarray:
    """``(N_pos, 2)`` raster positions in *object* pixels."""
    axis = np.arange(0.0, CTF_N, CTF_SCAN_STEP)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=-1)


def ctf_jittered_positions(amplitude_scan_px: float, seed: int = 1) -> np.ndarray:
    """Raster positions scattered off the lattice, with the outer ring pinned.

    Pinning the boundary keeps the bounding box exactly the scan field of view, so a
    ``boundary="wrap"`` montage and a regridded reconstruction stay directly comparable.
    """
    positions = ctf_raster_positions()
    rng = np.random.default_rng(seed)
    amplitude = amplitude_scan_px * CTF_SCAN_STEP
    jitter = rng.uniform(-amplitude, amplitude, positions.shape)
    edge = (
        (positions[:, 0] == 0)
        | (positions[:, 1] == 0)
        | (positions[:, 0] == CTF_N - CTF_SCAN_STEP)
        | (positions[:, 1] == CTF_N - CTF_SCAN_STEP)
    )
    jitter[edge] = 0.0
    return positions + jitter


def ctf_simulate(complex_obj: np.ndarray, probe: np.ndarray, positions_px: np.ndarray):
    """``(N_pos, n, n)`` intensities, using the Fourier shift theorem off-lattice."""
    n = CTF_N
    if np.allclose(positions_px, np.round(positions_px)):
        x0 = np.round(positions_px[:, 0]).astype(int)
        y0 = np.round(positions_px[:, 1]).astype(int)
        idx = np.fft.fftfreq(n, d=1 / n).astype(int)
        patches = complex_obj[
            (x0[:, None, None] + idx[None, :, None]) % n,
            (y0[:, None, None] + idx[None, None, :]) % n,
        ]
    else:
        fx = fy = np.fft.fftfreq(n)
        obj_fourier = np.fft.fft2(complex_obj)
        ramp = np.exp(
            2j
            * np.pi
            * (
                fx[None, :, None] * positions_px[:, 0, None, None]
                + fy[None, None, :] * positions_px[:, 1, None, None]
            )
        )
        patches = np.fft.ifft2(obj_fourier[None] * ramp, axes=(-2, -1))

    return (np.abs(np.fft.fft2(patches * probe)) ** 2).astype(np.float32)


def ctf_dataset4d(intensities: np.ndarray) -> Dataset4d:
    side = int(round(np.sqrt(len(intensities))))
    return Dataset4d.from_array(
        np.fft.fftshift(intensities.reshape(side, side, CTF_N, CTF_N), axes=(-1, -2)),
        name="white noise 4D-STEM",
        sampling=(CTF_SCAN_SAMPLING,) * 2 + (CTF_RECIPROCAL_SAMPLING,) * 2,
        units=("A", "A", "A^-1", "A^-1"),
    )


def ctf_dataset3d(intensities: np.ndarray) -> Dataset3d:
    return Dataset3d.from_array(
        np.fft.fftshift(intensities, axes=(-1, -2)),
        name="white noise ungridded",
        sampling=(1.0, CTF_RECIPROCAL_SAMPLING, CTF_RECIPROCAL_SAMPLING),
        units=("index", "A^-1", "A^-1"),
    )


def measured_ctf(obj: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.abs(np.fft.fft2(obj)) * 2)


def ctf_band_scores(measured: np.ndarray, analytic: np.ndarray, pixel_size: float):
    """Correlation with the analytical CTF, inside and beyond the scan Nyquist."""
    if measured.shape != analytic.shape:
        raise ValueError(f"shape mismatch {measured.shape} vs {analytic.shape}")

    freq = np.fft.fftshift(np.fft.fftfreq(measured.shape[0], pixel_size))
    q = np.hypot(freq[:, None], freq[None, :])
    nyquist = 1 / (2 * CTF_SCAN_SAMPLING)

    def corr(mask):
        a, b = measured[mask].astype(np.float64), analytic[mask].astype(np.float64)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    extension = (q > nyquist) & (q <= 2 * CTF_K_PROBE)
    return corr(q <= nyquist), corr(extension)


def ctf_kwargs() -> dict:
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=CTF_SEMIANGLE,
        rotation_angle=0.0,
        aberration_coefs=CTF_ABERRATIONS,
        verbose=False,
    )


@pytest.fixture(scope="module")
def ctf_scene():
    """``(complex_obj, chi, probe, ctf_full, ctf_sub)`` for the white-noise harness."""
    chi, probe = ctf_probe()
    full, sub = analytical_parallax_ctf(chi)
    return np.exp(1j * white_noise_object_2D()), chi, probe, full, sub


def ctf_radial_correlation(measured: np.ndarray, analytic: np.ndarray, pixel_size: float) -> float:
    """Correlation of radially averaged CTF profiles, independent of grid shape.

    A grid sized to cover scattered positions rarely lands on the same shape as the
    analytical CTF, so compare profiles rather than pixels.
    """

    def profile(image, px, bins=40, q_max=2 * CTF_K_PROBE):
        freq = np.fft.fftshift(np.fft.fftfreq(image.shape[0], px))
        q = np.hypot(freq[:, None], freq[None, :])
        index = np.clip((q / q_max * bins).astype(int), 0, bins - 1)
        inside = q <= q_max
        return np.array(
            [
                image[inside & (index == i)].mean() if (inside & (index == i)).any() else 0.0
                for i in range(bins)
            ]
        )

    a = profile(measured, pixel_size)
    b = profile(analytic, CTF_SAMPLING)
    return float(np.corrcoef(a, b)[0, 1])


def analytic_probe_array(reconstruction, aberration_coefs):
    """The analytic probe of a reconstruction, materialized as a complex array.

    Feeding this back in as a `FourierProbe.from_array` must reproduce the reconstruction it
    came from -- which is what pins the empirical path against the analytic one.
    """
    from quantem.diffractive_imaging.complex_probe import (
        evaluate_probe,
        polar_spatial_frequencies,
    )

    k, phi = polar_spatial_frequencies(
        reconstruction.gpts, reconstruction.sampling, device=reconstruction.device
    )
    return evaluate_probe(
        k * reconstruction.wavelength,
        phi,
        reconstruction.semiangle_cutoff,
        reconstruction.angular_sampling,
        reconstruction.wavelength,
        aberration_coefs=aberration_coefs,
    )


def ctf_interleaved_frames(
    complex_obj: np.ndarray,
    probe: np.ndarray,
    drift: np.ndarray,
    seed: int = 3,
):
    """Split the CTF raster into interleaved frames, each displaced by a known drift.

    Models a multi-frame acquisition: every frame samples the whole field of view sparsely,
    the frames interleave to fill it, and the specimen moves between them. The recorded
    positions stay nominal while the *object* is imaged displaced, which is what specimen
    drift does and what ``estimate_frame_drift`` has to undo.

    The sign follows the specimen: a feature at object coordinate ``x`` moves to ``x +
    drift`` in the microscope's frame, so it is measured when the probe is nominally at
    ``x + drift`` -- equivalently the probe illuminates ``nominal - drift``. The
    reconstruction then shows the object translated by ``+drift``, and ``positions - drift``
    puts it back.

    Returns ``(datasets, positions_A)``, both lists of length ``len(drift)``, with positions
    in Angstrom -- what the ungridded constructors take.
    """
    positions = ctf_raster_positions()
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(positions)) % len(drift)

    datasets, positions_A = [], []
    for frame, offset in enumerate(np.asarray(drift, dtype=np.float64)):
        nominal = positions[assignment == frame]
        illuminated = nominal - offset / CTF_SAMPLING
        intensities = ctf_simulate(complex_obj, probe, illuminated)
        datasets.append(ctf_dataset3d(intensities))
        positions_A.append(nominal * CTF_SAMPLING)

    return datasets, positions_A


# ---------------------------------------------------------------------------
# complex-valued and multislice objects
#
# A thin-layer multislice simulator on a 0.25 A grid with a 64 x 64 detector window and a
# step-1 scan, so the scan samples the full 2 k_alpha band and nothing aliases. The diffraction
# origin is pinned: fitting it resamples every pattern bilinearly, which mixes neighbouring
# detector pixels at the aperture edge, where the low-frequency amplitude/phase separation
# lives (4.7% amplitude-to-phase leakage fitted against 0.3% pinned).
# ---------------------------------------------------------------------------

ML_SAMPLING = 0.25  # Angstrom
ML_N_DET = 64  # detector window, pixels: the 16 A object period, which makes the simulation exact
ML_RECIPROCAL_SAMPLING = 1 / (ML_N_DET * ML_SAMPLING)
ML_K_PROBE = 1.0  # inverse Angstrom
ML_SEMIANGLE = ML_K_PROBE * electron_wavelength_angstrom(PROBE_ENERGY) * 1e3
ML_ABERRATIONS = {"C10": 50.0, "C12": 20.0, "phi12": np.deg2rad(11)}


def ml_probe_fourier(aberrations: dict) -> np.ndarray:
    """``psi(k)`` on the detector window, soft aperture, unit total intensity."""
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    k = np.fft.fftfreq(ML_N_DET, ML_SAMPLING)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return analytic_psi(kx, ky, aberrations, wavelength)


def library_probe_fourier(aberrations: dict) -> np.ndarray:
    """``psi(k)`` on the detector window from the library's own aberration surface.

    Unlike :func:`ml_probe_fourier`, which writes out only ``C10`` and ``C12``, this takes any
    coefficient, which the tests of odd aberrations need.
    """
    from quantem.diffractive_imaging.complex_probe import FourierProbe

    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    angular = ML_RECIPROCAL_SAMPLING * wavelength * 1e3
    probe = FourierProbe.from_aberrations(
        wavelength, ML_SEMIANGLE, (angular, angular), aberrations
    )
    k = torch.fft.fftfreq(ML_N_DET, ML_SAMPLING, dtype=torch.float64)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    return probe.at(kx, ky).numpy()


def analytic_psi(kx, ky, aberrations: dict, wavelength: float) -> np.ndarray:
    """Soft-aperture probe ``A(k) exp(-i chi(k))`` at arbitrary ``k``, matching the library."""
    k2 = kx**2 + ky**2
    phi = np.arctan2(ky, kx)
    aperture = np.clip((ML_K_PROBE - np.sqrt(k2)) / ML_RECIPROCAL_SAMPLING + 0.5, 0, 1)
    chi = (
        np.pi
        * wavelength
        * k2
        * (
            aberrations.get("C10", 0.0)
            + aberrations.get("C12", 0.0) * np.cos(2 * (phi - aberrations.get("phi12", 0.0)))
        )
    )
    return aperture * np.exp(-1j * chi)


def thin_layer_4dstem(
    log_transmissions,
    depths,
    aberrations: dict = ML_ABERRATIONS,
    dose=None,
    seed=0,
    probe_modes=None,
) -> Dataset4d:
    """4D-STEM through thin layers ``t = exp(a + i phi)`` at ``depths``, exact multislice.

    ``log_transmissions`` are ``a + i phi`` arrays on the ``ML_SAMPLING`` grid, periodic.
    Depths are downstream positive and relative to the plane the aberrations describe, so a
    layer at ``z`` sees the probe at ``C10 + z``.

    ``probe_modes``, ``(M, ML_N_DET, ML_N_DET)`` corner-centred ``psi_m(k)``, replaces the
    analytic probe: the patterns are then the incoherent sum over the modes.
    """
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    shape = np.asarray(log_transmissions[0]).shape
    k = torch.fft.fftfreq(ML_N_DET, ML_SAMPLING, dtype=torch.float64)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    if probe_modes is None:
        probe_modes = ml_probe_fourier(aberrations)[None]
    psi = torch.as_tensor(np.asarray(probe_modes), dtype=torch.complex128)
    psi = psi / psi.abs().square().sum().sqrt()

    order = np.argsort(depths)
    depths = np.asarray(depths, dtype=float)[order]
    layers = [
        torch.exp(torch.as_tensor(np.asarray(log_transmissions[i]), dtype=torch.complex128))
        for i in order
    ]
    probe = torch.fft.ifft2(psi * torch.exp(-1j * np.pi * wavelength * depths[0] * k2)) * ML_N_DET

    idx = torch.fft.fftfreq(ML_N_DET, d=1 / ML_N_DET).long()
    cols = torch.arange(shape[1])
    out = np.empty((shape[0], shape[1], ML_N_DET, ML_N_DET), dtype=np.float32)
    for row in range(shape[0]):
        rr = ((row + idx)[None, :, None] % shape[0]).expand(shape[1], -1, ML_N_DET)
        cc = ((cols[:, None] + idx[None, :])[:, None, :] % shape[1]).expand(-1, ML_N_DET, -1)
        wave = probe[None] * layers[0][rr, cc][:, None]
        for j in range(1, len(depths)):
            propagator = torch.exp(-1j * np.pi * wavelength * (depths[j] - depths[j - 1]) * k2)
            wave = torch.fft.ifft2(torch.fft.fft2(wave) * propagator) * layers[j][rr, cc][:, None]
        out[row] = torch.fft.fft2(wave).abs().square().sum(1).numpy()

    if dose is not None:
        rng = np.random.default_rng(seed)
        out = rng.poisson(out * dose * ML_SAMPLING**2).astype(np.float32)
    return Dataset4d.from_array(
        np.fft.fftshift(out, axes=(-2, -1)),
        name="thin-layer multislice",
        sampling=(ML_SAMPLING, ML_SAMPLING, ML_RECIPROCAL_SAMPLING, ML_RECIPROCAL_SAMPLING),
        units=("A", "A", "A^-1", "A^-1"),
    )


def ml_kwargs(aberrations: dict = ML_ABERRATIONS) -> dict:
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=ML_SEMIANGLE,
        rotation_angle=0.0,
        aberration_coefs=aberrations,
        force_fitted_origin=(ML_N_DET // 2, ML_N_DET // 2),
        edge_blend_pixels=0,
        verbose=False,
    )


def band_limit_ml(image: np.ndarray, cutoff: float = 2 * ML_K_PROBE) -> np.ndarray:
    """Low-pass to what the bright-field disk transfers, DC removed."""
    qx = np.fft.fftfreq(image.shape[0], ML_SAMPLING)
    qy = np.fft.fftfreq(image.shape[1], ML_SAMPLING)
    q = np.hypot(qx[:, None], qy[None, :])
    return np.fft.ifft2(np.fft.fft2(image) * ((q <= cutoff) & (q > 0))).real


def periodic_gaussians(positions, shape, sampling=ML_SAMPLING, sigma=0.35, weights=None):
    """Sum of periodic Gaussians at ``(row, col)`` positions in Angstrom."""
    rows = np.arange(shape[0]) * sampling
    cols = np.arange(shape[1]) * sampling
    extent = np.asarray(shape) * sampling
    weights = np.ones(len(positions)) if weights is None else weights
    out = np.zeros(shape)
    for (r, c), w in zip(positions, weights):
        dr = (rows[:, None] - r + extent[0] / 2) % extent[0] - extent[0] / 2
        dc = (cols[None, :] - c + extent[1] / 2) % extent[1] - extent[1] / 2
        out += w * np.exp(-(dr**2 + dc**2) / (2 * sigma**2))
    return out


def _lattice_in_box(basis_vectors, motif, rotation, box):
    """Lattice sites with a weighted motif, rotated and folded into a periodic box."""
    c, s = np.cos(rotation), np.sin(rotation)
    rot = np.array([[c, -s], [s, c]])
    a1, a2 = (rot @ v for v in basis_vectors)
    reach = int(np.ceil(2 * max(box) / min(np.linalg.norm(a1), np.linalg.norm(a2)))) + 2
    sites = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for offset, weight in motif:
                fractional = (i * a1 + j * a2 + rot @ offset) / np.asarray(box)
                fractional = fractional - np.floor(fractional + 1e-9)
                if np.all(fractional < 1 - 1e-9):
                    sites.append((*(fractional * box), weight))
    sites = np.unique(np.round(np.asarray(sites), 6), axis=0)
    return sites[:, :2], sites[:, 2]


def twisted_square_cell():
    """Two square lattices twisted by 36.87 degrees (Sigma 5), ``(shape, layers, twist)``.

    ``a = 3.2`` A puts five cells across a 16 A box, which holds both lattices exactly:
    ``(5a, 0)`` is ``4 a1' - 3 a2'`` of the rotated one. The motif is a strong corner site and
    a half-strength centre site, a perovskite-like column pattern. Each layer's (10)
    reflection then lands on the integer frequency pixels ``(5, 0)`` and ``(4, 3)``.
    """
    a, box = 3.2, (16.0, 16.0)
    basis = (np.array([a, 0.0]), np.array([0.0, a]))
    motif = [(np.zeros(2), 1.0), (np.array([a / 2, a / 2]), 0.5)]
    twist = 2 * np.arctan(1 / 3)
    shape = tuple(int(round(b / ML_SAMPLING)) for b in box)
    layers = []
    for angle in (0.0, twist):
        sites, weights = _lattice_in_box(basis, motif, angle, box)
        layers.append(periodic_gaussians(sites, shape, weights=weights))
    return shape, layers, float(np.degrees(twist))


def periodic_shift(image: np.ndarray, shift, sampling: float = ML_SAMPLING) -> np.ndarray:
    """``image`` translated by ``shift`` Angstrom, periodically, by the Fourier shift theorem."""
    q = np.fft.fftfreq(image.shape[0], sampling)
    ramp = np.exp(-2j * np.pi * (q[:, None] * shift[0] + q[None, :] * shift[1]))
    return np.fft.ifft2(np.fft.fft2(image) * ramp).real


def tilted_layers_4dstem(
    tilt_mrad=(0.0, 0.0), spacing: float = 6.0, dose=None, seed=0, drift=(0.0, 0.0)
) -> Dataset4d:
    """The twisted square layers, ``0.3`` rad strong and ``spacing`` apart about the C10 plane,
    seen at a specimen tilt: each displaced by its depth times the tilt, and both by ``drift``.
    """
    _, layers, _ = twisted_square_cell()
    theta = np.asarray(tilt_mrad, dtype=float) * 1e-3
    depths = (-spacing / 2, spacing / 2)
    logs = [
        1j * periodic_shift(0.3 * layer, z * theta + np.asarray(drift))
        for layer, z in zip(layers, depths)
    ]
    return thin_layer_4dstem(logs, list(depths), ML_ABERRATIONS, dose=dose, seed=seed)


def layer_purity(slices, truths) -> tuple[float, float]:
    """``(purity, correlation)`` averaged over layers, each true layer scored at its own slice.

    ``purity = c_own^2 / sum_m c_m^2`` with ``c_m`` the correlation of that slice with layer
    ``m``: 1 when a slice holds only its own layer, ``1 / n_layers`` when it holds all equally.
    """
    truths = [band_limit_ml(t) for t in truths]
    purities, owns = [], []
    for layer, image in enumerate(slices):
        image = band_limit_ml(np.asarray(image))
        c = np.array([correlation(image, t) for t in truths])
        purities.append(c[layer] ** 2 / (c**2).sum())
        owns.append(c[layer])
    return float(np.mean(purities)), float(np.mean(owns))


# --- Implicit-object (ObjectINR / ObjectTensorDecomp) reconstruction data ---------------------
# A smooth phase object larger than the scanned region, scanned in its interior with padding
# >= roi // 2, so patches never reach the object boundary (the non-toroidal regime the
# implicit models assume). Used by test_object_inr.py and test_object_tensor_decomp.py.

INR_N = 40  # detector / roi size (px)
INR_OGT = 64  # ground-truth object size (px); larger than the scanned region
INR_PAD = 20  # obj padding (>= roi // 2 so interior patches never hit the boundary)
INR_Q_MAX = 0.5  # inverse Angstroms
INR_Q_PROBE = INR_Q_MAX / 2
INR_ENERGY = 300e3  # eV
INR_C10 = 50.0  # defocus (Angstrom)
INR_STEP = 2  # scan step (px)
INR_SCAN_START = 20  # first scan position (px); INR_SCAN_START - roi//2 >= 0
INR_SCAN_STOP = 44  # exclusive; INR_SCAN_STOP - 1 + roi//2 - 1 < INR_OGT -> no wrap


def inr_smooth_phase() -> np.ndarray:
    """A smooth, band-limited phase object (friendly to a SIREN, unlike white noise)."""
    yy, xx = np.meshgrid(np.arange(INR_OGT), np.arange(INR_OGT), indexing="ij")
    phase = 0.7 * np.sin(2 * np.pi * xx / INR_OGT * 4) * np.cos(2 * np.pi * yy / INR_OGT * 3)
    return phase.astype(np.float32)


def inr_probe_array() -> np.ndarray:
    sampling = 1 / INR_Q_MAX / 2
    reciprocal_sampling = 2 * INR_Q_MAX / INR_N
    qx = qy = np.fft.fftfreq(INR_N, sampling)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)
    aperture = np.sqrt(np.clip((INR_Q_PROBE - q) / reciprocal_sampling + 0.5, 0, 1))
    chi = q**2 * electron_wavelength_angstrom(INR_ENERGY) * np.pi * INR_C10
    probe_fourier = aperture * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return (np.fft.ifft2(probe_fourier) * INR_N).astype(np.complex64)


def inr_semiangle_mrad() -> float:
    return electron_wavelength_angstrom(INR_ENERGY) * INR_Q_PROBE * 1e3


def inr_probe_params() -> dict:
    return {"energy": INR_ENERGY, "C10": INR_C10, "semiangle_cutoff": inr_semiangle_mrad()}


def inr_synthetic_dataset():
    """Simulate the non-toroidal 4D-STEM dataset; return ``(dataset_model, gt_phase, probe)``."""
    from quantem.core.datastructures.dataset4dstem import Dataset4dstem
    from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster

    phase = inr_smooth_phase()
    complex_obj = np.exp(1j * phase)
    probe = inr_probe_array()
    reciprocal_sampling = 2 * INR_Q_MAX / INR_N

    gpos = np.arange(INR_SCAN_START, INR_SCAN_STOP, INR_STEP)
    xx, yy = np.meshgrid(gpos, gpos, indexing="ij")
    positions = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    x0 = positions[:, 0].astype(int)
    y0 = positions[:, 1].astype(int)
    x_ind = np.fft.fftfreq(INR_N, d=1 / INR_N).astype(int)
    y_ind = np.fft.fftfreq(INR_N, d=1 / INR_N).astype(int)
    row = x0[:, None, None] + x_ind[None, :, None]  # no modulo: interior scan, no wrap
    col = y0[:, None, None] + y_ind[None, None, :]
    assert row.min() >= 0 and row.max() < INR_OGT and col.min() >= 0 and col.max() < INR_OGT
    exit_waves = complex_obj[row, col] * probe
    intensities = np.abs(np.fft.fft2(exit_waves)) ** 2

    sxy = len(gpos)
    dset = Dataset4dstem.from_array(
        array=np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((sxy, sxy, INR_N, INR_N)),
        sampling=(INR_STEP, INR_STEP, reciprocal_sampling, reciprocal_sampling),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.learn_scan_positions = False
    pdset.learn_descan = False
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=INR_ENERGY,
    )
    return pdset, phase, probe


def inr_ptychography(obj_model, pdset=None, probe_model=None):
    """``Ptychography`` on the synthetic dataset; the exact simulation probe, frozen, by default."""
    from quantem.diffractive_imaging.detector_models import DetectorPixelated
    from quantem.diffractive_imaging.probe_models import ProbePixelated
    from quantem.diffractive_imaging.ptychography import Ptychography

    gt_phase = None
    if pdset is None:
        pdset, gt_phase, probe = inr_synthetic_dataset()
        if probe_model is None:
            probe_model = ProbePixelated.from_array(
                num_probes=1, probe_params=inr_probe_params(), probe_array=probe.copy()
            )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=DetectorPixelated(),
        rng=0,
        verbose=False,
    )
    ptycho.preprocess(obj_padding_px=(INR_PAD, INR_PAD))
    return ptycho, gt_phase


def _inr_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum() + 1e-12))


def _inr_center_crop(a: np.ndarray, s: int) -> np.ndarray:
    r0 = (a.shape[0] - s) // 2
    c0 = (a.shape[1] - s) // 2
    return a[r0 : r0 + s, c0 : c0 + s]


def inr_best_corr(recon: np.ndarray, gt: np.ndarray, s: int = 32) -> float:
    """Best |correlation| over small integer shifts (handles ~1px registration offset)."""
    g = _inr_center_crop(gt, s)
    best = -1.0
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r = _inr_center_crop(np.roll(recon, (dr, dc), (0, 1)), s)
            best = max(best, abs(_inr_corr(r, g)))
    return best
