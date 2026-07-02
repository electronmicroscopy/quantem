"""
Utility tests for partitioned-PRISM ptychography in the regimes it was built for:
multislice (parent-wave amortization + thickness compensation) and mixed-state
(learnable beam coefficients as probe modes).

Ground truths follow the white-noise object convention: data is simulated with an
independent numpy multislice using quantem's own CTF, so the models can represent
it exactly (up to the partitioning approximation under test). See
scripts/benchmark_prism.py for the full comparison report these are distilled from.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    ProbePRISM,
    Ptychography,
    PtychographyDatasetRaster,
    PtychographyPRISM,
)
from quantem.diffractive_imaging._natural_neighbors_interpolation import beamlet_weights
from quantem.diffractive_imaging.complex_probe import fourier_space_probe
from quantem.diffractive_imaging.probe_models import (
    _partitioned_prism_wave_vectors,
    _prism_wave_vectors,
)

N = 64
K_MAX = 2.0  # inverse Angstroms
K_PROBE = 1.0  # inverse Angstroms
ENERGY = 300e3
WAVELENGTH = electron_wavelength_angstrom(ENERGY)
SAMPLING = 1 / K_MAX / 2  # Angstroms
RECIPROCAL_SAMPLING = 2 * K_MAX / N  # inverse Angstroms
SEMIANGLE = K_PROBE * WAVELENGTH * 1e3  # mrad
PHI0 = 1.0
BATCH_SIZE = N**2 // 8

ABERRATIONS = {"C10": 100.0, "C12": 50.0, "phi12": float(np.deg2rad(11))}
PROBE_PARAMS = {"energy": ENERGY, "semiangle_cutoff": SEMIANGLE, **ABERRATIONS}


# region --- simulation helpers (mirroring scripts/benchmark_prism.py) ---


def white_noise_object_2D(n: int, phi0: float, rng: np.random.Generator) -> np.ndarray:
    """Real 2D array whose FFT has unit amplitude and random (Hermitian) phase."""
    evenQ = n % 2 == 0
    pos_ind = np.arange(1, (n if evenQ else n + 1) // 2)
    neg_ind = np.flip(np.arange(n // 2 + 1, n))

    arr = rng.standard_normal((n, n))
    arr[pos_ind[:, None], pos_ind[None, :]] = -arr[neg_ind[:, None], neg_ind[None, :]]
    arr[pos_ind[:, None], neg_ind[None, :]] = -arr[neg_ind[:, None], pos_ind[None, :]]
    arr[0, pos_ind] = -arr[0, neg_ind]
    arr[pos_ind, 0] = -arr[neg_ind, 0]
    if evenQ:
        arr[n // 2, :] = 0
        arr[:, n // 2] = 0
    arr[0, 0] = 0

    return np.fft.ifft2(np.exp(2j * np.pi * arr) * phi0).real


def quantem_ctf(aberrations: dict = ABERRATIONS) -> np.ndarray:
    return (
        fourier_space_probe(
            gpts=(N, N),
            sampling=(SAMPLING, SAMPLING),
            energy=ENERGY,
            semiangle_cutoff=SEMIANGLE,
            aberration_coefs=aberrations,
            normalized=True,
        )
        .numpy()
        .astype(np.complex64)
    )


def simulate_multislice_intensities(
    transmissions: np.ndarray,
    probes: list[np.ndarray],
    mode_weights: np.ndarray,
    dz: float,
) -> np.ndarray:
    """Independent numpy multislice + incoherent mode sum on the full scan grid."""
    num_slices, n, _ = transmissions.shape
    positions = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    x_ind = np.fft.fftfreq(n, d=1 / n).astype(int)
    row = (positions[:, 0, None, None] + x_ind[None, :, None]) % n
    col = (positions[:, 1, None, None] + x_ind[None, None, :]) % n
    patches = transmissions[:, row, col].astype(np.complex64)

    k = np.fft.fftfreq(n, SAMPLING)
    k2 = (k[:, None] ** 2 + k[None, :] ** 2).astype(np.float32)
    propagator = np.exp(-1j * np.pi * WAVELENGTH * dz * k2).astype(np.complex64)

    intensities = np.zeros((n * n, n, n), dtype=np.float32)
    for w, probe in zip(mode_weights, probes):
        waves = patches[0] * probe.astype(np.complex64)[None]
        for s in range(1, num_slices):
            waves = np.fft.ifft2(np.fft.fft2(waves) * propagator[None])
            waves = waves * patches[s]
        intensities += w * np.abs(np.fft.fft2(waves)) ** 2
    return intensities


def make_pdset(intensities: np.ndarray) -> PtychographyDatasetRaster:
    dataset = Dataset4dstem.from_array(
        np.fft.fftshift(intensities.reshape((N, N, N, N)), axes=(-1, -2)),
        sampling=[SAMPLING, SAMPLING, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING],
        units=["A", "A", "A^-1", "A^-1"],
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dataset)
    pdset.preprocess(
        # synthetic patterns are perfectly centered; COM fitting would misread
        # asymmetric probe modes as detector misalignment and shift the patterns
        com_fit_function="no_shift",
        plot_rotation=False,
        plot_com=False,
        probe_energy=ENERGY,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    return pdset


def make_basis_modes(K: int, rng: np.random.Generator) -> tuple[list, np.ndarray]:
    """K source modes expressible exactly as CTF x Sibson weights x parent coefs.

    Gentle, non-orthogonal perturbations of the plain CTF: mixed states only
    depend on the density matrix, and this keeps the dominant eigenmode CTF-like
    and reachable from the CTF initialization.
    """
    cutoff = SEMIANGLE + float(np.linalg.norm([RECIPROCAL_SAMPLING * WAVELENGTH * 1e3] * 2))
    extent = np.array([N * SAMPLING] * 2)
    parents = _partitioned_prism_wave_vectors(cutoff, WAVELENGTH, num_rings=3)
    beamlets = _prism_wave_vectors(cutoff, extent, WAVELENGTH)
    weight_maps = beamlet_weights(parents, beamlets, (N, N), (SAMPLING, SAMPLING))

    coefs = 1.0 + 0.35 * (
        rng.standard_normal((K, len(parents))) + 1j * rng.standard_normal((K, len(parents)))
    ) / np.sqrt(2)
    kmodes = np.einsum("kb,bxy->kxy", coefs, weight_maps) * quantem_ctf()[None]
    kmodes /= np.linalg.norm(kmodes.reshape(K, -1), axis=1)[:, None, None]
    probes = [np.fft.ifft2(km.astype(np.complex64)) * N for km in kmodes]
    mode_weights = np.array([0.5, 0.3, 0.2][:K])
    return probes, mode_weights / mode_weights.sum()


def top_k_eigenmodes(probes: list[np.ndarray], weights: np.ndarray, K: int) -> np.ndarray:
    """Top-K eigenmodes (k-space) of the incoherent density matrix sum_j w_j |p_j><p_j|."""
    kmodes = np.stack([np.fft.fft2(pr) for pr in probes]).reshape(len(probes), -1)
    weighted = np.sqrt(weights)[:, None] * kmodes
    _, _, vh = np.linalg.svd(weighted, full_matrices=False)
    return vh[:K].reshape(K, N, N)


def make_source_blur_modes(blur_px: float = 0.8) -> tuple[list, np.ndarray]:
    """Center + 4 sub-pixel-shifted probes with Gaussian weights (source-size blur)."""
    ctf = quantem_ctf()
    k = np.fft.fftfreq(N, SAMPLING).astype(np.float32)
    offsets_px = np.array(
        [[0.0, 0.0], [blur_px, 0.0], [-blur_px, 0.0], [0.0, blur_px], [0.0, -blur_px]]
    )
    weights = np.exp(-0.5 * (np.linalg.norm(offsets_px, axis=1) / blur_px) ** 2)
    weights /= weights.sum()
    probes = []
    for dx, dy in offsets_px * SAMPLING:
        ramp = np.exp(-2j * np.pi * (k[:, None] * dx + k[None, :] * dy))
        probes.append(np.fft.ifft2(ctf * ramp) * N)
    return probes, weights


def principal_angles_deg(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the subspaces spanned by two mode stacks."""
    qa, _ = np.linalg.qr(modes_a.reshape(len(modes_a), -1).T)
    qb, _ = np.linalg.qr(modes_b.reshape(len(modes_b), -1).T)
    svals = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(svals, 0, 1)))


def build_prism(
    pdset,
    num_slices: int = 1,
    dz: float | None = None,
    num_probes: int = 1,
    num_partitions: int = 3,
    dense: bool = False,
    learn_beam_coefficients: bool = False,
) -> PtychographyPRISM:
    return PtychographyPRISM.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(
            num_slices=num_slices, obj_type="complex", slice_thicknesses=dz
        ),
        probe_model=ProbePRISM.from_params(
            num_probes=num_probes,
            probe_params=PROBE_PARAMS,
            num_partitions=num_partitions,
            dense=dense,
            learn_aberrations=False,
            learn_beam_coefficients=learn_beam_coefficients,
        ),
        detector_model=DetectorPixelated(),
        rng=42,
        verbose=False,
    ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)


def reconstruct_mixed(prism: PtychographyPRISM, num_iters: int) -> None:
    prism.reconstruct(
        num_iters=num_iters,
        reset=True,
        optimizer_params={
            "object": {"type": "SGD", "lr": 0.125},
            "probe": {"beam_coefficients": {"type": "adam", "lr": 1e-2}},
        },
        batch_size=BATCH_SIZE,
    )


def recovered_kmodes(prism: PtychographyPRISM) -> np.ndarray:
    return prism.probe_model._compute_beamlet_basis_fft().sum(dim=1).detach().cpu().numpy()


# endregion


@pytest.fixture
def multislice_data():
    """Thick 4-slice white-noise object and its simulated dataset."""
    rng = np.random.default_rng(42)
    num_slices, dz = 4, 25.0
    phases = np.stack(
        [white_noise_object_2D(N, PHI0 / num_slices, rng) for _ in range(num_slices)]
    )
    probe = np.fft.ifft2(quantem_ctf()) * N
    intensities = simulate_multislice_intensities(
        np.exp(1j * phases), [probe], np.array([1.0]), dz
    )
    return phases, dz, make_pdset(intensities)


class TestPartitionedForwardError:
    """Forward-only utility properties of the partitioned approximation."""

    def test_error_decreases_with_partitions_and_compensation_helps(self, multislice_data):
        phases, dz, pdset = multislice_data
        num_slices = phases.shape[0]
        transmissions = torch.tensor(np.exp(1j * phases), dtype=torch.complex64)
        batch_indices = torch.arange(0, N**2, 16)

        def forward(dense, parts, compensation):
            prism = build_prism(
                pdset, num_slices=num_slices, dz=dz, num_partitions=parts, dense=dense
            )
            prism.obj_model._obj.data = transmissions.clone()
            prism.thickness_compensation = compensation
            with torch.no_grad():
                pred, _ = prism._forward_batch(batch_indices)
            return pred

        dense_pred = forward(True, 3, False)

        # dense limit is exact: matches the independent numpy targets (amplitudes)
        targets = pdset._targets[batch_indices]
        dense_err = ((dense_pred.sqrt() - targets).norm() / targets.norm()).item()
        assert dense_err < 1e-4

        errors_on, errors_off = [], []
        for parts in (2, 3, 4):
            for compensation, errors in ((True, errors_on), (False, errors_off)):
                pred = forward(False, parts, compensation)
                errors.append(((pred - dense_pred).norm() / dense_pred.norm()).item())

        # accuracy improves monotonically with partitions (compensated)
        assert errors_on[0] > errors_on[1] > errors_on[2]
        # thickness compensation is what makes the partitioned expansion viable
        for err_on, err_off in zip(errors_on, errors_off):
            assert err_on < 0.1 * err_off


@pytest.mark.slow
class TestMultisliceReconstruction:
    def test_prism_converges_near_conventional(self, multislice_data):
        phases, dz, pdset = multislice_data
        num_slices = phases.shape[0]
        num_iters = 30
        opt = {"object": {"type": "SGD", "lr": 0.125}}

        conventional = Ptychography.from_models(
            dset=pdset,
            obj_model=ObjectPixelated.from_uniform(
                num_slices=num_slices, obj_type="complex", slice_thicknesses=dz
            ),
            probe_model=ProbePixelated.from_params(num_probes=1, probe_params=PROBE_PARAMS),
            detector_model=DetectorPixelated(),
            rng=42,
            verbose=False,
        ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)
        conventional.probe_model._probe.requires_grad_(False)
        conventional.reconstruct(
            num_iters=num_iters, reset=True, optimizer_params=opt, batch_size=BATCH_SIZE
        )

        prism = build_prism(pdset, num_slices=num_slices, dz=dz, num_partitions=4)
        prism.reconstruct(
            num_iters=num_iters, reset=True, optimizer_params=opt, batch_size=BATCH_SIZE
        )

        assert conventional._iter_losses[-1] < 1e-3
        # partitioned PRISM converges to (near) its forward-model floor, which on
        # this deliberately thick object sits well above the conventional floor
        assert prism._iter_losses[-1] < 6e-2
        assert prism._iter_losses[-1] < 0.3 * prism._iter_losses[0]


@pytest.mark.slow
class TestMixedState:
    NUM_ITERS = 150

    def test_basis_modes_recovered(self):
        """Modes built in PRISM's own parameterization are recovered by
        learn_beam_coefficients: the data is exactly representable, so the loss
        approaches machine precision and the dominant eigenmode is found. (The
        weak modes carry only a few % of intensity, so their subspace angles are
        ill-conditioned at this loss level and not asserted.)"""
        rng = np.random.default_rng(42)
        phase = white_noise_object_2D(N, PHI0, rng)
        probes, weights = make_basis_modes(3, rng)
        pdset = make_pdset(
            simulate_multislice_intensities(np.exp(1j * phase)[None], probes, weights, 1.0)
        )
        true_kmodes = top_k_eigenmodes(probes, weights, 3)

        prism = build_prism(pdset, num_probes=3, num_partitions=3, learn_beam_coefficients=True)
        reconstruct_mixed(prism, self.NUM_ITERS)

        assert prism._iter_losses[-1] < 5e-3
        angles = principal_angles_deg(recovered_kmodes(prism), true_kmodes)
        assert angles[0] < 5.0

    def test_source_blur_partial_coherence_utility(self):
        """K=3 modes capture source-size blur substantially better than a
        coherent (K=1) model, and the recovered mode subspace matches the blur
        kernel's top eigenmodes."""
        rng = np.random.default_rng(42)
        phase = white_noise_object_2D(N, PHI0, rng)
        probes, weights = make_source_blur_modes()
        pdset = make_pdset(
            simulate_multislice_intensities(np.exp(1j * phase)[None], probes, weights, 1.0)
        )
        true_kmodes = top_k_eigenmodes(probes, weights, 3)

        losses = {}
        for num_probes in (1, 3):
            prism = build_prism(
                pdset, num_probes=num_probes, num_partitions=4, learn_beam_coefficients=True
            )
            reconstruct_mixed(prism, self.NUM_ITERS)
            losses[num_probes] = prism._iter_losses[-1]

        assert losses[3] < 0.7 * losses[1]
        angles = principal_angles_deg(recovered_kmodes(prism), true_kmodes)
        assert angles.max() < 15.0
