"""Multislice / mixed-state utility benchmark for partitioned-PRISM ptychography.

Compares PtychographyPRISM against the conventional Ptychography engine on
clean synthetic ground truths in the spirit of the white-noise object test:

1. Multislice reconstruction: white-noise phase slices, data simulated by an
   independent numpy multislice; loss / per-slice SSIM / wall time / peak GPU
   memory for the conventional engine vs PRISM over a num_partitions sweep.
2. Forward error vs num_partitions (pure forward, no reconstruction): relative
   intensity error of partitioned PRISM against the dense forward on a thick
   object, with thickness compensation on and off; plus dense PRISM vs the
   independent numpy simulation.
3. Mixed-state, source-size blur (physical): incoherent sum of sub-pixel
   shifted probes; K-mode PRISM with learnable beam coefficients vs a
   single-mode PRISM and a pixelated mixed-state baseline.
4. Mixed-state, beamlet-basis modes (exactness anchor): modes built in PRISM's
   own parameterization; recovery should approach machine precision.

Usage:
    uv run python scripts/benchmark_prism.py                # CPU-small (n=64)
    uv run python scripts/benchmark_prism.py --large --device gpu
    uv run python scripts/benchmark_prism.py --sections 2,4 --num-iters 30
"""

import argparse
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from quantem.core import config
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

# region --- simulation helpers ---


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


def patch_indices_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/col gather indices for a full n x n integer scan grid with wraparound."""
    positions = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    x_ind = np.fft.fftfreq(n, d=1 / n).astype(int)
    row = (positions[:, 0, None, None] + x_ind[None, :, None]) % n
    col = (positions[:, 1, None, None] + x_ind[None, None, :]) % n
    return row, col


def simulate_multislice_intensities(
    transmissions: np.ndarray,  # (S, n, n) complex
    probes: list[np.ndarray],  # K real-space probes, each (n, n), unit L2
    mode_weights: np.ndarray,  # (K,) summing to 1
    dz: float,
    wavelength: float,
    sampling: float,
) -> np.ndarray:
    """Independent numpy multislice + incoherent mode sum.

    Fresnel propagator exp(-1j*pi*lambda*dz*k^2), matching
    ProbeBase._compute_propagator_arrays.
    """
    num_slices, n, _ = transmissions.shape
    row, col = patch_indices_grid(n)
    patches = transmissions[:, row, col].astype(np.complex64)  # (S, P, n, n)

    k = np.fft.fftfreq(n, sampling)
    k2 = (k[:, None] ** 2 + k[None, :] ** 2).astype(np.float32)
    propagator = np.exp(-1j * np.pi * wavelength * dz * k2).astype(np.complex64)

    intensities = np.zeros((n * n, n, n), dtype=np.float32)
    for w, probe in zip(mode_weights, probes):
        waves = patches[0] * probe.astype(np.complex64)[None]
        for s in range(1, num_slices):
            waves = np.fft.ifft2(np.fft.fft2(waves) * propagator[None])
            waves = waves * patches[s]
        intensities += w * np.abs(np.fft.fft2(waves)) ** 2
    return intensities


def make_pdset(
    intensities: np.ndarray, n: int, sampling: float, reciprocal_sampling: float, energy: float
) -> PtychographyDatasetRaster:
    dataset = Dataset4dstem.from_array(
        np.fft.fftshift(intensities.reshape((n, n, n, n)), axes=(-1, -2)),
        sampling=[sampling, sampling, reciprocal_sampling, reciprocal_sampling],
        units=["A", "A", "A^-1", "A^-1"],
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dataset)
    pdset.preprocess(
        # synthetic patterns are perfectly centered by construction; COM origin
        # fitting would misread asymmetric probe modes as detector misalignment
        # and sub-pixel-shift the patterns (unrepresentable by a fixed aperture)
        com_fit_function="no_shift",
        plot_rotation=False,
        plot_com=False,
        probe_energy=energy,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    return pdset


def aligned_phase_ssim(recon_slice: np.ndarray, true_phase: np.ndarray) -> float:
    """SSIM between reconstructed and true slice phases, mean-aligned."""
    phase = np.angle(recon_slice)
    phase -= phase.mean()
    ref = true_phase - true_phase.mean()
    rng_val = float(max(ref.max() - ref.min(), 1e-6))
    return float(ssim(phase, ref, data_range=rng_val))


def principal_angles_deg(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the subspaces spanned by two mode stacks.

    Modes are only defined up to unitary mixing (incoherent sums are invariant),
    so subspace angles are the right recovery metric.
    """
    qa, _ = np.linalg.qr(modes_a.reshape(len(modes_a), -1).T)
    qb, _ = np.linalg.qr(modes_b.reshape(len(modes_b), -1).T)
    svals = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(svals, 0, 1)))


# endregion

# region --- benchmark scaffolding ---


class Params:
    def __init__(self, large: bool):
        self.n = 128 if large else 64
        self.k_max = 2.0
        self.k_probe = 1.0
        self.energy = 300e3
        self.wavelength = electron_wavelength_angstrom(self.energy)
        self.sampling = 1 / self.k_max / 2
        self.reciprocal_sampling = 2 * self.k_max / self.n
        self.semiangle = self.k_probe * self.wavelength * 1e3
        self.num_slices = 8 if large else 2
        self.dz = 20.0
        self.phi0 = 1.0
        self.batch_size = self.n**2 // 8
        self.aberrations = {"C10": 100.0, "C12": 50.0, "phi12": float(np.deg2rad(11))}

    @property
    def probe_params(self) -> dict:
        return {"energy": self.energy, "semiangle_cutoff": self.semiangle, **self.aberrations}

    def ctf(self, aberrations: dict | None = None) -> np.ndarray:
        return (
            fourier_space_probe(
                gpts=(self.n, self.n),
                sampling=(self.sampling, self.sampling),
                energy=self.energy,
                semiangle_cutoff=self.semiangle,
                aberration_coefs=self.aberrations if aberrations is None else aberrations,
                normalized=True,
            )
            .numpy()
            .astype(np.complex64)
        )

    def real_probe(self, ctf: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(ctf) * self.n  # unit L2 in real space


def multislice_truth(p: Params, rng: np.random.Generator) -> np.ndarray:
    phases = np.stack(
        [white_noise_object_2D(p.n, p.phi0 / p.num_slices, rng) for _ in range(p.num_slices)]
    )
    return phases  # (S, n, n) real; transmission = exp(1j * phases)


def peak_gpu_mem_mb(device: str) -> float | None:
    if torch.cuda.is_available() and "cpu" not in str(device):
        return torch.cuda.max_memory_allocated() / 1e6
    return None


def reset_gpu_mem(device: str) -> None:
    if torch.cuda.is_available() and "cpu" not in str(device):
        torch.cuda.reset_peak_memory_stats()


def forward_intensities(engine, batch_indices: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred, _ = engine._forward_batch(batch_indices)
    return pred


# endregion

# region --- sections ---


def build_multislice_engines(p: Params, pdset, num_partitions, device, dense=False):
    """Conventional engine and one PRISM engine sharing probe params; probe frozen
    at the true aberrations to isolate multislice object recovery."""

    def obj_model():
        return ObjectPixelated.from_uniform(
            num_slices=p.num_slices, obj_type="complex", slice_thicknesses=p.dz
        )

    conventional = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model(),
        probe_model=ProbePixelated.from_params(num_probes=1, probe_params=p.probe_params),
        detector_model=DetectorPixelated(),
        rng=42,
        verbose=False,
        device=device,
    ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)
    # freeze the (correct) pixelated probe so both engines only learn the object
    conventional.probe_model._probe.requires_grad_(False)

    prism = PtychographyPRISM.from_models(
        dset=pdset,
        obj_model=obj_model(),
        probe_model=ProbePRISM.from_params(
            num_probes=1,
            probe_params=p.probe_params,
            num_partitions=num_partitions,
            dense=dense,
            learn_aberrations=False,
        ),
        detector_model=DetectorPixelated(),
        rng=42,
        verbose=False,
        device=device,
    ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)
    return conventional, prism


def section_multislice(p: Params, args, rng) -> None:
    print("\n=== 1. Multislice reconstruction: conventional vs PRISM ===")
    print(f"    n={p.n}, slices={p.num_slices}, dz={p.dz} A, batch={p.batch_size}")

    phases = multislice_truth(p, rng)
    transmissions = np.exp(1j * phases)
    probe = p.real_probe(p.ctf())
    intensities = simulate_multislice_intensities(
        transmissions, [probe], np.array([1.0]), p.dz, p.wavelength, p.sampling
    )
    pdset = make_pdset(intensities, p.n, p.sampling, p.reciprocal_sampling, p.energy)

    opt = {"object": {"type": "SGD", "lr": 0.125}}
    rows = []

    def run(engine, tag, **recon_kwargs):
        reset_gpu_mem(args.device)
        t0 = time.perf_counter()
        engine.reconstruct(
            num_iters=args.num_iters,
            reset=True,
            optimizer_params=opt,
            batch_size=p.batch_size,
            **recon_kwargs,
        )
        dt = time.perf_counter() - t0
        obj = engine.obj_model.obj.detach().cpu().numpy()
        ssims = [aligned_phase_ssim(obj[s], phases[s]) for s in range(p.num_slices)]
        mem = peak_gpu_mem_mb(args.device)
        rows.append((tag, engine._iter_losses[-1], np.mean(ssims), dt / args.num_iters, mem))

    conventional, _ = build_multislice_engines(p, pdset, 3, args.device)
    run(conventional, "conventional multislice")

    for parts in args.partitions:
        _, prism = build_multislice_engines(p, pdset, parts, args.device)
        run(prism, f"PRISM partitions={parts}", parent_batch_size=args.parent_batch_size)

    if args.checkpointing:
        _, prism = build_multislice_engines(p, pdset, max(args.partitions), args.device)
        run(
            prism,
            f"PRISM partitions={max(args.partitions)} +ckpt",
            parent_batch_size=args.parent_batch_size,
            use_checkpointing=True,
        )

    print(f"    {'config':34s} {'loss':>10s} {'slice SSIM':>10s} {'s/iter':>8s} {'GPU MB':>8s}")
    for tag, loss, s, t, mem in rows:
        mem_s = f"{mem:8.0f}" if mem is not None else "     n/a"
        print(f"    {tag:34s} {loss:10.3e} {s:10.4f} {t:8.2f} {mem_s}")


def section_forward_error(p: Params, args, rng) -> None:
    print("\n=== 2. Partitioned forward error vs num_partitions (pure forward) ===")
    num_slices = max(4, p.num_slices)
    dz = 25.0
    print(f"    n={p.n}, slices={num_slices}, dz={dz} A (thick on purpose)")

    p_thick = Params(args.large)
    p_thick.num_slices, p_thick.dz = num_slices, dz
    phases = multislice_truth(p_thick, rng)
    transmissions = np.exp(1j * phases)
    probe = p.real_probe(p.ctf())
    intensities = simulate_multislice_intensities(
        transmissions, [probe], np.array([1.0]), dz, p.wavelength, p.sampling
    )
    pdset = make_pdset(intensities, p.n, p.sampling, p.reciprocal_sampling, p.energy)
    batch_indices = torch.arange(0, p.n**2, max(1, p.n**2 // 256))

    def prism_forward(dense, parts, compensation):
        _, prism = build_multislice_engines(p_thick, pdset, parts, args.device, dense=dense)
        prism.obj_model._obj.data = torch.tensor(
            transmissions, dtype=torch.complex64, device=prism._single_device
        )
        prism.thickness_compensation = compensation
        return forward_intensities(prism, batch_indices)

    dense_pred = prism_forward(True, 3, True)

    # cross-check both engines' forwards (ground-truth object) against the stored
    # targets (amplitude space), which came from the independent numpy simulation;
    # the conventional line calibrates how much of the residual is dataset
    # preprocessing/normalization rather than the PRISM decomposition
    targets = pdset._targets[batch_indices].to(dense_pred.device)
    dense_vs_numpy = ((dense_pred.sqrt() - targets).norm() / targets.norm()).item()

    conventional, _ = build_multislice_engines(p_thick, pdset, 3, args.device)
    conventional.obj_model._obj.data = torch.tensor(
        transmissions, dtype=torch.complex64, device=conventional._single_device
    )
    conv_pred = forward_intensities(conventional, batch_indices)
    conv_vs_numpy = ((conv_pred.sqrt() - targets).norm() / targets.norm()).item()
    dense_vs_conv = ((dense_pred - conv_pred).norm() / conv_pred.norm()).item()
    print(f"    conventional vs independent numpy simulation: rel err {conv_vs_numpy:.2e}")
    print(f"    dense PRISM  vs independent numpy simulation: rel err {dense_vs_numpy:.2e}")
    print(f"    dense PRISM  vs conventional (intensities):   rel err {dense_vs_conv:.2e}")

    print(f"    {'partitions':>10s} {'rel err (comp ON)':>18s} {'rel err (comp OFF)':>19s}")
    for parts in args.partitions:
        errs = []
        for compensation in (True, False):
            pred = prism_forward(False, parts, compensation)
            errs.append(((pred - dense_pred).norm() / dense_pred.norm()).item())
        print(f"    {parts:>10d} {errs[0]:>18.3e} {errs[1]:>19.3e}")


def make_source_blur_modes(p: Params, blur_px: float = 0.8) -> tuple[list, np.ndarray]:
    """Center + 4 sub-pixel shifted probes with Gaussian weights (source-size blur)."""
    ctf = p.ctf()
    k = np.fft.fftfreq(p.n, p.sampling).astype(np.float32)
    offsets_px = np.array(
        [[0.0, 0.0], [blur_px, 0.0], [-blur_px, 0.0], [0.0, blur_px], [0.0, -blur_px]]
    )
    weights = np.exp(-0.5 * (np.linalg.norm(offsets_px, axis=1) / blur_px) ** 2)
    weights /= weights.sum()
    probes = []
    for dx, dy in offsets_px * p.sampling:  # to Angstroms
        ramp = np.exp(-2j * np.pi * (k[:, None] * dx + k[None, :] * dy))
        probes.append(p.real_probe(ctf * ramp))
    return probes, weights


def make_basis_modes(p: Params, K: int, rng) -> tuple[list, np.ndarray]:
    """K source modes expressible exactly as CTF x Sibson weights x per-parent coefs.

    The modes are gentle, non-orthogonal perturbations of the plain CTF (mixed
    states only depend on the density matrix, so orthogonality of the *source*
    modes is not required); recovery is compared against the density matrix's
    top-K eigenmodes, whose dominant mode stays CTF-like and reachable from the
    CTF initialization.
    """
    cutoff = p.semiangle + float(np.linalg.norm([p.reciprocal_sampling * p.wavelength * 1e3] * 2))
    extent = np.array([p.n * p.sampling] * 2)
    parents = _partitioned_prism_wave_vectors(cutoff, p.wavelength, num_rings=3)
    beamlets = _prism_wave_vectors(cutoff, extent, p.wavelength)
    weight_maps = beamlet_weights(parents, beamlets, (p.n, p.n), (p.sampling, p.sampling))

    ctf = p.ctf()
    coefs = 1.0 + 0.35 * (
        rng.standard_normal((K, len(parents))) + 1j * rng.standard_normal((K, len(parents)))
    ) / np.sqrt(2)
    kmodes = np.einsum("kb,bxy->kxy", coefs, weight_maps) * ctf[None]
    kmodes /= np.linalg.norm(kmodes.reshape(K, -1), axis=1)[:, None, None]  # unit power each
    probes = [p.real_probe(km.astype(np.complex64)) for km in kmodes]
    mode_weights = np.array([0.5, 0.3, 0.2][:K])
    mode_weights /= mode_weights.sum()
    return probes, mode_weights


def run_mixed_state(p, args, pdset, num_probes, num_partitions, dense, tag, rows, true_kmodes):
    prism = PtychographyPRISM.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(num_slices=1, obj_type="complex"),
        probe_model=ProbePRISM.from_params(
            num_probes=num_probes,
            probe_params=p.probe_params,
            num_partitions=num_partitions,
            dense=dense,
            learn_aberrations=False,
            learn_beam_coefficients=True,
        ),
        detector_model=DetectorPixelated(),
        rng=42,
        verbose=False,
        device=args.device,
    ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)

    prism.reconstruct(
        num_iters=args.num_iters,
        reset=True,
        optimizer_params={
            "object": {"type": "SGD", "lr": 0.125},
            "probe": {"beam_coefficients": {"type": "adam", "lr": 1e-2}},
        },
        batch_size=p.batch_size,
    )
    recovered = prism.probe_model._compute_beamlet_basis_fft().sum(dim=1).detach().cpu().numpy()
    angles = (
        principal_angles_deg(recovered, true_kmodes)
        if len(recovered) == len(true_kmodes)
        else None
    )
    rows.append((tag, prism._iter_losses[-1], angles))
    return prism


def print_mixed_rows(rows) -> None:
    print(f"    {'config':38s} {'loss':>10s}  principal angles (deg)")
    for tag, loss, angles in rows:
        ang = "n/a (K mismatch)" if angles is None else np.array2string(angles, precision=1)
        print(f"    {tag:38s} {loss:10.3e}  {ang}")


def top_k_eigenmodes(probes: list[np.ndarray], weights: np.ndarray, K: int) -> np.ndarray:
    """Top-K eigenmodes (k-space) of the incoherent density matrix sum_j w_j |p_j><p_j|.

    Recovered mixed-state modes are only defined up to the eigenbasis of the
    truth's density matrix, so this is the right stack for subspace comparison.
    """
    kmodes = np.stack([np.fft.fft2(pr) for pr in probes]).reshape(len(probes), -1)
    weighted = np.sqrt(weights)[:, None] * kmodes
    _, svals, vh = np.linalg.svd(weighted, full_matrices=False)
    captured = np.sum(svals[:K] ** 2) / np.sum(svals**2)
    print(f"    truth blur kernel: top-{K} eigenmodes capture {100 * captured:.2f}% of intensity")
    n = int(np.sqrt(vh.shape[1]))
    return vh[:K].reshape(K, n, n)


def section_source_blur(p: Params, args, rng) -> None:
    print("\n=== 3. Mixed-state: source-size blur (physical partial coherence) ===")
    phase = white_noise_object_2D(p.n, p.phi0, rng)
    probes, weights = make_source_blur_modes(p)
    intensities = simulate_multislice_intensities(
        np.exp(1j * phase)[None], probes, weights, p.dz, p.wavelength, p.sampling
    )
    pdset = make_pdset(intensities, p.n, p.sampling, p.reciprocal_sampling, p.energy)

    true_kmodes = top_k_eigenmodes(probes, weights, 3)

    rows = []
    run_mixed_state(p, args, pdset, 1, 3, False, "PRISM K=1 (coherent)", rows, true_kmodes)
    run_mixed_state(p, args, pdset, 3, 3, False, "PRISM K=3 partitions=3", rows, true_kmodes)
    run_mixed_state(p, args, pdset, 3, 4, False, "PRISM K=3 partitions=4", rows, true_kmodes)
    run_mixed_state(
        p, args, pdset, 3, 3, True, "PRISM K=3 dense (representable)", rows, true_kmodes
    )

    # pixelated mixed-state baseline
    conventional = Ptychography.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(num_slices=1, obj_type="complex"),
        probe_model=ProbePixelated.from_params(num_probes=3, probe_params=p.probe_params),
        detector_model=DetectorPixelated(),
        rng=42,
        verbose=False,
        device=args.device,
    ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)
    conventional.reconstruct(
        num_iters=args.num_iters,
        reset=True,
        optimizer_params={
            "object": {"type": "SGD", "lr": 0.125},
            "probe": {"type": "SGD", "lr": 0.125},
        },
        batch_size=p.batch_size,
    )
    recovered = np.stack(
        [np.fft.fft2(pr) for pr in conventional.probe_model.probe.detach().cpu().numpy()]
    )
    rows.append(
        (
            "pixelated K=3 (baseline)",
            conventional._iter_losses[-1],
            principal_angles_deg(recovered, true_kmodes),
        )
    )
    print_mixed_rows(rows)


def section_basis_modes(p: Params, args, rng) -> None:
    print("\n=== 4. Mixed-state: beamlet-basis modes (exactness anchor) ===")
    phase = white_noise_object_2D(p.n, p.phi0, rng)
    probes, weights = make_basis_modes(p, 3, rng)
    intensities = simulate_multislice_intensities(
        np.exp(1j * phase)[None], probes, weights, p.dz, p.wavelength, p.sampling
    )
    pdset = make_pdset(intensities, p.n, p.sampling, p.reciprocal_sampling, p.energy)
    true_kmodes = top_k_eigenmodes(probes, weights, 3)

    rows = []
    run_mixed_state(
        p, args, pdset, 3, 3, False, "PRISM K=3 partitions=3 (exact rep.)", rows, true_kmodes
    )
    print_mixed_rows(rows)


# endregion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--large", action="store_true", help="n=128, 8 slices (GPU scale)")
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--partitions", default="2,3,4")
    parser.add_argument("--parent-batch-size", type=int, default=None)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--sections", default="1,2,3,4")
    args = parser.parse_args()
    args.partitions = [int(s) for s in args.partitions.split(",")]

    if args.device != "cpu" and config.NUM_DEVICES > 0:
        config.set_device("gpu")

    p = Params(args.large)
    rng = np.random.default_rng(42)
    sections = {
        "1": section_multislice,
        "2": section_forward_error,
        "3": section_source_blur,
        "4": section_basis_modes,
    }
    for key in args.sections.split(","):
        sections[key.strip()](p, args, rng)


if __name__ == "__main__":
    main()
