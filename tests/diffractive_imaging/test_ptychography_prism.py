"""
Tests for partitioned-PRISM ptychography: Sibson interpolation weights, ProbePRISM,
and the PtychographyPRISM forward model (including exact equivalence with the
conventional engine in the dense limit).
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging._natural_neighbors_interpolation import (
    beamlet_weights,
    fourier_beamlet_weights,
    one_hot_beamlet_weights,
    pairwise_weights,
)
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectPixelated
from quantem.diffractive_imaging.probe_models import (
    ProbeParametric,
    ProbePRISM,
    _grid_prism_wave_vectors,
    _prism_wave_vectors,
)
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.diffractive_imaging.ptychography_prism import PtychographyPRISM

N = 64
Q_MAX = 0.5  # inverse Angstroms
Q_PROBE = Q_MAX / 2  # inverse Angstroms
PROBE_ENERGY = 300e3  # eV

RECIPROCAL_SAMPLING = np.array([2 * Q_MAX / N, 2 * Q_MAX / N])  # inverse Angstroms
SEMIANGLE_CUTOFF = Q_PROBE * electron_wavelength_angstrom(PROBE_ENERGY) * 1e3  # mrad
C10 = 50
MEAN_DIFFRACTION_INTENSITY = 100.0

PROBE_PARAMS = {
    "energy": PROBE_ENERGY,
    "semiangle_cutoff": SEMIANGLE_CUTOFF,
    "C10": C10,
}


# A cheap rings scheme (~3 rings / ~19 parents on this grid) so unit and
# reconstruction tests stay fast; the default grid+fourier uses many more parents
# (P~429) than these correctness checks need.
PARTITIONED_SCHEME = dict(parent_layout="rings", interpolation="sibson", interpolation_factor=12)


def make_prism_probe(num_probes=1, dense=False, **kwargs) -> ProbePRISM:
    probe_model = ProbePRISM.from_params(
        probe_params=PROBE_PARAMS,
        num_probes=num_probes,
        dense=dense,
        **kwargs,
    )
    probe_model.set_initial_probe((N, N), RECIPROCAL_SAMPLING, MEAN_DIFFRACTION_INTENSITY)
    return probe_model


def _hex_parents(cutoff_inv_A: float, num_rings: int = 3) -> np.ndarray:
    """Center point plus hexagonal rings of parent wave vectors, in inverse Angstroms."""
    rings = [np.array([[0.0, 0.0]])]
    n = 6
    for r in np.linspace(cutoff_inv_A / (num_rings - 1), cutoff_inv_A, num_rings - 1):
        angles = np.arange(n) * 2 * np.pi / n + np.pi / 2
        rings.append(np.stack([r * np.sin(angles), r * np.cos(-angles)], axis=1))
        n += 6
    return np.vstack(rings)


def _grid_beamlets(cutoff_inv_A: float, gpts, sampling) -> np.ndarray:
    """All fftfreq-grid wave vectors strictly inside the cutoff disk."""
    kx = np.fft.fftfreq(gpts[0], sampling[0])
    ky = np.fft.fftfreq(gpts[1], sampling[1])
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    mask = kxg**2 + kyg**2 < cutoff_inv_A**2
    return np.stack([kxg[mask], kyg[mask]], axis=1)


class TestSibsonWeights:
    """Unit tests for the natural-neighbor interpolation weights."""

    gpts = (N, N)
    sampling = (1 / Q_MAX / 2, 1 / Q_MAX / 2)
    cutoff = Q_PROBE

    def test_pairwise_weights_partition_of_unity(self):
        parents = _hex_parents(self.cutoff)
        beamlets = _grid_beamlets(self.cutoff, self.gpts, self.sampling)
        assert len(beamlets) > len(parents)

        weights = pairwise_weights(parents, beamlets)

        assert weights.shape == (len(parents), len(beamlets))
        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)

    def test_pairwise_weights_one_hot_at_parents(self):
        parents = _hex_parents(self.cutoff)
        weights = pairwise_weights(parents, parents)
        np.testing.assert_allclose(weights, np.eye(len(parents)), atol=1e-12)

    def test_beamlet_weights_scattered_onto_grid(self):
        parents = _hex_parents(self.cutoff)
        beamlets = _grid_beamlets(self.cutoff, self.gpts, self.sampling)

        maps = beamlet_weights(parents, beamlets, self.gpts, self.sampling)

        assert maps.shape == (len(parents), *self.gpts)
        total = maps.sum(axis=0)
        assert np.count_nonzero(total) == len(beamlets)
        np.testing.assert_allclose(total[total > 0], 1.0, atol=1e-12)

    def test_dense_fast_path_matches_general_path(self):
        beamlets = _grid_beamlets(self.cutoff, self.gpts, self.sampling)

        one_hot = one_hot_beamlet_weights(beamlets, self.gpts, self.sampling)
        general = beamlet_weights(beamlets, beamlets, self.gpts, self.sampling)

        assert one_hot.shape == (len(beamlets), *self.gpts)
        np.testing.assert_allclose(one_hot, general, atol=1e-12)

    def test_off_grid_wave_vectors_raise(self):
        beamlets = _grid_beamlets(self.cutoff, self.gpts, self.sampling)
        with pytest.raises(ValueError, match="reciprocal grid"):
            one_hot_beamlet_weights(beamlets + 1e-3, self.gpts, self.sampling)


class TestProbePRISM:
    """Unit tests for the PRISM probe model."""

    def _parametric_probe(self) -> torch.Tensor:
        probe_model = ProbeParametric.from_params(probe_params=PROBE_PARAMS)
        probe_model.set_initial_probe((N, N), RECIPROCAL_SAMPLING, MEAN_DIFFRACTION_INTENSITY)
        return probe_model.probe

    def test_probe_matches_parametric_partitioned(self):
        """Partition-of-unity weights (Sibson or Fourier) sum to 1 over the CTF
        support, so the summed beamlet basis reproduces the aberrated CTF probe
        exactly for any partitioning scheme."""
        reference = self._parametric_probe()
        for kwargs in ({"dense": True}, PARTITIONED_SCHEME, {"interpolation": "fourier"}):
            prism = make_prism_probe(**kwargs)
            probe = prism.probe
            assert probe.shape == (1, N, N)
            torch.testing.assert_close(probe, reference, rtol=1e-4, atol=1e-6)

    def test_beam_coefficient_init_preserves_probe(self):
        reference = make_prism_probe(learn_beam_coefficients=False).probe
        probe = make_prism_probe(learn_beam_coefficients=True).probe
        torch.testing.assert_close(probe, reference, rtol=0, atol=0)

    def test_vacuum_probe_intensity_replaces_aperture(self):
        """A measured aperture (e.g. a sqrt-edged one, unlike quantem's linear soft
        edge) propagates through the beamlet basis unchanged."""
        from quantem.diffractive_imaging.complex_probe import fourier_space_probe

        sampling = 1 / Q_MAX / 2
        kx = ky = np.fft.fftfreq(N, sampling)
        k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        aperture = np.sqrt(np.clip((Q_PROBE - k) / RECIPROCAL_SAMPLING[0] + 0.5, 0, 1))
        vacuum = (aperture**2 * 7.0).astype(np.float32)  # arbitrary scale

        prism = make_prism_probe(vacuum_probe_intensity=vacuum)

        expected_ctf = fourier_space_probe(
            gpts=(N, N),
            sampling=(sampling, sampling),
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            vacuum_probe_intensity=torch.tensor(vacuum),
            aberration_coefs={"C10": float(C10)},
            normalized=True,
        )
        expected = torch.fft.ifft2(expected_ctf)[None] * np.sqrt(
            MEAN_DIFFRACTION_INTENSITY * N * N
        )
        torch.testing.assert_close(prism.probe, expected, rtol=1e-4, atol=1e-6)

    def test_probe_intensity_normalization(self):
        probe = make_prism_probe().probe
        total = probe.abs().square().sum().item()
        assert total == pytest.approx(MEAN_DIFFRACTION_INTENSITY, rel=1e-4)

    def test_mixed_state_shapes_and_weights(self):
        weights = [0.7, 0.3]
        prism = make_prism_probe(
            num_probes=2,
            aberration_coefs=[{"C10": C10}, {"C10": -C10}],
            initial_probe_weights=weights,
        )
        probe = prism.probe
        assert probe.shape == (2, N, N)
        intensities = probe.abs().square().sum(dim=(-2, -1))
        ratios = (intensities / intensities.sum()).detach().numpy()
        np.testing.assert_allclose(ratios, weights, rtol=1e-4)

    def test_forward_returns_basis_and_phases(self):
        prism = make_prism_probe(num_probes=2, **PARTITIONED_SCHEME)
        fract_positions = torch.tensor([[0.25, -0.4], [0.0, 0.0], [0.5, 0.1]])
        beamlets_fft, position_coefs = prism.forward(fract_positions)
        assert beamlets_fft.shape == (2, prism.num_parent_beams, N, N)
        assert position_coefs.shape == (3, N, N)
        # zero fractional shift has unit phase
        torch.testing.assert_close(
            position_coefs[1],
            torch.ones_like(position_coefs[1]),
            rtol=0,
            atol=1e-6,
        )

    def test_gradient_flags(self):
        prism = make_prism_probe(learn_aberrations=True, learn_beam_coefficients=True)
        loss = prism.probe.abs().sum()
        loss.backward()
        assert prism.aberration_coefs[0]["C10"].grad is not None
        assert prism._beam_coefficients.grad is not None
        assert torch.any(prism._beam_coefficients.grad != 0)

        frozen = make_prism_probe(learn_aberrations=False, learn_beam_coefficients=False)
        assert not frozen.probe.requires_grad
        assert not frozen.aberration_coefs[0]["C10"].requires_grad
        assert not frozen._beam_coefficients.requires_grad

    def test_reset_restores_parameters(self):
        prism = make_prism_probe(learn_beam_coefficients=True)
        with torch.no_grad():
            prism.aberration_coefs[0]["C10"] += 25.0
            prism._beam_coefficients += 0.1
        prism.reset()
        assert prism.aberration_coefs[0]["C10"].item() == pytest.approx(C10)
        torch.testing.assert_close(
            prism._beam_coefficients, prism._initial_beam_coefficients, rtol=0, atol=0
        )

    def test_optimizer_single_spec_broadcasts_to_active_groups(self):
        prism = make_prism_probe(learn_aberrations=True, learn_beam_coefficients=True)
        prism.set_optimizer({"name": "adam", "lr": 1e-2})
        group_sizes = [len(g["params"]) for g in prism.optimizer.param_groups]
        assert len(group_sizes) == 2  # aberrations + beam_coefficients

        pplr = {
            "aberrations": {"name": "adam", "lr": 1.0},
            "beam_coefficients": {"name": "adam", "lr": 1e-3},
        }
        prism.set_optimizer(pplr)
        lrs = sorted(g["lr"] for g in prism.optimizer.param_groups)
        assert lrs == [1e-3, 1.0]

    def test_optimizer_bad_pplr_keys_raise(self):
        prism = make_prism_probe(learn_aberrations=True, learn_beam_coefficients=False)
        with pytest.raises(ValueError, match="do not match"):
            prism.set_optimizer({"beam_coefficients": {"name": "adam", "lr": 1e-3}})


# region --- engine fixtures (mirroring test_ptychography.py conventions) ---


@pytest.fixture
def complex_obj():
    """White-noise complex object, fixed seed for reproducibility."""
    rng = np.random.default_rng(42)
    arr = rng.random((N, N)).astype(np.float32)
    arr -= arr.mean()
    return np.exp(1.0j * arr)


def _simulate_dataset(complex_obj, probe_params) -> PtychographyDatasetRaster:
    """Synthetic single-slice 4D-STEM dataset on an N x N scan grid, simulated with
    quantem's own parametric CTF (so parametric probe models can represent it)."""
    positions = np.stack(np.meshgrid(np.arange(N), np.arange(N), indexing="ij"), axis=-1).reshape(
        -1, 2
    )

    x_ind = np.fft.fftfreq(N, d=1 / N).astype(int)
    row = (positions[:, 0, None, None] + x_ind[None, :, None]) % N
    col = (positions[:, 1, None, None] + x_ind[None, None, :]) % N

    probe_model = ProbeParametric.from_params(probe_params=probe_params)
    probe_model.set_initial_probe((N, N), RECIPROCAL_SAMPLING, 1.0)
    probe = probe_model.probe[0].detach().numpy()

    exit_waves = complex_obj[row, col] * probe
    intensities = np.abs(np.fft.fft2(exit_waves)) ** 2

    dset = Dataset4dstem.from_array(
        array=np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((N, N, N, N)),
        sampling=(1, 1, RECIPROCAL_SAMPLING[0], RECIPROCAL_SAMPLING[1]),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=PROBE_ENERGY,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    return pdset


@pytest.fixture
def ptycho_dataset(complex_obj):
    return _simulate_dataset(complex_obj, PROBE_PARAMS)


def _build_engines(
    ptycho_dataset,
    obj_arrays: np.ndarray,
    slice_thicknesses: float | None = None,
    conventional_c10: float = C10,
    obj_padding_px: tuple[int, int] = (0, 0),
    **prism_probe_kwargs,
) -> tuple[Ptychography, PtychographyPRISM]:
    """Conventional and PRISM engines sharing the dataset and ground-truth object."""
    conv_probe_params = dict(PROBE_PARAMS, C10=conventional_c10)

    conventional = Ptychography.from_models(
        dset=ptycho_dataset,
        obj_model=ObjectPixelated.from_array(obj_arrays, slice_thicknesses=slice_thicknesses),
        probe_model=ProbeParametric.from_params(probe_params=conv_probe_params),
        detector_model=DetectorPixelated(),
        rng=42,
    ).preprocess(obj_padding_px=obj_padding_px, plot_rotation=False, plot_com=False)

    prism = PtychographyPRISM.from_models(
        dset=ptycho_dataset,
        obj_model=ObjectPixelated.from_array(obj_arrays, slice_thicknesses=slice_thicknesses),
        probe_model=ProbePRISM.from_params(probe_params=PROBE_PARAMS, **prism_probe_kwargs),
        detector_model=DetectorPixelated(),
        rng=42,
    ).preprocess(obj_padding_px=obj_padding_px, plot_rotation=False, plot_com=False)

    return conventional, prism


def _forward_intensities(
    engine: Ptychography | PtychographyPRISM,
    batch_indices: torch.Tensor,
    descan: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run either engine's forward model on a batch, with an explicit descan."""
    patch_data, _pos, fract, _descan = engine.dset.forward(batch_indices, engine.obj_padding_px)
    if isinstance(engine, PtychographyPRISM):
        exit_waves = engine.forward_operator(patch_data, fract, descan)
    else:
        probes = engine.probe_model.forward(fract)
        obj_patches = engine.obj_model.forward(patch_data)
        _props, exit_waves = engine.forward_operator(obj_patches, probes, descan)
    return engine.detector_model.forward(exit_waves)


BATCH_INDICES = torch.arange(0, N * N, 131)  # 32 positions spread over the scan


# endregion --- engine fixtures ---


class TestPRISMForwardEquivalence:
    """Dense-limit PRISM must reproduce the conventional forward model exactly."""

    def test_vacuum_identity_multislice(self, ptycho_dataset):
        """Propagating parent waves through vacuum (with back-propagation and tilt
        correction) is the identity."""
        obj = np.ones((3, N, N), dtype=np.complex64)
        _, prism = _build_engines(ptycho_dataset, obj, slice_thicknesses=25.0, dense=False)
        prism._compute_object_propagators()
        waves = prism._propagate_parent_waves(
            prism.probe_model.parent_wave_vectors,
            torch.from_numpy(obj).to(torch.complex64),
        )
        torch.testing.assert_close(waves, torch.ones_like(waves), rtol=1e-4, atol=1e-5)

    def test_dense_single_slice_matches_conventional(self, ptycho_dataset, complex_obj):
        conventional, prism = _build_engines(ptycho_dataset, complex_obj[None], dense=True)

        pred_conv = _forward_intensities(conventional, BATCH_INDICES)
        pred_prism = _forward_intensities(prism, BATCH_INDICES)

        scale = pred_conv.abs().max()
        torch.testing.assert_close(pred_prism / scale, pred_conv / scale, rtol=1e-4, atol=1e-5)

    def test_dense_fractional_positions_and_descan(self, ptycho_dataset, complex_obj):
        conventional, prism = _build_engines(ptycho_dataset, complex_obj[None], dense=True)

        with torch.no_grad():
            conventional.dset._scan_positions_px += torch.tensor([0.3, -0.2])
        descan = 0.5 * torch.randn(
            len(BATCH_INDICES), 2, generator=torch.Generator().manual_seed(0)
        )

        pred_conv = _forward_intensities(conventional, BATCH_INDICES, descan)
        pred_prism = _forward_intensities(prism, BATCH_INDICES, descan)

        scale = pred_conv.abs().max()
        torch.testing.assert_close(pred_prism / scale, pred_conv / scale, rtol=1e-4, atol=1e-5)

    def test_dense_multislice_thickness_compensation(self, ptycho_dataset, complex_obj):
        """Thickness compensation (always-on back-propagation to the entrance plane)
        is a pure k-space phase in the far field, so dense PRISM still equals the
        conventional multislice forward exactly."""
        rng = np.random.default_rng(7)
        obj = np.stack(
            [complex_obj, np.exp(0.5j * (rng.random((N, N)) - 0.5)).astype(np.complex64)]
        )
        conventional, prism = _build_engines(
            ptycho_dataset,
            obj,
            slice_thicknesses=20.0,
            dense=True,
        )

        pred_conv = _forward_intensities(conventional, BATCH_INDICES)
        pred_prism = _forward_intensities(prism, BATCH_INDICES)

        scale = pred_conv.abs().max()
        torch.testing.assert_close(pred_prism / scale, pred_conv / scale, rtol=1e-4, atol=1e-5)

    def test_partitioned_close_to_conventional(self, ptycho_dataset, complex_obj):
        """Partitioned PRISM is an approximation; sanity-check it stays close."""
        conventional, prism = _build_engines(
            ptycho_dataset, complex_obj[None], dense=False, **PARTITIONED_SCHEME
        )

        pred_conv = _forward_intensities(conventional, BATCH_INDICES)
        pred_prism = _forward_intensities(prism, BATCH_INDICES)

        rel_error = (pred_prism - pred_conv).abs().sum() / pred_conv.abs().sum()
        assert rel_error < 0.05

    def test_prism_probe_rejected_by_conventional_engine(self, ptycho_dataset, complex_obj):
        with pytest.raises(TypeError, match="PtychographyPRISM"):
            Ptychography.from_models(
                dset=ptycho_dataset,
                obj_model=ObjectPixelated.from_array(complex_obj[None]),
                probe_model=ProbePRISM.from_params(probe_params=PROBE_PARAMS),
                detector_model=DetectorPixelated(),
            )


class TestPRISMGradients:
    def test_gradients_flow_to_all_learnables(self, ptycho_dataset, complex_obj):
        _, prism = _build_engines(
            ptycho_dataset,
            complex_obj[None],
            dense=False,
            **PARTITIONED_SCHEME,
            learn_aberrations=True,
            learn_beam_coefficients=True,
        )
        prism.dset.learn_scan_positions = True

        pred = _forward_intensities(prism, BATCH_INDICES[:8])
        pred.abs().sum().backward()

        assert prism.obj_model._obj.grad is not None
        assert prism.dset._scan_positions_px.grad is not None
        assert torch.any(prism.dset._scan_positions_px.grad[BATCH_INDICES[:8]] != 0)
        probe_model = prism.probe_model
        assert probe_model.aberration_coefs[0]["C10"].grad is not None
        assert probe_model._beam_coefficients.grad is not None

    def test_position_gradients_match_conventional_dense(self, ptycho_dataset, complex_obj):
        """In the dense limit the two engines are the same function of the scan
        positions, so their position gradients must agree.

        Uses a random-weighted functional of the predictions: the plain intensity
        sum is nearly shift-invariant, so its position gradient is a noise-level
        residual that cannot be compared meaningfully in float32.
        """
        idx = BATCH_INDICES[:8]
        weights = torch.randn(len(idx), N, N, generator=torch.Generator().manual_seed(1))
        conventional, prism = _build_engines(ptycho_dataset, complex_obj[None], dense=True)

        with torch.no_grad():
            ptycho_dataset._scan_positions_px += torch.tensor([0.3, -0.2])

        grads = []
        for engine in (conventional, prism):
            pred = _forward_intensities(engine, idx)
            (pred * weights).sum().backward()
            grads.append(engine.dset._scan_positions_px.grad[idx].clone())
            engine.dset._scan_positions_px.grad = None  # dset is shared between engines

        with torch.no_grad():
            ptycho_dataset._scan_positions_px -= torch.tensor([0.3, -0.2])

        scale = grads[0].abs().max()
        torch.testing.assert_close(grads[1] / scale, grads[0] / scale, rtol=1e-4, atol=1e-5)


class TestPRISMMemoryKnobs:
    """Parent-beam chunking and gradient checkpointing must not change the numbers.

    Uses a multislice object: single-slice objects take the collapsed fast path,
    which bypasses the parent-chunk loop entirely.
    """

    def test_chunked_and_checkpointed_match_unchunked(self, ptycho_dataset, complex_obj):
        rng = np.random.default_rng(7)
        obj = np.stack(
            [complex_obj, np.exp(0.5j * (rng.random((N, N)) - 0.5)).astype(np.complex64)]
        )
        _, prism = _build_engines(
            ptycho_dataset,
            obj,
            slice_thicknesses=20.0,
            dense=False,
            **PARTITIONED_SCHEME,
            learn_aberrations=True,
            learn_beam_coefficients=True,
        )
        idx = BATCH_INDICES[:8]
        weights = torch.randn(len(idx), N, N, generator=torch.Generator().manual_seed(2))
        watched = [
            prism.obj_model._obj,
            prism.probe_model._beam_coefficients,
            prism.probe_model.aberration_coefs[0]["C10"],
        ]

        def run() -> tuple[torch.Tensor, list[torch.Tensor]]:
            pred = _forward_intensities(prism, idx)
            (pred * weights).sum().backward()
            grads = [p.grad.clone() for p in watched]
            for model in (prism.obj_model, prism.probe_model, prism.dset):
                model.zero_grad()
            return pred.detach(), grads

        prism.parent_batch_size = None
        prism.use_checkpointing = False
        pred_ref, grads_ref = run()

        for parent_batch_size, use_checkpointing in ((5, False), (5, True), (None, True)):
            prism.parent_batch_size = parent_batch_size
            prism.use_checkpointing = use_checkpointing
            pred, grads = run()
            torch.testing.assert_close(pred, pred_ref, rtol=1e-4, atol=1e-6)
            for grad, grad_ref in zip(grads, grads_ref):
                scale = grad_ref.abs().max().clamp_min(1e-12)
                torch.testing.assert_close(grad / scale, grad_ref / scale, rtol=1e-4, atol=1e-5)


class TestParametricAberrationLearning:
    """Regression: the conventional engine's autograd backward used to rescale
    ProbeParametric gradients by sqrt(mean_diffraction_intensity), which sent
    radian-scale aberrations (phi12) unstable under SGD — learn_aberrations=True
    performed *worse* than a frozen probe. With aberrations initialized at the
    ground truth on representable data, learning must not degrade the fit."""

    def test_sgd_learning_at_truth_stays_converged(self, complex_obj):
        probe_params = dict(PROBE_PARAMS, C12=25.0, phi12=0.3)
        pdset = _simulate_dataset(complex_obj, probe_params)

        obj_model = ObjectPixelated.from_uniform(num_slices=1, obj_type="complex")
        probe_model = ProbeParametric.from_params(
            probe_params=probe_params, learn_aberrations=True
        )
        ptycho = Ptychography.from_models(
            dset=pdset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=DetectorPixelated(),
            rng=42,
            verbose=False,
        ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)

        ptycho.reconstruct(
            num_iters=15,
            reset=True,
            optimizer_params={
                "object": {"type": "SGD", "lr": 0.125},
                "probe": {"type": "SGD", "lr": 0.125},
            },
            batch_size=N**2 // 8,
        )

        losses = ptycho._iter_losses
        assert losses[-1] < 1e-2 * losses[0], (
            f"aberration learning degraded convergence: {losses[0]:.3e} -> {losses[-1]:.3e}"
        )
        assert probe_model.aberration_coefs["phi12"].item() == pytest.approx(0.3, abs=0.05)
        assert probe_model.aberration_coefs["C10"].item() == pytest.approx(C10, abs=1.0)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_gpu
class TestPRISMDeviceResidency:
    def test_probe_buffers_move_to_gpu(self):
        prism = make_prism_probe(learn_beam_coefficients=True)
        prism.to("cuda")
        assert prism._interpolation_weights.is_cuda
        assert prism._parent_wave_vectors.is_cuda
        assert prism._beam_coefficients.is_cuda
        assert prism._kxa.is_cuda
        assert prism.probe.is_cuda


@pytest.mark.slow
class TestPRISMIntegration:
    """End-to-end reconstruction through the inherited training loop."""

    def _reconstruct(self, ptycho_dataset, num_iters: int = 8) -> PtychographyPRISM:
        obj_model = ObjectPixelated.from_uniform(
            num_slices=1, obj_type="complex", slice_thicknesses=1
        )
        # start slightly defocused from the ground-truth C10 so aberration
        # learning has something to do
        probe_model = ProbePRISM.from_params(
            probe_params=dict(PROBE_PARAMS, C10=C10 - 10),
            **PARTITIONED_SCHEME,
            learn_aberrations=True,
            learn_beam_coefficients=True,
        )
        prism = PtychographyPRISM.from_models(
            dset=ptycho_dataset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=DetectorPixelated(),
            rng=42,
        ).preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)

        prism.reconstruct(
            num_iters=num_iters,
            reset=True,
            optimizer_params={
                "object": {"name": "adam", "lr": 5e-3},
                "probe": {
                    "aberrations": {"name": "adam", "lr": 1.0},
                    "beam_coefficients": {"name": "adam", "lr": 1e-2},
                },
            },
            batch_size=512,
            parent_batch_size=8,
            use_checkpointing=True,
        )
        return prism

    def test_reconstruction_converges(self, ptycho_dataset):
        prism = self._reconstruct(ptycho_dataset)
        losses = prism._iter_losses
        assert losses[-1] < 0.5 * losses[0], (
            f"loss should decrease: initial={losses[0]:.3e}, final={losses[-1]:.3e}"
        )

    def test_save_load_roundtrip(self, ptycho_dataset, tmp_path):
        from quantem.core.io.serialize import load as autoserialize_load

        prism = self._reconstruct(ptycho_dataset, num_iters=2)
        path = tmp_path / "prism.zip"
        prism.save(str(path), save_raw_data=True)
        reloaded = autoserialize_load(str(path))

        probe_model = prism.probe_model
        reloaded_probe = reloaded.probe_model
        torch.testing.assert_close(
            reloaded_probe._interpolation_weights.cpu(),
            probe_model._interpolation_weights.cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            reloaded_probe._beam_coefficients.detach().cpu(),
            probe_model._beam_coefficients.detach().cpu(),
            rtol=0,
            atol=0,
        )
        assert reloaded_probe.parent_layout == probe_model.parent_layout
        assert reloaded_probe.interpolation == probe_model.interpolation
        assert reloaded_probe.interpolation_factor == probe_model.interpolation_factor
        assert reloaded_probe.learn_beam_coefficients == probe_model.learn_beam_coefficients

        # continue training after reload
        reloaded.reconstruct(num_iters=1, batch_size=512, parent_batch_size=8)

    def test_reset_restores_probe_state(self, ptycho_dataset):
        prism = self._reconstruct(ptycho_dataset, num_iters=2)
        probe_model = prism.probe_model
        assert not torch.equal(
            probe_model._beam_coefficients.detach(), probe_model._initial_beam_coefficients
        )
        prism.reset_recon()
        torch.testing.assert_close(
            probe_model._beam_coefficients.detach(),
            probe_model._initial_beam_coefficients,
            rtol=0,
            atol=0,
        )
        assert probe_model.aberration_coefs[0]["C10"].item() == pytest.approx(C10 - 10)


def _thick_object(num_slices: int, seed: int = 7) -> np.ndarray:
    """Thick multislice white-noise transmission stack for forward-error tests."""
    rng = np.random.default_rng(seed)
    phases = (rng.random((num_slices, N, N)).astype(np.float32) - 0.5) * (1.0 / num_slices)
    return np.exp(1j * phases).astype(np.complex64)


def _scheme_forward_error(ptycho_dataset, obj, dense_pred, **scheme_kwargs) -> float:
    """Relative intensity error of a PRISM scheme vs the dense-PRISM reference."""
    _, prism = _build_engines(ptycho_dataset, obj, slice_thicknesses=20.0, **scheme_kwargs)
    pred = _forward_intensities(prism, BATCH_INDICES)
    return ((pred - dense_pred).norm() / dense_pred.norm()).item()


class TestFourierInterpolation:
    """Fourier-interpolation ('fourier'), grid-Sibson, and classic ('nearest') PRISM
    schemes on the shared ProbePRISM machinery (abTEM C-PRISM PR #318)."""

    def test_fourier_weights_partition_of_unity(self):
        """Nearest-fill trigonometric interpolation of a constant is a constant:
        the weights sum to one over parents at every beamlet."""
        gpts = (N, N)
        sampling = tuple(1 / (RECIPROCAL_SAMPLING * N))
        extent = np.array(gpts) * np.array(sampling)
        wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
        cutoff = SEMIANGLE_CUTOFF + float(np.linalg.norm(RECIPROCAL_SAMPLING * wavelength * 1e3))

        beamlets = _prism_wave_vectors(cutoff, extent, wavelength)
        for f in (2, 3, 4):
            parents = _grid_prism_wave_vectors(cutoff, extent, wavelength, f)
            weights = fourier_beamlet_weights(parents, beamlets, gpts, sampling, f)
            beamlet_support = weights.sum(axis=0)  # (N, N)
            # partition of unity exactly on the beamlet pixels
            n_int = np.rint(beamlets * extent).astype(int)
            bp = beamlet_support[n_int[:, 0] % N, n_int[:, 1] % N]
            np.testing.assert_allclose(bp, 1.0, atol=1e-6)

    def test_fourier_f1_matches_dense_forward(self, ptycho_dataset, complex_obj):
        """At interpolation factor 1 the coarse grid is the dense grid, so the
        Fourier scheme reproduces the dense-PRISM forward exactly."""
        obj = complex_obj[None]
        _, dense = _build_engines(ptycho_dataset, obj, dense=True)
        _, fourier = _build_engines(
            ptycho_dataset,
            obj,
            parent_layout="grid",
            interpolation="fourier",
            interpolation_factor=1,
        )
        pred_dense = _forward_intensities(dense, BATCH_INDICES)
        pred_fourier = _forward_intensities(fourier, BATCH_INDICES)
        scale = pred_dense.abs().max()
        torch.testing.assert_close(pred_fourier / scale, pred_dense / scale, rtol=1e-4, atol=1e-4)

    def test_nearest_f1_matches_dense_forward(self, ptycho_dataset, complex_obj):
        """Classic PRISM at factor 1 (full aperture, full window) is the identity."""
        obj = complex_obj[None]
        _, dense = _build_engines(ptycho_dataset, obj, dense=True)
        _, nearest = _build_engines(
            ptycho_dataset,
            obj,
            parent_layout="grid",
            interpolation="nearest",
            interpolation_factor=1,
        )
        pred_dense = _forward_intensities(dense, BATCH_INDICES)
        pred_nearest = _forward_intensities(nearest, BATCH_INDICES)
        scale = pred_dense.abs().max()
        torch.testing.assert_close(pred_nearest / scale, pred_dense / scale, rtol=1e-4, atol=1e-4)

    def test_fourier_error_decreases_with_parents(self, ptycho_dataset):
        """More coarse beams (smaller factor) -> smaller forward error vs dense."""
        obj = _thick_object(4)
        _, dense = _build_engines(ptycho_dataset, obj, slice_thicknesses=20.0, dense=True)
        dense_pred = _forward_intensities(dense, BATCH_INDICES)
        errs = [
            _scheme_forward_error(
                ptycho_dataset,
                obj,
                dense_pred,
                parent_layout="grid",
                interpolation="fourier",
                interpolation_factor=f,
            )
            for f in (4, 3, 2)
        ]
        assert errs[0] > errs[1] > errs[2]
        assert errs[-1] < 1e-2  # full-aperture interpolation is accurate

    def test_full_aperture_beats_nearest(self, ptycho_dataset):
        """The full-aperture interpolants reduce from the whole aperture and avoid the
        replica-overlap error of the classic crop; at a ptychographic ROI they are
        an order of magnitude more accurate than 'nearest' at the same factor."""
        obj = _thick_object(4)
        _, dense = _build_engines(ptycho_dataset, obj, slice_thicknesses=20.0, dense=True)
        dense_pred = _forward_intensities(dense, BATCH_INDICES)

        fourier = _scheme_forward_error(
            ptycho_dataset,
            obj,
            dense_pred,
            parent_layout="grid",
            interpolation="fourier",
            interpolation_factor=2,
        )
        sibson = _scheme_forward_error(
            ptycho_dataset,
            obj,
            dense_pred,
            parent_layout="grid",
            interpolation="sibson",
            interpolation_factor=2,
        )
        nearest = _scheme_forward_error(
            ptycho_dataset,
            obj,
            dense_pred,
            parent_layout="grid",
            interpolation="nearest",
            interpolation_factor=2,
        )
        assert fourier < 0.02 and sibson < 0.02
        assert nearest > 0.05
        assert fourier < 0.1 * nearest and sibson < 0.1 * nearest

    def test_coefficient_window_only_for_nearest(self):
        """Only the classic scheme carries a real-space crop window."""
        cases = [
            (dict(parent_layout="rings", interpolation="sibson", interpolation_factor=8), False),
            (dict(parent_layout="grid", interpolation="fourier", interpolation_factor=2), False),
            (dict(parent_layout="grid", interpolation="sibson", interpolation_factor=2), False),
            (dict(parent_layout="grid", interpolation="nearest", interpolation_factor=2), True),
        ]
        for kwargs, has_window in cases:
            probe = ProbePRISM.from_params(probe_params=PROBE_PARAMS, **kwargs)
            probe.set_initial_probe((N, N), RECIPROCAL_SAMPLING, MEAN_DIFFRACTION_INTENSITY)
            window = probe.coefficient_window
            if has_window:
                assert window is not None and tuple(window.shape) == (N, N)
                assert 0 < float(window.sum()) < N * N  # a proper crop
            else:
                assert window is None

    def test_invalid_scheme_combinations_raise(self):
        with pytest.raises(ValueError, match="requires parent_layout='grid'"):
            ProbePRISM.from_params(
                probe_params=PROBE_PARAMS, parent_layout="rings", interpolation="fourier"
            )
        with pytest.raises(ValueError, match="interpolation must be"):
            ProbePRISM.from_params(probe_params=PROBE_PARAMS, interpolation="spline")
        with pytest.raises(ValueError, match="parent_layout must be"):
            ProbePRISM.from_params(probe_params=PROBE_PARAMS, parent_layout="hex")
        with pytest.raises(ValueError, match="interpolation_factor must be >= 1"):
            ProbePRISM.from_params(probe_params=PROBE_PARAMS, interpolation_factor=0)

    def test_fourier_weights_reject_off_lattice_parents(self):
        """Parents whose reciprocal-grid indices are not divisible by the factor are
        not on the coarse sublattice."""
        gpts = (N, N)
        sampling = tuple(1 / (RECIPROCAL_SAMPLING * N))
        extent = np.array(gpts) * np.array(sampling)
        wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
        cutoff = SEMIANGLE_CUTOFF
        beamlets = _prism_wave_vectors(cutoff, extent, wavelength)
        # dense beamlets are on the f=1 lattice, hence not divisible by f=2
        with pytest.raises(ValueError, match="coarse sublattice"):
            fourier_beamlet_weights(beamlets, beamlets, gpts, sampling, 2)

    def test_mixed_state_grid_fourier_trains(self, ptycho_dataset):
        """Learnable per-parent beam coefficients on the Fourier scheme optimize
        without diverging (PPLR groups are scheme-independent)."""
        obj = np.exp(0.1j * np.random.default_rng(0).random((N, N))).astype(np.complex64)[None]
        _, prism = _build_engines(
            ptycho_dataset,
            obj,
            num_probes=2,
            parent_layout="grid",
            interpolation="fourier",
            interpolation_factor=2,
            learn_beam_coefficients=True,
            learn_aberrations=False,
        )
        prism.reconstruct(
            num_iters=4,
            reset=True,
            batch_size=512,
            optimizer_params={
                "object": {"name": "adam", "lr": 5e-3},
                "probe": {"beam_coefficients": {"name": "adam", "lr": 1e-2}},
            },
        )
        losses = np.asarray(prism._iter_losses)
        assert np.all(np.isfinite(losses))
        assert losses[-1] <= losses[0] * 1.5  # not diverging
