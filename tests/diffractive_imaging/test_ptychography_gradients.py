"""Autograd and the analytic path compute the same gradient.

``reconstruct(autograd=False)`` differentiates the ``l2_amplitude`` loss by hand. With
``analytic_step_normalization=False`` it must return exactly what autograd returns, in every
configuration: object type, batch, slices, probe modes, sub-pixel positions and descan. With the
normalization on it takes the ePIE step, which for a full batch on a scan whose step is one
object pixel coincides with the gradient, so ``SGD(lr)`` then runs identically on both paths.

The white-noise object checks the reconstruction itself. Its spectrum is flat, so the ratio of
the reconstructed to the true phase spectrum should be one at every spatial frequency.

Every problem here is matched and periodic: a unit-step raster over an ``N x N`` object with no
padding, simulated with the same corner-centred patch convention the reconstruction uses.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging import object_models
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography

from .conftest import white_noise_object_2D

N = 32
SAMPLING = 0.25  # Angstrom, so k_max = 2 / Angstrom
K_PROBE = 1.0  # inverse Angstrom
ENERGY = 300e3
ABERRATIONS = {"C10": 50.0, "C12": 25.0, "phi12": np.deg2rad(11)}
TOLERANCE = 1e-5  # relative, float32


def _probe() -> np.ndarray:
    k = np.fft.fftfreq(N, SAMPLING)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    phi = np.arctan2(ky, kx)
    aperture = np.sqrt(np.clip((K_PROBE - np.sqrt(k2)) * N * SAMPLING + 0.5, 0, 1))
    c1 = ABERRATIONS["C10"] + ABERRATIONS["C12"] * np.cos(2 * (phi - ABERRATIONS["phi12"]))
    chi = np.pi * electron_wavelength_angstrom(ENERGY) * k2 * c1
    fourier = aperture * np.exp(-1j * chi)
    fourier /= np.linalg.norm(fourier)
    return np.fft.ifft2(fourier) * N


def _simulate(phase: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """``(N, N, N, N)`` intensities of a unit-step periodic raster, corner-centred."""
    offsets = np.fft.fftfreq(N, 1 / N).astype(int)
    scan = np.arange(N)
    rows = (scan[:, None, None, None] + offsets[None, None, :, None]) % N
    cols = (scan[None, :, None, None] + offsets[None, None, None, :]) % N
    patches = np.exp(1j * phase)[rows, cols]
    return np.abs(np.fft.fft2(patches * probe)) ** 2


def _build(
    phase: np.ndarray | None = None,
    obj_type: str = "complex",
    num_slices: int = 1,
    num_probes: int = 1,
) -> Ptychography:
    if phase is None:
        phase = white_noise_object_2D(N, phi0=1.0)
    probe = _probe()
    dset = Dataset4dstem.from_array(
        np.fft.fftshift(_simulate(phase, probe), axes=(-2, -1)).astype(np.float32),
        sampling=(SAMPLING, SAMPLING, 1 / (N * SAMPLING), 1 / (N * SAMPLING)),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset, verbose=0)
    pdset.preprocess(
        com_fit_function="no_shift",
        plot_rotation=False,
        plot_com=False,
        probe_energy=ENERGY,
        force_com_rotation=0,
        force_com_transpose=False,
    )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=ObjectPixelated.from_uniform(
            num_slices=num_slices, obj_type=obj_type, slice_thicknesses=10.0
        ),
        probe_model=ProbePixelated.from_array(
            probe_array=probe, num_probes=num_probes, probe_params={"energy": ENERGY}
        ),
        detector_model=DetectorPixelated(),
        device="cpu",
        verbose=0,
        rng=0,
    )
    ptycho.preprocess(obj_padding_px=(0, 0), plot_rotation=False, plot_com=False)
    # probe orthogonalization is differentiated through under autograd only
    ptycho.constraints = {"object": {"positivity": False}, "probe": {"orthogonalize_probe": False}}
    assert tuple(ptycho.obj_model._obj.shape[-2:]) == (N, N)
    return ptycho


def _perturb(ptycho: Ptychography, seed: int = 1) -> None:
    """Move the object off its uniform start, including a global phase for complex objects."""
    g = torch.Generator().manual_seed(seed)
    obj = ptycho.obj_model
    phase = 0.3 * torch.randn(obj._obj.shape, generator=g)
    with torch.no_grad():
        if obj.obj_type == "complex":
            amplitude = 0.9 + 0.1 * torch.rand(obj._obj.shape, generator=g)
            obj._obj.data = (amplitude * torch.exp(1j * (phase + 0.7))).to(obj._obj.dtype)
        else:
            obj._obj.data = phase.to(obj._obj.dtype)


def _gradients(ptycho, indices, autograd: bool, step_normalization: bool = False):
    """Object and probe gradients for one batch, through the calls the reconstruction makes."""
    ptycho.criterion = "l2_amplitude"
    ptycho.dset._set_targets("amplitude")
    ptycho.compute_propagator_arrays()
    obj, probe = ptycho.obj_model._obj, ptycho.probe_model._probe
    obj.grad = probe.grad = None

    indices = torch.as_tensor(np.asarray(indices))
    n = ptycho.dset.num_positions
    targets = ptycho.dset.targets[indices]
    patch_data, _, fractional, descan = ptycho.dset.forward(indices, ptycho.obj_padding_px)
    shifted = ptycho.probe_model.forward(fractional)
    obj_patches = ptycho.obj_model.forward(patch_data)
    propagated, overlap = ptycho.forward_operator(obj_patches, shifted, descan)
    loss = ptycho.error_estimate(ptycho.detector_model.forward(overlap), targets, global_n=n)
    ptycho.backward(
        loss,
        autograd,
        obj_patches,
        propagated,
        overlap,
        patch_data,
        targets,
        positions_px_fractional=fractional,
        descan_shifts=descan,
        global_n=n,
        step_normalization=step_normalization,
    )
    return obj.grad.detach().clone(), probe.grad.detach().clone()


def _relative_difference(a, b) -> float:
    a = torch.as_tensor(a).to(torch.complex128).flatten()
    b = torch.as_tensor(b).to(torch.complex128).flatten()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


CONFIGURATIONS = {
    "complex": {},
    "potential": {"obj_type": "potential"},
    "pure_phase": {"obj_type": "pure_phase"},
    "minibatch": {"batch": 100},
    "multislice": {"num_slices": 2},
    "pure_phase_multislice": {"obj_type": "pure_phase", "num_slices": 3},
    "mixed_state": {"num_probes": 2},
    "subpixel_positions": {"subpixel": True},
    "descan": {"descan": True},
}


class TestAnalyticGradientMatchesAutograd:
    @pytest.mark.parametrize("name", list(CONFIGURATIONS))
    def test_object_and_probe_gradients(self, name):
        config = dict(CONFIGURATIONS[name])
        batch = config.pop("batch", None)
        subpixel = config.pop("subpixel", False)
        descan = config.pop("descan", False)

        ptycho = _build(**config)
        if subpixel:
            g = torch.Generator().manual_seed(2)
            positions = ptycho.dset._scan_positions_px
            with torch.no_grad():
                positions.data += 0.4 * (torch.rand(positions.shape, generator=g) - 0.5)
        if descan:
            # descan is only applied in the forward model while the dataset is optimized
            ptycho.optimizer_params = {"dataset": {"type": "sgd", "lr": 0.0}}
            ptycho.set_optimizers()
            with torch.no_grad():
                ptycho.dset._descan_shifts.data += torch.tensor([0.37, -0.21])
        _perturb(ptycho)

        n = ptycho.dset.num_positions
        indices = np.arange(n)
        if batch is not None:
            indices = np.sort(np.random.default_rng(3).choice(n, batch, replace=False))

        obj_ad, probe_ad = _gradients(ptycho, indices, autograd=True)
        obj_an, probe_an = _gradients(ptycho, indices, autograd=False)
        assert _relative_difference(obj_an, obj_ad) < TOLERANCE
        assert _relative_difference(probe_an, probe_ad) < TOLERANCE


class TestAnalyticStepNormalization:
    @pytest.mark.parametrize("batch", [None, 100])
    def test_normalized_step_is_the_epie_step(self, batch):
        """Checked against a numpy ePIE step, written without any of quantem's helpers."""
        ptycho = _build()
        _perturb(ptycho)
        n = ptycho.dset.num_positions
        indices = np.arange(n)
        if batch is not None:
            indices = np.sort(np.random.default_rng(3).choice(n, batch, replace=False))
        obj_step, probe_step = _gradients(ptycho, indices, autograd=False, step_normalization=True)

        # in float64, so the comparison is limited by quantem's float32 alone
        visible = ptycho.obj_model.obj[0].detach().numpy().ravel().astype(np.complex128)
        probe = ptycho.probe_model.probe[0].detach().numpy().astype(np.complex128)
        amplitudes = np.fft.ifftshift(
            ptycho.dset.centered_amplitudes.numpy()[indices].astype(np.float64), axes=(-2, -1)
        )
        flat = ptycho.dset.patch_indices.numpy()[indices].astype(int)

        obj_patches = visible[flat]
        exit_waves = probe * obj_patches
        fourier = np.fft.fft2(exit_waves, norm="ortho")
        projected = np.fft.ifft2(amplitudes * np.exp(1j * np.angle(fourier)), norm="ortho")
        difference = exit_waves - projected  # minus the ePIE correction

        obj_sum = np.zeros(N * N, dtype=complex)
        np.add.at(obj_sum, flat.ravel(), (np.conj(probe) * difference).ravel())
        probe_overlap = np.zeros(N * N)
        np.add.at(
            probe_overlap, flat.ravel(), np.broadcast_to(np.abs(probe) ** 2, flat.shape).ravel()
        )
        # the stored parameter differs from what the forward sees by the gauge rotation
        phasor = ptycho.obj_model.gauge_phasor(ptycho.obj_model._obj).item()
        expected_obj = obj_sum / probe_overlap.max() * phasor

        expected_probe = (np.conj(obj_patches) * difference).sum(0)
        expected_probe /= (np.abs(obj_patches) ** 2).sum(0).max()

        # the probe step sums the whole batch with heavy cancellation, and quantem does it in
        # float32 in a different order, so it sits nearer the float32 floor than the object
        assert _relative_difference(obj_step[0], expected_obj) < TOLERANCE
        assert _relative_difference(probe_step[0], expected_probe) < 10 * TOLERANCE

    def test_sgd_runs_identically_on_both_paths(self):
        """Full batch, unit-step scan: the ePIE step equals the gradient, so SGD(lr) agrees."""
        losses, objects = {}, {}
        for autograd in (True, False):
            ptycho = _build()
            ptycho.reconstruct(
                num_iters=10,
                reset=True,
                optimizer_params={"object": {"type": "sgd", "lr": 0.5}},
                batch_size=N * N,
                autograd=autograd,
                device="cpu",
            )
            losses[autograd] = np.asarray(ptycho._iter_losses)
            objects[autograd] = ptycho.obj_model.obj.detach()
        np.testing.assert_allclose(losses[True], losses[False], rtol=1e-4)
        assert _relative_difference(objects[False], objects[True]) < TOLERANCE


class TestGauge:
    @pytest.mark.parametrize("autograd", [True, False])
    @pytest.mark.parametrize("angle", [0.3, 2.0])
    def test_complex_gradient_is_returned_in_the_parameter_frame(self, autograd, angle):
        """Rotating the stored object by a global phase must leave the forward unchanged and
        rotate its gradient by the same phase.

        A straight-through gauge fix hands back the unrotated gradient, turning every update by
        the gauge angle (caught at 0.3 rad). Recentering on the mean of the wrapped angles also
        stops being a pure rotation once pixels cross the branch cut (caught at 2 rad, where the
        perturbed phases, centred on 0.7, pass pi).
        """
        ptycho = _build()
        _perturb(ptycho)
        indices = np.arange(ptycho.dset.num_positions)
        grad, _ = _gradients(ptycho, indices, autograd=autograd)
        visible = ptycho.obj_model.obj.detach().clone()

        rotation = np.exp(1j * angle)
        with torch.no_grad():
            ptycho.obj_model._obj.data *= rotation
        grad_rotated, _ = _gradients(ptycho, indices, autograd=autograd)

        assert _relative_difference(ptycho.obj_model.obj.detach(), visible) < TOLERANCE
        assert _relative_difference(grad_rotated, grad * rotation) < TOLERANCE

    def test_pure_phase_gauge_does_not_change_the_reconstruction(self, monkeypatch):
        """Subtracting the mean of a real phase is a global phase: the loss cannot see it, the
        gradient sums to zero, and nothing about the reconstruction may change."""

        def run():
            ptycho = _build(obj_type="pure_phase")
            ptycho.reconstruct(
                num_iters=10,
                reset=True,
                optimizer_params={"object": {"type": "sgd", "lr": 0.5}},
                batch_size=128,
                device="cpu",
            )
            return ptycho

        gauged = run()
        monkeypatch.setattr(
            object_models.ObjectConstraints, "_apply_hard_pure_phase", lambda self, obj, c: obj
        )
        ungauged = run()

        np.testing.assert_allclose(gauged._iter_losses, ungauged._iter_losses, rtol=1e-5)
        raw = gauged.obj_model._obj.detach()
        assert abs(float(raw.mean())) < 1e-5 * float(raw.std())
        phase_gauged = gauged.obj_model.obj.detach()
        phase_ungauged = ungauged.obj_model.obj.detach()
        assert _relative_difference(phase_gauged, phase_ungauged - phase_ungauged.mean()) < 1e-4


def _transfer_band_mean(reconstructed_phase: np.ndarray, true_phase: np.ndarray) -> complex:
    """Mean of ``FFT(rec) conj(FFT(true)) / |FFT(true)|^2`` over ``0 < q <= 2 k_probe``."""
    rec = np.fft.fft2(reconstructed_phase - reconstructed_phase.mean())
    true = np.fft.fft2(true_phase - true_phase.mean())
    power = np.abs(true) ** 2
    k = np.fft.fftfreq(N, SAMPLING)
    q = np.hypot(k[:, None], k[None, :])
    band = (power > 1e-6 * power.max()) & (q > 0) & (q <= 2 * K_PROBE)
    return complex(np.mean(rec[band] * np.conj(true[band]) / power[band]))


class TestWhiteNoiseTransfer:
    """A weak white-noise object has a flat spectrum, so a correct reconstruction transfers
    every spatial frequency with unit gain and no phase shift."""

    @pytest.mark.parametrize(
        "autograd, optimizer",
        [
            (False, {"type": "sgd", "lr": 0.5}),
            (True, {"type": "sgd", "lr": 0.5}),
            (True, {"type": "adam", "lr": 5e-3}),
        ],
    )
    def test_unit_transfer(self, autograd, optimizer):
        true_phase = white_noise_object_2D(N, phi0=1.0)
        ptycho = _build(phase=true_phase)
        ptycho.reconstruct(
            num_iters=30,
            reset=True,
            optimizer_params={"object": optimizer},
            batch_size=64,
            autograd=autograd,
            device="cpu",
        )
        obj = ptycho.obj_model.obj[0].detach().numpy()
        transfer = _transfer_band_mean(np.angle(obj), true_phase)
        assert abs(transfer - 1) < 1e-2, f"band-averaged transfer {transfer:.4f}"
