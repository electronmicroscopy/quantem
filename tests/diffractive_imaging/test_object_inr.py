"""
Tests for the implicit (INR) object model, ``ObjectINR``.

Covers the object model in isolation (forward/obj shapes, vacuum init, off-object
masking, multislice z-coordinates, gradient flow), the dataset's implicit-object
coordinate production, and an end-to-end reconstruction on a smooth synthetic object.

The reconstruction fixtures deliberately differ from ``test_ptychography.py``: that
fixture scans a torus-wrapped object (edge positions, zero padding), which the
pixelated path reproduces via ``% obj_shape`` indexing but the INR (vacuum outside the
object) cannot. Here the ground-truth object is larger than the scanned region and the
scan is confined to the interior with padding >= roi // 2, so patches never reach the
object boundary -- the physically realistic, non-toroidal regime both models agree on.
"""

import numpy as np
import pytest
import torch

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml import OptimizerParams, SchedulerParams
from quantem.core.ml.cnn import CNN2d
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectINR, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeDIP, ProbeParametric, ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography

if config.NUM_DEVICES > 0:
    config.set_device("gpu")

N = 40  # detector / roi size (px)
OGT = 64  # ground-truth object size (px); larger than the scanned region
PAD = 20  # obj padding (>= roi // 2 so interior patches never hit the boundary)
Q_MAX = 0.5  # inverse Angstroms
Q_PROBE = Q_MAX / 2
PROBE_ENERGY = 300e3  # eV
C10 = 50.0  # defocus (Angstrom)
STEP = 2  # scan step (px)
SCAN_START = 20  # first scan position (px); SCAN_START - roi//2 >= 0
SCAN_STOP = 44  # exclusive; SCAN_STOP - 1 + roi//2 - 1 < OGT  -> no wrap

# Reconstruction config (validated against the fixture below: pixelated reaches corr~0.89,
# the INR reaches corr~0.91 at omega_0=5 / lr=1e-2). omega_0=5 suits this smooth object;
# the default of 10 (and the image-fitting default of 30) stall here.
_INR_OMEGA = 5.0
_RECON_LR = 1e-2
_LOSS_RATIO = 0.3
_CORR_THRESHOLD = 0.7


def _smooth_phase() -> np.ndarray:
    """A smooth, band-limited phase object (friendly to a SIREN, unlike white noise)."""
    yy, xx = np.meshgrid(np.arange(OGT), np.arange(OGT), indexing="ij")
    return (0.7 * np.sin(2 * np.pi * xx / OGT * 4) * np.cos(2 * np.pi * yy / OGT * 3)).astype(
        np.float32
    )


def _probe_array() -> np.ndarray:
    sampling = 1 / Q_MAX / 2
    reciprocal_sampling = 2 * Q_MAX / N
    qx = qy = np.fft.fftfreq(N, sampling)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)
    aperture = np.sqrt(np.clip((Q_PROBE - q) / reciprocal_sampling + 0.5, 0, 1))
    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * C10
    probe_fourier = aperture * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return (np.fft.ifft2(probe_fourier) * N).astype(np.complex64)


def _semiangle_mrad() -> float:
    return electron_wavelength_angstrom(PROBE_ENERGY) * Q_PROBE * 1e3


def _build_synthetic_dataset() -> tuple[PtychographyDatasetRaster, np.ndarray, np.ndarray]:
    """Simulate a non-toroidal 4D-STEM dataset; return (dataset_model, gt_phase, probe)."""
    phase = _smooth_phase()
    complex_obj = np.exp(1j * phase)
    probe = _probe_array()
    reciprocal_sampling = 2 * Q_MAX / N

    gpos = np.arange(SCAN_START, SCAN_STOP, STEP)
    xx, yy = np.meshgrid(gpos, gpos, indexing="ij")
    positions = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    x0 = positions[:, 0].astype(int)
    y0 = positions[:, 1].astype(int)
    x_ind = np.fft.fftfreq(N, d=1 / N).astype(int)
    y_ind = np.fft.fftfreq(N, d=1 / N).astype(int)
    row = x0[:, None, None] + x_ind[None, :, None]  # no modulo: interior scan, no wrap
    col = y0[:, None, None] + y_ind[None, None, :]
    assert row.min() >= 0 and row.max() < OGT and col.min() >= 0 and col.max() < OGT
    exit_waves = complex_obj[row, col] * probe
    intensities = np.abs(np.fft.fft2(exit_waves)) ** 2

    sxy = len(gpos)
    dset = Dataset4dstem.from_array(
        array=np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((sxy, sxy, N, N)),
        sampling=(STEP, STEP, reciprocal_sampling, reciprocal_sampling),
        units=("A", "A", "A^-1", "A^-1"),
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.learn_scan_positions = False
    pdset.learn_descan = False
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=PROBE_ENERGY,
    )
    return pdset, phase, probe


def _build_inr_ptycho(
    num_slices: int = 1, slice_thicknesses=None, first_omega_0: float = _INR_OMEGA
) -> tuple[Ptychography, np.ndarray]:
    """Build an ObjectINR ptychography with the exact (frozen) simulation probe."""
    pdset, gt_phase, probe = _build_synthetic_dataset()
    obj = ObjectINR.from_uniform(
        num_slices=num_slices,
        slice_thicknesses=slice_thicknesses,
        obj_type="pure_phase",
        hidden_features=128,
        first_omega_0=first_omega_0,
        hidden_omega_0=first_omega_0,
        rng=0,
    )
    probe_model = ProbePixelated.from_array(
        num_probes=1,
        probe_params={"energy": PROBE_ENERGY, "C10": C10, "semiangle_cutoff": _semiangle_mrad()},
        probe_array=probe.copy(),
    )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj,
        probe_model=probe_model,
        detector_model=DetectorPixelated(),
        rng=0,
        verbose=False,
    )
    ptycho.preprocess(obj_padding_px=(PAD, PAD))
    return ptycho, gt_phase


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum() + 1e-12))


def _center_crop(a: np.ndarray, s: int) -> np.ndarray:
    r0 = (a.shape[0] - s) // 2
    c0 = (a.shape[1] - s) // 2
    return a[r0 : r0 + s, c0 : c0 + s]


def _best_corr(recon: np.ndarray, gt: np.ndarray, s: int = 32) -> float:
    """Best |correlation| over small integer shifts (handles ~1px registration offset)."""
    g = _center_crop(gt, s)
    best = -1.0
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r = _center_crop(np.roll(recon, (dr, dc), (0, 1)), s)
            best = max(best, abs(_corr(r, g)))
    return best


# --------------------------------------------------------------------------- #
# ObjectINR in isolation
# --------------------------------------------------------------------------- #
class TestObjectINRUnit:
    def test_forward_shape_dtype_and_vacuum(self):
        obj = ObjectINR.from_uniform(num_slices=1, hidden_features=64, rng=0)
        obj._initialize_obj((1, 32, 40))
        coords = torch.rand(5, 8, 8, 2) * 2 - 1
        patches = obj.forward(coords)
        assert patches.shape == (1, 5, 8, 8)
        assert patches.is_complex()
        # vacuum init (zeroed final layer) -> unit transmission everywhere
        assert torch.allclose(patches, torch.ones_like(patches), atol=1e-6)

    def test_off_object_is_vacuum(self):
        obj = ObjectINR.from_uniform(num_slices=1, hidden_features=64, rng=0)
        obj._initialize_obj((1, 16, 16))
        # train the final layer a little so the INR is not identically zero
        opt = torch.optim.Adam(obj.model.parameters(), lr=1e-2)
        inside = torch.rand(3, 4, 4, 2) * 2 - 1
        for _ in range(5):
            opt.zero_grad()
            obj.forward(inside).imag.sum().backward()
            opt.step()
        # coordinates outside [-1, 1] must map to identity transmission regardless of weights
        off = torch.full((1, 2, 2, 2), 5.0)
        out = obj.forward(off)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_multislice_z_coordinates(self):
        obj = ObjectINR.from_uniform(
            num_slices=3, slice_thicknesses=2.0, hidden_features=32, rng=0
        )
        obj._initialize_obj((3, 16, 16))
        z = obj._z_coords
        assert z.shape == (3,)
        # equally spaced slices span [-1, 1]
        assert torch.allclose(z, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)
        patches = obj.forward(torch.rand(4, 6, 6, 2) * 2 - 1)
        assert patches.shape == (3, 4, 6, 6)

    def test_single_slice_z_is_zero(self):
        obj = ObjectINR.from_uniform(num_slices=1, hidden_features=32, rng=0)
        obj._initialize_obj((1, 8, 8))
        assert torch.allclose(obj._z_coords, torch.zeros(1))

    def test_obj_materialization(self):
        obj = ObjectINR.from_uniform(
            num_slices=2, slice_thicknesses=1.0, hidden_features=32, rng=0
        )
        obj._initialize_obj((2, 20, 24))
        materialized = obj.obj
        assert materialized.shape == (2, 20, 24)
        assert not materialized.is_complex()  # pure_phase -> real phase array
        # vacuum init -> phase 0
        assert float(materialized.abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_gradients_flow_and_fit_smooth_phase(self):
        """The INR should be able to fit a smooth target phase via autograd."""
        obj = ObjectINR.from_uniform(num_slices=1, hidden_features=64, rng=1)
        obj._initialize_obj((1, 24, 24))
        opt = torch.optim.Adam(obj.model.parameters(), lr=1e-3)
        ys = torch.linspace(-1, 1, 24)
        xs = torch.linspace(-1, 1, 24)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        target = 0.5 * torch.sin(3 * gy) * torch.cos(2 * gx)
        coords = torch.stack([gy, gx], dim=-1)[None]  # (1, 24, 24, 2)

        losses = []
        for it in range(120):
            opt.zero_grad()
            pred_phase = obj.forward(coords)[0, 0].angle()
            loss = ((pred_phase - target) ** 2).mean()
            loss.backward()
            if it == 0:
                # zero-init final layer: after the first step only the final layer has a
                # gradient. parameters() order is [first-layer weight, bias, ..., final weight,
                # final bias], so params[0] is the first-layer weight and params[-2] the final.
                grads = [p.grad for p in obj.model.parameters()]
                assert grads[0] is not None and float(grads[0].abs().sum()) == 0.0
                assert grads[-2] is not None and float(grads[-2].abs().sum()) > 0.0
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.05 * losses[0]

    def test_autograd_only_backward_raises(self):
        obj = ObjectINR.from_uniform(num_slices=1, hidden_features=32, rng=0)
        with pytest.raises(NotImplementedError):
            obj.backward()

    def test_complex_obj_type_not_supported(self):
        with pytest.raises(NotImplementedError):
            ObjectINR.from_uniform(num_slices=1, obj_type="complex")

    def test_potential_obj_type_softplus_opt_in(self):
        """`potential` is real-valued; the default activation is now identity, but passing
        ``final_activation="softplus"`` opts back into output-activation positivity (min >= 0)."""
        obj = ObjectINR.from_uniform(
            num_slices=1,
            obj_type="potential",
            final_activation="softplus",
            hidden_features=32,
            rng=0,
        )
        obj._initialize_obj((1, 16, 16))
        assert obj.obj_type == "potential"
        assert not obj.dtype.is_complex  # real-valued object
        patches = obj.forward(torch.rand(3, 4, 4, 2) * 2 - 1)
        assert patches.shape == (1, 3, 4, 4) and patches.is_complex()
        # train a few steps so the potential is non-uniform, then check positivity holds
        opt = torch.optim.Adam(obj.model.parameters(), lr=1e-2)
        coords = torch.rand(2, 6, 6, 2) * 2 - 1
        for _ in range(5):
            opt.zero_grad()
            obj.forward(coords).imag.sum().backward()
            opt.step()
        materialized = obj.obj
        assert materialized.shape == (1, 16, 16) and not materialized.is_complex()
        assert float(materialized.min()) >= 0.0  # softplus enforces non-negative potential

    def test_potential_identity_default_and_positivity_penalty(self):
        """Default ``potential`` activation is identity (vacuum is exactly 0); the soft
        ``positivity_weight`` penalty drives a forced-negative potential non-negative."""
        torch.manual_seed(0)  # rng=0 only seeds coordinate sampling; HSiren init uses global RNG
        obj = ObjectINR.from_uniform(num_slices=1, obj_type="potential", hidden_features=32, rng=0)
        obj._initialize_obj((1, 24, 24))
        # identity + zeroed final layer -> vacuum is exactly 0 (not softplus(0) = ln 2)
        assert float(obj._materialize_obj().abs().max()) == pytest.approx(0.0, abs=1e-6)
        # force the whole potential negative via the (zero-weight) final-layer bias
        with torch.no_grad():
            obj.model.net[-2].bias.fill_(-0.5)  # type:ignore
        assert float(obj._materialize_obj().min()) == pytest.approx(-0.5, abs=1e-3)
        obj.constraints = {"positivity_weight": 1.0}
        assert float(obj._sampled_positivity_loss(1.0)) == pytest.approx(0.5, abs=0.05)
        opt = torch.optim.Adam(obj.model.parameters(), lr=1e-2)
        for _ in range(80):
            opt.zero_grad()
            obj.apply_soft_constraints().backward()
            opt.step()
        assert float(obj._materialize_obj().min()) > -1e-2  # driven non-negative

    def test_fix_potential_baseline_gauge(self):
        """``fix_potential_baseline`` subtracts the background offset from the materialized
        potential (display gauge; the reconstruction forward path is unaffected)."""
        obj = ObjectINR.from_uniform(num_slices=1, obj_type="potential", hidden_features=32, rng=0)
        obj._initialize_obj((1, 16, 16))
        with torch.no_grad():
            obj.model.net[-2].bias.fill_(1.0)  # type:ignore  # constant +1 background
        raw = obj._materialize_obj()
        assert float(raw.min()) == pytest.approx(1.0, abs=0.2)
        obj.constraints = {"fix_potential_baseline": True}
        disp = obj.apply_hard_constraints(raw, mask=obj.mask)
        assert float(disp.min()) == pytest.approx(0.0, abs=1e-3)  # background pinned to 0
        assert float(disp.min()) >= 0.0  # clamped non-negative

    def test_from_pixelated_with_model(self):
        """from_pixelated can wrap a directly-passed INR model (like ObjectDIP.from_pixelated)."""
        from quantem.core.ml.inr import HSiren

        h = w = 16
        pix = ObjectPixelated.from_array(initial_obj=torch.zeros(1, h, w), obj_type="pure_phase")
        pix._initialize_obj((1, h, w), sampling=(1.0, 1.0))
        my_model = HSiren(in_features=3, out_features=1, hidden_features=16, hidden_layers=2)
        inr = ObjectINR.from_pixelated(pix, model=my_model)
        assert inr.model is my_model
        assert tuple(inr.pretrain_target.shape) == (1, h, w)

    def test_from_pixelated_pretrain(self):
        """from_pixelated + pretrain warm-starts the INR to reproduce a pixelated object."""
        h = w = 32
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        phase = (0.5 * torch.sin(3 * gy) * torch.cos(2 * gx)).float()[None]  # (1, h, w)
        pix = ObjectPixelated.from_array(initial_obj=phase, obj_type="pure_phase")
        pix._initialize_obj((1, h, w), sampling=(1.0, 1.0))

        inr = ObjectINR.from_pixelated(
            pix, hidden_features=128, first_omega_0=5.0, hidden_omega_0=5.0
        )
        assert inr.num_slices == pix.num_slices
        assert tuple(inr.pretrain_target.shape) == tuple(pix.obj.shape)

        inr.pretrain(
            num_iters=150,
            optimizer_params=OptimizerParams.Adam(lr=1e-3),
            scheduler_params=SchedulerParams.Plateau(factor=0.5),
            show=False,
        )
        losses = inr.pretrain_losses
        assert losses[-1] < 0.1 * losses[0]

        gt = pix.obj[0].detach().cpu().numpy()
        gt -= gt.mean()

        def _corr_to_pix(arr: np.ndarray) -> float:
            a = arr - arr.mean()
            return float((a * gt).sum() / np.sqrt((a**2).sum() * (gt**2).sum() + 1e-12))

        assert _corr_to_pix(inr.obj[0].detach().cpu().numpy()) > 0.95
        # the pretrained weights are the reset state (reconstruct(reset=True) resumes from them)
        inr.reset()
        assert _corr_to_pix(inr.obj[0].detach().cpu().numpy()) > 0.95


# --------------------------------------------------------------------------- #
# Dataset implicit-object coordinate production
# --------------------------------------------------------------------------- #
class TestImplicitDatasetCoords:
    def test_scan_coords_normalization(self):
        """An integer scan position maps to linspace(-1,1,N) grid nodes; spacing = 1 px."""
        pdset, _, _ = _build_synthetic_dataset()
        pdset.implicit_object = True
        padding = (PAD, PAD)
        h_full, w_full = pdset._obj_shape_full_2d(padding)
        # place a clean integer position at the object center
        r, c = int(h_full) // 2, int(w_full) // 2
        pdset.scan_positions_px.data[0] = torch.tensor(
            [float(r), float(c)], device=pdset.scan_positions_px.device
        )
        coords = pdset._scan_coords(torch.tensor([0]), padding)[0].detach()  # (Hroi, Wroi, 2)

        # fftfreq offset 0 is the first ROI pixel -> coordinate of pixel (r, c)
        assert float(coords[0, 0, 0]) == pytest.approx(r / (int(h_full) - 1) * 2 - 1, abs=1e-5)
        assert float(coords[0, 0, 1]) == pytest.approx(c / (int(w_full) - 1) * 2 - 1, abs=1e-5)
        # adjacent rows/cols differ by exactly one normalized pixel
        d_row = float(coords[1, 0, 0] - coords[0, 0, 0])
        d_col = float(coords[0, 1, 1] - coords[0, 0, 1])
        assert d_row == pytest.approx(2 / (int(h_full) - 1), abs=1e-5)
        assert d_col == pytest.approx(2 / (int(w_full) - 1), abs=1e-5)

    def test_forward_returns_coords_and_zero_fractional_when_implicit(self):
        pdset, _, _ = _build_synthetic_dataset()
        pdset.implicit_object = True
        batch = torch.arange(4)
        patch_data, positions_px, fractional, _descan = pdset.forward(batch, (PAD, PAD))
        # implicit: patch_data are float coords (batch, Hroi, Wroi, 2), not integer indices
        assert patch_data.shape == (4, N, N, 2)
        assert patch_data.dtype.is_floating_point
        # the probe must not be subpixel-shifted -> fractional is zero
        assert torch.allclose(fractional, torch.zeros_like(fractional))

    def test_implicit_flag_synced_from_obj_model(self):
        ptycho, _ = _build_inr_ptycho()
        assert ptycho.obj_model.is_implicit is True
        assert ptycho.dset.implicit_object is True


# --------------------------------------------------------------------------- #
# End-to-end reconstruction
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestObjectINRReconstruction:
    def test_loss_decreases_and_recovers_object(self):
        ptycho, gt_phase = _build_inr_ptycho()
        ptycho.reconstruct(
            num_iters=150,
            optimizer_params={"object": {"name": "adam", "lr": _RECON_LR}},  # probe frozen
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert losses[-1] < _LOSS_RATIO * losses[0]
        assert _best_corr(ptycho.obj[0], gt_phase) > _CORR_THRESHOLD

    def test_multislice_runs(self):
        ptycho, _ = _build_inr_ptycho(num_slices=2, slice_thicknesses=20.0)
        ptycho.reconstruct(
            num_iters=5,
            optimizer_params={"object": {"name": "adam", "lr": _RECON_LR}},
            batch_size=200,
        )
        assert ptycho.obj.shape[0] == 2

    def test_data_loss_criteria_run(self):
        """The pluggable data-fidelity criteria run end-to-end and stay finite."""
        from quantem.diffractive_imaging.ptycho_losses import AmplitudeS3IM

        ptycho, _ = _build_inr_ptycho()
        for loss_type in ["l1_amplitude", "smooth_l1_amplitude", AmplitudeS3IM(repeats=3)]:
            ptycho.reconstruct(
                num_iters=5,
                reset=True,
                optimizer_params={"object": {"name": "adam", "lr": _RECON_LR}},
                batch_size=200,
                loss_type=loss_type,
            )
            losses = np.array(ptycho._iter_losses)
            assert np.isfinite(losses).all(), loss_type

    def test_save_load_roundtrip(self, tmp_path):
        ptycho, _ = _build_inr_ptycho()
        ptycho.reconstruct(
            num_iters=20,
            optimizer_params={"object": {"name": "adam", "lr": _RECON_LR}},
            batch_size=200,
        )
        obj_before = ptycho.obj.copy()
        path = tmp_path / "inr_ptycho.zip"
        ptycho.save(path, mode="o", save_raw_data=True)  # persist dset so loaded.dset works
        loaded = autoserialize_load(path)
        assert loaded.obj_model.is_implicit is True
        assert loaded.dset.implicit_object is True
        np.testing.assert_allclose(loaded.obj, obj_before, rtol=1e-5, atol=1e-6)
        # continued training still runs after reload
        loaded.reconstruct(
            num_iters=5,
            optimizer_params={"object": {"name": "adam", "lr": _RECON_LR}},
            batch_size=200,
        )


# --------------------------------------------------------------------------- #
# ObjectINR composes with every probe model (it is a drop-in object model)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestObjectINRProbeTypes:
    """ObjectINR is a drop-in object model: it composes with each probe representation.

    For all probe types the implicit object is queried at continuous coordinates (the dataset
    reports ``implicit_object=True``) and the probe receives a zero fractional shift, so the
    reconstruction runs end-to-end and the loss stays finite.
    """

    def _probe_params(self) -> dict:
        return {"energy": PROBE_ENERGY, "C10": C10, "semiangle_cutoff": _semiangle_mrad()}

    def test_inr_runs_with_each_probe_type(self):
        pdset, _gt, _probe = _build_synthetic_dataset()
        pp = self._probe_params()
        probe_builders = {
            "pixelated": lambda: ProbePixelated.from_params(probe_params=pp),
            "parametric": lambda: ProbeParametric.from_params(probe_params=pp),
            "dip": lambda: ProbeDIP.from_model(
                model=CNN2d(in_channels=1, dtype=torch.complex64, num_layers=3),
                roi_shape=(N, N),
                num_probes=1,
                probe_params=pp,
            ),
        }
        for name, build in probe_builders.items():
            obj = ObjectINR.from_uniform(
                num_slices=1,
                obj_type="pure_phase",
                hidden_features=128,
                first_omega_0=_INR_OMEGA,
                hidden_omega_0=_INR_OMEGA,
                rng=0,
            )
            ptycho = Ptychography.from_models(
                dset=pdset,
                obj_model=obj,
                probe_model=build(),
                detector_model=DetectorPixelated(),
                rng=0,
                verbose=False,
            )
            ptycho.preprocess(obj_padding_px=(PAD, PAD))
            assert ptycho.dset.implicit_object is True, name
            ptycho.reconstruct(
                num_iters=10,
                optimizer_params={
                    "object": {"name": "adam", "lr": _RECON_LR},
                    "probe": {"name": "adam", "lr": 1e-3},
                },
                batch_size=200,
            )
            losses = np.array(ptycho._iter_losses)
            assert np.isfinite(losses).all(), name
            assert ptycho.obj.shape[0] == 1, name


# --------------------------------------------------------------------------- #
# Data-fidelity criterion system (ptycho_losses)
# --------------------------------------------------------------------------- #
class TestDataCriteria:
    def test_registry_and_target_spaces(self):
        from quantem.diffractive_imaging.ptycho_losses import (
            L2,
            AmplitudeS3IM,
            get_data_criterion,
        )

        assert isinstance(get_data_criterion("l2_amplitude"), L2)
        assert get_data_criterion("l2_amplitude").target_space == "amplitude"
        assert get_data_criterion("l2_intensity").target_space == "intensity"
        assert get_data_criterion("poisson").target_space == "intensity"
        assert get_data_criterion("s3im_amplitude").target_space == "amplitude"
        # passing a DataCriterion instance returns it unchanged (tune params this way)
        crit = AmplitudeS3IM(lambda_s3im=0.5)
        assert get_data_criterion(crit) is crit
        with pytest.raises(ValueError):
            get_data_criterion("not_a_loss")

    def test_criterion_values(self):
        from quantem.diffractive_imaging.ptycho_losses import L1, L2

        preds = torch.tensor([[1.0, 2.0]])  # B = 1
        targets = torch.tensor([[1.5, 2.0]])
        # n == B -> global scale 1; matches the legacy sum-reduced amplitude losses
        assert torch.isclose(L2()(preds, targets, n=1), torch.tensor(0.25))
        assert torch.isclose(L1()(preds, targets, n=1), torch.tensor(0.5))
