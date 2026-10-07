"""
Tests for the tensor-decomposition (K-Planes) object model, ``ObjectTensorDecomp``.

``ObjectTensorDecomp`` subclasses ``ObjectINR`` and swaps the SIREN for a ``KPlanes``
network, so it reuses every coordinate-query path of the INR and only differs in the
optimizer wiring (per-parameter-group learning rates, PPLR). These tests cover the
shared-core ``KPlanes`` head fix, the object model in isolation (forward/vacuum/PPLR
parameter groups, bad-key rejection), the ptychography PPLR optimizer pass-through, and
an end-to-end reconstruction on the same non-toroidal synthetic object used by
``test_object_inr.py`` (fixtures duplicated here, matching that file's own duplication
from ``test_ptychography.py`` — ``--import-mode=importlib`` makes cross-test imports
unreliable).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml import OptimizerParams, SchedulerParams
from quantem.core.ml.models.kplanes import CPTilted, KPlanes
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import (
    ObjectModelType,
    ObjectPixelated,
    ObjectTensorDecomp,
)
from quantem.diffractive_imaging.probe_models import ProbePixelated
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

# K-Planes recon config (validated against the fixture below). grids learn faster than the
# decoder head; both run under a single cosine-annealing schedule.
_LR_GRIDS = 1e-2
_LR_SIGMA = 1e-3
_LOSS_RATIO = 0.3
_CORR_THRESHOLD = 0.6


def _smooth_phase() -> np.ndarray:
    """A smooth, band-limited phase object."""
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


def _build_kplanes_ptycho(
    num_slices: int = 1,
    slice_thicknesses=None,
    M_features: int = 24,
    resolution: tuple[int, int, int] = (16, 64, 64),
) -> tuple[Ptychography, np.ndarray]:
    """Build an ObjectTensorDecomp (K-Planes) ptychography with the exact (frozen) probe."""
    pdset, gt_phase, probe = _build_synthetic_dataset()
    obj = ObjectTensorDecomp.from_uniform(
        num_slices=num_slices,
        slice_thicknesses=slice_thicknesses,
        M_features=M_features,
        resolution=resolution,
        multiscale_res_multipliers=(0.25, 0.5, 1.0),
        use_hybrid_mlp=False,
        obj_type="pure_phase",
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


def _pplr_params() -> dict:
    return {
        "object": {
            "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
            "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
        }
    }


# --------------------------------------------------------------------------- #
# Shared-core KPlanes head fix (use_hybrid_mlp=False must build a decoder head)
# --------------------------------------------------------------------------- #
class TestKPlanesHead:
    def test_non_hybrid_builds_head_and_runs(self):
        m = KPlanes(
            M_features=8,
            resolution=(16, 16, 16),
            multiscale_res_multipliers=[0.5, 1.0],
            density_activation=nn.Identity(),
            use_hybrid_mlp=False,
        )
        assert isinstance(m.sigma_net, nn.Linear)
        out = m(torch.rand(20, 3) * 2 - 1)
        assert out.shape == (20, 1)
        assert set(m.get_params().keys()) == {"grids", "sigma_net"}

    def test_hybrid_still_builds_mlp_head(self):
        m = KPlanes(
            M_features=8,
            resolution=(16, 16, 16),
            multiscale_res_multipliers=[1.0],
            density_activation=nn.Identity(),
            use_hybrid_mlp=True,
            hybrid_hidden_dim=32,
            hybrid_num_layers=2,
        )
        assert isinstance(m.sigma_net, nn.Sequential)
        assert m(torch.rand(7, 3) * 2 - 1).shape == (7, 1)


# --------------------------------------------------------------------------- #
# ObjectTensorDecomp in isolation
# --------------------------------------------------------------------------- #
class TestObjectTensorDecompUnit:
    def _obj(self, **kw):
        return ObjectTensorDecomp.from_uniform(
            num_slices=kw.pop("num_slices", 1),
            slice_thicknesses=kw.pop("slice_thicknesses", None),
            M_features=8,
            resolution=kw.pop("resolution", (16, 32, 32)),
            multiscale_res_multipliers=(0.5, 1.0),
            rng=0,
            **kw,
        )

    def test_is_implicit_and_name(self):
        obj = self._obj()
        assert obj.is_implicit is True
        assert obj.name == "ObjectTensorDecomp"
        assert isinstance(obj, ObjectModelType)

    def test_forward_shape_dtype_and_vacuum(self):
        obj = self._obj()
        obj._initialize_obj((1, 24, 28))
        coords = torch.rand(5, 8, 8, 2) * 2 - 1
        patches = obj.forward(coords)
        assert patches.shape == (1, 5, 8, 8)
        assert patches.is_complex()
        # vacuum init (zeroed decoder head) -> unit transmission everywhere
        assert torch.allclose(patches, torch.ones_like(patches), atol=1e-6)

    def test_off_object_is_vacuum(self):
        obj = self._obj(resolution=(16, 16, 16))
        obj._initialize_obj((1, 16, 16))
        # train the head a little so the model is not identically zero
        opt = torch.optim.Adam(obj.model.parameters(), lr=1e-2)
        inside = torch.rand(3, 4, 4, 2) * 2 - 1
        for _ in range(5):
            opt.zero_grad()
            obj.forward(inside).imag.sum().backward()
            opt.step()
        off = torch.full((1, 2, 2, 2), 5.0)
        out = obj.forward(off)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_multislice_z_coordinates(self):
        obj = self._obj(num_slices=3, slice_thicknesses=2.0, resolution=(16, 16, 16))
        obj._initialize_obj((3, 16, 16))
        z = obj._z_coords
        assert z.shape == (3,)
        assert torch.allclose(z, torch.tensor([-1.0, 0.0, 1.0]).to(z), atol=1e-6)
        patches = obj.forward(torch.rand(4, 6, 6, 2) * 2 - 1)
        assert patches.shape == (3, 4, 6, 6)

    def test_get_optimization_parameters_keys(self):
        obj = self._obj()
        groups = obj.get_optimization_parameters()
        assert set(groups.keys()) == {"grids", "sigma_net"}
        for tensors in groups.values():
            assert len(tensors) > 0
            assert all(isinstance(t, nn.Parameter) and t.is_leaf for t in tensors)

    def test_pplr_optimizer_construction(self):
        obj = self._obj()
        obj.set_optimizer(
            {
                "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
                "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
            }
        )
        opt = obj.optimizer
        assert isinstance(opt, torch.optim.Adam)
        assert len(opt.param_groups) == 2
        assert sorted(pg["lr"] for pg in opt.param_groups) == [_LR_SIGMA, _LR_GRIDS]

    def test_optimizer_params_rejects_bad_keys(self):
        obj = self._obj()
        # missing a required group
        with pytest.raises(ValueError):
            obj.set_optimizer({"grids": OptimizerParams.Adam(lr=_LR_GRIDS)})
        # single-optimizer shorthand is not a valid PPLR spec here
        with pytest.raises(TypeError):
            obj.set_optimizer({"name": "adam", "lr": _LR_GRIDS})
        # a bare OptimizerParamsType (single optimizer) is also rejected
        with pytest.raises(TypeError):
            obj.set_optimizer(OptimizerParams.Adam(lr=_LR_GRIDS))

    def test_pretrain_without_target_raises(self):
        """pretrain is supported now, but needs a target (use from_pixelated or pass one)."""
        obj = self._obj()
        obj._initialize_obj((1, 24, 28))
        with pytest.raises(ValueError, match="pretrain target"):
            obj.pretrain(num_iters=1, show=False)

    def test_disable_sentinel_accepted(self):
        """The framework's "disabled" optimizer sentinel must pass through (used by reset).

        ``reconstruct(reset=True)`` replays ``reset_optimizer`` with the init default
        ``{"default": NoneOptimizer()}``; that must not be rejected by the PPLR key check.
        """
        obj = self._obj()
        # bare NoneOptimizer and the default-keyed dict both mean "no optimizer"
        obj.set_optimizer(OptimizerParams.NoneOptimizer())
        assert obj.optimizer is None
        obj.set_optimizer({"default": OptimizerParams.NoneOptimizer()})
        assert obj.optimizer is None

    def test_autograd_only_backward_raises(self):
        obj = self._obj()
        with pytest.raises(NotImplementedError):
            obj.backward()

    def test_from_model_warns_on_lambda_activation(self):
        # a lambda activation cannot be pickled by AutoSerialize -> from_model must warn
        model = KPlanes(
            M_features=4,
            resolution=(8, 8, 8),
            multiscale_res_multipliers=[1.0],
            use_hybrid_mlp=False,  # default density_activation is a lambda
        )
        with pytest.warns(UserWarning, match="density_activation"):
            ObjectTensorDecomp.from_model(model, num_slices=1, obj_type="pure_phase")

    def test_from_model_rejects_non_kplanes(self):
        with pytest.raises(TypeError):
            ObjectTensorDecomp.from_model(nn.Linear(3, 1), num_slices=1)  # pyright: ignore[reportArgumentType]

    def test_tilted_from_uniform(self):
        """tilted=True builds a KPlanesTILTED with an extra `so3` param group."""
        obj = ObjectTensorDecomp.from_uniform(
            num_slices=1,
            M_features=6,
            resolution=(16, 32, 32),
            multiscale_res_multipliers=(0.5, 1.0),
            tilted=True,
            T=4,
            obj_type="pure_phase",
            rng=0,
        )
        obj._initialize_obj((1, 24, 28))
        assert obj.model.tilted is True
        assert set(obj.get_optimization_parameters().keys()) == {"grids", "sigma_net", "so3"}
        patches = obj.forward(torch.rand(3, 6, 6, 2) * 2 - 1)
        assert patches.shape == (1, 3, 6, 6)
        # vacuum init holds for the tilted decoder too
        assert torch.allclose(patches, torch.ones_like(patches), atol=1e-6)
        obj.set_optimizer({k: OptimizerParams.Adam(lr=1e-2) for k in obj.model.param_keys})
        assert obj.optimizer is not None
        assert len(obj.optimizer.param_groups) == 3

    def test_from_model_accepts_cptilted(self):
        """from_model accepts the CPTilted bottleneck (TensorDecompositionModel, not a KPlanes)."""
        cp = CPTilted(
            C=4,
            resolution=(32, 32, 32),
            multiscale_res_multipliers=[1.0],
            T=4,
            density_activation=nn.Identity(),
        )
        obj = ObjectTensorDecomp.from_model(cp, num_slices=1, obj_type="pure_phase")
        assert set(obj.get_optimization_parameters().keys()) == {"grids", "sigma_net", "so3"}
        assert obj.forward(torch.rand(2, 4, 4, 2) * 2 - 1).shape == (1, 2, 4, 4)

    def test_from_pixelated_pretrain(self):
        """from_pixelated + pretrain warm-starts the K-Planes grid to a pixelated object."""
        h = w = 32
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        phase = (0.5 * torch.sin(3 * gy) * torch.cos(2 * gx)).float()[None]
        pix = ObjectPixelated.from_array(initial_obj=phase, obj_type="pure_phase")
        pix._initialize_obj((1, h, w), sampling=(1.0, 1.0))

        kp = ObjectTensorDecomp.from_pixelated(pix, M_features=16, resolution=(48, 48, 48))
        assert kp.num_slices == pix.num_slices
        assert tuple(kp.pretrain_target.shape) == tuple(pix.obj.shape)

        kp.pretrain(num_iters=120, show=False)  # default PPLR optimizer
        losses = kp.pretrain_losses
        assert losses[-1] < 0.05 * losses[0]

        gt = pix.obj[0].detach().cpu().numpy()
        gt -= gt.mean()

        def _corr_to_pix(arr):
            a = arr - arr.mean()
            return float((a * gt).sum() / np.sqrt((a**2).sum() * (gt**2).sum() + 1e-12))

        assert _corr_to_pix(kp.obj[0].detach().cpu().numpy()) > 0.9
        # pretrained weights are the reset state
        kp.reset()
        assert _corr_to_pix(kp.obj[0].detach().cpu().numpy()) > 0.9

    def test_potential_identity_default_and_positivity_penalty(self):
        """Potential K-Planes uses an identity decoder by default (softplus/relu fit zero-background
        potentials poorly); the inherited soft positivity penalty drives it non-negative instead."""
        obj = self._obj(obj_type="potential", resolution=(16, 16, 16))
        obj._initialize_obj((1, 16, 16))
        assert obj.obj_type == "potential"
        assert isinstance(obj.model.density_activation, nn.Identity)  # identity, not softplus
        # force the whole potential negative via the (zero-weight) decoder bias
        with torch.no_grad():
            obj.model.sigma_net.bias.fill_(-0.5)  # type:ignore
        assert float(obj._materialize_obj().min()) == pytest.approx(-0.5, abs=1e-3)
        obj.constraints = {"positivity_weight": 1.0}
        assert float(obj._sampled_positivity_loss(1.0)) == pytest.approx(0.5, abs=0.05)
        opt = torch.optim.Adam(obj.model.parameters(), lr=2e-2)
        for _ in range(100):
            opt.zero_grad()
            obj.apply_soft_constraints().backward()
            opt.step()
        assert float(obj._materialize_obj().min()) > -1e-2  # driven non-negative


# --------------------------------------------------------------------------- #
# Implicit-flag sync (ObjectTensorDecomp is implicit, like ObjectINR)
# --------------------------------------------------------------------------- #
class TestImplicitSync:
    def test_implicit_flag_synced_from_obj_model(self):
        ptycho, _ = _build_kplanes_ptycho()
        assert ptycho.obj_model.is_implicit is True
        assert ptycho.dset.implicit_object is True


# --------------------------------------------------------------------------- #
# PPLR optimizer pass-through through the ptychography optimizer layer
# --------------------------------------------------------------------------- #
class TestPPLRPassthrough:
    def test_setter_passes_nested_dict_through_unmutated(self):
        ptycho, _ = _build_kplanes_ptycho()
        params = _pplr_params()
        snapshot = {k: dict(v) for k, v in params.items()}  # shallow copy of inner dicts
        ptycho.optimizer_params = params
        ptycho.set_optimizers()
        # the model received the PPLR groups, not a corrupted single-optimizer dict
        stored = ptycho.obj_model.optimizer_params
        assert set(stored.keys()) == {"grids", "sigma_net"}
        # the ptychography setter must not have mutated the caller's nested dict
        assert params["object"] == snapshot["object"]
        assert "name" not in params["object"] and "lr" not in params["object"]
        # optimizer has the two named groups with the requested LRs
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        assert len(opt.param_groups) == 2
        assert sorted(pg["lr"] for pg in opt.param_groups) == [_LR_SIGMA, _LR_GRIDS]


# --------------------------------------------------------------------------- #
# End-to-end reconstruction
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestObjectTensorDecompReconstruction:
    def test_loss_decreases_and_recovers_object(self):
        ptycho, gt_phase = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=150,
            optimizer_params=_pplr_params(),  # probe frozen
            scheduler_params={"object": SchedulerParams.CosineAnnealing()},
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert losses[-1] < _LOSS_RATIO * losses[0]
        assert _best_corr(ptycho.obj[0], gt_phase) > _CORR_THRESHOLD

    def test_reconstruct_with_reset_runs(self):
        """reset=True replays reset_optimizer with the disabled sentinel before applying PPLR."""
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=5,
            reset=True,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        # the object optimizer still has its two PPLR groups after the reset cycle
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        assert len(opt.param_groups) == 2

    def test_tilted_reconstruct_runs(self):
        """A tilted K-Planes object reconstructs end-to-end with a 3-group PPLR optimizer."""
        pdset, gt_phase, probe = _build_synthetic_dataset()
        obj = ObjectTensorDecomp.from_uniform(
            num_slices=1,
            M_features=12,
            resolution=(16, 64, 64),
            multiscale_res_multipliers=(0.25, 0.5, 1.0),
            tilted=True,
            T=4,
            obj_type="pure_phase",
            rng=0,
        )
        probe_model = ProbePixelated.from_array(
            num_probes=1,
            probe_params={
                "energy": PROBE_ENERGY,
                "C10": C10,
                "semiangle_cutoff": _semiangle_mrad(),
            },
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
        ptycho.reconstruct(
            num_iters=60,
            optimizer_params={
                "object": {
                    "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
                    "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
                    "so3": OptimizerParams.Adam(lr=1e-3),
                }
            },
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        assert losses[-1] < losses[0]

    def test_tv_constraint_reconstruct_runs(self):
        """A reconstruct with the in-plane TV soft constraint runs and stays finite."""
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=20,
            optimizer_params=_pplr_params(),
            constraints={"object": {"tv_weight_xy": 1e-3}},
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        assert ptycho.obj_model.constraints.tv_weight_xy == 1e-3

    def test_multislice_runs(self):
        ptycho, _ = _build_kplanes_ptycho(
            num_slices=2, slice_thicknesses=20.0, resolution=(8, 64, 64)
        )
        ptycho.reconstruct(
            num_iters=5,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
        assert ptycho.obj.shape[0] == 2

    def test_scheduler_scales_pplr_groups(self):
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=10,
            optimizer_params=_pplr_params(),
            scheduler_params={"object": SchedulerParams.CosineAnnealing()},
            batch_size=200,
        )
        # one cosine schedule drives both param groups; both LRs have decayed below their start
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        lrs = sorted(pg["lr"] for pg in opt.param_groups)
        assert lrs[0] < _LR_SIGMA and lrs[1] < _LR_GRIDS

    def test_save_load_roundtrip(self, tmp_path):
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=20,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
        obj_before = ptycho.obj.copy()
        path = tmp_path / "kplanes_ptycho.zip"
        ptycho.save(path, mode="o", save_raw_data=True)  # persist dset so loaded.dset works
        loaded = autoserialize_load(path)
        assert loaded.obj_model.is_implicit is True
        assert loaded.dset.implicit_object is True
        assert loaded.obj_model.name == "ObjectTensorDecomp"
        np.testing.assert_allclose(loaded.obj, obj_before, rtol=1e-5, atol=1e-6)
        # continued training still runs after reload
        loaded.reconstruct(
            num_iters=5,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
