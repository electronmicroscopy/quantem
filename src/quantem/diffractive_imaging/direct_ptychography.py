import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import optuna
from numpy.typing import NDArray

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_gt,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    _polar_coordinates,
    aberration_surface,
    aberration_surface_cartesian_gradients,
    evaluate_probe,
    gamma_factor,
    polar_spatial_frequencies,
    spatial_frequencies,
)
from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


from quantem.core.utils.imaging_utils import cross_correlation_shift_torch
from quantem.diffractive_imaging.direct_ptycho_utils import (
    _bin_mask_and_stack_centered,
    _fourier_shift_stack,
    _make_periodic_pairs,
    _synchronize_shifts,
    _torch_polar,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationParameter:
    low: float
    high: float
    log: bool = False


class DirectPtychography(RNGMixin, AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float | None,
        soft_edges: bool,
        vacuum_probe_intensity: torch.Tensor | None,
        rng: np.random.Generator | int | None,
        device: str | int,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use DirectPtychography.from_dataset4dstem() or DirectPtychography.from_virtual_bfs() to instantiate this class."
            )

        self.device = device
        self.vbf_stack = vbf_dataset.array
        self.bf_mask = bf_mask_dataset.array

        self.reciprocal_sampling = bf_mask_dataset.sampling

        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.scan_gpts = vbf_dataset.shape[-2:]
        self.num_bf = vbf_dataset.shape[0]

        if semiangle_cutoff is None and vacuum_probe_intensity is None:
            raise ValueError(
                "one of semiangle_cutoff or vacuum_probe_intensity needs to be specified"
            )

        self.semiangle_cutoff = semiangle_cutoff
        self.vacuum_probe_intensity = vacuum_probe_intensity
        self.soft_edges = soft_edges
        self.rng = rng

        self.gpts = bf_mask_dataset.shape
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))
        self.wavelength = electron_wavelength_angstrom(energy)
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.rotation_angle = rotation_angle
        self.aberration_coefs = aberration_coefs

        self._preprocess()

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float | None = None,
        vacuum_probe_intensity: torch.Tensor | None = None,
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
    ):
        """ """

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            vacuum_probe_intensity=vacuum_probe_intensity,
            soft_edges=soft_edges,
            rng=rng,
            device=device,
            _token=cls._token,
        )

    @classmethod
    def from_dataset4d(
        cls,
        dataset: Dataset4d,
        energy: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        rotation_angle: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bicubic",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        vacuum_probe_intensity: torch.Tensor | None = None,
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
    ):
        """ """

        origin = CenterOfMassOriginModel.from_dataset(dataset, device=device)

        # measure and fit origin
        if force_fitted_origin is None:
            if force_measured_origin is None:
                origin.calculate_origin(max_batch_size)
            else:
                origin.origin_measured = force_measured_origin
            origin.fit_origin_background(fit_method=fit_method)
        else:
            origin.origin_fitted = force_fitted_origin

        if rotation_angle is None:
            origin.estimate_detector_rotation()
            rotation_angle = origin.detector_rotation_deg / 180 * math.pi

        # shift to origin
        origin.shift_origin_to(
            max_batch_size=max_batch_size,
            mode=mode,
        )
        shifted_tensor = origin.shifted_tensor

        # bf_mask
        mean_dp = shifted_tensor.mean(dim=(0, 1))
        bf_mask = mean_dp > mean_dp.max() * intensity_threshold

        bf_mask_dataset = Dataset2d.from_array(
            bf_mask.cpu().numpy(),
            name="BF mask",
            units=("A^-1", "A^-1"),
            sampling=dataset.sampling[-2:],
        )

        # vbf_stack
        vbf_stack = shifted_tensor[..., bf_mask]
        vbf_stack = vbf_stack / vbf_stack.mean((0, 1)) - 1
        vbf_stack = torch.moveaxis(vbf_stack, (0, 1, 2), (1, 2, 0))

        vbf_dataset = Dataset3d.from_array(
            vbf_stack.cpu().numpy(),
            name="vBF stack",
            units=("index", "A", "A"),
            sampling=(1,) + tuple(dataset.sampling[:2]),
        )

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            vacuum_probe_intensity=vacuum_probe_intensity,
            soft_edges=soft_edges,
            rng=rng,
            device=device,
            _token=cls._token,
        )

    def _preprocess(
        self,
    ):
        """ """

        self._bf_inds_i, self._bf_inds_j = torch.where(self.bf_mask)
        self._vbf_fourier = torch.fft.fft2(self.vbf_stack)
        self._corrected_stack = None

        return self

    def _return_upsampled_qgrid(
        self,
        upsampling_factor=None,
    ):
        """
        Assumes integer upsampling factor.
        """

        if upsampling_factor is None:
            scan_gpts = self.scan_gpts
            scan_sampling = self.scan_sampling
        else:
            scan_gpts = tuple(n * upsampling_factor for n in self.scan_gpts)
            scan_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)

        qxa, qya = spatial_frequencies(scan_gpts, scan_sampling, device=self.device)

        return qxa, qya

    @property
    def vbf_stack(self) -> torch.Tensor:
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value: torch.Tensor):
        self._vbf_stack = validate_tensor(value, "vbf_stack", dtype=torch.float).to(
            device=self.device
        )

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._bf_mask

    @bf_mask.setter
    def bf_mask(self, value: torch.Tensor):
        self._bf_mask = validate_tensor(value, "bf_mask", dtype=torch.bool).to(device=self.device)

    @property
    def rotation_angle(self) -> float:
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float):
        self._rotation_angle = float(value)

    @property
    def aberration_coefs(self) -> dict:
        return self._aberration_coefs

    @aberration_coefs.setter
    def aberration_coefs(self, value: dict):
        value = validate_aberration_coefficients(value)
        self._aberration_coefs = value

    @property
    def semiangle_cutoff(self) -> dict:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        validate_gt(value, 0.0, "semiangle_cutoff")
        self._semiangle_cutoff = value

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        # allow setting gpu/cpu, but not changing the device from the config gpu device
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev
            try:
                self.to(dev)
            except AttributeError:
                pass

    def _parallax_approximation(
        self,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
        flip_phase=True,
    ):
        """ """
        if aberration_coefs is None:
            aberration_coefs = self.aberration_coefs
        else:
            aberration_coefs = validate_aberration_coefficients(aberration_coefs)

        if rotation_angle is None:
            rotation_angle = self.rotation_angle
        else:
            rotation_angle = float(rotation_angle)

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if aberration_coefs:
            k, phi = polar_spatial_frequencies(
                self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
            )
            alpha = k * self.wavelength

            dx, dy = aberration_surface_cartesian_gradients(
                alpha,
                phi,
                aberration_coefs,
            )

            grad_k = torch.stack((dx[self.bf_mask], dy[self.bf_mask]), -1)

            qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
            qvec = torch.stack((qxa, qya), 0)

            grad_kq = torch.einsum("na,amp->nmp", grad_k, qvec)
            operator = torch.exp(-1j * grad_kq)

            if flip_phase:
                q = torch.sqrt(qxa.square() + qya.square())
                theta = torch.arctan2(qya, qxa)

                chi_q = aberration_surface(
                    q * self.wavelength,
                    theta,
                    self.wavelength,
                    aberration_coefs,
                )
                sign_sign_chi_q = torch.sign(torch.sin(chi_q))
                operator = operator * sign_sign_chi_q

            if max_batch_size is None:
                max_batch_size = self.num_bf

            batcher = SimpleBatcher(
                self.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng
            )

            corrected_stack = torch.empty((self.num_bf,) + qxa.shape, device=self.device)
            for batch_idx in batcher:
                vbf_fourier = torch.tile(
                    self._vbf_fourier[batch_idx],
                    (1, upsampling_factor, upsampling_factor),
                )
                corrected_stack[batch_idx] = (
                    torch.fft.ifft2(vbf_fourier * operator[batch_idx]).real * upsampling_factor
                )

            self.corrected_stack = corrected_stack

        return self

    def _kernel_deconvolution(
        self,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
    ):
        """ """
        if aberration_coefs is None:
            aberration_coefs = self.aberration_coefs
        else:
            aberration_coefs = validate_aberration_coefficients(aberration_coefs)

        if rotation_angle is None:
            rotation_angle = self.rotation_angle
        else:
            rotation_angle = float(rotation_angle)

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        kxa, kya = spatial_frequencies(
            self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
        )
        qxa, qya = self._return_upsampled_qgrid(
            upsampling_factor=upsampling_factor,
        )

        k, phi = _polar_coordinates(kxa, kya)
        alpha = k * self.wavelength
        cmplx_probe = evaluate_probe(
            alpha,
            phi,
            self.semiangle_cutoff,
            self.angular_sampling,
            self.wavelength,
            aberration_coefs=aberration_coefs,
        )

        if max_batch_size is None:
            max_batch_size = self.num_bf

        batcher = SimpleBatcher(
            self.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng
        )

        corrected_stack = torch.empty((self.num_bf,) + qxa.shape, device=self.device)
        for batch_idx in batcher:
            ind_i = self._bf_inds_i[batch_idx]
            ind_j = self._bf_inds_j[batch_idx]

            qmkxa = qxa.unsqueeze(0) - kxa[ind_i, ind_j].view(-1, 1, 1)
            qmkya = qya.unsqueeze(0) - kya[ind_i, ind_j].view(-1, 1, 1)

            qpkxa = qxa.unsqueeze(0) + kxa[ind_i, ind_j].view(-1, 1, 1)
            qpkya = qya.unsqueeze(0) + kya[ind_i, ind_j].view(-1, 1, 1)

            cmplx_probe_at_k = cmplx_probe[ind_i, ind_j].view(-1, 1, 1)

            gamma = gamma_factor(
                (qmkxa, qmkya),
                (qpkxa, qpkya),
                cmplx_probe_at_k,
                self.wavelength,
                self.semiangle_cutoff,
                self.vacuum_probe_intensity,
                self.soft_edges,
                angular_sampling=self.angular_sampling,
                aberration_coefs=aberration_coefs,
            )

            vbf_fourier = torch.tile(
                self._vbf_fourier[batch_idx],
                (1, upsampling_factor, upsampling_factor),
            )
            corrected_stack[batch_idx] = (
                torch.fft.ifft2(vbf_fourier * gamma).imag * upsampling_factor
            )

        self.corrected_stack = corrected_stack
        return self

    def reconstruct(
        self,
        use_parallax_approximation=False,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
        flip_phase=True,
    ):
        """ """
        if use_parallax_approximation:
            self._parallax_approximation(
                aberration_coefs=aberration_coefs,
                rotation_angle=rotation_angle,
                upsampling_factor=upsampling_factor,
                max_batch_size=max_batch_size,
                flip_phase=flip_phase,
            )
        else:
            self._kernel_deconvolution(
                aberration_coefs=aberration_coefs,
                rotation_angle=rotation_angle,
                upsampling_factor=upsampling_factor,
            )

        return self

    @property
    def corrected_stack(self) -> torch.Tensor:
        return self._corrected_stack

    @corrected_stack.setter
    def corrected_stack(self, value: torch.Tensor):
        self._corrected_stack = validate_tensor(value, "corrected_stack", dtype=torch.float).to(
            device=self.device
        )

    @property
    def mean_corrected_bf(self):
        if self.corrected_stack is None:
            return None
        return self.corrected_stack.mean(dim=0)

    def variance_loss(self):
        """ """
        if self.corrected_stack is None:
            return None
        return ((self.corrected_stack - self.mean_corrected_bf).abs().square()).mean()

    @property
    def obj(self) -> np.ndarray:
        obj = to_numpy(self.mean_corrected_bf)
        return obj

    def optimize_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] = None,
        rotation_angle: float | OptimizationParameter = None,
        n_trials=50,
        sampler=None,
        show_progress_bar=True,
        **reconstruct_kwargs,
    ):
        """
        Optimize hyperparameters (aberrations and/or rotation) using Optuna.

        Parameters
        ----------
        aberration_coefs : dict[str, float|OptimizationParameter]
            Dict of aberration names to either fixed values or optimization ranges.
        rotation_angle : float|OptimizationParameter
            Fixed rotation or optimization range.
        n_trials : int
            Number of Optuna trials.
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
        direction : str
            "minimize" or "maximize" (default: "minimize").
        show_progress_bar : bool
            Show progress bar during optimization.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().
        """

        aberration_coefs = aberration_coefs or {}
        sampler = sampler or optuna.samplers.TPESampler()

        def objective(trial):
            trial_aberrations = {}
            for name, val in aberration_coefs.items():
                if isinstance(val, OptimizationParameter):
                    trial_aberrations[name] = trial.suggest_float(
                        name, val.low, val.high, log=val.log
                    )
                else:
                    trial_aberrations[name] = val

            if isinstance(rotation_angle, OptimizationParameter):
                rot = trial.suggest_float(
                    "rotation_angle",
                    rotation_angle.low,
                    rotation_angle.high,
                    log=rotation_angle.log,
                )
            else:
                rot = rotation_angle

            self.reconstruct(
                aberration_coefs=trial_aberrations,
                rotation_angle=rot,
                **reconstruct_kwargs,
            )
            loss = self.variance_loss()
            return float(loss)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        self._hyperparameters_study = study
        self._best_hyperparameters = study.best_params
        self._reconstruct_optimized(**reconstruct_kwargs)
        return self

    def _reconstruct_optimized(
        self,
        **reconstruct_kwargs,
    ):
        """ """
        if not hasattr(self, "_best_hyperparameters"):
            raise ValueError("run self.optimize_hyperparameters first.")

        aberration_coefs = self._best_hyperparameters.copy()
        rotation_angle = aberration_coefs.pop("rotation_angle", None)
        return self.reconstruct(
            aberration_coefs=aberration_coefs,
            rotation_angle=rotation_angle,
            **reconstruct_kwargs,
        )

    def fit_low_order_aberrations(
        self, bin_factors: tuple[int] = (7, 6, 5, 4, 3, 2, 1), pair_connectivity: int = 4
    ):
        """ """
        bf_mask = self.bf_mask
        inds_i, inds_j = self._bf_inds_i, self._bf_inds_j

        vbf_stack = self.vbf_stack.clone()
        global_shifts = torch.zeros((self.num_bf, 2), device=self.device)

        for bin_factor in bin_factors:
            bf_mask_binned, inds_ib, inds_jb, vbf_binned, mapping = _bin_mask_and_stack_centered(
                bf_mask, inds_i, inds_j, vbf_stack, bin_factor=bin_factor
            )

            pairs = _make_periodic_pairs(bf_mask_binned, connectivity=pair_connectivity)

            rel_shifts = []
            for i, j in pairs:
                s_ij, _ = cross_correlation_shift_torch(
                    vbf_binned[i],
                    vbf_binned[j],
                    upsample_factor=1,
                )
                rel_shifts.append((i.item(), j.item(), s_ij))

            shifts = _synchronize_shifts(len(vbf_binned), rel_shifts, self.device)

            global_shifts += shifts[mapping]
            vbf_stack = _fourier_shift_stack(self.vbf_stack, global_shifts)

        self.corrected_stack = vbf_stack

        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            device=self.device,
        )
        kvec = torch.dstack((kxa[bf_mask], kya[bf_mask])).view((-1, 2))
        basis = kvec * self.wavelength

        shifts_ang = (global_shifts * torch.as_tensor(self.scan_sampling)).to(
            dtype=basis.dtype, device=basis.device
        )

        # least-squares fit
        m = torch.linalg.lstsq(basis, shifts_ang, rcond=None)[0]

        m_rotation, m_aberration = _torch_polar(m)
        rotation_rad = -1 * torch.arctan2(m_rotation[1, 0], m_rotation[0, 0])

        if 2 * torch.abs(torch.remainder(rotation_rad + math.pi, 2 * math.pi) - math.pi) > math.pi:
            rotation_rad = torch.remainder(rotation_rad, 2 * math.pi) - math.pi
            m_aberration = -m_aberration

        aberrations_C1 = (m_aberration[0, 0] + m_aberration[1, 1]) / 2
        aberrations_C12a = (m_aberration[0, 0] - m_aberration[1, 1]) / 2
        aberrations_C12b = (m_aberration[1, 0] + m_aberration[0, 1]) / 2

        aberrations_C12 = torch.sqrt(aberrations_C12a.square() + aberrations_C12b.square())
        aberrations_phi12 = torch.arctan2(aberrations_C12b, aberrations_C12a) / 2

        self._fitted_aberrations = {
            "C10": aberrations_C1.item(),
            "C12": aberrations_C12.item(),
            "phi12": aberrations_phi12.item(),
        }
        self._fitted_rotation_angle = rotation_rad.item()

        return self
