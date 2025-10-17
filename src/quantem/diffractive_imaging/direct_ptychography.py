import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import optuna
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_gt,
    validate_int,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface,
    aberration_surface_cartesian_gradients,
    evaluate_probe,
    gamma_factor,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

from quantem.diffractive_imaging.direct_ptycho_utils import (
    align_vbf_stack_multiscale,
    fit_aberrations_from_shifts,
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
        rng: np.random.Generator | int | None,
        device: str | int,
        verbose: int | bool,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use DirectPtychography.from_dataset4dstem() or DirectPtychography.from_virtual_bfs() to instantiate this class."
            )

        self.device = device
        self.verbose = verbose
        self.vbf_stack = vbf_dataset.array
        self.bf_mask = bf_mask_dataset.array

        self.reciprocal_sampling = bf_mask_dataset.sampling

        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.scan_gpts = vbf_dataset.shape[-2:]
        self.num_bf = vbf_dataset.shape[0]
        self.semiangle_cutoff = semiangle_cutoff
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
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """ """

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            soft_edges=soft_edges,
            rng=rng,
            device=device,
            verbose=verbose,
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
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
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
            soft_edges=soft_edges,
            rng=rng,
            device=device,
            verbose=verbose,
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
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

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

    def _compute_parallax_operator(
        self, alpha, phi, qxa, qya, aberration_coefs, rotation_angle, flip_phase=True
    ):
        """Compute parallax approximation operator."""
        dx, dy = aberration_surface_cartesian_gradients(
            alpha,
            phi,
            aberration_coefs,
        )
        grad_k = torch.stack((dx[self.bf_mask], dy[self.bf_mask]), -1)

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

        return operator

    def _compute_gamma_operator(
        self, kxa, kya, qxa, qya, cmplx_probe, batch_idx, asymmetric_version=True, normalize=True
    ):
        """Compute gamma deconvolution operator."""
        ind_i = self._bf_inds_i[batch_idx]
        ind_j = self._bf_inds_j[batch_idx]

        kx = kxa[ind_i, ind_j].view(-1, 1, 1)
        ky = kya[ind_i, ind_j].view(-1, 1, 1)

        qmkxa = qxa.unsqueeze(0) - kx
        qmkya = qya.unsqueeze(0) - ky
        qpkxa = qxa.unsqueeze(0) + kx
        qpkya = qya.unsqueeze(0) + ky

        cmplx_probe_at_k = cmplx_probe[ind_i, ind_j].view(-1, 1, 1)

        gamma = gamma_factor(
            (qmkxa, qmkya),
            (qpkxa, qpkya),
            cmplx_probe_at_k,
            self.wavelength,
            self.semiangle_cutoff,
            self.soft_edges,
            angular_sampling=self.angular_sampling,
            aberration_coefs=self.aberration_coefs,
            asymmetric_version=asymmetric_version,
            normalize=normalize,
        )

        return gamma

    def _compute_icom_weighting(self, qxa, qya, kxa, kya, batch_idx, q_highpass=None):
        """Compute iCOM Fourier-space weighting factors."""
        q2 = qxa.square() + qya.square()
        q2[0, 0] = torch.inf
        qx_op = -1.0j * qxa / q2
        qy_op = -1.0j * qya / q2

        env = torch.ones_like(q2)
        if q_highpass:
            butterworth_order = 12
            env *= 1 - 1 / (1 + (torch.sqrt(q2) / q_highpass) ** (2 * butterworth_order))

        ind_i = self._bf_inds_i[batch_idx]
        ind_j = self._bf_inds_j[batch_idx]
        kx = kxa[ind_i, ind_j].view(-1, 1, 1)
        ky = kya[ind_i, ind_j].view(-1, 1, 1)

        icom_weighting = (kx * qx_op + ky * qy_op) * env

        return icom_weighting

    def reconstruct(
        self,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
        deconvolution_kernel="full",
        use_center_of_mass_weighting=False,
        flip_phase=True,
        q_highpass=None,
        verbose=None,
    ):
        """
        Unified reconstruction method supporting multiple deconvolution techniques.

        Parameters
        ----------
        aberration_coefs : dict, optional
            Aberration coefficients for the probe
        upsampling_factor : int, optional
            Factor by which to upsample the reconstruction
        rotation_angle : float, optional
            Rotation angle for coordinate system
        max_batch_size : int, optional
            Maximum batch size for processing
        deconvolution_kernel : str, one of ['full', 'quadratic', 'none']
            deconvolution_kernel = 'full' -> SSB
            deconvolution_kernel = 'quadratic' -> parallax
            deconvolution_kernel = 'none' -> BF-STEM
        use_center_of_mass_weighting : bool, optional
            If True, apply iCOM Fourier-space weighting
        flip_phase : bool, optional
            If True, flip phase in parallax approximation (default: True)
        q_highpass : float, optional
            High-pass filter cutoff for iCOM weighting
        verbose : bool, optional
            If True, show progress bar

        Returns
        -------
        self
            Returns self with corrected_stack attribute set
        """
        # Validate and set parameters
        if aberration_coefs is None:
            aberration_coefs = self.aberration_coefs
        else:
            aberration_coefs = validate_aberration_coefficients(aberration_coefs)

        if rotation_angle is None:
            rotation_angle = self.rotation_angle
        else:
            rotation_angle = float(rotation_angle)

        if verbose is None:
            verbose = self.verbose

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if max_batch_size is None:
            max_batch_size = self.num_bf

        # Get upsampled q-space grid
        qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
        # Gamma deconvolution: need k-space grid and probe
        kxa, kya = spatial_frequencies(
            self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
        )
        k, phi = polar_coordinates(kxa, kya)
        alpha = k * self.wavelength

        # Compute operator based on method
        if deconvolution_kernel == "full":
            # operator calculated inside loop
            # compute common cmplx_probe instead
            cmplx_probe = evaluate_probe(
                alpha,
                phi,
                self.semiangle_cutoff,
                self.angular_sampling,
                self.wavelength,
                aberration_coefs=aberration_coefs,
            )
        elif deconvolution_kernel == "quadratic":
            # compute parallax operator once for all batches
            operator = self._compute_parallax_operator(
                alpha, phi, qxa, qya, aberration_coefs, rotation_angle, flip_phase=flip_phase
            )
        elif deconvolution_kernel == "none":
            # nothing to do really
            pass
        else:
            raise ValueError()

        # Process batches
        pbar = tqdm(range(self.num_bf), disable=not verbose)
        batcher = SimpleBatcher(
            self.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng
        )

        corrected_stack = torch.empty((self.num_bf,) + qxa.shape, device=self.device)

        for batch_idx in batcher:
            # Tile the Fourier-space VBF images
            vbf_fourier = torch.tile(
                self._vbf_fourier[batch_idx],
                (1, upsampling_factor, upsampling_factor),
            )

            if deconvolution_kernel == "full":
                # Compute gamma operator for this batch
                operator = self._compute_gamma_operator(
                    kxa,
                    kya,
                    qxa,
                    qya,
                    cmplx_probe,
                    batch_idx,
                    asymmetric_version=not use_center_of_mass_weighting,
                    normalize=not use_center_of_mass_weighting,
                )
                fourier_factor = vbf_fourier * operator
            elif deconvolution_kernel == "quadratic":
                # Use pre-computed operator
                fourier_factor = vbf_fourier * operator[batch_idx]
            else:
                fourier_factor = vbf_fourier

            # Apply iCOM weighting if requested
            if use_center_of_mass_weighting:
                icom_weighting = self._compute_icom_weighting(
                    qxa, qya, kxa, kya, batch_idx, q_highpass
                )
                fourier_factor = fourier_factor * icom_weighting
            elif deconvolution_kernel == "full":
                fourier_factor = fourier_factor * 1.0j

            # Inverse FFT and extract appropriate component
            corrected_stack[batch_idx] = torch.fft.ifft2(fourier_factor).real * upsampling_factor
            pbar.update(len(batch_idx))

        pbar.close()
        self.corrected_stack = corrected_stack

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
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        n_trials=50,
        sampler=None,
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
                verbose=False,
                **reconstruct_kwargs,
            )
            loss = self.variance_loss()
            return float(loss)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbose)

        self._optimization_study = study
        self._optimized_parameters = study.best_params
        self.reconstruct_with_optimized_parameters(verbose=False, **reconstruct_kwargs)
        return self

    def fit_hyperparameters(
        self,
        bin_factors: tuple[int, ...] = (7, 6, 5, 4, 3, 2, 1),
        pair_connectivity: int = 4,
        **reconstruct_kwargs,
    ):
        """ """
        bf_mask = self.bf_mask
        inds_i, inds_j = self._bf_inds_i, self._bf_inds_j
        vbf_stack = self.vbf_stack.clone()

        global_shifts, vbf_stack = align_vbf_stack_multiscale(
            vbf_stack,
            bf_mask,
            inds_i,
            inds_j,
            bin_factors,
            pair_connectivity=pair_connectivity,
            upsample_factor=1,
            verbose=self.verbose,
        )

        fit_results = fit_aberrations_from_shifts(
            global_shifts, bf_mask, self.wavelength, self.gpts, self.sampling, self.scan_sampling
        )

        self.corrected_stack = vbf_stack
        self._fitted_parameters = fit_results
        self.reconstruct_with_fitted_parameters(verbose=False, **reconstruct_kwargs)
        return self

    def reconstruct_with_optimized_parameters(
        self,
        **reconstruct_kwargs,
    ):
        """ """
        if not hasattr(self, "_optimized_parameters"):
            raise ValueError("run self.optimize_hyperparameters first.")

        aberration_coefs = self._optimized_parameters.copy()
        rotation_angle = aberration_coefs.pop("rotation_angle", None)
        return self.reconstruct(
            aberration_coefs=aberration_coefs,
            rotation_angle=rotation_angle,
            **reconstruct_kwargs,
        )

    def reconstruct_with_fitted_parameters(
        self,
        **reconstruct_kwargs,
    ):
        """ """
        if not hasattr(self, "_fitted_parameters"):
            raise ValueError("run self.fit_hyperparameters first.")

        aberration_coefs = self._fitted_parameters.copy()
        rotation_angle = aberration_coefs.pop("rotation_angle", None)
        return self.reconstruct(
            aberration_coefs=aberration_coefs,
            rotation_angle=rotation_angle,
            **reconstruct_kwargs,
        )
