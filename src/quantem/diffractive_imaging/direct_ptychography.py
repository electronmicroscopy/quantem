import gc
import math
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
from quantem.diffractive_imaging.optimize_hyperparameters import OptimizationParameter
from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

from itertools import product

from quantem.diffractive_imaging.direct_ptycho_utils import (
    align_vbf_stack_multiscale,
    create_edge_window,
    fit_aberrations_from_shifts,
    process_angle_parameters,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


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

        self.wavelength = electron_wavelength_angstrom(energy)
        self.scan_units = vbf_dataset.units[-2:]
        self.detector_units = bf_mask_dataset.units

        self.scan_gpts = vbf_dataset.shape[-2:]
        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.num_bf = vbf_dataset.shape[0]
        self.gpts = bf_mask_dataset.shape
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

        self.semiangle_cutoff = semiangle_cutoff
        self.soft_edges = soft_edges
        self.rng = rng

        self.rotation_angle = rotation_angle
        self.aberration_coefs = aberration_coefs

        self._preprocess()

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        aberration_coefs: dict,
        rotation_angle_rad: float | None = None,
        rotation_angle_deg: float | None = None,
        semiangle_cutoff: float | None = None,
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """
        Create a DirectPtychography instance from pre-computed virtual BFs.

        Parameters
        ----------
        rotation_angle_rad : float, optional
            Rotation angle in radians between scan and detector coordinates.
            Cannot be specified if rotation_angle_deg is provided.
        rotation_angle_deg : float, optional
            Rotation angle in degrees between scan and detector coordinates.
            Cannot be specified if rotation_angle_rad is provided.
        """

        # Process angle parameters
        rotation_angle = process_angle_parameters(rotation_angle_rad, rotation_angle_deg)

        if rotation_angle is None:
            raise ValueError("Either rotation_angle_rad or rotation_angle_deg must be provided.")

        if aberration_coefs is None:
            aberration_coefs = {}

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
        rotation_angle_rad: float | None = None,
        rotation_angle_deg: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        soft_edges: bool = True,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
        edge_blend_pixels: int = 0,
    ):
        """
        Create a DirectPtychography instance from a 4D-STEM dataset.

        Parameters
        ----------
        rotation_angle_rad : float, optional
            Rotation angle in radians between scan and detector coordinates.
            Cannot be specified if rotation_angle_deg is provided.
        rotation_angle_deg : float, optional
            Rotation angle in degrees between scan and detector coordinates.
            Cannot be specified if rotation_angle_rad is provided.
            If neither angle is provided, it will be auto-estimated.
        """

        # Process angle parameters
        rotation_angle = process_angle_parameters(rotation_angle_rad, rotation_angle_deg)

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
            if verbose:
                print(
                    f"Auto-estimated rotation angle: {origin.detector_rotation_deg:.2f}° ({rotation_angle:.4f} rad)"
                )
                print(
                    "Consider verifying this value and providing rotation_angle_deg explicitly for best results."
                )

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
            units=dataset.units[-2:],
            sampling=dataset.sampling[-2:],
        )

        # vbf_stack
        vbf_stack = shifted_tensor[..., bf_mask].cpu()
        gpts = vbf_stack.shape[:2]

        if normalization_order == 0:
            vbf_stack = vbf_stack / vbf_stack.mean((0, 1))  # unity mean, important

        elif normalization_order == 1:
            # Fit linear background to each BF image
            x = torch.linspace(-0.5, 0.5, gpts[0])
            y = torch.linspace(-0.5, 0.5, gpts[1])
            ya, xa = torch.meshgrid(y, x, indexing="ij")

            # Basis for linear fit: [1, x, y]
            basis = torch.stack(
                [torch.ones_like(xa.ravel()), xa.ravel(), ya.ravel()], dim=1
            )  # shape: [N_pixels, 3]

            # Fit each BF image
            for k in range(vbf_stack.shape[-1]):
                intensities = vbf_stack[..., k].ravel()

                # Least squares
                coefs = torch.linalg.lstsq(basis, intensities).solution

                # Normalize
                background = (basis @ coefs).reshape(gpts)
                vbf_stack[..., k] /= background
        else:
            raise ValueError()

        # smooth window
        window_edge = create_edge_window(
            shape=gpts, edge_blend_pixels=edge_blend_pixels, device="cpu"
        )
        vbf_stack = (1 - window_edge[..., None]) + window_edge[..., None] * vbf_stack

        vbf_stack = torch.moveaxis(vbf_stack, (0, 1, 2), (1, 2, 0))
        vbf_dataset = Dataset3d.from_array(
            vbf_stack.numpy(),
            name="vBF stack",
            units=("index",) + tuple(dataset.units[:2]),
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
        self._dc_per_image = self._vbf_fourier[..., 0, 0].mean(0)
        self._vbf_fourier[..., 0, 0] = 0  # zero DC
        self._corrected_stack = None
        self._corrected_stack_amplitude = None

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
        """Rotation angle in radians (internal representation)."""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float):
        self._rotation_angle = float(value)

    @property
    def rotation_angle_rad(self) -> float:
        """Rotation angle in radians."""
        return self._rotation_angle

    @property
    def rotation_angle_deg(self) -> float:
        """Rotation angle in degrees."""
        return np.rad2deg(self._rotation_angle)

    @property
    def fitted_rotation_angle_rad(self) -> float | None:
        """Fitted rotation angle in radians (from fit_hyperparameters)."""
        if hasattr(self, "_fitted_parameters") and "rotation_angle_rad" in self._fitted_parameters:
            return self._fitted_parameters["rotation_angle_rad"]
        return None

    @property
    def fitted_rotation_angle_deg(self) -> float | None:
        """Fitted rotation angle in degrees (from fit_hyperparameters)."""
        angle_rad = self.fitted_rotation_angle_rad
        return np.rad2deg(angle_rad) if angle_rad is not None else None

    @property
    def fitted_aberration_coefs(self) -> dict | None:
        """Fitted aberration coefficients (from fit_hyperparameters)."""
        if hasattr(self, "_fitted_parameters"):
            # Return copy without rotation_angle_rad key
            return {k: v for k, v in self._fitted_parameters.items() if k != "rotation_angle_rad"}
        return None

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

    @property
    def scan_sampling(self) -> NDArray:
        return self._scan_sampling

    @scan_sampling.setter
    def scan_sampling(self, value: NDArray | tuple | list) -> None:
        """
        Units A or raises error
        """
        units = self.scan_units
        if units[0] == "A":
            self._scan_sampling = value
        else:
            raise ValueError("real-space needs to be given in 'A'")

    @property
    def reciprocal_sampling(self) -> NDArray:
        return self._reciprocal_sampling

    @reciprocal_sampling.setter
    def reciprocal_sampling(self, value: NDArray | tuple | list) -> None:
        """
        Units A or raises error
        """
        units = self.detector_units
        if units[0] == "A^-1":
            self._reciprocal_sampling = value
        elif units[0] == "mrad":
            self._reciprocal_sampling = tuple(val / self.wavelength / 1e3 for val in value)
        else:
            raise ValueError("reciprocal-space needs to be given in 'A^-1' or 'mrad'")

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

        return operator, grad_k / 2 / np.pi

    def _compute_gamma_operator(
        self,
        kxa,
        kya,
        qxa,
        qya,
        aberration_coefs,
        cmplx_probe,
        batch_idx,
        asymmetric_version=True,
        normalize=True,
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
            aberration_coefs=aberration_coefs,
            asymmetric_version=asymmetric_version,
            normalize=normalize,
        )

        return gamma

    def _compute_icom_weighting(self, qxa, qya, kxa, kya, batch_idx, q_highpass=None):
        """Compute iCOM Fourier-space weighting factors."""
        q2 = qxa.square() + qya.square()
        qx_op = -1.0j * qxa / q2
        qy_op = -1.0j * qya / q2
        qx_op[0, 0] = 0.0
        qy_op[0, 0] = 0.0

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
        rotation_angle_rad=None,
        rotation_angle_deg=None,
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
        rotation_angle_rad : float, optional
            Rotation angle in radians for coordinate system.
            Cannot be specified if rotation_angle_deg is provided.
        rotation_angle_deg : float, optional
            Rotation angle in degrees for coordinate system.
            Cannot be specified if rotation_angle_rad is provided.
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

        # Process angle parameters
        rotation_angle = process_angle_parameters(rotation_angle_rad, rotation_angle_deg)

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

        if deconvolution_kernel is None:
            deconvolution_kernel == "none"
        elif deconvolution_kernel == "parallax":
            deconvolution_kernel = "quadratic"
        elif deconvolution_kernel == "ssb":
            deconvolution_kernel = "full"
        elif deconvolution_kernel not in ("full", "quadratic", "none"):
            raise ValueError(
                f"deconvolution_kernel needs to be one on 'full' or 'ssb','quadratic' or 'parallax', 'none' or None, not '{deconvolution_kernel}'"
            )

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
            operator, self.lateral_shifts = self._compute_parallax_operator(
                alpha, phi, qxa, qya, aberration_coefs, rotation_angle, flip_phase=flip_phase
            )

        # Process batches
        pbar = tqdm(range(self.num_bf), disable=not verbose)
        batcher = SimpleBatcher(
            self.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng
        )

        if deconvolution_kernel == "full":
            corrected_stack = torch.empty(
                (self.num_bf,) + qxa.shape, device=self.device, dtype=torch.complex64
            )
        else:
            corrected_stack = torch.empty((self.num_bf,) + qxa.shape, device=self.device)

        for batch_idx in batcher:
            # Fourier-space tiling
            vbf_fourier = torch.cat(
                [torch.cat([self._vbf_fourier[batch_idx]] * upsampling_factor, dim=-1)]
                * upsampling_factor,
                dim=-2,
            )

            if use_center_of_mass_weighting:
                icom_weighting = self._compute_icom_weighting(
                    qxa, qya, kxa, kya, batch_idx, q_highpass
                )
                if deconvolution_kernel == "full":
                    icom_weighting = 1.0j * icom_weighting
            else:
                icom_weighting = 1.0

            if deconvolution_kernel == "full":
                # Compute gamma operator for this batch
                operator = self._compute_gamma_operator(
                    kxa,
                    kya,
                    qxa,
                    qya,
                    aberration_coefs,
                    cmplx_probe,
                    batch_idx,
                    asymmetric_version=not use_center_of_mass_weighting,
                    normalize=not use_center_of_mass_weighting,
                )
                fourier_factor = vbf_fourier * operator * icom_weighting
                fourier_factor[..., 0, 0] = self._dc_per_image  # normalize by mean
            elif deconvolution_kernel == "quadratic":
                fourier_factor = vbf_fourier * operator[batch_idx] * icom_weighting
            else:
                fourier_factor = vbf_fourier * icom_weighting

            # Inverse FFT and extract appropriate component
            if deconvolution_kernel == "full":
                corrected_stack[batch_idx] = torch.fft.ifft2(
                    fourier_factor
                )  # * upsampling_factor**2
            else:
                corrected_stack[batch_idx] = (
                    torch.fft.ifft2(fourier_factor).real * upsampling_factor**2
                )
            pbar.update(len(batch_idx))

        pbar.close()

        # memory management
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        if deconvolution_kernel == "full":
            self.corrected_stack = corrected_stack.angle()
            self.corrected_stack_amplitude = (
                2 - corrected_stack.abs()
            )  # amplitude contrast flipping
        else:
            self.corrected_stack = corrected_stack
            self.corrected_stack_amplitude = None

        return self

    @property
    def corrected_stack(self) -> torch.Tensor:
        return self._corrected_stack

    @corrected_stack.setter
    def corrected_stack(self, value: torch.Tensor):
        self._corrected_stack = validate_tensor(value, "corrected_stack", dtype=torch.float32).to(
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
        if self.corrected_stack.abs().sum() > 0:
            variance_loss = ((self.corrected_stack - self.mean_corrected_bf).abs().square()).mean()
        else:
            variance_loss = torch.tensor(
                torch.inf, dtype=self.corrected_stack.dtype, device=self.device
            )
        return variance_loss

    @property
    def obj(self) -> np.ndarray:
        obj = to_numpy(self.mean_corrected_bf)
        return obj

    def optimize_hyperparameters(
        self,
        aberration_coefs: dict[str, float | tuple[float, float]] | None = None,
        rotation_angle_rad: float | tuple[float, float] | None = None,
        rotation_angle_deg: float | tuple[float, float] | None = None,
        n_trials=50,
        sampler=None,
        **reconstruct_kwargs,
    ):
        """
        Optimize hyperparameters (aberrations and/or rotation) using Optuna.

        Parameters
        ----------
        aberration_coefs : dict[str, float | tuple[float, float]]
            Dict of aberration names to either fixed values or (min, max) ranges.
            Example: {"C10": (-400, 400), "C12": 0, "phi12": (-np.pi/2, np.pi/2)}
        rotation_angle_rad : float | tuple[float, float], optional
            Fixed rotation in radians or (min, max) range.
            Cannot be specified if rotation_angle_deg is provided.
        rotation_angle_deg : float | tuple[float, float], optional
            Fixed rotation in degrees or (min, max) range.
            Cannot be specified if rotation_angle_rad is provided.
        n_trials : int
            Number of Optuna trials.
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().

        Examples
        --------
        >>> # Optimize with degree ranges (more intuitive)
        >>> direct_ptycho.optimize_hyperparameters(
        ...     aberration_coefs={"C10": (-400, 400), "C12": (0, 100)},
        ...     rotation_angle_deg=(-180, -160),
        ...     n_trials=50
        ... )
        """

        # Handle tuple conversion for degrees to radians
        rotation_angle_param: float | tuple[float, float] | OptimizationParameter | None

        if isinstance(rotation_angle_deg, OptimizationParameter):
            # Convert OptimizationParameter bounds from degrees to radians
            rotation_angle_param = OptimizationParameter(
                low=np.deg2rad(rotation_angle_deg.low),
                high=np.deg2rad(rotation_angle_deg.high),
                log=rotation_angle_deg.log,
                n_points=rotation_angle_deg.n_points,
            )
        elif isinstance(rotation_angle_rad, OptimizationParameter):
            # Already in radians
            rotation_angle_param = rotation_angle_rad
        elif isinstance(rotation_angle_deg, tuple):
            # Convert tuple bounds from degrees to radians
            rotation_angle_param = (
                np.deg2rad(rotation_angle_deg[0]),
                np.deg2rad(rotation_angle_deg[1]),
            )
        elif isinstance(rotation_angle_rad, tuple):
            # Already in radians, use as-is
            rotation_angle_param = rotation_angle_rad
        elif rotation_angle_deg is None and rotation_angle_rad is None:
            # Neither provided, use None
            rotation_angle_param = None
        else:
            # Scalar values: use utility function for conversion and validation
            rotation_angle_param = process_angle_parameters(rotation_angle_rad, rotation_angle_deg)

        aberration_coefs = aberration_coefs or {}
        sampler = sampler or optuna.samplers.TPESampler()

        def objective(trial):
            trial_aberrations = {}
            for name, val in aberration_coefs.items():
                if isinstance(val, OptimizationParameter):
                    # Handle OptimizationParameter objects
                    if val.log:
                        trial_aberrations[name] = trial.suggest_float(
                            name, val.low, val.high, log=True
                        )
                    else:
                        trial_aberrations[name] = trial.suggest_float(name, val.low, val.high)
                elif isinstance(val, tuple):
                    # Tuple represents (min, max) range for optimization
                    trial_aberrations[name] = trial.suggest_float(name, val[0], val[1])
                else:
                    # Fixed value
                    trial_aberrations[name] = val

            if isinstance(rotation_angle_param, OptimizationParameter):
                # Handle OptimizationParameter for rotation
                if rotation_angle_param.log:
                    rot = trial.suggest_float(
                        "rotation_angle_rad",
                        rotation_angle_param.low,
                        rotation_angle_param.high,
                        log=True,
                    )
                else:
                    rot = trial.suggest_float(
                        "rotation_angle_rad", rotation_angle_param.low, rotation_angle_param.high
                    )
            elif isinstance(rotation_angle_param, tuple):
                # Tuple represents (min, max) range
                rot = trial.suggest_float(
                    "rotation_angle_rad", rotation_angle_param[0], rotation_angle_param[1]
                )
            else:
                # Fixed value
                rot = rotation_angle_param

            self.reconstruct(
                aberration_coefs=trial_aberrations,
                rotation_angle_rad=rot,
                verbose=False,
                **reconstruct_kwargs,
            )
            loss = self.variance_loss()
            return float(loss)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbose)

        self._optimization_study = study
        self._optimized_parameters = study.best_params
        if self.verbose:
            print(f"Optimized parameters: {self._optimized_parameters}")

        self.reconstruct_with_optimized_parameters(verbose=False, **reconstruct_kwargs)
        return self

    def grid_search_hyperparameters(
        self,
        aberration_coefs: dict[str, float | tuple[float, float, int]] | None = None,
        rotation_angle_rad: float | tuple[float, float, int] | None = None,
        rotation_angle_deg: float | tuple[float, float, int] | None = None,
        **reconstruct_kwargs,
    ):
        """
        Perform a grid search over specified hyperparameter combinations.

        Parameters
        ----------
        aberration_coefs : dict[str, float | tuple[float, float, int]], optional
            Dict of aberration names to either fixed values or (min, max, n_points) ranges.
            Example: {"C10": (-400, 400, 10), "C12": 0}
        rotation_angle_rad : float | tuple[float, float, int], optional
            Fixed rotation in radians or (min, max, n_points) range.
            Cannot be specified if rotation_angle_deg is provided.
        rotation_angle_deg : float | tuple[float, float, int], optional
            Fixed rotation in degrees or (min, max, n_points) range.
            Cannot be specified if rotation_angle_rad is provided.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().

        Examples
        --------
        >>> # Grid search with degree ranges (more intuitive)
        >>> direct_ptycho.grid_search_hyperparameters(
        ...     aberration_coefs={"C10": (-400, 400, 10), "C12": (0, 100, 5)},
        ...     rotation_angle_deg=(-180, -160, 20),
        ... )
        """

        # Handle tuple conversion for degrees to radians
        rotation_angle_param: float | tuple[float, float, int] | None
        if isinstance(rotation_angle_deg, tuple):
            # Convert tuple bounds from degrees to radians
            min_rad = np.deg2rad(rotation_angle_deg[0])
            max_rad = np.deg2rad(rotation_angle_deg[1])
            n_points = rotation_angle_deg[2] if len(rotation_angle_deg) > 2 else 10
            rotation_angle_param = (min_rad, max_rad, n_points)
        elif isinstance(rotation_angle_rad, tuple):
            # Already in radians, use as-is
            rotation_angle_param = rotation_angle_rad
        else:
            # Scalar values: use utility function for conversion and validation
            rotation_angle_param = process_angle_parameters(rotation_angle_rad, rotation_angle_deg)

        aberration_coefs = aberration_coefs or {}

        # Build parameter grid
        param_grid = {}
        for name, val in aberration_coefs.items():
            if isinstance(val, tuple):
                # Tuple: (min, max, n_points)
                low, high = val[0], val[1]
                n_points = val[2] if len(val) > 2 else 10
                param_grid[name] = np.linspace(low, high, n_points)
            else:
                # Fixed value
                param_grid[name] = [val]

        if rotation_angle_param is not None:
            if isinstance(rotation_angle_param, tuple):
                # Tuple: (min, max, n_points)
                low, high = rotation_angle_param[0], rotation_angle_param[1]
                n_points = rotation_angle_param[2] if len(rotation_angle_param) > 2 else 10
                param_grid["rotation_angle_rad"] = np.linspace(low, high, n_points)
            else:
                # Fixed value
                param_grid["rotation_angle_rad"] = [rotation_angle_param]

        # Cartesian product of all parameter combinations
        keys = list(param_grid.keys())
        grid = list(product(*(param_grid[k] for k in keys)))

        results = []
        best_loss = float("inf")
        best_params = None

        for combo in tqdm(grid):
            params = dict(zip(keys, combo))
            trial_aberration_coefs = params.copy()
            trial_rotation_angle = trial_aberration_coefs.pop("rotation_angle_rad", None)

            self.reconstruct(
                aberration_coefs=trial_aberration_coefs,
                rotation_angle_rad=trial_rotation_angle,
                verbose=False,
                **reconstruct_kwargs,
            )

            loss = float(self.variance_loss())
            results.append((params, loss))

            if loss < best_loss:
                best_loss = loss
                best_params = params

        self._grid_search_results = results
        self._optimized_parameters = best_params

        if self.verbose:
            print(f"Best grid parameters: {self._optimized_parameters}")

        self.reconstruct_with_optimized_parameters(verbose=False, **reconstruct_kwargs)

        return self

    def fit_hyperparameters(
        self,
        bin_factors: tuple[int, ...] = (3, 2, 1),
        pair_connectivity: int = 4,
        alignment_method: str = "reference",
        reference: torch.Tensor | NDArray | None = None,
        running_average: bool = False,
        regularize_shifts: bool = True,
        dft_upsample_factor: int = 4,
        **reconstruct_kwargs,
    ):
        """
        Fit aberrations and rotation angle from virtual BF stack.

        Parameters
        ----------
        bin_factors : tuple of int
            Sequence of binning factors from coarse to fine
        pair_connectivity : int
            Neighbor connectivity for pairwise alignment (4 or 8)
        alignment_method : str
            Alignment strategy:
            - "pairwise": Graph-based pairwise alignment (most robust)
            - "reference": Align all images to a reference
        reference : array-like, optional
            Reference image for alignment_method="reference".
            If None, uses mean of initial reconstruction.
        running_average : bool
            If True and alignment_method="reference", updates reference as running
            average during alignment. Can help with noisy data.
        regularize_shifts : bool
            If True, constrains shifts to physical aberration model at each iteration.
        **reconstruct_kwargs
            Additional arguments passed to reconstruct methods.
            If aberration coefficients are provided (e.g., C10, C12, phi12) or rotation_angle,
            performs initial reconstruction to seed the alignment.

        Returns
        -------
        self : object
            Returns self with fitted parameters stored in self._fitted_parameters

        The running_average option updates the reference at each bin level as:
            ref_new = ref_old * n/(n+1) + aligned_mean / (n+1)
        """
        bf_mask = self.bf_mask
        inds_i, inds_j = self._bf_inds_i, self._bf_inds_j
        scan_sampling = torch.as_tensor(
            self.scan_sampling, device=self.device, dtype=torch.float32
        )

        # initial reconstruction
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["deconvolution_kernel", "verbose"]
        }
        flip_phase = safe_kwargs.pop("flip_phase", False)

        self.reconstruct(
            deconvolution_kernel="parallax",
            flip_phase=flip_phase,
            verbose=False,
            **safe_kwargs,
        )

        vbf_stack = self.corrected_stack.clone()
        initial_shifts = self.lateral_shifts / scan_sampling

        if alignment_method == "reference":
            reference = (
                vbf_stack.mean(0)
                if reference is None
                else torch.as_tensor(reference, dtype=torch.float32, device=self.device)
            )
        else:
            reference = None

        if regularize_shifts:
            kxa, kya = spatial_frequencies(self.gpts, self.sampling, device=self.device)
            kvec = torch.dstack((kxa[bf_mask], kya[bf_mask])).view((-1, 2))
            basis = kvec * self.wavelength / scan_sampling
        else:
            basis = None

        shifts_px, vbf_stack = align_vbf_stack_multiscale(
            vbf_stack,
            bf_mask,
            inds_i,
            inds_j,
            bin_factors,
            pair_connectivity=pair_connectivity,
            upsample_factor=dft_upsample_factor,
            reference=reference,
            initial_shifts=initial_shifts,
            running_average=running_average,
            basis=basis,
            verbose=self.verbose,
        )

        self.lateral_shifts = shifts_px * scan_sampling

        fit_results = fit_aberrations_from_shifts(
            self.lateral_shifts,
            bf_mask,
            self.wavelength,
            self.gpts,
            self.sampling,
        )

        self.corrected_stack = vbf_stack
        self._fitted_parameters = fit_results

        if self.verbose:
            print(f"Fitted parameters: {self._fitted_parameters}")

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
        rotation_angle = aberration_coefs.pop("rotation_angle_rad", None)

        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["aberration_coefs", "rotation_angle_rad"]
        }
        return self.reconstruct(
            aberration_coefs=aberration_coefs,
            rotation_angle_rad=rotation_angle,
            **safe_kwargs,
        )

    def reconstruct_with_fitted_parameters(
        self,
        **reconstruct_kwargs,
    ):
        """ """
        if not hasattr(self, "_fitted_parameters"):
            raise ValueError("run self.fit_hyperparameters first.")

        aberration_coefs = self._fitted_parameters.copy()
        rotation_angle = aberration_coefs.pop("rotation_angle_rad", None)
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["aberration_coefs", "rotation_angle_rad"]
        }
        return self.reconstruct(
            aberration_coefs=aberration_coefs,
            rotation_angle_rad=rotation_angle,
            **safe_kwargs,
        )

    def _reconstruct_all_permutations(self, **reconstruct_kwargs):
        """ """
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["deconvolution_kernel", "use_center_of_mass_weighting", "verbose"]
        }

        combo = list(
            product(
                [False, True],
                ["none", "quadratic", "full"],
            )
        )

        recons = [
            self.reconstruct(
                deconvolution_kernel=kernel,
                use_center_of_mass_weighting=icom,
                verbose=False,
                **safe_kwargs,
            ).obj
            for icom, kernel in tqdm(combo, disable=not self.verbose)
        ]

        return recons
