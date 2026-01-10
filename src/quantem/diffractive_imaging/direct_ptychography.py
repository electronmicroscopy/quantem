import gc
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

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

from itertools import product

from quantem.diffractive_imaging.direct_ptycho_utils import (
    ABERRATION_PRESETS,
    align_vbf_stack_multiscale,
    concentric_ring_wavevectors,
    create_edge_window,
    find_nearest_k_indices,
    fit_aberrations_from_shifts,
    fit_aberrations_using_least_squares,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationParameter:
    low: float
    high: float
    log: bool = False
    n_points: int | None = None

    def grid_values(self):
        """Return an array of grid values for this parameter."""
        if self.n_points is None:
            raise ValueError("n_points must be specified for grid search parameters.")
        if self.log:
            return np.geomspace(self.low, self.high, self.n_points)
        else:
            return np.linspace(self.low, self.high, self.n_points)


@dataclass
class HyperparameterState:
    fixed_aberrations: Dict[str, float] = field(default_factory=dict)
    fixed_rotation_angle: float | None = None
    optimized_aberrations: Dict[str, float] = field(default_factory=dict)
    optimized_rotation_angle: float | None = None
    optimized_keys: set[str] = field(default_factory=set)
    study: optuna.Study | None = None

    def merged_aberrations(self) -> Dict[str, float]:
        """Return full aberration dictionary (fixed ⊕ optimized)."""
        out = dict(self.fixed_aberrations)
        out.update(self.optimized_aberrations)
        return out

    def merged_rotation_angle(self) -> float | None:
        """Return rotation angle (optimized takes precedence)."""
        return (
            self.optimized_rotation_angle
            if self.optimized_rotation_angle is not None
            else self.fixed_rotation_angle
        )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        lines = []

        if self.fixed_aberrations:
            lines.append(f"  fixed_aberrations={self.fixed_aberrations!r},")
        if self.fixed_rotation_angle is not None:
            lines.append(f"  fixed_rotation_angle={self.fixed_rotation_angle!r},")
        if self.optimized_aberrations:
            lines.append(f"  optimized_aberrations={self.optimized_aberrations!r},")
        if self.optimized_rotation_angle is not None:
            lines.append(f"  optimized_rotation_angle={self.optimized_rotation_angle!r},")

        if not lines:
            return f"{cls}()"

        body = "\n".join(lines)
        return f"{cls}(\n{body}\n)"


@dataclass(frozen=True)
class BrightFieldContext:
    bf_mask: torch.Tensor
    bf_inds_i: torch.Tensor
    bf_inds_j: torch.Tensor
    num_bf: int
    vbf_index_mapping: torch.Tensor


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
        self.vbf_stack = vbf_dataset.array  # ty:ignore[invalid-assignment]
        self.bf_mask = bf_mask_dataset.array  # ty:ignore[invalid-assignment]

        self.wavelength = electron_wavelength_angstrom(energy)
        self.scan_units = vbf_dataset.units[-2:]
        self.detector_units = bf_mask_dataset.units

        self.scan_gpts = vbf_dataset.shape[-2:]
        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.num_bf = vbf_dataset.shape[0]
        self.gpts = bf_mask_dataset.shape[:2]
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

        self.semiangle_cutoff = semiangle_cutoff  # ty:ignore[invalid-assignment]
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
        """ """

        origin = CenterOfMassOriginModel.from_dataset(dataset, device=device)

        # measure and fit origin
        if force_fitted_origin is None:
            if force_measured_origin is None:
                origin.calculate_origin(max_batch_size)
            else:
                origin.origin_measured = force_measured_origin  # ty:ignore[invalid-assignment]
            origin.fit_origin_background(fit_method=fit_method)
        else:
            origin.origin_fitted = force_fitted_origin  # ty:ignore[invalid-assignment]

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

        self._vbf_fourier = torch.fft.fft2(self.vbf_stack)
        self._dc_per_image = self._vbf_fourier[..., 0, 0].mean(0)
        self._vbf_fourier[..., 0, 0] = 0  # zero DC
        self._corrected_stack = None

        return self

    def _return_bf_context(self, bf_mask):
        """
        Given a BF mask, compute all BF-dependent geometry and indexing.
        """
        bf_mask = torch.as_tensor(bf_mask, dtype=torch.bool, device=self.device)

        bf_inds_i, bf_inds_j = torch.nonzero(bf_mask, as_tuple=True)
        vbf_index_mapping = torch.where(bf_mask[self.bf_mask])[0]
        num_bf = bf_inds_i.numel()

        return BrightFieldContext(
            bf_mask=bf_mask,
            bf_inds_i=bf_inds_i,
            bf_inds_j=bf_inds_j,
            num_bf=num_bf,
            vbf_index_mapping=vbf_index_mapping,
        )

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
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        validate_gt(value, 0.0, "semiangle_cutoff")
        self._semiangle_cutoff = value

    @property
    def device(self) -> str | torch.device:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device  # ty:ignore[invalid-return-type]
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev

    @property
    def scan_sampling(self) -> NDArray:
        return self._scan_sampling  # ty:ignore[invalid-return-type]

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
        return self._reciprocal_sampling  # ty:ignore[invalid-return-type]

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

    def _return_kernel_contributions(
        self,
        bf,
        deconvolution_kernel,
        vbf_fourier,
        kxa,
        kya,
        qxa,
        qya,
        cmplx_probe_k,
        grad_k,
        sign_sin_chi_q,
        aberration_coefs,
        batch_idx,
    ):
        """ """
        ind_i = bf.bf_inds_i[batch_idx]
        ind_j = bf.bf_inds_j[batch_idx]

        kx = kxa[ind_i, ind_j].view(-1, 1, 1)
        ky = kya[ind_i, ind_j].view(-1, 1, 1)

        power = None

        if deconvolution_kernel in ("ssb", "obf", "mf"):
            qmkxa = qxa.unsqueeze(0) - kx
            qmkya = qya.unsqueeze(0) - ky
            qpkxa = qxa.unsqueeze(0) + kx
            qpkya = qya.unsqueeze(0) + ky

            cmplx_probe_at_k = cmplx_probe_k[ind_i, ind_j].view(-1, 1, 1)

            gamma = gamma_factor(
                (qmkxa, qmkya),
                (qpkxa, qpkya),
                cmplx_probe_at_k,
                self.wavelength,
                self.semiangle_cutoff,
                self.soft_edges,
                angular_sampling=self.angular_sampling,
                aberration_coefs=aberration_coefs,
                normalize=False,
            )

            fourier_factor = -1.0j * vbf_fourier * gamma.conj()
            abs_gamma = gamma.abs()

            if deconvolution_kernel == "ssb":
                fourier_factor = fourier_factor / abs_gamma.clip(1e-8)
            else:
                power = abs_gamma.square().sum(0)

        elif deconvolution_kernel == "prlx":
            qvec = torch.stack((qxa, qya), 0)
            grad_kq = torch.einsum("na,amp->nmp", grad_k[batch_idx], qvec)
            operator = torch.exp(-1j * grad_kq) * sign_sin_chi_q
            fourier_factor = vbf_fourier * operator

        else:
            q2 = qxa.square() + qya.square()
            qx_op = -1.0j * qxa / q2
            qy_op = -1.0j * qya / q2
            qx_op[0, 0] = 0.0
            qy_op[0, 0] = 0.0

            operator = kx * qx_op + ky * qy_op
            fourier_factor = vbf_fourier * operator

        return fourier_factor, power

    def _normalize_kernel_name(self, kernel):
        kernel = kernel.lower()

        aliases = {
            "ssb": "ssb",
            "single-sideband": "ssb",
            "acbf": "ssb",
            "aberration-corrected-bright-field": "ssb",
            "obf": "obf",
            "optimum-bright-field": "obf",
            "mf": "mf",
            "matched-filter": "mf",
            "prlx": "prlx",
            "parallax": "prlx",
            "tcbf": "prlx",
            "tilt-corrected-bright-field": "prlx",
            "icom": "icom",
            "center-of-mass": "icom",
        }

        if kernel not in aliases:
            raise ValueError(f"Unknown deconvolution kernel '{kernel}'")

        return aliases[kernel]

    def reconstruct(
        self,
        bf_mask=None,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
        deconvolution_kernel="single-sideband",
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        matched_filter_norm_epsilon=1e-1,
        parallax_flip_phase=True,
        verbose=None,
    ):
        """
        Unified reconstruction method supporting multiple deconvolution techniques.

        Parameters
        ----------
        bf_mask: torch.Tensor, optional
            Subset of bright field mask to use for reconstruction. Note this must be
            strictly smaller than the bf_mask used for initialization.
        aberration_coefs : dict, optional
            Aberration coefficients for the probe
        upsampling_factor : int, optional
            Factor by which to upsample the reconstruction
        rotation_angle : float, optional
            Rotation angle for coordinate system
        max_batch_size : int, optional
            Maximum batch size for processing
        deconvolution_kernel : str, one of ['ssb', 'obf', 'mf','prlx','icom']
        q_highpass : float, optional
            High-pass filter cutoff
        q_lowpass : float, optional
            Low-pass filter cutoff
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

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)

        num_bf = bf.num_bf
        bf_mask = bf.bf_mask
        vbf_index_mapping = bf.vbf_index_mapping

        if max_batch_size is None:
            max_batch_size = num_bf

        deconvolution_kernel = self._normalize_kernel_name(deconvolution_kernel)

        # Get upsampled q-space grid
        qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
        q, theta = polar_coordinates(qxa, qya)

        # Get k-space grid
        kxa, kya = spatial_frequencies(
            self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
        )
        k, phi = polar_coordinates(kxa, kya)

        # compute global / cheap functions for prlx
        if deconvolution_kernel == "prlx":
            dx, dy = aberration_surface_cartesian_gradients(
                k * self.wavelength,
                phi,
                aberration_coefs=aberration_coefs,
            )
            grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)

            if parallax_flip_phase:
                chi_q = aberration_surface(
                    q * self.wavelength,
                    theta,
                    self.wavelength,
                    aberration_coefs=aberration_coefs,
                )
                sign_sin_chi_q = torch.sign(torch.sin(chi_q))
            else:
                sign_sin_chi_q = torch.ones_like(q)
        else:
            grad_k = None
            sign_sin_chi_q = None

        # compute global / cheap functions for all
        cmplx_probe_k = evaluate_probe(
            k * self.wavelength,
            phi,
            self.semiangle_cutoff,
            self.angular_sampling,
            self.wavelength,
            aberration_coefs=aberration_coefs,
        )
        BF_weights = cmplx_probe_k[bf_mask].abs().square().sum()

        butterworth_env = torch.ones_like(q)
        if q_lowpass:
            butterworth_env *= 1 / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            butterworth_env *= 1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order))

        # Process batches
        pbar = tqdm(range(num_bf), disable=not verbose)
        batcher = SimpleBatcher(num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng)

        fourier_factor = torch.empty(
            (num_bf,) + qxa.shape, device=self.device, dtype=torch.complex64
        )
        if deconvolution_kernel in ("obf", "mf"):
            power = torch.zeros(qxa.shape, device=self.device)
        else:
            power = None

        # first pass
        for batch_idx in batcher:
            mapped_idx = vbf_index_mapping[batch_idx]
            vbf_fourier = self._vbf_fourier[mapped_idx]

            # Fourier-space tiling
            vbf_fourier = torch.cat(
                [torch.cat([vbf_fourier] * upsampling_factor, dim=-1)] * upsampling_factor,
                dim=-2,
            )

            num, pow = self._return_kernel_contributions(
                bf,
                deconvolution_kernel,
                vbf_fourier,
                kxa,
                kya,
                qxa,
                qya,
                cmplx_probe_k,
                grad_k,
                sign_sin_chi_q,
                aberration_coefs,
                batch_idx,
            )
            if power is None:
                num *= butterworth_env
                num[:, 0, 0] = self._dc_per_image

                fourier_factor[batch_idx] = torch.fft.ifft2(num)
            else:
                fourier_factor[batch_idx] = num
                power += pow

            pbar.update(len(batch_idx))
        pbar.close()

        if power is not None:
            power /= BF_weights

            if deconvolution_kernel == "obf":
                norm = power.sqrt().clamp_min(1e-8)
            elif deconvolution_kernel == "mf":
                norm = (power + matched_filter_norm_epsilon * power.max()).clamp_min(1e-8)

            # second pass
            for batch_idx in batcher:
                ff = fourier_factor[batch_idx]

                if power is not None:
                    ff /= norm

                ff *= butterworth_env
                ff[:, 0, 0] = self._dc_per_image

                fourier_factor[batch_idx] = torch.fft.ifft2(ff)

        self.corrected_stack = fourier_factor.real / BF_weights

        # memory management
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    @property
    def corrected_stack(self):
        return self._corrected_stack

    @corrected_stack.setter
    def corrected_stack(self, value: torch.Tensor):
        self._corrected_stack = validate_tensor(value, "corrected_stack", dtype=torch.float32).to(
            device=self.device
        )

    @property
    def corrected_bf(self):
        if self.corrected_stack is None:
            return None
        return self.corrected_stack.sum(dim=0)

    def variance_loss(self):
        """ """
        if self.corrected_stack is None:
            return None
        if self.corrected_stack.abs().sum() > 0:
            mean_corrected_bf = self.corrected_stack.mean(dim=0)
            variance_loss = ((self.corrected_stack - mean_corrected_bf).abs().square()).mean()
        else:
            variance_loss = torch.tensor(
                torch.inf, dtype=self.corrected_stack.dtype, device=self.device
            )
        return variance_loss

    @property
    def obj(self) -> np.ndarray:
        obj = to_numpy(self.corrected_bf)
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
        sampler = sampler or optuna.samplers.TPESampler()

        state = HyperparameterState()
        aberration_coefs = aberration_coefs or {}

        for name, val in aberration_coefs.items():
            if isinstance(val, OptimizationParameter):
                state.optimized_keys.add(name)
            else:
                state.fixed_aberrations[name] = val

            if isinstance(rotation_angle, OptimizationParameter):
                state.optimized_keys.add("rotation_angle")
            else:
                state.fixed_rotation_angle = rotation_angle

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

        state.study = study
        opt_aberration_coefs = study.best_params.copy()
        opt_rotation_angle = opt_aberration_coefs.pop("rotation_angle", None)

        state.optimized_aberrations = opt_aberration_coefs
        state.optimized_rotation_angle = opt_rotation_angle

        self._hyperparameter_state = state

        if self.verbose:
            print(self._hyperparameter_state)

        self.reconstruct_with_hyperparameters(verbose=False, **reconstruct_kwargs)
        return self

    def grid_search_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        **reconstruct_kwargs,
    ):
        """
        Perform a grid search over specified hyperparameter combinations.

        Parameters
        ----------
        aberration_coefs : dict[str, float | OptimizationParameter], optional
            Dict of aberration names to either fixed values or OptimizationParameter ranges.
        rotation_angle : float | OptimizationParameter, optional
            Fixed rotation or optimization range.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().
        """
        aberration_coefs = aberration_coefs or {}

        optimized_keys: set[str] = {
            name
            for name, val in aberration_coefs.items()
            if isinstance(val, OptimizationParameter)
        }
        if isinstance(rotation_angle, OptimizationParameter):
            optimized_keys.add("rotation_angle")

        # Build parameter grid
        param_grid = {}
        for name, val in aberration_coefs.items():
            if isinstance(val, OptimizationParameter):
                param_grid[name] = val.grid_values()
            else:
                param_grid[name] = [val]

        if rotation_angle is not None:
            if isinstance(rotation_angle, OptimizationParameter):
                param_grid["rotation_angle"] = rotation_angle.grid_values()
            else:
                param_grid["rotation_angle"] = [rotation_angle]

        # Cartesian product of all parameter combinations
        keys = list(param_grid.keys())
        grid = list(product(*(param_grid[k] for k in keys)))

        results = []
        best_loss = float("inf")
        best_params: dict[str, float] | None = None

        for combo in tqdm(grid):
            params = dict(zip(keys, combo))

            trial_aberration_coefs = {k: v for k, v in params.items() if k != "rotation_angle"}
            trial_rotation_angle = params.get("rotation_angle", None)

            self.reconstruct(
                aberration_coefs=trial_aberration_coefs,
                rotation_angle=trial_rotation_angle,
                verbose=False,
                **reconstruct_kwargs,
            )

            loss = float(self.variance_loss())
            results.append((params, loss))

            if loss < best_loss:
                best_loss = loss
                best_params = params

        self._grid_search_results = results

        fixed_aberrations = {}
        optimized_aberrations = {}
        fixed_rotation_angle = None
        optimized_rotation_angle = None

        for name, val in best_params.items():  # ty:ignore[possibly-missing-attribute]
            if name == "rotation_angle":
                if "rotation_angle" in optimized_keys:
                    optimized_rotation_angle = val
                else:
                    fixed_rotation_angle = val
            elif name in optimized_keys:
                optimized_aberrations[name] = val
            else:
                fixed_aberrations[name] = val

        self._hyperparameter_state = HyperparameterState(
            fixed_aberrations=fixed_aberrations,
            fixed_rotation_angle=fixed_rotation_angle,
            optimized_aberrations=optimized_aberrations,
            optimized_rotation_angle=optimized_rotation_angle,
            optimized_keys=optimized_keys,
        )

        if self.verbose:
            print(self._hyperparameter_state)

        self.reconstruct_with_hyperparameters(verbose=False, **reconstruct_kwargs)

        return self

    def _return_lateral_shifts(
        self,
        rotation_angle,
        aberration_coefs,
        bf_mask,
    ):
        # Get initial shifts
        kxa, kya = spatial_frequencies(
            self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
        )
        k, phi = polar_coordinates(kxa, kya)

        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength,
            phi,
            aberration_coefs=aberration_coefs,  # ty:ignore[invalid-argument-type]
        )
        grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)
        lateral_shifts = grad_k / 2 / np.pi
        return lateral_shifts

    def fit_hyperparameters(
        self,
        bf_mask: torch.Tensor | None = None,
        rotation_angle: float | None = None,
        aberration_coefs: Mapping[str, int | float | torch.Tensor] = {},
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
        bf_mask: torch.Tensor, optional
            Subset of bright field mask to use for reconstruction. Note this must be
            strictly smaller than the bf_mask used for initialization.
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

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)
        bf_mask = bf.bf_mask
        inds_i, inds_j = bf.bf_inds_i, bf.bf_inds_j

        scan_sampling = torch.as_tensor(
            self.scan_sampling, device=self.device, dtype=torch.float32
        )

        # initial reconstruction
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["deconvolution_kernel", "verbose", "parallax_flip_phase"]
        }

        self.reconstruct(
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            deconvolution_kernel="parallax",
            parallax_flip_phase=False,
            verbose=False,
            **safe_kwargs,
        )

        vbf_stack = self.corrected_stack.clone()

        # Get initial shifts
        lateral_shifts = self._return_lateral_shifts(rotation_angle, aberration_coefs, bf_mask)
        initial_shifts = lateral_shifts / scan_sampling

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

        lateral_shifts = shifts_px * scan_sampling

        fit_results = fit_aberrations_from_shifts(
            lateral_shifts,
            bf_mask,
            self.wavelength,
            self.gpts,
            self.sampling,
        )

        self.corrected_stack = vbf_stack

        fitted_aberration_coefs = fit_results.copy()
        fitted_rotation_angle = fitted_aberration_coefs.pop("rotation_angle", None)
        self._hyperparameter_state = HyperparameterState(
            optimized_aberrations=fitted_aberration_coefs,
            optimized_rotation_angle=fitted_rotation_angle,
            optimized_keys={"C10", "C12", "phi12", "rotation_angle"},
        )

        if self.verbose:
            print(self._hyperparameter_state)

        self.reconstruct_with_hyperparameters(verbose=False, **reconstruct_kwargs)

        return self

    def fit_hyperparameters_least_squares(
        self,
        cartesian_basis: str | list[str] = "low_order",
        rotation_angle: float | None = None,
        aberration_coefs: dict[str, int | float | torch.Tensor] | None = None,
        num_bf_rings: int = 3,
        num_bf_points_per_ring: int = 6,
    ):
        if aberration_coefs is None:
            aberration_coefs = self.aberration_coefs
        else:
            aberration_coefs = validate_aberration_coefficients(aberration_coefs)

        if rotation_angle is None:
            rotation_angle = self.rotation_angle
        else:
            rotation_angle = float(rotation_angle)

        if isinstance(cartesian_basis, str):
            cartesian_basis = ABERRATION_PRESETS[cartesian_basis]

        # spatial frequencies
        qxa, qya = self._return_upsampled_qgrid()

        # BF geometry
        kxa, kya = spatial_frequencies(
            self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
        )

        k_bf_target = concentric_ring_wavevectors(
            self.semiangle_cutoff * 0.95,
            num_rings=num_bf_rings,
            num_points_per_ring=num_bf_points_per_ring,
            wavelength=self.wavelength,
            include_center=False,
            device=self.device,
        )

        inds_i, inds_j = find_nearest_k_indices(k_bf_target, kxa, kya)

        k_bf = torch.stack((kxa[inds_i, inds_j], kya[inds_i, inds_j]), -1)

        bf_mask = torch.zeros_like(self.bf_mask)
        bf_mask[inds_i, inds_j] = True
        vbf_index_mapping = torch.where(bf_mask[self.bf_mask])[0]
        vbf_fourier = self._vbf_fourier[vbf_index_mapping]

        updated_aberrations_polar, delta_cartesian, phi_obj = fit_aberrations_using_least_squares(
            vbf_fourier=vbf_fourier,
            qxa=qxa,
            qya=qya,
            k_bf=k_bf,
            cartesian_basis=cartesian_basis,
            wavelength=self.wavelength,
            semiangle_cutoff=self.semiangle_cutoff,
            angular_sampling=self.angular_sampling,
            soft_edges=True,
            aberration_coefs_init=aberration_coefs,
        )

        return updated_aberrations_polar

    def reconstruct_with_hyperparameters(
        self,
        **reconstruct_kwargs,
    ):
        """ """
        if not hasattr(self, "_hyperparameter_state"):
            raise ValueError(
                "run self.optimize_hyperparameters or self.fit_hyperparameters first."
            )

        state = self._hyperparameter_state

        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["aberration_coefs", "rotation_angle"]
        }

        return self.reconstruct(
            aberration_coefs=state.merged_aberrations(),
            rotation_angle=state.merged_rotation_angle(),
            **safe_kwargs,
        )

    def _reconstruct_all_permutations(self, **reconstruct_kwargs):
        """ """
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["deconvolution_kernel", "verbose"]
        }

        kernels = ["ssb", "obf", "mf", "prlx", "icom"]

        recons = [
            self.reconstruct(
                deconvolution_kernel=kernel,
                verbose=False,
                **safe_kwargs,
            ).obj
            for kernel in tqdm(kernels, disable=not self.verbose)
        ]

        return recons

    def _make_checkerboard_bf_masks(self, gpts, bf_mask):
        """ """
        i_coords = torch.arange(gpts[0], device=self.device)
        j_coords = torch.arange(gpts[1], device=self.device)
        i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing="ij")
        checkerboard = torch.fft.ifftshift(((i_grid + j_grid) % 2).bool())

        bf1 = bf_mask & checkerboard
        bf2 = bf_mask & (~checkerboard)

        return [bf1, bf2]

    def _reconstruct_with_halfsets(self, **reconstruct_kwargs):
        """
        Compute two half-set reconstructions using alternating BF pixels (checkerboard pattern).

        Returns
        -------
        halfset_1 : torch.Tensor
            Reconstruction using first half of BF pixels
        halfset_2 : torch.Tensor
            Reconstruction using second half of BF pixels
        """

        bf1, bf2 = self._make_checkerboard_bf_masks(self.gpts, self.bf_mask)
        safe_kwargs = {
            k: v for k, v in reconstruct_kwargs.items() if k not in ["verbose", "bf_mask"]
        }

        self.reconstruct(**safe_kwargs, bf_mask=bf1, verbose=False)
        halfset_1 = self.corrected_bf

        self.reconstruct(**safe_kwargs, bf_mask=bf2, verbose=False)
        halfset_2 = self.corrected_bf

        return [halfset_1, halfset_2]
